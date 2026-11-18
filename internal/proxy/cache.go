package proxy

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/tfyl/strata-diskcache/internal/arbiter"
	"github.com/tfyl/strata-diskcache/internal/engine"
	"github.com/tfyl/strata-diskcache/internal/store"
	"github.com/tfyl/strata-diskcache/internal/strata"
)

// Policy tunes when the cache saves.
type Policy struct {
	MinTokens     int           // save conversations of at least this many tokens
	ParkMinTokens int           // the engine parks only these; -1 asks the engine
	RegrowTokens  int           // save again once a conversation has grown this much past its file
	IdleDelay     time.Duration // quiet time after a turn before it is saved; 0: never
	// MaxIdleSaveTokens: the background idle save is skipped above this. A
	// 100K-token session writes 1.4 GB in ~110 s, and the save holds the save
	// lock, so a queued request waits for it. The yield save at a switch writes the
	// same file in ~4 s, and the shutdown save at SIGTERM, so skipping the idle save
	// costs nothing the next request will not get. 0: no cap.
	MaxIdleSaveTokens int
	// RAM per token of a parked conversation's snapshot, to predict a refused park
	SnapshotBytesPerToken int64
	MemAvailable          func() (int64, error) // default strata.MemAvailable
	// BodyDivisor estimates the tokens of a conversation from its request body:
	// bytes per token. 0: never estimate, only what the engine reports.
	BodyDivisor int
}

// The bounds a learned bytes-per-token figure is clamped to. Measured: Strata k8v4
// 13.8 KB/token, a llama slot 85.7 KB/token.
const (
	minBPT = 4 << 10
	maxBPT = 256 << 10
)

// Actions, one per generation request.
const (
	actContinue = "continue" // continues the live conversation
	actRAM      = "ram"      // switches to a conversation parked in RAM
	actDisk     = "disk"     // switches to a conversation restored from its file
	actMiss     = "miss"     // switches to a conversation with no copy anywhere
	actBusy     = "busy"     // other requests in flight: passed through untouched
	actUnkeyed  = "unkeyed"  // a body the cache can't key
	actPass     = "pass"     // the engine didn't answer, or the body was too big to look at
	actRefused  = "refused"  // the model the request named is not one this arbiter serves
	actDenied   = "denied"   // a served model the card was busy for: 503
	actError    = "error"    // the cache's own logic failed
)

// route is one (backend, model) identity and everything the cache remembers
// about it. The session namespace is its own, so a 27B route learning a
// fingerprint never touches flash-next's files.
type route struct {
	id         string
	ns         string
	eng        engine.Engine
	fp         string
	live       []string   // keys of the conversation this engine holds; nil: unknown
	ram        [][]string // believed parked in RAM, most recent first
	started    int64
	liveTokens int   // tokens this engine holds for live, estimated when it cannot say
	bpt        int64 // file bytes per token of this route's last save
}

// cache decides, per generation request, what the engine needs before it. An
// engine can save only the live conversation (a parked one is out of reach), and
// its restore drops the live one without parking it. So: a continuing turn
// passes straight through; a switch restores the incoming conversation's file
// unless it is parked in RAM; the outgoing one is saved at the switch only when
// it would be lost there, and otherwise while idle. See docs/design.md.
type cache struct {
	arb    *arbiter.Arbiter // nil: one upstream, no arbitration
	routes map[string]*route
	files  *store.Store
	m      *Metrics
	log    *slog.Logger
	pol    Policy
	keys   keyer

	ctx    context.Context // ends with close; idle saves run under it
	cancel context.CancelFunc
	wg     sync.WaitGroup

	saveMu sync.Mutex // one switch decision or idle save at a time

	mu       sync.Mutex
	cur      *route // the route the last request used, the one idle saves watch
	inflight int
	queued   int // requests waiting for the card, as the arbiter last reported them
	seq      uint64
	idle     *time.Timer
}

// ticket is one request's decision.
type ticket struct {
	route          *route
	action, reason string
	tokens         int // restored, or the outgoing conversation's
	seq            uint64
	counted        bool
	keys           []string
	prev           []string // the live conversation this request switched away from
	prevParks      bool
}

// route returns the cache's state for one route id; the proxy needs it to
// prepare a request the arbiter just admitted.
func (c *cache) route(id string) *route {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.routes[id]
}

// Swap is the arbiter's note that the card moved. The route that gave the card
// away has no live conversation and nothing parked, and the route taking it has
// just been loaded, so its belief is stale too. Clear both, so the next request
// is decided from the session files and not from a conversation the engine no
// longer holds: llama.cpp's router reports no start time, so the restart check
// cannot see a model that was unloaded and loaded again.
func (c *cache) Swap(from, to string, _ float64) {
	c.mu.Lock()
	for _, id := range []string{from, to} {
		if r := c.routes[id]; r != nil {
			r.live, r.liveTokens, r.ram = nil, 0, nil
		}
	}
	c.mu.Unlock()
}

// Reloaded is the arbiter's note that a route's model was loaded from cold: the
// engine it starts with holds no conversation, so a continuation is not possible
// and the restore path is the one that can help.
func (c *cache) Reloaded(route string) {
	c.mu.Lock()
	if r := c.routes[route]; r != nil {
		r.live, r.liveTokens, r.ram = nil, 0, nil
	}
	c.mu.Unlock()
}

// newCache builds the cache over the arbiter's routes. Without a store, there is
// nothing to cache and the proxy is a plain forwarder.
func newCache(arb *arbiter.Arbiter, files *store.Store, m *Metrics, log *slog.Logger, pol Policy) *cache {
	if pol.MemAvailable == nil {
		pol.MemAvailable = strata.MemAvailable
	}
	ctx, cancel := context.WithCancel(context.Background())
	c := &cache{arb: arb, files: files, m: m, log: log, pol: pol, ctx: ctx, cancel: cancel, routes: map[string]*route{}}
	for _, r := range arbRoutes(arb) {
		c.routes[r.ID] = &route{id: r.ID, ns: r.Namespace, eng: r.Engine}
	}
	if arb != nil {
		// The arbiter asks for the save before it gives the card away.
		arb.SetSaveLive(c.SaveLive)
	}
	return c
}

// arbRoutes is the arbiter's route list; nil-safe for a single-upstream proxy.
func arbRoutes(arb *arbiter.Arbiter) []arbiter.Route {
	if arb == nil {
		return nil
	}
	return arb.Routes()
}

// close stops idle saves and waits for one under way until ctx ends. A save
// the engine is still writing completes on its own; the next start indexes it.
func (c *cache) close(ctx context.Context) error {
	c.mu.Lock()
	c.cancel() // under mu: no turn ending now can schedule another idle save
	if c.idle != nil && c.idle.Stop() {
		c.wg.Done()
	}
	c.mu.Unlock()
	done := make(chan struct{})
	go func() { c.wg.Wait(); close(done) }()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("a save was still running: %w", ctx.Err())
	}
}

// identity is the route's RAM cache state and the fingerprint it must be keyed
// by. Strata answers both from /metrics; llama.cpp has no RAM cache, so its
// identity comes from GET /models, which never loads a model. An unloaded engine
// answers neither, and the error is the answer: there is nothing to key by.
func (c *cache) identity(ctx context.Context, r *route) (engine.CacheInfo, error) {
	en, err := r.eng.CacheInfo(ctx)
	if err != nil || en.Fingerprint != "" {
		return en, err
	}
	fp, ferr := r.eng.Fingerprint(ctx)
	if ferr != nil {
		return engine.CacheInfo{}, ferr
	}
	en.Fingerprint = fp
	return en, nil
}

// prepare runs before a generation request is forwarded, once the arbiter has
// admitted it and made its route the resident. Whatever happens in it, the
// request goes on: errors and panics only cost the cache.
func (c *cache) prepare(ctx context.Context, path string, body []byte, r *route) (t *ticket) {
	t = &ticket{route: r, action: actError}
	defer func() {
		if v := recover(); v != nil {
			c.log.Error("cache panic; request passed through", "path", path, "panic", fmt.Sprint(v))
			c.m.panics.Inc()
			t.action, t.reason = actError, "panic"
		}
	}()
	keys, keyed := c.keys.keys(path, body)
	c.mu.Lock()
	c.inflight++
	c.seq++
	t.seq, t.counted, t.keys = c.seq, true, keys
	busy := c.inflight > 1
	c.mu.Unlock()

	switch {
	case busy:
		t.action, t.reason = actBusy, "concurrent"
		return t
	case !keyed:
		t.action, t.reason = actUnkeyed, "no_messages"
		return t
	}
	st, err := r.eng.Status(ctx)
	if err != nil {
		t.action, t.reason = actPass, "status_"+engine.ErrorKind(err)
		return t
	}
	if c.restarted(r, st) {
		r.live = nil
	}
	// How many tokens the engine holds for the conversation it last served. Strata
	// says; the llama.cpp router says nothing, so the request body sizes it: a
	// session save is keyed by the last key and the engine's own count, the
	// estimate only decides whether a save is worth starting.
	live := st.LiveTokens
	if live <= 0 && c.pol.BodyDivisor > 0 {
		live = len(body) / c.pol.BodyDivisor
	}
	c.mu.Lock()
	r.liveTokens = live
	c.mu.Unlock()
	if r.live != nil && extends(keys, r.live) {
		t.action, t.reason = actContinue, "live"
		return t
	}

	// A switch. An idle save under way finishes first; the engine would queue this
	// request behind it anyway.
	c.saveMu.Lock()
	defer c.saveMu.Unlock()
	en, err := c.identity(ctx, r)
	var fp string
	switch {
	case err == nil:
		fp = c.learn(r, en)
	case !st.Loaded:
		// An unloaded engine has no /metrics to read a fingerprint from. The last one we
		// learned, or the newest file of this route on disk, is what a restore has to
		// match; a cold start with no files has nothing to restore.
		c.mu.Lock()
		fp = r.fp
		c.mu.Unlock()
		if fp == "" {
			fp = c.files.AnyFingerprint(r.ns)
			if fp != "" {
				c.files.Adopt(r.ns, fp)
			}
		}
		if fp == "" {
			t.action, t.reason = actMiss, "engine_unloaded_cold"
			return t
		}
	default:
		t.action, t.reason = actMiss, "engine_"+engine.ErrorKind(err)
		return t
	}
	inRAM := c.inRAM(r, keys)
	var file store.Entry
	var onDisk bool
	if !inRAM {
		// The prompt estimate is the only bound: the engine reuses a restored
		// conversation only when its live count is a prefix of the prompt, so a
		// file longer than the estimate can never be reused. The engine's live
		// count is NOT a bound — a fork request shares the parent's prefix, and
		// the parent can grow past the fork point, so the fork's prompt can be
		// SHORTER than the live count; admitting the parent's file there restores
		// a conversation the engine reads whole. 0 (no estimate, --body-divisor
		// 0): prefix aliases are skipped. Lookup's exact last-key entry — the
		// conversation's own turn boundary — passes unfiltered.
		m := 0
		if c.pol.BodyDivisor > 0 {
			m = len(body) / c.pol.BodyDivisor
		}
		file, onDisk = c.files.Lookup(r.ns, fp, keys, m)
	}
	if r.live != nil {
		t.prev, t.tokens = r.live, live
		t.prevParks = c.willPark(live, en, r)
		if t.prevParks && !onDisk {
			c.skip("parks", live)
		} else {
			c.save(ctx, r, fp, r.live, live, "switch")
		}
	}
	switch {
	case inRAM:
		t.action, t.reason = actRAM, "parked"
	case !onDisk:
		t.action, t.reason = actMiss, "no_copy"
		c.m.prefill(len(keys)) // nothing to restore: the engine reads the whole prompt
	case c.restore(ctx, r, file, t):
		t.action, t.reason = actDisk, "restored"
		t.prevParks = false // a restore drops the outgoing conversation
	default:
		t.action, t.reason = actMiss, "restore_failed"
		c.m.prefill(len(keys))
	}
	return t
}

// touch registers a request the cache cannot key — a body over --max-body-mib,
// streamed through unbuffered — as engine-touching at admission, the bookkeeping
// prepare does for a keyed request: inflight++ and seq++, decremented at finish.
// It matters because an oversized request runs on the very engine a background save
// writes to: idleSave's quiet reads inflight and seq, so a request that is not
// registered lets the idle save of the turn before it start on a busy card and hold
// the save lock for the request's length. The live-conversation belief is cleared
// here too: this engine now holds a conversation the cache cannot name.
func (c *cache) touch(r *route) *ticket {
	c.mu.Lock()
	c.inflight++
	c.seq++
	t := &ticket{route: r, action: actPass, reason: "body_too_large", seq: c.seq, counted: true}
	c.mu.Unlock()
	c.invalidate()
	return t
}

// SaveLive saves the conversation this route holds before the arbiter gives the
// card away. Best effort: the arbiter unloads after it, whatever it cost.
func (c *cache) SaveLive(ctx context.Context, r arbiter.Route) {
	c.mu.Lock()
	rt := c.routes[r.ID]
	if rt == nil {
		c.mu.Unlock()
		return
	}
	live, fp, tokens := rt.live, rt.fp, rt.liveTokens
	c.mu.Unlock()
	if live == nil || fp == "" {
		return
	}
	if st, err := rt.eng.Status(ctx); err == nil && st.Loaded && st.LiveTokens > tokens {
		tokens = st.LiveTokens // Strata says how much it holds; the router says nothing
	}
	if tokens <= 0 {
		return
	}
	c.saveMu.Lock()
	defer c.saveMu.Unlock()
	c.save(context.WithoutCancel(ctx), rt, fp, live, tokens, "yield")
}

// invalidate forgets every route's live conversation: a request the cache didn't
// see into (another endpoint, an oversized body) may have changed one.
func (c *cache) invalidate() {
	c.mu.Lock()
	for _, r := range c.routes {
		r.live = nil
	}
	c.mu.Unlock()
}

// restarted notes a server restart or an unloaded engine: the RAM cache and the
// live conversation of that route are gone.
func (c *cache) restarted(r *route, st engine.Status) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	gone := (r.started != 0 && st.Started != r.started) || !st.Loaded
	r.started = st.Started
	if gone {
		r.live, r.ram = nil, nil
	}
	return gone
}

// learn records the route's fingerprint and how many conversations it has
// parked. A new fingerprint makes every file of that route's other fingerprints
// useless, and adopts the legacy files written for it. Files of other routes are
// never touched: this is the scope that keeps flash-next's sessions alive while
// the 27B route learns its own.
func (c *cache) learn(r *route, en engine.CacheInfo) string {
	fp := en.Fingerprint
	if fp == "" {
		return r.fp // nothing was read: the route keeps the identity it knows
	}
	c.mu.Lock()
	changed := fp != "" && fp != r.fp
	r.fp = fp
	if changed {
		r.ram = nil
	}
	r.ram = r.ram[:min(len(r.ram), en.Parked)]
	c.mu.Unlock()
	if changed {
		c.log.Info("engine fingerprint", "route", r.id, "fingerprint", fp)
		c.files.Adopt(r.ns, fp)
		c.files.KeepFingerprint(r.ns, fp)
	}
	return fp
}

func (c *cache) inRAM(r *route, keys []string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, k := range r.ram {
		if extends(keys, k) {
			return true
		}
	}
	return false
}

// willPark reports whether the engine parks a conversation of this many tokens
// on the switch away from it. It evicts parked ones to make room, so it refuses
// only one bigger than the whole budget, or one that leaves less than the RAM
// floor even with every parked one evicted.
func (c *cache) willPark(tokens int, en engine.CacheInfo, r *route) bool {
	minTokens := c.pol.ParkMinTokens
	if minTokens < 0 {
		minTokens = en.MinTokens
	}
	est := int64(tokens) * c.pol.SnapshotBytesPerToken
	if en.BudgetMiB <= 0 || tokens < minTokens || est > en.BudgetMiB<<20 {
		return false
	}
	avail, err := c.pol.MemAvailable()
	return err == nil && est+en.MinFreeMiB<<20 <= avail+en.ParkedBytes
}

func (c *cache) skip(reason string, tokens int) {
	c.m.skipped.WithLabelValues(reason).Inc()
	c.log.Info("save skipped", "reason", reason, "tokens", tokens)
}

// save writes the live conversation to disk unless it is short, already on disk
// or doesn't fit. It outlives the request's context: the engine finishes a save
// it started anyway, and the index must know about it.
func (c *cache) save(ctx context.Context, r *route, fp string, live []string, tokens int, trigger string) {
	switch {
	case tokens < c.pol.MinTokens:
		c.skip("short", tokens)
		return
	case c.files.Covered(r.ns, fp, live, tokens-c.pol.RegrowTokens):
		c.skip("on_disk", tokens)
		return
	}
	c.mu.Lock()
	bpt := r.bpt
	c.mu.Unlock()
	if bpt <= 0 {
		// The fallback for a route that has never saved. Measured figures: Strata
		// k8v4 is 13.8 KB/token, a llama slot with f16 KV + draft layers is
		// 85.7 KB/token, so 32 KB/token over-reserves Strata by ~2.3x (no_room fires
		// early and long sessions are never saved) and under-reserves llama by ~2.6x
		// (reservations pass a cap the real files then blow through). A route learns
		// its own figure from its first save worth saving.
		bpt = 32 << 10
	}
	if !c.files.Reserve(int64(tokens) * bpt) {
		c.skip("no_room", tokens)
		return
	}
	t0 := time.Now()
	n, _, err := r.eng.Save(context.WithoutCancel(ctx), store.FileName(r.ns, fp, live[len(live)-1], tokens))
	d := time.Since(t0)
	if err != nil {
		c.m.failures.WithLabelValues("save", engine.ErrorKind(err)).Inc()
		c.log.Warn("save failed", "route", r.id, "trigger", trigger, "tokens", tokens, "ms", d.Milliseconds(),
			"reason", engine.ErrorKind(err), "err", err)
		return
	}
	e, err := c.files.Commit(r.ns, fp, live, tokens, n)
	if err != nil {
		c.m.failures.WithLabelValues("save", "index").Inc()
		c.log.Warn("save not indexed", "route", r.id, "err", err)
		return
	}
	c.m.saved(trigger, e.Size, d)
	// A save's bytes-per-token is learned, and clamped. The measured figures are
	// Strata k8v4 at 13.8 KB/token and a llama slot (f16 KV + draft layers) at
	// 85.7 KB/token, so 4 KB..256 KB/token covers both dialects and nothing sane: a
	// file that is mostly fixed overhead (226 MB for 74 tokens = 3 MB/token) is
	// clamped down, and a route whose files are unusually dense is clamped up. A
	// clamped figure keeps the next reserve finite: the poison case read a 168K
	// session as 515 GB, over the cap, so the save was skipped and the session lost.
	if e.Tokens >= c.pol.MinTokens {
		bpt := e.Size / int64(e.Tokens)
		if bpt < minBPT {
			bpt = minBPT
		}
		if bpt > maxBPT {
			bpt = maxBPT
		}
		c.mu.Lock()
		r.bpt = bpt
		c.mu.Unlock()
	}
	c.log.Info("save", "route", r.id, "trigger", trigger, "tokens", e.Tokens, "mb", e.Size>>20, "ms", d.Milliseconds())
}

func (c *cache) restore(ctx context.Context, r *route, e store.Entry, t *ticket) bool {
	uCtx := context.WithoutCancel(ctx)
	t0 := time.Now()
	n, b, err := r.eng.Restore(uCtx, e.Name())
	if r.eng.IsUnloaded(err) {
		// A --lazy engine loads on the next request: wake it, check it is still the
		// engine the file was written for, and give it the restore it refused.
		c.log.Info("waking engine for restore", "route", r.id, "tokens", e.Tokens)
		if err = r.eng.Wake(uCtx); err == nil {
			en, eerr := c.identity(uCtx, r)
			if eerr != nil {
				err = eerr
			} else {
				switch fp := c.learn(r, en); {
				case fp != e.Fingerprint:
					c.log.Warn("woken into another engine; reading fresh", "route", r.id, "wanted", e.Fingerprint, "got", fp)
					c.m.failures.WithLabelValues("restore", "fingerprint").Inc()
					return false
				default:
					t0 = time.Now()
					n, b, err = r.eng.Restore(uCtx, e.Name())
				}
			}
		}
	}
	d := time.Since(t0)
	if err != nil {
		kind := engine.ErrorKind(err)
		c.m.failures.WithLabelValues("restore", kind).Inc()
		c.log.Warn("restore failed; reading fresh", "route", r.id, "tokens", e.Tokens, "ms", d.Milliseconds(),
			"reason", kind, "err", err)
		if kind == "not_found" || kind == "invalid" || kind == "io" { // one it had no RAM for stays
			c.files.Forget(e, "unreadable")
		}
		return false
	}
	if n < e.Tokens/2 {
		// The file on disk restored far less than its name claims: a save cut short.
		// Keep it and every later turn finds it "covered" and is never saved.
		c.log.Warn("truncated session file; forgetting", "route", r.id, "expected", e.Tokens, "actual", n)
		c.files.Forget(e, "truncated")
	} else {
		c.files.Touch(e)
	}
	c.m.restored(b, d)
	t.tokens = n
	c.log.Info("restore", "route", r.id, "tokens", n, "mb", b>>20, "ms", d.Milliseconds())
	return true
}

// finish records the end of t's request: its HTTP status, and whether the client
// went away before the answer was complete.
func (c *cache) finish(t *ticket, status int, clientGone bool) {
	c.m.requests.WithLabelValues(t.action).Inc()
	c.mu.Lock()
	defer c.mu.Unlock()
	if t.counted {
		c.inflight--
	}
	r := t.route
	if status >= 500 {
		r.ram = nil // the engine may have died, and its RAM cache with it
	}
	switch {
	case t.action != actContinue && t.action != actRAM && t.action != actDisk && t.action != actMiss,
		c.seq != t.seq, status != 200, clientGone:
		r.live = nil // another request may have run in between, or this one left no clean state
		return
	}
	if t.prevParks {
		r.ram = append([][]string{t.prev}, dropTurnsOf(r.ram, t.prev)...)
	}
	r.ram = dropTurnsOf(r.ram, t.keys) // now live, not parked
	r.live = t.keys
	c.cur = r
	if c.pol.IdleDelay > 0 && c.ctx.Err() == nil {
		if c.idle != nil && c.idle.Stop() {
			c.wg.Done()
		}
		c.wg.Add(1)
		seq := t.seq
		c.idle = time.AfterFunc(c.pol.IdleDelay, func() { defer c.wg.Done(); c.idleSave(seq) })
	}
}

// Queue is the arbiter's Observers.Queue. The cache keeps the count so a
// background save can step out of the way of requests that are already waiting
// for the card: the save would hold the save lock for its whole duration, and the
// waiting request pays for it before its own swap.
func (c *cache) Queue(n int, _ string) {
	c.mu.Lock()
	c.queued = n
	c.mu.Unlock()
}

// idleSave saves the live conversation of the route the last turn used, if
// nothing has started since the turn numbered seq ended. Two guards keep the save
// from costing the next request more than it gains: a conversation over
// MaxIdleSaveTokens is left to the yield save at the switch, and a queue that is
// not empty means a request is already waiting for the save lock.
func (c *cache) idleSave(seq uint64) {
	defer func() {
		if v := recover(); v != nil {
			c.log.Error("cache panic in an idle save", "panic", fmt.Sprint(v))
			c.m.panics.Inc()
		}
	}()
	c.saveMu.Lock()
	defer c.saveMu.Unlock()
	c.mu.Lock()
	r, live, fp, est := c.cur, c.cur.live, c.cur.fp, c.cur.liveTokens
	queued := c.queued
	quiet := c.seq == seq && c.inflight == 0 && live != nil && fp != "" && c.ctx.Err() == nil
	c.mu.Unlock()
	if !quiet {
		return
	}
	if queued > 0 {
		c.skip("queue_busy", est)
		return
	}
	if st, err := r.eng.Status(c.ctx); err == nil && !c.restarted(r, st) {
		tokens := st.LiveTokens
		if tokens < est {
			tokens = est // the router reports none: keep the estimate
		}
		if c.pol.MaxIdleSaveTokens > 0 && tokens > c.pol.MaxIdleSaveTokens {
			c.skip("idle_too_long", tokens)
			return
		}
		c.save(c.ctx, r, fp, live, tokens, "idle")
	}
}

// dropTurnsOf removes the conversations keys extends (earlier turns of it).
func dropTurnsOf(l [][]string, keys []string) [][]string {
	out := l[:0:0]
	for _, r := range l {
		if !extends(keys, r) {
			out = append(out, r)
		}
	}
	return out
}
