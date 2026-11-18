// Package arbiter owns the one GPU. It treats each (backend, model) pair as a
// distinct resident identity, keeps one route resident at a time, and gives the
// VRAM to another route only between requests: drain the running turn, save the
// session, unload, check the free VRAM, load. It never runs inside a forwarded
// request, so a swap is always paid for between requests. The admission of a
// request for the resident route can also pay for a load, but only when that
// model has disappeared from the card (the router's idle sleep, a dead engine, a
// swap refused after the unload) — a request never takes the card.
//
// The queue rule is resident-drain: requests for the route that is resident now
// are admitted ahead of a request that wants to take the card, so a 27B parent
// calling a flash-next child does not ping-pong the card between the two
// engines while flash-next still has turns queued. The drain is bounded by
// MaxResidentStreak, so a route cannot starve the card forever.
package arbiter

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/frodosik/gpu-arbiter/internal/engine"
)

// Route is one resident identity: a backend and the model it is running.
type Route struct {
	ID        string // "llama/Swift-27B": the session namespace and the metric label
	Dialect   string // "strata" or "llama", for logs
	Model     string // the model name a client puts in its request
	Base      string // the server URL the proxy forwards to, http://127.0.0.1:8001
	APIKey    string // the bearer token this backend needs, if any
	Namespace string // the store namespace: 12 hex, sanitized from the route id
	Need      int64  // bytes of VRAM this route needs (weights + KV); 0: gate on margin only
	Engine    engine.Engine
}

// SetObservers installs the observer after construction: the cache is built over
// the arbiter, so a wire that needs both can only be made afterwards.
func (a *Arbiter) SetObservers(o Observers) {
	a.mu.Lock()
	a.opt.Observers = o
	a.mu.Unlock()
}

// Observers reports swaps to the metrics; optional.
type Observers interface {
	Swap(from, to string, seconds float64)
	SwapBlocked(reason string)
	Queue(n int, resident string)
	// Reloaded says the resident route's model was loaded from cold, so what the
	// engine held for the last conversation is gone.
	Reloaded(route string)
}

// Options configure the arbiter.
type Options struct {
	Routes []Route
	// Strict makes a request name one of the routes: a request whose model is not
	// one of them is refused with NoRoute, not answered by the resident, and a
	// control POST that names a model of another route is refused too — the rule is
	// one rule across every path. Set when --route flags were configured. The single
	// route built from --upstream is not strict: it answers every request, and the
	// proxy logs the model each request asked for so the aliasing is visible in the
	// log.
	Strict bool
	// MaxResidentStreak: admissions of the resident before a waiter may take the
	// card; <= 0: unbounded
	MaxResidentStreak int
	DrainWait         time.Duration // how long an unload waits for a running turn (Strata's ?wait=)
	LoadTimeout       time.Duration // the whole swap: drain, VRAM gate, load
	VRAMMargin        int64         // bytes that must stay free on the card after a load
	VRAMFree          func() (int64, error)
	PollInterval      time.Duration
	// InitialResident is the route that owns the card at startup. A Strata server
	// with --lazy has not loaded its model yet, so the arbiter treats it as resident
	// and loads it on that route's first admission (ensureLoaded), not as idle.
	InitialResident string
	Log             *slog.Logger
	Observers       Observers
	// SaveLive lets the cache write the resident's live conversation to its
	// session file before the card is given away. Best effort; nil: no save.
	SaveLive func(ctx context.Context, r Route)
}

// Arbiter is the lease: who holds the card, the queue of waiters, and the swap.
// It is safe for concurrent use.
type Arbiter struct {
	opt     Options
	byID    map[string]Route
	byModel map[string]Route

	mu        sync.Mutex
	cond      *sync.Cond
	resident  string    // route id, "": the card is idle
	streak    int       // admissions of the resident since the last swap
	queued    []*waiter // waiting requests, in arrival order
	next      int       // the next waiter token
	busy      bool      // a request is in flight on the card: one at a time
	vramFails int
}

// waiter is one queued request. The scheduler marks exactly one of them chosen
// when the slot is free; the chosen one runs its own swap, then forwards.
type waiter struct {
	route  string // route id this request asks for
	token  int
	chosen bool
	err    error
}

// Routes is the route list, in the order given: the first is the initial
// resident candidate.
func (a *Arbiter) Routes() []Route { return a.opt.Routes }

// SetSaveLive installs the callback the arbiter calls to save the resident's
// live conversation before giving the card away. It is set once, at startup,
// before any request is admitted.
func (a *Arbiter) SetSaveLive(f func(ctx context.Context, r Route)) {
	a.mu.Lock()
	a.opt.SaveLive = f
	a.mu.Unlock()
}

// New returns an arbiter over opt.Routes. The first route is the initial
// resident candidate; the card is idle until the first request takes it.
func New(opt Options) *Arbiter {
	if opt.Log == nil {
		opt.Log = slog.Default()
	}
	if opt.VRAMFree == nil {
		opt.VRAMFree = engine.FreeVRAM
	}
	if opt.PollInterval <= 0 {
		opt.PollInterval = 2 * time.Second
	}
	if opt.LoadTimeout <= 0 {
		opt.LoadTimeout = 10 * time.Minute
	}
	if opt.DrainWait <= 0 {
		opt.DrainWait = 300 * time.Second
	}
	a := &Arbiter{opt: opt, byID: map[string]Route{}, byModel: map[string]Route{}}
	a.cond = sync.NewCond(&a.mu)
	for _, r := range opt.Routes {
		a.byID[r.ID] = r
		a.byModel[r.Model] = r
	}
	if _, ok := a.byID[opt.InitialResident]; ok {
		a.resident = opt.InitialResident
	}
	return a
}

// RouteOf resolves the route a control call belongs to: the resident, and the first
// route when the card is idle. Control calls (/slots, /props, /load, a raw prompt)
// carry no model, and a per-model endpoint on an idle router loads that model, so
// this is the only routing a control call gets. Generation requests name a model and
// go through Acquire, which refuses a model it does not serve when Strict is set.
func (a *Arbiter) RouteOf(model string) Route {
	a.mu.Lock()
	defer a.mu.Unlock()
	if r, ok := a.byModel[model]; ok {
		return r
	}
	if a.resident != "" {
		return a.byID[a.resident]
	}
	for _, r := range a.opt.Routes {
		return r
	}
	return Route{}
}

// Resident is the route holding the card now, "" when idle.
func (a *Arbiter) Resident() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.resident
}

// Serves reports whether model is one of the arbiter's routes. It is the strict
// rule's source, and load-bearing on the one path Acquire does not cover: a control
// POST that names a model is answered 404 when this arbiter does not serve it, and
// 404 again when its route is not the resident. The rule has to be readable where
// the request is forwarded, not only inside Acquire.
func (a *Arbiter) Serves(model string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	_, ok := a.byModel[model]
	return ok
}

// Strict reports that a request has to name one of the arbiter's routes. It is set
// when the routes came from --route flags. Load-bearing: the rule is read in two
// places — Acquire for a generation request, and the proxy's control path for a
// POST that names a model — and both read this one source.
func (a *Arbiter) Strict() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.opt.Strict
}

// Acquire makes model's route the resident, waiting behind the resident's queued
// requests, and returns the route with the release that gives the slot back. One
// request runs at a time; the queue rule picks which one, and the chosen request
// does its own swap, so a swap is always paid for by the request that asked for
// the model. It is the arbiter's only entry point: it runs between requests,
// never inside a forwarded one.
func (a *Arbiter) Acquire(ctx context.Context, model string) (Route, func(), error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	r, ok := a.byModel[model]
	if !ok {
		if a.opt.Strict {
			// The arbiter serves exactly the models it was given. A request naming
			// a model that is not one of them, or naming none, is refused: answering
			// it with the route that happens to hold the card gives the client a
			// different model under the label it asked for, and its token counts,
			// session files and the harness's model name all inherit the lie.
			return Route{}, nil, &NoRoute{Model: model}
		}
		// Single-route mode: the one route answers every request, whatever the body
		// names. The proxy logs the requested model, so the aliasing is visible.
		switch {
		case a.resident != "":
			r = a.byID[a.resident]
		case len(a.opt.Routes) > 0:
			r = a.opt.Routes[0]
		default:
			return Route{}, nil, &NoRoute{Model: model}
		}
	}
	w := &waiter{route: r.ID, token: a.next}
	a.next++
	a.queued = append(a.queued, w)
	a.observe()
	// A waiter parked in cond.Wait has to hear about its own request ending.
	go func() {
		<-ctx.Done()
		a.mu.Lock()
		a.cond.Broadcast()
		a.mu.Unlock()
	}()
	for {
		if ctx.Err() != nil {
			a.unqueue(w)
			a.observe()
			a.cond.Broadcast()
			return Route{}, nil, ctx.Err()
		}
		if !a.busy {
			a.pick()
		}
		if w.chosen {
			if a.resident != r.ID {
				w.err = a.swap(ctx, r)
				if w.err != nil {
					// The slot is given back: the next waiter is picked.
					a.unqueue(w)
					a.busy = false
					a.observe()
					a.cond.Broadcast()
					return Route{}, nil, w.err
				}
			} else if w.err = a.ensureLoaded(ctx, r); w.err != nil {
				a.unqueue(w)
				a.busy = false
				a.observe()
				a.cond.Broadcast()
				return Route{}, nil, w.err
			}
			break
		}
		a.cond.Wait()
	}
	a.unqueue(w)
	a.busy = true
	if a.resident == r.ID {
		a.streak++
	}
	a.observe()
	release := func() {
		a.mu.Lock()
		a.busy = false
		a.cond.Broadcast()
		a.mu.Unlock()
	}
	return r, release, nil
}

// pick marks the waiter that runs next. The resident's queued turns go first, so
// the card does not oscillate; once the resident's streak reaches the cap, the
// first waiter that wants another route is chosen, so a route cannot starve.
// Call with mu held and the slot free.
func (a *Arbiter) pick() *waiter {
	if len(a.queued) == 0 {
		return nil
	}
	capped := a.opt.MaxResidentStreak > 0 && a.streak >= a.opt.MaxResidentStreak
	winner := a.queued[0]
	if a.resident != "" && !capped {
		for _, w := range a.queued {
			if w.route == a.resident {
				winner = w
				break
			}
		}
	}
	for _, w := range a.queued {
		if w.route != a.resident && (a.resident == "" || capped) {
			winner = w
			break
		}
	}
	// Claim the slot here: the chosen request has not taken it yet, and a second
	// pick must not mark another waiter in the same moment.
	winner.chosen = true
	a.busy = true
	// Wake everyone: the picker is often not the winner (an early waiter runs the
	// scheduler and goes back to sleep), so the chosen request has to be woken
	// too, or it parks in cond.Wait forever.
	a.cond.Broadcast()
	return winner
}

// ensureLoaded loads the resident route's model when the engine says it is not
// loaded: the router's idle sleep, a dead engine, or a swap that was refused
// after the unload, all leave the arbiter's resident route with nothing to
// answer. Only a positive "not loaded" answer counts: a status read that fails
// (a 3 s timeout, a router blip, a non-200) is not evidence the model is gone,
// and loading on that evidence would clear a live conversation that is there.
// One LoadTimeout covers the gate and the load together, as the option says.
func (a *Arbiter) ensureLoaded(ctx context.Context, r Route) error {
	st, err := r.Engine.Status(ctx)
	if err != nil {
		return nil
	}
	if st.Loaded {
		return nil
	}
	window, cancel := context.WithTimeout(ctx, a.opt.LoadTimeout)
	defer cancel()
	if err := a.waitForVRAM(window, r); err != nil {
		a.blocked("vram")
		a.log().Warn("not enough VRAM free to reload", "route", r.ID, "need", r.Need, "margin", a.opt.VRAMMargin, "err", err)
		return err
	}
	if err := r.Engine.Load(window); err != nil {
		a.blocked("reload")
		// Fail open: an engine that cannot load answers its own reason, and that is
		// the client's best answer. The admission is not refused on a failed load,
		// and the route's stale live/ram beliefs are dropped by finish when the
		// engine's 5xx reaches the client. Reloaded fires only on a load that
		// succeeded: a failed load did not replace any conversation.
		a.log().Warn("reload refused; the request goes to the engine", "route", r.ID, "err", err)
		return nil
	}
	if a.opt.Observers != nil {
		a.opt.Observers.Reloaded(r.ID)
	}
	a.log().Info("resident model reloaded", "route", r.ID)
	return nil
}

// swap gives the card to to. The order is fixed: save what the resident holds,
// drain and unload it, check the free VRAM, then load. Nothing is polled on a
// model the arbiter does not own.
func (a *Arbiter) swap(ctx context.Context, to Route) error {
	from := a.resident
	if from == to.ID {
		return nil
	}
	t0 := time.Now()
	if from != "" {
		cur := a.byID[from]
		if a.opt.SaveLive != nil {
			a.opt.SaveLive(ctx, cur)
		}
		if err := cur.Engine.Unload(ctx, a.opt.DrainWait); err != nil {
			a.blocked("unload")
			a.log().Warn("yield refused", "route", cur.ID, "err", err)
			return err
		}
	}
	if err := a.waitForVRAM(ctx, to); err != nil {
		a.blocked("vram")
		a.log().Warn("not enough VRAM free to load", "route", to.ID, "need", to.Need, "margin", a.opt.VRAMMargin, "err", err)
		return err
	}
	if err := to.Engine.Load(ctx); err != nil {
		a.blocked("load")
		a.log().Warn("load refused", "route", to.ID, "err", err)
		return err
	}
	a.resident = to.ID
	a.streak = 0
	d := time.Since(t0)
	a.log().Info("swap", "from", from, "to", to.ID, "ms", d.Milliseconds())
	if a.opt.Observers != nil {
		a.opt.Observers.Swap(from, to.ID, d.Seconds())
	}
	return nil
}

// waitForVRAM gates on the card: bytes free must cover the route's need plus the
// margin. A card whose figure cannot be read three times in a row is trusted to
// the operator, so an unreadable sysfs does not wedge the queue.
func (a *Arbiter) waitForVRAM(ctx context.Context, to Route) error {
	need := to.Need + a.opt.VRAMMargin
	if need <= 0 {
		return nil
	}
	deadline := time.Now().Add(a.opt.LoadTimeout)
	for {
		free, err := a.opt.VRAMFree()
		switch {
		case err == nil && free >= need:
			a.vramFails = 0
			return nil
		case err != nil:
			a.vramFails++
			if a.vramFails >= 3 {
				a.log().Warn("free VRAM unreadable; loading on the operator's word", "err", err)
				return nil
			}
		}
		left := time.Until(deadline)
		if left <= 0 {
			return context.DeadlineExceeded
		}
		if left < a.opt.PollInterval {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			time.Sleep(left)
			continue
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(a.opt.PollInterval):
		}
	}
}

// unqueue drops one waiter; mu held. A request that gave up, or a swap that was
// refused, leaves no phantom behind: the queue is the order the next pick sees.
func (a *Arbiter) unqueue(w *waiter) {
	for i, q := range a.queued {
		if q == w {
			a.queued = append(a.queued[:i], a.queued[i+1:]...)
			return
		}
	}
}

func (a *Arbiter) blocked(reason string) {
	if a.opt.Observers != nil {
		a.opt.Observers.SwapBlocked(reason)
	}
}

func (a *Arbiter) observe() {
	if a.opt.Observers != nil {
		a.opt.Observers.Queue(len(a.queued), a.resident)
	}
}

func (a *Arbiter) log() *slog.Logger { return a.opt.Log }

// NoRoute is a model this arbiter does not serve. The proxy answers it with 404.
type NoRoute struct{ Model string }

func (e *NoRoute) Error() string {
	if e.Model == "" {
		return "the request names no model"
	}
	return "model " + e.Model + " is not served by this arbiter"
}

// NotResident is a model this arbiter serves on a route that does not hold the
// card. Naming it is the aliasing strict mode exists to catch: answered by the
// resident, the client gets a model the card does not hold; answered by its own
// route, that model is loaded on an idle card with no yield save of the
// conversation the card holds. The proxy answers 404 and touches no engine.
type NotResident struct{ Model, ResidentModel string }

func (e *NotResident) Error() string {
	if e.ResidentModel == "" {
		return "the card is idle, so model " + e.Model + " has no resident route"
	}
	return "model " + e.Model + " is not the resident; " + e.ResidentModel + " holds the card"
}
