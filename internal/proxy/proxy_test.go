package proxy

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/frodosik/gpu-arbiter/internal/arbiter"
	"github.com/frodosik/gpu-arbiter/internal/engine"
	"github.com/frodosik/gpu-arbiter/internal/store"
	"github.com/frodosik/gpu-arbiter/internal/strata/stratatest"
)

type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}
func (s *syncBuffer) String() string { s.mu.Lock(); defer s.mu.Unlock(); return s.b.String() }

type env struct {
	t     *testing.T
	f     *stratatest.Fake
	p     *Proxy
	st    *store.Store
	url   string
	dir   string
	logs  *syncBuffer
	mu    sync.Mutex
	avail int64
}

// setup starts a fake Strata and the proxy in front of it. Idle saves are off
// unless idle > 0. One route, no VRAM gate: the single-engine cache behaves as
// it did before the arbiter existed.
func setup(t *testing.T, idle time.Duration, maxBody int64) *env {
	t.Helper()
	return setupCap(t, idle, maxBody, 64<<30)
}

// setupCap is setup with the store's size cap: a cap the saves overshoot is the
// production shape, and the only way to watch the cap and the disk agree.
func setupCap(t *testing.T, idle time.Duration, maxBody, maxBytes int64) *env {
	t.Helper()
	e := &env{t: t, dir: t.TempDir(), logs: &syncBuffer{}, avail: 8 << 30}
	e.f = stratatest.New(e.dir)
	up := httptest.NewServer(e.f)
	t.Cleanup(up.Close)
	log := slog.New(slog.NewJSONHandler(e.logs, nil))
	e.st = store.New(store.Options{Dir: e.dir, MaxBytes: maxBytes, MaxAge: time.Hour, Log: log,
		FreeSpace: func(string) (int64, error) { return 1 << 40, nil }})
	if err := e.st.Load(); err != nil {
		t.Fatal(err)
	}
	e.p = New(Options{Arbiter: oneRoute(up.URL, "test-model", log), Store: e.st, MaxBody: maxBody,
		Metrics: NewMetrics(prometheus.NewRegistry()), Log: log,
		Policy: Policy{MinTokens: 15000, ParkMinTokens: -1, RegrowTokens: 4096, IdleDelay: idle,
			SnapshotBytesPerToken: 64 << 10, BodyDivisor: 4,
			MemAvailable: func() (int64, error) { e.mu.Lock(); defer e.mu.Unlock(); return e.avail, nil }}})
	px := httptest.NewServer(e.p)
	t.Cleanup(func() { px.Close(); _ = e.p.Close(context.Background()) })
	e.url = px.URL
	return e
}

// oneRoute is the arbiter over a single route, with no VRAM gate. The route is
// resident from the start. A --lazy Strata takes its model on the first request,
// so a cold engine is loaded by the arbiter at admission and the cache's restore
// path stays reachable.
func oneRoute(base, model string, log *slog.Logger) *arbiter.Arbiter {
	id := "strata/" + model
	return arbiter.New(arbiter.Options{
		Routes: []arbiter.Route{{ID: id, Dialect: "strata", Model: model, Base: base,
			Namespace: store.NamespaceOf("strata", model),
			Engine:    engine.NewStrata(engine.Options{ID: id, Model: model, Base: base})}},
		InitialResident: id,
		Log:             log,
		VRAMFree:        func() (int64, error) { return 1 << 40, nil },
	})
}

func (e *env) setAvail(n int64) { e.mu.Lock(); e.avail = n; e.mu.Unlock() }

func conversation(name string, turns int) string {
	msgs := []string{fmt.Sprintf(`{"role":"system","content":"%s"}`, name)}
	for i := 0; i < turns; i++ {
		if i > 0 {
			msgs = append(msgs, fmt.Sprintf(`{"role":"assistant","content":"answer-text-%d"}`, i))
		}
		msgs = append(msgs, fmt.Sprintf(`{"role":"user","content":"u%d"}`, i))
	}
	return `{"stream":true,"messages":[` + strings.Join(msgs, ",") + `]}`
}

// post sends a request and reads the whole answer; it returns the status and the SSE event count.
func (e *env) post(ctx context.Context, path, body string) (int, int) {
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, e.url+path, strings.NewReader(body))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, 0
	}
	defer func() { _ = resp.Body.Close() }()
	n := 0
	for sc := bufio.NewScanner(resp.Body); sc.Scan(); {
		if strings.HasPrefix(sc.Text(), "data: ") {
			n++
		}
	}
	return resp.StatusCode, n
}

// turn sends name's conversation at that many turns and returns the cache's action.
func (e *env) turn(name string, turns int) string {
	e.t.Helper()
	if status, events := e.post(context.Background(), "/v1/chat/completions", conversation(name, turns)); status != 200 || events != 4 {
		e.t.Fatalf("%s/%d: status %d, %d events", name, turns, status, events)
	}
	return e.last()
}

func (e *env) last() string {
	e.p.mu.Lock()
	defer e.p.mu.Unlock()
	return e.p.recent[len(e.p.recent)-1].Action
}

func (e *env) files() int {
	m, _ := filepath.Glob(filepath.Join(e.dir, "dc-*.session"))
	return len(m)
}

// nameOf is the one file on disk, for following it across a turn.
func (e *env) nameOf() string {
	m, _ := filepath.Glob(filepath.Join(e.dir, "dc-*.session"))
	if len(m) != 1 {
		e.t.Fatalf("%d files on disk, want 1", len(m))
	}
	return filepath.Base(m[0])
}

func (e *env) exists(name string) bool {
	_, err := os.Stat(filepath.Join(e.dir, name))
	return err == nil
}

// restores counts the restore calls the fake saw.
func restores(actions []string) (n int) {
	for _, a := range actions {
		if strings.HasPrefix(a, "restore ") {
			n++
		}
	}
	return n
}

func saves(actions []string) (n int) {
	for _, a := range actions {
		if strings.HasPrefix(a, "save ") {
			n++
		}
	}
	return n
}

func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	for deadline := time.Now().Add(3 * time.Second); time.Now().Before(deadline); time.Sleep(10 * time.Millisecond) {
		if cond() {
			return
		}
	}
	t.Fatalf("timed out waiting for %s", what)
}

func want(t *testing.T, step, got, want string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %q, want %q", step, got, want)
	}
}

func TestSavesOnlyWhatWouldBeLost(t *testing.T) {
	e := setup(t, 0, 64<<20)
	want(t, "first request", e.turn("A", 1), actMiss)
	want(t, "next turn", e.turn("A", 2), actContinue)
	e.f.Take()

	e.f.Set(20000, 0) // A: 20000 tokens, a ~1.3 GB snapshot: it parks
	e.turn("B", 1)
	if n := saves(e.f.Take()); n != 0 {
		t.Fatalf("A parks, yet %d saves at the switch", n)
	}
	e.f.Set(20000, 1)
	want(t, "back to A", e.turn("A", 3), actRAM)

	e.f.Set(100000, 1) // 6.4 GB: over the 3.5 GiB budget, never parks
	e.turn("C", 1)
	if n := saves(e.f.Take()); n != 1 {
		t.Fatalf("too big to park: %d saves at the switch, want 1", n)
	}
	e.setAvail(2 << 30) // under the RAM floor
	e.f.Set(20000, 0)
	e.turn("D", 1)
	if n := saves(e.f.Take()); n != 1 {
		t.Fatalf("no room above the RAM floor: %d saves, want 1", n)
	}
	e.setAvail(8 << 30)

	// RAM lost everything: A comes back from disk, and D is saved first since the restore drops it
	want(t, "A from disk", e.turn("A", 4), actDisk)
	if got := e.f.Take(); len(got) != 3 || !strings.HasPrefix(got[0], "save ") || !strings.HasPrefix(got[1], "restore ") {
		t.Fatalf("A from disk: %q", got)
	}
	e.f.Set(500, 0)
	e.turn("E", 1)
	e.turn("F", 1)
	if n := saves(e.f.Take()); n != 0 {
		t.Fatal("a short conversation was saved")
	}
}

func TestIdleSave(t *testing.T) {
	e := setup(t, 50*time.Millisecond, 64<<20)
	e.f.Set(20000, 0)
	e.turn("A", 1)
	e.turn("A", 2) // a quick next turn cancels the first idle save
	eventually(t, "the idle save", func() bool { return e.files() == 1 })
	if n := saves(e.f.Take()); n != 1 {
		t.Fatalf("%d idle saves, want 1", n)
	}
	e.f.Set(22000, 0) // past the file's last key: the conversation is not covered
	e.turn("A", 3)
	time.Sleep(200 * time.Millisecond)
	if n := saves(e.f.Take()); n != 1 {
		t.Fatal("the grown conversation was not saved at its own boundary")
	}
	e.f.Set(30000, 0)
	e.turn("A", 4)
	eventually(t, "a new save", func() bool { return saves(e.f.Take()) == 1 })
	// Every save keys its own turn boundary, and the older files survive as the
	// turn boundaries their keys point to in the prefix index.
	if e.files() != 3 {
		t.Errorf("%d files, want the save of each turn boundary", e.files())
	}
}

// saveAB leaves A and B both on disk (via idle saves) with B live.
func saveAB(e *env) {
	e.f.Set(20000, 0)
	e.turn("A", 1)
	eventually(e.t, "A's file", func() bool { return e.files() == 1 })
	e.turn("B", 1)
	eventually(e.t, "B's file", func() bool { return e.files() == 2 })
}

func TestOverlappingRequests(t *testing.T) {
	e := setup(t, 10*time.Millisecond, 64<<20)
	saveAB(e)

	// a request that arrives mid-restore is passed through untouched
	entered := e.f.Entered()
	e.f.Update(func(f *stratatest.Fake) { f.SlotDelay = 300 * time.Millisecond })
	done := make(chan int)
	go func() { s, _ := e.post(context.Background(), "/v1/chat/completions", conversation("A", 2)); done <- s }()
	want(t, "slot call", <-entered, "restore")
	e.f.Update(func(f *stratatest.Fake) { f.SlotDelay = 0 })
	t0 := time.Now()
	if s, _ := e.post(context.Background(), "/v1/chat/completions", conversation("B", 2)); s != 200 {
		t.Fatalf("status %d", s)
	}
	// The card runs one request at a time: B waited for A's restore and then
	// switched from disk on a settled cache, never mid-restore.
	if time.Since(t0) < 300*time.Millisecond {
		t.Error("the second request ran while the first's restore was under way")
	}
	want(t, "after the restore", e.last(), actDisk)
	if s := <-done; s != 200 {
		t.Fatalf("restored request: status %d", s)
	}
	// The restores are the ones the two requests asked for, once each; the saves
	// that follow are the idle saves of the two turns, which is what the queue
	// order is for.
	if n := restores(e.f.Take()); n != 2 {
		t.Errorf("restores: %d, want one per request", n)
	}
	if e.p.cache.routes["strata/test-model"].live == nil {
		t.Error("the cache lost track of the conversation after the restore")
	}

	// the same conversation from four clients at once: all answered
	e.f.Update(func(f *stratatest.Fake) { f.GenDelay = 100 * time.Millisecond })
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Go(func() {
			if s, _ := e.post(context.Background(), "/v1/chat/completions", conversation("C", 1)); s != 200 {
				t.Errorf("concurrent request: status %d", s)
			}
		})
	}
	wg.Wait()
}

func TestRequestDuringIdleSave(t *testing.T) {
	e := setup(t, 20*time.Millisecond, 64<<20)
	e.f.Set(20000, 0)
	entered := e.f.Entered()
	e.f.Update(func(f *stratatest.Fake) { f.SlotDelay = 300 * time.Millisecond })
	e.turn("A", 1)
	want(t, "slot call", <-entered, "save")
	t0 := time.Now()
	want(t, "a continuing turn", e.turn("A", 2), actContinue)
	if time.Since(t0) > 250*time.Millisecond {
		t.Error("a continuing turn waited for the idle save in the proxy")
	}
	e.f.Update(func(f *stratatest.Fake) { f.SlotDelay = 0 })
	want(t, "a switch after the save", e.turn("B", 1), actMiss)
}

func TestClientCancel(t *testing.T) {
	e := setup(t, 10*time.Millisecond, 64<<20)
	saveAB(e)

	// mid-restore: the restore finishes anyway and the file stays
	entered := e.f.Entered()
	e.f.Update(func(f *stratatest.Fake) { f.SlotDelay = 300 * time.Millisecond })
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { e.post(ctx, "/v1/chat/completions", conversation("A", 2)); close(done) }()
	want(t, "slot call", <-entered, "restore")
	cancel()
	<-done
	e.f.Update(func(f *stratatest.Fake) { f.SlotDelay = 0 })
	eventually(t, "the restore", func() bool { return strings.Contains(e.logs.String(), `"msg":"restore"`) })
	if e.files() != 2 {
		t.Errorf("%d files after a canceled restore, want 2", e.files())
	}

	// mid-stream: the cut-off state is not trusted as live
	e.turn("A", 2)
	e.f.Update(func(f *stratatest.Fake) { f.GenDelay = 5 * time.Second })
	ctx, cancel = context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, e.url+"/v1/chat/completions", strings.NewReader(conversation("A", 3)))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	line, _ := bufio.NewReader(resp.Body).ReadString('\n')
	want(t, "the first event is not held back", line, "data: {\"i\":0}\n")
	cancel()
	_ = resp.Body.Close()
	eventually(t, "the request to end", func() bool { return strings.Contains(e.logs.String(), `"client_gone":true`) })
	e.f.Update(func(f *stratatest.Fake) { f.GenDelay = 0 })
	if got := e.turn("A", 4); got == actContinue {
		t.Error("a cut-off answer was trusted as the live state")
	}
}

// stream sends a request, reads its answer until a data line carries stop, and closes
// the socket right there; it returns the SSE lines read.
func (e *env) stream(path, body, stop string) []string {
	e.t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, e.url+path, strings.NewReader(body))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		e.t.Fatal(err)
	}
	var lines []string
	for sc := bufio.NewScanner(resp.Body); sc.Scan(); {
		if line := sc.Text(); strings.HasPrefix(line, "data: ") {
			lines = append(lines, line)
			if strings.Contains(line, stop) {
				break
			}
		}
	}
	cancel()
	_ = resp.Body.Close()
	return lines
}

// lastReq is the most recent request line in the log.
func (e *env) lastReq() string {
	lines := strings.Split(e.logs.String(), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if strings.Contains(lines[i], `"msg":"request"`) {
			return lines[i]
		}
	}
	return ""
}

// A marker quoted inside streamed content is not the end of the stream, even when
// the rolling tail starts right at it.
func TestStatusWriterDone(t *testing.T) {
	rec := httptest.NewRecorder()
	s := &statusWriter{ResponseWriter: rec, status: http.StatusOK, tail: []byte{'\n'}}

	marker := "data: [DONE]"
	rest := `"}}]}` + "\n\n" // closes the JSON the marker sits inside
	chunk := `data: {"choices":[{"delta":{"content":"example: ` + marker +
		strings.Repeat("p", tailKeep-len(marker)-len(rest)) + rest
	if _, err := s.Write([]byte(chunk)); err != nil {
		t.Fatal(err)
	}
	if s.done {
		t.Fatal("a marker quoted inside content was taken for the end of the stream")
	}
	if !bytes.HasPrefix(s.tail, []byte(marker)) {
		t.Fatalf("the tail is %q, want it to start at the quoted marker", s.tail)
	}
	if _, err := s.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"next\"}}]}\n\n")); err != nil {
		t.Fatal(err)
	}
	if s.done {
		t.Fatal("the newline seeded at stream start made the quoted marker read as a stream end")
	}
	if _, err := s.Write([]byte("data: [DONE]\n\n")); err != nil {
		t.Fatal(err)
	}
	if !s.done {
		t.Fatal("the real end of the stream was not seen")
	}
}

func TestClientGoneAfterCompletion(t *testing.T) {
	e := setup(t, 10*time.Millisecond, 64<<20)
	e.f.Set(20000, 0)

	// an agent harness that closes the socket as soon as the stream finished: the answer
	// is complete, so the conversation stays live and the idle save runs
	for i, c := range []struct{ path, body, stop string }{
		{"/v1/chat/completions", conversation("A", 1), "[DONE]"},
		{"/v1/chat/completions", conversation("B", 1), `"finish_reason":"tool_calls"`},
		{"/v1/messages", `{"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`, `"type":"message_stop"`},
		{"/v1/responses", `{"input":"hi"}`, `"type":"response.completed"`},
	} {
		e.f.Update(func(f *stratatest.Fake) { f.Finish, f.TailDelay = "", 300*time.Millisecond })
		if c.stop == `"finish_reason":"tool_calls"` {
			e.f.Update(func(f *stratatest.Fake) { f.Finish = "tool_calls" })
		}
		lines := e.stream(c.path, c.body, c.stop)
		if last := lines[len(lines)-1]; !strings.Contains(last, c.stop) {
			t.Fatalf("%s: stream ended at %q, want %s", c.path, last, c.stop)
		}
		if c.stop == `"type":"response.completed"` && len(lines[len(lines)-1]) < 1024 {
			t.Fatalf("the completed event is %d bytes: the marker must be buried by its payload", len(lines[len(lines)-1]))
		}
		eventually(t, "the request to end", func() bool {
			return strings.Contains(e.lastReq(), `"client_gone":false`)
		})
		if status, _ := e.post(context.Background(), c.path, c.body); status != 200 {
			t.Fatalf("%s: the turn after a clean close: status %d", c.path, status)
		}
		want(t, c.path+" after a clean close", e.last(), actContinue) // the state was trusted
		eventually(t, "the idle save of "+c.path, func() bool { return e.files() == i+1 })
	}

	// a cancel while the stream is still running is a mid-stream cancellation: the state
	// is forgotten, so the next turn is read fresh
	e.f.Update(func(f *stratatest.Fake) { f.Finish, f.TailDelay, f.GenDelay = "", 0, 300*time.Millisecond })
	e.stream("/v1/chat/completions", conversation("G", 1), `"i":0`)
	eventually(t, "the canceled request to end", func() bool {
		return strings.Contains(e.lastReq(), `"client_gone":true`)
	})
	e.post(context.Background(), "/v1/chat/completions", conversation("G", 2))
	want(t, "the turn after a mid-stream cancel", e.last(), actMiss)
	e.f.Update(func(f *stratatest.Fake) { f.GenDelay = 0 })

	// a model quoting "data: [DONE]" inside the answer is not the end of the stream
	e.f.Update(func(f *stratatest.Fake) { f.GenDelay, f.Content = 300*time.Millisecond, "talking about data: [DONE]" })
	lines := e.stream("/v1/chat/completions", conversation("H", 1), "data: [DONE]")
	if last := lines[len(lines)-1]; !strings.Contains(last, `"content"`) {
		t.Fatalf("the marker was not streamed inside content: %q", last)
	}
	eventually(t, "the canceled request to end", func() bool {
		return strings.Contains(e.lastReq(), `"client_gone":true`)
	})
	e.post(context.Background(), "/v1/chat/completions", conversation("H", 2))
	want(t, "the turn after a cancel at a quoted marker", e.last(), actMiss)
	e.f.Update(func(f *stratatest.Fake) { f.GenDelay, f.Content = 0, "" })
}

// A cold engine with the file on disk: the arbiter loads the model at admission,
// and the restore runs on the loaded engine. No request is spent waking it, and
// the file stays.
func TestColdEngineIsLoadedThenRestored(t *testing.T) {
	e := setup(t, 50*time.Millisecond, 64<<20)
	e.f.Set(20000, 0)
	e.turn("A", 1)
	eventually(t, "A's file", func() bool { return e.files() == 1 })

	// the engine unloaded (--lazy, idle_unload_s): /metrics has no context or KV type
	e.f.Take()
	e.f.Update(func(f *stratatest.Fake) { f.Unloaded, f.Loaded = true, false })
	want(t, "a cold engine with the file on disk", e.turn("A", 2), actDisk)
	if acts := e.f.Take(); len(acts) != 3 || acts[0] != "load" || !strings.HasPrefix(acts[1], "restore ") ||
		acts[2] != "gen /v1/chat/completions" {
		t.Errorf("actions %q: want the load, the restore, the turn", acts)
	}
	if e.files() != 1 {
		t.Errorf("%d files: a restore on a loaded engine evicted them", e.files())
	}
	want(t, "the turn after the restore", e.turn("A", 3), actContinue)
}

// The arbiter loads a cold engine on admission, so the cache never sees a route
// that is resident with nothing loaded: the load is explicit, and the request is
// forwarded after it.
func TestColdEngineIsLoadedOnAdmission(t *testing.T) {
	e := setup(t, 10*time.Millisecond, 64<<20)
	e.f.Update(func(f *stratatest.Fake) { f.Unloaded, f.Loaded = true, false })
	want(t, "a cold engine with no files", e.turn("A", 1), actMiss)
	if acts := e.f.Take(); len(acts) != 2 || acts[0] != "load" || acts[1] != "gen /v1/chat/completions" {
		t.Errorf("actions %q: the arbiter must load the cold engine, then forward", acts)
	}
}

func TestUnloadedEngineKeepsFiles(t *testing.T) {
	e := setup(t, 50*time.Millisecond, 64<<20)
	e.f.Set(20000, 0)
	e.turn("A", 1)
	eventually(t, "A's file", func() bool { return e.files() == 1 })

	// /metrics of an unloaded engine has no fingerprint: it must not read as one, and
	// the files of the loaded engine must stay
	e.f.Update(func(f *stratatest.Fake) { f.Unloaded, f.Loaded = true, false })
	aFile := e.nameOf()
	want(t, "a switch while unloaded", e.turn("B", 1), actMiss)
	if e.files() != 1 {
		t.Fatalf("%d files after a request against an unloaded engine, want 1", e.files())
	}
	if strings.Contains(e.logs.String(), `"reason":"fingerprint"`) {
		t.Error("files were evicted for a fingerprint read off an unloaded engine")
	}
	want(t, "back to A", e.turn("A", 2), actDisk)
	if !e.exists(aFile) {
		t.Errorf("A's file %s went to a fingerprint read off an unloaded engine", aFile)
	}
	if e.files() != 2 { // A's file, plus B's idle save
		t.Errorf("%d files after the restore, want 2", e.files())
	}
}

// A cold load into a different model: the fingerprint is read after the arbiter's
// load, so the files of the model that is gone are evicted before any restore is
// tried on them.
func TestColdLoadIntoAnotherModelEvictsFiles(t *testing.T) {
	e := setup(t, 50*time.Millisecond, 64<<20)
	e.f.Set(20000, 0)
	e.turn("A", 1)
	eventually(t, "A's file", func() bool { return e.files() == 1 })

	e.f.Update(func(f *stratatest.Fake) { f.Unloaded, f.Loaded, f.Model = true, false, "another-model" })
	want(t, "loaded into another model", e.turn("A", 2), actMiss)
	if !strings.Contains(e.logs.String(), `"reason":"fingerprint"`) {
		t.Error("the files of the old model were not evicted for the fingerprint change")
	}
	if e.files() != 0 {
		t.Errorf("%d files: a fingerprint change kept the files of the old engine", e.files())
	}
}

// A load the engine refuses: the admission does not make that its own 503. The
// request goes to the engine, and the engine's answer is the client's answer.
func TestRefusedLoadFailsOpen(t *testing.T) {
	e := setup(t, 50*time.Millisecond, 64<<20)
	e.f.Set(20000, 0)
	e.turn("A", 1)
	eventually(t, "A's file", func() bool { return e.files() == 1 })

	e.f.Take() // the turn above is not part of this check
	e.f.Update(func(f *stratatest.Fake) {
		f.Unloaded, f.Loaded, f.LoadStatus, f.GenStatus = true, false, 502, 502
	})
	if status, _ := e.post(context.Background(), "/v1/chat/completions", conversation("A", 2)); status != 502 {
		t.Fatalf("a turn whose engine cannot load: status %d", status)
	}
	if !strings.Contains(e.logs.String(), `"msg":"reload refused; the request goes to the engine"`) {
		t.Error("the refused load was not recorded")
	}
	if acts := e.f.Take(); len(acts) < 2 || acts[0] != "load" {
		t.Errorf("actions %q: the arbiter must try the load before forwarding", acts)
	}
	if e.files() != 1 {
		t.Errorf("%d files: a refused load evicted the file", e.files())
	}
}

func TestTruncatedSessionFile(t *testing.T) {
	e := setup(t, 50*time.Millisecond, 64<<20)
	e.f.Set(20000, 0)
	e.turn("A", 1)
	eventually(t, "A's file", func() bool { return e.files() == 1 })

	// the file on disk restores far fewer tokens than its name claims
	e.f.Update(func(f *stratatest.Fake) { f.Started++; f.Tokens = 8000 })
	want(t, "restoring a truncated file", e.turn("A", 2), actDisk)
	if !strings.Contains(e.logs.String(), `"msg":"truncated session file; forgetting"`) {
		t.Error("a truncated file was kept")
	}
	if e.files() != 0 {
		t.Errorf("%d files: a truncated file was kept", e.files())
	}
	// with the broken file gone, the conversation is saved afresh instead of being
	// skipped as already on disk
	e.f.Set(20000, 0)
	e.turn("A", 3)
	eventually(t, "the save of the regrown conversation", func() bool { return e.files() == 1 })
}

func TestStrataRestartAndFingerprint(t *testing.T) {
	e := setup(t, 10*time.Millisecond, 64<<20)
	e.f.Set(20000, 0)
	e.turn("A", 1)
	eventually(t, "A's file", func() bool { return e.files() == 1 })

	e.f.Update(func(f *stratatest.Fake) { f.Started++ }) // the server restarted: A is no longer live
	want(t, "after a restart", e.turn("A", 2), actDisk)

	e.f.Set(20000, 1)
	e.turn("B", 1)
	e.f.Update(func(f *stratatest.Fake) { f.Loaded = false })
	if got := e.turn("A", 3); got == actRAM {
		t.Error("an unloaded engine was believed to have A in RAM")
	}

	e.f.Update(func(f *stratatest.Fake) { f.Loaded = true; f.Model = "another-model" }) // new fingerprint
	e.turn("C", 1)
	if e.files() != 0 {
		t.Errorf("%d files of the old model left", e.files())
	}
}

func TestRestoreFailure(t *testing.T) {
	e := setup(t, 10*time.Millisecond, 64<<20)
	saveAB(e)
	e.f.Update(func(f *stratatest.Fake) { f.SlotStatus, f.SlotKind = 507, "memory" })
	want(t, "no RAM to restore", e.turn("A", 2), actMiss)
	if e.files() != 2 {
		t.Error("a file Strata had no RAM for was deleted")
	}
	e.f.Update(func(f *stratatest.Fake) { f.SlotKind = "invalid" })
	e.turn("B", 2)
	if e.files() != 1 {
		t.Error("a file Strata rejects was kept")
	}
}

func TestFailOpen(t *testing.T) {
	e := setup(t, 0, 1<<20)

	// Strata down: its own error reaches the client
	e.f.Update(func(f *stratatest.Fake) { f.Down = true })
	if s, _ := e.post(context.Background(), "/v1/chat/completions", conversation("A", 1)); s != http.StatusServiceUnavailable {
		t.Errorf("Strata down: status %d", s)
	}
	want(t, "Strata down", e.last(), actPass)
	e.f.Update(func(f *stratatest.Fake) { f.Down = false })

	// malformed JSON: passed through, Strata answers
	if s, _ := e.post(context.Background(), "/v1/chat/completions", `{"messages": [`); s != http.StatusBadRequest {
		t.Errorf("malformed JSON: status %d", s)
	}
	want(t, "malformed JSON", e.last(), actUnkeyed)

	// a body over the limit streams through whole, unbuffered
	big := `{"messages":[{"role":"user","content":"` + strings.Repeat("x", 2<<20) + `"}]}`
	if s, _ := e.post(context.Background(), "/v1/chat/completions", big); s != 200 {
		t.Errorf("big body: status %d", s)
	}
	want(t, "big body", e.last(), actPass)
	e.f.Update(func(f *stratatest.Fake) {
		if f.Bodies[len(f.Bodies)-1] != big {
			t.Error("Strata did not get the whole big body")
		}
	})

	// other endpoints pass through with the client's Host, and make the live conversation unknown
	e.turn("A", 1)
	want(t, "next turn", e.turn("A", 2), actContinue)
	req, _ := http.NewRequest(http.MethodPost, e.url+"/v1/completions", strings.NewReader(`{"prompt":"x"}`))
	req.Host = "llm.example:8080"
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	want(t, "pass-through", string(b), "host=llm.example:8080 path=/v1/completions")
	if got := e.turn("A", 3); got == actContinue {
		t.Error("a raw completion did not make the live conversation unknown")
	}

	// a POST outside the generation paths that names a model is read for that name
	// before it reaches the engine: naming the resident's model passes through in
	// strict mode too (the rule is the model the card holds, not the endpoint), and
	// the conversation it touched becomes unknown. Naming a model the arbiter does
	// not serve, or one that is not resident, is answered 404: see
	// TestStrictNonGenPOSTToOtherModelIsRefused.
	req, _ = http.NewRequest(http.MethodPost, e.url+"/v1/completions", strings.NewReader(`{"model":"test-model","prompt":"x"}`))
	req.Host = "llm.example:8080"
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	b, _ = io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	want(t, "model-naming pass-through", string(b), "host=llm.example:8080 path=/v1/completions")
	if got := e.turn("A", 4); got == actContinue {
		t.Error("a completion naming the resident's model did not make the live conversation unknown")
	}

	// a panic in the cache logic: the request still goes through
	e.p.cache.files = nil
	e.f.Set(20000, 0)
	if s, _ := e.post(context.Background(), "/v1/chat/completions", conversation("Z", 1)); s != 200 {
		t.Errorf("after a panic: status %d", s)
	}
	want(t, "panic", e.last(), actError)
	if e.p.cache.inflight != 0 {
		t.Errorf("inflight %d after a panic", e.p.cache.inflight)
	}
}

func TestDialectsAndNoContentInLogs(t *testing.T) {
	e := setup(t, 10*time.Millisecond, 64<<20)
	if s, n := e.post(context.Background(), "/v1/messages", `{"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`); s != 200 || n != 6 {
		t.Errorf("/v1/messages: status %d, %d events", s, n)
	}
	if s, n := e.post(context.Background(), "/v1/responses", `{"input":"hi"}`); s != 200 || n != 5 {
		t.Errorf("/v1/responses: status %d, %d events", s, n)
	}
	secret := "the-launch-code-is-4471"
	e.f.Set(20000, 0)
	e.post(context.Background(), "/v1/chat/completions", strings.ReplaceAll(conversation("A", 1), "u0", secret))
	eventually(t, "A's file", func() bool { return e.files() == 1 })
	e.turn("B", 1)
	e.post(context.Background(), "/v1/chat/completions", strings.ReplaceAll(conversation("A", 2), "u0", secret))
	e.post(context.Background(), "/v1/chat/completions", `{"messages":[{"role":"user","content":"`+secret+`"`)

	logs := e.logs.String()
	for _, msg := range []string{`"msg":"request"`, `"msg":"save"`, `"msg":"restore"`} {
		if !strings.Contains(logs, msg) {
			t.Errorf("no %s line", msg)
		}
	}
	if strings.Contains(logs, secret) || strings.Contains(logs, "answer-text") {
		t.Error("message content reached the logs")
	}
}

func TestKeys(t *testing.T) {
	var k keyer
	key := func(path, body string) []string {
		t.Helper()
		keys, ok := k.keys(path, []byte(body))
		if !ok {
			t.Fatalf("not keyed: %s", body)
		}
		return keys
	}
	const chat = "/v1/chat/completions"
	turn1 := key(chat, `{"tools":[{"type":"function"}],"messages":[{"role":"system","content":"s"},{"role":"user","content":"hi"}]}`)
	// other sampling, key order, whitespace, a moved cache_control
	turn2 := key(chat, `{"temperature":0.2,"tools":[{"type":"function"}],"messages":[ {"content":"s","role":"system"},
		{"role":"user","content":"hi","cache_control":{"type":"ephemeral"}},{"role":"assistant","content":"x"},{"role":"user","content":"y"}]}`)
	if !extends(turn2, turn1) || extends(turn1, turn2) || !extends(turn1, turn1) {
		t.Error("turn 2 should extend turn 1, not the reverse; a retry extends itself")
	}
	msgs := `"messages":[{"role":"user","content":"hi"}]`
	base := key(chat, `{`+msgs+`}`)
	for _, other := range [][]string{
		key(chat, `{"tools":[{"type":"function"}],`+msgs+`}`),
		key(chat, `{"reasoning_effort":"high",`+msgs+`}`),
		key("/v1/messages", `{`+msgs+`}`),
		key("/v1/messages", `{"system":"be brief",`+msgs+`}`),
		key(chat, `{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}`),
	} {
		if extends(other, base) {
			t.Errorf("%v extends the base conversation", other)
		}
	}
	for _, b := range []struct{ path, body string }{
		{chat, `not json`}, {chat, `{}`}, {chat, `{"messages":[]}`}, {chat, `{"messages":"nope"}`},
		{"/v1/responses", `{"input":[{"role":"user","content":"x"}],"previous_response_id":"r1"}`},
	} {
		if _, ok := k.keys(b.path, []byte(b.body)); ok {
			t.Errorf("keyed: %s", b.body)
		}
	}
	if keys := key("/v1/responses", `{"input":"hello"}`); len(keys) != 1 {
		t.Error("string input")
	}
}

// A save's bytes-per-token is learned only from a conversation worth saving. The
// engine can report a live count of 175K tokens and write a file dominated by
// fixed overhead, answering 74 tokens saved over 226 MB — 3 MB/token. With that
// estimate the next reserve reads a 175K conversation as 534 GB over the cap, the
// save is skipped, and the session is lost at the switch. The guard keeps the
// 32 KB/token default, which fits.
func TestSmallSaveDoesNotPoisonTheEstimate(t *testing.T) {
	e := setup(t, time.Second, 1<<20)
	e.f.Update(func(f *stratatest.Fake) { f.Tokens, f.SavedTokens, f.SaveBytes = 175000, 74, 226<<20 })

	if status, _ := e.post(context.Background(), "/v1/chat/completions", longBody(20000)); status != 200 {
		t.Fatalf("the first long body: status %d", status)
	}
	eventually(t, "the first file", func() bool { return e.files() == 1 })

	if status, _ := e.post(context.Background(), "/v1/chat/completions", longBody(200000)); status != 200 {
		t.Fatalf("the next long body: status %d", status)
	}
	eventually(t, "the second file", func() bool { return e.files() == 2 })
}

// longBody is a conversation whose body is about words*5 bytes, so the proxy's
// token estimate is words*5/4.
func longBody(words int) string {
	return `{"stream":true,"messages":[{"role":"system","content":"` + strings.Repeat("word ", words) +
		`"},{"role":"user","content":"u"}]}`
}

// The turn-1 boundary below --min-tokens lands on disk — that is the shared root
// a fork, a session with the same opening, or the summarizer restores — and its
// bytes-per-token never reaches the route's estimate: the estimate is learned
// only from a save of at least MinTokens, so the next long reserve is sized by
// the measured big file, not by a small file's fixed overhead.
func TestSmallRootSaveDoesNotPoisonBPT(t *testing.T) {
	e := setup(t, 50*time.Millisecond, 64<<20)
	e.p.cache.pol.MinTokens = 4096 // the arbiter's --min-tokens
	e.f.Update(func(f *stratatest.Fake) { f.Tokens, f.SavedTokens, f.SaveBytes = 3309, 3309, 3309*3000 })

	// A's chain is the turn-1 boundary: two keys, below MinTokens, above the floor
	want(t, "the root conversation", e.turn("A", 1), actMiss)
	eventually(t, "A's root file", func() bool { return e.files() == 1 })
	if m, _ := filepath.Glob(filepath.Join(e.dir, "dc-*-3309.session")); len(m) != 1 {
		t.Fatalf("the root boundary: %v, want A's file at 3309 tokens", m)
	}
	e.p.cache.mu.Lock()
	bpt := e.p.cache.routes["strata/test-model"].bpt
	e.p.cache.mu.Unlock()
	if bpt != 0 {
		t.Errorf("bytes per token %d after a save below MinTokens, want the untouched default", bpt)
	}

	// B's own boundary lands too, at the same count; its save is what switches A out
	want(t, "the switch away from the root", e.turn("B", 1), actMiss)
	eventually(t, "B's boundary file", func() bool { return e.files() == 2 })
	e.p.cache.mu.Lock()
	bpt = e.p.cache.routes["strata/test-model"].bpt
	e.p.cache.mu.Unlock()
	if bpt != 0 {
		t.Errorf("bytes per token %d after two saves below MinTokens, want the untouched default", bpt)
	}

	// The first save worth saving teaches the route its own figure: the long
	// file's ratio, never the small file's fixed overhead. The long conversation
	// is too big to park, so its save runs at the switch.
	time.Sleep(120 * time.Millisecond) // let the idle saves finish
	e.f.Update(func(f *stratatest.Fake) { f.Tokens, f.SavedTokens, f.SaveBytes = 100000, 100000, 100000*13824 })
	want(t, "the switch away from B", e.turn("C", 1), actMiss)
	eventually(t, "the long file", func() bool { return e.files() == 3 })
	e.p.cache.mu.Lock()
	bpt = e.p.cache.routes["strata/test-model"].bpt
	e.p.cache.mu.Unlock()
	if bpt != 13824 {
		t.Errorf("bytes per token %d, want the figure measured from the long save", bpt)
	}
}

// The measured double prefill: a fresh session's launch prefills the engine, and
// its next turn must continue the conversation the engine just built, paying for
// its new messages only.
func TestFreshSessionSecondTurnContinuesTheLiveSlot(t *testing.T) {
	e := setup(t, 50*time.Millisecond, 64<<20)
	e.f.Set(20248, 0)
	want(t, "the launch request", e.turn("A", 4), actMiss)
	e.f.Set(20834, 0)
	want(t, "the next turn", e.turn("A", 5), actContinue)
	if acts := e.f.Take(); saves(acts) != 0 || restores(acts) != 0 {
		t.Fatalf("the second turn: %q, want the live slot continued with no save and no restore", acts)
	}
}

// The single route built from --upstream answers every request, whatever the body
// names. The request line carries the model that was asked for next to the route
// that answered, so an alias is visible in the log and not only in the client's
// label. Strict mode refuses it instead; see TestStrictRefusesAnUnknownModel.
func TestSingleRouteLogsTheRequestedModel(t *testing.T) {
	e := setup(t, 0, 64<<20)
	body := `{"model":"qwen3.8-swift-1.5-architect","stream":true,"messages":[{"role":"user","content":"u"}]}`
	if status, _ := e.post(context.Background(), "/v1/chat/completions", body); status != 200 {
		t.Fatalf("status %d, want the single route to answer", status)
	}
	logs := e.logs.String()
	if !strings.Contains(logs, `"model":"qwen3.8-swift-1.5-architect"`) {
		t.Error("the log does not name the model the request asked for")
	}
	if !strings.Contains(logs, `"route":"strata/test-model"`) {
		t.Error("the log does not name the route that answered")
	}
	if !strings.Contains(e.p.requestsJSON(), `"model":"qwen3.8-swift-1.5-architect"`) {
		t.Error("/requests does not show the requested model")
	}
}

// modelOf reads the model from a truncated prefix: a body cut at --max-body-mib
// keeps it, because it is the first key. A body that names none, a model that is
// not a string, and a body that is not an object all read as no model.
func TestModelOfReadsATruncatedPrefix(t *testing.T) {
	full := `{"model":"swift","stream":true,"messages":[{"role":"user","content":"u"}]}`
	if got := modelOf([]byte(full)); got != "swift" {
		t.Errorf("whole body: %q, want swift", got)
	}
	cut := `{"model":"swift","str` // the body cut mid-value, after the model
	if got := modelOf([]byte(cut)); got != "swift" {
		t.Errorf("truncated prefix %q: %q, want swift", cut, got)
	}
	if got := modelOf([]byte(`{"stream":true,"messages":[]}`)); got != "" {
		t.Errorf("no model: %q, want empty", got)
	}
	if got := modelOf([]byte(`{"model":123}`)); got != "" {
		t.Errorf("model is not a string: %q, want empty", got)
	}
	if got := modelOf([]byte(`[1,2]`)); got != "" {
		t.Errorf("not an object: %q, want empty", got)
	}
	if got := modelOf([]byte(`{"tools":[{"model":"x"}],"stream":true}`)); got != "" {
		t.Errorf("nested model key: %q, want empty: only the top level names the model", got)
	}
}

// forkOf extends a body by one user message: its chain shares the whole prefix.
func forkOf(body, tail string) string {
	return strings.TrimSuffix(body, `]}`) + `,{"role":"user","content":"` + tail + `"}]}`
}

// A fork whose chain reaches a saved turn boundary gets the file of the deepest
// boundary with live ≤ prompt restored: longer files on the way down are
// skipped, a fork that extends the live card pays nothing, and the fork's own
// switch-away save is skipped as covered by the shared prefix.
func TestForkRestoresPrefix(t *testing.T) {
	e := setup(t, 50*time.Millisecond, 64<<20)
	e.p.cache.pol.MaxIdleSaveTokens = 32768 // above the cap, saves fire only at switches
	e.f.Set(124000, 0)
	a1 := longBody(100000)
	if status, _ := e.post(context.Background(), "/v1/chat/completions", a1); status != 200 {
		t.Fatalf("the long first request: status %d", status)
	}
	if act := e.last(); act != actMiss {
		t.Fatalf("the long first request: %q, want %q", act, actMiss)
	}
	eventually(t, "the idle save above the cap is skipped",
		func() bool { return strings.Contains(e.logs.String(), `"reason":"idle_too_long"`) })
	if e.files() != 0 {
		t.Fatalf("%d files: an idle save above the cap ran", e.files())
	}

	// B switches away from A1: A1 is saved at the switch, keyed by its last key
	want(t, "the switch to B", e.turn("B", 1), actMiss)
	if n := saves(e.f.Take()); n != 1 {
		t.Fatalf("A1 at the switch: %d saves, want 1", n)
	}
	m, _ := filepath.Glob(filepath.Join(e.dir, "dc-*.session"))
	if len(m) != 1 || !strings.Contains(filepath.Base(m[0]), "-124000.session") {
		t.Fatalf("A1's file: %v, want one file at 124000 tokens", m)
	}

	// A fork of A1 does not extend B, so it switches: the lookup descends past
	// B's key to A1's, 124000 ≤ the fork's prompt, and the restore fires
	e.f.Set(124000, 0)
	f1 := forkOf(a1, "fork tail")
	if status, _ := e.post(context.Background(), "/v1/chat/completions", f1); status != 200 {
		t.Fatalf("the fork of A1: status %d", status)
	}
	want(t, "fork restore from A1's file", e.last(), actDisk)
	if acts := e.f.Take(); restores(acts) != 1 || !strings.Contains(strings.Join(acts, " "), filepath.Base(m[0])) {
		t.Fatalf("fork restore: %q, want one restore of %s", acts, filepath.Base(m[0]))
	}

	// A fork of the fork extends the live card: the engine keeps it, no save, no restore
	f2 := forkOf(f1, "next tail")
	e.f.Set(126000, 0)
	if status, _ := e.post(context.Background(), "/v1/chat/completions", f2); status != 200 {
		t.Fatalf("the fork of the fork: status %d", status)
	}
	want(t, "the deeper fork extends live", e.last(), actContinue)
	if n := saves(e.f.Take()); n != 0 {
		t.Fatalf("the switch away from f1: %d saves, want 0 covered by A1's file", n)
	}

	// Z switches away: f2 is saved at the switch, the deeper boundary at 130000
	e.f.Set(130000, 0)
	want(t, "the switch away from f2", e.turn("Z", 1), actMiss)
	if n := saves(e.f.Take()); n != 1 {
		t.Fatalf("f2 at the switch: %d saves, want 1", n)
	}
	if e.files() != 3 {
		t.Fatalf("%d files, want A1, B and f2", e.files())
	}

	// A fork behind the deep boundary: the engine's live count is the prompt's
	// lower bound, so the deepest prefix file is at the fork's prompt and is the
	// restore target, exactly as the engine can reuse it.
	f3 := forkOf(f2, "deeper tail")
	e.f.Set(130000, 0)
	if status, _ := e.post(context.Background(), "/v1/chat/completions", f3); status != 200 {
		t.Fatalf("the fork behind the deep boundary: status %d", status)
	}
	want(t, "the fork restores the deepest prefix at its prompt", e.last(), actDisk)
	if acts := e.f.Take(); restores(acts) != 1 || !strings.Contains(strings.Join(acts, " "), "-130000.session") {
		t.Fatalf("restore past the deep boundary: %q, want one restore of the 130000-token file", acts)
	}

	// The switch away from f3: its last key has no file, so the deeper boundary
	// is saved — f2's file covers only the chain f2 ended at, not f3's tail.
	// Z, the conversation that switched the card, is saved at its own boundary.
	e.f.Set(134096, 0)
	want(t, "the switch away from f3", e.turn("Y", 1), actMiss)
	if n := saves(e.f.Take()); n != 1 {
		t.Fatalf("the switch away from f3: %d saves, want 1 at its own boundary", n)
	}
	if e.files() != 5 {
		t.Fatalf("%d files, want A1, B, f2, Z and f3", e.files())
	}
}

// --max-idle-save-tokens 32768 skips the idle save of a live conversation above
// the cap. The switch-save is what a new conversation gets, and the fork that
// arrives after it restores A1's file: the handoff pays one save, not a full
// re-prefill.
func TestForkHandoffAt32k(t *testing.T) {
	e := setup(t, 50*time.Millisecond, 64<<20)
	e.p.cache.pol.MaxIdleSaveTokens = 32768
	e.f.Set(100000, 0)
	a1 := longBody(100000)
	if status, _ := e.post(context.Background(), "/v1/chat/completions", a1); status != 200 {
		t.Fatalf("the long first request: status %d", status)
	}
	if act := e.last(); act != actMiss {
		t.Fatalf("the long first request: %q, want %q", act, actMiss)
	}
	eventually(t, "the idle save above the cap is skipped",
		func() bool { return strings.Contains(e.logs.String(), `"reason":"idle_too_long"`) })
	if e.files() != 0 {
		t.Fatalf("%d files: an idle save above the cap ran", e.files())
	}

	// A fork of the live conversation continues it: no save, no restore
	f1 := forkOf(a1, "fork tail")
	if status, _ := e.post(context.Background(), "/v1/chat/completions", f1); status != 200 {
		t.Fatalf("the fork of the live conversation: status %d", status)
	}
	want(t, "fork extends live", e.last(), actContinue)
	if acts := e.f.Take(); saves(acts) != 0 || restores(acts) != 0 {
		t.Fatalf("fork extends live: %q, want no save and no restore", acts)
	}

	// A new conversation switches away from the fork: the switch-save fires
	// above the idle cap, before the new request is served
	want(t, "the new conversation", e.turn("B", 1), actMiss)
	if n := saves(e.f.Take()); n != 1 {
		t.Fatalf("the switch away from the fork: %d saves, want 1", n)
	}
	m, _ := filepath.Glob(filepath.Join(e.dir, "dc-*.session"))
	if len(m) != 1 || !strings.Contains(filepath.Base(m[0]), "-100000.session") {
		t.Fatalf("the switch-save file: %v, want one file at 100000 tokens", m)
	}

	// A fork sharing the saved chain restores it: 100000 ≤ the fork's prompt
	e.f.Set(100400, 0)
	f2 := forkOf(f1, "next tail")
	if status, _ := e.post(context.Background(), "/v1/chat/completions", f2); status != 200 {
		t.Fatalf("the fork after the switch: status %d", status)
	}
	want(t, "fork restore from the saved chain", e.last(), actDisk)
	if acts := e.f.Take(); restores(acts) != 1 || !strings.Contains(strings.Join(acts, " "), filepath.Base(m[0])) {
		t.Fatalf("fork restore: %q, want one restore of %s", acts, filepath.Base(m[0]))
	}

	// The fork's switch-away: its last key has no file, so its own boundary is
	// saved; the shared prefix file covers only the chain it was keyed to.
	e.f.Set(104096, 0)
	want(t, "the switch away from the fork", e.turn("Z", 1), actMiss)
	if n := saves(e.f.Take()); n != 1 {
		t.Fatalf("the switch away from the fork: %d saves, want 1 at its own boundary", n)
	}
	if m, _ := filepath.Glob(filepath.Join(e.dir, "dc-*-104096.session")); len(m) != 1 {
		t.Fatalf("the fork's own boundary: %v, want one file at 104096 tokens", m)
	}
}

// TestForkHandoffGrownParent is the measured fork bug: the parent keeps the card
// and grows past the fork point, so its checkpoint (70,611 tokens, committed
// after the fork) registers prefix aliases for EVERY prefix key of its chain,
// the fork-point hash included. A fork's prompt (~66,000 tokens) is SHORTER than
// that file, so the engine reads the whole prompt after restoring it. The bound
// is the prompt estimate only: the grown alias is skipped, and the fork restores
// a file at or below the estimate — its own turn boundary when the store has one,
// full prefill when it has none.
func TestForkHandoffGrownParent(t *testing.T) {
	e := setup(t, 50*time.Millisecond, 64<<20)
	e.p.cache.pol.MaxIdleSaveTokens = 32768 // above the cap, saves fire only at switches

	// The parent conversation: its prompt estimate is ~66,000 tokens.
	e.f.Set(66000, 0)
	a1 := longBody(52800)
	if status, _ := e.post(context.Background(), "/v1/chat/completions", a1); status != 200 {
		t.Fatalf("the long first request: status %d", status)
	}
	if act := e.last(); act != actMiss {
		t.Fatalf("the long first request: %q, want %q", act, actMiss)
	}
	eventually(t, "the idle save above the cap is skipped",
		func() bool { return strings.Contains(e.logs.String(), `"reason":"idle_too_long"`) })
	if e.files() != 0 {
		t.Fatalf("%d files: an idle save above the cap ran", e.files())
	}

	// The fork-point file: a new conversation switches the card away from a1,
	// and a1 is saved at its own turn boundary, below the fork's estimate.
	e.f.Set(65000, 0)
	want(t, "the switch away from a1", e.turn("B", 1), actMiss)
	if n := saves(e.f.Take()); n != 1 {
		t.Fatalf("a1 at the switch: %d saves, want 1", n)
	}
	boundary, _ := filepath.Glob(filepath.Join(e.dir, "dc-*-65000.session"))
	if len(boundary) != 1 {
		t.Fatalf("the fork-point file: %v, want one at 65000 tokens", boundary)
	}

	// The parent takes the card back, continues past the fork point, and is
	// saved at 70,611: every prefix key of its chain, the fork-point hash
	// included, now aliases to the grown file.
	e.f.Set(67000, 0)
	a2 := forkOf(a1, "parent tail")
	if status, _ := e.post(context.Background(), "/v1/chat/completions", a2); status != 200 {
		t.Fatalf("the parent's next turn: status %d", status)
	}
	want(t, "the parent restores its own boundary", e.last(), actDisk)
	e.f.Take()
	e.f.Set(70611, 0)
	a3 := forkOf(a2, "deeper parent")
	if status, _ := e.post(context.Background(), "/v1/chat/completions", a3); status != 200 {
		t.Fatalf("the parent's deeper turn: status %d", status)
	}
	want(t, "the deeper turn continues the card", e.last(), actContinue)
	e.f.Take()
	want(t, "the switch away from the grown parent", e.turn("C", 1), actMiss)
	if n := saves(e.f.Take()); n != 1 {
		t.Fatalf("the grown parent at the switch: %d saves, want 1", n)
	}
	if m, _ := filepath.Glob(filepath.Join(e.dir, "dc-*-70611.session")); len(m) != 1 {
		t.Fatalf("the grown parent's file: %v, want one at 70611 tokens", m)
	}

	// A fork behind the grown file: its chain reaches the grown aliases, its
	// prompt estimate is ~66,000, and the parent is live on the card at 70,611.
	// The engine answers the restore with the file's own count, so the request
	// line shows what the restore really carried.
	e.f.Update(func(f *stratatest.Fake) { f.SavedTokens = 65000 })
	e.f.Set(70611, 0)
	f2 := forkOf(a2, "fork tail")
	if status, _ := e.post(context.Background(), "/v1/chat/completions", f2); status != 200 {
		t.Fatalf("the fork of the grown parent: status %d", status)
	}
	want(t, "the fork restores its own boundary", e.last(), actDisk)
	acts := e.f.Take()
	restored := ""
	for _, a := range acts {
		if strings.HasPrefix(a, "restore ") {
			restored = a
		}
	}
	if restores(acts) != 1 || !strings.Contains(restored, filepath.Base(boundary[0])) {
		t.Fatalf("the fork restore: %q, want one restore of %s", acts, filepath.Base(boundary[0]))
	}
	if strings.Contains(restored, "-70611.session") {
		t.Fatalf("the fork restored the grown parent's file: %q", acts)
	}
	if !strings.Contains(e.logs.String(), `"action":"disk","reason":"restored","tokens":65000`) {
		t.Error("the fork's request line does not log the restored 65000-token boundary")
	}

	// A fork whose only copies are longer than its prompt pays full prefill: the
	// grown parent's chain has no file at or below the estimate, so nothing is
	// restored.
	e.f.Update(func(f *stratatest.Fake) { f.SavedTokens = 0 })
	fresh := `{"stream":true,"messages":[{"role":"system","content":"` + strings.Repeat("word ", 52800) +
		`x"},{"role":"user","content":"u"}]}`
	e.f.Set(66000, 0)
	if status, _ := e.post(context.Background(), "/v1/chat/completions", fresh); status != 200 {
		t.Fatalf("the fresh chain: status %d", status)
	}
	want(t, "the fresh chain", e.last(), actMiss)
	e.f.Set(67000, 0)
	fParent := forkOf(fresh, "parent tail")
	if status, _ := e.post(context.Background(), "/v1/chat/completions", fParent); status != 200 {
		t.Fatalf("the fresh parent's next turn: status %d", status)
	}
	want(t, "the fresh parent continues the card", e.last(), actContinue)
	e.f.Take()
	e.f.Set(70611, 0)
	want(t, "the switch away from the fresh parent", e.turn("D", 1), actMiss)
	if n := saves(e.f.Take()); n != 1 {
		t.Fatalf("the fresh parent at the switch: %d saves, want 1", n)
	}
	if m, _ := filepath.Glob(filepath.Join(e.dir, "dc-*-70611.session")); len(m) != 2 {
		t.Fatalf("the fresh parent's grown file: %v, want two files at 70611 tokens", m)
	}
	e.f.Set(70611, 0)
	fFork := forkOf(fresh, "fork tail")
	if status, _ := e.post(context.Background(), "/v1/chat/completions", fFork); status != 200 {
		t.Fatalf("the fork of the fresh parent: status %d", status)
	}
	want(t, "the fork pays full prefill", e.last(), actMiss)
	if acts := e.f.Take(); restores(acts) != 0 {
		t.Fatalf("the fork of the fresh parent restored %q, want no restore", acts)
	}
	if strings.Contains(e.logs.String(), `"action":"disk","reason":"restored","tokens":70611`) {
		t.Error("the arbiter restored the grown 70611-token file for a shorter fork prompt")
	}
}

// A conversation of a few tokens is nobody's prefix: the floor keeps it off disk.
func TestFewTokenConversationIsNotSaved(t *testing.T) {
	e := setup(t, 50*time.Millisecond, 64<<20)
	e.p.cache.pol.MinTokens = 4096
	e.f.Set(64, 0)
	want(t, "the tiny conversation", e.turn("A", 1), actMiss)
	time.Sleep(200 * time.Millisecond)
	if n := saves(e.f.Take()); n != 0 {
		t.Fatalf("the tiny conversation: %d saves, want 0 below the floor", n)
	}
	if e.files() != 0 {
		t.Fatalf("%d files, want none for a conversation below the floor", e.files())
	}
}

// A few-token conversation arriving after a big one on the same route inherits
// the route's stored count. The short rule reads the engine's own count for the
// conversation, so the inherited count admits nothing: a 236 MB file of a 64-token
// conversation is nobody's prefix, and the route's byte estimate stays with the big file.
func TestStaleLiveCountDoesNotAdmitTinySave(t *testing.T) {
	e := setup(t, 50*time.Millisecond, 64<<20)
	e.p.cache.pol.MinTokens = 4096
	e.f.Set(8069, 0)
	want(t, "the big conversation", e.turn("A", 2), actMiss)
	eventually(t, "the big boundary file", func() bool { return e.files() == 1 })
	if n := saves(e.f.Take()); n != 1 {
		t.Fatalf("the big conversation: %d saves, want its own boundary file", n)
	}

	// The engine now serves a 64-token conversation; the route still carries 8069.
	e.f.Set(64, 0)
	want(t, "the tiny conversation", e.turn("B", 1), actMiss)
	time.Sleep(200 * time.Millisecond)
	if n := saves(e.f.Take()); n != 0 {
		t.Fatalf("the tiny conversation: %d saves, want 0 with the route count inherited", n)
	}
	if e.files() != 1 {
		t.Fatalf("%d files, want only the big conversation's boundary file", e.files())
	}
}

// The root conversation's boundary file lands despite the route carrying a stale
// big count, and the fork launch restores it. The idle cap and floor read the
// engine's own count: the inherited 67,691 estimate would skip the 8,069-token
// boundary as idle_too_long, and the fork would have nothing to restore.
func TestRootBoundarySurvivesStaleBigCount(t *testing.T) {
	e := setup(t, 100*time.Millisecond, 8<<30) // room for the 8k-token reserve
	e.p.cache.pol.MinTokens = 4096
	e.p.cache.pol.MaxIdleSaveTokens = 32768

	// A leaves 67,691 tokens in the engine, so A's own idle save is over the cap
	// and nothing lands. B arrives with the engine reporting 8,069: the cap reads
	// the engine's count, not the 67,691 the route stored, and B's boundary file
	// lands. A's file comes from the switch-save at B's prepare.
	e.f.Set(67691, 0)
	want(t, "the conversation that leaves a big count", e.turn("A", 1), actMiss)
	time.Sleep(200 * time.Millisecond)
	if e.files() != 0 {
		t.Fatalf("%d files, want none: the big conversation is over the idle cap", e.files())
	}
	e.f.Set(8069, 0)
	want(t, "the switch from A to B", e.turn("B", 1), actMiss)
	eventually(t, "A's switch-save and B's boundary file", func() bool { return e.files() == 2 })
	if m, _ := filepath.Glob(filepath.Join(e.dir, "dc-*-8069.session")); len(m) != 2 {
		t.Fatalf("the boundary files: %v, want A's switch-save and B's own boundary", m)
	}

	// Z arrives far below the floor: B's switch-save is covered by B's own file,
	// Z is short, and the fork launch restores B's boundary.
	e.f.Set(60, 0)
	want(t, "the switch away from B", e.turn("Z", 1), actMiss)
	e.f.Set(8100, 0)
	want(t, "the fork launch", e.turn("B", 2), actDisk)
}

// folderMatchesStore is the store's invariant as the proxy sees it: every session
// file in the folder is in the index, the bytes the index counts are the bytes on
// disk, and the counted total is under the cap. The window between the engine
// writing a file and Commit indexing it is microseconds, so the check waits for
// it; a file the index dropped without unlinking never heals, so the check fails.
func folderMatchesStore(t *testing.T, dir string, st *store.Store, maxBytes int64) {
	t.Helper()
	var last string
	for deadline := time.Now().Add(3 * time.Second); ; time.Sleep(10 * time.Millisecond) {
		m, _ := filepath.Glob(filepath.Join(dir, "dc-*.session"))
		var bytes int64
		for _, p := range m {
			info, err := os.Stat(p)
			if err != nil {
				t.Fatalf("%s is gone: %v", filepath.Base(p), err)
			}
			bytes += info.Size()
		}
		files, counted := st.Stats()
		last = fmt.Sprintf("%d files %d bytes on disk; the index counts %d files %d bytes against cap %d",
			len(m), bytes, files, counted, maxBytes)
		if files == len(m) && counted == bytes && counted <= maxBytes {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("the folder does not match the index: %s", last)
		}
	}
}

// The cache's own save path under a tight cap: 100 saves of one growing
// conversation, each at its own turn boundary, through cache.save — reserve the
// room, the engine writes, Commit indexes. The folder must match the store's
// counts at every step: the prefix-index turn-boundary files are under the cap
// exactly like the last-key files, so the disk never grows past the cap. This is
// the production shape: 569 files on disk against 39 counted.
func TestCacheSaveKeepsTheFolderMatchingTheIndex(t *testing.T) {
	maxBytes := int64(3 << 20)
	e := setupCap(t, 0, 64<<20, maxBytes)
	e.f.Update(func(f *stratatest.Fake) { f.SaveBytes = 200000 })
	c := e.p.cache
	c.pol.MinTokens = 10 // the conversation is short; the cap is the point
	r := c.route("strata/test-model")
	e.turn("A", 1) // the route learns its fingerprint
	if r.fp == "" {
		t.Fatal("the route learned no fingerprint")
	}
	r.bpt = minBPT // the figure a route learns from a dense save, at the clamp floor
	tokens, seen := 300, 0
	for i := 1; i <= 100; i++ {
		e.f.Set(tokens, 0)
		keys, ok := c.keys.keys("/v1/chat/completions", []byte(conversation("A", i)))
		if !ok {
			t.Fatalf("turn %d cannot be keyed", i)
		}
		c.save(context.Background(), r, r.fp, keys, tokens, tokens, "test")
		seen += saves(e.f.Take())
		folderMatchesStore(t, e.dir, e.st, maxBytes)
	}
	if seen < 90 {
		t.Errorf("%d saves ran, want the saves to run at each turn boundary", seen)
	}
	if int64(seen)*200000 < 5*maxBytes {
		t.Errorf("the saves wrote %d bytes, want over 5x the cap %d", int64(seen)*200000, maxBytes)
	}
	if n, bytes := e.st.Stats(); bytes > maxBytes {
		t.Errorf("%d bytes counted, %d files: the cap does not bound the folder", bytes, n)
	}
	if n, _ := e.st.Stats(); n < 2 {
		t.Errorf("%d files, want the newest turn boundaries kept under the cap", n)
	}
}
