package arbiter

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/frodosik/gpu-arbiter/internal/engine"
)

// card is the shared GPU: both routes need the same bytes.
type card struct {
	mu    sync.Mutex
	free  int64
	calls []string
}

func (c *card) setFree(n int64) { c.mu.Lock(); c.free = n; c.mu.Unlock() }

func (c *card) freeBytes() (int64, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.free, nil
}

func (c *card) record(s string) { c.mu.Lock(); c.calls = append(c.calls, s); c.mu.Unlock() }

func (c *card) callsSince(n int) []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.calls[n:]...)
}

// fakeEngine is one (backend, model) identity. It records every call the
// arbiter makes, so a test can see the order of the handoff.
type fakeEngine struct {
	id, model   string
	card        *card
	mu          sync.Mutex
	loaded      bool
	loadErr     error
	unloadErr   error
	saveTokens  int
	restoreToke int
}

func (f *fakeEngine) calls() []string { return f.card.callsSince(0) }

// loads counts the load calls the card recorded.
func loads(calls []string) (n int) {
	for _, c := range calls {
		if strings.HasPrefix(c, "load ") {
			n++
		}
	}
	return n
}

func (f *fakeEngine) ID() string    { return f.id }
func (f *fakeEngine) Model() string { return f.model }

func (f *fakeEngine) Status(context.Context) (engine.Status, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return engine.Status{Loaded: f.loaded}, nil
}

func (f *fakeEngine) Fingerprint(context.Context) (string, error) { return "fp-" + f.id, nil }

func (f *fakeEngine) CacheInfo(context.Context) (engine.CacheInfo, error) {
	return engine.CacheInfo{}, nil
}

func (f *fakeEngine) Save(ctx context.Context, name string) (int, int64, error) {
	f.card.record("save " + f.id + " " + name)
	return f.saveTokens, 1000, nil
}

func (f *fakeEngine) Restore(ctx context.Context, name string) (int, int64, error) {
	f.card.record("restore " + f.id + " " + name)
	return f.restoreToke, 1000, nil
}

func (f *fakeEngine) Load(ctx context.Context) error {
	f.card.record("load " + f.id)
	if f.loadErr != nil {
		return f.loadErr
	}
	f.mu.Lock()
	f.loaded = true
	f.mu.Unlock()
	return nil
}

func (f *fakeEngine) Unload(ctx context.Context, wait time.Duration) error {
	f.card.record("unload " + f.id + " " + wait.String())
	if f.unloadErr != nil {
		return f.unloadErr
	}
	f.mu.Lock()
	f.loaded = false
	f.mu.Unlock()
	return nil
}

func (f *fakeEngine) Wake(context.Context) error { f.card.record("wake " + f.id); return nil }

func (f *fakeEngine) VramFree(context.Context) (int64, error) { return f.card.freeBytes() }

func (f *fakeEngine) IsUnloaded(error) bool { return false }

// newTest builds an arbiter over two routes on one card, with no waiting: the
// tests drive admissions by hand.
func newTest(t *testing.T, card *card, streak int) (*Arbiter, *fakeEngine, *fakeEngine) {
	t.Helper()
	a, b := &fakeEngine{id: "strata/flash", model: "flash", card: card}, &fakeEngine{id: "llama/27b", model: "27b", card: card}
	a.loaded, b.loaded = true, false
	ar := New(Options{
		Routes: []Route{
			{ID: a.id, Model: a.model, Namespace: "aaaaaaaaaaaa", Engine: a},
			{ID: b.id, Model: b.model, Namespace: "bbbbbbbbbbbb", Engine: b},
		},
		MaxResidentStreak: streak,
		DrainWait:         300 * time.Second,
		LoadTimeout:       30 * time.Second,
		VRAMMargin:        1 << 30,
		VRAMFree:          card.freeBytes,
		PollInterval:      10 * time.Millisecond,
		Log:               slog.New(slog.NewJSONHandler(quietWriter{}, nil)),
	})
	return ar, a, b
}

type quietWriter struct{}

func (quietWriter) Write(p []byte) (int, error) { return len(p), nil }

// request drives one Acquire and reports the route it was admitted to.
type request struct {
	model    string
	admitted chan string
	err      chan error
	release  func()
}

func do(ctx context.Context, ar *Arbiter, model string) *request {
	r := &request{model: model, admitted: make(chan string, 1), err: make(chan error, 1)}
	go func() {
		rt, rel, err := ar.Acquire(ctx, model)
		if err != nil {
			r.err <- err
			return
		}
		r.release = rel
		r.admitted <- rt.Model
	}()
	return r
}

// doQueued launches one request and returns once its waiter is in the queue, so
// a test builds the queue in the order it launches. A bare do() appends in
// goroutine start order, which is not launch order: a test that read the queue
// as launch order hung when the scheduler started the goroutines in another
// order. The request is queued, so its waiter is counted, and the launches are
// serial, so a growth of one belongs to this request.
func doQueued(t *testing.T, ar *Arbiter, model string) *request {
	t.Helper()
	ar.mu.Lock()
	before := len(ar.queued)
	ar.mu.Unlock()
	r := do(context.Background(), ar, model)
	for deadline := time.Now().Add(2 * time.Second); time.Now().Before(deadline); time.Sleep(5 * time.Millisecond) {
		ar.mu.Lock()
		got := len(ar.queued)
		ar.mu.Unlock()
		if got > before {
			return r
		}
	}
	t.Fatalf("the request for %s never reached the queue", model)
	return nil
}

// admittedAt admits r's request and returns the route, keeping the release for
// the test to run at the end.
// admittedAt waits for r's request to be admitted and returns the model of the
// route it got. A release keeps the card busy: a request that is admitted and
// not released holds the queue behind it, which is how the tests set up the
// resident-drain order deterministically.
func admittedAt(t *testing.T, r *request) string {
	t.Helper()
	select {
	case model := <-r.admitted:
		return model
	case err := <-r.err:
		t.Fatalf("admission failed: %v", err)
		return ""
	}
}

// awaitQueued waits until n requests are parked in the queue, so a test knows
// the requests it launched are all waiting before it reads the queue rule's
// outcome.
func awaitQueued(t *testing.T, ar *Arbiter, n int) {
	t.Helper()
	for deadline := time.Now().Add(2 * time.Second); time.Now().Before(deadline); time.Sleep(5 * time.Millisecond) {
		ar.mu.Lock()
		got := len(ar.queued)
		ar.mu.Unlock()
		if got == n {
			return
		}
	}
	t.Fatalf("the queue never reached %d waiting requests", n)
}

func notAdmitted(t *testing.T, r *request, what string) {
	t.Helper()
	select {
	case model := <-r.admitted:
		t.Fatalf("%s: the card moved to %s", what, model)
	case err := <-r.err:
		t.Fatalf("%s: admission failed: %v", what, err)
	default:
	}
}

// The queue rule, end to end: the resident's queued turn runs first, the swap
// runs once, in the fixed order, and the card ends on the route that asked for it.
func TestResidentDrainsBeforeASwap(t *testing.T) {
	c := &card{free: 20 << 30}
	ar, a, b := newTest(t, c, 32)
	ar.resident = a.id // flash-next is resident, as a parent's child runs

	first := do(context.Background(), ar, "flash")
	if admittedAt(t, first) != "flash" {
		t.Fatal("the resident's own turn was not admitted")
	}
	second := do(context.Background(), ar, "flash")
	awaitQueued(t, ar, 1)
	parent := do(context.Background(), ar, "27b")
	awaitQueued(t, ar, 2)
	notAdmitted(t, parent, "the 27B route")

	order := []string{"flash"}
	first.release()
	if m := admittedAt(t, second); m != "flash" {
		t.Fatalf("the resident's queued turn went to %s", m)
	}
	order = append(order, "flash")
	second.release()
	if m := admittedAt(t, parent); m != "27b" {
		t.Fatalf("the parent went to %s", m)
	}
	order = append(order, "27b")
	parent.release()

	if order[0] != "flash" || order[1] != "flash" || order[2] != "27b" {
		t.Fatalf("admitted %v, want the resident's turns, then the swap", order)
	}
	want := []string{"unload " + a.id + " 5m0s", "load " + b.id}
	if got := c.callsSince(0); len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Errorf("card calls %v, want the resident unloaded, then the new route loaded", got)
	}
	if ar.Resident() != b.id {
		t.Errorf("resident is %q, want %q", ar.Resident(), b.id)
	}
}

// The point of the rule: parent and child share one card, and the card moves
// once per exchange, not once per turn.
func TestNoOscillation(t *testing.T) {
	c := &card{free: 20 << 30}
	ar, _, _ := newTest(t, c, 32)
	ar.resident = "strata/flash"
	callsBefore := len(c.callsSince(0))

	var order []string
	for i := 0; i < 3; i++ {
		child1 := do(context.Background(), ar, "flash")
		order = append(order, admittedAt(t, child1))
		child2 := doQueued(t, ar, "flash")
		parent := doQueued(t, ar, "27b")
		notAdmitted(t, parent, fmt.Sprintf("round %d", i))
		child1.release()
		order = append(order, admittedAt(t, child2))
		child2.release()
		order = append(order, admittedAt(t, parent))
		parent.release()
	}
	wantOrder := []string{"flash", "flash", "27b", "flash", "flash", "27b", "flash", "flash", "27b"}
	if strings.Join(order, ",") != strings.Join(wantOrder, ",") {
		t.Fatalf("admitted %v, want %v", order, wantOrder)
	}
	loads := 0
	for _, call := range c.callsSince(callsBefore) {
		if strings.HasPrefix(call, "load ") {
			loads++
		}
	}
	// Each exchange is one swap, never a load per turn: 3 parent turns move the
	// card 3 times, and flash is loaded back 2 times.
	if loads != 5 {
		t.Errorf("%d loads for 6 child + 3 parent turns, want one swap per exchange", loads)
	}
}

// A route cannot starve the card: the streak cap admits the waiter while the
// resident still has turns queued.
func TestStreakCapBreaksStarvation(t *testing.T) {
	c := &card{free: 20 << 30}
	ar, _, b := newTest(t, c, 2)
	ar.resident = "strata/flash"

	first := do(context.Background(), ar, "flash")
	if admittedAt(t, first) != "flash" {
		t.Fatal("the resident's turn was not admitted")
	}
	queued := make([]*request, 3)
	for i := range queued {
		queued[i] = doQueued(t, ar, "flash")
	}
	parent := doQueued(t, ar, "27b")
	notAdmitted(t, parent, "the parent before the cap")

	order := []string{"flash"}
	first.release()
	order = append(order, admittedAt(t, queued[0]))
	queued[0].release()
	// The streak is at the cap: the parent is admitted while three flash turns are
	// still queued. The card runs one request at a time, so it comes in as soon as
	// the resident's turn releases.
	order = append(order, admittedAt(t, parent))
	parent.release()
	for i := 1; i < 3; i++ {
		order = append(order, admittedAt(t, queued[i]))
		queued[i].release()
	}
	if order[2] != "27b" {
		t.Fatalf("admitted %v, want the parent after exactly 2 resident turns", order)
	}
	if ar.Resident() != "strata/flash" {
		t.Errorf("resident %q, want the route the queue returns to", ar.Resident())
	}
	_ = b
}

// A cancelled waiter leaves no phantom in the queue: once its request ends, the
// next request of the resident drains straight through.
func TestCancelledWaitLeavesNoPhantom(t *testing.T) {
	c := &card{free: 20 << 30}
	ar, _, _ := newTest(t, c, 32)
	ar.resident = "strata/flash"

	hold := do(context.Background(), ar, "flash")
	if admittedAt(t, hold) != "flash" {
		t.Fatal("the resident's turn was not admitted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	waiter := do(ctx, ar, "27b")
	awaitQueued(t, ar, 1)
	cancel()
	if err := <-waiter.err; err == nil {
		t.Fatal("a cancelled waiter was admitted")
	}
	hold.release()
	next := do(context.Background(), ar, "flash")
	if m := admittedAt(t, next); m != "flash" {
		t.Fatalf("admitted %s", m)
	}
	next.release()
	if ar.Resident() != "strata/flash" {
		t.Errorf("resident %q, want the one that never yielded", ar.Resident())
	}
}

func TestSwapOrderSaveUnloadLoad(t *testing.T) {
	c := &card{free: 20 << 30}
	ar, a, b := newTest(t, c, 32)
	ar.resident = a.id
	ar.opt.SaveLive = func(ctx context.Context, r Route) { a.card.record("yield " + r.ID) }

	r := do(context.Background(), ar, "27b")
	if m := <-r.admitted; m != "27b" {
		t.Fatalf("admitted %s", m)
	}
	want := []string{"yield " + a.id, "unload " + a.id + " 5m0s", "load " + b.id}
	if got := c.callsSince(0); strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("card calls %v, want the save, the unload, then the load", got)
	}
	r.release()
}

func TestVRAMGateWaitsForFreeMemory(t *testing.T) {
	c := &card{free: 2 << 20} // almost nothing free
	ar, _, b := newTest(t, c, 32)
	ar.opt.Routes[1].Need = 18 << 30
	ar.resident = "strata/flash"

	r := do(context.Background(), ar, "27b")
	select {
	case m := <-r.admitted:
		t.Fatalf("admitted %s with no VRAM free", m)
	case <-time.After(200 * time.Millisecond):
	}
	// The unload runs first, since it is what frees the VRAM; the load waits for
	// the gate, so nothing is loaded while the card is full.
	if n := loads(c.callsSince(0)); n != 0 {
		t.Errorf("card calls %v, want no load while the VRAM is short", c.callsSince(0))
	}
	c.setFree(20 << 30) // the unload freed the card
	if m := <-r.admitted; m != "27b" {
		t.Fatalf("admitted %s", m)
	}
	if got := c.callsSince(0); len(got) != 2 || got[0] != "unload strata/flash 5m0s" || got[1] != "load "+b.id {
		t.Errorf("card calls %v, want the unload, then the load after the gate opened", got)
	}
	r.release()
}

// A load that is refused leaves the resident alone: the card does not move, and
// the request is told the truth.
func TestLoadRefusalKeepsResident(t *testing.T) {
	c := &card{free: 20 << 30}
	ar, a, b := newTest(t, c, 32)
	b.loadErr = &engine.SlotError{Status: 503, Message: "GPU busy, retry after 30"}
	ar.resident = a.id

	r := do(context.Background(), ar, "27b")
	if err := <-r.err; err == nil {
		t.Fatal("the swap reported success")
	}
	if ar.Resident() != a.id {
		t.Errorf("resident %q after a refused load, want the old one", ar.Resident())
	}
	// the queue is clean: the next request drains straight through
	next := do(context.Background(), ar, "flash")
	if m := <-next.admitted; m != "flash" {
		t.Fatalf("admitted %s", m)
	}
	next.release()
}

func TestIntraRouterSwapUnloadsOldModel(t *testing.T) {
	c := &card{free: 20 << 30}
	swift := &fakeEngine{id: "llama/Swift-27B", model: "swift", card: c, loaded: true}
	orca := &fakeEngine{id: "llama/OrcaCyber", model: "orca", card: c}
	ar := New(Options{
		Routes: []Route{
			{ID: swift.id, Model: swift.model, Namespace: "cccccccccccc", Engine: swift},
			{ID: orca.id, Model: orca.model, Namespace: "dddddddddddd", Engine: orca},
		},
		MaxResidentStreak: 32, LoadTimeout: 30 * time.Second, VRAMMargin: 1 << 30,
		VRAMFree: c.freeBytes, PollInterval: 10 * time.Millisecond,
		Log: slog.New(slog.NewJSONHandler(quietWriter{}, nil)),
	})
	ar.resident = swift.id

	r := do(context.Background(), ar, "orca")
	if m := <-r.admitted; m != "orca" {
		t.Fatalf("admitted %s", m)
	}
	want := []string{"unload " + swift.id + " 5m0s", "load " + orca.id}
	got := c.callsSince(0)
	if len(got) != len(want) {
		t.Fatalf("card calls %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("card calls %v, want the old model unloaded before the new is loaded", got)
		}
	}
	r.release()
}
