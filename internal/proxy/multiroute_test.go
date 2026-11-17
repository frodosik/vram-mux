package proxy

import (
	"bufio"
	"context"
	"encoding/json"
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

	"github.com/tfyl/strata-diskcache/internal/arbiter"
	"github.com/tfyl/strata-diskcache/internal/engine"
	"github.com/tfyl/strata-diskcache/internal/store"
	"github.com/tfyl/strata-diskcache/internal/strata/stratatest"
)

// llamaFake is a llama.cpp b12445 router with two presets: it loads a model only
// when asked, reports the presets in GET /models without loading any of them,
// saves and restores a model's session, and answers 400 "model is not running"
// for a model that is not resident.
type llamaFake struct {
	Dir string

	mu      sync.Mutex
	models  map[string]*llamaModel
	order   []string // the presets, in the order given, for a stable /models answer
	actions []string
	bodies  []string
	entered chan string
}

type llamaModel struct {
	id     string
	args   []string
	loaded bool
}

func newLlamaFake(dir string, ids ...string) *llamaFake {
	f := &llamaFake{Dir: dir, models: map[string]*llamaModel{}}
	for _, id := range ids {
		f.models[id] = &llamaModel{id: id, args: []string{"llama-server", "--model", "/models/" + id + ".gguf", "--ctx-size", "131072"}}
		f.order = append(f.order, id)
	}
	return f
}

func (f *llamaFake) Take() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	a := f.actions
	f.actions = nil
	return a
}

func (f *llamaFake) Bodies() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	b := f.bodies
	f.bodies = nil
	return b
}

func (f *llamaFake) Loaded(id string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.models[id].loaded
}

func (f *llamaFake) Entered() <-chan string {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.entered == nil {
		f.entered = make(chan string, 16)
	}
	return f.entered
}

func (f *llamaFake) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	switch {
	case r.URL.Path == "/models" && r.Method == http.MethodGet:
		var b strings.Builder
		b.WriteString(`{"data":[`)
		for i, id := range f.order {
			m := f.models[id]
			if i > 0 {
				b.WriteString(",")
			}
			state := "unloaded"
			if m.loaded {
				state = "loaded"
			}
			args, _ := json.Marshal(m.args)
			fmt.Fprintf(&b, `{"id":%q,"status":{"value":%q},"args":%s}`, m.id, state, args)
		}
		b.WriteString("]}")
		f.mu.Unlock()
		fmt.Fprint(w, b.String())
	case r.URL.Path == "/models/load":
		var body struct{ Model string }
		_ = json.NewDecoder(r.Body).Decode(&body)
		m := f.models[body.Model]
		if m == nil {
			f.mu.Unlock()
			http.Error(w, `{"error":"model is not a preset"}`, http.StatusNotFound)
			return
		}
		f.actions = append(f.actions, "load "+body.Model)
		m.loaded = true
		f.mu.Unlock()
		fmt.Fprint(w, `{"status":"loaded"}`)
	case r.URL.Path == "/models/unload":
		var body struct{ Model string }
		_ = json.NewDecoder(r.Body).Decode(&body)
		m := f.models[body.Model]
		f.actions = append(f.actions, "unload "+body.Model)
		if m == nil {
			f.mu.Unlock()
			http.Error(w, `{"error":"model is not a preset"}`, http.StatusNotFound)
			return
		}
		if !m.loaded {
			f.mu.Unlock()
			http.Error(w, `{"error":"model is not running"}`, http.StatusBadRequest)
			return
		}
		m.loaded = false
		f.mu.Unlock()
		fmt.Fprint(w, `{"status":"unloaded"}`)
	case strings.HasPrefix(r.URL.Path, "/slots/0"):
		f.slot(w, r) // unlocks
	case r.URL.Path == "/v1/chat/completions":
		f.gen(w, r) // unlocks
	default:
		f.mu.Unlock()
		http.Error(w, `{"error":"not found"}`, http.StatusNotFound)
	}
}

func (f *llamaFake) slot(w http.ResponseWriter, r *http.Request) {
	var body struct{ Filename string }
	_ = json.NewDecoder(r.Body).Decode(&body)
	action, model := r.URL.Query().Get("action"), r.URL.Query().Get("model")
	m := f.models[model]
	if m == nil {
		f.mu.Unlock()
		http.Error(w, `{"error":"model is not a preset"}`, http.StatusNotFound)
		return
	}
	if !m.loaded {
		f.mu.Unlock()
		http.Error(w, `{"error":"model is not running"}`, http.StatusBadRequest)
		return
	}
	f.actions = append(f.actions, "llama "+action+" "+body.Filename)
	entered := f.entered
	f.mu.Unlock()
	if entered != nil {
		entered <- action
	}
	p := filepath.Join(f.Dir, body.Filename)
	if action == "save" {
		_ = os.WriteFile(p, make([]byte, 1000), 0o600)
		fmt.Fprint(w, `{"n_saved":20000,"n_written":1000}`)
		return
	}
	if _, err := os.Stat(p); err != nil {
		http.Error(w, `{"error":"no saved session"}`, http.StatusNotFound)
		return
	}
	fmt.Fprint(w, `{"n_restored":20000,"n_read":1000}`)
}

// gen answers a generation request. ServeHTTP holds the lock on entry, as the
// Strata fake does.
func (f *llamaFake) gen(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	var b struct {
		Model string
	}
	_ = json.Unmarshal(body, &b)
	f.actions = append(f.actions, "gen "+b.Model)
	f.bodies = append(f.bodies, string(body))
	if m := f.models[b.Model]; m != nil {
		m.loaded = true // the router loads the model this request asks for
	}
	f.mu.Unlock()
	w.Header().Set("Content-Type", "text/event-stream")
	fl := w.(http.Flusher)
	fmt.Fprint(w, "data: {\"i\":0}\n\n")
	fl.Flush()
	fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	fl.Flush()
}

// twoRouteEnv starts Strata (flash) and a llama router (swift) behind one proxy,
// with a small MinTokens so the short test conversations are saved.
func twoRouteEnv(t *testing.T) (*Proxy, *stratatest.Fake, *llamaFake, *store.Store, string, *syncBuffer, string) {
	t.Helper()
	dir := t.TempDir()
	logs := &syncBuffer{}
	log := slog.New(slog.NewJSONHandler(logs, nil))

	strataFake := stratatest.New(dir)
	strataFake.Model = "flash"
	strataFake.MinTokens = 10
	strataServer := httptest.NewServer(strataFake)
	t.Cleanup(strataServer.Close)
	llamaFake := newLlamaFake(dir, "swift")
	llamaServer := httptest.NewServer(llamaFake)
	t.Cleanup(llamaServer.Close)

	st := store.New(store.Options{Dir: dir, MaxBytes: 64 << 30, MaxAge: time.Hour, Log: log,
		FreeSpace: func(string) (int64, error) { return 1 << 40, nil }})
	if err := st.Load(); err != nil {
		t.Fatal(err)
	}
	arb := arbiter.New(arbiter.Options{
		Routes: []arbiter.Route{
			{ID: "strata/flash", Dialect: "strata", Model: "flash", Base: strataServer.URL,
				Namespace: store.NamespaceOf("strata", "flash"),
				Engine:    engine.NewStrata(engine.Options{ID: "strata/flash", Model: "flash", Base: strataServer.URL})},
			{ID: "llama/swift", Dialect: "llama", Model: "swift", Base: llamaServer.URL, APIKey: "k",
				Namespace: store.NamespaceOf("llama", "swift"),
				Engine:    engine.NewLlama(engine.Options{ID: "llama/swift", Model: "swift", Base: llamaServer.URL, APIKey: "k"})},
		},
		MaxResidentStreak: 32, LoadTimeout: 30 * time.Second, VRAMMargin: 0,
		VRAMFree:     func() (int64, error) { return 1 << 40, nil },
		PollInterval: 10 * time.Millisecond,
		Log:          log,
	})
	p := New(Options{Arbiter: arb, Store: st, MaxBody: 64 << 20, Metrics: NewMetrics(prometheus.NewRegistry()), Log: log,
		Policy: Policy{MinTokens: 10, ParkMinTokens: -1, RegrowTokens: 0, IdleDelay: 10 * time.Millisecond,
			SnapshotBytesPerToken: 64 << 10, BodyDivisor: 4,
			MemAvailable: func() (int64, error) { return 8 << 30, nil }}})
	px := httptest.NewServer(p)
	t.Cleanup(func() { px.Close(); _ = p.Close(context.Background()) })
	return p, strataFake, llamaFake, st, px.URL, logs, dir
}

// postModel sends a named model's request and returns the status.
func postModel(t *testing.T, url, model string, body string) int {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, url+"/v1/chat/completions", strings.NewReader(body))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	n := 0
	for sc := bufio.NewScanner(resp.Body); sc.Scan(); {
		if strings.HasPrefix(sc.Text(), "data: ") {
			n++
		}
	}
	if resp.StatusCode != 200 {
		return resp.StatusCode
	}
	return n
}

// withModel puts the model name in a conversation body.
func withModel(model, body string) string {
	return strings.Replace(body, `{"stream":true,`, `{"model":"`+model+`","stream":true,`, 1)
}

func TestTwoRoutesShareOneSessionFolder(t *testing.T) {
	p, strata, llama, _, url, _, dir := twoRouteEnv(t)
	body := conversation("A", 3)

	// flash-next runs; its conversation is saved in its own namespace.
	if n := postModel(t, url, "flash", withModel("flash", body)); n == 0 {
		t.Fatal("flash turn failed")
	}
	eventually(t, "flash's file", func() bool { return len(filesIn(t, p, "strata/flash", dir)) == 1 })

	// the 27B route takes the card: yield, unload, load, then its own turn.
	if n := postModel(t, url, "swift", withModel("swift", conversation("B", 3))); n == 0 {
		t.Fatal("swift turn failed")
	}
	eventually(t, "swift's file", func() bool { return len(filesIn(t, p, "llama/swift", dir)) == 1 })

	if p.Resident() != "llama/swift" {
		t.Fatalf("resident %q, want the route the last request named", p.Resident())
	}
	if !llama.Loaded("swift") {
		t.Error("the 27B route is not resident")
	}
	if strata.Loaded {
		t.Error("Strata is still resident: the card did not move")
	}
}

// filesIn lists the store's files of one route's namespace.
func filesIn(t *testing.T, p *Proxy, routeID, dir string) []string {
	t.Helper()
	ns := p.cache.routes[routeID].ns
	m, _ := filepath.Glob(filepath.Join(dir, "dc-"+ns+"-*.session"))
	return m
}

func TestSwapSavesThenUnloadsThenLoads(t *testing.T) {
	_, strata, llama, _, url, _, _ := twoRouteEnv(t)
	entered := strata.Entered()

	// flash-next has a live conversation of 20000 tokens; a 27B request arrives.
	strata.Set(20000, 0)
	if n := postModel(t, url, "flash", withModel("flash", conversation("A", 3))); n == 0 {
		t.Fatal("flash turn failed")
	}
	strata.Update(func(f *stratatest.Fake) { f.MinTokens = 10 })
	if n := postModel(t, url, "swift", withModel("swift", conversation("B", 3))); n == 0 {
		t.Fatal("swift turn failed")
	}

	// Strata saw: the save of the live conversation, then the unload, then its own
	// generation request. The llama router saw the load and the generation.
	strataActions := strata.Take()
	// The yield order is the point: the live conversation is saved, and only then
	// is Strata unloaded. A cold start loads Strata first and the idle save lands
	// before the switch, so the log is load, gen, save, unload.
	save, unload := -1, -1
	for i, a := range strataActions {
		if strings.HasPrefix(a, "save") {
			save = i
		}
		if strings.HasPrefix(a, "unload 300") {
			unload = i
		}
	}
	if save < 0 || unload < 0 || save > unload {
		t.Fatalf("strata actions %v, want the save before the unload", strataActions)
	}
	if got := llama.Take(); len(got) < 2 || got[0] != "load swift" || got[1] != "gen swift" {
		t.Fatalf("llama actions %v, want the load then the request", got)
	}
	_ = entered
}

// The point of the namespace: the 27B route learning a fingerprint must leave
// flash-next's sessions on disk.
func TestFingerprintLearningIsScopedPerRoute(t *testing.T) {
	p, strata, _, st, url, _, dir := twoRouteEnv(t)
	if n := postModel(t, url, "flash", withModel("flash", conversation("A", 3))); n == 0 {
		t.Fatal("flash turn failed")
	}
	eventually(t, "flash's file", func() bool { return len(filesIn(t, p, "strata/flash", dir)) == 1 })
	strataFiles := filesIn(t, p, "strata/flash", dir)
	if len(strataFiles) != 1 {
		t.Fatalf("no flash file to protect: %v", strataFiles)
	}

	// The 27B route runs, and flash's engine is reloaded with another model: its
	// fingerprint changes, and only its route's stale files go.
	if n := postModel(t, url, "swift", withModel("swift", conversation("B", 3))); n == 0 {
		t.Fatal("swift turn failed")
	}
	eventually(t, "swift's file", func() bool { return len(filesIn(t, p, "llama/swift", dir)) == 1 })
	swiftFiles := filesIn(t, p, "llama/swift", dir)
	strata.Update(func(f *stratatest.Fake) { f.Model, f.Unloaded, f.Loaded = "swift-27b", true, false })
	if n := postModel(t, url, "flash", withModel("flash", conversation("C", 1))); n == 0 {
		t.Fatal("flash turn after a new fingerprint failed")
	}
	if len(filesIn(t, p, "llama/swift", dir)) != 1 {
		t.Errorf("learning flash's new fingerprint deleted the 27B route's sessions: %v", swiftFiles)
	}
	if st.AnyFingerprint(store.NamespaceOf("llama", "swift")) == "" {
		t.Error("the 27B route's namespace lost its fingerprint")
	}
	_ = strataFiles
}

// Requests for the resident route are never preempted, and the cross-engine
// request is admitted only when its turn comes.
// The queue rule, from the client's side: a turn already queued on the resident
// runs before the cross-engine request gets the card, so the card moves once.
func TestResidentDrainsAcrossTheProxy(t *testing.T) {
	p, strata, llama, _, url, _, _ := twoRouteEnv(t)
	strata.Set(20000, 0)
	postModel(t, url, "flash", withModel("flash", conversation("A", 1)))
	strata.Take()
	llama.Take()

	// flash-next has a turn queued; the parent asks for the card behind it.
	queued := make(chan int, 1)
	go func() { queued <- postModel(t, url, "flash", withModel("flash", conversation("A", 2))) }()
	time.Sleep(40 * time.Millisecond)
	if n := postModel(t, url, "swift", withModel("swift", conversation("P", 2))); n == 0 {
		t.Fatal("the parent's turn failed")
	}
	if n := <-queued; n == 0 {
		t.Fatal("the resident's queued turn failed")
	}
	if p.Resident() != "llama/swift" {
		t.Fatalf("resident %q, want the card moved after the drain", p.Resident())
	}
	if loads := count(llama.Take(), "load swift"); loads != 1 {
		t.Errorf("%d loads of the 27B route, want exactly one (no oscillation)", loads)
	}
	if unloads := count(strata.Take(), "unload"); unloads != 1 {
		t.Errorf("%d unloads of Strata, want exactly one", unloads)
	}
}

func count(actions []string, prefix string) int {
	n := 0
	for _, a := range actions {
		if strings.HasPrefix(a, prefix) {
			n++
		}
	}
	return n
}

// A model name the proxy does not know goes to the resident, and a request with
// no model name at all never causes a swap.
func TestUnknownModelGoesToTheResident(t *testing.T) {
	p, strata, llama, _, url, _, _ := twoRouteEnv(t)
	strata.Set(20000, 0)
	if n := postModel(t, url, "flash", withModel("flash", conversation("A", 2))); n == 0 {
		t.Fatal("flash turn failed")
	}
	before := llama.Take()
	if n := postModel(t, url, "flash", `{"messages":[{"role":"user","content":"x"}]}`); n == 0 {
		t.Fatal("a body with no model failed")
	}
	if llama.Take() != nil && len(llama.Take()) > 0 {
		t.Errorf("a model-less request moved the card: %v", before)
	}
	if p.Resident() != "strata/flash" {
		t.Fatalf("resident %q, want flash still resident", p.Resident())
	}
	if n := postModel(t, url, "flash", withModel("unknown-model", conversation("A", 3))); n == 0 {
		t.Fatal("an unknown model name failed")
	}
	if p.Resident() != "strata/flash" {
		t.Errorf("resident %q after an unknown model, want no swap", p.Resident())
	}
}
