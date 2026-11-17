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
	"sort"
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
// for a model that is not resident. LoadDelay makes a load slow, as loading 19 GiB
// of weights is on a real card.
type llamaFake struct {
	Dir       string
	LoadDelay time.Duration

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
		delay := f.LoadDelay
		f.mu.Unlock()
		if delay > 0 {
			time.Sleep(delay) // 19 GiB of weights take seconds
		}
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
	var body struct{ Filename, Model string }
	_ = json.NewDecoder(r.Body).Decode(&body)
	// llama.cpp names the model in the body, not in the query
	action, model := r.URL.Query().Get("action"), body.Model
	if model == "" {
		model = r.URL.Query().Get("model")
	}
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
	return twoRouteEnvWith(t, false, 0, 10*time.Millisecond, 64<<20, "")
}

// twoRouteEnvWith is the same, with strict mode (a request must name one of the
// routes), the idle-save token cap, the idle delay, the body limit, and the route
// that owns the card at startup ("" = nobody, as a fresh arbiter sees).
func twoRouteEnvWith(t *testing.T, strict bool, maxIdle int, idle time.Duration, maxBody int64, initialResident string) (*Proxy, *stratatest.Fake, *llamaFake, *store.Store, string, *syncBuffer, string) {
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
		Strict: strict,
		Routes: []arbiter.Route{
			{ID: "strata/flash", Dialect: "strata", Model: "flash", Base: strataServer.URL,
				Namespace: store.NamespaceOf("strata", "flash"),
				Engine:    engine.NewStrata(engine.Options{ID: "strata/flash", Model: "flash", Base: strataServer.URL})},
			{ID: "llama/swift", Dialect: "llama", Model: "swift", Base: llamaServer.URL, APIKey: "k",
				Namespace: store.NamespaceOf("llama", "swift"),
				Engine:    engine.NewLlama(engine.Options{ID: "llama/swift", Model: "swift", Base: llamaServer.URL, APIKey: "k"})},
		},
		MaxResidentStreak: 32, LoadTimeout: 30 * time.Second, VRAMMargin: 0,
		VRAMFree:        func() (int64, error) { return 1 << 40, nil },
		PollInterval:    10 * time.Millisecond,
		InitialResident: initialResident,
		Log:             log,
	})
	p := New(Options{Arbiter: arb, Store: st, MaxBody: maxBody, Metrics: NewMetrics(prometheus.NewRegistry()), Log: log,
		Policy: Policy{MinTokens: 10, ParkMinTokens: -1, RegrowTokens: 0, IdleDelay: idle,
			MaxIdleSaveTokens: maxIdle, SnapshotBytesPerToken: 64 << 10, BodyDivisor: 4,
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

// A route that gave the card away holds no live conversation, so its next request
// must restore the session file. llama.cpp's router reports no start time, so the
// cache cannot see that the model was unloaded and loaded again; the arbiter's
// swap note is what tells it.
func TestSwapBackRestoresTheFile(t *testing.T) {
	p, _, _, _, url, _, dir := twoRouteEnv(t)

	postModel(t, url, "flash", withModel("flash", conversation("A", 3)))
	eventually(t, "flash's file", func() bool { return len(filesIn(t, p, "strata/flash", dir)) == 1 })
	postModel(t, url, "swift", withModel("swift", conversation("B", 3)))
	eventually(t, "swift's file", func() bool { return len(filesIn(t, p, "llama/swift", dir)) == 1 })

	// The card goes back to flash and then to Swift. Swift's engine was unloaded
	// and loaded again, so its conversation is on disk, not in the engine.
	postModel(t, url, "flash", withModel("flash", conversation("A", 3)))
	postModel(t, url, "swift", withModel("swift", conversation("B", 3)))
	p.mu.Lock()
	last := p.recent[len(p.recent)-1].Action
	p.mu.Unlock()
	if last != "disk" {
		t.Errorf("Swift after a swap back: action %q, want disk", last)
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

// postBody sends a generation request and returns the status and the answer body.
func postBody(t *testing.T, url, body string) (int, string) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, url+"/v1/chat/completions", strings.NewReader(body))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

// The arbiter serves exactly the models it was given. A request naming a model
// that is not one of the routes is refused, not answered by the route that
// happens to hold the card: answering it gives the client a different model under
// the label it asked for, and its token counts, session files and the harness's
// model name all inherit the substitution.
func TestStrictRefusesAnUnknownModel(t *testing.T) {
	p, _, llama, _, url, logs, _ := twoRouteEnvWith(t, true, 0, 10*time.Millisecond, 64<<20, "")
	if n := postModel(t, url, "flash", withModel("flash", conversation("A", 2))); n == 0 {
		t.Fatal("flash turn failed")
	}
	status, body := postBody(t, url, withModel("ghost", conversation("A", 3)))
	if status != http.StatusNotFound {
		t.Fatalf("status %d, want 404", status)
	}
	if !strings.Contains(body, `"type":"invalid_request_error"`) {
		t.Errorf("body %q, want the client's error shape", body)
	}
	if !strings.Contains(body, "model ghost is not served by this arbiter") {
		t.Errorf("body %q, want the refused model named", body)
	}
	if llama.Take() != nil {
		t.Error("a request for a model that is not a route touched the other engine")
	}
	if p.Resident() != "strata/flash" {
		t.Errorf("resident %q, want flash still resident", p.Resident())
	}
	if !strings.Contains(logs.String(), `"model":"ghost"`) {
		t.Error("the log does not name the refused model")
	}
	// A refusal is a request: /requests shows the model asked for and the refusal.
	rec := p.requestsJSON()
	if !strings.Contains(rec, `"model":"ghost"`) || !strings.Contains(rec, `"diskcache":"refused"`) {
		t.Errorf("/requests %q, want the refused model and the refusal recorded", rec)
	}
	// The refusal is counted, not only logged: a scrape shows what the arbiter
	// turned away, which is how an operator finds a model-name mixup.
	if v := counterValue(t, p.m.registry(), "strata_diskcache_requests_total", "action", "refused"); v != 1 {
		t.Errorf("requests_total{action=\"refused\"} = %v, want 1", v)
	}
}

// Strict mode refuses a generation request that names no model: the arbiter does
// not choose for a client that asked for none.
func TestStrictRefusesARequestWithNoModel(t *testing.T) {
	p, _, _, _, url, logs, _ := twoRouteEnvWith(t, true, 0, 10*time.Millisecond, 64<<20, "")
	status, body := postBody(t, url, conversation("A", 3))
	if status != http.StatusNotFound {
		t.Fatalf("status %d, want 404", status)
	}
	if !strings.Contains(body, "the request names no model") {
		t.Errorf("body %q, want the missing model named as the reason", body)
	}
	if p.Resident() != "" {
		t.Errorf("resident %q, want the card untouched", p.Resident())
	}
	if !strings.Contains(logs.String(), `"model":""`) {
		t.Error("the log does not show the empty model")
	}
}

// A background idle save of a conversation over --max-idle-save-tokens is
// skipped: a 100K-token session writes 1.4 GB in ~110 s behind the save lock, and
// the yield save at the next switch writes the same file in ~4 s.
func TestIdleSaveSkipsALongConversation(t *testing.T) {
	p, _, _, _, url, logs, dir := twoRouteEnvWith(t, false, 1000, 10*time.Millisecond, 64<<20, "")
	if status, _ := postBody(t, url, withModel("flash", longBody(2000))); status != 200 {
		t.Fatalf("status %d, want 200", status)
	}
	eventually(t, "the skip", func() bool { return strings.Contains(logs.String(), `"reason":"idle_too_long"`) })
	if len(filesIn(t, p, "strata/flash", dir)) != 0 {
		t.Error("a conversation over the cap was saved in the background")
	}
}

// A request already queued for the card is not made to wait for a background
// save: the idle save steps out of the way, and the switch save on that request
// writes the file.
func TestIdleSaveStepsOutForAQueuedRequest(t *testing.T) {
	p, _, _, _, url, logs, dir := twoRouteEnvWith(t, false, 0, 200*time.Millisecond, 64<<20, "")
	if status, _ := postBody(t, url, withModel("flash", conversation("A", 3))); status != 200 {
		t.Fatalf("status %d", status)
	}
	p.cache.Queue(1, "strata/flash") // a request is queued inside the idle delay
	eventually(t, "the skip", func() bool { return strings.Contains(logs.String(), `"reason":"queue_busy"`) })
	if len(filesIn(t, p, "strata/flash", dir)) != 0 {
		t.Error("the background save ran with a request queued")
	}
}

// A route's bytes-per-token is clamped to 4 KB..256 KB/token. Measured: Strata
// k8v4 13.8 KB/token, a llama slot 85.7 KB/token. A file that is mostly fixed
// overhead (3 MB for 10 tokens) would set 300 KB/token, and the next reserve reads
// a long session as hundreds of gigabytes over the cap: the save is skipped and
// the session lost.
func TestLearnedBytesPerTokenIsClamped(t *testing.T) {
	p, strata, _, _, url, _, dir := twoRouteEnv(t)
	strata.SavedTokens, strata.SaveBytes = 10, 3072000 // 300 KB/token
	if status, _ := postBody(t, url, withModel("flash", conversation("A", 2))); status != 200 {
		t.Fatalf("status %d", status)
	}
	eventually(t, "the file", func() bool { return len(filesIn(t, p, "strata/flash", dir)) == 1 })
	p.cache.mu.Lock()
	bpt := p.cache.routes["strata/flash"].bpt
	p.cache.mu.Unlock()
	if bpt != 256<<10 {
		t.Errorf("bytes per token %d, want the 256 KB/token bound", bpt)
	}
	strata.SavedTokens, strata.SaveBytes = 10, 1000 // 100 bytes/token
	if status, _ := postBody(t, url, withModel("flash", conversation("B", 2))); status != 200 {
		t.Fatalf("status %d", status)
	}
	eventually(t, "the second file", func() bool { return len(filesIn(t, p, "strata/flash", dir)) == 2 })
	p.cache.mu.Lock()
	bpt = p.cache.routes["strata/flash"].bpt
	p.cache.mu.Unlock()
	if bpt != 4<<10 {
		t.Errorf("bytes per token %d, want the 4 KB/token floor", bpt)
	}
}

// counterValue is one label value of a CounterVec.
func counterValue(t *testing.T, reg *prometheus.Registry, name, label, value string) float64 {
	t.Helper()
	g, err := reg.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, mf := range g {
		if mf.GetName() != name {
			continue
		}
		for _, m := range mf.GetMetric() {
			for _, l := range m.GetLabel() {
				if l.GetName() == label && l.GetValue() == value {
					return m.GetCounter().GetValue()
				}
			}
		}
	}
	return -1
}

// gaugeSeries is the named gauge's samples, keyed by its labels in sorted order.
func gaugeSeries(t *testing.T, reg *prometheus.Registry, name string) map[string]float64 {
	t.Helper()
	g, err := reg.Gather()
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]float64{}
	for _, mf := range g {
		if mf.GetName() != name {
			continue
		}
		for _, m := range mf.GetMetric() {
			ls := m.GetLabel()
			sort.Slice(ls, func(i, j int) bool { return ls[i].GetName() < ls[j].GetName() })
			key := ""
			for _, l := range ls {
				key += l.GetName() + "=" + l.GetValue() + " "
			}
			out[strings.TrimSpace(key)] = m.GetGauge().GetValue()
		}
	}
	return out
}

// The route→model table is written at startup, so every route has its
// resident_info series from the first scrape: 0 for the routes that do not hold
// the card, 1 for the one that does. A gauge that only appears after a route's
// first swap leaves an operator guessing which model owns the card.
func TestResidentInfoNamesTheModelFromTheStart(t *testing.T) {
	p, _, _, _, url, _, _ := twoRouteEnvWith(t, true, 0, 10*time.Millisecond, 64<<20, "")
	reg := p.m.registry()
	want := map[string]float64{"model=flash route=strata/flash": 0, "model=swift route=llama/swift": 0}
	got := gaugeSeries(t, reg, "strata_diskcache_resident_info")
	if len(got) != len(want) {
		t.Fatalf("series %v, want both routes named before any request", got)
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("%s = %v, want %v", k, got[k], v)
		}
	}
	if n := postModel(t, url, "flash", withModel("flash", conversation("A", 2))); n == 0 {
		t.Fatal("flash turn failed")
	}
	got = gaugeSeries(t, reg, "strata_diskcache_resident_info")
	if got["model=flash route=strata/flash"] != 1 {
		t.Errorf("the resident gauge is %v, want 1 for the route holding the card", got["model=flash route=strata/flash"])
	}
	if got["model=swift route=llama/swift"] != 0 {
		t.Errorf("the idle route's gauge is %v, want 0", got["model=swift route=llama/swift"])
	}
}

// A generation body over --max-body-mib is streamed through unbuffered, and it
// pays admission like any request: the model in its prefix is made the resident
// before the bytes reach a server, so a request never runs on a route whose model
// is not loaded, and the conversation the card holds is yielded by the swap.
// The model is read from the prefix, so a body cut at the limit is still routed.
func TestOversizedBodyIsAdmittedToItsRoute(t *testing.T) {
	p, strata, llama, _, url, _, _ := twoRouteEnvWith(t, true, 0, 10*time.Millisecond, 4096, "")
	if status, _ := postBody(t, url, withModel("flash", conversation("A", 3))); status != 200 {
		t.Fatalf("status %d, want 200", status)
	}
	body := withModel("swift", longBody(1500)) // ~7.5 KB: over the 4 KiB limit
	if len(body) <= 4096 {
		t.Fatalf("test body is %d bytes, want over the limit", len(body))
	}
	if status, _ := postBody(t, url, body); status != 200 {
		t.Fatalf("status %d, want 200", status)
	}
	if p.Resident() != "llama/swift" {
		t.Errorf("resident %q, want the card moved to the model the body named", p.Resident())
	}
	if acts := llama.Take(); len(acts) < 2 || acts[0] != "load swift" {
		t.Fatalf("llama actions %v, want the load before the request", acts)
	}
	got := llama.Bodies()
	if len(got) != 1 || len(got[0]) != len(body) {
		t.Errorf("the engine got %d bytes of the body, want all %d streamed through", len(got), len(body))
	}
	if strata.Loaded {
		t.Error("Strata is still resident: the card did not move")
	}
}

// The oversized path is the same rule: a model the arbiter does not serve is
// refused, not answered by the resident. Its name is in the prefix the proxy read.
func TestStrictOversizedBodyIsRefusedByTheSameRule(t *testing.T) {
	p, _, llama, _, url, logs, _ := twoRouteEnvWith(t, true, 0, 10*time.Millisecond, 4096, "")
	if status, _ := postBody(t, url, withModel("flash", conversation("A", 2))); status != 200 {
		t.Fatalf("status %d, want 200", status)
	}
	status, body := postBody(t, url, withModel("ghost", longBody(1500)))
	if status != http.StatusNotFound {
		t.Fatalf("status %d, want 404", status)
	}
	if !strings.Contains(body, "model ghost is not served by this arbiter") {
		t.Errorf("body %q, want the refused model named", body)
	}
	if llama.Take() != nil {
		t.Error("a refused oversized body touched the other engine")
	}
	if p.Resident() != "strata/flash" {
		t.Errorf("resident %q, want flash untouched", p.Resident())
	}
	if !strings.Contains(logs.String(), `"model":"ghost"`) {
		t.Error("the log does not name the refused model")
	}
}

// The queue the arbiter observes is what makes a background save step out of the
// way. The wiring, not a direct call: a slow load keeps one admission running, the
// next request for that route is queued, and the idle save fires while the queue
// is not empty, so it skips and the switch save on the queued request writes the
// file.
func TestQueueWiringMakesTheIdleSaveStepOut(t *testing.T) {
	p, _, llama, _, url, logs, dir := twoRouteEnvWith(t, false, 0, 250*time.Millisecond, 64<<20, "")
	llama.LoadDelay = 400 * time.Millisecond
	if status, _ := postBody(t, url, withModel("flash", conversation("A", 3))); status != 200 {
		t.Fatalf("status %d, want 200", status)
	}
	send := func(model, body string) <-chan int {
		ch := make(chan int, 1)
		go func() {
			req, _ := http.NewRequest(http.MethodPost, url+"/v1/chat/completions",
				strings.NewReader(withModel(model, body)))
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				ch <- 0
				return
			}
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()
			ch <- resp.StatusCode
		}()
		return ch
	}
	// B takes the card: its admission runs for the length of the load. C queues
	// behind it, inside the idle delay, so the queue is not empty when the
	// background save for flash's turn fires.
	b, c := send("swift", conversation("B", 2)), send("swift", conversation("C", 2))
	time.Sleep(120 * time.Millisecond)
	c2 := send("swift", conversation("D", 2))
	eventually(t, "the skip", func() bool { return strings.Contains(logs.String(), `"reason":"queue_busy"`) })
	if status := <-b; status != 200 {
		t.Errorf("B's status %d, want 200", status)
	}
	for _, ch := range []<-chan int{c, c2} {
		if status := <-ch; status != 200 {
			t.Errorf("a queued turn's status %d, want 200", status)
		}
	}
	if n := count(llama.Take(), "load swift"); n != 1 {
		t.Errorf("%d loads of the 27B route, want exactly one", n)
	}
	// The background save never ran; the switch save on B's admission did.
	eventually(t, "flash's file from the switch save", func() bool { return len(filesIn(t, p, "strata/flash", dir)) == 1 })
}

// The arbiter is told which route owns the card at startup (the --lazy Strata
// route), so the first scrape says who holds the card. SetRoutes writing 0 for
// every route is not enough: a reader of resident_info at startup sees nobody
// owning the card, which is the mixup this gauge exists to make visible.
func TestResidentInfoSeedsTheInitialResident(t *testing.T) {
	p, _, _, _, _, _, _ := twoRouteEnvWith(t, true, 0, 10*time.Millisecond, 64<<20, "strata/flash")
	got := gaugeSeries(t, p.m.registry(), "strata_diskcache_resident_info")
	if got["model=flash route=strata/flash"] != 1 {
		t.Errorf("the initial resident's gauge is %v, want 1 before any request", got["model=flash route=strata/flash"])
	}
	if got["model=swift route=llama/swift"] != 0 {
		t.Errorf("the idle route's gauge is %v, want 0", got["model=swift route=llama/swift"])
	}
}
