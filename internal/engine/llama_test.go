package engine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// the adapter must keep satisfying the interface the arbiter uses
var _ Engine = (*Llama)(nil)

// llamaMock is a fake llama.cpp b12445 router: the presets the test sets and
// every request it saw. GET /models never loads anything, so a read of it
// changes nothing here.
type llamaMock struct {
	mu sync.Mutex

	models                               []llamaModel
	slotStatus                           int // non-zero: a slot call answers this status
	slotErr                              string
	slotBody                             string // the success body; default llama's save answer
	loadStatus, unloadStatus, wakeStatus int

	loadStays  bool // load answered success, the instance keeps reporting unloaded
	loadFailed bool // load answered success, the instance died on start
	loadBusy   bool // load refused: the model is already running
	exitCode   int  // the exit code a failed instance reports

	reqs []llamaRec
}

type llamaModel struct {
	id, value string
	args      []string
}

type llamaRec struct {
	method, path, query, body, origin, auth, ctype string
}

func newLlamaMock() *llamaMock {
	return &llamaMock{models: []llamaModel{
		{id: "Swift-27B", value: "unloaded", args: []string{"/models/swift.gguf", "-ngl", "99", "--ctx-size", "16384"}},
		{id: "flash-next", value: "loaded", args: []string{"/models/flash.gguf", "-ngl", "99"}},
	}}
}

func (m *llamaMock) update(fn func(m *llamaMock)) {
	m.mu.Lock()
	fn(m)
	m.mu.Unlock()
}

func (m *llamaMock) requests() []llamaRec {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]llamaRec(nil), m.reqs...)
}

func (m *llamaMock) statusOf(id string) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, e := range m.models {
		if e.id == id {
			return e.value
		}
	}
	return ""
}

func (m *llamaMock) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	m.mu.Lock()
	defer m.mu.Unlock()
	raw, _ := io.ReadAll(r.Body)
	m.reqs = append(m.reqs, llamaRec{method: r.Method, path: r.URL.Path, query: r.URL.RawQuery, body: string(raw),
		origin: r.Header.Get("Origin"), auth: r.Header.Get("Authorization"), ctype: r.Header.Get("Content-Type")})
	w.Header().Set("Content-Type", "application/json")
	switch {
	case r.Method == http.MethodGet && r.URL.Path == "/models":
		var v struct {
			Data []modelEntry `json:"data"`
		}
		for _, e := range m.models {
			var me modelEntry
			me.ID = e.id
			me.Status.Value = e.value
			me.Status.Args = e.args
			if m.loadFailed {
				me.Status.Failed = true
				me.Status.ExitCode = m.exitCode
			}
			v.Data = append(v.Data, me)
		}
		b, _ := json.Marshal(v)
		_, _ = io.WriteString(w, string(b))
	case r.Method == http.MethodPost && r.URL.Path == "/slots/0":
		if m.slotStatus != 0 {
			msg := m.slotErr
			if msg == "" {
				msg = "refused"
			}
			w.WriteHeader(m.slotStatus)
			fmt.Fprintf(w, `{"error":%q}`, msg)
			return
		}
		body := m.slotBody
		if body == "" {
			body = `{"n_saved":123,"n_written":1000}`
		}
		_, _ = io.WriteString(w, body)
	case r.Method == http.MethodPost && r.URL.Path == "/models/load":
		if m.loadBusy {
			// "already running" means the model is resident: the list says loaded
			want := modelName(raw)
			for i := range m.models {
				if m.models[i].id == want {
					m.models[i].value = "loaded"
				}
			}
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, `{"error":{"code":400,"message":"model is already running"}}`)
			return
		}
		if m.loadStatus != 0 {
			w.WriteHeader(m.loadStatus)
			_, _ = io.WriteString(w, `{"error":"the model does not fit in VRAM"}`)
			return
		}
		// the router answers at once; the instance starts in the background
		want := modelName(raw)
		for i := range m.models {
			if m.models[i].id == want && !m.loadStays {
				m.models[i].value = "loaded"
			}
		}
		_, _ = io.WriteString(w, `{"success":true}`)
	case r.Method == http.MethodPost && r.URL.Path == "/models/unload":
		if m.unloadStatus == http.StatusBadRequest {
			w.WriteHeader(m.unloadStatus)
			_, _ = io.WriteString(w, `{"error":"model is not running"}`)
			return
		}
		if m.unloadStatus != 0 {
			w.WriteHeader(m.unloadStatus)
			_, _ = io.WriteString(w, `{"error":"the router refused"}`)
			return
		}
		_, _ = io.WriteString(w, `{"success":true}`)
	case r.Method == http.MethodPost && r.URL.Path == "/v1/chat/completions":
		if m.wakeStatus != 0 {
			w.WriteHeader(m.wakeStatus)
			_, _ = io.WriteString(w, `{"error":"refused"}`)
			return
		}
		_, _ = io.WriteString(w, `{"choices":[{"finish_reason":"length"}]}`)
	default:
		// in router mode POST /unload and POST /model do not exist
		w.WriteHeader(http.StatusNotFound)
		_, _ = io.WriteString(w, `{"error":"unknown endpoint"}`)
	}
}

// modelName is the model a POST /models/load|unload body names.
func modelName(raw []byte) string {
	var v struct {
		Model string `json:"model"`
	}
	_ = json.Unmarshal(raw, &v)
	return v.Model
}

func TestLlamaStatus(t *testing.T) {
	m := newLlamaMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewLlama(Options{ID: "llama/Swift-27B", Model: "Swift-27B", Base: srv.URL})
	ctx := context.Background()

	if e.ID() != "llama/Swift-27B" || e.Model() != "Swift-27B" {
		t.Errorf("ID %q, Model %q", e.ID(), e.Model())
	}
	st, err := e.Status(ctx)
	if err != nil || st != (Status{Loaded: false}) {
		t.Fatalf("status of an unloaded preset %+v %v", st, err)
	}
	m.update(func(m *llamaMock) { m.models[0].value = "loaded" })
	if st, err := e.Status(ctx); err != nil || st != (Status{Loaded: true}) {
		t.Errorf("status of a loaded preset %+v %v", st, err)
	}
	// a preset the router does not have is an error, not a loaded model
	e2 := NewLlama(Options{ID: "llama/no-such-model", Model: "no-such-model", Base: srv.URL})
	if st, err := e2.Status(ctx); err == nil || st.Loaded {
		t.Errorf("a preset the router does not have: %+v %v", st, err)
	}
	if reqs := m.requests(); len(reqs) != 3 || reqs[0].method != http.MethodGet || reqs[0].path != "/models" {
		t.Errorf("status reads: %q", reqs)
	}
}

func TestLlamaFingerprintFromArgs(t *testing.T) {
	m := newLlamaMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewLlama(Options{ID: "llama/Swift-27B", Model: "Swift-27B", Base: srv.URL})
	ctx := context.Background()

	want := hex6("Swift-27B|/models/swift.gguf -ngl 99 --ctx-size 16384")
	got, err := e.Fingerprint(ctx)
	if err != nil || got != want {
		t.Fatalf("fingerprint %q, want %q, err %v", got, want, err)
	}
	// the args the router starts the preset with change it
	m.update(func(m *llamaMock) {
		m.models[0].args = []string{"/models/swift.gguf", "-ngl", "99", "--ctx-size", "8192"}
	})
	if got2, err := e.Fingerprint(ctx); err != nil || got2 == want {
		t.Errorf("other args: %q %v", got2, err)
	}
	e2 := NewLlama(Options{ID: "llama/no-such-model", Model: "no-such-model", Base: srv.URL})
	if got, err := e2.Fingerprint(ctx); err == nil || got != "" {
		t.Errorf("a preset the router does not have read as a fingerprint: %q %v", got, err)
	}
	// learning it never loaded the model
	if m.statusOf("Swift-27B") != "unloaded" {
		t.Errorf("GET /models loaded the model")
	}
	reqs := m.requests()
	if len(reqs) != 3 {
		t.Fatalf("fingerprint reads: %q", reqs)
	}
	for _, r := range reqs {
		if r.method != http.MethodGet || r.path != "/models" {
			t.Errorf("a fingerprint read used %s %s: /slots, /props and POST /slots load a model", r.method, r.path)
		}
	}
}

func TestLlamaCacheInfo(t *testing.T) {
	m := newLlamaMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewLlama(Options{ID: "llama/Swift-27B", Model: "Swift-27B", Base: srv.URL})

	c, err := e.CacheInfo(context.Background())
	if err != nil || c != (CacheInfo{}) {
		t.Errorf("llama has no RAM cache: %+v %v", c, err)
	}
	if reqs := m.requests(); len(reqs) != 0 {
		t.Errorf("CacheInfo contacted the router: %q", reqs)
	}
}

func TestLlamaSaveRestore(t *testing.T) {
	m := newLlamaMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewLlama(Options{ID: "llama/Swift-27B", Model: "Swift-27B", Base: srv.URL, APIKey: "llama-key"})
	ctx := context.Background()

	if n, b, err := e.Save(ctx, "dc-abc-123.session"); err != nil || n != 123 || b != 1000 {
		t.Errorf("save %d %d %v", n, b, err)
	}
	if n, b, err := e.Restore(ctx, "dc-abc-123.session"); err != nil || n != 123 || b != 1000 {
		t.Errorf("restore %d %d %v", n, b, err)
	}
	reqs := m.requests()
	if len(reqs) != 2 {
		t.Fatalf("requests: %q", reqs)
	}
	for _, r := range reqs {
		if r.origin != "" {
			t.Errorf("Origin header sent: %q", r.origin)
		}
		if r.auth != "Bearer llama-key" {
			t.Errorf("api key: %q", r.auth)
		}
		if r.ctype != "application/json" {
			t.Errorf("content type: %q", r.ctype)
		}
	}
	if reqs[0].path != "/slots/0" || reqs[0].query != "action=save" ||
		reqs[0].body != `{"filename":"dc-abc-123.session","model":"Swift-27B"}` {
		t.Errorf("save call: %s?%s %s", reqs[0].path, reqs[0].query, reqs[0].body)
	}
	if reqs[1].path != "/slots/0" || reqs[1].query != "action=restore" {
		t.Errorf("restore call: %s?%s", reqs[1].path, reqs[1].query)
	}
}

func TestLlamaSlotUnknownSuccessShape(t *testing.T) {
	m := newLlamaMock()
	m.update(func(m *llamaMock) { m.slotBody = `{"saved":5,"slot":0}` })
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewLlama(Options{ID: "llama/Swift-27B", Model: "Swift-27B", Base: srv.URL})

	n, b, err := e.Save(context.Background(), "dc-abc-123.session")
	if err != nil || n != 0 || b != 0 {
		t.Errorf("an unknown success shape is a success: %d %d %v", n, b, err)
	}
	m.update(func(m *llamaMock) { m.slotBody = `{"n_restored":42,"n_read":999999}` })
	if n, b, err := e.Restore(context.Background(), "dc-abc-123.session"); err != nil || n != 42 || b != 999999 {
		t.Errorf("counts %d %d %v", n, b, err)
	}
}

func TestLlamaIsUnloaded(t *testing.T) {
	m := newLlamaMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewLlama(Options{ID: "llama/Swift-27B", Model: "Swift-27B", Base: srv.URL})
	ctx := context.Background()

	m.update(func(m *llamaMock) { m.slotStatus, m.slotErr = http.StatusBadRequest, "model is not running" })
	_, _, err := e.Restore(ctx, "dc-abc-123.session")
	var se *SlotError
	if !e.IsUnloaded(err) {
		t.Errorf("refused by a model that is not running: %v", err)
	}
	if !errors.As(err, &se) || se.Status != http.StatusBadRequest || se.Message != "model is not running" {
		t.Errorf("the refusal carried: %+v", se)
	}
	m.update(func(m *llamaMock) { m.slotStatus, m.slotErr = http.StatusNotFound, "the model is not a preset" })
	if _, _, err := e.Restore(ctx, "dc-abc-123.session"); !e.IsUnloaded(err) {
		t.Errorf("a 404 read as not unloaded: %v", err)
	}
	m.update(func(m *llamaMock) { m.slotStatus, m.slotErr = http.StatusBadRequest, "model is loaded in slot 1" })
	if _, _, err := e.Restore(ctx, "dc-abc-123.session"); e.IsUnloaded(err) {
		t.Errorf("a loaded model read as unloaded: %v", err)
	}
	m.update(func(m *llamaMock) { m.slotStatus, m.slotErr = http.StatusInternalServerError, "boom" })
	if _, _, err := e.Save(ctx, "dc-abc-123.session"); e.IsUnloaded(err) {
		t.Errorf("a 500 read as unloaded: %v", err)
	}
	if e.IsUnloaded(errors.New("something else")) {
		t.Error("an ordinary error read as unloaded")
	}
}

func TestLlamaLoadUnload(t *testing.T) {
	m := newLlamaMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewLlama(Options{ID: "llama/Swift-27B", Model: "Swift-27B", Base: srv.URL, APIKey: "llama-key"})
	ctx := context.Background()

	if err := e.Load(ctx); err != nil {
		t.Errorf("load: %v", err)
	}
	if err := e.Unload(ctx, 30*time.Second); err != nil {
		t.Errorf("unload: %v", err)
	}
	reqs := m.requests()
	if reqs[0].path != "/models/load" || reqs[0].body != `{"model":"Swift-27B"}` {
		t.Errorf("load call: %s %s", reqs[0].path, reqs[0].body)
	}
	if reqs[1].path != "/models" || reqs[1].method != http.MethodGet {
		t.Errorf("Load must read the model list, not trust the load answer: %s %s", reqs[1].method, reqs[1].path)
	}
	if reqs[2].path != "/models/unload" || reqs[2].body != `{"model":"Swift-27B"}` {
		t.Errorf("unload call: %s %s", reqs[2].path, reqs[2].body)
	}
	for _, r := range reqs {
		if r.path == "/unload" || r.path == "/model" {
			t.Errorf("router mode does not have %s", r.path)
		}
		if r.origin != "" || r.auth != "Bearer llama-key" {
			t.Errorf("origin %q auth %q", r.origin, r.auth)
		}
	}
	m.update(func(m *llamaMock) { m.unloadStatus = http.StatusBadRequest })
	if err := e.Unload(ctx, time.Second); err != nil {
		t.Errorf("nothing to unload is not an error: %v", err)
	}
	m.update(func(m *llamaMock) { m.unloadStatus = http.StatusNotFound })
	var se *SlotError
	if err := e.Unload(ctx, time.Second); !errors.As(err, &se) || se.Status != http.StatusNotFound {
		t.Errorf("a preset the router does not have: %v", err)
	}
	m.update(func(m *llamaMock) { m.loadStatus = http.StatusInternalServerError })
	if err := e.Load(ctx); err == nil {
		t.Error("a refused load accepted")
	}
}

func TestLlamaLoadWaitsForTheInstance(t *testing.T) {
	m := newLlamaMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewLlama(Options{ID: "llama/Swift-27B", Model: "Swift-27B", Base: srv.URL, Timeout: 2 * time.Second})
	old := loadPoll
	loadPoll = 10 * time.Millisecond
	defer func() { loadPoll = old }()

	m.update(func(m *llamaMock) { m.loadStays = true }) // the router answers at once; the instance is still starting
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	if err := e.Load(ctx); err == nil || !strings.Contains(err.Error(), "not loaded") {
		t.Errorf("a load that never became resident accepted: %v", err)
	}
	reqs := m.requests()
	if reqs[0].path != "/models/load" {
		t.Errorf("load call: %s", reqs[0].path)
	}
	if n := len(reqs); n < 3 {
		t.Errorf("the model list was read %d times while the instance started", n)
	}
}

func TestLlamaLoadReportsAFailedInstance(t *testing.T) {
	m := newLlamaMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewLlama(Options{ID: "llama/Swift-27B", Model: "Swift-27B", Base: srv.URL})
	old := loadPoll
	loadPoll = 10 * time.Millisecond
	defer func() { loadPoll = old }()

	m.update(func(m *llamaMock) { m.loadStays, m.loadFailed, m.exitCode = true, true, 1 })
	if err := e.Load(context.Background()); err == nil || !strings.Contains(err.Error(), "exit code 1") {
		t.Errorf("a dead instance: %v", err)
	}
}

// A swap back to a model that was never evicted: the router refuses the load as
// already running, which is the load having succeeded. Treating it as a failure
// wedges the queue: the request gets 503 and the card never moves.
func TestLlamaLoadOfARunningModelIsASuccess(t *testing.T) {
	m := newLlamaMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewLlama(Options{ID: "llama/Swift-27B", Model: "Swift-27B", Base: srv.URL})
	m.update(func(m *llamaMock) { m.loadBusy = true })
	if err := e.Load(context.Background()); err != nil {
		t.Errorf("a model already running: %v", err)
	}
}

// The router assigns a new port on every load. A fingerprint that keeps the port
// reads as a new engine after a swap and evicts the route's saved slot files, so
// the next request re-prefills. The port must not be part of the identity.
func TestLlamaFingerprintIgnoresThePort(t *testing.T) {
	m := newLlamaMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewLlama(Options{ID: "llama/Swift-27B", Model: "Swift-27B", Base: srv.URL})
	m.update(func(m *llamaMock) {
		m.models[0].value = "loaded"
		m.models[0].args = []string{"/models/swift.gguf", "--port", "53985", "--ctx-size", "229376"}
	})
	first, err := e.Fingerprint(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	m.update(func(m *llamaMock) {
		m.models[0].args = []string{"/models/swift.gguf", "--port", "54001", "--ctx-size", "229376"}
	})
	second, err := e.Fingerprint(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Errorf("a reload on a new port read as a new engine: %s vs %s", first, second)
	}
	m.update(func(m *llamaMock) {
		m.models[0].args = []string{"/models/swift.gguf", "--port", "54001", "--ctx-size", "131072"}
	})
	if third, err := e.Fingerprint(context.Background()); err != nil || third == first {
		t.Errorf("a context change must change the fingerprint, got %s vs %s (err %v)", third, first, err)
	}
}

func TestLlamaWake(t *testing.T) {
	m := newLlamaMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewLlama(Options{ID: "llama/Swift-27B", Model: "Swift-27B", Base: srv.URL})
	ctx := context.Background()

	if err := e.Wake(ctx); err != nil {
		t.Errorf("wake: %v", err)
	}
	reqs := m.requests()
	last := reqs[len(reqs)-1]
	if last.path != "/v1/chat/completions" {
		t.Errorf("wake path: %s", last.path)
	}
	if !strings.Contains(last.body, `"model":"Swift-27B"`) || !strings.Contains(last.body, `"max_tokens":1`) {
		t.Errorf("wake body: %s", last.body)
	}
	m.update(func(m *llamaMock) { m.wakeStatus = http.StatusServiceUnavailable })
	var se *SlotError
	if err := e.Wake(ctx); !errors.As(err, &se) || se.Status != http.StatusServiceUnavailable {
		t.Errorf("wake on a 503: %v", err)
	}
}

func TestLlamaVramFree(t *testing.T) {
	dir := t.TempDir()
	old := SysfsVRAM
	SysfsVRAM = dir
	t.Cleanup(func() { SysfsVRAM = old })
	if err := os.WriteFile(filepath.Join(dir, "mem_info_vram_total"), []byte("24576 MiB"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "mem_info_vram_used"), []byte("8192 MiB"), 0o600); err != nil {
		t.Fatal(err)
	}
	e := NewLlama(Options{ID: "llama/Swift-27B", Model: "Swift-27B", Base: "http://127.0.0.1:1"})

	n, err := e.VramFree(context.Background())
	if err != nil || n != 16384<<20 {
		t.Errorf("VramFree %d %v", n, err)
	}
	NoteFree(1234 << 20)
	SysfsVRAM = filepath.Join(dir, "not-a-card")
	n, err = e.VramFree(context.Background())
	if err != nil || n != 1234<<20 {
		t.Errorf("the fallback to the last figure: %d %v", n, err)
	}
}
