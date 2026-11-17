package engine

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
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
var _ Engine = (*Strata)(nil)

// strataMock is a fake Strata server: the answers the test sets and every
// request it saw. It is an httptest server, so nothing outside the test is
// contacted.
type strataMock struct {
	mu sync.Mutex

	started   int64
	loaded    bool
	timings   int // cache_n + prompt_n + predicted_n
	model     string
	version   string
	context   any // nil: the engine is unloaded
	kv        any
	cvec      any
	parked    *int // nil: /metrics has no parked count
	vramMiB   int64
	budget    int64
	minFree   int64
	minTokens int

	slotStatus                           int    // non-zero: a slot call answers this status
	slotKind, slotMsg                    string // with this kind and message
	slotTokens                           int    // the count a success reports
	slotDelay                            time.Duration
	loadStatus, unloadStatus, wakeStatus int
	retryAfter                           string
	loadBody, unloadBody                 string // answers other than the default

	reqs []strataRec
}

type strataRec struct {
	method, path, query, body, origin, auth, ctype string
}

func newStrataMock() *strataMock {
	parked := 2
	return &strataMock{started: 1000, loaded: true, timings: 16000, model: "flash-next", version: "0.1.45",
		context: 262144, kv: "k8v4", parked: &parked, vramMiB: 4096, budget: 3584, minFree: 2048,
		minTokens: 15000, slotTokens: 123}
}

func (m *strataMock) update(fn func(m *strataMock)) {
	m.mu.Lock()
	fn(m)
	m.mu.Unlock()
}

func (m *strataMock) requests() []strataRec {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]strataRec(nil), m.reqs...)
}

func (m *strataMock) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	m.mu.Lock()
	defer m.mu.Unlock()
	raw, _ := io.ReadAll(r.Body)
	m.reqs = append(m.reqs, strataRec{method: r.Method, path: r.URL.Path, query: r.URL.RawQuery, body: string(raw),
		origin: r.Header.Get("Origin"), auth: r.Header.Get("Authorization"), ctype: r.Header.Get("Content-Type")})
	w.Header().Set("Content-Type", "application/json")
	switch {
	case r.Method == http.MethodGet && r.URL.Path == "/v1/status":
		fmt.Fprintf(w, `{"started":%d,"loaded":%t,"last_timings":{"cache_n":%d,"prompt_n":0,"predicted_n":0}}`,
			m.started, m.loaded, m.timings)
	case r.Method == http.MethodGet && r.URL.Path == "/metrics":
		_, _ = io.WriteString(w, m.metrics())
	case r.Method == http.MethodPost && r.URL.Path == "/slots/0":
		m.slot(w, r)
	case r.Method == http.MethodPost && r.URL.Path == "/v1/load":
		if m.loadStatus != 0 {
			if m.retryAfter != "" {
				w.Header().Set("Retry-After", m.retryAfter)
			}
			w.WriteHeader(m.loadStatus)
			_, _ = io.WriteString(w, m.answer(m.loadBody, `{"error":{"message":"model does not fit in VRAM"}}`))
			return
		}
		_, _ = io.WriteString(w, `{"status":"loaded"}`)
	case r.Method == http.MethodPost && r.URL.Path == "/v1/unload":
		if m.unloadStatus != 0 {
			w.WriteHeader(m.unloadStatus)
			def := `{"status":"busy"}`
			if m.unloadStatus == http.StatusServiceUnavailable {
				def = `{"error":{"message":"engine is stuck"}}`
			}
			_, _ = io.WriteString(w, m.answer(m.unloadBody, def))
			return
		}
		_, _ = io.WriteString(w, `{"status":"unloaded"}`)
	case r.Method == http.MethodPost && r.URL.Path == "/v1/chat/completions":
		if m.wakeStatus != 0 {
			w.WriteHeader(m.wakeStatus)
			_, _ = io.WriteString(w, `{"error":{"message":"refused"}}`)
			return
		}
		_, _ = io.WriteString(w, `{"choices":[{"finish_reason":"length"}]}`)
	default:
		w.WriteHeader(http.StatusNotFound)
		_, _ = io.WriteString(w, `{"error":{"message":"no endpoint"}}`)
	}
}

func (m *strataMock) answer(body, def string) string {
	if body != "" {
		return body
	}
	return def
}

func (m *strataMock) metrics() string {
	var v struct {
		CC struct {
			Parked    *int  `json:"parked"`
			Bytes     int64 `json:"bytes"`
			BudgetMiB int64 `json:"budget_mib"`
		} `json:"conversation_cache"`
		Engine struct {
			Model       string `json:"model"`
			Version     any    `json:"version"`
			Context     any    `json:"context"`
			KV          any    `json:"kv"`
			CVec        any    `json:"cvec"`
			VramFreeMiB int64  `json:"vram_free_mib"`
			MinFreeMiB  int64  `json:"conversation_cache_min_free_mib"`
			MinTokens   int    `json:"conversation_cache_min_tokens"`
		} `json:"engine"`
	}
	v.CC.Parked, v.CC.Bytes, v.CC.BudgetMiB = m.parked, 4096, m.budget
	v.Engine.Model, v.Engine.Version, v.Engine.Context, v.Engine.KV, v.Engine.CVec =
		m.model, m.version, m.context, m.kv, m.cvec
	v.Engine.VramFreeMiB, v.Engine.MinFreeMiB, v.Engine.MinTokens = m.vramMiB, m.minFree, m.minTokens
	b, _ := json.Marshal(v)
	return string(b)
}

func (m *strataMock) slot(w http.ResponseWriter, r *http.Request) {
	time.Sleep(m.slotDelay)
	if m.slotStatus != 0 {
		msg := m.slotMsg
		if msg == "" {
			msg = "refused"
		}
		w.WriteHeader(m.slotStatus)
		fmt.Fprintf(w, `{"error":{"kind":%q,"message":%q}}`, m.slotKind, msg)
		return
	}
	if r.URL.Query().Get("action") == "save" {
		fmt.Fprintf(w, `{"n_saved":%d,"n_written":1000}`, m.slotTokens)
		return
	}
	fmt.Fprintf(w, `{"n_restored":%d,"n_read":1000}`, m.slotTokens)
}

func TestStrataStatus(t *testing.T) {
	m := newStrataMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewStrata(Options{ID: "strata/flash-next", Model: "flash-next", Base: srv.URL, APIKey: "strata-key"})
	ctx := context.Background()

	if e.ID() != "strata/flash-next" || e.Model() != "flash-next" {
		t.Errorf("ID %q, Model %q", e.ID(), e.Model())
	}
	st, err := e.Status(ctx)
	if err != nil || st != (Status{Started: 1000, Loaded: true, LiveTokens: 16000}) {
		t.Fatalf("status %+v %v", st, err)
	}
	if reqs := m.requests(); len(reqs) != 1 || reqs[0].auth != "Bearer strata-key" || reqs[0].origin != "" {
		t.Errorf("a status read: %q", reqs)
	}
	m.update(func(m *strataMock) { m.loaded, m.timings = false, 0 })
	if st, err := e.Status(ctx); err != nil || st.Loaded || st.LiveTokens != 0 {
		t.Errorf("unloaded status %+v %v", st, err)
	}
	if reqs := m.requests(); len(reqs) != 2 || reqs[0].method != http.MethodGet {
		t.Errorf("status calls: %q", reqs)
	}
}

func TestStrataFingerprint(t *testing.T) {
	m := newStrataMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewStrata(Options{ID: "strata/flash-next", Model: "flash-next", Base: srv.URL})
	ctx := context.Background()

	want := hex6("flash-next|0.1.45|262144|k8v4|<nil>")
	got, err := e.Fingerprint(ctx)
	if err != nil || got != want {
		t.Fatalf("fingerprint %q, want %q, err %v", got, want, err)
	}
	m.update(func(m *strataMock) { m.model = "other-model" })
	if got2, err := e.Fingerprint(ctx); err != nil || got2 == want {
		t.Errorf("another model: %q %v", got2, err)
	}
	m.update(func(m *strataMock) { m.model, m.context, m.kv = "flash-next", nil, nil })
	if got, err := e.Fingerprint(ctx); err == nil || got != "" {
		t.Errorf("an unloaded engine read as a fingerprint: %q %v", got, err)
	}
	m.update(func(m *strataMock) { m.context, m.kv, m.parked = 262144, "k8v4", nil })
	if got, err := e.Fingerprint(ctx); err == nil || got != "" {
		t.Errorf("no parked count read as a fingerprint: %q %v", got, err)
	}
	parked := 0
	m.update(func(m *strataMock) { m.parked, m.model = &parked, "" })
	if got, err := e.Fingerprint(ctx); err == nil || got != "" {
		t.Errorf("an empty model name read as a fingerprint: %q %v", got, err)
	}
}

func TestStrataCacheInfo(t *testing.T) {
	m := newStrataMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewStrata(Options{ID: "strata/flash-next", Model: "flash-next", Base: srv.URL})

	c, err := e.CacheInfo(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if c != (CacheInfo{Parked: 2, ParkedBytes: 4096, BudgetMiB: 3584, MinFreeMiB: 2048, MinTokens: 15000}) {
		t.Errorf("cache %+v", c)
	}
	m.update(func(m *strataMock) { m.kv = nil })
	if c, err := e.CacheInfo(context.Background()); err == nil || c != (CacheInfo{}) {
		t.Errorf("cache of an unloaded engine: %+v %v", c, err)
	}
}

func TestStrataSaveRestore(t *testing.T) {
	m := newStrataMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewStrata(Options{ID: "strata/flash-next", Model: "flash-next", Base: srv.URL, APIKey: "strata-key"})
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
		if r.auth != "Bearer strata-key" {
			t.Errorf("api key: %q", r.auth)
		}
		if r.ctype != "application/json" {
			t.Errorf("content type: %q", r.ctype)
		}
	}
	if reqs[0].path != "/slots/0" || reqs[0].query != "action=save" || reqs[0].body != `{"filename":"dc-abc-123.session"}` {
		t.Errorf("save call: %s?%s %s", reqs[0].path, reqs[0].query, reqs[0].body)
	}
	if reqs[1].query != "action=restore" {
		t.Errorf("restore call: %s?%s", reqs[1].path, reqs[1].query)
	}
}

func TestStrataSlotTimeout(t *testing.T) {
	m := newStrataMock()
	m.update(func(m *strataMock) { m.slotDelay = 300 * time.Millisecond })
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewStrata(Options{ID: "strata/flash-next", Model: "flash-next", Base: srv.URL, Timeout: 20 * time.Millisecond})

	_, _, err := e.Save(context.Background(), "dc-abc-123.session")
	if err == nil || !strings.Contains(err.Error(), "deadline") {
		t.Errorf("a slot call outliving Options.Timeout: %v", err)
	}
}

func TestStrataSlotRefusal(t *testing.T) {
	m := newStrataMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewStrata(Options{ID: "strata/flash-next", Model: "flash-next", Base: srv.URL})
	ctx := context.Background()

	m.update(func(m *strataMock) { m.slotStatus, m.slotMsg = http.StatusServiceUnavailable, "engine is not loaded" })
	_, _, err := e.Restore(ctx, "dc-abc-123.session")
	var se *SlotError
	if !e.IsUnloaded(err) {
		t.Errorf("refused by an unloaded engine: %v", err)
	}
	if !errors.As(err, &se) || se.Status != http.StatusServiceUnavailable || se.Message != "engine is not loaded" {
		t.Errorf("the refusal carried: %+v", se)
	}
	m.update(func(m *strataMock) {
		m.slotStatus, m.slotKind, m.slotMsg = http.StatusServiceUnavailable, "", "refused"
	})
	if _, _, err := e.Restore(ctx, "dc-abc-123.session"); e.IsUnloaded(err) {
		t.Errorf("a plain 503 read as unloaded: %v", err)
	}
	m.update(func(m *strataMock) {
		m.slotStatus, m.slotKind, m.slotMsg = http.StatusInsufficientStorage, "storage", "no space"
	})
	_, _, err = e.Save(ctx, "dc-abc-123.session")
	if e.IsUnloaded(err) {
		t.Errorf("a 507 read as unloaded: %v", err)
	}
	if !errors.As(err, &se) || se.Status != 507 || se.Kind != "storage" || se.Message != "no space" {
		t.Errorf("the storage refusal carried: %+v", se)
	}
	if e.IsUnloaded(errors.New("something else")) {
		t.Error("an ordinary error read as unloaded")
	}
}

func TestStrataLoad(t *testing.T) {
	m := newStrataMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewStrata(Options{ID: "strata/flash-next", Model: "flash-next", Base: srv.URL, APIKey: "strata-key"})
	ctx := context.Background()

	if err := e.Load(ctx); err != nil {
		t.Errorf("load: %v", err)
	}
	m.update(func(m *strataMock) { m.loadStatus, m.retryAfter = http.StatusServiceUnavailable, "30" })
	err := e.Load(ctx)
	if err == nil || !strings.Contains(err.Error(), "retry after 30") {
		t.Errorf("a GPU that cannot fit the model: %v", err)
	}
	reqs := m.requests()
	last := reqs[len(reqs)-1]
	if last.path != "/v1/load" || last.body != "{}" || last.auth != "Bearer strata-key" || last.origin != "" {
		t.Errorf("load call: %s %s auth %q origin %q", last.path, last.body, last.auth, last.origin)
	}
	m.update(func(m *strataMock) {
		m.loadStatus, m.retryAfter, m.loadBody = http.StatusBadRequest, "", `{"error":{"message":"bad args"}}`
	})
	var se *SlotError
	if err := e.Load(ctx); !errors.As(err, &se) || se.Status != http.StatusBadRequest || se.Message != "bad args" {
		t.Errorf("a refused load: %v", err)
	}
}

func TestStrataUnload(t *testing.T) {
	m := newStrataMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewStrata(Options{ID: "strata/flash-next", Model: "flash-next", Base: srv.URL})
	ctx := context.Background()

	if err := e.Unload(ctx, 30*time.Second); err != nil {
		t.Errorf("unload: %v", err)
	}
	reqs := m.requests()
	last := reqs[len(reqs)-1]
	if last.path != "/v1/unload" || last.query != "wait=30" || last.body != "{}" {
		t.Errorf("unload call: %s?%s %s", last.path, last.query, last.body)
	}
	m.update(func(m *strataMock) { m.unloadBody = `{"status":"not loaded"}` })
	if err := e.Unload(ctx, 5*time.Second); err != nil {
		t.Errorf("an engine that was not loaded: %v", err)
	}
	if reqs := m.requests(); reqs[len(reqs)-1].query != "wait=5" {
		t.Errorf("wait: %s", reqs[len(reqs)-1].query)
	}
	m.update(func(m *strataMock) { m.unloadStatus, m.unloadBody = http.StatusConflict, "" })
	err := e.Unload(ctx, 2*time.Second)
	var se *SlotError
	if err == nil || !errors.As(err, &se) || se.Status != http.StatusConflict || !strings.Contains(err.Error(), "busy") {
		t.Errorf("a turn that did not drain: %v", err)
	}
	m.update(func(m *strataMock) { m.unloadStatus = http.StatusServiceUnavailable })
	if err := e.Unload(ctx, time.Second); err == nil {
		t.Error("a stuck engine accepted")
	}
}

func TestStrataWake(t *testing.T) {
	m := newStrataMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewStrata(Options{ID: "strata/flash-next", Model: "flash-next", Base: srv.URL})
	ctx := context.Background()

	if err := e.Wake(ctx); err != nil {
		t.Errorf("wake: %v", err)
	}
	reqs := m.requests()
	last := reqs[len(reqs)-1]
	if last.path != "/v1/chat/completions" {
		t.Errorf("wake path: %s", last.path)
	}
	if last.body != `{"messages":[{"role":"user","content":"."}],"max_tokens":1,"stream":false}` {
		t.Errorf("wake body: %s", last.body)
	}
	m.update(func(m *strataMock) { m.wakeStatus = http.StatusServiceUnavailable })
	var se *SlotError
	if err := e.Wake(ctx); !errors.As(err, &se) || se.Status != http.StatusServiceUnavailable {
		t.Errorf("wake on a 503: %v", err)
	}
}

func TestStrataVramFree(t *testing.T) {
	dir := t.TempDir()
	old := SysfsVRAM
	SysfsVRAM = dir
	t.Cleanup(func() { SysfsVRAM = old })
	if err := os.WriteFile(filepath.Join(dir, "mem_info_vram_total"), []byte("24576 MiB\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "mem_info_vram_used"), []byte("8192 MiB"), 0o600); err != nil {
		t.Fatal(err)
	}
	m := newStrataMock()
	srv := httptest.NewServer(m)
	defer srv.Close()
	e := NewStrata(Options{ID: "strata/flash-next", Model: "flash-next", Base: srv.URL})

	n, err := e.VramFree(context.Background())
	if err != nil || n != 4096<<20 {
		t.Errorf("VramFree %d %v", n, err)
	}
	if LastFree() != 4096<<20 {
		t.Errorf("NoteFree recorded %d", LastFree())
	}
	if n, err := FreeVRAM(); err != nil || n != 16384<<20 {
		t.Errorf("sysfs %d %v", n, err)
	}
}

func hex6(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:6])
}
