package engine

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Strata is a Strata server and the one model behind it: its own load, unload
// and slot dialect, so a request carries no model name.
type Strata struct {
	cl client
}

// NewStrata returns an Engine for the Strata server at o.Base, e.g.
// http://127.0.0.1:8095.
func NewStrata(o Options) *Strata { return &Strata{cl: newClient(o)} }

// ID is the route id, e.g. "strata/flash-next".
func (s *Strata) ID() string { return s.cl.id }

// Model is the model the server runs.
func (s *Strata) Model() string { return s.cl.model }

// Status reads /v1/status; it never loads a model.
func (s *Strata) Status(ctx context.Context) (Status, error) {
	var st struct {
		Started     int64 `json:"started"`
		Loaded      bool  `json:"loaded"`
		LastTimings *struct {
			CacheN     int `json:"cache_n"`
			PromptN    int `json:"prompt_n"`
			PredictedN int `json:"predicted_n"`
		} `json:"last_timings"`
	}
	if err := s.cl.getJSON(ctx, "/v1/status", &st); err != nil {
		return Status{}, err
	}
	out := Status{Started: st.Started, Loaded: st.Loaded}
	if t := st.LastTimings; t != nil {
		out.LiveTokens = t.CacheN + t.PromptN + t.PredictedN
	}
	return out, nil
}

// Fingerprint is the identity of the engine Strata has loaded, from JSON
// /metrics. An unloaded engine reports no context and no KV type: that is not a
// fingerprint, so it is an error, and no model is loaded to learn one.
func (s *Strata) Fingerprint(ctx context.Context) (string, error) {
	m, err := s.metrics(ctx)
	if err != nil {
		return "", err
	}
	if err := m.complete(); err != nil {
		return "", err
	}
	e := m.Engine
	sum := sha256.Sum256(fmt.Appendf(nil, "%s|%v|%v|%v|%v", e.Model, e.Version, e.Context, e.KV, e.CVec))
	return hex.EncodeToString(sum[:6]), nil
}

// CacheInfo is Strata's in-RAM conversation cache: what it has parked, its
// budget and its park floors.
func (s *Strata) CacheInfo(ctx context.Context) (CacheInfo, error) {
	m, err := s.metrics(ctx)
	if err != nil {
		return CacheInfo{}, err
	}
	if err := m.complete(); err != nil {
		return CacheInfo{}, err
	}
	return CacheInfo{Parked: *m.CC.Parked, ParkedBytes: m.CC.Bytes, BudgetMiB: m.CC.BudgetMiB,
		MinFreeMiB: m.Engine.MinFreeMiB, MinTokens: m.Engine.MinTokens}, nil
}

// VramFree is the card as Strata reports it; the figure is recorded, so a sysfs
// read that fails later falls back to it.
func (s *Strata) VramFree(ctx context.Context) (int64, error) {
	m, err := s.metrics(ctx)
	if err != nil {
		return 0, err
	}
	free := m.Engine.VramFreeMiB << 20
	NoteFree(free)
	return free, nil
}

// Save writes the live conversation to name in Strata's slot_save_path.
func (s *Strata) Save(ctx context.Context, name string) (int, int64, error) {
	return s.slot(ctx, "save", name)
}

// Restore makes name the live conversation; Strata drops the live one without
// parking it.
func (s *Strata) Restore(ctx context.Context, name string) (int, int64, error) {
	return s.slot(ctx, "restore", name)
}

func (s *Strata) slot(ctx context.Context, action, name string) (int, int64, error) {
	raw, err := s.cl.slot(ctx, "/slots/0?action="+action, name)
	if err != nil {
		return 0, 0, err
	}
	tokens, size, err := counts(raw)
	if err != nil {
		return 0, 0, fmt.Errorf("engine: strata %s answer: %w", action, err)
	}
	return tokens, size, nil
}

// Load makes the model resident. A 503 carrying a Retry-After is the GPU being
// busy: the model does not fit now, so Load reports that.
func (s *Strata) Load(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, s.cl.loadWindow())
	defer cancel()
	r, err := s.cl.do(ctx, http.MethodPost, "/v1/load", emptyBody)
	if err != nil {
		return err
	}
	if r.status == http.StatusOK {
		return nil
	}
	_, msg := r.refusal()
	if r.status == http.StatusServiceUnavailable {
		return &SlotError{Status: r.status,
			Message: fmt.Sprintf("GPU busy, retry after %s: %s", r.header.Get("Retry-After"), msg)}
	}
	return &SlotError{Status: r.status, Message: msg}
}

// Unload gives the VRAM back and keeps the server process alive. ?wait= is how
// long Strata drains a running turn, so 409 busy is the turn not draining and
// 503 a stuck engine. A 200 answering "not loaded" is fine: there was nothing
// to give back.
func (s *Strata) Unload(ctx context.Context, wait time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, wait+15*time.Second)
	defer cancel()
	r, err := s.cl.do(ctx, http.MethodPost, "/v1/unload?wait="+strconv.Itoa(int(wait.Seconds())), emptyBody)
	if err != nil {
		return err
	}
	if r.status == http.StatusOK {
		return nil
	}
	_, msg := r.refusal()
	switch r.status {
	case http.StatusConflict:
		return &SlotError{Status: r.status, Message: fmt.Sprintf("%s: the turn did not drain within %s", msg, wait)}
	case http.StatusServiceUnavailable:
		return &SlotError{Status: r.status, Message: "engine is stuck: the unload did not finish"}
	}
	return &SlotError{Status: r.status, Message: msg}
}

// Wake sends one 1-token request to trigger Strata's lazy load.
func (s *Strata) Wake(ctx context.Context) error {
	return s.cl.wake(ctx, []byte(`{"messages":[{"role":"user","content":"."}],"max_tokens":1,"stream":false}`))
}

// IsUnloaded reports whether Strata refused a slot call because its engine is
// not resident: a 503 saying "not loaded".
func (s *Strata) IsUnloaded(err error) bool {
	var se *SlotError
	return errors.As(err, &se) && se.Status == http.StatusServiceUnavailable && ContainsUnloaded(se.Message)
}

// metricsAnswer is the part of Strata's JSON /metrics the arbiter reads.
type metricsAnswer struct {
	CC struct {
		Parked    *int  `json:"parked"`
		Bytes     int64 `json:"bytes"`
		BudgetMiB int64 `json:"budget_mib"`
	} `json:"conversation_cache"`
	Engine struct {
		Model            string `json:"model"`
		Version, Context any
		KV               any   `json:"kv"`
		CVec             any   `json:"cvec"`
		VramFreeMiB      int64 `json:"vram_free_mib"`
		MinFreeMiB       int64 `json:"conversation_cache_min_free_mib"`
		MinTokens        int   `json:"conversation_cache_min_tokens"`
	} `json:"engine"`
}

func (s *Strata) metrics(ctx context.Context) (*metricsAnswer, error) {
	var m metricsAnswer
	if err := s.cl.getJSON(ctx, "/metrics", &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// complete reports an engine Strata has not loaded: it has no RAM cache, no
// context and no KV type, so its /metrics is not an identity.
func (m *metricsAnswer) complete() error {
	if m.CC.Parked == nil || m.Engine.Model == "" || m.Engine.Context == nil || m.Engine.KV == nil {
		return errors.New("engine: strata engine unloaded, /metrics incomplete")
	}
	return nil
}

// emptyBody is the body /v1/load and /v1/unload take.
var emptyBody = []byte(`{}`)

// client is the HTTP plumbing the two dialects share: one route, one base URL,
// one bearer token.
type client struct {
	id, model, base, apiKey string
	timeout, loadTimeout    time.Duration
	http                    http.Client
}

// newClient defaults the slot timeout to 15 minutes: a 200K-token session file
// is several GB.
func newClient(o Options) client {
	if o.Timeout <= 0 {
		o.Timeout = 15 * time.Minute
	}
	return client{id: o.ID, model: o.Model, base: strings.TrimRight(o.Base, "/"), apiKey: o.APIKey,
		timeout: o.Timeout, loadTimeout: o.LoadTimeout}
}

// loadWindow is how long a load may take: the load timeout when one was set,
// else the slot timeout.
func (c *client) loadWindow() time.Duration {
	if c.loadTimeout > 0 && c.loadTimeout < c.timeout {
		return c.loadTimeout
	}
	return c.timeout
}

// resp is one answer from a server.
type resp struct {
	status int
	header http.Header
	body   []byte
}

// refusal is the message a refusal carries: Strata's {"error":{"kind","message"}},
// llama.cpp's {"error":"text"}, and a load or unload answer's {"status":"busy"}.
func (r *resp) refusal() (kind, message string) {
	var v struct {
		Error  json.RawMessage `json:"error"`
		Status string          `json:"status"`
	}
	if json.Unmarshal(r.body, &v) != nil {
		return "", strings.TrimSpace(string(r.body))
	}
	if len(v.Error) > 0 {
		var obj struct{ Kind, Message string }
		if json.Unmarshal(v.Error, &obj) == nil {
			return obj.Kind, obj.Message
		}
		var s string
		if json.Unmarshal(v.Error, &s) == nil {
			return "", s
		}
	}
	return "", v.Status
}

// do sends one request: the bearer token when the server wants one, JSON for a
// POST body, and no Origin header — Strata reads a missing Origin as a
// non-browser client and accepts the call.
func (c *client) do(ctx context.Context, method, path string, body []byte) (*resp, error) {
	var r io.Reader
	if body != nil {
		r = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.base+path, r)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	res, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = res.Body.Close() }()
	raw, err := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	return &resp{status: res.StatusCode, header: res.Header, body: raw}, nil
}

// getJSON reads a GET answer; a status or fingerprint read is 3 seconds.
func (c *client) getJSON(ctx context.Context, path string, v any) error {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+path, nil)
	if err != nil {
		return err
	}
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	req.Header.Set("Accept", "application/json")
	res, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = res.Body.Close() }()
	if res.StatusCode != http.StatusOK {
		return fmt.Errorf("engine: GET %s: HTTP %d", path, res.StatusCode)
	}
	return json.NewDecoder(res.Body).Decode(v)
}

// postJSON sends one POST and answers a refusal; the caller can allow one status.
func (c *client) postJSON(ctx context.Context, path string, body []byte) (*resp, error) {
	r, err := c.do(ctx, http.MethodPost, path, body)
	if err != nil {
		return nil, err
	}
	if r.status == http.StatusOK {
		return r, nil
	}
	_, msg := r.refusal()
	return r, &SlotError{Status: r.status, Message: msg}
}

// slot POSTs one /slots/0 call with {"filename":name} and returns its answer.
// slot sends one slot action with the file name in the body.
func (c *client) slot(ctx context.Context, path, name string) ([]byte, error) {
	return c.slotBody(ctx, path, map[string]string{"filename": name})
}

// slotBody POSTs one slot action with the body the server asks for and returns
// its answer. A slot for a model that is not resident is a refusal, not a 200.
func (c *client) slotBody(ctx context.Context, path string, body map[string]string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	raw, _ := json.Marshal(body)
	r, err := c.do(ctx, http.MethodPost, path, raw)
	if err != nil {
		return nil, err
	}
	if r.status != http.StatusOK {
		kind, msg := r.refusal()
		return nil, &SlotError{Status: r.status, Kind: kind, Message: msg}
	}
	return r.body, nil
}

// wake sends one 1-token chat request; 5 minutes covers the lazy load it triggers.
func (c *client) wake(ctx context.Context, body []byte) error {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()
	r, err := c.do(ctx, http.MethodPost, "/v1/chat/completions", body)
	if err != nil {
		return err
	}
	if r.status == http.StatusOK {
		return nil
	}
	_, msg := r.refusal()
	return &SlotError{Status: r.status, Message: msg}
}

// counts is the n_saved/n_written/n_restored/n_read a slot call answers with.
func counts(raw []byte) (int, int64, error) {
	var ok struct {
		Saved    int   `json:"n_saved"`
		Written  int64 `json:"n_written"`
		Restored int   `json:"n_restored"`
		Read     int64 `json:"n_read"`
	}
	if err := json.Unmarshal(raw, &ok); err != nil {
		return 0, 0, err
	}
	return ok.Saved + ok.Restored, ok.Written + ok.Read, nil
}
