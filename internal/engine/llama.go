package engine

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Llama is a llama.cpp b12445 router running several presets; this Engine is the
// one o.Model names. Every call carries the model name, and the router loads a
// model on demand.
type Llama struct {
	cl client
}

// NewLlama returns an Engine for the llama.cpp router at o.Base, e.g.
// http://127.0.0.1:8001, for the model o.Model.
func NewLlama(o Options) *Llama { return &Llama{cl: newClient(o)} }

// ID is the route id, e.g. "llama/Swift-27B".
func (l *Llama) ID() string { return l.cl.id }

// Model is the model name a client puts in its request.
func (l *Llama) Model() string { return l.cl.model }

// modelEntry is one entry of the router's GET /models answer.
type modelEntry struct {
	ID     string `json:"id"`
	Status struct {
		Value string `json:"value"` // "loaded", "unloaded"
	} `json:"status"`
	Args []string `json:"args"` // the router's own command line for this preset
}

// Status reads GET /models: it is the only endpoint that reports a model without
// loading it. The router gives no start time and no live token count.
func (l *Llama) Status(ctx context.Context) (Status, error) {
	e, err := l.entry(ctx)
	if err != nil {
		return Status{}, err
	}
	return Status{Loaded: e.Status.Value == "loaded"}, nil
}

// Fingerprint is the model name and the args the router starts it with. GET
// /models never loads a model; /slots?model=, GET and POST /props and
// POST /slots do, so none of them is used here.
func (l *Llama) Fingerprint(ctx context.Context) (string, error) {
	e, err := l.entry(ctx)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256([]byte(e.ID + "|" + strings.Join(e.Args, " ")))
	return hex.EncodeToString(sum[:6]), nil
}

// CacheInfo is zeros: llama.cpp's router has no in-RAM conversation cache.
func (l *Llama) CacheInfo(context.Context) (CacheInfo, error) { return CacheInfo{}, nil }

// VramFree is the shared card from sysfs; the last figure an engine reported is
// the fallback when sysfs cannot be read.
func (l *Llama) VramFree(context.Context) (int64, error) {
	if n, err := FreeVRAM(); err == nil {
		NoteFree(n)
		return n, nil
	}
	return LastFree(), nil
}

// Save writes the live conversation of this model to name in the router's
// session folder.
func (l *Llama) Save(ctx context.Context, name string) (int, int64, error) {
	return l.slot(ctx, "save", name)
}

// Restore makes name the live conversation. The router's action is "load", not
// "restore".
func (l *Llama) Restore(ctx context.Context, name string) (int, int64, error) {
	return l.slot(ctx, "load", name)
}

func (l *Llama) slot(ctx context.Context, action, name string) (int, int64, error) {
	raw, err := l.cl.slot(ctx, "/slots/0?action="+action+"&model="+url.QueryEscape(l.cl.model), name)
	if err != nil {
		return 0, 0, err
	}
	tokens, size, _ := counts(raw) // the success shape varies with the build; a 200 is a success
	return tokens, size, nil
}

// Load loads this model into a free slot.
func (l *Llama) Load(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, l.cl.timeout)
	defer cancel()
	_, err := l.cl.postJSON(ctx, "/models/load", modelBody(l.cl.model))
	return err
}

// Unload gives this model's VRAM back. The router has no drain wait, so wait
// only bounds the call. "model is not running" is not an error: there was
// nothing to unload. A 404 is: the model is not a preset.
func (l *Llama) Unload(ctx context.Context, wait time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, wait+15*time.Second)
	defer cancel()
	_, err := l.cl.postJSON(ctx, "/models/unload", modelBody(l.cl.model))
	var se *SlotError
	if errors.As(err, &se) && se.Status == http.StatusBadRequest &&
		strings.Contains(strings.ToLower(se.Message), "model is not running") {
		return nil
	}
	return err
}

// Wake sends one 1-token request for this model to trigger its lazy load.
func (l *Llama) Wake(ctx context.Context) error {
	b, _ := json.Marshal(struct {
		Model     string    `json:"model"`
		Messages  []message `json:"messages"`
		MaxTokens int       `json:"max_tokens"`
		Stream    bool      `json:"stream"`
	}{l.cl.model, []message{{Role: "user", Content: "."}}, 1, false})
	return l.cl.wake(ctx, b)
}

// message is one chat turn in a Wake request.
type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// IsUnloaded reports whether the router refused a slot call because the model is
// not resident: a 400 saying "model is not running", or a 404 for a model the
// router does not know.
func (l *Llama) IsUnloaded(err error) bool {
	var se *SlotError
	if !errors.As(err, &se) {
		return false
	}
	return se.Status == http.StatusNotFound ||
		se.Status == http.StatusBadRequest && strings.Contains(strings.ToLower(se.Message), "model is not running")
}

// entry is this model's entry in the router's GET /models answer.
func (l *Llama) entry(ctx context.Context) (*modelEntry, error) {
	var m struct {
		Data []modelEntry `json:"data"`
	}
	if err := l.cl.getJSON(ctx, "/models", &m); err != nil {
		return nil, err
	}
	for i := range m.Data {
		if m.Data[i].ID == l.cl.model {
			return &m.Data[i], nil
		}
	}
	return nil, fmt.Errorf("engine: %s is not in the router's model list", l.cl.model)
}

// modelBody is the router's {"model":NAME} request body.
func modelBody(model string) []byte {
	b, _ := json.Marshal(map[string]string{"model": model})
	return b
}
