// Package engine is one resident model identity behind the arbiter: a
// (backend, model) pair with the endpoints the arbiter needs to yield and take
// VRAM, and to save/restore a conversation. Two dialects: Strata's own server
// (POST /v1/unload, /v1/load, POST /slots/0?action=save|restore) and llama.cpp's
// router (POST /models/unload, /models/load, POST /slots/0?action=save|load&model=).
package engine

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

// SlotError is a backend refusing or failing a slot, load or unload call.
type SlotError struct {
	Status  int
	Kind    string // the backend's own kind, may be empty
	Message string
}

func (e *SlotError) Error() string {
	return fmt.Sprintf("engine: HTTP %d %s: %s", e.Status, e.Kind, e.Message)
}

// ErrorKind names an error for metrics and logs: not_found, the backend's kind,
// http_<status>, canceled or transport.
func ErrorKind(err error) string {
	var se *SlotError
	switch {
	case err == nil:
		return ""
	case errors.As(err, &se) && se.Status == 404:
		return "not_found"
	case errors.As(err, &se) && se.Kind != "":
		return se.Kind
	case errors.As(err, &se):
		return fmt.Sprintf("http_%d", se.Status)
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return "canceled"
	default:
		return "transport"
	}
}

// UnloadedMessage is what both dialects say when the model is not resident;
// adapters report it in SlotError.Message.
const UnloadedMessage = "not loaded"

// ContainsUnloaded reports whether a message says the model is not resident.
func ContainsUnloaded(s string) bool { return strings.Contains(strings.ToLower(s), UnloadedMessage) }

// Status is what a backend reports between requests.
type Status struct {
	Started    int64 // when the server started (unix seconds), 0 unknown
	Loaded     bool  // the model is resident
	LiveTokens int   // tokens the engine holds for the last request's conversation
}

// CacheInfo is the engine's in-RAM conversation cache: what it has parked, what
// it can park, and the RAM floor it applies. Fingerprint is the same 12-hex
// identity Fingerprint() returns, read in the same call, so a switch costs one
// request and not two.
type CacheInfo struct {
	Fingerprint string // "", when the engine cannot say what it is
	Parked      int    // conversations parked in RAM now
	ParkedBytes int64  // their snapshots' total
	BudgetMiB   int64  // the RAM cache's cap, 0: no RAM cache
	MinFreeMiB  int64  // the free-RAM floor a park must leave
	MinTokens   int    // the shortest conversation the engine parks
}

// Engine is one (backend, model) resident identity. All calls are for the model
// this Engine names; an Engine is never used to probe a model it does not own.
type Engine interface {
	// ID is the route id, e.g. "strata/flash-next" — the session namespace.
	ID() string
	// Model is the model name a client puts in its request.
	Model() string

	// Status reads the server's state between requests. It must never load a model.
	Status(ctx context.Context) (Status, error)
	// Fingerprint is the 12-hex identity of model + build + settings that a session
	// file is tied to. It must never load a model; error when it cannot be learned.
	Fingerprint(ctx context.Context) (string, error)

	// Save writes the live conversation to name in the server's session folder.
	Save(ctx context.Context, name string) (tokens int, bytes int64, err error)
	// Restore makes name the live conversation. Dialect note: llama's action is
	// "load", Strata's is "restore".
	Restore(ctx context.Context, name string) (tokens int, bytes int64, err error)

	// Load makes the model resident (explicit, so it is never inside a client's
	// request). Load must fail fast when the model cannot fit.
	Load(ctx context.Context) error
	// Unload gives the VRAM back, keeping the server process alive; it waits up
	// to wait for a running turn (Strata's ?wait=), and returns the engine's answer.
	Unload(ctx context.Context, wait time.Duration) error

	// Wake sends one 1-token request to trigger a lazy load.
	Wake(ctx context.Context) error

	// VramFree reports bytes free on the GPU this engine would need, 0 when the
	// server cannot answer; used by the arbiter's margin gate.
	VramFree(ctx context.Context) (int64, error)

	// CacheInfo is the engine's in-RAM conversation cache. An engine with no RAM
	// cache (llama.cpp) returns zeros and no error.
	CacheInfo(ctx context.Context) (CacheInfo, error)

	// IsUnloaded reports whether err is the server refusing a slot call because
	// its model is not resident.
	IsUnloaded(err error) bool
}

// Options configures one Engine.
type Options struct {
	// ID is the route id, "strata/flash-next".
	ID string
	// Model is the model name for client requests, e.g. qwen3.8-flash-next-iq2_xs.
	Model string
	// Base is the server URL, e.g. http://127.0.0.1:8095.
	Base string
	// APIKey is the bearer token when the server requires one (llama's --api-key).
	APIKey string
	// Timeout bounds one slot call; a 200K-token session file is several GB.
	Timeout time.Duration
}
