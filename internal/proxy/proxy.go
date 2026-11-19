// Package proxy is the client-facing end of the arbiter: one address, many
// engines. Each generation request names a model; the arbiter makes that
// model's route resident, and the request is forwarded to that route's server
// with its session restored if the engine no longer holds it. Requests for the
// route already resident are served in arrival order and never preempted, so a
// child's run of turns finishes before the card moves.
package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"sync"
	"time"

	"github.com/frodosik/vram-mux/internal/arbiter"
	"github.com/frodosik/vram-mux/internal/store"
)

// Options configure the proxy.
type Options struct {
	Arbiter *arbiter.Arbiter
	Store   *store.Store // nil: a plain proxy, no disk cache
	Policy  Policy
	MaxBody int64 // generation bodies larger than this are streamed through unbuffered
	Metrics *Metrics
	Log     *slog.Logger
}

// Proxy is the client-facing http.Handler.
type Proxy struct {
	arb     *arbiter.Arbiter
	cache   *cache                            // built over the arbiter; its store may be nil, so no session files
	proxies map[string]*httputil.ReverseProxy // by route id
	models  map[string]string                 // route id → the model it answers for: resident_info and the refusal message
	m       *Metrics
	maxBody int64
	log     *slog.Logger

	mu     sync.Mutex
	recent []record // the last generation requests, for /requests
}

// New returns a proxy over the arbiter's routes.
func New(opt Options) *Proxy {
	p := &Proxy{arb: opt.Arbiter, maxBody: opt.MaxBody, m: opt.Metrics, log: opt.Log,
		proxies: map[string]*httputil.ReverseProxy{}, models: map[string]string{}}
	if opt.Arbiter != nil {
		p.cache = newCache(opt.Arbiter, opt.Store, opt.Metrics, opt.Log, opt.Policy)
		// A swap ends both engines' live conversations: the arbiter says so, and the
		// cache must stop calling a request a continuation of a conversation the
		// engine no longer holds.
		opt.Arbiter.SetObservers(observers{m: opt.Metrics, c: p.cache})
		for _, r := range opt.Arbiter.Routes() {
			p.proxies[r.ID] = reverseProxy(r, opt.Log)
			p.models[r.ID] = r.Model
		}
		if opt.Metrics != nil {
			// resident_info needs the route→model table before the first request,
			// and the route the arbiter says owns the card at startup gets its 1.
			opt.Metrics.SetRoutes(p.models)
			opt.Metrics.SeedResident(opt.Arbiter.Resident())
		}
	}
	return p
}

// observers is the arbiter's observer: the metrics, plus the cache's note that
// a swap cleared both routes' live conversation, and that a cold load cleared the
// conversation the loaded engine used to hold.
type observers struct {
	m *Metrics
	c *cache
}

func (o observers) Swap(from, to string, seconds float64) {
	if o.m != nil {
		o.m.Swap(from, to, seconds)
	}
	if o.c != nil {
		o.c.Swap(from, to, seconds)
	}
}

func (o observers) SwapBlocked(reason string) {
	if o.m != nil {
		o.m.SwapBlocked(reason)
	}
}

func (o observers) Queue(n int, resident string) {
	if o.m != nil {
		o.m.Queue(n, resident)
	}
	if o.c != nil {
		o.c.Queue(n, resident)
	}
}

func (o observers) Reloaded(route string) {
	if o.c != nil {
		o.c.Reloaded(route)
	}
}

// reverseProxy forwards to one route's server. The client's Host and Origin are
// kept so the server's own checks see the client's name; the route's api-key is
// added, since the arbiter, not the client, picked the server.
func reverseProxy(r arbiter.Route, log *slog.Logger) *httputil.ReverseProxy {
	base, err := url.Parse(r.Base)
	if err != nil {
		log.Error("route base", "route", r.ID, "base", r.Base, "err", err)
		base = &url.URL{Scheme: "http", Host: "127.0.0.1:0"}
	}
	key := r.APIKey
	return &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.SetURL(base)
			pr.Out.Host = pr.In.Host
			for _, h := range []string{"X-Forwarded-For", "X-Forwarded-Host", "X-Forwarded-Proto"} {
				if v, ok := pr.In.Header[h]; ok { // Rewrite drops them; a client behind another proxy keeps them
					pr.Out.Header[h] = v
				}
			}
			if key != "" {
				pr.Out.Header.Set("Authorization", "Bearer "+key)
			}
		},
		Transport: &http.Transport{
			DialContext:         (&net.Dialer{Timeout: 5 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
			MaxIdleConnsPerHost: 64,
			IdleConnTimeout:     90 * time.Second,
			DisableCompression:  true, // the client's Accept-Encoding goes through as it is
		},
		FlushInterval: -1, // SSE: every chunk as it comes
		ErrorHandler: func(w http.ResponseWriter, req *http.Request, err error) {
			if req.Context().Err() != nil {
				return // the client went away; nobody to answer
			}
			log.Warn("upstream", "route", r.ID, "method", req.Method, "path", req.URL.Path, "err", err)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			_, _ = io.WriteString(w, `{"error":{"type":"server_error","code":502,"message":"`+r.ID+` is not answering (starting or restarting?)"}}`)
		},
	}
}

// Close stops the cache, waiting for a save under way until ctx ends.
func (p *Proxy) Close(ctx context.Context) error {
	if p.cache == nil {
		return nil
	}
	return p.cache.close(ctx)
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	gen := r.Method == http.MethodPost &&
		(r.URL.Path == "/v1/chat/completions" || r.URL.Path == "/v1/messages" || r.URL.Path == "/v1/responses")
	if !gen {
		// A control call goes to the route that is resident, never to a model that
		// is not: a per-model endpoint on an idle router loads that model. In strict
		// mode the body is read for a model name before a byte reaches an engine, so
		// the rule is one rule across every path — the model a request names has to
		// be the model the card holds, and the route the request is forwarded to is
		// the route the rule checked, read once. A POST that names none (/slots,
		// /v1/unload, a raw prompt) goes to the resident, as before. In single-route
		// mode there is no rule to check, so no control body is read at all.
		resident := p.arb.Resident()
		strict := p.arb.Strict()
		if p.cache != nil && r.Method != http.MethodGet && r.Method != http.MethodHead && r.Method != http.MethodOptions {
			if strict {
				head, err := io.ReadAll(io.LimitReader(r.Body, prefixRead))
				if err != nil {
					http.Error(w, "reading the request body: "+err.Error(), http.StatusBadRequest)
					return
				}
				r.Body = struct {
					io.Reader
					io.Closer
				}{io.MultiReader(bytes.NewReader(head), r.Body), r.Body}
				if model := modelOf(head); model != "" {
					if !p.arb.Serves(model) {
						p.notServed(w, r, model)
						return
					}
					if p.arb.RouteOf(model).ID != resident {
						p.refused(w, r, model, (&arbiter.NotResident{Model: model, ResidentModel: p.models[resident]}).Error(), actRefused)
						return
					}
				}
			}
			p.cache.invalidate() // /v1/completions, /slots, /load ...: a live conversation may have changed
		}
		if resident == "" {
			resident = p.arb.RouteOf("").ID // the card is idle: the first route answers
		}
		p.proxies[resident].ServeHTTP(w, r)
		return
	}

	start := time.Now()
	t := &ticket{action: actPass, reason: "body_too_large"}
	body, err := io.ReadAll(io.LimitReader(r.Body, p.maxBody+1))
	if err != nil {
		http.Error(w, "reading the request body: "+err.Error(), http.StatusBadRequest)
		return
	}
	// The model the client asked for, read from the buffered prefix. A body cut at
	// the limit still gives it: in every client the model is the first key.
	model := modelOf(body)
	// The arbiter owns the card for every generation request, oversized ones
	// included: this request's model is made the resident, behind the resident's
	// queued turns, before a byte reaches a server. Forwarding an oversized body to
	// a route that is not resident runs it on an engine whose model is not loaded,
	// with no yield save of the conversation the card holds, beside the resident's
	// turn. Admission is the same call for both paths.
	rt, rel, err := p.arb.Acquire(r.Context(), model)
	if err != nil {
		var nr *arbiter.NoRoute
		if errors.As(err, &nr) {
			p.notServed(w, r, nr.Model)
			return
		}
		p.log.Warn("not admitted", "model", model, "err", err)
		if r.Context().Err() != nil {
			// The client walked away while the request waited for the card. Nobody is
			// left to answer, and a request that never got the card is not a denial
			// the operator has to read: no record, no counter, no body.
			return
		}
		p.denied(w, r, model, err)
		return
	}
	defer rel()
	if int64(len(body)) > p.maxBody {
		// The body is streamed through as it is, unbuffered from here. No key, so
		// no session copy, no restore: the request is registered as engine-touching
		// at admission (touch), and that clears every route's live-conversation
		// belief on the way in — the engine is about to hold a conversation this
		// cache cannot name — which finish then keeps cleared.
		r.Body = struct {
			io.Reader
			io.Closer
		}{io.MultiReader(bytes.NewReader(body), r.Body), r.Body}
		t = p.cache.touch(p.cache.route(rt.ID))
	} else {
		r.Body, r.ContentLength, r.TransferEncoding = io.NopCloser(bytes.NewReader(body)), int64(len(body)), nil
		t = p.cache.prepare(r.Context(), r.URL.Path, body, p.cache.route(rt.ID))
	}
	prepared := time.Since(start)
	client, _, _ := net.SplitHostPort(r.RemoteAddr)
	p.remember(record{At: float64(time.Now().UnixMicro()) / 1e6, Path: r.URL.Path, Client: client,
		UserAgent: r.UserAgent(), Model: model, Action: t.action, Route: t.route.id})

	// tail starts with a newline so the first chunk can be seen as a line start
	sw := &statusWriter{ResponseWriter: w, status: http.StatusOK, tail: []byte{'\n'}}
	defer func() {
		gone := r.Context().Err() != nil && !sw.done
		p.cache.finish(t, sw.status, gone)
		// one line per request: what the cache decided, never what was said. The
		// requested model is logged with the route that answered, so an alias
		// (single-route mode) is visible in the log, not only in the client's label.
		p.log.Info("request", "path", r.URL.Path, "client", client, "model", model, "route", t.route.id,
			"action", t.action, "reason", t.reason, "tokens", t.tokens, "prepare_ms", prepared.Milliseconds(),
			"status", sw.status, "client_gone", gone, "ms", time.Since(start).Milliseconds())
	}()
	p.proxies[t.route.id].ServeHTTP(sw, r)
}

// refused answers a request the proxy turned away, in the error shape the client
// reads, and records it: a 404 has to be visible in /requests and in
// requests_total, not only in the log. The route is left empty — the engine was
// never touched.
func (p *Proxy) refused(w http.ResponseWriter, r *http.Request, model, msg, action string) {
	p.log.Warn("refused", "model", model, "msg", msg)
	client, _, _ := net.SplitHostPort(r.RemoteAddr)
	p.remember(record{At: float64(time.Now().UnixMicro()) / 1e6, Path: r.URL.Path, Client: client,
		UserAgent: r.UserAgent(), Model: model, Action: action, Route: ""})
	if p.m != nil {
		p.m.Request(action)
	}
	b, _ := json.Marshal(map[string]any{"error": map[string]any{"type": "invalid_request_error",
		"message": msg}})
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusNotFound)
	_, _ = w.Write(b)
}

// notServed answers a request whose model this arbiter does not serve. The arbiter
// is the source of what it serves, so the message comes from its NoRoute: one
// place says what the rule is.
func (p *Proxy) notServed(w http.ResponseWriter, r *http.Request, model string) {
	p.refused(w, r, model, (&arbiter.NoRoute{Model: model}).Error(), actRefused)
}

// denied answers a request whose model this arbiter serves, when the card was busy
// for it: 503, recorded in /requests and counted as action="denied", so a denial
// is read next to the refusals, not only in the log.
func (p *Proxy) denied(w http.ResponseWriter, r *http.Request, model string, err error) {
	p.log.Warn("not admitted", "model", model, "err", err)
	client, _, _ := net.SplitHostPort(r.RemoteAddr)
	p.remember(record{At: float64(time.Now().UnixMicro()) / 1e6, Path: r.URL.Path, Client: client,
		UserAgent: r.UserAgent(), Model: model, Action: actDenied, Route: ""})
	if p.m != nil {
		p.m.Request(actDenied)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusServiceUnavailable)
	_, _ = io.WriteString(w, `{"error":{"type":"server_error","code":503,"message":"the GPU is busy: `+err.Error()+`"}}`)
}

// prefixRead is how much of a control POST is read for its model name. llama.cpp's
// per-model endpoints take the model in the body, so the strict rule has to see it
// before a byte reaches an engine; a longer body is forwarded whole, unbuffered.
const prefixRead = 64 << 10

// modelOf is the request's model name, "" when the body names none. It reads the
// prefix, not the whole body: a body cut at --max-body-mib keeps the model, which
// is the first key in every client's request. A generation request whose prefix
// names no model is refused in strict mode, because no route can be chosen for it;
// a control POST that names no model goes to the resident, as it always has, so it
// never triggers a swap.
func modelOf(body []byte) string {
	dec := json.NewDecoder(bytes.NewReader(body))
	tok, err := dec.Token()
	if err != nil || tok != json.Delim('{') {
		return "" // not an object: no model to read
	}
	for dec.More() {
		ktok, err := dec.Token()
		if err != nil {
			return ""
		}
		key, ok := ktok.(string)
		if !ok {
			return ""
		}
		if key == "model" {
			var v string
			if dec.Decode(&v) != nil {
				return ""
			}
			return v
		}
		var v any // the key we want is at the top level; a nested value is skipped whole
		if dec.Decode(&v) != nil {
			return ""
		}
	}
	return ""
}

// statusWriter records the status; Unwrap keeps Flush (SSE) and hijacking.
type statusWriter struct {
	http.ResponseWriter
	status int
	wrote  bool
	done   bool   // the stream reached its terminal event
	tail   []byte // the last bytes written, seeded with a newline at stream start
}

// tailKeep is what Write remembers across calls: enough to see a marker split
// between two writes.
const tailKeep = 64

func (s *statusWriter) WriteHeader(code int) {
	if !s.wrote && code >= 200 {
		s.status, s.wrote = code, true
	}
	s.ResponseWriter.WriteHeader(code)
}

func (s *statusWriter) Write(b []byte) (int, error) {
	s.wrote = true
	if !s.done && len(b) > 0 {
		scanned := make([]byte, 0, len(s.tail)+len(b))
		scanned = append(scanned, s.tail...)
		scanned = append(scanned, b...)
		for _, m := range doneMarkers {
			if bytes.Contains(scanned, m) {
				s.done = true
				s.tail = nil
				break
			}
		}
		if !s.done {
			if len(scanned) > tailKeep {
				s.tail = append(s.tail[:0], scanned[len(scanned)-tailKeep:]...)
			} else {
				s.tail = scanned
			}
		}
	}
	return s.ResponseWriter.Write(b)
}

// doneMarkers are the events that close a stream in the three dialects; none of
// them appears mid-stream. A marker is searched in the whole chunk, because the
// Responses API writes kilobytes after "response.completed". [DONE] is anchored
// to its line: inside JSON strings newlines are escaped, so a model quoting
// "data: [DONE]" in streamed content is not the end of the stream.
var doneMarkers = [][]byte{
	[]byte("\ndata: [DONE]"),
	[]byte(`"finish_reason":"stop"`), []byte(`"finish_reason": "stop"`),
	[]byte(`"finish_reason":"tool_calls"`), []byte(`"finish_reason": "tool_calls"`),
	[]byte(`"finish_reason":"length"`), []byte(`"finish_reason": "length"`),
	[]byte(`"type":"message_stop"`), []byte(`"type": "message_stop"`),
	[]byte(`"type":"response.completed"`), []byte(`"type": "response.completed"`),
}

func (s *statusWriter) Unwrap() http.ResponseWriter { return s.ResponseWriter }

// record is one generation request as the proxy saw it. The server sees only the
// proxy's address, so a tracer that reads its request log can join these on by
// path and start time.
type record struct {
	At        float64 `json:"at"` // unix seconds, when it was forwarded
	Path      string  `json:"path"`
	Client    string  `json:"client"`
	UserAgent string  `json:"user_agent"`
	Model     string  `json:"model"` // the model the request named
	Route     string  `json:"route"` // the route that answered it
	Action    string  `json:"diskcache"`
}

func (p *Proxy) remember(r record) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.recent = append(p.recent, r)
	if len(p.recent) > 256 {
		p.recent = append([]record(nil), p.recent[len(p.recent)-256:]...)
	}
}

// requestsJSON is the /requests body, shared by the handler and the tests.
func (p *Proxy) requestsJSON() string {
	p.mu.Lock()
	recs := append([]record(nil), p.recent...)
	p.mu.Unlock()
	b, _ := json.Marshal(map[string]any{"requests": recs})
	return string(b)
}

// Requests serves the last 256 generation requests as JSON.
func (p *Proxy) Requests(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = io.WriteString(w, p.requestsJSON())
}

// Resident is the route holding the card, for /health.
func (p *Proxy) Resident() string { return p.arb.Resident() }
