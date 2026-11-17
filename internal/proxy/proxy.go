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
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"sync"
	"time"

	"github.com/tfyl/strata-diskcache/internal/arbiter"
	"github.com/tfyl/strata-diskcache/internal/store"
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
	cache   *cache                            // nil without a store
	proxies map[string]*httputil.ReverseProxy // by route id
	maxBody int64
	log     *slog.Logger

	mu     sync.Mutex
	recent []record // the last generation requests, for /requests
}

// New returns a proxy over the arbiter's routes.
func New(opt Options) *Proxy {
	p := &Proxy{arb: opt.Arbiter, maxBody: opt.MaxBody, log: opt.Log, proxies: map[string]*httputil.ReverseProxy{}}
	if opt.Arbiter != nil {
		p.cache = newCache(opt.Arbiter, opt.Store, opt.Metrics, opt.Log, opt.Policy)
		for _, r := range opt.Arbiter.Routes() {
			p.proxies[r.ID] = reverseProxy(r, opt.Log)
		}
	}
	return p
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
		// A control call goes to the route that is resident, never to a model that is
		// not: a per-model endpoint on an idle router loads that model.
		route := p.arb.Resident()
		if route == "" {
			route = p.arb.RouteOf("").ID
		}
		if p.cache != nil && r.Method != http.MethodGet && r.Method != http.MethodHead && r.Method != http.MethodOptions {
			p.cache.invalidate() // /v1/completions, /slots, /load ...: a live conversation may have changed
		}
		p.proxies[route].ServeHTTP(w, r)
		return
	}

	start := time.Now()
	t := &ticket{action: actPass, reason: "body_too_large"}
	body, err := io.ReadAll(io.LimitReader(r.Body, p.maxBody+1))
	switch {
	case err != nil:
		http.Error(w, "reading the request body: "+err.Error(), http.StatusBadRequest)
		return
	case int64(len(body)) > p.maxBody: // stream it through as it is, unbuffered from here
		r.Body = struct {
			io.Reader
			io.Closer
		}{io.MultiReader(bytes.NewReader(body), r.Body), r.Body}
		p.cache.invalidate()
		t.route = p.cache.route(p.home()) // streamed whole: the resident serves it
	default:
		r.Body, r.ContentLength, r.TransferEncoding = io.NopCloser(bytes.NewReader(body)), int64(len(body)), nil
		// The arbiter owns the card: make this request's model the resident, behind
		// the resident's queued turns, before anything is sent to a server.
		model := modelOf(body)
		rt, rel, err := p.arb.Acquire(r.Context(), model)
		if err != nil {
			p.log.Warn("not admitted", "model", model, "err", err)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, `{"error":{"type":"server_error","code":503,"message":"the GPU is busy: `+err.Error()+`"}}`)
			return
		}
		defer rel()
		t = p.cache.prepare(r.Context(), r.URL.Path, body, p.cache.route(rt.ID))
	}
	prepared := time.Since(start)
	client, _, _ := net.SplitHostPort(r.RemoteAddr)
	p.remember(record{At: float64(time.Now().UnixMicro()) / 1e6, Path: r.URL.Path, Client: client,
		UserAgent: r.UserAgent(), Action: t.action, Route: t.route.id})

	// tail starts with a newline so the first chunk can be seen as a line start
	sw := &statusWriter{ResponseWriter: w, status: http.StatusOK, tail: []byte{'\n'}}
	defer func() {
		gone := r.Context().Err() != nil && !sw.done
		p.cache.finish(t, sw.status, gone)
		// one line per request: what the cache decided, never what was said
		p.log.Info("request", "path", r.URL.Path, "client", client, "route", t.route.id, "action", t.action,
			"reason", t.reason, "tokens", t.tokens, "prepare_ms", prepared.Milliseconds(), "status", sw.status,
			"client_gone", gone, "ms", time.Since(start).Milliseconds())
	}()
	p.proxies[t.route.id].ServeHTTP(sw, r)
}

// home is the route a request that names no model, or cannot be keyed, belongs
// to: the resident, and the first route when the card is idle.
func (p *Proxy) home() string {
	if id := p.arb.Resident(); id != "" {
		return id
	}
	if rs := p.arb.Routes(); len(rs) > 0 {
		return rs[0].ID
	}
	return ""
}

// modelOf is the request's model name, "" when the body names none. A request
// with no model goes to the resident route, so it never triggers a swap.
func modelOf(body []byte) string {
	var b struct {
		Model string
	}
	if json.Unmarshal(body, &b) != nil {
		return ""
	}
	return b.Model
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
	Route     string  `json:"route"`
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

// Requests serves the last 256 generation requests as JSON.
func (p *Proxy) Requests(w http.ResponseWriter, _ *http.Request) {
	p.mu.Lock()
	recs := append([]record(nil), p.recent...)
	p.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"requests": recs})
}

// Resident is the route holding the card, for /health.
func (p *Proxy) Resident() string { return p.arb.Resident() }
