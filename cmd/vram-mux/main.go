// Command vram-mux is a reverse proxy that owns one GPU for several engine
// processes (Strata, github.com/Niko1221/Strata, and llama.cpp servers), caches
// conversations on disk, and restores them when a request needs one.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/alecthomas/kong"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"golang.org/x/sync/errgroup"
	"gopkg.in/natefinch/lumberjack.v2"

	"github.com/frodosik/vram-mux/internal/arbiter"
	"github.com/frodosik/vram-mux/internal/engine"
	"github.com/frodosik/vram-mux/internal/proxy"
	"github.com/frodosik/vram-mux/internal/store"
)

// version is set at build time: -ldflags "-X main.version=v1.2.3".
var version = "dev"

// defaultVRAMSysfs is the --vram-sysfs default, kept here so the code can tell
// "the operator named a card" from "the card has to be found".
const defaultVRAMSysfs = "/sys/class/drm/card0/device"

// vramCard is the node the free-VRAM gate reads. An explicit flag is taken as
// given; the default names card0, which on a box with an integrated GPU is not
// the card the engines share, so the default auto-detects the card with the most
// VRAM. A detect that finds nothing keeps the flag.
func vramCard(flag string, detect func() string) string {
	if flag != defaultVRAMSysfs {
		return flag
	}
	if card := detect(); card != "" {
		return card
	}
	return flag
}

// config is the command line. Every flag can also be set by its environment
// variable; the flag wins.
type config struct {
	Listen        string `env:"LISTEN" default:":8080" help:"Address clients connect to."`
	Upstream      string `env:"UPSTREAM" default:"http://127.0.0.1:8081" help:"Strata's URL."`
	MetricsListen string `env:"METRICS_LISTEN" default:"127.0.0.1:8082" help:"Address for /metrics, /requests and /health. Keep it private."`
	SessionDir    string `env:"SESSION_DIR" help:"Strata's slot_save_path. Empty: a plain proxy, no disk cache."`

	MaxGiB        float64 `name:"max-gib" env:"MAX_GIB" default:"40" help:"Total size cap of the session files, GiB. Least recently used go first."`
	RetentionDays float64 `env:"RETENTION_DAYS" default:"7" help:"Delete session files unused for this many days."`
	MinFreeMiB    float64 `name:"min-free-mib" env:"MIN_FREE_MIB" default:"4096" help:"Never leave less than this free on the session filesystem. Match Strata's --session-min-free-mib."`

	MinTokens             int     `env:"MIN_TOKENS" default:"15000" help:"Save conversations of at least this many tokens."`
	ParkMinTokens         int     `env:"PARK_MIN_TOKENS" default:"-1" help:"Strata's --conversation-cache-min-tokens. -1 reads it from Strata."`
	RegrowTokens          int     `env:"REGROW_TOKENS" default:"4096" help:"Save a conversation again once it has grown this much past its file."`
	IdleSaveS             float64 `name:"idle-save-s" env:"IDLE_SAVE_S" default:"3" help:"Quiet seconds after a turn before it is saved. 0: save only when a switch would lose it."`
	MaxIdleSaveTokens     int     `name:"max-idle-save-tokens" env:"MAX_IDLE_SAVE_TOKENS" default:"32768" help:"Skip a background idle save of a conversation longer than this: a 100K-token session writes 1.4 GB in ~110 s and holds the save lock, so a queued request waits for it. The yield save at a switch writes the same file in ~4 s. 0: no cap."`
	SnapshotBytesPerToken int64   `env:"SNAPSHOT_BYTES_PER_TOKEN" default:"65536" help:"RAM per token of a parked conversation, to predict when Strata will refuse to park one."`
	MaxBodyMiB            int64   `name:"max-body-mib" env:"MAX_BODY_MIB" default:"64" help:"Generation request bodies larger than this are streamed through without caching."`
	ShutdownS             float64 `name:"shutdown-s" env:"SHUTDOWN_S" default:"20" help:"On SIGTERM, how long running requests and a save under way get to finish."`

	StrataModel   string `name:"strata-model" env:"STRATA_MODEL" help:"The model the Strata route answers for. Required: it names the route id and the session namespace."`
	StrataAPIKey  string `name:"strata-api-key" env:"STRATA_API_KEY" help:"Bearer token Strata needs, if any."`
	StrataNeedMiB int64  `name:"strata-need-mib" env:"STRATA_NEED_MIB" default:"0" help:"VRAM the Strata route must have free to load: its floor, not what it fills (expert-cache auto sizes to the free VRAM it finds). 0: gate on the margin only."`

	Routes []string `name:"route" env:"ROUTES" sep:" " help:"One route per occurrence: dialect|id|model|base|api-key|need-mib. Example: llama|llama/Swift-27B|qwen3.8-swift-1.5-architect|http://127.0.0.1:8001|secret|18000"`

	MaxResidentStreak int     `name:"max-resident-streak" env:"MAX_RESIDENT_STREAK" default:"32" help:"Requests of the resident route drained before a waiter may take the card. 0: unbounded."`
	VRAMMarginMiB     int64   `name:"vram-margin-mib" env:"VRAM_MARGIN_MIB" default:"1024" help:"Bytes that must stay free on the card before a route is loaded."`
	DrainWaitS        float64 `name:"drain-wait-s" env:"DRAIN_WAIT_S" default:"300" help:"How long a yield waits for a running turn to finish."`
	LoadTimeoutS      float64 `name:"load-timeout-s" env:"LOAD_TIMEOUT_S" default:"600" help:"The whole swap: drain, free-VRAM gate, load."`
	VRAMSysfs         string  `name:"vram-sysfs" env:"VRAM_SYSFS" default:"/sys/class/drm/card0/device" help:"The amdgpu device node the free-VRAM gate reads."`
	GPUIndex          int64   `name:"gpu-index" env:"GPU_INDEX" default:"-1" help:"The card nvidia-smi answers for when no amdgpu card is found: -1 picks the card with the most VRAM."`
	VRAMPollS         float64 `name:"vram-poll-s" env:"VRAM_POLL_S" default:"2" help:"How often the free-VRAM gate re-reads the card while it waits for a load."`
	BodyDivisor       int     `name:"body-divisor" env:"BODY_DIVISOR" default:"4" help:"Request bytes per token, the estimate for an engine that reports no token count. 0: never estimate."`

	Log struct {
		Output     string `env:"OUTPUT" enum:"stderr,file,both" default:"stderr" help:"Where logs go: stderr, file or both."`
		File       string `env:"FILE" default:"/var/log/vram-mux/vram-mux.log" help:"Log file, for file and both."`
		Format     string `env:"FORMAT" enum:"text,json" default:"text" help:"text or json."`
		Level      string `env:"LEVEL" enum:"debug,info,warn,error" default:"info" help:"debug, info, warn or error."`
		MaxSizeMiB int    `name:"max-size-mib" env:"MAX_SIZE_MIB" default:"100" help:"Rotate the log file at this size."`
		MaxBackups int    `env:"MAX_BACKUPS" default:"10" help:"Rotated log files to keep (0: all)."`
		MaxAgeDays int    `env:"MAX_AGE_DAYS" default:"30" help:"Delete rotated log files older than this (0: never)."`
		Compress   bool   `env:"COMPRESS" default:"true" negatable:"" help:"gzip rotated log files."`
	} `embed:"" prefix:"log-" envprefix:"LOG_" group:"Logging"`

	Version kong.VersionFlag `help:"Print the version and exit."`
}

// Validate is called by kong after parsing.
func (c *config) Validate() error {
	if _, err := c.routes(); err != nil {
		return err
	}
	if c.MaxGiB <= 0 || c.RetentionDays <= 0 || c.MinTokens < 1 || c.SnapshotBytesPerToken <= 0 || c.MaxBodyMiB < 1 ||
		c.ShutdownS <= 0 || c.MinFreeMiB < 0 || c.IdleSaveS < 0 || c.MaxIdleSaveTokens < 0 || c.RegrowTokens < 0 || c.Log.MaxSizeMiB < 1 ||
		c.VRAMMarginMiB < 0 || c.VRAMPollS <= 0 || c.DrainWaitS <= 0 || c.LoadTimeoutS <= 0 || c.BodyDivisor < 0 || c.MaxResidentStreak < 0 || c.GPUIndex < -1 {
		return errors.New("sizes, limits and timeouts must be positive (--min-free-mib, --idle-save-s, --regrow-tokens, --max-resident-streak, --vram-margin-mib, --body-divisor and --gpu-index may be -1)")
	}
	if c.StrataModel == "" && len(c.Routes) == 0 {
		return errors.New("--strata-model is required, or give at least one --route")
	}
	return nil
}

// resolveKey turns an api-key field into the bearer token to send. "file:PATH"
// reads the first line of that file at startup, so the key never sits in the
// process arguments where `ps` shows it; a plain value is used as it is.
func resolveKey(spec string) (string, error) {
	if !strings.HasPrefix(spec, "file:") {
		return spec, nil
	}
	path := strings.TrimPrefix(spec, "file:")
	b, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("api-key file %q: %w", path, err)
	}
	line := strings.SplitN(strings.TrimSpace(string(b)), "\n", 2)[0]
	if line == "" {
		return "", fmt.Errorf("api-key file %q is empty", path)
	}
	return line, nil
}

// routes is the arbiter's route list: the Strata route first (the resident a
// request that names no model goes to), then each --route in order. The Strata
// route exists only when --strata-model is set, so a llama-only box needs none.
func (c *config) routes() ([]arbiter.Route, error) {
	strataKey, err := resolveKey(c.StrataAPIKey)
	if err != nil {
		return nil, err
	}
	var out []arbiter.Route
	if c.StrataModel != "" {
		out = append(out, arbiter.Route{
			ID:        "strata/" + c.StrataModel,
			Dialect:   "strata",
			Model:     c.StrataModel,
			Base:      c.Upstream,
			APIKey:    strataKey,
			Need:      c.StrataNeedMiB << 20,
			Namespace: store.NamespaceOf("strata", c.StrataModel),
		})
	}
	for _, spec := range c.Routes {
		f := strings.Split(spec, "|")
		if len(f) != 6 {
			return nil, fmt.Errorf("--route needs dialect|id|model|base|api-key|need-mib, got %q", spec)
		}
		if f[0] != "strata" && f[0] != "llama" {
			return nil, fmt.Errorf("--route dialect must be strata or llama, got %q", f[0])
		}
		need, err := strconv.ParseInt(f[5], 10, 64)
		if err != nil || need < 0 {
			return nil, fmt.Errorf("--route need-mib must be a whole number of MiB, got %q", f[5])
		}
		if f[1] == "" || f[2] == "" || f[3] == "" {
			return nil, fmt.Errorf("--route needs an id, a model and a base URL: %q", spec)
		}
		for _, r := range out {
			if r.ID == f[1] {
				return nil, fmt.Errorf("--route id %q is used twice", f[1])
			}
		}
		key, err := resolveKey(f[4])
		if err != nil {
			return nil, err
		}
		out = append(out, arbiter.Route{
			ID: f[1], Dialect: f[0], Model: f[2], Base: f[3], APIKey: key,
			Need: need << 20, Namespace: store.NamespaceOf(f[0], f[2]),
		})
	}
	for i, r := range out {
		u, err := url.Parse(r.Base)
		if err != nil || u.Scheme == "" || u.Host == "" {
			return nil, fmt.Errorf("route %q: base must be a URL like http://127.0.0.1:8001", r.ID)
		}
		load := time.Duration(c.LoadTimeoutS * float64(time.Second))
		switch r.Dialect {
		case "strata":
			out[i].Engine = engine.NewStrata(engine.Options{ID: r.ID, Model: r.Model, Base: r.Base, APIKey: r.APIKey, LoadTimeout: load})
		case "llama":
			out[i].Engine = engine.NewLlama(engine.Options{ID: r.ID, Model: r.Model, Base: r.Base, APIKey: r.APIKey, LoadTimeout: load})
		}
	}
	return out, nil
}

func parse(args []string) (*config, error) {
	var c config
	k, err := kong.New(&c, kong.Name("vram-mux"),
		kong.Description("One GPU, several engine processes: the arbiter owns model loads, and conversations are cached on disk."),
		kong.Vars{"version": "vram-mux " + version}, kong.UsageOnError())
	if err != nil {
		return nil, err
	}
	_, err = k.Parse(args)
	return &c, err
}

// newLogger builds the slog logger and a closer for its file. A log file is
// appended to across restarts and rotates by size into
// <name>-<timestamp>.log(.gz) next to it.
func newLogger(c *config, stderr io.Writer) (*slog.Logger, io.Closer, error) {
	var level slog.Level
	if err := level.UnmarshalText([]byte(c.Log.Level)); err != nil {
		return nil, nil, err
	}
	w, closer := stderr, io.Closer(io.NopCloser(nil))
	if c.Log.Output != "stderr" {
		if err := os.MkdirAll(filepath.Dir(c.Log.File), 0o755); err != nil {
			return nil, nil, fmt.Errorf("log folder: %w", err)
		}
		f := &lumberjack.Logger{Filename: c.Log.File, MaxSize: c.Log.MaxSizeMiB, MaxBackups: c.Log.MaxBackups,
			MaxAge: c.Log.MaxAgeDays, Compress: c.Log.Compress}
		w, closer = f, f
		if c.Log.Output == "both" {
			w = io.MultiWriter(stderr, f)
		}
	}
	opts := &slog.HandlerOptions{Level: level}
	if c.Log.Format == "json" {
		return slog.New(slog.NewJSONHandler(w, opts)), closer, nil
	}
	return slog.New(slog.NewTextHandler(w, opts)), closer, nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "vram-mux:", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := parse(os.Args[1:])
	if err != nil {
		return err
	}
	log, closeLog, err := newLogger(cfg, os.Stderr)
	if err != nil {
		return err
	}
	defer func() { _ = closeLog.Close() }()
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	var lc net.ListenConfig
	pl, err := lc.Listen(ctx, "tcp", cfg.Listen)
	if err != nil {
		return err
	}
	al, err := lc.Listen(ctx, "tcp", cfg.MetricsListen)
	if err != nil {
		return err
	}
	return serve(ctx, cfg, log.With("version", version), pl, al)
}

// serve runs the proxy and the admin server on the listeners until ctx ends,
// then gives running requests and a save under way --shutdown-s to finish.
func serve(ctx context.Context, cfg *config, log *slog.Logger, proxyL, adminL net.Listener) error {

	maxBytes := int64(cfg.MaxGiB * (1 << 30))
	reg := prometheus.NewRegistry()
	reg.MustRegister(collectors.NewGoCollector(), collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))
	m := proxy.NewMetrics(reg)

	var st *store.Store
	if cfg.SessionDir != "" {
		st = store.New(store.Options{Dir: cfg.SessionDir, MaxBytes: maxBytes, Log: log, OnEvict: m.Evicted,
			MaxAge: time.Duration(cfg.RetentionDays * 24 * float64(time.Hour)), MinFree: int64(cfg.MinFreeMiB * (1 << 20))})
		if err := st.Load(); err != nil {
			log.Warn("session folder unreadable; disk cache off, plain proxy", "dir", cfg.SessionDir, "err", err)
			st = nil
		} else {
			m.WatchStore(reg, st, maxBytes)
			n, _ := st.Stats()
			log.Info("disk cache", "dir", cfg.SessionDir, "max_gib", cfg.MaxGiB, "retention_days", cfg.RetentionDays, "files", n)
		}
	}
	engine.SysfsVRAM = vramCard(cfg.VRAMSysfs, engine.DetectCard)
	engine.NvidiaGPUIndex = cfg.GPUIndex
	if _, err := engine.SysfsFree(); err == nil {
		log.Info("vram gate", "backend", "sysfs", "card", engine.SysfsVRAM)
	} else {
		log.Info("vram gate", "backend", "nvidia-smi", "gpu_index", cfg.GPUIndex)
	}
	routes, err := cfg.routes()
	if err != nil {
		return err
	}
	arb := arbiter.New(arbiter.Options{
		Routes:            routes,
		MaxResidentStreak: cfg.MaxResidentStreak,
		DrainWait:         time.Duration(cfg.DrainWaitS * float64(time.Second)),
		LoadTimeout:       time.Duration(cfg.LoadTimeoutS * float64(time.Second)),
		VRAMMargin:        cfg.VRAMMarginMiB << 20,
		VRAMFree:          engine.FreeVRAM,
		PollInterval:      time.Duration(cfg.VRAMPollS * float64(time.Second)),
		// The routes are strict: a request must name one of the models that were
		// configured. Without --route flags the single route built from --upstream
		// answers every request, and the proxy logs the model each one asked for.
		Strict: len(cfg.Routes) > 0,
		// The Strata route owns the card at startup: a --lazy server takes its model
		// on the first request, so the arbiter must not load it and must not read the
		// engine as idle. The strata route is always routes[0].
		InitialResident: routes[0].ID,
		Log:             log,
		Observers:       m,
	})
	p := proxy.New(proxy.Options{Arbiter: arb, Store: st, MaxBody: cfg.MaxBodyMiB << 20, Metrics: m, Log: log,
		Policy: proxy.Policy{MinTokens: cfg.MinTokens, ParkMinTokens: cfg.ParkMinTokens, RegrowTokens: cfg.RegrowTokens,
			IdleDelay: time.Duration(cfg.IdleSaveS * float64(time.Second)), MaxIdleSaveTokens: cfg.MaxIdleSaveTokens,
			SnapshotBytesPerToken: cfg.SnapshotBytesPerToken, BodyDivisor: cfg.BodyDivisor}})

	mux := http.NewServeMux()
	mux.Handle("GET /metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))
	mux.HandleFunc("GET /requests", p.Requests)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "ok resident="+p.Resident()+"\n")
	})
	srv := &http.Server{Handler: p, ReadHeaderTimeout: 30 * time.Second}
	admin := &http.Server{Handler: mux, ReadHeaderTimeout: 10 * time.Second}

	g, gctx := errgroup.WithContext(ctx)
	for _, s := range []struct {
		srv *http.Server
		l   net.Listener
	}{{srv, proxyL}, {admin, adminL}} {
		g.Go(func() error {
			if err := s.srv.Serve(s.l); !errors.Is(err, http.ErrServerClosed) {
				return err
			}
			return nil
		})
	}
	g.Go(func() error {
		tick := time.NewTicker(10 * time.Minute)
		defer tick.Stop()
		for {
			select {
			case <-gctx.Done():
				log.Info("shutting down")
				sctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), time.Duration(cfg.ShutdownS*float64(time.Second)))
				defer cancel()
				return errors.Join(srv.Shutdown(sctx), admin.Shutdown(sctx), p.Close(sctx))
			case <-tick.C:
				if st != nil {
					st.Prune()
				}
			}
		}
	})
	log.Info("listening", "listen", proxyL.Addr().String(), "upstream", cfg.Upstream, "admin", adminL.Addr().String())
	return g.Wait()
}
