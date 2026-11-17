package proxy

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/tfyl/strata-diskcache/internal/store"
)

// Metrics are the proxy's Prometheus metrics, all strata_diskcache_*.
type Metrics struct {
	requests, savesBy, skipped, failures, evictions, evictedBytes, reloads *prometheus.CounterVec
	saves, restores, savedBytes, restoredBytes, panics                     prometheus.Counter
	saveSeconds, restoreSeconds                                            prometheus.Histogram
	lastSave, lastRestore                                                  prometheus.Gauge
	swaps, prefillTokens                                                   prometheus.Counter
	swapSeconds                                                            prometheus.Histogram
	swapBlocked                                                            *prometheus.CounterVec
	queue                                                                  prometheus.Gauge
	resident                                                               *prometheus.GaugeVec
}

// NewMetrics registers the metrics with reg.
func NewMetrics(reg prometheus.Registerer) *Metrics {
	opts := func(name, help string) prometheus.Opts {
		return prometheus.Opts{Namespace: "strata_diskcache", Name: name, Help: help}
	}
	vec := func(name, help string, labels ...string) *prometheus.CounterVec {
		return prometheus.NewCounterVec(prometheus.CounterOpts(opts(name, help)), labels)
	}
	counter := func(name, help string) prometheus.Counter {
		return prometheus.NewCounter(prometheus.CounterOpts(opts(name, help)))
	}
	hist := func(name, help string) prometheus.Histogram {
		return prometheus.NewHistogram(prometheus.HistogramOpts{Namespace: "strata_diskcache", Name: name, Help: help,
			Buckets: []float64{0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60}})
	}
	m := &Metrics{
		requests:       vec("requests_total", "Generation requests by what the cache did.", "action"),
		savesBy:        vec("saves_by_trigger_total", "Session files saved, by trigger.", "trigger"),
		skipped:        vec("saves_skipped_total", "Saves not made, by reason.", "reason"),
		failures:       vec("failures_total", "Saves and restores the engine refused or failed.", "op", "reason"),
		evictions:      vec("evictions_total", "Session files deleted, by reason.", "reason"),
		evictedBytes:   vec("evicted_bytes_total", "Bytes of session files deleted, by reason.", "reason"),
		saves:          counter("saves_total", "Session files saved."),
		restores:       counter("restores_total", "Session files restored."),
		savedBytes:     counter("saved_bytes_total", "Bytes written to session files."),
		restoredBytes:  counter("restored_bytes_total", "Bytes read from session files."),
		panics:         counter("panics_total", "Panics in the cache logic (the request went through)."),
		swaps:          counter("swaps_total", "Times the card was given to another route."),
		reloads:        vec("reloads_total", "Times a resident route's model was loaded from cold, by route.", "route"),
		prefillTokens:  counter("re_prefill_tokens_total", "Tokens of requests that ran with no session to restore, an estimate of what a swap cost in re-prefill."),
		saveSeconds:    hist("save_seconds", "Time to save a session file."),
		restoreSeconds: hist("restore_seconds", "Time to restore a session file."),
		swapSeconds: prometheus.NewHistogram(prometheus.HistogramOpts{Namespace: "strata_diskcache", Name: "swap_seconds",
			Help: "Time from the yield to the new route being resident.", Buckets: []float64{1, 3, 10, 30, 60, 120, 300, 600}}),
		swapBlocked: vec("swap_blocked_total", "A swap that did not happen, by reason.", "reason"),
		queue:       prometheus.NewGauge(prometheus.GaugeOpts(opts("queue_length", "Requests waiting for the card."))),
		lastSave:    prometheus.NewGauge(prometheus.GaugeOpts(opts("last_save_ms", "Duration of the last save."))),
		lastRestore: prometheus.NewGauge(prometheus.GaugeOpts(opts("last_restore_ms", "Duration of the last restore."))),
		resident:    prometheus.NewGaugeVec(prometheus.GaugeOpts(opts("resident_route", "1 for the route holding the card.")), []string{"route"}),
	}
	reg.MustRegister(m.requests, m.savesBy, m.skipped, m.failures, m.evictions, m.evictedBytes, m.reloads,
		m.saves, m.restores, m.savedBytes, m.restoredBytes, m.panics, m.swaps, m.prefillTokens,
		m.saveSeconds, m.restoreSeconds, m.swapSeconds, m.swapBlocked, m.queue, m.resident,
		m.lastSave, m.lastRestore)
	return m
}

// WatchStore exports the store's file count, total size and cap.
func (m *Metrics) WatchStore(reg prometheus.Registerer, st *store.Store, maxBytes int64) {
	gauge := func(name, help string, f func() float64) prometheus.Collector {
		return prometheus.NewGaugeFunc(prometheus.GaugeOpts{Namespace: "strata_diskcache", Name: name, Help: help}, f)
	}
	reg.MustRegister(
		gauge("files", "Session files on disk.", func() float64 { n, _ := st.Stats(); return float64(n) }),
		gauge("bytes", "Total size of the session files.", func() float64 { _, b := st.Stats(); return float64(b) }),
		gauge("max_bytes", "The session files' size cap.", func() float64 { return float64(maxBytes) }),
	)
}

// Evicted counts a deleted file (the store's OnEvict).
func (m *Metrics) Evicted(reason string, bytes int64) {
	m.evictions.WithLabelValues(reason).Inc()
	m.evictedBytes.WithLabelValues(reason).Add(float64(bytes))
}

func (m *Metrics) saved(trigger string, bytes int64, d time.Duration) {
	m.saves.Inc()
	m.savesBy.WithLabelValues(trigger).Inc()
	m.savedBytes.Add(float64(bytes))
	m.saveSeconds.Observe(d.Seconds())
	m.lastSave.Set(float64(d.Milliseconds()))
}

func (m *Metrics) restored(bytes int64, d time.Duration) {
	m.restores.Inc()
	m.restoredBytes.Add(float64(bytes))
	m.restoreSeconds.Observe(d.Seconds())
	m.lastRestore.Set(float64(d.Milliseconds()))
}

// prefill counts a request that ran with nothing to restore: what a swap cost in
// tokens the engine had to read again.
func (m *Metrics) prefill(tokens int) {
	m.prefillTokens.Add(float64(tokens))
}

// Swap is the arbiter's Observers.Swap.
func (m *Metrics) Swap(from, to string, seconds float64) {
	m.swaps.Inc()
	m.swapSeconds.Observe(seconds)
	m.resident.WithLabelValues(to).Set(1)
	if from != "" {
		m.resident.WithLabelValues(from).Set(0)
	}
}

// SwapBlocked is the arbiter's Observers.SwapBlocked.
func (m *Metrics) SwapBlocked(reason string) {
	m.swapBlocked.WithLabelValues(reason).Inc()
}

// Reloaded is the arbiter's Observers.Reloaded.
func (m *Metrics) Reloaded(route string) {
	m.reloads.WithLabelValues(route).Inc()
}

// Queue is the arbiter's Observers.Queue.
func (m *Metrics) Queue(n int, resident string) {
	m.queue.Set(float64(n))
	if resident != "" {
		m.resident.WithLabelValues(resident).Set(1)
	}
}
