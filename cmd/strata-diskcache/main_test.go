package main

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tfyl/strata-diskcache/internal/strata/stratatest"
)

func TestConfig(t *testing.T) {
	c, err := parse(nil)
	if err != nil {
		t.Fatal(err)
	}
	if c.Listen != ":8080" || c.Upstream != "http://127.0.0.1:8081" || c.MaxGiB != 40 || c.RetentionDays != 7 ||
		c.ParkMinTokens != -1 || c.Log.Output != "stderr" || !c.Log.Compress {
		t.Errorf("defaults: %+v", c)
	}

	t.Setenv("SESSION_DIR", "/var/lib/sessions")
	t.Setenv("MAX_GIB", "10")
	t.Setenv("LOG_COMPRESS", "false")
	c, err = parse([]string{"--max-gib", "20"})
	if err != nil || c.SessionDir != "/var/lib/sessions" || c.MaxGiB != 20 || c.Log.Compress {
		t.Errorf("env, then the flag: %+v %v", c, err)
	}

	for _, args := range [][]string{{"--retention-days", "0"}, {"--upstream", "127.0.0.1:8081"}, {"--log-output", "syslog"}, {"--nope"}} {
		if _, err := parse(args); err == nil {
			t.Errorf("%v accepted", args)
		}
	}
}

func TestLogFileRotates(t *testing.T) {
	dir := t.TempDir()
	c, _ := parse([]string{"--log-output", "both", "--log-file", filepath.Join(dir, "logs", "p.log"),
		"--log-format", "json", "--log-max-size-mib", "1", "--log-max-backups", "2"})
	var stderr bytes.Buffer
	log, closer, err := newLogger(c, &stderr)
	if err != nil {
		t.Fatal(err)
	}
	pad := strings.Repeat("x", 1000)
	for i := 0; i < 3500; i++ { // ~3.5 MiB: three rotations
		log.Info("request", "pad", pad)
	}
	_ = closer.Close()
	if !strings.Contains(stderr.String(), `"msg":"request"`) {
		t.Error("nothing on stderr with --log-output both")
	}
	for deadline := time.Now().Add(5 * time.Second); ; time.Sleep(50 * time.Millisecond) { // gzip runs in the background
		gz, _ := filepath.Glob(filepath.Join(dir, "logs", "p-*.log.gz"))
		plain, _ := filepath.Glob(filepath.Join(dir, "logs", "p-*.log"))
		if len(gz) == 2 && len(plain) == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("want 2 gzipped backups and none plain, got %v %v", gz, plain)
		}
	}
	before, _ := os.Stat(filepath.Join(dir, "logs", "p.log"))
	log2, closer2, _ := newLogger(c, io.Discard) // a restart appends
	log2.Info("after restart")
	_ = closer2.Close()
	if after, _ := os.Stat(filepath.Join(dir, "logs", "p.log")); after.Size() <= before.Size() {
		t.Error("a restart truncated the log file")
	}
}

func TestShutdownMidSave(t *testing.T) {
	dir := t.TempDir()
	f := stratatest.New(dir)
	up := httptest.NewServer(f)
	defer up.Close()
	c, err := parse([]string{"--upstream", up.URL, "--session-dir", dir, "--idle-save-s", "0.02", "--shutdown-s", "5"})
	if err != nil {
		t.Fatal(err)
	}
	pl, _ := net.Listen("tcp", "127.0.0.1:0")
	al, _ := net.Listen("tcp", "127.0.0.1:0")
	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan error, 1)
	go func() { served <- serve(ctx, c, slog.New(slog.NewTextHandler(io.Discard, nil)), pl, al) }()

	f.Set(20000, 0)
	entered := f.Entered()
	f.Update(func(f *stratatest.Fake) { f.SlotDelay = 400 * time.Millisecond })
	resp, err := http.Post("http://"+pl.Addr().String()+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	if got := <-entered; got != "save" {
		t.Fatalf("slot call %q", got)
	}
	cancel() // SIGTERM while the save runs
	select {
	case err := <-served:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("shutdown hung")
	}
	if m, _ := filepath.Glob(filepath.Join(dir, "dc-*.session")); len(m) != 1 {
		t.Errorf("the save did not finish before exit: %v", m)
	}
	if r, err := http.Get("http://" + al.Addr().String() + "/health"); err == nil {
		_ = r.Body.Close()
		t.Error("the admin server is still up")
	}
}

// A route api-key may be "file:PATH": the bearer token is read from the file at
// startup, so it never appears in the process arguments.
func TestRouteAPIKeyFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "keys")
	if err := os.WriteFile(path, []byte("secret1\nsecret2\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	c, err := parse([]string{"--route", "llama|llama/27b|swift|http://127.0.0.1:8001|file:" + path + "|18000"})
	if err != nil {
		t.Fatal(err)
	}
	routes, err := c.routes()
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) != 2 || routes[1].APIKey != "secret1" {
		t.Errorf("api key from file: %+v", routes)
	}

	empty := filepath.Join(dir, "empty")
	if err := os.WriteFile(empty, []byte("\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := parse([]string{"--route", "llama|llama/27b|swift|http://127.0.0.1:8001|file:" + empty + "|18000"}); err == nil {
		t.Error("an empty key file accepted")
	}
	if _, err := parse([]string{"--route", "llama|llama/27b|swift|http://127.0.0.1:8001|file:" + filepath.Join(dir, "missing") + "|18000"}); err == nil {
		t.Error("a missing key file accepted")
	}
	c, err = parse([]string{"--route", "llama|llama/27b|swift|http://127.0.0.1:8001|plain|18000"})
	if err != nil {
		t.Fatal(err)
	}
	if routes, err := c.routes(); err != nil || routes[1].APIKey != "plain" {
		t.Errorf("plain key: %v %v", routes, err)
	}
}

// The boot wiring end to end: the Strata route is resident from the start (the
// --lazy server loads its model itself, the arbiter must not), the free-VRAM gate
// reads the sysfs node the operator pointed at, the route key comes from a file,
// and a request for the 27B model moves the card: save, unload, gate, load.
func TestBootWiring(t *testing.T) {
	dir := t.TempDir()
	card := t.TempDir()
	if err := os.WriteFile(filepath.Join(card, "mem_info_vram_total"), []byte("25165824000"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(card, "mem_info_vram_used"), []byte("18000000000"), 0o600); err != nil {
		t.Fatal(err)
	}
	keys := filepath.Join(dir, "keys")
	if err := os.WriteFile(keys, []byte("llama-key\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	var llamaCalls []string
	var llamaKey string
	llama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		llamaKey = r.Header.Get("Authorization")
		switch r.URL.Path {
		case "/models":
			if r.Method == http.MethodPost {
				llamaCalls = append(llamaCalls, "load")
				_, _ = w.Write([]byte(`{"success":true}`))
				return
			}
			_, _ = w.Write([]byte(`{"models":[{"id":"swift","status":{"value":"loaded"},"args":["--model","swift.gguf"]}]}`))
		case "/models/load":
			llamaCalls = append(llamaCalls, "load")
			_, _ = w.Write([]byte(`{"success":true}`))
		case "/models/unload":
			llamaCalls = append(llamaCalls, "unload")
			_, _ = w.Write([]byte(`{"success":true}`))
		case "/slots/0":
			llamaCalls = append(llamaCalls, "slot "+r.URL.Query().Get("action"))
			_, _ = w.Write([]byte(`{"type":"save","n_saved":100,"size":400}`))
		case "/v1/chat/completions":
			llamaCalls = append(llamaCalls, "generate")
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
		default:
			t.Errorf("llama fake got %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer llama.Close()

	strata := stratatest.New(t.TempDir())
	strata.Set(20000, 0) // the live conversation the swap must save before the card moves
	up := httptest.NewServer(strata)
	defer up.Close()
	entered := strata.Entered()

	c, err := parse([]string{"--upstream", up.URL, "--session-dir", dir, "--strata-model", "flash",
		"--route", "llama|llama/27b|swift|" + llama.URL + "|file:" + keys + "|1",
		"--vram-sysfs", card, "--vram-margin-mib", "1024", "--drain-wait-s", "0.1", "--load-timeout-s", "10",
		"--idle-save-s", "0"})
	if err != nil {
		t.Fatal(err)
	}
	pl, _ := net.Listen("tcp", "127.0.0.1:0")
	al, _ := net.Listen("tcp", "127.0.0.1:0")
	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = serve(ctx, c, slog.New(slog.NewTextHandler(io.Discard, nil)), pl, al) }()
	defer cancel()

	body := strings.NewReader(`{"model":"flash","messages":[{"role":"user","content":"hi"}]}`)
	if resp, err := http.Post("http://"+pl.Addr().String()+"/v1/chat/completions", "application/json", body); err != nil {
		t.Fatal(err)
	} else {
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
	}
	health, err := http.Get("http://" + al.Addr().String() + "/health")
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(health.Body)
	_ = health.Body.Close()
	if !strings.Contains(string(raw), "resident=strata/flash") {
		t.Errorf("boot resident: %q, want the strata route", string(raw))
	}

	body = strings.NewReader(`{"model":"swift","messages":[{"role":"user","content":"hi"}]}`)
	resp, err := http.Post("http://"+pl.Addr().String()+"/v1/chat/completions", "application/json", body)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ = io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if !strings.Contains(string(raw), "ok") {
		t.Fatalf("27B request: %s", string(raw))
	}
	deadline := time.Now().Add(2 * time.Second)
	for len(llamaCalls) < 2 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	var strataSeen []string
	for len(strataSeen) < 2 && time.Now().Before(deadline) {
		select {
		case g := <-entered:
			strataSeen = append(strataSeen, g)
		case <-time.After(200 * time.Millisecond):
		}
	}
	if len(llamaCalls) < 2 || llamaCalls[0] != "load" {
		t.Errorf("llama calls: %v, want the load first", llamaCalls)
	}
	if llamaKey != "Bearer llama-key" {
		t.Errorf("the route key from the file: %q", llamaKey)
	}
	if len(strataSeen) < 1 || strataSeen[0] != "save" {
		t.Errorf("strata calls: %v, want the live session saved before the card moves", strataSeen)
	}
}
