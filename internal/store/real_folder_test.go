package store

import (
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// The deployed build names files dc-<12hex fingerprint>-<64hex key>-<tokens>.session,
// which is the store's legacy form: the real folder from the running stack must load
// into the index, be counted against the cap, and be adopted into its route namespace.
func TestRealFolderFromTheDeployedBuild(t *testing.T) {
	dir := t.TempDir()
	files := []string{
		"dc-eb98614d3db3-2df95693013de4d041c5301f4058efbd8dfd9e96f61c8fcd81c1fa52dfa3e049-180315.session",
		"dc-eb98614d3db3-62418c9a4a9e4907ad8927301d60521bb68329556f5649246a069dc807bdc738-185536.session",
		"dc-eb98614d3db3-c8ebad5fff0b6613dba9eed3ccfe8c499b6ba0af8bd60eab49ea9ed50a4ca614-89411.session",
	}
	for _, f := range files {
		if err := os.WriteFile(filepath.Join(dir, f), make([]byte, 1000), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	s := New(Options{Dir: dir, MaxBytes: 1 << 30, MaxAge: 7 * 24 * time.Hour, MinFree: 0,
		Log: slog.New(slog.NewTextHandler(io.Discard, nil)), FreeSpace: func(string) (int64, error) { return 1 << 40, nil }})
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	if n, b := s.Stats(); n != 3 || b != 3000 {
		t.Fatalf("the deployed folder did not load: %d files %d bytes", n, b)
	}
	key := "2df95693013de4d041c5301f4058efbd8dfd9e96f61c8fcd81c1fa52dfa3e049"
	if _, ok := s.Lookup("", "eb98614d3db3", []string{key}); !ok {
		t.Fatal("the conversation is not found in the legacy namespace")
	}
	ns := NamespaceOf("strata", "qwen3.8-flash-next-iq2_xs")
	if moved := s.Adopt(ns, "eb98614d3db3"); moved != 3 {
		t.Fatalf("Adopt moved %d files, want 3", moved)
	}
	if _, ok := s.Lookup(ns, "eb98614d3db3", []string{key}); !ok {
		t.Fatal("the adopted conversation is not found in the route namespace")
	}
}
