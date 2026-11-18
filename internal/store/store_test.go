package store

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const fp = "0123456789ab"

// ns is a route namespace, the 12-hex prefix a (backend, model) route carries.
const ns = "111111111111"

func key(i int) string { return fmt.Sprintf("%064x", i) }

func name(i, tokens int) string { return FileName(ns, fp, key(i), tokens) }

func write(t *testing.T, dir, name string, size int, mtime time.Time) {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, make([]byte, size), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(p, mtime, mtime); err != nil {
		t.Fatal(err)
	}
}

func exists(dir, name string) bool { _, err := os.Stat(filepath.Join(dir, name)); return err == nil }

func newStore(t *testing.T, maxBytes, free int64) (*Store, string, map[string]int) {
	t.Helper()
	dir := t.TempDir()
	evicted := map[string]int{}
	s := New(Options{Dir: dir, MaxBytes: maxBytes, MaxAge: 7 * 24 * time.Hour, MinFree: 100,
		OnEvict:   func(why string, _ int64) { evicted[why]++ },
		Log:       slog.New(slog.NewTextHandler(io.Discard, nil)),
		FreeSpace: func(string) (int64, error) { return free, nil }})
	return s, dir, evicted
}

func TestLoadCleansUp(t *testing.T) {
	s, dir, _ := newStore(t, 1000, 1<<40)
	now := time.Now()
	write(t, dir, name(1, 20000), 10, now.Add(-8*24*time.Hour)) // past the age limit
	write(t, dir, name(2, 20000), 10, now.Add(-time.Hour))
	write(t, dir, name(3, 20000), 0, now)                        // empty: a broken write
	write(t, dir, "dc-"+key(4)+"-20000.session", 10, now)        // an older format
	write(t, dir, ".dc-abc.1234.tmp", 10, now.Add(-2*time.Hour)) // a dead save's temporary
	write(t, dir, ".dc-def.5678.tmp", 10, now)                   // maybe a save in progress
	write(t, dir, "saved-by-hand.bin", 10, now.Add(-30*24*time.Hour))
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	for file, want := range map[string]bool{
		name(1, 20000): false, name(2, 20000): true, name(3, 20000): false,
		"dc-" + key(4) + "-20000.session": false, ".dc-abc.1234.tmp": false,
		".dc-def.5678.tmp": true, "saved-by-hand.bin": true,
	} {
		if exists(dir, file) != want {
			t.Errorf("%s: exists=%v, want %v", file, !want, want)
		}
	}
	if n, _ := s.Stats(); n != 1 {
		t.Errorf("%d files indexed, want 1", n)
	}
}

func TestLoadIndexesLegacyNames(t *testing.T) {
	s, dir, _ := newStore(t, 1<<30, 1<<40)
	now := time.Now()
	write(t, dir, FileName("", fp, key(1), 20000), 10, now)
	write(t, dir, FileName(ns, fp, key(2), 20000), 10, now)
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	if n, bytes := s.Stats(); n != 2 || bytes != 20 {
		t.Errorf("legacy and namespaced files: %d files %d bytes, want 2 20", n, bytes)
	}
	if _, ok := s.Lookup("", fp, []string{key(1)}, 20000); !ok {
		t.Error("the legacy file is not found in the legacy namespace")
	}
	if _, ok := s.Lookup(ns, fp, []string{key(2)}, 20000); !ok {
		t.Error("the namespaced file is not found in its namespace")
	}
}

func TestKeepFingerprintIsScopedPerRoute(t *testing.T) {
	s, dir, evicted := newStore(t, 1<<30, 1<<40)
	now := time.Now()
	strata, llama := NamespaceOf("strata", "flash-next"), NamespaceOf("llama", "swift-27b")
	write(t, dir, FileName(strata, fp, key(1), 20000), 10, now)
	write(t, dir, FileName(strata, "ffffffffffff", key(2), 20000), 10, now)
	write(t, dir, FileName(llama, "eeeeeeeeeeee", key(3), 20000), 10, now)
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	// The 27B parent learns its own fingerprint: only its route's stale files go.
	s.KeepFingerprint(llama, "eeeeeeeeeeee")
	if !exists(dir, FileName(strata, fp, key(1), 20000)) || !exists(dir, FileName(strata, "ffffffffffff", key(2), 20000)) {
		t.Error("learning the 27B fingerprint evicted flash-next's sessions")
	}
	if evicted["fingerprint"] != 0 {
		t.Errorf("nothing of llama's route was evicted: %v", evicted)
	}
	// Its route's other fingerprints go, and only those.
	s.KeepFingerprint(llama, "dddddddddddd")
	if !exists(dir, FileName(strata, fp, key(1), 20000)) || exists(dir, FileName(llama, "eeeeeeeeeeee", key(3), 20000)) {
		t.Errorf("KeepFingerprint kept the wrong files: %v", evicted)
	}
}

func TestAdoptMovesLegacyFilesIntoTheirRoute(t *testing.T) {
	s, dir, _ := newStore(t, 1<<30, 1<<40)
	now := time.Now()
	strata := NamespaceOf("strata", "flash-next")
	write(t, dir, FileName("", fp, key(1), 20000), 10, now)
	write(t, dir, FileName("", "ffffffffffff", key(2), 20000), 10, now)
	write(t, dir, FileName(NamespaceOf("llama", "swift-27b"), "dddddddddddd", key(3), 20000), 10, now)
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	if n := s.Adopt(strata, fp); n != 1 {
		t.Fatalf("Adopt moved %d files, want 1", n)
	}
	if exists(dir, FileName("", fp, key(1), 20000)) {
		t.Error("the legacy file is still under its old name")
	}
	moved := FileName(strata, fp, key(1), 20000)
	if !exists(dir, moved) {
		t.Fatalf("%s is gone", moved)
	}
	if _, ok := s.Lookup(strata, fp, []string{key(1)}, 20000); !ok {
		t.Error("the adopted file is not found in its namespace")
	}
	if _, ok := s.Lookup("", fp, []string{key(1)}, 20000); ok {
		t.Error("the adopted file is still found under the legacy namespace")
	}
	if _, ok := s.Lookup("", "ffffffffffff", []string{key(2)}, 20000); !ok {
		t.Error("Adopt took a legacy file of another fingerprint")
	}
}

func TestCapIsGlobalAcrossRoutes(t *testing.T) {
	s, dir, _ := newStore(t, 250, 1<<40)
	now := time.Now()
	strata, llama := NamespaceOf("strata", "flash-next"), NamespaceOf("llama", "swift-27b")
	for i := 1; i <= 4; i++ { // 1 is the least recently used, across both routes
		nsOf := strata
		if i%2 == 0 {
			nsOf = llama
		}
		write(t, dir, FileName(nsOf, fp, key(i), 20000), 100, now.Add(time.Duration(i-10)*time.Minute))
	}
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	if !exists(dir, FileName(strata, fp, key(3), 20000)) || !exists(dir, FileName(llama, fp, key(4), 20000)) ||
		exists(dir, FileName(strata, fp, key(2), 20000)) {
		t.Fatal("the cap kept the wrong files across routes")
	}
}

func TestCapAndLRU(t *testing.T) {
	s, dir, _ := newStore(t, 250, 1<<40)
	now := time.Now()
	for i := 1; i <= 4; i++ { // 1 is the least recently used
		write(t, dir, name(i, 20000), 100, now.Add(time.Duration(i-10)*time.Minute))
	}
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	if !exists(dir, name(3, 20000)) || !exists(dir, name(4, 20000)) || exists(dir, name(2, 20000)) {
		t.Fatal("the cap kept the wrong files")
	}
	s.Touch(Entry{Namespace: ns, Fingerprint: fp, Key: key(3), Tokens: 20000}) // 3 is now the most recent
	if !s.Reserve(100) || exists(dir, name(4, 20000)) || !exists(dir, name(3, 20000)) {
		t.Fatal("Reserve evicted the wrong file")
	}
	if s.Reserve(300) {
		t.Error("a file bigger than the cap fits")
	}
}

func TestDiskFloorWins(t *testing.T) {
	s, dir, evicted := newStore(t, 1<<30, 0)
	s.opt.MinFree = 1000
	for i := 1; i <= 3; i++ {
		write(t, dir, name(i, 20000), 100, time.Now().Add(time.Duration(i)*time.Second))
	}
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	s.opt.FreeSpace = func(string) (int64, error) { return 1050, nil } // needs 50 more: the oldest goes
	if !s.Reserve(100) || exists(dir, name(1, 20000)) || !exists(dir, name(2, 20000)) {
		t.Fatal("wrong eviction for the disk floor")
	}
	s.opt.FreeSpace = func(string) (int64, error) { return 500, nil } // no room even with every file gone
	if s.Reserve(100) {
		t.Fatal("saved below the disk floor")
	}
	if n, _ := s.Stats(); n != 2 || evicted["disk"] != 1 {
		t.Errorf("a hopeless reserve evicted files: %d left, %v", n, evicted)
	}
	s.opt.FreeSpace = func(string) (int64, error) { return 0, os.ErrPermission }
	if s.Reserve(1) {
		t.Error("statfs failing still reserved")
	}
}

func TestAnyFingerprint(t *testing.T) {
	s, dir, _ := newStore(t, 1<<30, 1<<40)
	if s.AnyFingerprint(ns) != "" {
		t.Error("an empty store gave a fingerprint")
	}
	now := time.Now()
	write(t, dir, FileName(ns, "aaaaaaaaaaaa", key(1), 20000), 10, now.Add(-2*time.Hour))
	write(t, dir, FileName(ns, fp, key(2), 20000), 10, now.Add(-time.Hour))
	write(t, dir, FileName("222222222222", fp, key(3), 20000), 10, now) // another route, the newest
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	if got := s.AnyFingerprint(ns); got != fp {
		t.Errorf("AnyFingerprint = %q, want the most recently used file of this route", got)
	}
	s.Touch(Entry{Namespace: ns, Fingerprint: "aaaaaaaaaaaa", Key: key(1), Tokens: 20000})
	if got := s.AnyFingerprint(ns); got != "aaaaaaaaaaaa" {
		t.Errorf("After a Touch, AnyFingerprint = %q, want the file just used", got)
	}
	if got := s.AnyFingerprint("222222222222"); got != fp {
		t.Errorf("the other route's AnyFingerprint = %q, want %q", got, fp)
	}
}

func TestAnyFingerprintTie(t *testing.T) {
	s, dir, _ := newStore(t, 1<<30, 1<<40)
	now := time.Now()
	write(t, dir, FileName(ns, "aaaaaaaaaaaa", key(1), 20000), 10, now)
	write(t, dir, FileName(ns, "bbbbbbbbbbbb", key(2), 20000), 10, now)
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 20; i++ { // map order is randomised: the tie must break the same way
		if got := s.AnyFingerprint(ns); got != "bbbbbbbbbbbb" {
			t.Fatalf("AnyFingerprint = %q, want the higher file name for the same mtime", got)
		}
	}
}

func TestCommitLookupAndCorruptFiles(t *testing.T) {
	s, dir, _ := newStore(t, 1<<30, 1<<40)
	t1 := []string{key(1)}
	t2 := []string{key(1), key(2), key(3)}
	t3 := []string{key(1), key(2), key(3), key(4), key(5)}

	write(t, dir, name(1, 20000), 10, time.Now())
	if _, err := s.Commit(ns, fp, t1, 20000, 20000); err != nil {
		t.Fatal(err)
	}
	if e, ok := s.Lookup(ns, fp, t3, 40000); !ok || e.Key != key(1) {
		t.Fatal("turn 3 does not find turn 1's file")
	}
	if e, ok := s.Lookup(ns, fp, t3, 10000); !ok || e.Key != key(1) {
		t.Error("the conversation's own turn boundary must pass the prompt filter")
	}
	if _, ok := s.Lookup(ns, "ffffffffffff", t3, 40000); ok {
		t.Error("found under another fingerprint")
	}
	if _, ok := s.Lookup("222222222222", fp, t3, 40000); ok {
		t.Error("found in another route's namespace")
	}
	if !s.Covered(ns, fp, t2, 20000) || s.Covered(ns, fp, t2, 20001) {
		t.Error("Covered is wrong")
	}

	// The engine counted 30123 tokens where 30000 were planned: renamed, and turn 1's
	// file survives as a turn-boundary, its key in the prefix index
	write(t, dir, name(3, 30000), 10, time.Now())
	if e, err := s.Commit(ns, fp, t2, 30000, 30123); err != nil || e.Tokens != 30123 || !exists(dir, name(3, 30123)) {
		t.Fatalf("commit with another count: %+v %v", e, err)
	}
	if !exists(dir, name(1, 20000)) {
		t.Error("turn 1's file did not survive as a turn-boundary")
	}
	if e, ok := s.Lookup(ns, fp, t2, 25000); !ok || e.Key != key(3) || e.Tokens != 30123 {
		t.Errorf("the conversation's own last key: %+v %v, want turn 2's file", e, ok)
	}
	// Prefix aliases are filtered by the prompt: key(2) points at the turn 2 file
	// of 30123 tokens, restored only by a prompt that can hold it.
	if _, ok := s.Lookup(ns, fp, []string{key(2), key(9)}, 25000); ok {
		t.Error("a prefix alias of a longer file was restored")
	}
	if e, ok := s.Lookup(ns, fp, []string{key(2), key(9)}, 31000); !ok || e.Key != key(3) || e.Tokens != 30123 {
		t.Errorf("a prefix alias at a long-enough prompt: %+v %v, want turn 2's file", e, ok)
	}
	if _, err := s.Commit(ns, fp, t3, 1, 1); err == nil {
		t.Error("committed a file the engine did not write")
	}

	// truncated behind our back: dropped on lookup, and the descent lands on turn 1's file
	if err := os.Truncate(filepath.Join(dir, name(3, 30123)), 5); err != nil {
		t.Fatal(err)
	}
	if e, ok := s.Lookup(ns, fp, t3, 40000); !ok || e.Key != key(1) || exists(dir, name(3, 30123)) {
		t.Errorf("a changed file was restored, or the descent lost turn 1's file: %+v %v", e, ok)
	}
}

// A fork whose chain reaches an earlier turn finds that turn's own file, the
// files of turn boundaries survive every later commit, and a restart rebuilds
// the prefix index from the chain side index.
func TestPrefixIndexSurvivesCommitsAndRestart(t *testing.T) {
	s, dir, _ := newStore(t, 1<<30, 1<<40)
	t1 := []string{key(1)}
	t2 := []string{key(1), key(2), key(3)}
	t3 := []string{key(1), key(2), key(3), key(4), key(5)}
	t4 := append(append([]string(nil), t3...), key(6), key(7))

	write(t, dir, name(1, 20000), 10, time.Now())
	if _, err := s.Commit(ns, fp, t1, 20000, 20000); err != nil {
		t.Fatal(err)
	}
	write(t, dir, name(3, 30000), 20, time.Now())
	if _, err := s.Commit(ns, fp, t2, 30000, 30000); err != nil {
		t.Fatal(err)
	}
	write(t, dir, name(5, 40000), 30, time.Now())
	if _, err := s.Commit(ns, fp, t3, 40000, 40000); err != nil {
		t.Fatal(err)
	}

	// Every turn boundary's file survives the commits after it
	if !exists(dir, name(1, 20000)) || !exists(dir, name(3, 30000)) || !exists(dir, name(5, 40000)) {
		t.Fatal("a turn-boundary file did not survive the next commit")
	}
	// A fork reaches the newest turn boundary: its own last-key entry passes
	// unfiltered
	if e, ok := s.Lookup(ns, fp, t4, 35000); !ok || e.Key != key(5) || e.Tokens != 40000 {
		t.Errorf("fork lookup: %+v %v, want turn 3's file", e, ok)
	}
	// A prefix alias of turn 2's file is restored by a prompt that can hold its
	// 30000 tokens, and skipped by a shorter one
	if e, ok := s.Lookup(ns, fp, []string{key(2), key(6)}, 35000); !ok || e.Key != key(3) || e.Tokens != 30000 {
		t.Errorf("fork alias lookup: %+v %v, want turn 2's file", e, ok)
	}
	if _, ok := s.Lookup(ns, fp, []string{key(2), key(6)}, 25000); ok {
		t.Error("a prompt shorter than the alias file restored it")
	}

	// Restart: a fresh store on the same folder rebuilds the prefix index from
	// the chain side index, and the fork finds turn 2's file again
	s2 := New(Options{Dir: dir, MaxBytes: 1 << 30, MaxAge: 7 * 24 * time.Hour, MinFree: 100,
		Log: slog.New(slog.NewTextHandler(io.Discard, nil)), FreeSpace: func(string) (int64, error) { return 1 << 40, nil }})
	if err := s2.Load(); err != nil {
		t.Fatal(err)
	}
	if e, ok := s2.Lookup(ns, fp, []string{key(2), key(6)}, 35000); !ok || e.Key != key(3) || e.Tokens != 30000 {
		t.Errorf("after a restart: %+v %v, want turn 2's file", e, ok)
	}
}

// The chain side index is written atomically at Commit; a half-written line is
// cleaned at Load, and the rebuild keeps the chains of files that are on disk.
func TestChainTornLinesCleanAtLoad(t *testing.T) {
	s, dir, _ := newStore(t, 1<<30, 1<<40)
	write(t, dir, name(2, 20000), 10, time.Now())
	if _, err := s.Commit(ns, fp, []string{key(1), key(2)}, 20000, 20000); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "chain.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"`+key(1)+`"`) {
		t.Errorf("the chain side index does not name the prefix key: %s", raw)
	}

	// a half-written line, as a crash leaves it: torn, and no file behind it
	f, err := os.OpenFile(filepath.Join(dir, "chain.jsonl"), os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	torn := `{"Namespace":"` + ns + `","Fingerprint":"` + fp + `","Key":"` + key(4) +
		`","Toks":50000,"Name":"` + name(4, 50000) + `","Keys":["` + key(1) + `","` + key(4) + `")`
	if _, err := f.WriteString(torn); err != nil {
		t.Fatal(err)
	}
	_ = f.Close()

	s2 := New(Options{Dir: dir, MaxBytes: 1 << 30, MaxAge: 7 * 24 * time.Hour, MinFree: 100,
		Log: slog.New(slog.NewTextHandler(io.Discard, nil)), FreeSpace: func(string) (int64, error) { return 1 << 40, nil }})
	if err := s2.Load(); err != nil {
		t.Fatal(err)
	}
	raw, err = os.ReadFile(filepath.Join(dir, "chain.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), `"`+key(4)+`"`) {
		t.Errorf("the torn line survived Load: %s", raw)
	}
	if e, ok := s2.Lookup(ns, fp, []string{key(1), key(5)}, 20000); !ok || e.Tokens != 20000 || e.Name() != name(2, 20000) {
		t.Errorf("after a torn line: %+v %v, want the file of the clean chain", e, ok)
	}
}

// The startup case: the folder is full of legacy unprefixed files from the
// single-route build, and the first save of a namespaced route has to count them
// against the global cap. Legacy files are in the LRU like any other, so the
// least recently used go first, and the disk pressure the owner worried about
// does not appear: the save is admitted after the prune, not refused.
func TestLegacyFilesCountAgainstTheGlobalCap(t *testing.T) {
	s, dir, _ := newStore(t, 300, 1<<40)
	now := time.Now()
	strata := NamespaceOf("strata", "flash-next")
	for i := 1; i <= 3; i++ { // legacy 3-group names, oldest first
		write(t, dir, FileName("", fp, key(i), 20000), 100, now.Add(time.Duration(i-30)*time.Minute))
	}
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	if n, bytes := s.Stats(); n != 3 || bytes != 300 {
		t.Fatalf("startup: %d files %d bytes, want 3 300", n, bytes)
	}
	if !s.Reserve(250) {
		t.Fatal("the first prefixed save was refused")
	}
	write(t, dir, FileName(strata, fp, key(4), 250), 250, now)
	if _, err := s.Commit(strata, fp, []string{key(4)}, 250, 250); err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 3; i++ {
		if exists(dir, FileName("", fp, key(i), 20000)) {
			t.Errorf("legacy file %d survived the cap", i)
		}
	}
	if !exists(dir, FileName(strata, fp, key(4), 250)) {
		t.Error("the prefixed save is not on disk")
	}
	if n, bytes := s.Stats(); n != 1 || bytes != 250 {
		t.Errorf("after the save: %d files %d bytes, want 1 250", n, bytes)
	}
}
