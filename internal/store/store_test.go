package store

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
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

// diskFiles is the folder as the disk reports it: every regular dc-*.session
// whose name the store can index, by name and size. chain.jsonl, the engine's
// temporaries, and files of other programs are not the store's. A dc-*.session
// whose name the store cannot index is listed in unparseable: Load deletes
// those, so one surviving means the index never owned it.
func diskFiles(t *testing.T, dir string) (map[string]int64, []string) {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Errorf("the folder is unreadable: %v", err)
		return map[string]int64{}, nil
	}
	disk, unparseable := map[string]int64{}, []string(nil)
	for _, d := range ents {
		name := d.Name()
		info, err := d.Info()
		if err != nil || !info.Mode().IsRegular() {
			continue
		}
		if !strings.HasPrefix(name, "dc-") || !strings.HasSuffix(name, ".session") {
			continue
		}
		if _, _, _, m := splitName(name); m == nil {
			unparseable = append(unparseable, name)
			continue
		}
		disk[name] = info.Size()
	}
	return disk, unparseable
}

// trackedNames is the index as the store reports it: one entry per file it owns,
// the last-key entries in entries and the turn-boundary files held by the prefix
// index, deduped.
func trackedNames(s *Store) map[string]int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := map[string]int64{}
	for _, e := range s.all() {
		out[e.Name()] = e.Size
	}
	return out
}

// assertDiskMatchesStats is the store's invariant: the index owns every session
// file in the folder, every file it owns is on disk at the size it counts, and
// the counted total is under the cap. A file dropped from entries, prefixes, or
// totalBytes without unlinking it fails here, and so does a byte total the cap
// never saw — the production leak: 569 files on disk, 39 counted, 712 GiB
// against an 85.9 GB cap.
func assertDiskMatchesStats(t *testing.T, dir string, s *Store) {
	t.Helper()
	disk, unparseable := diskFiles(t, dir)
	if len(unparseable) > 0 {
		t.Errorf("unparseable session files on disk: %v", unparseable)
	}
	index := trackedNames(s)
	for name, size := range disk {
		if _, ok := index[name]; !ok {
			t.Errorf("orphan: %s (%d bytes) is on disk and not in the index", name, size)
		}
	}
	for name, size := range index {
		info, err := os.Stat(filepath.Join(dir, name))
		if err != nil {
			t.Errorf("%s is in the index and not on disk: %v", name, err)
			continue
		}
		if info.Size() != size {
			t.Errorf("%s: the index counts %d bytes, disk has %d", name, size, info.Size())
		}
	}
	var bytes int64
	for _, size := range disk {
		bytes += size
	}
	files, counted := s.Stats()
	if files != len(disk) {
		t.Errorf("the index counts %d files, disk has %d", files, len(disk))
	}
	if counted != bytes {
		t.Errorf("the index counts %d bytes, disk has %d", counted, bytes)
	}
	if counted > s.opt.MaxBytes {
		t.Errorf("%d bytes counted exceeds the cap %d", counted, s.opt.MaxBytes)
	}
}

// engineSave runs one save exactly as cache.save does: reserve the room, write
// the file under the planned name, index it. The invariant is checked after every
// step, so a leak fails at the save that made it.
func engineSave(t *testing.T, s *Store, dir string, keys []string, tokens, size int, when time.Time) Entry {
	t.Helper()
	e := reserveWriteCommit(t, s, dir, keys, tokens, size, when)
	if e.Tokens > 0 {
		assertDiskMatchesStats(t, dir, s)
	}
	return e
}

// reserveWriteCommit is the same save without the invariant check: the window
// between the engine writing and Commit indexing is a file the index does not
// own yet, so parallel saves check the folder once, when they are all done.
func reserveWriteCommit(t *testing.T, s *Store, dir string, keys []string, tokens, size int, when time.Time) Entry {
	t.Helper()
	name := FileName(ns, fp, keys[len(keys)-1], tokens)
	if !s.Reserve(int64(size)) {
		return Entry{}
	}
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, make([]byte, size), 0o600); err != nil {
		t.Errorf("write %s: %v", name, err)
		return Entry{}
	}
	if err := os.Chtimes(p, when, when); err != nil {
		t.Errorf("chtimes %s: %v", name, err)
		return Entry{}
	}
	e, err := s.Commit(ns, fp, keys, tokens, tokens)
	if err != nil {
		t.Errorf("commit %s: %v", name, err)
		return Entry{}
	}
	return e
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
	for i := 0; i < 20; i++ { // map order is randomized: the tie must break the same way
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
	// Covered looks only at the conversation's own last key: t2's key(2) has no
	// file, so no count covers it, and turn 1's file covers exactly its own count.
	if s.Covered(ns, fp, t2, 1) || !s.Covered(ns, fp, t1, 20000) || s.Covered(ns, fp, t1, 20001) {
		t.Error("Covered must look only at the conversation's own last key")
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
	// A chain key that was never terminal names no file: key(2) resolves to
	// nothing at any prompt, and the descent reaches turn 1's boundary, whose
	// file is restored only by a prompt that can hold its 20000 tokens.
	if _, ok := s.Lookup(ns, fp, []string{key(2), key(9)}, 31000); ok {
		t.Error("a key that was never terminal resolved to a longer file")
	}
	if e, ok := s.Lookup(ns, fp, []string{key(1), key(2), key(9)}, 25000); !ok || e.Key != key(1) || e.Tokens != 20000 {
		t.Errorf("the descent to turn 1's boundary: %+v %v, want turn 1's file", e, ok)
	}
	if _, ok := s.Lookup(ns, fp, []string{key(1), key(2), key(9)}, 15000); ok {
		t.Error("a prompt shorter than the boundary file restored it")
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
	// key(2) was never a terminal key, so it names no file at any prompt. key(3),
	// the turn-boundary file of 30000 tokens, is restored by a prompt that can
	// hold it and skipped by a shorter one.
	if _, ok := s.Lookup(ns, fp, []string{key(2), key(6)}, 35000); ok {
		t.Error("a key that was never terminal resolved to a deeper file")
	}
	if e, ok := s.Lookup(ns, fp, []string{key(3), key(6)}, 35000); !ok || e.Key != key(3) || e.Tokens != 30000 {
		t.Errorf("fork boundary lookup: %+v %v, want turn 2's file", e, ok)
	}
	if _, ok := s.Lookup(ns, fp, []string{key(3), key(6)}, 25000); ok {
		t.Error("a prompt shorter than the boundary file restored it")
	}

	// Restart: a fresh store on the same folder rebuilds the prefix index from
	// the chain side index, and the fork finds turn 2's file again
	s2 := New(Options{Dir: dir, MaxBytes: 1 << 30, MaxAge: 7 * 24 * time.Hour, MinFree: 100,
		Log: slog.New(slog.NewTextHandler(io.Discard, nil)), FreeSpace: func(string) (int64, error) { return 1 << 40, nil }})
	if err := s2.Load(); err != nil {
		t.Fatal(err)
	}
	if e, ok := s2.Lookup(ns, fp, []string{key(3), key(6)}, 35000); !ok || e.Key != key(3) || e.Tokens != 30000 {
		t.Errorf("after a restart: %+v %v, want turn 2's file", e, ok)
	}
	if _, ok := s2.Lookup(ns, fp, []string{key(2), key(6)}, 35000); ok {
		t.Error("the restart rebuilt an alias for a key that was never terminal")
	}
}

// The measured session-save bug: a new conversation shares the opening chain keys
// of an older, much deeper one — the same system prompt and the same first
// messages, a 3,309-token root inside an 116k conversation — and every shared key
// resolves to the deep session's file. Covered must consult only the
// conversation's own last key, so the 20k and 43k conversations are never marked
// on_disk and their saves run.
func TestNewSessionSharingRootKeyIsNotCovered(t *testing.T) {
	s, dir, _ := newStore(t, 1<<30, 1<<40)
	deep := []string{key(1), key(2), key(3), key(4)}
	write(t, dir, name(4, 116000), 60, time.Now())
	if _, err := s.Commit(ns, fp, deep, 116000, 116000); err != nil {
		t.Fatal(err)
	}
	// The new session: its launch boundary at 20,802 tokens, then 43,278, both
	// sharing key(1) and key(2) with the deep session.
	launch := []string{key(1), key(2), key(9)}
	grown := []string{key(1), key(2), key(9), key(10)}
	if s.Covered(ns, fp, launch, 20802-4096) || s.Covered(ns, fp, grown, 43278-4096) {
		t.Error("sharing an ancestor key with a deep session marked the new conversation on_disk")
	}
	// Its own saves run, each keyed by its own last key
	write(t, dir, name(9, 20802), 12, time.Now())
	if _, err := s.Commit(ns, fp, launch, 20802, 20802); err != nil {
		t.Fatal(err)
	}
	if !exists(dir, name(9, 20802)) || !s.Covered(ns, fp, launch, 20802-4096) {
		t.Error("the 20k conversation's own boundary is not on disk and covering it")
	}
	write(t, dir, name(10, 43278), 24, time.Now())
	if _, err := s.Commit(ns, fp, grown, 43278, 43278); err != nil {
		t.Fatal(err)
	}
	if !exists(dir, name(10, 43278)) || !s.Covered(ns, fp, grown, 43278-4096) {
		t.Error("the 43k conversation's own boundary is not on disk and covering it")
	}
	// The deep session's file is still its own, and a chain reaching key(4) finds it
	if e, ok := s.Lookup(ns, fp, []string{key(1), key(2), key(3), key(4), key(11)}, 120000); !ok || e.Tokens != 116000 {
		t.Errorf("the deep conversation's own boundary: %+v %v, want its 116000-token file", e, ok)
	}
}

// The prefix index never aliases forward. A lookup for a chain that shares
// key(1) with a deep session finds only files whose own saved count is at or
// below the prompt, and a restart rebuilds the same rule.
func TestPrefixIndexNeverAliasesLongerDescendant(t *testing.T) {
	s, dir, _ := newStore(t, 1<<30, 1<<40)
	deep := []string{key(1), key(2), key(3), key(4)}
	write(t, dir, name(4, 116000), 60, time.Now())
	if _, err := s.Commit(ns, fp, deep, 116000, 116000); err != nil {
		t.Fatal(err)
	}
	// The new session's 20k prompt shares key(1): the deep file is not restored
	if _, ok := s.Lookup(ns, fp, []string{key(1), key(7)}, 20000); ok {
		t.Error("a 20k prompt restored the 116k file through a shared key")
	}
	// The session's own boundary is what its fork finds
	write(t, dir, name(7, 19000), 12, time.Now())
	if _, err := s.Commit(ns, fp, []string{key(1), key(7)}, 19000, 19000); err != nil {
		t.Fatal(err)
	}
	if e, ok := s.Lookup(ns, fp, []string{key(1), key(7), key(8)}, 20000); !ok || e.Name() != name(7, 19000) {
		t.Errorf("the fork of the 20k session: %+v %v, want its own 19000-token file", e, ok)
	}
	// A restart rebuilds the prefix index with the same rule
	s2 := New(Options{Dir: dir, MaxBytes: 1 << 30, MaxAge: 7 * 24 * time.Hour, MinFree: 100,
		Log: slog.New(slog.NewTextHandler(io.Discard, nil)), FreeSpace: func(string) (int64, error) { return 1 << 40, nil }})
	if err := s2.Load(); err != nil {
		t.Fatal(err)
	}
	if _, ok := s2.Lookup(ns, fp, []string{key(1), key(9)}, 20000); ok {
		t.Error("the restart aliased key(1) forward to a longer file")
	}
	if e, ok := s2.Lookup(ns, fp, []string{key(1), key(7), key(8)}, 20000); !ok || e.Name() != name(7, 19000) {
		t.Errorf("after a restart: %+v %v, want the session's own file", e, ok)
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
	// key(1) was never terminal, so the rebuild gives it no file, and a chain
	// reaching key(2), the terminal key of the clean line, finds that file.
	if _, ok := s2.Lookup(ns, fp, []string{key(1), key(5)}, 20000); ok {
		t.Error("the rebuild aliased a key that was never terminal")
	}
	if e, ok := s2.Lookup(ns, fp, []string{key(1), key(2), key(5)}, 20000); !ok || e.Tokens != 20000 || e.Name() != name(2, 20000) {
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

// The production leak: one conversation saved every 2-3 minutes, 150875 ->
// 152461 -> 154050 -> 155032 tokens. A save is keyed by the last chain key, so
// repeated saves of the same conversation overwrite the last-key entry with a
// longer file, and the older file of that key stayed on disk with nothing
// counting it. 50 growth saves of one key, 10k -> 155k tokens, must leave
// exactly one file, and the cap must see every byte.
func TestRepeatedGrowthSavesOnTheSameKeyLeaveNoOrphans(t *testing.T) {
	maxBytes := int64(400 << 10)
	s, dir, evicted := newStore(t, maxBytes, 1<<40)
	start := time.Now()
	chain := []string{key(1), key(2), key(3)}
	tokens, written := 10000, int64(0)
	saves := 0
	for i := 0; i < 50; i++ {
		size := 2 * tokens // 2 bytes per token: 50 saves are ~20x the cap
		if e := engineSave(t, s, dir, chain, tokens, size, start.Add(time.Duration(i)*time.Minute)); e.Tokens > 0 {
			saves++
			written += int64(size)
		}
		tokens += 2900
	}
	if saves < 50 {
		t.Fatalf("%d saves ran, want 50", saves)
	}
	if written < 5*maxBytes {
		t.Fatalf("the saves wrote %d bytes, want over 5x the cap %d", written, maxBytes)
	}
	// One file: the newest save of the key. Every superseded file is unlinked.
	want := FileName(ns, fp, key(3), tokens-2900)
	if !exists(dir, want) {
		t.Errorf("%s is not on disk", want)
	}
	if n, bytes := s.Stats(); n != 1 {
		t.Errorf("%d files in the index, want 1 (the disk has %d bytes)", n, bytes)
	}
	assertDiskMatchesStats(t, dir, s)
	// Every save unlinked exactly one older file of this key, either as the entry
	// it replaced (superseded) or as the oldest under the cap (size).
	if evicted["superseded"]+evicted["size"] != 49 {
		t.Errorf("evictions %v, want one older file of the key unlinked by every save", evicted)
	}
}

// 50 saves, each at its own turn boundary: the files of the earlier turns live
// only in the prefix index, and the cap must count them. A restart loads the
// folder, prunes it under the cap, rebuilds the prefix index, and the disk still
// matches the index.
func TestGrowthAcrossTurnBoundariesSurvivesRestart(t *testing.T) {
	maxBytes := int64(2 << 20)
	s, dir, _ := newStore(t, maxBytes, 1<<40)
	start := time.Now()
	tokens := 10000
	var chain []string
	var turns []struct {
		keys   []string
		tokens int
	}
	for i := 0; i < 50; i++ {
		chain = append(chain, key(2*i+1), key(2*i+2)) // two new chain keys per turn
		engineSave(t, s, dir, chain, tokens, 2*tokens, start.Add(time.Duration(i)*time.Minute))
		turns = append(turns, struct {
			keys   []string
			tokens int
		}{append([]string(nil), chain...), tokens})
		tokens += 2900
	}
	// A fresh store on the same folder: Load indexes every file, replays the
	// chain, and prunes the over-cap total at startup.
	s2 := New(Options{Dir: dir, MaxBytes: maxBytes, MaxAge: 7 * 24 * time.Hour, MinFree: 100,
		Log: slog.New(slog.NewTextHandler(io.Discard, nil)), FreeSpace: func(string) (int64, error) { return 1 << 40, nil }})
	if err := s2.Load(); err != nil {
		t.Fatal(err)
	}
	assertDiskMatchesStats(t, dir, s2)
	if _, bytes := s2.Stats(); bytes > maxBytes {
		t.Errorf("%d bytes after startup, want under the cap %d", bytes, maxBytes)
	}
	// Every turn boundary on disk is what its own key resolves to, and a file
	// that is gone leaves no alias behind: Covered is exactly the disk.
	surviving := 0
	for i, tr := range turns {
		name := FileName(ns, fp, tr.keys[len(tr.keys)-1], tr.tokens)
		onDisk := exists(dir, name)
		if onDisk != s2.Covered(ns, fp, tr.keys, tr.tokens) {
			t.Errorf("turn %d: %s exists=%v, the index covers=%v", i, name, onDisk, s2.Covered(ns, fp, tr.keys, tr.tokens))
		}
		if !onDisk {
			continue
		}
		surviving++
		// A fork behind this boundary finds this file, not a deeper one.
		fork := append(append([]string(nil), tr.keys...), key(1000+i))
		if e, ok := s2.Lookup(ns, fp, fork, tr.tokens); !ok || e.Name() != name {
			t.Errorf("fork behind turn %d: %+v %v, want %s", i, e, ok, name)
		}
	}
	if surviving < 2 {
		t.Errorf("%d turn-boundary files survived, want the newest turns kept", surviving)
	}
}

// A commit whose rename lands on a name the index already owns: the file under
// that name is unlinked with its entry before the rename buries it.
func TestCommitRenameOntoATrackedNameUnlinksIt(t *testing.T) {
	s, dir, evicted := newStore(t, 1<<30, 1<<40)
	now := time.Now()
	turn := []string{key(1), key(2), key(3)}
	// The engine writes under the planned count, Commit renames it to the count
	// the engine reported.
	write(t, dir, name(3, 30000), 30000, now)
	if e, err := s.Commit(ns, fp, turn, 30000, 30123); err != nil || e.Tokens != 30123 {
		t.Fatalf("the first save: %+v %v", e, err)
	}
	assertDiskMatchesStats(t, dir, s)
	// The next save planned 40000, the engine counted 30123: the rename lands on
	// the name the index owns, so that entry and file go first.
	write(t, dir, name(3, 40000), 40000, now.Add(time.Minute))
	e, err := s.Commit(ns, fp, turn, 40000, 30123)
	if err != nil || e.Tokens != 30123 {
		t.Fatalf("the rename onto a tracked name: %+v %v", e, err)
	}
	assertDiskMatchesStats(t, dir, s)
	if n, bytes := s.Stats(); n != 1 || bytes != 40000 {
		t.Errorf("after the rename: %d files %d bytes, want 1 40000", n, bytes)
	}
	if evicted["superseded"] != 1 {
		t.Errorf("evictions %v, want the overwritten file unlinked", evicted)
	}
}

// Startup over the cap: every file is indexed, the total is pruned to the cap at
// Load, and nothing is left on disk outside the index.
func TestLoadPrunesTheFolderOverTheCapAtStartup(t *testing.T) {
	s, dir, evicted := newStore(t, 300, 1<<40)
	now := time.Now()
	for i := 1; i <= 10; i++ { // 10 files, 1000 bytes total, over a 300-byte cap
		write(t, dir, name(i, 10000+i), 100, now.Add(time.Duration(i-100)*time.Minute))
	}
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	assertDiskMatchesStats(t, dir, s)
	if evicted["size"] != 7 {
		t.Errorf("evictions %v, want 7 files pruned at startup", evicted)
	}
}

// One key, several token counts on disk: Load keeps the longest and unlinks the
// duplicates, so a key names one file.
func TestLoadKeepsOneFilePerKey(t *testing.T) {
	s, dir, _ := newStore(t, 1<<30, 1<<40)
	now := time.Now()
	write(t, dir, name(3, 10000), 100, now)
	write(t, dir, name(3, 15000), 150, now.Add(time.Minute))
	write(t, dir, name(3, 12000), 120, now.Add(2*time.Minute))
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	if !exists(dir, name(3, 15000)) {
		t.Error("the longest file of the key did not survive")
	}
	for _, tokens := range []int{10000, 12000} {
		if exists(dir, name(3, tokens)) {
			t.Errorf("the duplicate %d-token file stayed on disk", tokens)
		}
	}
	assertDiskMatchesStats(t, dir, s)
}

// Adopt renames a legacy file onto a name the index already owns: the file under
// that name goes with its entry, and the folder keeps one file per key.
func TestAdoptUnlinksTheFileItsRenameOverwrites(t *testing.T) {
	s, dir, _ := newStore(t, 1<<30, 1<<40)
	now := time.Now()
	strata := NamespaceOf("strata", "flash-next")
	write(t, dir, FileName("", fp, key(1), 20000), 100, now)
	write(t, dir, FileName(strata, fp, key(1), 20000), 100, now.Add(time.Minute))
	write(t, dir, FileName("", fp, key(2), 30000), 100, now.Add(2*time.Minute))
	write(t, dir, FileName(strata, fp, key(2), 30000), 100, now.Add(3*time.Minute))
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	assertDiskMatchesStats(t, dir, s)
	if n := s.Adopt(strata, fp); n != 2 {
		t.Fatalf("Adopt moved %d files, want 2", n)
	}
	for _, k := range []string{key(1), key(2)} {
		if exists(dir, FileName("", fp, k, 20000)) || exists(dir, FileName("", fp, k, 30000)) {
			t.Errorf("the legacy file of %s is still under its old name", k)
		}
	}
	assertDiskMatchesStats(t, dir, s)
	if n, bytes := s.Stats(); n != 2 || bytes != 200 {
		t.Errorf("after Adopt: %d files %d bytes, want 2 200", n, bytes)
	}
	for _, k := range []string{key(1), key(2)} {
		if _, ok := s.Lookup(strata, fp, []string{k}, 30000); !ok {
			t.Errorf("the adopted file of %s is not found", k)
		}
	}
}

// The turn-boundary files the prefix index holds are under the cap: a save that
// needs room evicts them, and the counted total is the disk.
func TestTurnBoundaryFilesAreUnderTheCap(t *testing.T) {
	s, dir, evicted := newStore(t, 250, 1<<40)
	now := time.Now()
	engineSave(t, s, dir, []string{key(1)}, 20000, 100, now)
	engineSave(t, s, dir, []string{key(1), key(2), key(3)}, 30000, 100, now.Add(time.Minute))
	// turn 1's file lives only in the prefix index; its bytes were never counted
	e := engineSave(t, s, dir, []string{key(1), key(2), key(3), key(4), key(5)}, 40000, 100, now.Add(2*time.Minute))
	if e.Tokens == 0 {
		t.Fatal("the third save was refused")
	}
	if exists(dir, name(1, 20000)) {
		t.Error("the oldest turn-boundary file survived the cap")
	}
	if _, ok := s.Lookup(ns, fp, []string{key(1), key(9)}, 25000); ok {
		t.Error("a prefix key points at a file that is gone")
	}
	if evicted["size"] == 0 {
		t.Errorf("evictions %v: the cap never saw a turn-boundary file", evicted)
	}
	assertDiskMatchesStats(t, dir, s)
}

// An aborted save: no file, so Commit refuses and nothing is left to index.
func TestAbortedSaveLeavesNothingToIndex(t *testing.T) {
	s, dir, _ := newStore(t, 1<<30, 1<<40)
	if _, err := s.Commit(ns, fp, []string{key(1), key(2)}, 20000, 20000); err == nil {
		t.Error("a commit with no file on disk succeeded")
	}
	assertDiskMatchesStats(t, dir, s)
	// A save the engine finished, and a count that matches its file: indexed.
	write(t, dir, name(2, 20000), 100, time.Now())
	if e, err := s.Commit(ns, fp, []string{key(1), key(2)}, 20000, 20000); err != nil || e.Size != 100 {
		t.Errorf("the planned file was not indexed: %+v %v", e, err)
	}
	assertDiskMatchesStats(t, dir, s)
}

// Saves run concurrently: two routes' idle and switch saves hit one store. Every
// save reserves, writes, commits, and the folder must match the index at the
// end. -race is what this test runs under.
func TestConcurrentGrowthKeepsDiskAndIndexEqual(t *testing.T) {
	s, dir, _ := newStore(t, 600<<10, 1<<40)
	start := time.Now()
	var wg sync.WaitGroup
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			tokens := 10000 + g*5000
			for i := 0; i < 20; i++ {
				reserveWriteCommit(t, s, dir, []string{key(1), key(2), key(3 + g)}, tokens, 2*tokens,
					start.Add(time.Duration(i)*time.Minute+time.Duration(g)*time.Second))
				tokens += 2900
			}
		}(g)
	}
	wg.Wait()
	assertDiskMatchesStats(t, dir, s)
	if n, bytes := s.Stats(); n == 0 || bytes > s.opt.MaxBytes {
		t.Errorf("%d files %d bytes, want files under the cap %d", n, bytes, s.opt.MaxBytes)
	}
}

// A turn-boundary alias and a last-key file can share a key: evicting the
// boundary must not take the newer entry of that key with it. Deleting the map
// slot by key, not by entry, leaves the newer file on disk with nothing counting
// it — an orphan the cap never sees again.
func TestEvictingATurnBoundaryKeepsTheLastKeyFileOfTheSameKey(t *testing.T) {
	s, dir, evicted := newStore(t, 300, 1<<40)
	now := time.Now()
	engineSave(t, s, dir, []string{key(1)}, 10000, 100, now)                          // A at key(1)
	engineSave(t, s, dir, []string{key(1), key(2)}, 15000, 100, now.Add(time.Minute)) // B at key(2), and A becomes a boundary
	engineSave(t, s, dir, []string{key(1)}, 20000, 100, now.Add(2*time.Minute))       // C at key(1): entries[key(1)] is C, prefixes[key(1)] is A
	// 300 bytes on disk, at the cap: a save of 100 needs room, so the oldest goes.
	if !s.Reserve(100) {
		t.Fatal("the save was refused at the cap")
	}
	if exists(dir, name(1, 10000)) {
		t.Error("the turn-boundary file survived the cap")
	}
	if !exists(dir, name(1, 20000)) {
		t.Error("the last-key file of key(1) was evicted with the boundary")
	}
	assertDiskMatchesStats(t, dir, s)
	if evicted["size"] != 1 {
		t.Errorf("evictions %v, want exactly the boundary file", evicted)
	}
	if e, ok := s.Lookup(ns, fp, []string{key(1), key(9)}, 12000); !ok || e.Name() != name(1, 20000) {
		t.Errorf("key(1) after the boundary eviction: %+v %v, want the last-key file", e, ok)
	}
}

// The legacy folder of the deployed build, with a namespaced file of the same key
// already indexed: Adopt unlinks the file its rename overwrites, so one key keeps
// one file and the counts stay the disk.
func TestAdoptCollisionOverTheSameKey(t *testing.T) {
	s, dir, _ := newStore(t, 1<<30, 1<<40)
	now := time.Now()
	strata := NamespaceOf("strata", "flash-next")
	// key(1): the namespaced file has a different token count, so Adopt's map
	// write replaces the entry; key(2): the namespaced name is exactly where the
	// rename lands.
	write(t, dir, FileName("", fp, key(1), 20000), 100, now)
	write(t, dir, FileName(strata, fp, key(1), 40000), 100, now.Add(time.Minute))
	write(t, dir, FileName("", fp, key(2), 30000), 100, now.Add(2*time.Minute))
	write(t, dir, FileName(strata, fp, key(2), 30000), 100, now.Add(3*time.Minute))
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	assertDiskMatchesStats(t, dir, s)
	if n := s.Adopt(strata, fp); n != 2 {
		t.Fatalf("Adopt moved %d files, want 2", n)
	}
	assertDiskMatchesStats(t, dir, s)
	if n, bytes := s.Stats(); n != 2 || bytes != 200 {
		t.Errorf("after Adopt: %d files %d bytes, want 2 200", n, bytes)
	}
	for _, k := range []string{key(1), key(2)} {
		if exists(dir, FileName("", fp, k, 20000)) || exists(dir, FileName("", fp, k, 30000)) {
			t.Errorf("the legacy file of %s is still under its old name", k)
		}
	}
	if !exists(dir, FileName(strata, fp, key(2), 30000)) {
		t.Error("the adopted file of key(2) is not on disk")
	}
	if _, ok := s.Lookup(strata, fp, []string{key(1)}, 20000); !ok {
		t.Error("the adopted file of key(1) is not found")
	}
}
