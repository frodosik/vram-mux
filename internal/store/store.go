// Package store indexes the session files the proxy owns and enforces their
// limits: an age limit, a total size cap and a free-space floor.
//
// Files are named dc-<namespace>-<fingerprint>-<key>-<tokens>.session in the
// shared session folder. The namespace names the (backend, model) identity that
// owns the file: "llama/Swift-27B" and "strata/flash-next" share one folder, and
// a file is only ever evicted by the fingerprint of its own namespace, so
// learning one model's fingerprint never deletes another model's sessions. The
// size cap and the LRU are global: they manage total disk across all models.
//
// The fingerprint names the build, model and settings that wrote the file; the
// engine refuses a file of another one, so those go. A file's mtime is its last
// use (save or restore), for the LRU and the age limit. Other files in the
// folder are never touched. The engine writes a file under a temporary name and
// renames it into place, so a file under its final name is complete; a leftover
// temporary (.dc-*.tmp) is from a save that died.
//
// The legacy name, dc-<fingerprint>-<key>-<tokens>.session, has no namespace.
// Load indexes it in namespace ""; Adopt moves a legacy file into a namespace
// when its fingerprint is learned for that route, and renames it on disk.
package store

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Options are the store's folder and limits.
type Options struct {
	Dir      string
	MaxBytes int64
	MaxAge   time.Duration
	MinFree  int64
	OnEvict  func(reason string, bytes int64) // optional
	Log      *slog.Logger
	// for tests; default to statfs and the wall clock
	FreeSpace func(dir string) (int64, error)
	Now       func() time.Time
}

// Entry is one file.
type Entry struct {
	Namespace, Fingerprint, Key string
	Tokens                      int
	Size                        int64
	Used                        time.Time
}

// FileName is the name of the file for this namespace, fingerprint, key and
// token count. An empty namespace is the legacy form.
func FileName(namespace, fingerprint, key string, tokens int) string {
	if namespace == "" {
		return fmt.Sprintf("dc-%s-%s-%d.session", fingerprint, key, tokens)
	}
	return fmt.Sprintf("dc-%s-%s-%s-%d.session", namespace, fingerprint, key, tokens)
}

// Name is the entry's file name.
func (e *Entry) Name() string { return FileName(e.Namespace, e.Fingerprint, e.Key, e.Tokens) }

// NamespaceOf is the 12-hex namespace for a route: a sanitized (backend, model)
// identity, so a route id with punctuation maps to a name-safe, stable prefix.
func NamespaceOf(backend, model string) string {
	sum := sha256.Sum256([]byte(backend + "/" + model))
	return hex.EncodeToString(sum[:6])
}

var (
	nameRe      = regexp.MustCompile(`^dc-([0-9a-f]{12})-([0-9a-f]{64})-([1-9][0-9]*)\.session$`)
	nameReNS    = regexp.MustCompile(`^dc-([0-9a-f]{12})-([0-9a-f]{12})-([0-9a-f]{64})-([1-9][0-9]*)\.session$`)
	tempRe      = regexp.MustCompile(`^\.dc-.*\.tmp$`)
	namespaceRe = regexp.MustCompile(`^([0-9a-f]{12})$`)
)

// Store is the index. It is safe for concurrent use.
type Store struct {
	opt     Options
	mu      sync.Mutex
	entries map[string]*Entry // by namespace/fingerprint/key
}

// New returns an empty store; Load reads the folder.
func New(opt Options) *Store {
	if opt.FreeSpace == nil {
		opt.FreeSpace = freeSpace
	}
	if opt.Now == nil {
		opt.Now = time.Now
	}
	if opt.OnEvict == nil {
		opt.OnEvict = func(string, int64) {}
	}
	if opt.Log == nil {
		opt.Log = slog.Default()
	}
	return &Store{opt: opt, entries: map[string]*Entry{}}
}

func freeSpace(dir string) (int64, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(dir, &st); err != nil {
		return 0, err
	}
	return int64(st.Bavail) * st.Bsize, nil
}

// Load reads the folder: our files into the index, in both name formats; empty
// files, files in an unknown format and leftover temporaries removed; then the
// limits applied.
func (s *Store) Load() error {
	ents, err := os.ReadDir(s.opt.Dir)
	if err != nil {
		return err
	}
	s.mu.Lock()
	for _, d := range ents {
		name := d.Name()
		info, err := d.Info()
		switch {
		case err != nil || !info.Mode().IsRegular():
		case tempRe.MatchString(name):
			if s.opt.Now().Sub(info.ModTime()) > time.Hour { // a recent one may still be written
				s.remove(name, "stale_temporary", info.Size())
			}
		case !strings.HasPrefix(name, "dc-") || !strings.HasSuffix(name, ".session"):
			// not ours
		default:
			ns, fp, key, m := splitName(name)
			if m == nil || info.Size() == 0 {
				s.remove(name, "unreadable", info.Size()) // an older format, or a broken write
				continue
			}
			tokens, _ := strconv.Atoi(m[len(m)-1])
			e := &Entry{Namespace: ns, Fingerprint: fp, Key: key, Tokens: tokens, Size: info.Size(), Used: info.ModTime()}
			if old := s.entries[ns+"/"+fp+"/"+key]; old != nil { // one key, two token counts: keep the longer
				if old.Tokens >= tokens {
					s.remove(name, "duplicate", e.Size)
					continue
				}
				s.drop(old, "duplicate")
			}
			s.entries[ns+"/"+fp+"/"+key] = e
		}
	}
	s.mu.Unlock()
	s.Prune()
	return nil
}

// splitName parses both formats: the namespaced one wins, then the legacy one
// with namespace "". It returns the parts and the match, nil when neither fits.
func splitName(name string) (namespace, fingerprint, key string, m []string) {
	if m := nameReNS.FindStringSubmatch(name); m != nil {
		return m[1], m[2], m[3], m
	}
	if m := nameRe.FindStringSubmatch(name); m != nil {
		return "", m[1], m[2], m
	}
	return "", "", "", nil
}

func (s *Store) remove(name, why string, size int64) {
	if err := os.Remove(filepath.Join(s.opt.Dir, name)); err != nil && !os.IsNotExist(err) {
		s.opt.Log.Warn("remove file", "file", name, "reason", why, "err", err)
		return
	}
	s.opt.Log.Info("evict", "file", name, "reason", why, "bytes", size)
	s.opt.OnEvict(why, size)
}

// drop removes e from the index and the disk; mu held.
func (s *Store) drop(e *Entry, why string) {
	delete(s.entries, e.Namespace+"/"+e.Fingerprint+"/"+e.Key)
	s.remove(e.Name(), why, e.Size)
}

// byAge lists the entries, least recently used first; mu held.
func (s *Store) byAge() []*Entry {
	l := make([]*Entry, 0, len(s.entries))
	for _, e := range s.entries {
		l = append(l, e)
	}
	slices.SortFunc(l, func(a, b *Entry) int { return a.Used.Compare(b.Used) })
	return l
}

// Stats is the number of files and their total size.
func (s *Store) Stats() (files int, bytes int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.entries), s.total()
}

func (s *Store) total() (t int64) {
	for _, e := range s.entries {
		t += e.Size
	}
	return t
}

// Prune applies the age limit and the size cap, oldest use first. Global:
// total disk across every namespace, so one model's cache cannot starve the
// folder for the others.
func (s *Store) Prune() {
	s.mu.Lock()
	defer s.mu.Unlock()
	total := s.total()
	for _, e := range s.byAge() {
		if s.opt.Now().Sub(e.Used) > s.opt.MaxAge {
			total -= e.Size
			s.drop(e, "age")
		} else if total > s.opt.MaxBytes {
			total -= e.Size
			s.drop(e, "size")
		}
	}
}

// KeepFingerprint removes every file of namespace namespace written under
// another fingerprint. Files of other namespaces are never touched: a 27B parent
// learning its fingerprint must not evict the flash-next sessions it shares the
// folder with.
func (s *Store) KeepFingerprint(namespace, fingerprint string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, e := range s.byAge() {
		if e.Namespace == namespace && e.Fingerprint != fingerprint {
			s.drop(e, "fingerprint")
		}
	}
}

// Adopt moves the legacy files of fingerprint fingerprint into namespace,
// renaming them on disk so the scope is recorded in the name and survives a
// restart. It returns how many files moved. Files of another namespace, and
// files of another fingerprint, are not touched.
func (s *Store) Adopt(namespace, fingerprint string) int {
	if namespace == "" || !namespaceRe.MatchString(namespace) {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, e := range s.entries {
		if e.Namespace != "" || e.Fingerprint != fingerprint {
			continue
		}
		old := filepath.Join(s.opt.Dir, e.Name())
		e.Namespace = namespace
		if err := os.Rename(old, filepath.Join(s.opt.Dir, e.Name())); err != nil {
			e.Namespace = ""
			s.opt.Log.Warn("adopt rename", "file", filepath.Base(old), "namespace", namespace, "err", err)
			continue
		}
		move := e // re-key the index entry in place
		delete(s.entries, "/"+e.Fingerprint+"/"+e.Key)
		s.entries[namespace+"/"+e.Fingerprint+"/"+e.Key] = move
		n++
	}
	if n > 0 {
		s.opt.Log.Info("adopt", "namespace", namespace, "fingerprint", fingerprint, "files", n)
	}
	return n
}

// AnyFingerprint returns the fingerprint of the most recently used session file
// of namespace, or "".
func (s *Store) AnyFingerprint(namespace string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var best *Entry
	for _, e := range s.entries {
		if e.Namespace != namespace {
			continue
		}
		if best == nil || e.Used.After(best.Used) || (e.Used.Equal(best.Used) && e.Name() > best.Name()) {
			best = e
		}
	}
	if best != nil {
		return best.Fingerprint
	}
	return ""
}

// Reserve makes room for a file of about need bytes: the least recently used
// files go until it fits under the cap and leaves MinFree on the disk. False
// when it cannot fit even with every file of ours gone; then nothing goes.
func (s *Store) Reserve(need int64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	free, err := s.opt.FreeSpace(s.opt.Dir)
	if err != nil {
		s.opt.Log.Warn("free space", "dir", s.opt.Dir, "err", err)
		return false
	}
	total := s.total()
	if need > s.opt.MaxBytes || free+total-need < s.opt.MinFree {
		return false
	}
	for _, e := range s.byAge() {
		if total+need <= s.opt.MaxBytes && free-need >= s.opt.MinFree {
			break
		}
		why := "size"
		if total+need <= s.opt.MaxBytes {
			why = "disk"
		}
		total, free = total-e.Size, free+e.Size
		s.drop(e, why)
	}
	return true
}

// Lookup finds the file of the longest earlier turn of a conversation whose
// prefix keys are keys (shortest first), in namespace namespace. A file that is
// gone or changed size since it was indexed is dropped and the search goes on.
func (s *Store) Lookup(namespace, fingerprint string, keys []string) (Entry, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := len(keys) - 1; i >= 0; i-- {
		e := s.entries[namespace+"/"+fingerprint+"/"+keys[i]]
		if e == nil {
			continue
		}
		if info, err := os.Stat(filepath.Join(s.opt.Dir, e.Name())); err != nil || info.Size() != e.Size {
			s.drop(e, "changed")
			continue
		}
		return *e, true
	}
	return Entry{}, false
}

// Covered reports whether a file of one of keys in namespace namespace holds at
// least tokens tokens. A file of an earlier turn still counts: the engine reads
// only what came after it.
func (s *Store) Covered(namespace, fingerprint string, keys []string, tokens int) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, k := range keys {
		if e := s.entries[namespace+"/"+fingerprint+"/"+k]; e != nil && e.Tokens >= tokens {
			return true
		}
	}
	return false
}

// Commit indexes the file the engine just saved as FileName(namespace,
// fingerprint, keys[last], planned), renamed to carry the engine's own token
// count when that differs. The files of the conversation's earlier turns are
// superseded.
func (s *Store) Commit(namespace, fingerprint string, keys []string, planned, tokens int) (Entry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	key := keys[len(keys)-1]
	e := &Entry{Namespace: namespace, Fingerprint: fingerprint, Key: key, Tokens: planned}
	if tokens > 0 && tokens != planned {
		from := filepath.Join(s.opt.Dir, e.Name())
		e.Tokens = tokens
		if err := os.Rename(from, filepath.Join(s.opt.Dir, e.Name())); err != nil {
			e.Tokens = planned
		}
	}
	info, err := os.Stat(filepath.Join(s.opt.Dir, e.Name()))
	if err != nil {
		return Entry{}, fmt.Errorf("saved file: %w", err)
	}
	e.Size, e.Used = info.Size(), info.ModTime()
	for _, k := range keys {
		if old := s.entries[namespace+"/"+fingerprint+"/"+k]; old != nil && old.Name() != e.Name() {
			s.drop(old, "superseded")
		}
	}
	s.entries[namespace+"/"+fingerprint+"/"+key] = e
	return *e, nil
}

// Touch marks e used now (a restore).
func (s *Store) Touch(e Entry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if cur := s.entries[e.Namespace+"/"+e.Fingerprint+"/"+e.Key]; cur != nil {
		cur.Used = s.opt.Now()
		_ = os.Chtimes(filepath.Join(s.opt.Dir, cur.Name()), cur.Used, cur.Used)
	}
}

// Forget drops e from the index and the disk.
func (s *Store) Forget(e Entry, why string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if cur := s.entries[e.Namespace+"/"+e.Fingerprint+"/"+e.Key]; cur != nil && cur.Name() == e.Name() {
		s.drop(cur, why)
	}
}
