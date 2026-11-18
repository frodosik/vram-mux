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
//
// The index is two-level: entries key a conversation's last key to its newest
// file, and prefixes key the chain keys of earlier turns to the file saved for
// that key — the turn boundary. Commit moves a superseded last-key entry into
// prefixes, so a turn boundary's file survives every later commit and a fork
// whose chain shares the prefix finds it. Lookup descends from the request's
// last key and returns the first file whose live token count is at most the
// requesting prompt: the engine reuses a restore only when live ≤ prompt.
// chain.jsonl, one line per Commit, records each committed file's full key
// chain; Load replays it to rebuild the prefix index at startup.
package store

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
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
	chainRe     = regexp.MustCompile(`^\.chain\.jsonl\.tmp$`)
	namespaceRe = regexp.MustCompile(`^([0-9a-f]{12})$`)
)

// Store is the index. It is safe for concurrent use.
type Store struct {
	opt      Options
	mu       sync.Mutex
	entries  map[string]*Entry // by namespace/fingerprint/key
	prefixes map[string]*Entry // chain keys of earlier turns, keyed the same; an entry is the file saved for its key, and alias-to-latest entries are filtered by the prompt length in Lookup
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
	return &Store{opt: opt, entries: map[string]*Entry{}, prefixes: map[string]*Entry{}}
}

func freeSpace(dir string) (int64, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(dir, &st); err != nil {
		return 0, err
	}
	return int64(st.Bavail) * st.Bsize, nil
}

// Load reads the folder: our files into the index, in both name formats; empty
// files, files in an unknown format and leftover temporaries removed; the chain
// side index replayed to rebuild the prefix index, torn lines cleaned; then the
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
		case tempRe.MatchString(name) || chainRe.MatchString(name):
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
	s.rebuildChains()
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

// drop removes e from the index and the disk; mu held. Prefix keys that point
// to this entry go with it: a prefix key must never name a file that is gone.
func (s *Store) drop(e *Entry, why string) {
	delete(s.entries, e.Namespace+"/"+e.Fingerprint+"/"+e.Key)
	for id, p := range s.prefixes {
		if p == e {
			delete(s.prefixes, id)
		}
	}
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
		for id, p := range s.prefixes { // re-key its prefix aliases the same way
			if p == move {
				delete(s.prefixes, id)
				k := strings.TrimPrefix(id, "/"+p.Fingerprint+"/") // the prefix key the alias is keyed by
				s.prefixes[namespace+"/"+p.Fingerprint+"/"+k] = p
			}
		}
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
// promptTokens is the requesting prompt's length. A prefix-key candidate points
// to a file of a newer, possibly diverged chain, so one of more than
// promptTokens tokens is skipped and the descent continues — the engine only
// reuses a restored conversation when its live token count is a prefix of the
// prompt. An exact last-key entry is the conversation's own turn boundary: its
// chain is inside the request, so it passes unfiltered. promptTokens 0: the
// length is unknown, prefix aliases are skipped.
func (s *Store) Lookup(namespace, fingerprint string, keys []string, promptTokens int) (Entry, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := len(keys) - 1; i >= 0; i-- {
		id := namespace + "/" + fingerprint + "/" + keys[i]
		e := s.entries[id]
		if e == nil {
			e = s.prefixes[id]
			if e == nil {
				continue
			}
			if promptTokens > 0 && e.Tokens > promptTokens {
				continue // alias-to-newest: live > the requesting prompt, the engine cannot reuse it
			}
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
// only what came after it. Prefix keys count as their turn-boundary file, so a
// fork's later save is skipped when the shared prefix is already on disk.
func (s *Store) Covered(namespace, fingerprint string, keys []string, tokens int) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, k := range keys {
		id := namespace + "/" + fingerprint + "/" + k
		e := s.entries[id]
		if e == nil {
			e = s.prefixes[id]
		}
		if e != nil && e.Tokens >= tokens {
			return true
		}
	}
	return false
}

// Commit indexes the file the engine just saved as FileName(namespace,
// fingerprint, keys[last], planned), renamed to carry the engine's own token
// count when that differs. The files of the conversation's earlier turns are
// superseded for the last key, and survive as turn-boundaries: each prefix key
// moves to the prefix index pointing to the file that was saved for it, and a
// prefix key with no such file aliases to this entry (Lookup's prompt filter
// skips such an alias when its live count is longer than the prompt).
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
	for _, k := range keys[:len(keys)-1] {
		id := namespace + "/" + fingerprint + "/" + k
		if old := s.entries[id]; old != nil && old.Name() != e.Name() {
			delete(s.entries, id)
			s.prefixes[id] = old // the turn-boundary file keeps its key; the file stays
		} else if s.prefixes[id] == nil {
			s.prefixes[id] = e
		}
	}
	s.entries[namespace+"/"+fingerprint+"/"+key] = e
	s.appendChain(namespace, fingerprint, keys, e)
	return *e, nil
}

// chainLine is one entry of the chain side index: the full key chain of a
// committed conversation and the file that holds it.
type chainLine struct {
	Namespace, Fingerprint, Key string
	Toks                        int
	Name                        string
	Keys                        []string
}

func (s *Store) chainPath() string { return filepath.Join(s.opt.Dir, "chain.jsonl") }

// appendChain records the committed file's full key chain in the side index,
// atomically: the file is read whole, the line lands at its end, and the result
// is renamed into place, so a half-written chain is impossible. Load replays it
// to rebuild the prefix index.
func (s *Store) appendChain(namespace, fingerprint string, keys []string, e *Entry) {
	line, err := json.Marshal(chainLine{Namespace: namespace, Fingerprint: fingerprint, Key: e.Key, Toks: e.Tokens, Name: e.Name(), Keys: keys})
	if err != nil {
		s.opt.Log.Warn("chain line", "err", err)
		return
	}
	prev, err := os.ReadFile(s.chainPath())
	if err != nil && !os.IsNotExist(err) {
		s.opt.Log.Warn("chain read", "err", err)
		return
	}
	tmp := filepath.Join(s.opt.Dir, ".chain.jsonl.tmp")
	out := append(append([]byte(nil), prev...), append(line, '\n')...)
	if err := os.WriteFile(tmp, out, 0o600); err != nil {
		s.opt.Log.Warn("chain temp", "err", err)
		return
	}
	if err := os.Rename(tmp, s.chainPath()); err != nil {
		s.opt.Log.Warn("chain rename", "err", err)
	}
}

// rebuildChains replays the chain side index at startup: every prefix key of a
// line maps to the entry of the file the line names, when that file is in the
// index with that name. Lines of files that are gone are dropped, and the
// cleaned index is written back through the same atomic rename. mu held.
func (s *Store) rebuildChains() {
	raw, err := os.ReadFile(s.chainPath())
	if err != nil {
		return
	}
	var kept []string
	for _, line := range strings.Split(string(raw), "\n") {
		if line == "" {
			continue
		}
		var l chainLine
		if json.Unmarshal([]byte(line), &l) != nil {
			continue // a torn line from a half write: it goes
		}
		e := s.entries[l.Namespace+"/"+l.Fingerprint+"/"+l.Key]
		if e == nil || e.Name() != l.Name {
			continue // the file is gone, renamed, or pruned: the line goes
		}
		for _, k := range l.Keys {
			if k == l.Key {
				continue
			}
			id := l.Namespace + "/" + l.Fingerprint + "/" + k
			if s.prefixes[id] == nil {
				s.prefixes[id] = e
			}
		}
		kept = append(kept, line)
	}
	if len(kept) > 0 {
		tmp := filepath.Join(s.opt.Dir, ".chain.jsonl.tmp")
		out := strings.Join(kept, "\n") + "\n"
		if err := os.WriteFile(tmp, []byte(out), 0o600); err == nil {
			if err := os.Rename(tmp, s.chainPath()); err != nil {
				s.opt.Log.Warn("chain rename", "err", err)
			}
		}
	}
}

// Touch marks e used now (a restore), in entries for a last-key hit and in the
// prefix index for a prefix hit.
func (s *Store) Touch(e Entry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	id := e.Namespace + "/" + e.Fingerprint + "/" + e.Key
	cur := s.entries[id]
	if cur == nil {
		cur = s.prefixes[id]
	}
	if cur != nil && cur.Name() == e.Name() {
		cur.Used = s.opt.Now()
		_ = os.Chtimes(filepath.Join(s.opt.Dir, cur.Name()), cur.Used, cur.Used)
	}
}

// Forget drops e from the index and the disk: from entries for a last-key hit,
// and from every prefix key that points to it.
func (s *Store) Forget(e Entry, why string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	id := e.Namespace + "/" + e.Fingerprint + "/" + e.Key
	cur := s.entries[id]
	if cur == nil {
		cur = s.prefixes[id]
	}
	if cur != nil && cur.Name() == e.Name() {
		s.drop(cur, why)
	}
}
