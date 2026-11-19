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
//
// The invariant: the index owns every file in the folder that is ours, and
// every file it owns is counted. A key names one file; an entry superseded,
// adopted, pruned, forgotten, or pointed at a file that has changed is dropped
// with os.Remove on its file, and the prefix aliases to it go with it. Stats is
// the disk: the cap, the LRU, the age limit and the floor all read every file
// the store tracks, the last-key files in entries and the turn-boundary files
// the prefix index holds, so a file can never leave the accounting and grow on
// disk untracked.
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
// The last-key slot is deleted only when it holds this entry: a turn-boundary
// alias and the last-key file can share a key, and evicting the boundary must
// not drop the newer entry of that key, whose file would then sit on disk with
// nothing counting it.
func (s *Store) drop(e *Entry, why string) {
	id := e.Namespace + "/" + e.Fingerprint + "/" + e.Key
	if s.entries[id] == e {
		delete(s.entries, id)
	}
	for id, p := range s.prefixes {
		if p == e {
			delete(s.prefixes, id)
		}
	}
	s.remove(e.Name(), why, e.Size)
}

// byAge lists the tracked files, least recently used first; mu held.
func (s *Store) byAge() []*Entry {
	l := s.all()
	slices.SortFunc(l, func(a, b *Entry) int { return a.Used.Compare(b.Used) })
	return l
}

// all lists every file the store tracks, once: the last-key entries in entries
// and the turn-boundary files the prefix index holds. A prefix alias points to
// the entry of the file saved when its key was terminal, so the same file can
// be named by a last-key entry and a prefix key at once: it is counted once.
// The size cap, the LRU, the age limit and Stats read this set, so a
// turn-boundary file is under the cap exactly like a last-key file, and the
// count is the number of files on disk. mu held.
func (s *Store) all() []*Entry {
	seen := make(map[*Entry]bool, len(s.entries)+len(s.prefixes))
	out := make([]*Entry, 0, len(s.entries)+len(s.prefixes))
	for _, e := range s.entries {
		if !seen[e] {
			seen[e] = true
			out = append(out, e)
		}
	}
	for _, e := range s.prefixes {
		if !seen[e] {
			seen[e] = true
			out = append(out, e)
		}
	}
	return out
}

// owners lists the tracked entries whose file is name, across both maps, once;
// mu held. The index must never point two entries at one file, and it must
// never overwrite an entry that owns name: the file would stay on disk with
// nothing counting it.
func (s *Store) owners(name string) []*Entry {
	var out []*Entry
	for _, e := range s.all() {
		if e.Name() == name {
			out = append(out, e)
		}
	}
	return out
}

// Stats is the number of files and their total size: what the store tracks is
// what is on disk.
func (s *Store) Stats() (files int, bytes int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.all()), s.total()
}

func (s *Store) total() (t int64) {
	for _, e := range s.all() {
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
// files of another fingerprint, are not touched. A namespaced file already at the
// name the rename lands on, and the entry already keyed by the name the legacy
// file moves to, are unlinked with their entries first: the rename would
// overwrite them, and an entry replaced by the map write would leave its file on
// disk with nothing counting it.
func (s *Store) Adopt(namespace, fingerprint string) int {
	if namespace == "" || !namespaceRe.MatchString(namespace) {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, e := range s.all() { // a legacy file can be a turn-boundary in the prefix index
		if e.Namespace != "" || e.Fingerprint != fingerprint {
			continue
		}
		old := filepath.Join(s.opt.Dir, e.Name())
		moved := FileName(namespace, fingerprint, e.Key, e.Tokens)
		// Two entries cannot own one key: the entry already keyed here goes with
		// its file, and so does any entry whose name the rename lands on.
		if prev := s.entries[namespace+"/"+fingerprint+"/"+e.Key]; prev != nil && prev != e {
			s.drop(prev, "adopt")
		}
		for _, o := range s.owners(moved) {
			s.drop(o, "adopt")
		}
		e.Namespace = namespace
		if err := os.Rename(old, filepath.Join(s.opt.Dir, e.Name())); err != nil {
			e.Namespace = ""
			s.opt.Log.Warn("adopt rename", "file", filepath.Base(old), "namespace", namespace, "err", err)
			continue
		}
		delete(s.entries, "/"+e.Fingerprint+"/"+e.Key)
		s.entries[namespace+"/"+e.Fingerprint+"/"+e.Key] = e // re-key the entry in place
		for id, p := range s.prefixes {                      // re-key its prefix aliases the same way
			if p == e {
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
// of namespace, or "". Files the prefix index holds count too: a route whose
// last-key files were pruned still has its turn-boundary files on disk.
func (s *Store) AnyFingerprint(namespace string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var best *Entry
	for _, e := range s.all() {
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
// promptTokens is the requesting prompt's length. The prefix index holds only
// files saved when their key was terminal, so a prefix entry's count is the live
// count of that turn; the engine reuses a restored conversation only when its
// live count is a prefix of the prompt, so one of more than promptTokens tokens
// is skipped and the descent continues. An exact last-key entry is the
// conversation's own turn boundary: its chain is inside the request, so it passes
// unfiltered. promptTokens 0: the length is unknown, prefix entries are skipped.
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

// Covered reports whether the conversation's own last key has a file of at least
// tokens tokens: its turn-boundary file in entries, or the file saved when that
// key was terminal, which is all the prefix index holds. Ancestor keys are never
// consulted. A file keyed by an ancestor is a shallower conversation, and a
// conversation that only shares an ancestor key with an older, deeper one is not
// covered by it: sharing a 3,300-token system prompt with an 116k conversation
// must not mark a 20k conversation on disk, because the engine cannot read a
// longer file as a prefix of a shorter prompt.
func (s *Store) Covered(namespace, fingerprint string, keys []string, tokens int) bool {
	if len(keys) == 0 {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	id := namespace + "/" + fingerprint + "/" + keys[len(keys)-1]
	e := s.entries[id]
	if e == nil {
		e = s.prefixes[id]
	}
	return e != nil && e.Tokens >= tokens
}

// Commit indexes the file the engine just saved as FileName(namespace,
// fingerprint, keys[last], planned), renamed to carry the engine's own token
// count when that differs. The files of the conversation's earlier turns are
// superseded for the last key, and survive as turn-boundaries: each prefix key
// moves to the prefix index pointing to the file that was saved for it. A prefix
// key with no file of its own stays empty — a prefix key at chain index i may
// only name a file saved when keys[i] was that save's terminal key, never a
// longer descendant file, so Lookup's prompt bound is never fooled by an alias.
//
// The last key is the one key Commit overwrites, and it is where the leak starts.
// A conversation saved again past --regrow-tokens keeps its chain keys, so its
// next save is keyed by the SAME last key with a longer token count: a different
// file name. The older file of that key is superseded and unlinked here, so a key
// names one file and the disk never keeps a state the index forgot. The same rule
// covers the name the rename lands on — a tracked file under it is unlinked
// before the rename buries it — and a file the engine wrote that indexing cannot
// confirm, whose leftover is removed on the way out. Nothing is left on disk that
// Stats does not count.
func (s *Store) Commit(namespace, fingerprint string, keys []string, planned, tokens int) (Entry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	key := keys[len(keys)-1]
	id := namespace + "/" + fingerprint + "/" + key
	plannedName := FileName(namespace, fingerprint, key, planned)
	e := &Entry{Namespace: namespace, Fingerprint: fingerprint, Key: key, Tokens: planned}
	// An older save of this same key is superseded. Its file is a different name,
	// so it goes now: deleting the entry alone leaves it on disk, uncounted,
	// outside the cap, the LRU and the age limit for good.
	if old := s.entries[id]; old != nil && old.Name() != plannedName {
		s.drop(old, "superseded")
	}
	if tokens > 0 && tokens != planned {
		// The file lands on the engine's own count. A tracked file of that name is
		// overwritten by the rename, so its entry and file go first.
		target := FileName(namespace, fingerprint, key, tokens)
		for _, o := range s.owners(target) {
			s.drop(o, "superseded")
		}
		e.Tokens = tokens
		if err := os.Rename(filepath.Join(s.opt.Dir, plannedName), filepath.Join(s.opt.Dir, e.Name())); err != nil {
			e.Tokens = planned
		}
	}
	info, err := os.Stat(filepath.Join(s.opt.Dir, e.Name()))
	if err != nil {
		// Nothing to index. The engine's file is then under the planned name: the
		// leftover of a rename that failed. Unlink it; the caller never hears of it.
		if e.Name() != plannedName {
			s.remove(plannedName, "not_indexed", 0)
		}
		return Entry{}, fmt.Errorf("saved file: %w", err)
	}
	e.Size, e.Used = info.Size(), info.ModTime()
	for _, k := range keys[:len(keys)-1] {
		pid := namespace + "/" + fingerprint + "/" + k
		if old := s.entries[pid]; old != nil && old.Name() != e.Name() {
			delete(s.entries, pid)
			s.prefixes[pid] = old // the turn-boundary file keeps its key; the file stays
		}
	}
	s.entries[id] = e
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

// rebuildChains replays the chain side index at startup: a prefix key maps to
// the entry of the file saved when it was terminal — the entry indexed by that
// key. A key with no such file stays empty: the chain never rebuilds an alias
// to a deeper file. Lines of files that are gone are dropped, and the cleaned
// index is written back through the same atomic rename. mu held.
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
				if pe := s.entries[id]; pe != nil {
					s.prefixes[id] = pe // the file saved when this key was terminal
				}
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
