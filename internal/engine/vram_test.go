package engine

import (
	"os"
	"path/filepath"
	"testing"
)

// The gate's card is found from sysfs: a box with an integrated GPU puts the
// discrete card at card1, and a fixed card0 path reads nothing.

func TestDetectCardPicksTheBiggest(t *testing.T) {
	root := t.TempDir()
	for _, c := range []struct{ card, total string }{{"card0", "1073741824\n"}, {"card1", "25753026560\n"}} {
		dir := filepath.Join(root, c.card, "device")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		for name, v := range map[string]string{"mem_info_vram_total": c.total, "mem_info_vram_used": "0\n"} {
			if err := os.WriteFile(filepath.Join(dir, name), []byte(v), 0o600); err != nil {
				t.Fatal(err)
			}
		}
	}
	oldRoot, oldVRAM := SysfsRoot, SysfsVRAM
	SysfsRoot, SysfsVRAM = root, filepath.Join(root, "card0", "device")
	defer func() { SysfsRoot, SysfsVRAM = oldRoot, oldVRAM }()

	card := filepath.Join(root, "card1", "device")
	if got := DetectCard(); got != card {
		t.Errorf("card picked: %s", got)
	}
	SysfsVRAM = card
	free, err := FreeVRAM()
	if err != nil || free != 25753026560 {
		t.Errorf("free VRAM read from card1: %d %v", free, err)
	}
}

func TestDetectCardWithNothingReadable(t *testing.T) {
	root := t.TempDir()
	oldRoot := SysfsRoot
	SysfsRoot = root
	defer func() { SysfsRoot = oldRoot }()

	if got := DetectCard(); got != "" {
		t.Errorf("an empty sysfs root answered %s", got)
	}
	SysfsVRAM = filepath.Join(root, "card9", "device")
	if _, err := FreeVRAM(); err == nil {
		t.Error("a card that is not there read as free VRAM")
	}
}
