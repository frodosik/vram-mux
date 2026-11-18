package engine

import (
	"context"
	"errors"
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
	oldSmi, oldIdx := NvidiaSmi, NvidiaGPUIndex
	NvidiaSmi, NvidiaGPUIndex = func(context.Context) ([]byte, error) {
		return nil, errors.New("no nvidia-smi on this box")
	}, -1
	defer func() { NvidiaSmi, NvidiaGPUIndex = oldSmi, oldIdx }()
	if _, err := FreeVRAM(); err == nil {
		t.Error("a card that is not there read as free VRAM")
	}
}

// With no amdgpu card the gate asks nvidia-smi; its runner is settable.

func TestFreeVRAMFallsBackToNvidia(t *testing.T) {
	oldVRAM, oldSmi, oldIdx := SysfsVRAM, NvidiaSmi, NvidiaGPUIndex
	SysfsVRAM = filepath.Join(t.TempDir(), "card9", "device")
	NvidiaSmi = func(context.Context) ([]byte, error) {
		return []byte("0, 23950, 24576\n1, 12000, 24576\n"), nil
	}
	NvidiaGPUIndex = int64(-1)
	defer func() { SysfsVRAM, NvidiaSmi, NvidiaGPUIndex = oldVRAM, oldSmi, oldIdx }()

	free, err := FreeVRAM()
	if err != nil || free != 23950<<20 {
		t.Errorf("largest card: %d %v, want 23950 MiB", free, err)
	}
	NvidiaGPUIndex = int64(1)
	free, err = FreeVRAM()
	if err != nil || free != 12000<<20 {
		t.Errorf("--gpu-index 1: %d %v, want 12000 MiB", free, err)
	}
}

func TestFreeVRAMUnparseableNvidiaAnswer(t *testing.T) {
	oldVRAM, oldSmi := SysfsVRAM, NvidiaSmi
	SysfsVRAM = filepath.Join(t.TempDir(), "card9", "device")
	NvidiaSmi = func(context.Context) ([]byte, error) { return []byte("no gpu here\n"), nil }
	defer func() { SysfsVRAM, NvidiaSmi = oldVRAM, oldSmi }()

	if _, err := FreeVRAM(); err == nil {
		t.Error("an unparseable nvidia-smi answer read as free VRAM")
	}
}

func TestFreeVRAMPrefersSysfs(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "card1", "device")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, v := range map[string]string{"mem_info_vram_total": "25753026560\n", "mem_info_vram_used": "1000\n"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(v), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	oldVRAM, oldSmi := SysfsVRAM, NvidiaSmi
	SysfsVRAM, NvidiaSmi = dir, func(context.Context) ([]byte, error) {
		t.Error("the amdgpu card answered, nvidia-smi must not be called")
		return nil, errors.New("called")
	}
	defer func() { SysfsVRAM, NvidiaSmi = oldVRAM, oldSmi }()

	free, err := FreeVRAM()
	if err != nil || free != 25753026560-1000 {
		t.Errorf("sysfs preferred: %d %v", free, err)
	}
}

func TestFreeVRAMSkipsUnparseableNvidiaRows(t *testing.T) {
	oldVRAM, oldSmi, oldIdx := SysfsVRAM, NvidiaSmi, NvidiaGPUIndex
	SysfsVRAM = filepath.Join(t.TempDir(), "card9", "device")
	NvidiaSmi = func(context.Context) ([]byte, error) {
		return []byte("0, [Not Supported], [Not Supported]\n"), nil
	}
	NvidiaGPUIndex = int64(-1)
	defer func() { SysfsVRAM, NvidiaSmi, NvidiaGPUIndex = oldVRAM, oldSmi, oldIdx }()

	if _, err := FreeVRAM(); err == nil {
		t.Error("a [Not Supported] row read as 0 free VRAM")
	}
}

func TestFreeVRAMPicksTheGoodRow(t *testing.T) {
	oldVRAM, oldSmi, oldIdx := SysfsVRAM, NvidiaSmi, NvidiaGPUIndex
	SysfsVRAM = filepath.Join(t.TempDir(), "card9", "device")
	NvidiaSmi = func(context.Context) ([]byte, error) {
		return []byte("0, [Not Supported], [Not Supported]\n1, 22000, 24576\n"), nil
	}
	NvidiaGPUIndex = int64(-1)
	defer func() { SysfsVRAM, NvidiaSmi, NvidiaGPUIndex = oldVRAM, oldSmi, oldIdx }()

	free, err := FreeVRAM()
	if err != nil || free != 22000<<20 {
		t.Errorf("row 1 after a bad row: %d %v, want 22000 MiB", free, err)
	}
}
