package engine

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// VRAM is read from the amdgpu sysfs node: the card is shared by every engine,
// so the arbiter gates on the real free memory, not on one engine's idea of it.
// Paths are settable for tests. With no amdgpu card on the box the gate falls
// back to nvidia-smi, whose runner is settable for tests.

// SysfsVRAM is the amdgpu device node for the card the engines share.
var SysfsVRAM = "/sys/class/drm/card0/device"

// SysfsRoot is where the DRM cards live; settable for tests.
var SysfsRoot = "/sys/class/drm"

// NvidiaGPUIndex pins the card nvidia-smi answers for, -1 = the largest card.
var NvidiaGPUIndex int64 = -1

// NvidiaSmi runs the query and returns its stdout: "index, free, total" per
// line, MiB. Settable for tests.
var NvidiaSmi = func(ctx context.Context) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	return exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits").Output()
}

// DetectCard names the device node of the card with the most VRAM: on a box with
// an integrated GPU the discrete card is card1, so a fixed card0 path reads
// nothing and the gate goes blind. "" when no card answers.
func DetectCard() string {
	best, bestTotal := "", int64(-1)
	dirs, err := filepath.Glob(SysfsRoot + "/card*/device")
	if err != nil {
		return ""
	}
	for _, dir := range dirs {
		total, err := readSysfs(dir + "/mem_info_vram_total")
		if err != nil {
			continue
		}
		if total > bestTotal {
			best, bestTotal = dir, total
		}
	}
	return best
}

// FreeVRAM returns bytes free on the card: total - used. It is the figure the
// arbiter's margin gate is measured against. The amdgpu sysfs node is preferred
// (zero fork); a box with no amdgpu card gets nvidia-smi.
func FreeVRAM() (int64, error) {
	if n, err := SysfsFree(); err == nil {
		return n, nil
	}
	return freeNvidia()
}

// SysfsFree reads the amdgpu node only, so the startup log can say which backend
// answers before any load.
func SysfsFree() (int64, error) {
	total, err := readSysfs(SysfsVRAM + "/mem_info_vram_total")
	if err != nil {
		return 0, err
	}
	used, err := readSysfs(SysfsVRAM + "/mem_info_vram_used")
	if err != nil {
		return 0, err
	}
	if total < used {
		return 0, nil
	}
	return total - used, nil
}

// freeNvidia parses nvidia-smi's csv rows "index, free, total" in MiB: --gpu-index
// picks that card, -1 picks the card with the most VRAM.
func freeNvidia() (int64, error) {
	out, err := NvidiaSmi(context.Background())
	if err != nil {
		return 0, err
	}
	best, bestTotal := int64(0), int64(-1)
	for _, line := range strings.Split(string(out), "\n") {
		f := strings.Split(line, ",")
		if len(f) < 3 {
			continue
		}
		idx, ok1 := parseNum(f[0])
		free, ok2 := parseNum(f[1])
		total, ok3 := parseNum(f[2])
		// A row like "0, [Not Supported], [Not Supported]" is not a card: a
		// row that does not parse end to end is skipped, not read as 0 free.
		if !ok1 || !ok2 || !ok3 {
			continue
		}
		if idx == NvidiaGPUIndex {
			return free << 20, nil
		}
		if total > bestTotal {
			best, bestTotal = free<<20, total
		}
	}
	if bestTotal < 0 {
		return 0, errors.New("nvidia-smi answered no card")
	}
	return best, nil
}

func parseNum(s string) (int64, bool) {
	n, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
	return n, err == nil
}

// readSysfs reads one amdgpu file: a decimal, optionally suffixed with " MiB".
func readSysfs(path string) (int64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	f := strings.Fields(string(raw))
	if len(f) == 0 {
		return 0, os.ErrInvalid
	}
	n, err := strconv.ParseInt(f[0], 10, 64)
	if err != nil {
		return 0, err
	}
	if len(f) > 1 && f[1] == "MiB" {
		return n << 20, nil
	}
	return n, nil
}

// cacheFree keeps the last readable figure so one engine's answer can be
// cross-checked against the card.
var cacheFree struct {
	sync.Mutex
	bytes int64
}

// NoteFree records a figure an engine reported (its own vram_free_mib), so a
// sysfs read that fails can fall back to the engine's last known answer.
func NoteFree(bytes int64) {
	cacheFree.Lock()
	cacheFree.bytes = bytes
	cacheFree.Unlock()
}

// LastFree returns the last figure recorded, 0 when none.
func LastFree() int64 {
	cacheFree.Lock()
	defer cacheFree.Unlock()
	return cacheFree.bytes
}
