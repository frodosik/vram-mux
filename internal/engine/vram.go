package engine

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

// VRAM is read from the amdgpu sysfs node: the card is shared by every engine,
// so the arbiter gates on the real free memory, not on one engine's idea of it.
// Paths are settable for tests.

// SysfsVRAM is the amdgpu device node for the card the engines share.
var SysfsVRAM = "/sys/class/drm/card0/device"

// SysfsRoot is where the DRM cards live; settable for tests.
var SysfsRoot = "/sys/class/drm"

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
// arbiter's margin gate is measured against.
func FreeVRAM() (int64, error) {
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
