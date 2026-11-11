package anchorann

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

// FormatVersion is the version of the standalone clustering artifact.
const FormatVersion uint32 = 1

var magic = [8]byte{'A', 'N', 'C', 'H', 'O', 'R', '0', '1'}

// Artifact contains cluster c's original IDs and its corresponding anchor c.
// Both Members and Anchors are indexed by the shared cluster/anchor ID. Anchors
// remain float64 on disk so accumulation precision is not lost in serialization.
type Artifact struct {
	Version    uint32
	N          int
	Size       int
	Count      int
	Dim        int
	Seed       int64
	Method     Method
	Iterations int
	Members    [][]uint32
	Anchors    [][]float64
}

// Validate enforces the version, shape, finite anchors, equal real-point counts,
// and exact-once membership of IDs 0..N-1. It needs no original vector dataset.
// Recompute centroids against the original vectors to verify their provenance.
func (a *Artifact) Validate() error {
	if a == nil {
		return fmt.Errorf("nil clustering artifact")
	}
	if a.Version != FormatVersion {
		return fmt.Errorf("unsupported artifact version=%d", a.Version)
	}
	if err := validateSize(a.N, a.Size); err != nil {
		return err
	}
	if uint64(a.N) > uint64(^uint32(0)) || a.Dim <= 0 || uint64(a.Dim) > uint64(^uint32(0)) || a.Count != a.N/a.Size || len(a.Members) != a.Count || len(a.Anchors) != a.Count {
		return fmt.Errorf("invalid artifact shape: N=%d s=%d C=%d dimension=%d member rows=%d anchor rows=%d", a.N, a.Size, a.Count, a.Dim, len(a.Members), len(a.Anchors))
	}
	if a.Iterations <= 0 || uint64(a.Iterations) > uint64(^uint32(0)) || (a.Method != Recursive && a.Method != Global && a.Method != Common) {
		return fmt.Errorf("invalid artifact algorithm: method=%q iterations=%d", a.Method, a.Iterations)
	}
	seen := make([]bool, a.N)
	for c, ids := range a.Members {
		if len(ids) != a.Size {
			return fmt.Errorf("cluster=%d size=%d expected s=%d N=%d", c, len(ids), a.Size, a.N)
		}
		if len(a.Anchors[c]) != a.Dim {
			return fmt.Errorf("anchor=%d dimension=%d expected=%d", c, len(a.Anchors[c]), a.Dim)
		}
		for _, id := range ids {
			if uint64(id) >= uint64(a.N) {
				return fmt.Errorf("cluster=%d original ID=%d outside 0..%d", c, id, a.N-1)
			}
			if seen[id] {
				return fmt.Errorf("duplicate original ID=%d at cluster=%d (another ID is missing)", id, c)
			}
			seen[id] = true
		}
		for j, x := range a.Anchors[c] {
			if math.IsNaN(x) || math.IsInf(x, 0) {
				return fmt.Errorf("nonfinite anchor=%d coordinate=%d", c, j)
			}
		}
	}
	// N slots, no out-of-range IDs, and no duplicates imply no missing IDs.
	return nil
}

type diskHeader struct {
	Magic      [8]byte
	Version    uint32
	N          uint64
	Size       uint32
	Count      uint32
	Dim        uint32
	Seed       int64
	Method     uint32
	Iterations uint32
}

func header(a *Artifact) diskHeader {
	m := uint32(1)
	if a.Method == Global {
		m = 2
	}
	if a.Method == Common {
		m = 3
	}
	return diskHeader{magic, a.Version, uint64(a.N), uint32(a.Size), uint32(a.Count), uint32(a.Dim), a.Seed, m, uint32(a.Iterations)}
}

// Save validates before writing and atomically replaces path only after a
// complete, synced file has been closed. The parent directory must exist.
func Save(path string, a *Artifact) error {
	if err := a.Validate(); err != nil {
		return fmt.Errorf("save artifact: %w", err)
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".anchor-*")
	if err != nil {
		return fmt.Errorf("create artifact: %w", err)
	}
	defer os.Remove(f.Name())
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)
	if err := encode(w, a); err != nil {
		return fmt.Errorf("write artifact: %w", err)
	}
	if err := w.Flush(); err != nil {
		return fmt.Errorf("flush artifact: %w", err)
	}
	if err := f.Sync(); err != nil {
		return fmt.Errorf("sync artifact: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close artifact: %w", err)
	}
	if err := os.Rename(f.Name(), path); err != nil {
		return fmt.Errorf("replace artifact: %w", err)
	}
	return nil
}

func encode(w io.Writer, a *Artifact) error {
	if err := binary.Write(w, binary.LittleEndian, header(a)); err != nil {
		return err
	}
	for c, ids := range a.Members {
		if err := binary.Write(w, binary.LittleEndian, uint32(c)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, ids); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, a.Anchors[c]); err != nil {
			return err
		}
	}
	return nil
}

// Load checks file length before allocation, explicitly verifies each recorded
// anchor c -> cluster c mapping, then always calls the shared contract validator.
func Load(path string) (*Artifact, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open artifact: %w", err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat artifact: %w", err)
	}
	r := bufio.NewReaderSize(f, 1<<20)
	var h diskHeader
	if err := binary.Read(r, binary.LittleEndian, &h); err != nil {
		return nil, fmt.Errorf("read artifact header: %w", err)
	}
	if h.Magic != magic || h.Version != FormatVersion {
		return nil, fmt.Errorf("unsupported artifact magic=%q version=%d", h.Magic, h.Version)
	}
	if h.N == 0 || h.N > uint64(^uint32(0)) || h.Size == 0 || h.N%uint64(h.Size) != 0 || uint64(h.Count) != h.N/uint64(h.Size) || h.Dim == 0 || h.Iterations == 0 || (h.Method != 1 && h.Method != 2 && h.Method != 3) {
		return nil, fmt.Errorf("invalid artifact header: N=%d s=%d C=%d dimension=%d method=%d iterations=%d", h.N, h.Size, h.Count, h.Dim, h.Method, h.Iterations)
	}
	rowBytes := uint64(4) + 4*uint64(h.Size) + 8*uint64(h.Dim)
	headerBytes := uint64(binary.Size(h))
	if rowBytes > (uint64(math.MaxInt64)-headerBytes)/uint64(h.Count) {
		return nil, fmt.Errorf("artifact length overflows")
	}
	expected := headerBytes + uint64(h.Count)*rowBytes
	if uint64(info.Size()) != expected {
		return nil, fmt.Errorf("artifact length=%d expected=%d (truncated or trailing data)", info.Size(), expected)
	}
	maxInt := uint64(int(^uint(0) >> 1))
	if h.N > maxInt || uint64(h.Count)*uint64(h.Dim) > maxInt/8 {
		return nil, fmt.Errorf("artifact shape exceeds address space")
	}
	m := Recursive
	if h.Method == 2 {
		m = Global
	}
	if h.Method == 3 {
		m = Common
	}
	a := &Artifact{Version: h.Version, N: int(h.N), Size: int(h.Size), Count: int(h.Count), Dim: int(h.Dim), Seed: h.Seed, Method: m, Iterations: int(h.Iterations)}
	a.Members = make([][]uint32, a.Count)
	a.Anchors = make([][]float64, a.Count)
	ids := make([]uint32, a.N)
	centers := make([]float64, a.Count*a.Dim)
	for c := 0; c < a.Count; c++ {
		var anchorID uint32
		if err := binary.Read(r, binary.LittleEndian, &anchorID); err != nil {
			return nil, fmt.Errorf("read anchor ID: %w", err)
		}
		if anchorID != uint32(c) {
			return nil, fmt.Errorf("anchor ID=%d does not match cluster ID=%d", anchorID, c)
		}
		a.Members[c] = ids[c*a.Size : (c+1)*a.Size]
		a.Anchors[c] = centers[c*a.Dim : (c+1)*a.Dim]
		if err := binary.Read(r, binary.LittleEndian, a.Members[c]); err != nil {
			return nil, fmt.Errorf("read members: %w", err)
		}
		if err := binary.Read(r, binary.LittleEndian, a.Anchors[c]); err != nil {
			return nil, fmt.Errorf("read anchor: %w", err)
		}
	}
	if err := a.Validate(); err != nil {
		return nil, fmt.Errorf("load artifact: %w", err)
	}
	return a, nil
}
