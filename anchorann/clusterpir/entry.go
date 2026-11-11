// Package clusterpir stores each AnchorANN cluster (its s original IDs and real
// vectors) as one fixed-length Piano PIR entry whose index is the cluster ID,
// serves it with the existing rpc/server PirService, and privately fetches the
// n_probe selected clusters of a query with a fixed number of PIR requests.
package clusterpir

import (
	"encoding/binary"
	"fmt"
	"math"
)

// WordBytes is the Piano PIR word size; every entry length is a multiple of it.
const WordBytes = 8

// Layout is the public shape of the cluster PIR DB. Entry c holds cluster c.
//
// Entry bytes (little-endian): s uint32 original IDs, then s·Dim float32
// coordinates (row-major, point j at [j·Dim, (j+1)·Dim)), then zero bytes up to
// the next multiple of WordBytes. No dummy points are stored.
type Layout struct {
	Count int `json:"c"`         // entries = clusters
	Size  int `json:"s"`         // real points per cluster
	Dim   int `json:"dimension"` // vector dimension
}

// NewLayout validates the shape and that byte sizes fit in an int.
func NewLayout(count, size, dim int) (Layout, error) {
	l := Layout{Count: count, Size: size, Dim: dim}
	if count <= 0 || size <= 0 || dim <= 0 {
		return l, fmt.Errorf("invalid cluster PIR layout: C=%d s=%d dimension=%d (all must be positive)", count, size, dim)
	}
	if uint64(count)*uint64(size) > uint64(math.MaxUint32)+1 {
		return l, fmt.Errorf("invalid cluster PIR layout: N=C·s=%d exceeds uint32 original IDs", uint64(count)*uint64(size))
	}
	maxInt := uint64(math.MaxInt)
	if uint64(size) > maxInt/4/(uint64(dim)+1) {
		return l, fmt.Errorf("invalid cluster PIR layout: entry for s=%d dimension=%d overflows", size, dim)
	}
	if uint64(l.EntryBytes()) > maxInt/uint64(count) {
		return l, fmt.Errorf("invalid cluster PIR layout: DB for C=%d entry bytes=%d overflows", count, l.EntryBytes())
	}
	return l, nil
}

// N is the number of real points, C·s.
func (l Layout) N() int { return l.Count * l.Size }

// PayloadBytes is the unpadded entry length 4s + 4s·Dim.
func (l Layout) PayloadBytes() int { return 4*l.Size + 4*l.Size*l.Dim }

// EntryBytes is PayloadBytes rounded up to WordBytes.
func (l Layout) EntryBytes() int {
	return (l.PayloadBytes() + WordBytes - 1) / WordBytes * WordBytes
}

// EntryWords is the entry length in uint64 words.
func (l Layout) EntryWords() int { return l.EntryBytes() / WordBytes }

// DBBytes is the total DB size C·EntryBytes.
func (l Layout) DBBytes() int64 { return int64(l.Count) * int64(l.EntryBytes()) }

// Cluster is one decoded entry: point j has original ID IDs[j] and vector
// Vectors[j·Dim:(j+1)·Dim].
type Cluster struct {
	ID      int
	IDs     []uint32
	Vectors []float32
}

// EncodeEntry writes ids and vectors into dst (EntryWords long) and zeroes the
// alignment padding.
func EncodeEntry(l Layout, ids []uint32, vectors []float32, dst []uint64) error {
	if len(ids) != l.Size || len(vectors) != l.Size*l.Dim {
		return fmt.Errorf("encode entry: ids=%d vectors=%d expected s=%d s·dimension=%d", len(ids), len(vectors), l.Size, l.Size*l.Dim)
	}
	if len(dst) != l.EntryWords() {
		return fmt.Errorf("encode entry: dst words=%d expected=%d", len(dst), l.EntryWords())
	}
	buf := make([]byte, l.EntryBytes())
	for j, id := range ids {
		binary.LittleEndian.PutUint32(buf[4*j:], id)
	}
	off := 4 * l.Size
	for j, x := range vectors {
		binary.LittleEndian.PutUint32(buf[off+4*j:], math.Float32bits(x))
	}
	for w := range dst {
		dst[w] = binary.LittleEndian.Uint64(buf[WordBytes*w:])
	}
	return nil
}

// DecodeEntry checks the entry length, that every original ID is < N and
// distinct within the entry, that coordinates are finite, and that the padding
// is zero. An all-zero response (e.g. a dropped PIR answer) fails the
// distinctness check whenever s > 1.
func DecodeEntry(l Layout, words []uint64) (Cluster, error) {
	var c Cluster
	if len(words) != l.EntryWords() {
		return c, fmt.Errorf("decode entry: length=%d words expected=%d (entry bytes=%d)", len(words), l.EntryWords(), l.EntryBytes())
	}
	buf := make([]byte, l.EntryBytes())
	for w, x := range words {
		binary.LittleEndian.PutUint64(buf[WordBytes*w:], x)
	}
	n := uint64(l.N())
	c.IDs = make([]uint32, l.Size)
	seen := make(map[uint32]struct{}, l.Size)
	for j := range c.IDs {
		id := binary.LittleEndian.Uint32(buf[4*j:])
		if uint64(id) >= n {
			return c, fmt.Errorf("decode entry: point=%d original ID=%d outside 0..%d", j, id, n-1)
		}
		if _, dup := seen[id]; dup {
			return c, fmt.Errorf("decode entry: duplicate original ID=%d at point=%d", id, j)
		}
		seen[id] = struct{}{}
		c.IDs[j] = id
	}
	off := 4 * l.Size
	c.Vectors = make([]float32, l.Size*l.Dim)
	for j := range c.Vectors {
		x := math.Float32frombits(binary.LittleEndian.Uint32(buf[off+4*j:]))
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return c, fmt.Errorf("decode entry: nonfinite coordinate point=%d coordinate=%d", j/l.Dim, j%l.Dim)
		}
		c.Vectors[j] = x
	}
	for i := l.PayloadBytes(); i < len(buf); i++ {
		if buf[i] != 0 {
			return c, fmt.Errorf("decode entry: nonzero padding byte at offset=%d", i)
		}
	}
	return c, nil
}
