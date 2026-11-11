// Package anchorann implements server-side equal-size clustering and anchors.
package anchorann

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// Dataset stores real vectors in row-major order. Original IDs are row indices.
type Dataset struct {
	N      int
	Dim    int
	Values []float32
}

// Validate checks vector shape and finite coordinates before clustering starts.
func (d Dataset) Validate() error {
	if d.N <= 0 || d.Dim <= 0 || d.N > int(^uint(0)>>1)/d.Dim || len(d.Values) != d.N*d.Dim {
		return fmt.Errorf("invalid dataset: N=%d dimension=%d coordinates=%d", d.N, d.Dim, len(d.Values))
	}
	for i, x := range d.Values {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return fmt.Errorf("nonfinite coordinate: N=%d dimension=%d point=%d coordinate=%d", d.N, d.Dim, i/d.Dim, i%d.Dim)
		}
	}
	return nil
}

func validateSize(n, s int) error {
	if s <= 0 {
		return fmt.Errorf("invalid cluster size: N=%d s=%d (s must be positive)", n, s)
	}
	if n <= 0 || n%s != 0 {
		return fmt.Errorf("cannot form equal-size clusters: N=%d s=%d remainder=%d (N must be positive and divisible by s)", n, s, n%s)
	}
	return nil
}

// ReadVectors reads fvecs or bvecs using each record's little-endian dimension.
// limit=0 reads every record; a positive limit explicitly selects a prefix and
// fails if fewer records exist. No records within that prefix may be malformed.
func ReadVectors(path, format string, limit int) (Dataset, error) {
	var d Dataset
	if limit < 0 {
		return d, fmt.Errorf("negative vector limit: %d", limit)
	}
	if format == "auto" {
		format = strings.TrimPrefix(strings.ToLower(filepath.Ext(path)), ".")
	}
	width := 4
	switch format {
	case "fvecs":
	case "bvecs":
		width = 1
	default:
		return d, fmt.Errorf("unsupported vector format %q", format)
	}
	f, err := os.Open(path)
	if err != nil {
		return d, fmt.Errorf("open vectors: %w", err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return d, fmt.Errorf("stat vectors: %w", err)
	}
	var header [4]byte
	if _, err = io.ReadFull(f, header[:]); err != nil {
		return d, fmt.Errorf("read vector dimension (N=0): %w", err)
	}
	d.Dim = int(binary.LittleEndian.Uint32(header[:]))
	record := int64(4) + int64(width)*int64(d.Dim)
	if d.Dim <= 0 || record > info.Size() {
		return d, fmt.Errorf("invalid vector dimension=%d file bytes=%d", d.Dim, info.Size())
	}
	if limit == 0 && info.Size()%record != 0 {
		return d, fmt.Errorf("truncated vectors: dimension=%d record bytes=%d file bytes=%d", d.Dim, record, info.Size())
	}
	n := info.Size() / record
	if limit > 0 {
		if int64(limit) > n {
			return d, fmt.Errorf("vector limit=%d exceeds available N=%d", limit, n)
		}
		n = int64(limit)
	}
	if n > int64(int(^uint(0)>>1)/d.Dim) {
		return d, fmt.Errorf("vector shape too large: N=%d dimension=%d", n, d.Dim)
	}
	d.N = int(n)
	d.Values = make([]float32, d.N*d.Dim)
	// Read in blocks without allocating a second full copy of the dataset.
	const blockRecords = 1024
	buf := make([]byte, int(record)*min(blockRecords, d.N))
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return Dataset{}, fmt.Errorf("rewind vectors: %w", err)
	}
	for start := 0; start < d.N; start += blockRecords {
		count := min(blockRecords, d.N-start)
		block := buf[:count*int(record)]
		if _, err := io.ReadFull(f, block); err != nil {
			return Dataset{}, fmt.Errorf("read vectors at ID=%d: %w", start, err)
		}
		for i := 0; i < count; i++ {
			row := block[i*int(record) : (i+1)*int(record)]
			got := int(binary.LittleEndian.Uint32(row[:4]))
			if got != d.Dim {
				return Dataset{}, fmt.Errorf("dimension mismatch: N=%d ID=%d dimension=%d expected=%d", d.N, start+i, got, d.Dim)
			}
			for j := 0; j < d.Dim; j++ {
				x := float32(row[4+j])
				if width == 4 {
					x = math.Float32frombits(binary.LittleEndian.Uint32(row[4+4*j:]))
				}
				d.Values[(start+i)*d.Dim+j] = x
			}
		}
	}
	if err := d.Validate(); err != nil {
		return Dataset{}, err
	}
	return d, nil
}
