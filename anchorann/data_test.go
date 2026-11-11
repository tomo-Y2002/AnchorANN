package anchorann

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func writeVectors(t *testing.T, format string, rows [][]float32) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "vectors."+format)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	for _, row := range rows {
		if err := binary.Write(f, binary.LittleEndian, uint32(len(row))); err != nil {
			t.Fatal(err)
		}
		if format == "fvecs" {
			err = binary.Write(f, binary.LittleEndian, row)
		} else {
			b := make([]byte, len(row))
			for j, x := range row {
				b[j] = byte(x)
			}
			_, err = f.Write(b)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	return path
}

func TestReadVectors(t *testing.T) {
	for _, format := range []string{"fvecs", "bvecs"} {
		path := writeVectors(t, format, [][]float32{{0, 1}, {127, 255}, {2, 3}})
		for _, requestedFormat := range []string{format, "auto"} {
			d, err := ReadVectors(path, requestedFormat, 0)
			want := Dataset{N: 3, Dim: 2, Values: []float32{0, 1, 127, 255, 2, 3}}
			if err != nil || !reflect.DeepEqual(d, want) {
				t.Fatalf("%s: got=%+v error=%v", requestedFormat, d, err)
			}
			prefix, err := ReadVectors(path, requestedFormat, 2)
			if err != nil || prefix.N != 2 || !reflect.DeepEqual(prefix.Values, want.Values[:4]) {
				t.Fatalf("%s prefix: %+v error=%v", requestedFormat, prefix, err)
			}
		}
	}
	path := writeVectors(t, "fvecs", [][]float32{{-1.5, 0.125}, {3.25, -10}})
	d, err := ReadVectors(path, "auto", 0)
	if err != nil || !reflect.DeepEqual(d.Values, []float32{-1.5, 0.125, 3.25, -10}) {
		t.Fatalf("float coordinates changed: %+v %v", d, err)
	}
}

func TestReadVectorsBlockBoundary(t *testing.T) {
	rows := make([][]float32, 2051)
	for i := range rows {
		rows[i] = []float32{float32(i), float32(-i)}
	}
	d, err := ReadVectors(writeVectors(t, "fvecs", rows), "auto", 0)
	if err != nil || d.N != len(rows) {
		t.Fatalf("read multi-block input: N=%d %v", d.N, err)
	}
	for i, row := range rows {
		if !reflect.DeepEqual(d.Values[i*2:i*2+2], row) {
			t.Fatalf("row %d changed", i)
		}
	}
}

func TestReadVectorsRejectsMalformedInput(t *testing.T) {
	good := writeVectors(t, "fvecs", [][]float32{{1, 2}, {3, 4}})
	for _, tc := range []struct {
		name, path, format string
		limit              int
		want               string
	}{
		{"missing", filepath.Join(t.TempDir(), "absent"), "fvecs", 0, "open vectors"},
		{"format", good, "ivecs", 0, "unsupported"},
		{"negative limit", good, "fvecs", -1, "negative"},
		{"oversized limit", good, "fvecs", 3, "available N=2"},
		{"empty", writeVectors(t, "fvecs", nil), "auto", 0, "N=0"},
		{"zero dimension", writeVectors(t, "fvecs", [][]float32{{}}), "auto", 0, "dimension=0"},
		{"mismatch", writeVectors(t, "fvecs", [][]float32{{1, 2}, {3, 4, 5}}), "auto", 2, "dimension mismatch"},
		{"nan", writeVectors(t, "fvecs", [][]float32{{float32(math.NaN())}}), "auto", 0, "nonfinite"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ReadVectors(tc.path, tc.format, tc.limit); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ReadVectors()=%v; want %q", err, tc.want)
			}
		})
	}
	bytes, err := os.ReadFile(good)
	if err != nil {
		t.Fatal(err)
	}
	for _, content := range [][]byte{bytes[:2], bytes[:len(bytes)-1]} {
		path := filepath.Join(t.TempDir(), "truncated.fvecs")
		if err := os.WriteFile(path, content, 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := ReadVectors(path, "auto", 0); err == nil {
			t.Fatal("accepted truncated input")
		}
	}
	// A malformed record inside an explicitly selected prefix is still rejected.
	bytes[12] = 3
	path := filepath.Join(t.TempDir(), "mismatch.fvecs")
	if err := os.WriteFile(path, bytes, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadVectors(path, "auto", 2); err == nil || !strings.Contains(err.Error(), "dimension mismatch") {
		t.Fatalf("accepted mismatched dimension: %v", err)
	}
}
