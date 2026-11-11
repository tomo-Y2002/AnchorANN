package anchorann

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestArtifactContractValidation(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*Artifact)
		want   string
	}{
		{"version", func(a *Artifact) { a.Version++ }, "version"},
		{"zero size", func(a *Artifact) { a.Size = 0 }, "s=0"},
		{"remainder", func(a *Artifact) { a.N++ }, "remainder=1"},
		{"count", func(a *Artifact) { a.Count++ }, "shape"},
		{"missing cluster", func(a *Artifact) { a.Members = a.Members[:1] }, "shape"},
		{"missing point", func(a *Artifact) { a.Members[0] = a.Members[0][:1] }, "size=1"},
		{"oversized cluster", func(a *Artifact) { a.Members[0] = append(a.Members[0], 0) }, "size=3"},
		{"duplicate and missing", func(a *Artifact) { a.Members[1][0] = a.Members[0][0] }, "duplicate"},
		{"out of range", func(a *Artifact) { a.Members[0][0] = uint32(a.N) }, "outside"},
		{"anchor count", func(a *Artifact) { a.Anchors = a.Anchors[:1] }, "shape"},
		{"anchor dimension", func(a *Artifact) { a.Anchors[0] = nil }, "dimension"},
		{"zero dimension", func(a *Artifact) { a.Dim = 0 }, "shape"},
		{"nan", func(a *Artifact) { a.Anchors[0][0] = math.NaN() }, "nonfinite"},
		{"infinity", func(a *Artifact) { a.Anchors[0][0] = math.Inf(1) }, "nonfinite"},
		{"method", func(a *Artifact) { a.Method = "unknown" }, "algorithm"},
		{"iterations", func(a *Artifact) { a.Iterations = 0 }, "algorithm"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := testCluster(t, testDataset(6, 2), 2, Recursive, 1)
			tc.change(a)
			if err := a.Validate(); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Validate()=%v; want %q", err, tc.want)
			}
			path := filepath.Join(t.TempDir(), "invalid.anchor")
			if err := Save(path, a); err == nil {
				t.Fatal("Save accepted invalid artifact")
			}
			if _, err := os.Stat(path); !os.IsNotExist(err) {
				t.Fatalf("invalid save created file: %v", err)
			}
		})
	}
	var a *Artifact
	if err := a.Validate(); err == nil {
		t.Fatal("nil artifact accepted")
	}
}

func TestArtifactRoundTrip(t *testing.T) {
	for _, method := range []Method{Recursive, Global, Common} {
		t.Run(string(method), func(t *testing.T) {
			a := testCluster(t, testDataset(30, 4), 2, method, 3)
			a.Seed = -711
			path := filepath.Join(t.TempDir(), "clusters.anchor")
			if err := Save(path, a); err != nil {
				t.Fatal(err)
			}
			got, err := Load(path)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(a, got) {
				t.Fatalf("round trip mismatch: %+v %+v", a, got)
			}
			before, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			a.Members[0][0] = a.Members[0][1]
			if err := Save(path, a); err == nil {
				t.Fatal("invalid replacement succeeded")
			}
			after, err := os.ReadFile(path)
			if err != nil || !bytes.Equal(before, after) {
				t.Fatalf("invalid replacement changed existing file: %v", err)
			}
		})
	}
}

func TestLoadRejectsCorruption(t *testing.T) {
	a := testCluster(t, testDataset(6, 2), 2, Recursive, 1)
	var buf bytes.Buffer
	if err := encode(&buf, a); err != nil {
		t.Fatal(err)
	}
	base := buf.Bytes()
	hsize := binary.Size(header(a))
	row := 4 + 4*a.Size + 8*a.Dim
	for _, tc := range []struct {
		name   string
		change func([]byte) []byte
		want   string
	}{
		{"short header", func(b []byte) []byte { return b[:5] }, "header"},
		{"magic", func(b []byte) []byte { b[0] = 0; return b }, "magic"},
		{"version", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[8:], 2); return b }, "version"},
		{"N", func(b []byte) []byte { binary.LittleEndian.PutUint64(b[12:], 0); return b }, "header"},
		{"s", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[20:], 0); return b }, "header"},
		{"C", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[24:], 4); return b }, "header"},
		{"dimension", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[28:], 0); return b }, "header"},
		{"method", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[40:], 4); return b }, "header"},
		{"iterations", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[44:], 0); return b }, "header"},
		{"truncation", func(b []byte) []byte { return b[:len(b)-1] }, "length"},
		{"trailing bytes", func(b []byte) []byte { return append(b, 0) }, "length"},
		{"mapping", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[hsize:], 1); return b }, "does not match"},
		{"duplicate", func(b []byte) []byte { copy(b[hsize+row+4:], b[hsize+4:hsize+8]); return b }, "duplicate"},
		{"range", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[hsize+4:], 6); return b }, "outside"},
		{"nan centroid", func(b []byte) []byte {
			binary.LittleEndian.PutUint64(b[hsize+4+4*a.Size:], math.Float64bits(math.NaN()))
			return b
		}, "nonfinite"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "corrupt.anchor")
			if err := os.WriteFile(path, tc.change(append([]byte(nil), base...)), 0600); err != nil {
				t.Fatal(err)
			}
			if _, err := Load(path); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Load()=%v; want %q", err, tc.want)
			}
		})
	}
}

func TestArtifactFilesystemErrors(t *testing.T) {
	a := testCluster(t, testDataset(6, 2), 2, Recursive, 1)
	dir := t.TempDir()
	if _, err := Load(filepath.Join(dir, "absent.anchor")); err == nil {
		t.Fatal("loaded missing file")
	}
	if err := Save(filepath.Join(dir, "absent", "data.anchor"), a); err == nil {
		t.Fatal("saved to missing parent")
	}
	if err := Save(dir, a); err == nil {
		t.Fatal("replaced directory with artifact")
	}
	files, err := os.ReadDir(dir)
	if err != nil || len(files) != 0 {
		t.Fatalf("failed save leaked temporary file: %v %v", files, err)
	}
}
