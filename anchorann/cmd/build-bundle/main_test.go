package main

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"path/filepath"
	"strings"
	"testing"

	"privatesearch/anchorann"
	"privatesearch/anchorann/bundle"
)

func writeArtifact(t *testing.T, dir string) string {
	t.Helper()
	d := anchorann.Dataset{N: 256, Dim: 4, Values: make([]float32, 256*4)}
	rng := rand.New(rand.NewSource(1))
	for i := range d.Values {
		d.Values[i] = float32(rng.Intn(100))
	}
	a, err := anchorann.Cluster(d, anchorann.Options{Size: 8, Seed: 1, Method: anchorann.Recursive, Iterations: 2, Workers: 2})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "a.anchor")
	if err := anchorann.Save(path, a); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestRun(t *testing.T) {
	dir := t.TempDir()
	art := writeArtifact(t, dir)
	out := filepath.Join(dir, "a.bundle")
	args := func(output, workers string) []string {
		return []string{"-artifact", art, "-output", output, "-knn", "8", "-l", "16", "-r", "6", "-c-nsg", "32", "-workers", workers, "-pir-entry-count", "32", "-pir-entry-bytes", "16", "-pir-batch-size", "2"}
	}
	var buf bytes.Buffer
	if err := run(args(out, "2"), &buf); err != nil {
		t.Fatal(err)
	}
	var r report
	if err := json.Unmarshal(buf.Bytes(), &r); err != nil {
		t.Fatal(err)
	}
	b, err := bundle.Load(out, bundle.Expect{Count: 32, Dim: 4, Size: 8})
	if err != nil {
		t.Fatal(err)
	}
	if !r.ReloadVerified || r.Count != 32 || r.BundleBytes == 0 || r.PeakRSSBytes <= 0 || r.Build.Nodes != 32 || r.R != 6 || b.PIR.EntryBytes != 16 || r.Entry != b.Graph.Entry {
		t.Fatalf("report %+v", r)
	}
	sum := r.Sections.Header + r.Sections.Mapping + r.Sections.Vectors + r.Sections.Degrees + r.Sections.Neighbors + r.Sections.Checksum
	if sum != r.BundleBytes {
		t.Fatalf("sections %d != bundle %d", sum, r.BundleBytes)
	}
	// Rebuilding with different parallelism yields the same bundle.
	var again bytes.Buffer
	if err := run(args(filepath.Join(dir, "b.bundle"), "7"), &again); err != nil {
		t.Fatal(err)
	}
	var r2 report
	json.Unmarshal(again.Bytes(), &r2)
	if r2.BundleSHA256 != r.BundleSHA256 || r2.Workers != 7 {
		t.Fatalf("non-deterministic: %s vs %s (workers %d)", r2.BundleSHA256, r.BundleSHA256, r2.Workers)
	}
}

func TestRunRejectsInvalidInput(t *testing.T) {
	dir := t.TempDir()
	art := writeArtifact(t, dir)
	cases := [][]string{
		{},
		{"-artifact", art},
		{"-artifact", art, "-output", art},
		{"-artifact", filepath.Join(dir, "missing"), "-output", filepath.Join(dir, "x")},
		{"-artifact", art, "-output", filepath.Join(dir, "x"), "-r", "0"},
		{"-artifact", art, "-output", filepath.Join(dir, "x"), "-pir-entry-count", "3", "-pir-entry-bytes", "8"},
		{"-artifact", art, "-output", filepath.Join(dir, "nodir", "x")},
		{"-bogus"},
	}
	for _, args := range cases {
		if err := run(args, &bytes.Buffer{}); err == nil {
			t.Errorf("run(%s) succeeded", strings.Join(args, " "))
		}
	}
}
