// Command build-bundle builds the server-side anchor NSG from a clustering
// artifact and writes the distribution bundle with a JSON cost report.
package main

import (
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	"privatesearch/anchorann"
	"privatesearch/anchorann/bundle"
	"privatesearch/anchorann/nsg"
)

type sections struct {
	Header    int `json:"header"`
	Mapping   int `json:"anchor_cluster_mapping"`
	Vectors   int `json:"anchor_vectors_float32"`
	Degrees   int `json:"degrees"`
	Neighbors int `json:"neighbors"`
	Checksum  int `json:"sha256"`
}

type report struct {
	Artifact       string           `json:"artifact"`
	ArtifactSHA256 string           `json:"artifact_sha256"`
	Output         string           `json:"output"`
	N              int              `json:"n"`
	Size           int              `json:"s"`
	Count          int              `json:"c"`
	Dimension      int              `json:"dimension"`
	Method         anchorann.Method `json:"method"`
	Seed           int64            `json:"seed"`
	KNN            int              `json:"knn"`
	L              int              `json:"l"`
	R              int              `json:"r"`
	CNSG           int              `json:"c_nsg"`
	Workers        int              `json:"workers"`
	Entry          uint32           `json:"entry"`
	PIR            bundle.PIRParams `json:"pir"`
	LoadSeconds    float64          `json:"artifact_load_seconds"`
	Build          nsg.Stats        `json:"nsg"`
	SaveSeconds    float64          `json:"encode_save_seconds"`
	PeakRSSBytes   int64            `json:"peak_rss_bytes"`
	BundleBytes    int              `json:"bundle_bytes"`
	BundleSHA256   string           `json:"bundle_sha256"`
	Sections       sections         `json:"sections_bytes"`
	ReloadVerified bool             `json:"reload_verified"`
	GoVersion      string           `json:"go_version"`
	Platform       string           `json:"platform"`
	GOMAXPROCS     int              `json:"gomaxprocs"`
}

// PeakRSS returns the process peak resident set size in bytes.
func peakRSS() (int64, error) {
	var u syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &u); err != nil {
		return 0, fmt.Errorf("measure peak RSS: %w", err)
	}
	if runtime.GOOS == "darwin" {
		return u.Maxrss, nil
	}
	return u.Maxrss * 1024, nil
}

func run(args []string, out io.Writer) error {
	def := nsg.DefaultParams()
	f := flag.NewFlagSet("build-bundle", flag.ContinueOnError)
	artifact := f.String("artifact", "", "clustering artifact from anchorann/cmd/cluster (required)")
	output := f.String("output", "", "output bundle file (required)")
	knn := f.Int("knn", def.KNN, "exact initial kNN graph degree K")
	l := f.Int("l", def.L, "NSG build-time search pool L")
	r := f.Int("r", def.R, "NSG out-degree cap R")
	cnsg := f.Int("c-nsg", def.CNSG, "NSG candidate pool cap C_nsg (not the cluster count)")
	workers := f.Int("workers", def.Workers, "build parallelism (does not change the output)")
	pirEntries := f.Uint64("pir-entry-count", 0, "PIR DB entry count (0 = unset)")
	pirBytes := f.Uint64("pir-entry-bytes", 0, "PIR DB entry bytes (0 = unset)")
	pirBatch := f.Uint("pir-batch-size", 0, "PIR batch size (0 = unset)")
	if err := f.Parse(args); err != nil {
		return err
	}
	if *artifact == "" || *output == "" {
		return fmt.Errorf("-artifact and -output are required")
	}
	if a, err := filepath.Abs(*artifact); err == nil {
		if o, err := filepath.Abs(*output); err == nil && a == o {
			return fmt.Errorf("-output must differ from -artifact")
		}
	}
	start := time.Now()
	raw, err := os.ReadFile(*artifact)
	if err != nil {
		return fmt.Errorf("read artifact: %w", err)
	}
	sum := sha256.Sum256(raw)
	raw = nil
	a, err := anchorann.Load(*artifact)
	if err != nil {
		return err
	}
	loaded := time.Now()
	p := nsg.Params{KNN: *knn, L: *l, R: *r, CNSG: *cnsg, Workers: *workers}
	pir := bundle.PIRParams{EntryCount: *pirEntries, EntryBytes: *pirBytes, BatchSize: uint32(*pirBatch)}
	b, st, err := bundle.FromArtifact(a, sum, p, pir)
	if err != nil {
		return err
	}
	built := time.Now()
	data, err := bundle.Save(*output, b)
	if err != nil {
		return err
	}
	saved := time.Now()
	peak, err := peakRSS()
	if err != nil {
		return err
	}
	if _, err := bundle.Load(*output, bundle.Expect{Count: a.Count, Dim: a.Dim, Size: a.Size}); err != nil {
		return fmt.Errorf("reload bundle: %w", err)
	}
	edges := b.Graph.Edges()
	rep := report{
		Artifact: *artifact, ArtifactSHA256: fmt.Sprintf("%x", sum), Output: *output,
		N: b.N, Size: b.Size, Count: b.Count, Dimension: b.Dim, Method: b.Method, Seed: b.Seed,
		KNN: p.KNN, L: p.L, R: p.R, CNSG: p.CNSG, Workers: p.Workers, Entry: b.Graph.Entry, PIR: pir,
		LoadSeconds: loaded.Sub(start).Seconds(), Build: st, SaveSeconds: saved.Sub(built).Seconds(),
		PeakRSSBytes: peak, BundleBytes: len(data), BundleSHA256: bundle.Checksum(data),
		Sections:       sections{Header: bundle.HeaderBytes, Mapping: 4 * b.Count, Vectors: 4 * b.Count * b.Dim, Degrees: 4 * b.Count, Neighbors: 4 * edges, Checksum: sha256.Size},
		ReloadVerified: true, GoVersion: runtime.Version(), Platform: runtime.GOOS + "/" + runtime.GOARCH, GOMAXPROCS: runtime.GOMAXPROCS(0),
	}
	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	if err := enc.Encode(rep); err != nil {
		return fmt.Errorf("write report: %w", err)
	}
	return nil
}

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "anchorann build-bundle:", err)
		os.Exit(1)
	}
}
