// Command search-server serves everything an AnchorANN client needs on one
// gRPC address: the cluster PIR DB (PirService, built from the clustering
// artifact and the original vectors) and the anchor NSG bundle
// (AnchorBundle). The bundle must have been built from the same artifact
// file. Once listening it writes one JSON line with the build costs.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"example.com/private-search/rpc"
	"google.golang.org/grpc"

	"privatesearch/anchorann"
	"privatesearch/anchorann/bundle"
	"privatesearch/anchorann/bundlerpc"
	"privatesearch/anchorann/clusterpir"
)

type report struct {
	Addr            string            `json:"addr"`
	Artifact        string            `json:"artifact"`
	ArtifactSHA256  string            `json:"artifact_sha256"`
	Vectors         string            `json:"vectors"`
	Bundle          string            `json:"bundle"`
	BundleSHA256    string            `json:"bundle_sha256"`
	BundleBytes     int               `json:"bundle_bytes"`
	BundlePIR       bundle.PIRParams  `json:"bundle_pir"`
	Layout          clusterpir.Layout `json:"layout"`
	EntryBytes      int               `json:"entry_bytes"`
	DBBytes         int64             `json:"db_bytes"`
	Partitions      int               `json:"partitions"`
	PartitionSize   uint64            `json:"partition_size"`
	ChunkSize       uint64            `json:"chunk_size"`
	SetSize         uint64            `json:"set_size"`
	FailureProbLog2 uint64            `json:"failure_prob_log2"`
	ArtifactSeconds float64           `json:"artifact_load_seconds"`
	VectorsSeconds  float64           `json:"vectors_load_seconds"`
	BuildSeconds    float64           `json:"db_build_seconds"`
	BundleSeconds   float64           `json:"bundle_load_seconds"`
	PeakRSSBytes    int64             `json:"peak_rss_bytes"`
	MaxMessageBytes int               `json:"grpc_max_message_bytes"`
	GoVersion       string            `json:"go_version"`
	Platform        string            `json:"platform"`
	GOMAXPROCS      int               `json:"gomaxprocs"`
}

func peakRSS() int64 {
	var u syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &u); err != nil {
		return 0
	}
	if runtime.GOOS == "darwin" {
		return u.Maxrss
	}
	return u.Maxrss * 1024
}

// checkBundle requires the bundle to come from the artifact file (same
// SHA-256 and shape) and its PIR parameters, when set, to describe the DB.
func checkBundle(b *bundle.Bundle, artifactSHA [32]byte, l clusterpir.Layout) error {
	if b.SourceSHA256 != artifactSHA {
		return fmt.Errorf("bundle was built from artifact sha256=%x, serving %x", b.SourceSHA256, artifactSHA)
	}
	if b.Count != l.Count || b.Size != l.Size || b.Dim != l.Dim || b.N != l.N() {
		return fmt.Errorf("bundle C=%d s=%d dimension=%d N=%d differ from DB C=%d s=%d dimension=%d N=%d", b.Count, b.Size, b.Dim, b.N, l.Count, l.Size, l.Dim, l.N())
	}
	if b.PIR.IsSet() && (b.PIR.EntryCount != uint64(l.Count) || b.PIR.EntryBytes != uint64(l.EntryBytes())) {
		return fmt.Errorf("bundle PIR entries=%d entry bytes=%d differ from DB entries=%d entry bytes=%d", b.PIR.EntryCount, b.PIR.EntryBytes, l.Count, l.EntryBytes())
	}
	return nil
}

// serve builds the DB, loads and checks the bundle, writes the report as one
// JSON line to out once listening, and serves until ctx is done.
func serve(ctx context.Context, args []string, out io.Writer) error {
	f := flag.NewFlagSet("search-server", flag.ContinueOnError)
	artifact := f.String("artifact", "", "clustering artifact from anchorann/cmd/cluster (required)")
	vectors := f.String("vectors", "", "original base vectors, fvecs or bvecs (required; first N records are read)")
	format := f.String("format", "auto", "vector format: auto, fvecs, or bvecs")
	bundlePath := f.String("bundle", "", "bundle from build-bundle for the same artifact (required)")
	addr := f.String("addr", ":50091", "listen address for PirService and AnchorBundle")
	partitions := f.Int("partitions", 1, "Piano partitions P (contiguous cluster ID ranges)")
	failure := f.Uint64("failure-prob-log2", 40, "Piano failure probability 2^-x")
	maxMsgMB := f.Int("max-msg-mb", 1024, "gRPC max message MiB (one preprocess reply carries all hints of a partition)")
	chunk := f.Int("chunk-bytes", bundlerpc.DefaultChunkBytes, "bytes per streamed bundle chunk")
	if err := f.Parse(args); err != nil {
		return err
	}
	if *artifact == "" || *vectors == "" || *bundlePath == "" {
		return fmt.Errorf("-artifact, -vectors and -bundle are required")
	}
	if *maxMsgMB <= 0 {
		return fmt.Errorf("-max-msg-mb must be positive")
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
	d, err := anchorann.ReadVectors(*vectors, *format, a.N)
	if err != nil {
		return err
	}
	read := time.Now()
	db, err := clusterpir.Build(a, d)
	if err != nil {
		return err
	}
	built := time.Now()
	a, d = nil, anchorann.Dataset{}
	runtime.GC()

	data, err := os.ReadFile(*bundlePath)
	if err != nil {
		return fmt.Errorf("read bundle: %w", err)
	}
	bundleSrv, err := bundlerpc.NewServer(data, *chunk)
	if err != nil {
		return err
	}
	b, err := bundle.Decode(data, bundle.Expect{})
	if err != nil {
		return err
	}
	if err := checkBundle(b, sum, db.Layout); err != nil {
		return err
	}
	bundleLoaded := time.Now()

	pirSrv, err := db.NewServer(*partitions, *failure)
	if err != nil {
		return err
	}
	meta, err := pirSrv.GetMetadata(ctx, &rpc.MetadataRequest{})
	if err != nil {
		return err
	}
	lis, err := net.Listen("tcp", *addr)
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}
	maxBytes := *maxMsgMB << 20
	gs := grpc.NewServer(grpc.MaxRecvMsgSize(maxBytes), grpc.MaxSendMsgSize(maxBytes))
	rpc.RegisterPirServiceServer(gs, pirSrv)
	bundlerpc.RegisterAnchorBundleServer(gs, bundleSrv)
	l := db.Layout
	rep := report{
		Addr: lis.Addr().String(), Artifact: *artifact, ArtifactSHA256: fmt.Sprintf("%x", sum), Vectors: *vectors,
		Bundle: *bundlePath, BundleSHA256: bundle.Checksum(data), BundleBytes: len(data), BundlePIR: b.PIR,
		Layout: l, EntryBytes: l.EntryBytes(), DBBytes: l.DBBytes(),
		Partitions: *partitions, PartitionSize: meta.PartitionSize, ChunkSize: meta.ChunkSize, SetSize: meta.SetSize, FailureProbLog2: *failure,
		ArtifactSeconds: loaded.Sub(start).Seconds(), VectorsSeconds: read.Sub(loaded).Seconds(), BuildSeconds: built.Sub(read).Seconds(),
		BundleSeconds: bundleLoaded.Sub(built).Seconds(), PeakRSSBytes: peakRSS(), MaxMessageBytes: maxBytes,
		GoVersion: runtime.Version(), Platform: runtime.GOOS + "/" + runtime.GOARCH, GOMAXPROCS: runtime.GOMAXPROCS(0),
	}
	b = nil
	if err := json.NewEncoder(out).Encode(rep); err != nil {
		lis.Close()
		return fmt.Errorf("write report: %w", err)
	}
	go func() {
		<-ctx.Done()
		gs.GracefulStop()
	}()
	return gs.Serve(lis)
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := serve(ctx, os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "anchorann search-server:", err)
		os.Exit(1)
	}
}
