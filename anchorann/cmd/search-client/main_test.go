package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"example.com/private-search/rpc"
	"google.golang.org/grpc"

	"privatesearch/anchorann"
	"privatesearch/anchorann/bundle"
	"privatesearch/anchorann/bundlerpc"
	"privatesearch/anchorann/clusterpir"
	"privatesearch/anchorann/nsg"
)

const (
	count = 16
	size  = 3
	dim   = 4
)

type env struct {
	addr, bundle, queries, gt string
	queryN                    int
}

func writeVectors(t *testing.T, path string, values []float32) {
	t.Helper()
	var buf bytes.Buffer
	for i := 0; i < len(values)/dim; i++ {
		binary.Write(&buf, binary.LittleEndian, int32(dim))
		for _, x := range values[i*dim : (i+1)*dim] {
			binary.Write(&buf, binary.LittleEndian, math.Float32bits(x))
		}
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

// setup serves a synthetic cluster PIR DB and its bundle on one loopback gRPC
// server (as search-server does) and writes queries and an exact ivecs GT.
func setup(t *testing.T) env {
	t.Helper()
	dir := t.TempDir()
	rng := rand.New(rand.NewSource(3))
	n := count * size
	d := anchorann.Dataset{N: n, Dim: dim, Values: make([]float32, n*dim)}
	for i := range d.Values {
		d.Values[i] = float32(rng.Intn(32))
	}
	perm := rng.Perm(n)
	a := &anchorann.Artifact{Version: anchorann.FormatVersion, N: n, Size: size, Count: count, Dim: dim, Seed: 3, Method: anchorann.Common, Iterations: 1}
	for c := 0; c < count; c++ {
		ids := make([]uint32, size)
		mean := make([]float64, dim)
		for j := range ids {
			ids[j] = uint32(perm[c*size+j])
			for k := 0; k < dim; k++ {
				mean[k] += float64(d.Values[int(ids[j])*dim+k]) / size
			}
		}
		a.Members = append(a.Members, ids)
		a.Anchors = append(a.Anchors, mean)
	}
	db, err := clusterpir.Build(a, d)
	if err != nil {
		t.Fatal(err)
	}
	pirSrv, err := db.NewServer(1, 40)
	if err != nil {
		t.Fatal(err)
	}
	b, _, err := bundle.FromArtifact(a, sha256.Sum256(nil), nsg.Params{KNN: 6, L: 12, R: 4, CNSG: 24, Workers: 1}, bundle.PIRParams{})
	if err != nil {
		t.Fatal(err)
	}
	e := env{bundle: filepath.Join(dir, "b.bundle"), queries: filepath.Join(dir, "q.fvecs"), gt: filepath.Join(dir, "gt.ivecs"), queryN: 6}
	data, err := bundle.Save(e.bundle, b)
	if err != nil {
		t.Fatal(err)
	}
	bundleSrv, err := bundlerpc.NewServer(data, 128)
	if err != nil {
		t.Fatal(err)
	}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	gs := grpc.NewServer()
	rpc.RegisterPirServiceServer(gs, pirSrv)
	bundlerpc.RegisterAnchorBundleServer(gs, bundleSrv)
	go gs.Serve(lis)
	t.Cleanup(gs.Stop)
	e.addr = lis.Addr().String()

	q := make([]float32, e.queryN*dim)
	for i := range q {
		q[i] = float32(rng.Intn(32)) + 0.5
	}
	writeVectors(t, e.queries, q)
	var gt bytes.Buffer
	for i := 0; i < e.queryN; i++ {
		rs := nsg.Exact(d.Values, dim, q[i*dim:(i+1)*dim], 12)
		binary.Write(&gt, binary.LittleEndian, int32(len(rs)))
		for _, r := range rs {
			binary.Write(&gt, binary.LittleEndian, r.ID)
		}
	}
	if err := os.WriteFile(e.gt, gt.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return e
}

func runJSON(t *testing.T, args []string) report {
	t.Helper()
	var out bytes.Buffer
	if err := run(context.Background(), args, &out); err != nil {
		t.Fatal(err)
	}
	var rep report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatal(err)
	}
	return rep
}

func TestRunOverGRPC(t *testing.T) {
	e := setup(t)
	perQuery := filepath.Join(t.TempDir(), "q.jsonl")
	rep := runJSON(t, []string{"-addr", e.addr, "-queries", e.queries, "-gt", e.gt, "-k", "5",
		"-expect-c", "16", "-expect-dim", "4", "-expect-s", "3",
		"-n-probe", "2,6", "-l-search", "4,16", "-modes", "nsg,exact", "-per-query", perQuery})
	if rep.Count != count || rep.Size != size || rep.Dimension != dim || rep.Queries != e.queryN || rep.BundleFetch == nil || rep.BundleFetch.Chunks < 2 {
		t.Fatalf("report header %+v", rep)
	}
	if len(rep.Probes) != 2 {
		t.Fatalf("probes %d", len(rep.Probes))
	}
	if rep.SourceSHA256 != fmt.Sprintf("%x", sha256.Sum256(nil)) {
		t.Fatalf("artifact sha256 %s", rep.SourceSHA256)
	}
	for _, pr := range rep.Probes {
		// C=16: MaxQueryNum 11 and 6 backups per chunk of 8, so n_probe=6
		// fits one construction and never exhausts a chunk.
		if pr.MaxQueryNum != 11 || pr.BackupPerChunk != 6 || pr.QueriesPerRefresh != 11/pr.NProbe || pr.DBBytes <= 0 {
			t.Fatalf("n_probe=%d max_query_num=%d db_bytes=%d", pr.NProbe, pr.MaxQueryNum, pr.DBBytes)
		}
	}
	lines := 0
	for _, pr := range rep.Probes {
		if pr.Offline.DownBytes == 0 || pr.Offline.Seconds <= 0 {
			t.Fatalf("n_probe=%d offline %+v", pr.NProbe, pr.Offline)
		}
		// n_probe=6 drops L_search=4 (< n_probe).
		want := map[int]int{2: 3, 6: 2}[pr.NProbe]
		if len(pr.Runs) != want {
			t.Fatalf("n_probe=%d runs=%d want %d", pr.NProbe, len(pr.Runs), want)
		}
		for i, r := range pr.Runs {
			lines += r.Queries
			if r.Recall == nil || r.IncompleteQueries != 0 || r.MissedClusters != 0 || r.Requests != r.NProbe*e.queryN {
				t.Fatalf("run %+v", r)
			}
			if r.OneRecall == nil || r.CandidateOneContainment == nil ||
				*r.OneRecall != float64(r.NearestHits)/float64(r.Queries) ||
				r.CandidateNearestHits != r.NearestHits || *r.OneRecall != *r.CandidateOneContainment {
				t.Fatalf("nearest-neighbor metrics %+v", r)
			}
			// Exact reranking of the retrieved points loses nothing: every GT
			// neighbor among the candidates reaches the final top-k.
			if r.CandidateContainment == nil || r.CandidateHits != r.RecallHits || *r.CandidateContainment != *r.Recall {
				t.Fatalf("candidate containment %+v", r)
			}
			if r.UpBytes <= 0 || r.DownBytes <= 0 || r.Online.Mean <= 0 || r.FetchSeconds <= 0 || r.CacheStorageBytes <= 0 {
				t.Fatalf("run measurements %+v", r)
			}
			if (i > 0) != (r.ForcedRefreshSeconds > 0) {
				t.Fatalf("run %d forced refresh %v", i, r.ForcedRefreshSeconds)
			}
			if r.Mode == "exact" && r.AnchorEvaluations != count {
				t.Fatalf("exact mode evaluations %v", r.AnchorEvaluations)
			}
		}
	}
	f, err := os.Open(perQuery)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(nil, 1<<20)
	got := 0
	for sc.Scan() {
		var rec queryRecord
		if err := json.Unmarshal(sc.Bytes(), &rec); err != nil {
			t.Fatal(err)
		}
		if rec.Result == nil || len(rec.Result.IDs) != 5 || len(rec.Result.Clusters) != rec.NProbe || rec.Hits == nil {
			t.Fatalf("record %+v", rec)
		}
		got++
	}
	if got != lines {
		t.Fatalf("per-query lines=%d want %d", got, lines)
	}
}

// The batch mode returns the same answers with one round trip per query
// (one partition, one round) and records the mode in the report.
func TestRunBatchModeMatchesSequential(t *testing.T) {
	e := setup(t)
	args := []string{"-addr", e.addr, "-queries", e.queries, "-gt", e.gt, "-k", "5", "-n-probe", "2,6", "-l-search", "8", "-modes", "nsg,exact", "-head-queries", "3"}
	seq := runJSON(t, args)
	bat := runJSON(t, append(args, "-pir-rpc", "batch"))
	for i, pr := range bat.Probes {
		if pr.Batch.RPC != "batch" || seq.Probes[i].Batch.RPC != "sequential" {
			t.Fatalf("modes %q %q", seq.Probes[i].Batch.RPC, pr.Batch.RPC)
		}
		for j, r := range pr.Runs {
			s := seq.Probes[i].Runs[j]
			if r.RPCs != e.queryN || s.RPCs != s.Requests || r.Requests != s.Requests {
				t.Fatalf("n_probe=%d %s: rpcs batch=%d sequential=%d requests=%d", pr.NProbe, r.Mode, r.RPCs, s.RPCs, s.Requests)
			}
			if r.RecallHits != s.RecallHits || r.NearestHits != s.NearestHits || r.MissedClusters != s.MissedClusters || r.CandidateHits != s.CandidateHits {
				t.Fatalf("n_probe=%d %s: batch %+v sequential %+v", pr.NProbe, r.Mode, r, s)
			}
			h, sh := r.Head, s.Head
			if h == nil || sh == nil || h.Queries != 3 || h.RPCs != 3 || sh.RPCs != h.Requests || h.Requests != 3*r.NProbe ||
				h.OneRecall == nil || *h.OneRecall != *sh.OneRecall || h.Online.Mean <= 0 || h.DownBytes <= 0 {
				t.Fatalf("n_probe=%d %s: first queries batch %+v sequential %+v", pr.NProbe, r.Mode, h, sh)
			}
		}
	}
}

func TestRunWithBundleFileAndNoGT(t *testing.T) {
	e := setup(t)
	rep := runJSON(t, []string{"-addr", e.addr, "-bundle", e.bundle, "-queries", e.queries, "-query-limit", "3", "-k", "2", "-n-probe", "1", "-l-search", "2", "-refresh-between-runs=false", "-modes", "nsg,exact"})
	if rep.BundleFetch != nil || rep.BundleSource != e.bundle || rep.Queries != 3 {
		t.Fatalf("report %+v", rep)
	}
	for _, r := range rep.Probes[0].Runs {
		if r.Recall != nil || r.OneRecall != nil || r.CandidateContainment != nil || r.CandidateOneContainment != nil || r.ForcedRefreshSeconds != 0 {
			t.Fatalf("run %+v", r)
		}
	}
	// A separate bundle address works the same way as -addr.
	rep = runJSON(t, []string{"-addr", e.addr, "-bundle-addr", e.addr, "-queries", e.queries, "-n-probe", "4", "-l-search", "4"})
	if rep.BundleSource != e.addr || rep.BundleFetch == nil {
		t.Fatalf("report %+v", rep)
	}
}

func TestRunRejectsBadArguments(t *testing.T) {
	e := setup(t)
	shortGT := filepath.Join(t.TempDir(), "short.ivecs")
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, int32(1))
	binary.Write(&buf, binary.LittleEndian, uint32(0))
	os.WriteFile(shortGT, buf.Bytes(), 0o644)
	otherDim := filepath.Join(t.TempDir(), "q3.fvecs")
	os.WriteFile(otherDim, []byte{3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0o644)
	base := []string{"-addr", e.addr, "-queries", e.queries}
	for _, tc := range []struct {
		args []string
		want string
	}{
		{[]string{"-addr", e.addr}, "-queries is required"},
		{append(base, "-k", "0"), "positive"},
		{append(base, "-n-probe", "0"), "-n-probe"},
		{append(base, "-l-search", "a"), "-l-search"},
		{append(base, "-modes", "fast"), "unknown mode"},
		{append(base, "-pir-rpc", "fast"), "-pir-rpc"},
		{append(base, "-bundle", e.bundle, "-bundle-addr", e.addr), "exclusive"},
		{append(base, "-expect-c", "17"), "17"},
		{append(base, "-n-probe", "17", "-l-search", "17"), "n_probe=17"},
		{append(base, "-k", "40", "-n-probe", "2", "-l-search", "4"), "exceeds retrieved points"},
		{append(base, "-gt", shortGT), "ground truth"},
		{[]string{"-addr", e.addr, "-queries", otherDim}, "dimension"},
	} {
		err := run(context.Background(), tc.args, &bytes.Buffer{})
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("%v: err=%v want %q", tc.args, err, tc.want)
		}
	}
}

func TestRunChecksPinnedSHA256(t *testing.T) {
	e := setup(t)
	data, err := os.ReadFile(e.bundle)
	if err != nil {
		t.Fatal(err)
	}
	bundleSHA := bundle.Checksum(data)
	artifactSHA := fmt.Sprintf("%x", sha256.Sum256(nil))
	base := []string{"-addr", e.addr, "-queries", e.queries, "-k", "2", "-n-probe", "1", "-l-search", "2"}
	// Matching pins (case-insensitive) pass, for a fetched and a file bundle.
	rep := runJSON(t, append(base, "-expect-bundle-sha256", strings.ToUpper(bundleSHA), "-expect-artifact-sha256", artifactSHA))
	if rep.BundleSHA256 != bundleSHA || rep.SourceSHA256 != artifactSHA {
		t.Fatalf("report %+v", rep)
	}
	runJSON(t, append(base, "-bundle", e.bundle, "-expect-bundle-sha256", bundleSHA))
	other := strings.Repeat("0", 64)
	for _, tc := range []struct {
		args []string
		want string
	}{
		{append(base, "-expect-bundle-sha256", other), "bundle sha256=" + bundleSHA + ", expected " + other},
		{append(base, "-expect-artifact-sha256", other), "built from artifact sha256=" + artifactSHA},
		{append(base, "-bundle", e.bundle, "-expect-bundle-sha256", other), "expected " + other},
	} {
		err := run(context.Background(), tc.args, &bytes.Buffer{})
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("%v: err=%v want %q", tc.args, err, tc.want)
		}
	}
}

func TestCheckPins(t *testing.T) {
	if err := checkPins("ab", "cd", "", ""); err != nil {
		t.Fatal(err)
	}
	if err := checkPins("ab", "cd", "AB", "CD"); err != nil {
		t.Fatal(err)
	}
	if checkPins("ab", "cd", "ac", "") == nil || checkPins("ab", "cd", "", "ce") == nil {
		t.Fatal("mismatch accepted")
	}
}

func TestHitsAndSummarize(t *testing.T) {
	if h := hits([]uint32{1, 2, 3}, []uint32{3, 4, 1}); h != 2 {
		t.Fatalf("hits=%d", h)
	}
	l := summarize([]float64{4, 1, 3, 2})
	if l.Mean != 2.5 || l.Max != 4 || l.P50 != 3 {
		t.Fatalf("summary %+v", l)
	}
	if (summarize(nil) != latency{}) {
		t.Fatal("empty summary")
	}
}

func TestReadExistingNpyGroundTruth(t *testing.T) {
	path := filepath.Join(t.TempDir(), "groundtruth_1M.npy")
	if err := saveIntMatrix(path, [][]int{{7, 1, 2}, {9, 3, 4}}); err != nil {
		t.Fatal(err)
	}
	got, err := readIvecs(path, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || len(got[0]) != 2 || got[0][0] != 7 || got[1][0] != 9 {
		t.Fatalf("GT ordering changed: %v", got)
	}
	if err := saveIntMatrix(path, [][]int{{-1, 2}}); err != nil {
		t.Fatal(err)
	}
	if _, err := readIvecs(path, 1, 2); err == nil {
		t.Fatal("negative ground truth ID accepted")
	}
}

func TestDeepAndGistVectorDimensions(t *testing.T) {
	for _, dimension := range []int{96, 960} {
		t.Run(fmt.Sprint(dimension), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "queries.fvecs")
			f, err := os.Create(path)
			if err != nil {
				t.Fatal(err)
			}
			for row := 0; row < 2; row++ {
				if err := binary.Write(f, binary.LittleEndian, int32(dimension)); err != nil {
					t.Fatal(err)
				}
				coords := make([]float32, dimension)
				coords[dimension-1] = float32(row + 1)
				if err := binary.Write(f, binary.LittleEndian, coords); err != nil {
					t.Fatal(err)
				}
			}
			if err := f.Close(); err != nil {
				t.Fatal(err)
			}
			got, err := anchorann.ReadVectors(path, "auto", 2)
			if err != nil || got.N != 2 || got.Dim != dimension || got.Values[2*dimension-1] != 2 {
				t.Fatalf("shape/coordinates mismatch N=%d dim=%d err=%v", got.N, got.Dim, err)
			}
		})
	}
}
