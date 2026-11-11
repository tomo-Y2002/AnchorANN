package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"example.com/private-search/rpc"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"privatesearch/anchorann"
	"privatesearch/anchorann/bundle"
	"privatesearch/anchorann/bundlerpc"
	"privatesearch/anchorann/clusterpir"
	"privatesearch/anchorann/nsg"
	"privatesearch/anchorann/search"
)

type inputs struct {
	artifact, vectors, bundle string
	a                         *anchorann.Artifact
	d                         anchorann.Dataset
}

// writeInputs writes a random fvecs dataset, an artifact with mean anchors
// and a bundle built from that artifact file (sourceSHA overrides the
// recorded artifact digest when non-nil).
func writeInputs(t *testing.T, count, size, dim int, sourceSHA *[32]byte) inputs {
	t.Helper()
	dir := t.TempDir()
	rng := rand.New(rand.NewSource(5))
	n := count * size
	in := inputs{d: anchorann.Dataset{N: n, Dim: dim, Values: make([]float32, n*dim)}}
	var buf bytes.Buffer
	for i := 0; i < n; i++ {
		binary.Write(&buf, binary.LittleEndian, int32(dim))
		for j := 0; j < dim; j++ {
			x := float32(rng.Intn(64))
			in.d.Values[i*dim+j] = x
			binary.Write(&buf, binary.LittleEndian, math.Float32bits(x))
		}
	}
	in.vectors = filepath.Join(dir, "v.fvecs")
	if err := os.WriteFile(in.vectors, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	perm := rng.Perm(n)
	in.a = &anchorann.Artifact{Version: anchorann.FormatVersion, N: n, Size: size, Count: count, Dim: dim, Seed: 5, Method: anchorann.Common, Iterations: 1}
	for c := 0; c < count; c++ {
		ids := make([]uint32, size)
		mean := make([]float64, dim)
		for j := range ids {
			ids[j] = uint32(perm[c*size+j])
			for k := 0; k < dim; k++ {
				mean[k] += float64(in.d.Values[int(ids[j])*dim+k]) / float64(size)
			}
		}
		in.a.Members = append(in.a.Members, ids)
		in.a.Anchors = append(in.a.Anchors, mean)
	}
	in.artifact = filepath.Join(dir, "a.anchor")
	if err := anchorann.Save(in.artifact, in.a); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(in.artifact)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	if sourceSHA != nil {
		sum = *sourceSHA
	}
	l, _ := clusterpir.NewLayout(count, size, dim)
	b, _, err := bundle.FromArtifact(in.a, sum, nsg.Params{KNN: 8, L: 16, R: 6, CNSG: 32, Workers: 1}, clusterpir.DefaultBatch(2, 1).PIRParams(l, 1))
	if err != nil {
		t.Fatal(err)
	}
	in.bundle = filepath.Join(dir, "b.bundle")
	if _, err := bundle.Save(in.bundle, b); err != nil {
		t.Fatal(err)
	}
	return in
}

func start(t *testing.T, args []string) (report, context.CancelFunc, chan error) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	pr, pw := io.Pipe()
	done := make(chan error, 1)
	go func() {
		err := serve(ctx, args, pw)
		pw.CloseWithError(err)
		done <- err
	}()
	var rep report
	if err := json.NewDecoder(pr).Decode(&rep); err != nil {
		cancel()
		t.Fatalf("server did not start: %v (%v)", err, <-done)
	}
	return rep, cancel, done
}

// TestServeEndToEnd starts the server, fetches the bundle over AnchorBundle,
// preprocesses Piano over PirService on the same address, and checks the
// pipeline's top-k against the plaintext answer over the selected clusters.
func TestServeEndToEnd(t *testing.T) {
	in := writeInputs(t, 24, 4, 6, nil)
	rep, cancel, done := start(t, []string{"-artifact", in.artifact, "-vectors", in.vectors, "-bundle", in.bundle, "-addr", "127.0.0.1:0", "-chunk-bytes", "512"})
	if rep.Layout.Count != 24 || rep.EntryBytes != 4*4+4*4*6 || rep.BundlePIR.EntryCount != 24 || rep.BundleBytes == 0 || rep.Partitions != 1 {
		t.Fatalf("report %+v", rep)
	}
	conn, err := grpc.Dial(rep.Addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	ctx := context.Background()
	b, _, st, err := bundlerpc.Fetch(ctx, bundlerpc.NewAnchorBundleClient(conn), bundle.Expect{Count: 24, Dim: 6, Size: 4})
	if err != nil || st.SHA256 != rep.BundleSHA256 {
		t.Fatalf("bundle fetch: %v sha=%s", err, st.SHA256)
	}
	ix, err := bundle.NewIndex(b)
	if err != nil {
		t.Fatal(err)
	}
	p := search.Params{LSearch: 8, NProbe: 2, K: 4}
	fe, err := clusterpir.NewFetcher(ctx, rpc.NewPirServiceClient(conn), clusterpir.BatchConfig{NProbe: p.NProbe})
	if err != nil {
		t.Fatal(err)
	}
	sc, _ := search.New(ix, search.Piano{F: fe})
	if err := sc.Preprocess(ctx); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(9))
	for qi := 0; qi < 5; qi++ {
		q := make([]float32, 6)
		for j := range q {
			q[j] = float32(rng.Intn(64)) + 0.5
		}
		res, err := sc.Search(ctx, q, p)
		if err != nil {
			t.Fatal(err)
		}
		var all []nsg.Result
		for _, c := range res.Clusters {
			for _, id := range in.a.Members[c] {
				all = append(all, nsg.Result{ID: id, Distance: nsg.Distance(q, in.d.Values[int(id)*6:(int(id)+1)*6])})
			}
		}
		sort.Slice(all, func(i, j int) bool {
			return all[i].Distance < all[j].Distance || (all[i].Distance == all[j].Distance && all[i].ID < all[j].ID)
		})
		want := make([]uint32, p.K)
		for i := range want {
			want[i] = all[i].ID
		}
		if !reflect.DeepEqual(res.IDs, want) {
			t.Fatalf("q=%d: got %v want %v", qi, res.IDs, want)
		}
	}
	cancel()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestServeRejectsBadInputs(t *testing.T) {
	in := writeInputs(t, 12, 2, 3, nil)
	other := [32]byte{1}
	foreign := writeInputs(t, 12, 2, 3, &other)
	smaller := writeInputs(t, 6, 2, 3, nil)
	for _, tc := range []struct {
		args []string
		want string
	}{
		{[]string{"-artifact", in.artifact, "-vectors", in.vectors}, "required"},
		{[]string{"-artifact", in.artifact, "-vectors", in.vectors, "-bundle", foreign.bundle}, "built from artifact"},
		{[]string{"-artifact", in.artifact, "-vectors", in.vectors, "-bundle", smaller.bundle}, "built from artifact"},
		{[]string{"-artifact", in.artifact, "-vectors", in.vectors, "-bundle", in.vectors}, "bundle"},
		{[]string{"-artifact", in.artifact, "-vectors", in.vectors, "-bundle", in.bundle, "-max-msg-mb", "0"}, "positive"},
		{[]string{"-artifact", in.artifact, "-vectors", in.vectors, "-bundle", in.bundle, "-partitions", "13"}, "partition"},
	} {
		err := serve(context.Background(), append(tc.args, "-addr", "127.0.0.1:0"), io.Discard)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("%v: err=%v want %q", tc.args, err, tc.want)
		}
	}
}

func TestCheckBundle(t *testing.T) {
	in := writeInputs(t, 12, 2, 3, nil)
	b, err := bundle.Load(in.bundle, bundle.Expect{})
	if err != nil {
		t.Fatal(err)
	}
	l, _ := clusterpir.NewLayout(12, 2, 3)
	if err := checkBundle(b, b.SourceSHA256, l); err != nil {
		t.Fatal(err)
	}
	wrongShape, _ := clusterpir.NewLayout(6, 4, 3)
	if err := checkBundle(b, b.SourceSHA256, wrongShape); err == nil || !strings.Contains(err.Error(), "differ") {
		t.Fatalf("shape mismatch: %v", err)
	}
	b.PIR.EntryBytes += 8
	if err := checkBundle(b, b.SourceSHA256, l); err == nil || !strings.Contains(err.Error(), "PIR") {
		t.Fatalf("PIR mismatch: %v", err)
	}
	b.PIR = bundle.PIRParams{}
	if err := checkBundle(b, b.SourceSHA256, l); err != nil {
		t.Fatalf("unset PIR params rejected: %v", err)
	}
}
