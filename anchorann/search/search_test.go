package search

import (
	"context"
	"crypto/sha256"
	"errors"
	"math/rand"
	"net"
	"os"
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
)

// fixture returns a dataset of C·s integer-valued points, an artifact whose
// clusters are a seeded permutation and whose anchors are the cluster means,
// and the bundle built from it with matching PIR parameters.
func fixture(t testing.TB, count, size, dim int, seed int64) (*anchorann.Artifact, anchorann.Dataset, *bundle.Bundle) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	n := count * size
	d := anchorann.Dataset{N: n, Dim: dim, Values: make([]float32, n*dim)}
	for i := range d.Values {
		d.Values[i] = float32(rng.Intn(64))
	}
	perm := rng.Perm(n)
	a := &anchorann.Artifact{Version: anchorann.FormatVersion, N: n, Size: size, Count: count, Dim: dim, Seed: seed, Method: anchorann.Common, Iterations: 1}
	for c := 0; c < count; c++ {
		ids := make([]uint32, size)
		mean := make([]float64, dim)
		for j := range ids {
			ids[j] = uint32(perm[c*size+j])
			for k := 0; k < dim; k++ {
				mean[k] += float64(d.Values[int(ids[j])*dim+k]) / float64(size)
			}
		}
		a.Members = append(a.Members, ids)
		a.Anchors = append(a.Anchors, mean)
	}
	l, err := clusterpir.NewLayout(count, size, dim)
	if err != nil {
		t.Fatal(err)
	}
	pir := clusterpir.DefaultBatch(1, 1).PIRParams(l, 1)
	b, _, err := bundle.FromArtifact(a, sha256.Sum256(nil), nsg.Params{KNN: 8, L: 16, R: 6, CNSG: 32, Workers: 2}, pir)
	if err != nil {
		t.Fatal(err)
	}
	return a, d, b
}

func index(t testing.TB, b *bundle.Bundle) *bundle.Index {
	t.Helper()
	ix, err := bundle.NewIndex(b)
	if err != nil {
		t.Fatal(err)
	}
	return ix
}

// plainRetriever answers from the artifact and dataset without any PIR.
type plainRetriever struct {
	a           *anchorann.Artifact
	d           anchorann.Dataset
	miss        map[int]bool
	fail        error
	wrong       bool // return a different cluster than requested
	calls       [][]int
	pre, refrsh int
}

func (r *plainRetriever) Preprocess(context.Context) error { r.pre++; return nil }
func (r *plainRetriever) Refresh(context.Context) error    { r.refrsh++; return nil }

func (r *plainRetriever) Fetch(_ context.Context, ids []int) (*Fetched, error) {
	r.calls = append(r.calls, append([]int(nil), ids...))
	if r.fail != nil {
		return nil, r.fail
	}
	f := &Fetched{Clusters: make([]*Cluster, len(ids)), Stats: FetchStats{Requests: len(ids), UpBytes: 10 * int64(len(ids)), DownBytes: 100 * int64(len(ids))}}
	for i, c := range ids {
		if r.miss[c] {
			f.Missed = append(f.Missed, c)
			continue
		}
		src := c
		if r.wrong {
			src = (c + 1) % r.a.Count
		}
		f.Clusters[i] = r.cluster(src)
	}
	return f, nil
}

func (r *plainRetriever) cluster(c int) *Cluster {
	cl := &Cluster{ID: c, IDs: append([]uint32(nil), r.a.Members[c]...)}
	for _, id := range cl.IDs {
		cl.Vectors = append(cl.Vectors, r.d.Values[int(id)*r.d.Dim:(int(id)+1)*r.d.Dim]...)
	}
	return cl
}

// bruteForce is the plaintext exact top-k over the members of clusters.
func bruteForce(a *anchorann.Artifact, d anchorann.Dataset, q []float32, clusters []int, k int) []uint32 {
	var all []nsg.Result
	for _, c := range clusters {
		for _, id := range a.Members[c] {
			all = append(all, nsg.Result{ID: id, Distance: nsg.Distance(q, d.Values[int(id)*d.Dim:(int(id)+1)*d.Dim])})
		}
	}
	sort.Slice(all, func(i, j int) bool {
		return all[i].Distance < all[j].Distance || (all[i].Distance == all[j].Distance && all[i].ID < all[j].ID)
	})
	out := make([]uint32, min(k, len(all)))
	for i := range out {
		out[i] = all[i].ID
	}
	return out
}

func queries(n, dim int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([][]float32, n)
	for i := range out {
		out[i] = make([]float32, dim)
		for j := range out[i] {
			out[i][j] = float32(rng.Intn(64)) + 0.5
		}
	}
	return out
}

func TestSearchMatchesPlaintextOfSelectedClusters(t *testing.T) {
	a, d, b := fixture(t, 40, 4, 6, 1)
	r := &plainRetriever{a: a, d: d}
	c, err := New(index(t, b), r)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []Params{{LSearch: 8, NProbe: 3, K: 5}, {LSearch: 4, NProbe: 4, K: 16}, {NProbe: 2, K: 1, ExactAnchors: true}} {
		for qi, q := range queries(20, 6, 2) {
			res, err := c.Search(context.Background(), q, p)
			if err != nil {
				t.Fatalf("%+v q=%d: %v", p, qi, err)
			}
			if !res.Complete || res.Shortfall != 0 || len(res.MissedClusters) != 0 || res.Error != "" {
				t.Fatalf("%+v q=%d: incomplete %+v", p, qi, res)
			}
			if len(res.Clusters) != p.NProbe || res.Candidates != p.NProbe*4 || !reflect.DeepEqual(r.calls[len(r.calls)-1], res.Clusters) {
				t.Fatalf("%+v q=%d: clusters=%v candidates=%d call=%v", p, qi, res.Clusters, res.Candidates, r.calls[len(r.calls)-1])
			}
			if want := bruteForce(a, d, q, res.Clusters, p.K); !reflect.DeepEqual(res.IDs, want) {
				t.Fatalf("%+v q=%d: got %v want %v", p, qi, res.IDs, want)
			}
			// The fetched clusters are those of the selected anchors.
			anchors, clusters, _, err := c.SelectClusters(q, p)
			if err != nil || !reflect.DeepEqual(anchors, res.Anchors) || !reflect.DeepEqual(clusters, res.Clusters) {
				t.Fatalf("%+v q=%d: SelectClusters %v %v %v", p, qi, anchors, clusters, err)
			}
			if res.UpBytes != 10*int64(p.NProbe) || res.DownBytes != 100*int64(p.NProbe) || res.Requests != p.NProbe {
				t.Fatalf("%+v q=%d: traffic not propagated: %+v", p, qi, res)
			}
			if res.Timing.OnlineSeconds < res.Timing.AnchorSeconds || res.Timing.OnlineSeconds <= 0 {
				t.Fatalf("timing %+v", res.Timing)
			}
		}
	}
}

func TestDiagnosticFullProbeMatchesExhaustiveSearch(t *testing.T) {
	a, d, b := fixture(t, 30, 3, 5, 3)
	c, _ := New(index(t, b), &plainRetriever{a: a, d: d})
	all := make([]int, a.Count)
	for i := range all {
		all[i] = i
	}
	for _, p := range []Params{{NProbe: 30, K: 10, ExactAnchors: true}, {LSearch: 30, NProbe: 30, K: 7}} {
		for qi, q := range queries(15, 5, 4) {
			res, err := c.Search(context.Background(), q, p)
			if err != nil {
				t.Fatal(err)
			}
			want := bruteForce(a, d, q, all, p.K)
			// Also against a scan of the raw dataset (no clustering at all).
			raw := nsg.Exact(d.Values, d.Dim, q, p.K)
			for i, r := range raw {
				if want[i] != r.ID {
					t.Fatalf("reference mismatch %v %v", want, raw)
				}
			}
			if !reflect.DeepEqual(res.IDs, want) {
				t.Fatalf("%+v q=%d: got %v want %v", p, qi, res.IDs, want)
			}
		}
	}
	res, _ := c.Search(context.Background(), queries(1, 5, 9)[0], Params{NProbe: 30, K: 1, ExactAnchors: true})
	if res.AnchorDistanceEvaluations != 30 {
		t.Fatalf("diagnostic evaluations=%d", res.AnchorDistanceEvaluations)
	}
}

func TestDiagnosticUsesExactAnchorRanking(t *testing.T) {
	a, d, b := fixture(t, 50, 2, 4, 5)
	c, _ := New(index(t, b), &plainRetriever{a: a, d: d})
	for _, q := range queries(10, 4, 6) {
		res, err := c.Search(context.Background(), q, Params{LSearch: 0, NProbe: 5, K: 3, ExactAnchors: true})
		if err != nil {
			t.Fatal(err)
		}
		exact := nsg.Exact(b.Vectors, b.Dim, q, 5)
		for i, r := range exact {
			if res.Anchors[i] != r.ID || res.Clusters[i] != int(b.ClusterOf[r.ID]) {
				t.Fatalf("anchors %v exact %v", res.Anchors, exact)
			}
		}
	}
}

func TestParameterValidation(t *testing.T) {
	a, d, b := fixture(t, 10, 2, 3, 7)
	r := &plainRetriever{a: a, d: d}
	c, _ := New(index(t, b), r)
	q := []float32{1, 2, 3}
	for _, tc := range []struct {
		p    Params
		q    []float32
		want string
	}{
		{Params{LSearch: 4, NProbe: 2, K: 0}, q, "positive"},
		{Params{LSearch: 4, NProbe: 0, K: 1}, q, "positive"},
		{Params{LSearch: 4, NProbe: -1, K: 1}, q, "positive"},
		{Params{LSearch: 20, NProbe: 11, K: 1}, q, "exceeds cluster count"},
		{Params{NProbe: 11, K: 1, ExactAnchors: true}, q, "exceeds cluster count"},
		{Params{LSearch: 2, NProbe: 3, K: 1}, q, "smaller than n_probe"},
		{Params{LSearch: 4, NProbe: 2, K: 5}, q, "exceeds retrieved points"},
		{Params{LSearch: 4, NProbe: 2, K: 1}, []float32{1, 2}, "dimension"},
		{Params{LSearch: 4, NProbe: 2, K: 1}, []float32{1, 2, 3, 4}, "dimension"},
	} {
		_, err := c.Search(context.Background(), tc.q, tc.p)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("%+v dim=%d: err=%v want %q", tc.p, len(tc.q), err, tc.want)
		}
	}
	if len(r.calls) != 0 {
		t.Fatalf("invalid queries reached the retriever: %v", r.calls)
	}
	if _, err := New(nil, r); err == nil {
		t.Fatal("nil index accepted")
	}
	if _, err := New(index(t, b), nil); err == nil {
		t.Fatal("nil retriever accepted")
	}
}

func TestMissesAreReportedNotSilent(t *testing.T) {
	a, d, b := fixture(t, 20, 3, 4, 8)
	r := &plainRetriever{a: a, d: d}
	c, _ := New(index(t, b), r)
	q := queries(1, 4, 9)[0]
	p := Params{LSearch: 8, NProbe: 4, K: 12}
	full, err := c.Search(context.Background(), q, p)
	if err != nil {
		t.Fatal(err)
	}
	r.miss = map[int]bool{full.Clusters[1]: true}
	res, err := c.Search(context.Background(), q, p)
	if !errors.Is(err, ErrIncomplete) || res == nil {
		t.Fatalf("err=%v res=%v", err, res)
	}
	kept := []int{full.Clusters[0], full.Clusters[2], full.Clusters[3]}
	if res.Complete || !reflect.DeepEqual(res.MissedClusters, []int{full.Clusters[1]}) || res.Shortfall != 3 || len(res.IDs) != 9 || res.Error == "" {
		t.Fatalf("partial result not flagged: %+v", res)
	}
	if !reflect.DeepEqual(res.IDs, bruteForce(a, d, q, kept, 12)) {
		t.Fatalf("partial IDs %v", res.IDs)
	}
	// A miss with enough remaining candidates is still incomplete.
	res, err = c.Search(context.Background(), q, Params{LSearch: 8, NProbe: 4, K: 2})
	if !errors.Is(err, ErrIncomplete) || res.Complete || res.Shortfall != 0 || len(res.MissedClusters) != 1 {
		t.Fatalf("err=%v res=%+v", err, res)
	}
}

func TestFetchErrorsAreReturned(t *testing.T) {
	a, d, b := fixture(t, 20, 3, 4, 10)
	boom := errors.New("connection reset")
	r := &plainRetriever{a: a, d: d, fail: boom}
	c, _ := New(index(t, b), r)
	q := queries(1, 4, 11)[0]
	res, err := c.Search(context.Background(), q, Params{LSearch: 4, NProbe: 2, K: 2})
	if !errors.Is(err, boom) || res == nil || res.Error == "" || res.IDs != nil || res.Complete {
		t.Fatalf("err=%v res=%+v", err, res)
	}
	r.fail, r.wrong = nil, true
	if _, err := c.Search(context.Background(), q, Params{LSearch: 4, NProbe: 2, K: 2}); err == nil || errors.Is(err, ErrIncomplete) {
		t.Fatalf("wrong cluster accepted: %v", err)
	}
}

func TestCheckFetchedRejectsInconsistentRetriever(t *testing.T) {
	ok := &Cluster{ID: 1, IDs: []uint32{0}, Vectors: []float32{1, 2}}
	for name, f := range map[string]*Fetched{
		"length":        {Clusters: []*Cluster{ok}},
		"silent nil":    {Clusters: []*Cluster{ok, nil}},
		"missed+value":  {Clusters: []*Cluster{ok, {ID: 2, IDs: []uint32{1}, Vectors: []float32{1, 2}}}, Missed: []int{2}},
		"shape":         {Clusters: []*Cluster{ok, {ID: 2, IDs: []uint32{1}, Vectors: []float32{1}}}},
		"extra missing": {Clusters: []*Cluster{ok, nil}, Missed: []int{2, 3}},
	} {
		if _, err := checkFetched([]int{1, 2}, f, 2); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
}

func TestRerankMergesDuplicatesAndBreaksTiesByID(t *testing.T) {
	q := []float32{0, 0}
	cs := []*Cluster{
		{ID: 0, IDs: []uint32{5, 3}, Vectors: []float32{1, 0, 0, 1}},
		{ID: 1, IDs: []uint32{3, 9, 1}, Vectors: []float32{0, 1, 2, 0, 0, 3}},
	}
	got, n := Rerank(q, cs, 2, 10)
	want := []nsg.Result{{ID: 3, Distance: 1}, {ID: 5, Distance: 1}, {ID: 9, Distance: 4}, {ID: 1, Distance: 9}}
	if n != 4 || !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v (%d)", got, n)
	}
	got, _ = Rerank(q, cs, 2, 2)
	if !reflect.DeepEqual(got, want[:2]) {
		t.Fatalf("k=2 got %v", got)
	}
	if got, n := Rerank(q, nil, 2, 3); len(got) != 0 || n != 0 {
		t.Fatalf("empty got %v", got)
	}
}

func TestPreprocessAndRefreshGoThroughRetriever(t *testing.T) {
	a, d, b := fixture(t, 10, 2, 3, 12)
	r := &plainRetriever{a: a, d: d}
	c, _ := New(index(t, b), r)
	if err := c.Preprocess(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := c.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if r.pre != 1 || r.refrsh != 1 || c.Index() == nil {
		t.Fatalf("pre=%d refresh=%d", r.pre, r.refrsh)
	}
}

// TestPianoRetrieverMatchesPlaintext runs the pipeline through the real
// clusterpir Fetcher (in-process PirService) and checks every answer against
// the plaintext retriever.
func TestPianoRetrieverMatchesPlaintext(t *testing.T) {
	a, d, b := fixture(t, 64, 4, 8, 13)
	db, err := clusterpir.Build(a, d)
	if err != nil {
		t.Fatal(err)
	}
	srv, err := db.NewServer(1, 40)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	p := Params{LSearch: 10, NProbe: 3, K: 5}
	f, err := clusterpir.NewFetcher(ctx, clusterpir.InProcess(srv), clusterpir.BatchConfig{NProbe: p.NProbe})
	if err != nil {
		t.Fatal(err)
	}
	pir, plain := Piano{F: f}, &plainRetriever{a: a, d: d}
	c, _ := New(index(t, b), pir)
	ref, _ := New(index(t, b), plain)
	if err := c.Preprocess(ctx); err != nil {
		t.Fatal(err)
	}
	for qi, q := range queries(12, 8, 14) {
		if qi == 6 {
			if err := c.Refresh(ctx); err != nil {
				t.Fatal(err)
			}
		}
		got, err := c.Search(ctx, q, p)
		if err != nil {
			t.Fatal(err)
		}
		want, err := ref.Search(ctx, q, p)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got.IDs, want.IDs) || !reflect.DeepEqual(got.Clusters, want.Clusters) || !reflect.DeepEqual(got.Distances, want.Distances) {
			t.Fatalf("q=%d: piano %v plain %v", qi, got.IDs, want.IDs)
		}
		if got.Requests != p.NProbe || got.UpBytes <= 0 || got.DownBytes < int64(got.Requests*db.Layout.EntryBytes()) {
			t.Fatalf("q=%d: traffic %+v", qi, got)
		}
	}
	// n_probe differing from the Fetcher's fixed batch is an error, not a
	// silently resized request.
	if _, err := c.Search(ctx, queries(1, 8, 15)[0], Params{LSearch: 10, NProbe: 2, K: 2}); err == nil {
		t.Fatal("n_probe different from the fetcher batch accepted")
	}
}

// TestLoopbackEndToEnd serves the PIR DB and the bundle over one TCP loopback
// gRPC server, then fetches and loads the bundle, preprocesses Piano and
// answers queries, comparing them with the plaintext computation.
func TestLoopbackEndToEnd(t *testing.T) {
	a, d, b := fixture(t, 48, 4, 8, 16)
	db, err := clusterpir.Build(a, d)
	if err != nil {
		t.Fatal(err)
	}
	pirSrv, err := db.NewServer(1, 40)
	if err != nil {
		t.Fatal(err)
	}
	data, err := bundle.Encode(b)
	if err != nil {
		t.Fatal(err)
	}
	bundleSrv, err := bundlerpc.NewServer(data, 256)
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
	defer gs.Stop()

	conn, err := grpc.Dial(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	ctx := context.Background()
	got, _, _, err := bundlerpc.Fetch(ctx, bundlerpc.NewAnchorBundleClient(conn), bundle.Expect{Count: 48, Dim: 8, Size: 4})
	if err != nil {
		t.Fatal(err)
	}
	p := Params{LSearch: 12, NProbe: 4, K: 10}
	f, err := clusterpir.NewFetcher(ctx, rpc.NewPirServiceClient(conn), clusterpir.BatchConfig{NProbe: p.NProbe})
	if err != nil {
		t.Fatal(err)
	}
	c, _ := New(index(t, got), Piano{F: f})
	if err := c.Preprocess(ctx); err != nil {
		t.Fatal(err)
	}
	for qi, q := range queries(8, 8, 17) {
		res, err := c.Search(ctx, q, p)
		if err != nil {
			t.Fatal(err)
		}
		if want := bruteForce(a, d, q, res.Clusters, p.K); !reflect.DeepEqual(res.IDs, want) {
			t.Fatalf("q=%d: got %v want %v", qi, res.IDs, want)
		}
		if res.Requests == 0 || res.DownBytes == 0 || res.Timing.FetchSeconds <= 0 {
			t.Fatalf("q=%d: no PIR traffic recorded %+v", qi, res)
		}
	}
	// A dead server surfaces as an error with the failure recorded.
	gs.Stop()
	res, err := c.Search(ctx, queries(1, 8, 18)[0], p)
	if err == nil || res == nil || res.Error == "" || res.IDs != nil {
		t.Fatalf("server failure not reported: err=%v res=%+v", err, res)
	}
}

// TestPipelineHasNoPIRDependency keeps the pipeline code independent of a
// concrete PIR scheme: only the Piano adapter may import clusterpir/pirclient.
func TestPipelineHasNoPIRDependency(t *testing.T) {
	for _, file := range []string{"pipeline.go", "retriever.go"} {
		src, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		for _, banned := range []string{"clusterpir", "pirclient", "pianopir", "example.com/private-search/rpc"} {
			if strings.Contains(string(src), banned) {
				t.Fatalf("%s references %s", file, banned)
			}
		}
	}
}
