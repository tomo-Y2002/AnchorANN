package nsg

import (
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"
)

func randomVectors(n, dim int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	v := make([]float32, n*dim)
	for i := range v {
		v[i] = float32(rng.Intn(256))
	}
	return v
}

func smallParams() Params {
	return Params{KNN: 12, L: 24, R: 8, CNSG: 48, Workers: 4}
}

func checkGraph(t *testing.T, g *Graph, n int) {
	t.Helper()
	if len(g.Neighbors) != n || int(g.Entry) >= n {
		t.Fatalf("shape: nodes=%d entry=%d want n=%d", len(g.Neighbors), g.Entry, n)
	}
	for v, adj := range g.Neighbors {
		seen := map[uint32]bool{}
		for _, u := range adj {
			if int(u) >= n || int(u) == v || seen[u] {
				t.Fatalf("node %d bad neighbor %d in %v", v, u, adj)
			}
			seen[u] = true
		}
	}
	if u := g.Unreachable(); u != 0 {
		t.Fatalf("%d nodes unreachable", u)
	}
}

func TestBuildSmallReachableAndBounded(t *testing.T) {
	const n, dim = 500, 8
	v := randomVectors(n, dim, 1)
	p := smallParams()
	g, st, err := Build(v, dim, p)
	if err != nil {
		t.Fatal(err)
	}
	checkGraph(t, g, n)
	if st.Nodes != n || st.Edges != g.Edges() || st.MaxDegree != g.MaxDegree() || st.EffectiveKNN != p.KNN {
		t.Fatalf("stats mismatch: %+v", st)
	}
	if st.AboveR > st.RepairEdges {
		t.Fatalf("only repair edges may exceed R: above=%d repair=%d", st.AboveR, st.RepairEdges)
	}
	// The navigating node is the node nearest to the mean vector.
	mean := make([]float32, dim)
	sum := make([]float64, dim)
	for i := 0; i < n; i++ {
		for j := 0; j < dim; j++ {
			sum[j] += float64(v[i*dim+j])
		}
	}
	for j := range mean {
		mean[j] = float32(sum[j] / n)
	}
	if want := Exact(v, dim, mean, 1)[0].ID; g.Entry != want {
		t.Fatalf("entry=%d want medoid %d", g.Entry, want)
	}
}

func TestBuildDeterministicAcrossWorkers(t *testing.T) {
	v := randomVectors(400, 6, 2)
	p := smallParams()
	p.Workers = 1
	g1, _, err := Build(v, 6, p)
	if err != nil {
		t.Fatal(err)
	}
	p.Workers = 9
	g2, _, err := Build(v, 6, p)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(g1, g2) {
		t.Fatal("graph depends on worker count")
	}
}

func TestBuildTinyAndDegenerate(t *testing.T) {
	g, st, err := Build([]float32{1, 2, 3}, 3, smallParams())
	if err != nil {
		t.Fatal(err)
	}
	if g.Entry != 0 || len(g.Neighbors) != 1 || len(g.Neighbors[0]) != 0 || st.EffectiveKNN != 0 {
		t.Fatalf("C=1 graph %+v stats %+v", g, st)
	}
	s, err := NewSearcher(g, []float32{1, 2, 3}, 3)
	if err != nil {
		t.Fatal(err)
	}
	rs, _, err := s.Search([]float32{0, 0, 0}, 5)
	if err != nil || len(rs) != 1 || rs[0].ID != 0 || rs[0].Distance != 14 {
		t.Fatalf("C=1 search %v %v", rs, err)
	}

	g, _, err = Build([]float32{0, 10}, 1, smallParams())
	if err != nil {
		t.Fatal(err)
	}
	checkGraph(t, g, 2)

	// Identical vectors: MRNG pruning keeps only one edge per node, so the
	// graph relies on reverse edges and repair to stay connected.
	same := make([]float32, 64*4)
	for i := range same {
		same[i] = 7
	}
	g, _, err = Build(same, 4, smallParams())
	if err != nil {
		t.Fatal(err)
	}
	checkGraph(t, g, 64)
}

func TestBuildConnectsSeparatedGroups(t *testing.T) {
	// Two far-apart groups: the kNN graph (K=3) never crosses between them;
	// long edges come from the candidates found by searching from the entry.
	const per, dim = 40, 2
	v := make([]float32, 0, 2*per*dim)
	rng := rand.New(rand.NewSource(3))
	for g := 0; g < 2; g++ {
		for i := 0; i < per; i++ {
			v = append(v, float32(g*100000)+rng.Float32(), rng.Float32())
		}
	}
	g, _, err := Build(v, dim, Params{KNN: 3, L: 8, R: 4, CNSG: 16, Workers: 2})
	if err != nil {
		t.Fatal(err)
	}
	checkGraph(t, g, 2*per)
}

func TestConnectRepairsUnreachableNodes(t *testing.T) {
	// 0 -> 1, nodes 2 and 3 only point at each other and 4 is isolated.
	v := []float32{0, 1, 10, 11, 20}
	p := Params{KNN: 1, L: 4, R: 1, CNSG: 4}
	adj := [][]uint32{{1}, {}, {3}, {2}, {}}
	if added := connect(adj, v, 1, 0, p); added != 2 {
		t.Fatalf("added=%d want 2", added)
	}
	g := &Graph{Entry: 0, Neighbors: adj}
	if g.Unreachable() != 0 {
		t.Fatalf("still unreachable: %v", adj)
	}
	// Node 1 has spare degree (R=1) and is nearest to 2; node 0 is full.
	if !reflect.DeepEqual(adj[1], []uint32{2}) || !reflect.DeepEqual(adj[0], []uint32{1}) {
		t.Fatalf("repair edges %v", adj)
	}
	// 4 is attached from the nearest found node even though all are full.
	if !reflect.DeepEqual(adj[3], []uint32{2, 4}) {
		t.Fatalf("full-degree fallback %v", adj)
	}
}

func TestBuildRejectsInvalidInput(t *testing.T) {
	cases := []struct {
		v   []float32
		dim int
		p   Params
		msg string
	}{
		{[]float32{1, 2}, 2, Params{KNN: 0, L: 1, R: 1, CNSG: 1}, "positive"},
		{[]float32{1, 2, 3}, 2, smallParams(), "invalid vectors"},
		{nil, 2, smallParams(), "invalid vectors"},
		{[]float32{1, float32(math.NaN())}, 2, smallParams(), "nonfinite"},
	}
	for _, c := range cases {
		if _, _, err := Build(c.v, c.dim, c.p); err == nil || !strings.Contains(err.Error(), c.msg) {
			t.Errorf("Build(%v,%d,%+v) err=%v want %q", c.v, c.dim, c.p, err, c.msg)
		}
	}
}

func TestSearchMatchesExhaustiveWhenPoolCoversGraph(t *testing.T) {
	const n, dim = 300, 8
	v := randomVectors(n, dim, 4)
	g, _, err := Build(v, dim, smallParams())
	if err != nil {
		t.Fatal(err)
	}
	s, err := NewSearcher(g, v, dim)
	if err != nil {
		t.Fatal(err)
	}
	queries := randomVectors(20, dim, 5)
	for q := 0; q < 20; q++ {
		query := queries[q*dim : (q+1)*dim]
		got, evals, err := s.Search(query, n)
		if err != nil {
			t.Fatal(err)
		}
		if want := Exact(v, dim, query, n); !reflect.DeepEqual(got, want) {
			t.Fatalf("query %d: L=n search differs from full scan", q)
		}
		if evals != n {
			t.Fatalf("query %d: evaluations=%d want %d", q, evals, n)
		}
		small, _, err := s.Search(query, 10)
		if err != nil || len(small) != 10 {
			t.Fatalf("L=10 returned %d results err=%v", len(small), err)
		}
		for i := 1; i < len(small); i++ {
			if less(small[i], small[i-1]) {
				t.Fatal("results not ordered by distance")
			}
		}
	}
}

func TestSearchQualityOnSmallGraph(t *testing.T) {
	const n, dim, k = 2000, 8, 10
	v := randomVectors(n, dim, 6)
	g, _, err := Build(v, dim, Params{KNN: 24, L: 40, R: 16, CNSG: 100, Workers: 4})
	if err != nil {
		t.Fatal(err)
	}
	s, _ := NewSearcher(g, v, dim)
	queries := randomVectors(50, dim, 7)
	hits := 0
	for q := 0; q < 50; q++ {
		query := queries[q*dim : (q+1)*dim]
		got, _, _ := s.Search(query, 40)
		want := map[uint32]bool{}
		for _, r := range Exact(v, dim, query, k) {
			want[r.ID] = true
		}
		for _, r := range got[:k] {
			if want[r.ID] {
				hits++
			}
		}
	}
	if recall := float64(hits) / (50 * k); recall < 0.9 {
		t.Fatalf("recall@%d=%.3f below 0.9", k, recall)
	}
}

func TestSearcherValidation(t *testing.T) {
	v := randomVectors(10, 2, 8)
	g, _, err := Build(v, 2, smallParams())
	if err != nil {
		t.Fatal(err)
	}
	s, err := NewSearcher(g, v, 2)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := s.Search([]float32{1}, 3); err == nil {
		t.Error("dimension mismatch accepted")
	}
	if _, _, err := s.Search([]float32{1, 2}, 0); err == nil {
		t.Error("L_search=0 accepted")
	}
	if _, _, err := s.Search([]float32{1, float32(math.Inf(1))}, 3); err == nil {
		t.Error("infinite query accepted")
	}
	if _, err := NewSearcher(nil, v, 2); err == nil {
		t.Error("nil graph accepted")
	}
	if _, err := NewSearcher(&Graph{Entry: 10, Neighbors: g.Neighbors}, v, 2); err == nil {
		t.Error("entry out of range accepted")
	}
	bad := &Graph{Neighbors: make([][]uint32, 10)}
	bad.Neighbors[0] = []uint32{10}
	if _, err := NewSearcher(bad, v, 2); err == nil {
		t.Error("neighbor out of range accepted")
	}
	// Epoch wrap-around clears marks instead of treating nodes as visited.
	s.w.epoch = math.MaxUint32
	a, _, _ := s.Search([]float32{1, 2}, 10)
	b, _, _ := s.Search([]float32{1, 2}, 10)
	if len(a) != 10 || !reflect.DeepEqual(a, b) {
		t.Fatalf("epoch wrap changed results: %v vs %v", a, b)
	}
}

func TestExactAndDistance(t *testing.T) {
	if d := Distance([]float32{1, 2, 3, 4, 5}, []float32{0, 0, 0, 0, 0}); d != 55 {
		t.Fatalf("distance=%v", d)
	}
	v := []float32{1, -1, 0, 5}
	got := Exact(v, 1, []float32{0}, 3)
	want := []Result{{2, 0}, {0, 1}, {1, 1}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("exact=%v want %v (ties by ID)", got, want)
	}
	if len(Exact(v, 1, []float32{0}, 10)) != 4 || Exact(v, 1, []float32{0}, 0) != nil {
		t.Fatal("exact k clamping")
	}
}
