// Package nsg builds and searches a Navigating Spreading-out Graph (NSG) over
// AnchorANN anchors. The server builds the graph once; clients only search it.
package nsg

import (
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"
	"time"
)

// Params are the requested NSG construction parameters. KNN is the degree of
// the exact initial kNN graph, L the build-time search pool, R the pruning
// out-degree cap and CNSG the candidate pool cap (distinct from the cluster
// count C). Values larger than the node count are clamped during Build.
type Params struct {
	KNN  int `json:"knn"`
	L    int `json:"l"`
	R    int `json:"r"`
	CNSG int `json:"c_nsg"`
	// Workers only controls parallelism; the built graph does not depend on it.
	Workers int `json:"-"`
}

// DefaultParams returns the parameters adopted for SIFT1M (C=31,250); see
// anchorann/docs/nsg-bundle.md for the rationale.
func DefaultParams() Params {
	return Params{KNN: 64, L: 100, R: 32, CNSG: 300, Workers: runtime.NumCPU()}
}

// Validate rejects non-positive parameters.
func (p Params) Validate() error {
	if p.KNN <= 0 || p.L <= 0 || p.R <= 0 || p.CNSG <= 0 {
		return fmt.Errorf("invalid NSG params: K=%d L=%d R=%d C_nsg=%d (all must be positive)", p.KNN, p.L, p.R, p.CNSG)
	}
	return nil
}

// Graph is a directed graph over node IDs 0..len(Neighbors)-1 with a single
// navigating node Entry. Neighbor lists are ordered by distance to the node.
type Graph struct {
	Entry     uint32
	Neighbors [][]uint32
}

// Edges returns the total number of directed edges.
func (g *Graph) Edges() int {
	n := 0
	for _, adj := range g.Neighbors {
		n += len(adj)
	}
	return n
}

// MaxDegree returns the largest out-degree.
func (g *Graph) MaxDegree() int {
	m := 0
	for _, adj := range g.Neighbors {
		m = max(m, len(adj))
	}
	return m
}

// Unreachable returns the number of nodes not reachable from Entry. It
// assumes neighbor IDs are in range.
func (g *Graph) Unreachable() int {
	n := len(g.Neighbors)
	if n == 0 {
		return 0
	}
	seen := make([]bool, n)
	return n - reach(g.Neighbors, g.Entry, seen)
}

// reach marks every node reachable from start in seen and returns how many
// nodes were newly marked.
func reach(adj [][]uint32, start uint32, seen []bool) int {
	if seen[start] {
		return 0
	}
	seen[start] = true
	stack := []uint32{start}
	count := 1
	for len(stack) > 0 {
		v := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for _, u := range adj[v] {
			if !seen[u] {
				seen[u] = true
				count++
				stack = append(stack, u)
			}
		}
	}
	return count
}

// Stats records construction cost and the resulting graph shape.
type Stats struct {
	Nodes          int     `json:"nodes"`
	EffectiveKNN   int     `json:"effective_knn"`
	KNNSeconds     float64 `json:"knn_seconds"`
	EntrySeconds   float64 `json:"entry_seconds"`
	PruneSeconds   float64 `json:"prune_seconds"`
	ConnectSeconds float64 `json:"connect_seconds"`
	TotalSeconds   float64 `json:"total_seconds"`
	Edges          int     `json:"edges"`
	MaxDegree      int     `json:"max_degree"`
	MeanDegree     float64 `json:"mean_degree"`
	RepairEdges    int     `json:"repair_edges"`
	AboveR         int     `json:"nodes_above_r"`
}

// Distance is squared L2 accumulated in float64. The explicit conversion
// forbids fused multiply-add so every platform computes identical values.
func Distance(a, b []float32) float64 {
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+4 <= len(a); i += 4 {
		d0 := float64(a[i]) - float64(b[i])
		d1 := float64(a[i+1]) - float64(b[i+1])
		d2 := float64(a[i+2]) - float64(b[i+2])
		d3 := float64(a[i+3]) - float64(b[i+3])
		s0 += float64(d0 * d0)
		s1 += float64(d1 * d1)
		s2 += float64(d2 * d2)
		s3 += float64(d3 * d3)
	}
	for ; i < len(a); i++ {
		d := float64(a[i]) - float64(b[i])
		s0 += float64(d * d)
	}
	return (s0 + s1) + (s2 + s3)
}

// Result is a node and its squared L2 distance to a query.
type Result struct {
	ID       uint32
	Distance float64
}

func less(a, b Result) bool {
	return a.Distance < b.Distance || (a.Distance == b.Distance && a.ID < b.ID)
}

func sortResults(r []Result) {
	sort.Slice(r, func(i, j int) bool { return less(r[i], r[j]) })
}

func checkVectors(vectors []float32, dim int) (int, error) {
	if dim <= 0 || len(vectors) == 0 || len(vectors)%dim != 0 {
		return 0, fmt.Errorf("invalid vectors: coordinates=%d dimension=%d", len(vectors), dim)
	}
	n := len(vectors) / dim
	if uint64(n) > uint64(math.MaxUint32) {
		return 0, fmt.Errorf("too many nodes: %d", n)
	}
	for i, x := range vectors {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return 0, fmt.Errorf("nonfinite coordinate: node=%d coordinate=%d", i/dim, i%dim)
		}
	}
	return n, nil
}

// Exact returns the k nearest nodes to query by full scan, ordered by
// (distance, ID). It is the diagnostic reference for local search.
func Exact(vectors []float32, dim int, query []float32, k int) []Result {
	n := len(vectors) / dim
	k = min(k, n)
	if k <= 0 {
		return nil
	}
	h := make(maxHeap, 0, k)
	for i := 0; i < n; i++ {
		h.offer(Result{uint32(i), Distance(query, vectors[i*dim:(i+1)*dim])}, k)
	}
	out := []Result(h)
	sortResults(out)
	return out
}

// maxHeap keeps the k smallest results; the root is the worst kept result.
type maxHeap []Result

func (h *maxHeap) offer(r Result, k int) {
	s := *h
	if len(s) < k {
		s = append(s, r)
		for i := len(s) - 1; i > 0; {
			p := (i - 1) / 2
			if !less(s[p], s[i]) {
				break
			}
			s[p], s[i] = s[i], s[p]
			i = p
		}
		*h = s
		return
	}
	if !less(r, s[0]) {
		return
	}
	s[0] = r
	for i := 0; ; {
		l, big := 2*i+1, i
		if l < len(s) && less(s[big], s[l]) {
			big = l
		}
		if l+1 < len(s) && less(s[big], s[l+1]) {
			big = l + 1
		}
		if big == i {
			break
		}
		s[i], s[big] = s[big], s[i]
		i = big
	}
}

// parallel runs fn(i) for i in [0,n) on workers goroutines. Each i is handled
// independently, so the output never depends on scheduling.
func parallel(n, workers int, fn func(i int)) {
	workers = max(1, min(workers, n))
	var wg sync.WaitGroup
	var mu sync.Mutex
	next := 0
	const chunk = 64
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				mu.Lock()
				start := next
				next += chunk
				mu.Unlock()
				if start >= n {
					return
				}
				for i := start; i < min(start+chunk, n); i++ {
					fn(i)
				}
			}
		}()
	}
	wg.Wait()
}

// Build constructs an NSG over row-major vectors. Steps: exact kNN graph,
// navigating node = node nearest to the mean vector, per-node candidate
// search on the kNN graph plus MRNG pruning to R, reverse-edge insertion with
// re-pruning, and connectivity repair from the navigating node. Every step is
// deterministic for the same input regardless of Workers. Build returns an
// error if any node remains unreachable.
func Build(vectors []float32, dim int, p Params) (*Graph, Stats, error) {
	var st Stats
	if err := p.Validate(); err != nil {
		return nil, st, err
	}
	n, err := checkVectors(vectors, dim)
	if err != nil {
		return nil, st, err
	}
	if p.Workers <= 0 {
		p.Workers = 1
	}
	st.Nodes = n
	start := time.Now()
	vec := func(i uint32) []float32 { return vectors[int(i)*dim : (int(i)+1)*dim] }

	k := min(p.KNN, n-1)
	st.EffectiveKNN = k
	t := time.Now()
	knn := make([][]uint32, n)
	parallel(n, p.Workers, func(i int) {
		h := make(maxHeap, 0, k)
		q := vec(uint32(i))
		for j := 0; j < n; j++ {
			if j != i {
				h.offer(Result{uint32(j), Distance(q, vec(uint32(j)))}, k)
			}
		}
		rs := []Result(h)
		sortResults(rs)
		ids := make([]uint32, len(rs))
		for x, r := range rs {
			ids[x] = r.ID
		}
		knn[i] = ids
	})
	st.KNNSeconds = time.Since(t).Seconds()

	t = time.Now()
	mean := make([]float64, dim)
	for i := 0; i < n; i++ {
		for j, x := range vec(uint32(i)) {
			mean[j] += float64(x)
		}
	}
	center := make([]float32, dim)
	for j := range mean {
		center[j] = float32(mean[j] / float64(n))
	}
	entry := Exact(vectors, dim, center, 1)[0].ID
	st.EntrySeconds = time.Since(t).Seconds()

	t = time.Now()
	pruned := make([][]Result, n)
	searchers := sync.Pool{New: func() any { return newWalker(n) }}
	parallel(n, p.Workers, func(i int) {
		w := searchers.Get().(*walker)
		defer searchers.Put(w)
		q := vec(uint32(i))
		visited := w.search(knn, vectors, dim, q, entry, p.L, true)
		cands := make([]Result, 0, len(visited)+len(knn[i]))
		for _, r := range visited {
			if r.ID != uint32(i) {
				cands = append(cands, r)
			}
		}
		for _, j := range knn[i] {
			cands = append(cands, Result{j, Distance(q, vec(j))})
		}
		cands = dedupe(cands)
		if len(cands) > p.CNSG {
			cands = cands[:p.CNSG]
		}
		pruned[i] = prune(cands, p.R, vectors, dim)
	})

	// Reverse edges are gathered sequentially in node order, then each node
	// re-prunes only its own list, so the result is scheduling independent.
	reverse := make([][]uint32, n)
	for v, adj := range pruned {
		for _, r := range adj {
			reverse[r.ID] = append(reverse[r.ID], uint32(v))
		}
	}
	adj := make([][]uint32, n)
	parallel(n, p.Workers, func(i int) {
		q := vec(uint32(i))
		pool := append([]Result(nil), pruned[i]...)
		for _, v := range reverse[i] {
			pool = append(pool, Result{v, Distance(q, vec(v))})
		}
		pool = dedupe(pool)
		if len(pool) > p.R {
			pool = prune(pool, p.R, vectors, dim)
		}
		ids := make([]uint32, len(pool))
		for x, r := range pool {
			ids[x] = r.ID
		}
		adj[i] = ids
	})
	st.PruneSeconds = time.Since(t).Seconds()

	t = time.Now()
	st.RepairEdges = connect(adj, vectors, dim, entry, p)
	st.ConnectSeconds = time.Since(t).Seconds()

	g := &Graph{Entry: entry, Neighbors: adj}
	if u := g.Unreachable(); u != 0 {
		return nil, st, fmt.Errorf("NSG build left %d of %d nodes unreachable from entry=%d", u, n, entry)
	}
	st.Edges = g.Edges()
	st.MaxDegree = g.MaxDegree()
	st.MeanDegree = float64(st.Edges) / float64(n)
	for _, a := range adj {
		if len(a) > p.R {
			st.AboveR++
		}
	}
	st.TotalSeconds = time.Since(start).Seconds()
	return g, st, nil
}

// dedupe sorts by (distance, ID) and drops repeated IDs, which are adjacent
// because a node always has the same distance to a fixed query.
func dedupe(r []Result) []Result {
	sortResults(r)
	out := r[:0]
	for _, x := range r {
		if len(out) == 0 || out[len(out)-1].ID != x.ID {
			out = append(out, x)
		}
	}
	return out
}

// prune applies the MRNG edge-selection rule to candidates sorted by distance
// to the node: p is kept unless an already kept r satisfies d(p,r) < d(p,node).
func prune(cands []Result, r int, vectors []float32, dim int) []Result {
	out := make([]Result, 0, r)
	for _, c := range cands {
		if len(out) >= r {
			break
		}
		pv := vectors[int(c.ID)*dim : (int(c.ID)+1)*dim]
		keep := true
		for _, s := range out {
			if Distance(pv, vectors[int(s.ID)*dim:(int(s.ID)+1)*dim]) < c.Distance {
				keep = false
				break
			}
		}
		if keep {
			out = append(out, c)
		}
	}
	return out
}

// connect attaches every node unreachable from entry. For the smallest
// unreached ID u it searches for u from entry and adds an edge from the
// nearest found node whose degree is below R (or the nearest found node if all
// are full, letting that degree exceed R), then continues the traversal from
// u. It returns the number of added edges.
func connect(adj [][]uint32, vectors []float32, dim int, entry uint32, p Params) int {
	n := len(adj)
	seen := make([]bool, n)
	reached := reach(adj, entry, seen)
	added := 0
	w := newWalker(n)
	for u := 0; reached < n; u++ {
		if seen[u] {
			continue
		}
		q := vectors[u*dim : (u+1)*dim]
		found := w.search(adj, vectors, dim, q, entry, p.L, false)
		from := found[0].ID
		for _, r := range found {
			if len(adj[r.ID]) < p.R {
				from = r.ID
				break
			}
		}
		adj[from] = append(adj[from], uint32(u))
		added++
		reached += reach(adj, uint32(u), seen)
	}
	return added
}
