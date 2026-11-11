package nsg

import (
	"fmt"
	"math"
	"sort"
)

type poolEntry struct {
	Result
	checked bool
}

// walker holds reusable best-first search state. The epoch-tagged visit marks
// avoid clearing an O(n) array per query.
type walker struct {
	mark      []uint32
	epoch     uint32
	pool      []poolEntry
	visited   []Result
	distances int
}

func newWalker(n int) *walker {
	return &walker{mark: make([]uint32, n)}
}

func (w *walker) reset() {
	w.epoch++
	if w.epoch == 0 {
		clear(w.mark)
		w.epoch = 1
	}
	w.pool = w.pool[:0]
	w.visited = w.visited[:0]
	w.distances = 0
}

// search runs best-first search from entry with a pool of at most l nodes.
// It returns the pool sorted by (distance, ID), or every visited node when
// collect is set (the NSG build candidate set).
func (w *walker) search(adj [][]uint32, vectors []float32, dim int, q []float32, entry uint32, l int, collect bool) []Result {
	w.reset()
	visit := func(id uint32) Result {
		w.mark[id] = w.epoch
		w.distances++
		r := Result{id, Distance(q, vectors[int(id)*dim:(int(id)+1)*dim])}
		if collect {
			w.visited = append(w.visited, r)
		}
		return r
	}
	w.pool = append(w.pool, poolEntry{Result: visit(entry)})
	for next := 0; next < len(w.pool); {
		cur := &w.pool[next]
		cur.checked = true
		id := cur.ID
		// Entries before next are all checked, so the first unchecked entry
		// is at the lowest insertion position or at/after next.
		lowest := next
		for _, nb := range adj[id] {
			if w.mark[nb] == w.epoch {
				continue
			}
			r := visit(nb)
			if len(w.pool) == l && !less(r, w.pool[l-1].Result) {
				continue
			}
			pos := sort.Search(len(w.pool), func(i int) bool { return less(r, w.pool[i].Result) })
			if len(w.pool) < l {
				w.pool = append(w.pool, poolEntry{})
			}
			copy(w.pool[pos+1:], w.pool[pos:len(w.pool)-1])
			w.pool[pos] = poolEntry{Result: r}
			lowest = min(lowest, pos)
		}
		next = min(lowest, len(w.pool))
		for next < len(w.pool) && w.pool[next].checked {
			next++
		}
	}
	if collect {
		return append([]Result(nil), w.visited...)
	}
	out := make([]Result, len(w.pool))
	for i, e := range w.pool {
		out[i] = e.Result
	}
	return out
}

// Searcher performs local NSG search over a loaded graph. It never performs
// I/O. A Searcher reuses buffers and is not safe for concurrent use; create
// one per goroutine.
type Searcher struct {
	graph   *Graph
	vectors []float32
	dim     int
	w       *walker
}

// NewSearcher checks that the graph and vectors agree in node count and that
// every ID is in range. It does not copy the graph or vectors.
func NewSearcher(g *Graph, vectors []float32, dim int) (*Searcher, error) {
	if g == nil {
		return nil, fmt.Errorf("nil graph")
	}
	n, err := checkVectors(vectors, dim)
	if err != nil {
		return nil, err
	}
	if len(g.Neighbors) != n || int(g.Entry) >= n {
		return nil, fmt.Errorf("graph shape: nodes=%d vectors=%d entry=%d", len(g.Neighbors), n, g.Entry)
	}
	for v, adj := range g.Neighbors {
		for _, u := range adj {
			if int(u) >= n {
				return nil, fmt.Errorf("node=%d neighbor=%d outside 0..%d", v, u, n-1)
			}
		}
	}
	return &Searcher{graph: g, vectors: vectors, dim: dim, w: newWalker(n)}, nil
}

// Search returns up to lSearch nodes found by best-first search from the
// navigating node, ordered by (squared L2 distance, ID), and the number of
// distance evaluations performed.
func (s *Searcher) Search(query []float32, lSearch int) ([]Result, int, error) {
	if lSearch <= 0 {
		return nil, 0, fmt.Errorf("L_search=%d must be positive", lSearch)
	}
	if len(query) != s.dim {
		return nil, 0, fmt.Errorf("query dimension=%d expected=%d", len(query), s.dim)
	}
	for j, x := range query {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return nil, 0, fmt.Errorf("nonfinite query coordinate=%d", j)
		}
	}
	out := s.w.search(s.graph.Neighbors, s.vectors, s.dim, query, s.graph.Entry, lSearch, false)
	return out, s.w.distances, nil
}
