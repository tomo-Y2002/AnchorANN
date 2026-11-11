package bundle

import (
	"fmt"

	"privatesearch/anchorann/nsg"
)

// SearchParams separates the three client-side knobs. LSearch is the NSG
// search pool, NProbe the number of clusters to retrieve and K the final
// answer count after reranking the NProbe*s retrieved points.
type SearchParams struct {
	LSearch int
	NProbe  int
	K       int
}

// Validate requires 1 <= NProbe <= LSearch, NProbe <= C and 1 <= K <= NProbe*s.
// LSearch < NProbe is rejected instead of silently widening the search.
func (p SearchParams) Validate(count, size int) error {
	if p.LSearch <= 0 || p.NProbe <= 0 || p.K <= 0 {
		return fmt.Errorf("L_search=%d n_probe=%d k=%d must be positive", p.LSearch, p.NProbe, p.K)
	}
	if p.NProbe > count {
		return fmt.Errorf("n_probe=%d exceeds cluster count C=%d", p.NProbe, count)
	}
	if p.LSearch < p.NProbe {
		return fmt.Errorf("L_search=%d is smaller than n_probe=%d", p.LSearch, p.NProbe)
	}
	if p.K > p.NProbe*size {
		return fmt.Errorf("k=%d exceeds retrieved points n_probe*s=%d", p.K, p.NProbe*size)
	}
	return nil
}

// Probe is a selected anchor and the cluster to retrieve for it.
type Probe struct {
	Anchor   uint32
	Cluster  uint32
	Distance float64
}

// Index holds a validated bundle in client memory and searches it locally.
// It has no network dependency. It is not safe for concurrent use.
type Index struct {
	Bundle   *Bundle
	searcher *nsg.Searcher
}

// NewIndex wraps a bundle that already passed Decode/Validate.
func NewIndex(b *Bundle) (*Index, error) {
	if err := b.Validate(Expect{}); err != nil {
		return nil, err
	}
	s, err := nsg.NewSearcher(&b.Graph, b.Vectors, b.Dim)
	if err != nil {
		return nil, err
	}
	return &Index{Bundle: b, searcher: s}, nil
}

// Search returns up to LSearch anchors ordered by (squared L2 distance, ID)
// and the number of distance evaluations.
func (ix *Index) Search(query []float32, lSearch int) ([]nsg.Result, int, error) {
	return ix.searcher.Search(query, lSearch)
}

// Probe searches with p.LSearch and returns the first p.NProbe anchors with
// their clusters. The anchor -> cluster map is a validated permutation, so
// the returned clusters are distinct. Fewer than NProbe probes are returned
// only if the search found fewer nodes, which a connected graph prevents.
func (ix *Index) Probe(query []float32, p SearchParams) ([]Probe, error) {
	if err := p.Validate(ix.Bundle.Count, ix.Bundle.Size); err != nil {
		return nil, err
	}
	rs, _, err := ix.searcher.Search(query, p.LSearch)
	if err != nil {
		return nil, err
	}
	rs = rs[:min(len(rs), p.NProbe)]
	out := make([]Probe, len(rs))
	for i, r := range rs {
		out[i] = Probe{Anchor: r.ID, Cluster: ix.Bundle.ClusterOf[r.ID], Distance: r.Distance}
	}
	return out, nil
}
