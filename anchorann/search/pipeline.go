package search

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"time"

	"privatesearch/anchorann/bundle"
	"privatesearch/anchorann/nsg"
)

// ErrIncomplete marks a result that was computed from fewer clusters than
// requested (PIR misses) or holds fewer than k IDs. The partial Result is
// still returned, with the shortfall recorded in its fields.
var ErrIncomplete = errors.New("incomplete search result")

// Params are the per-query knobs. LSearch is the local NSG search pool,
// NProbe the number of clusters retrieved and K the answer count. With
// ExactAnchors the n_probe anchors are the exact top n_probe of all C
// anchors instead of the NSG search result (diagnostic mode, LSearch unused);
// it separates the loss of the graph search from the limit of cluster
// selection itself.
type Params struct {
	LSearch      int  `json:"l_search"`
	NProbe       int  `json:"n_probe"`
	K            int  `json:"k"`
	ExactAnchors bool `json:"exact_anchors"`
}

// Validate requires 1 <= NProbe <= C, NProbe <= LSearch (unless
// ExactAnchors) and 1 <= K <= NProbe·s, the most points NProbe clusters hold.
func (p Params) Validate(count, size int) error {
	sp := bundle.SearchParams{LSearch: p.LSearch, NProbe: p.NProbe, K: p.K}
	if p.ExactAnchors {
		sp.LSearch = max(p.NProbe, 1)
	}
	if err := sp.Validate(count, size); err != nil {
		return fmt.Errorf("search params: %w", err)
	}
	return nil
}

// Timing splits the online latency of one query. Online is the sum of the
// three phases; refreshes the retriever ran during Fetch are excluded and
// reported in Result.Refresh.
type Timing struct {
	AnchorSeconds float64 `json:"anchor_seconds"`
	FetchSeconds  float64 `json:"fetch_seconds"`
	RerankSeconds float64 `json:"rerank_seconds"`
	OnlineSeconds float64 `json:"online_seconds"`
}

// Refresh is the maintenance cost a retriever spent during one query.
type Refresh struct {
	Count     int     `json:"count"`
	Seconds   float64 `json:"seconds"`
	UpBytes   int64   `json:"up_bytes"`
	DownBytes int64   `json:"down_bytes"`
}

// Result is the answer and measurements of one query. Clusters (and Anchors)
// are client-local information and must never be sent to the server.
type Result struct {
	IDs       []uint32  `json:"ids"`       // top-k original IDs, by (distance, ID)
	Distances []float64 `json:"distances"` // squared L2 distances of IDs
	Anchors   []uint32  `json:"anchors"`   // selected anchors, best first
	Clusters  []int     `json:"clusters"`  // cluster IDs requested from the retriever
	// AnchorDistanceEvaluations counts distance computations of the anchor
	// selection (C in diagnostic mode).
	AnchorDistanceEvaluations int     `json:"anchor_distance_evaluations"`
	Candidates                int     `json:"candidates"`      // distinct retrieved points reranked
	MissedClusters            []int   `json:"missed_clusters"` // requested but not retrieved
	Shortfall                 int     `json:"shortfall"`       // K - len(IDs)
	Complete                  bool    `json:"complete"`        // no misses, no shortfall, no error
	Error                     string  `json:"error,omitempty"`
	Timing                    Timing  `json:"timing"`
	UpBytes                   int64   `json:"online_up_bytes"`
	DownBytes                 int64   `json:"online_down_bytes"`
	Requests                  int     `json:"requests"`
	RPCs                      int     `json:"rpcs"` // round trips carrying the requests
	CacheHits                 int     `json:"cache_hits"`
	PIRMissed                 int     `json:"pir_missed"` // missed clusters for lack of PIR hints (part of MissedClusters)
	Refresh                   Refresh `json:"refresh"`
}

// Client holds the loaded bundle and a Retriever. It is not safe for
// concurrent use (the NSG searcher reuses buffers).
type Client struct {
	ix  *bundle.Index
	pir Retriever
}

// New wraps a validated bundle index and a retriever. The retriever must
// already be preprocessed (or call Client.Preprocess).
func New(ix *bundle.Index, pir Retriever) (*Client, error) {
	if ix == nil || ix.Bundle == nil || pir == nil {
		return nil, fmt.Errorf("new search client: index and retriever are required")
	}
	return &Client{ix: ix, pir: pir}, nil
}

// Index returns the local anchor index.
func (c *Client) Index() *bundle.Index { return c.ix }

// Preprocess runs the retriever's offline phase.
func (c *Client) Preprocess(ctx context.Context) error { return c.pir.Preprocess(ctx) }

// Refresh rebuilds the retriever's offline state.
func (c *Client) Refresh(ctx context.Context) error { return c.pir.Refresh(ctx) }

// SelectClusters returns the anchors and their distinct clusters for query q:
// the first NProbe distinct clusters of the anchor ranking (local NSG search,
// or the exact ranking in diagnostic mode). No server traffic is involved.
func (c *Client) SelectClusters(q []float32, p Params) ([]uint32, []int, int, error) {
	b := c.ix.Bundle
	if err := p.Validate(b.Count, b.Size); err != nil {
		return nil, nil, 0, err
	}
	if len(q) != b.Dim {
		return nil, nil, 0, fmt.Errorf("query dimension=%d bundle dimension=%d", len(q), b.Dim)
	}
	var ranked []nsg.Result
	evals := b.Count
	if p.ExactAnchors {
		ranked = nsg.Exact(b.Vectors, b.Dim, q, p.NProbe)
	} else {
		var err error
		if ranked, evals, err = c.ix.Search(q, p.LSearch); err != nil {
			return nil, nil, 0, fmt.Errorf("local anchor search: %w", err)
		}
	}
	// The anchor -> cluster map of a validated bundle is a permutation, so
	// duplicates cannot occur; they are still merged rather than fetched twice.
	anchors := make([]uint32, 0, p.NProbe)
	clusters := make([]int, 0, p.NProbe)
	seen := make(map[uint32]struct{}, p.NProbe)
	for _, r := range ranked {
		if len(clusters) == p.NProbe {
			break
		}
		cl := b.ClusterOf[r.ID]
		if _, dup := seen[cl]; dup {
			continue
		}
		seen[cl] = struct{}{}
		anchors = append(anchors, r.ID)
		clusters = append(clusters, int(cl))
	}
	if len(clusters) < p.NProbe {
		return nil, nil, 0, fmt.Errorf("anchor search found %d distinct clusters, n_probe=%d", len(clusters), p.NProbe)
	}
	return anchors, clusters, evals, nil
}

// Search runs the full pipeline for query q. A retriever error is returned
// with a Result whose Error is set and which holds no IDs. If clusters were
// missed or fewer than K IDs were found, the Result is returned together with
// an error wrapping ErrIncomplete; partial answers are never silent.
func (c *Client) Search(ctx context.Context, q []float32, p Params) (*Result, error) {
	start := time.Now()
	anchors, clusters, evals, err := c.SelectClusters(q, p)
	if err != nil {
		return nil, err
	}
	res := &Result{Anchors: anchors, Clusters: clusters, AnchorDistanceEvaluations: evals}
	res.Timing.AnchorSeconds = time.Since(start).Seconds()

	start = time.Now()
	f, err := c.pir.Fetch(ctx, clusters)
	wall := time.Since(start).Seconds()
	if err != nil {
		res.Error = err.Error()
		res.Timing.FetchSeconds = wall
		res.Timing.OnlineSeconds = res.Timing.AnchorSeconds + wall
		return res, fmt.Errorf("fetch clusters: %w", err)
	}
	st := f.Stats
	res.UpBytes, res.DownBytes, res.Requests, res.CacheHits, res.PIRMissed = st.UpBytes, st.DownBytes, st.Requests, st.CacheHits, st.PIRMissed
	res.RPCs = st.RPCs
	res.Refresh = Refresh{Count: st.Refreshes, Seconds: st.RefreshSeconds, UpBytes: st.RefreshUpBytes, DownBytes: st.RefreshDownBytes}
	res.Timing.FetchSeconds = max(0, wall-st.RefreshSeconds)
	res.MissedClusters = append([]int{}, f.Missed...)
	got, err := checkFetched(clusters, f, c.ix.Bundle.Dim)
	if err != nil {
		res.Error = err.Error()
		return res, err
	}

	start = time.Now()
	top, cands := Rerank(q, got, c.ix.Bundle.Dim, p.K)
	res.Timing.RerankSeconds = time.Since(start).Seconds()
	res.Timing.OnlineSeconds = res.Timing.AnchorSeconds + res.Timing.FetchSeconds + res.Timing.RerankSeconds
	res.Candidates = cands
	res.IDs = make([]uint32, len(top))
	res.Distances = make([]float64, len(top))
	for i, r := range top {
		res.IDs[i], res.Distances[i] = r.ID, r.Distance
	}
	res.Shortfall = p.K - len(top)
	res.Complete = len(res.MissedClusters) == 0 && res.Shortfall == 0
	if !res.Complete {
		err := fmt.Errorf("%w: %d of %d clusters missed, %d of k=%d IDs", ErrIncomplete, len(res.MissedClusters), len(clusters), len(top), p.K)
		res.Error = err.Error()
		return res, err
	}
	return res, nil
}

// checkFetched verifies that the retriever answered exactly the requested
// clusters (nil only for listed misses) with consistent shapes, and returns
// the retrieved ones.
func checkFetched(ids []int, f *Fetched, dim int) ([]*Cluster, error) {
	if len(f.Clusters) != len(ids) {
		return nil, fmt.Errorf("retriever returned %d clusters for %d IDs", len(f.Clusters), len(ids))
	}
	missed := make(map[int]bool, len(f.Missed))
	for _, m := range f.Missed {
		missed[m] = true
	}
	got := make([]*Cluster, 0, len(ids))
	for i, c := range f.Clusters {
		if c == nil {
			if !missed[ids[i]] {
				return nil, fmt.Errorf("retriever returned no cluster %d without reporting a miss", ids[i])
			}
			continue
		}
		if missed[ids[i]] || c.ID != ids[i] {
			return nil, fmt.Errorf("retriever returned cluster %d (missed=%v) for requested cluster %d", c.ID, missed[ids[i]], ids[i])
		}
		if len(c.Vectors) != len(c.IDs)*dim {
			return nil, fmt.Errorf("cluster %d: %d IDs with %d coordinates, dimension=%d", c.ID, len(c.IDs), len(c.Vectors), dim)
		}
		got = append(got, c)
	}
	if len(got)+len(f.Missed) != len(ids) {
		return nil, fmt.Errorf("retriever reported %d misses with %d of %d clusters", len(f.Missed), len(got), len(ids))
	}
	return got, nil
}

// Rerank computes the squared L2 distance from q to every retrieved point,
// keeps each original ID once, and returns the k nearest by (distance, ID)
// together with the number of distinct candidates.
func Rerank(q []float32, clusters []*Cluster, dim, k int) ([]nsg.Result, int) {
	n := 0
	for _, c := range clusters {
		n += len(c.IDs)
	}
	all := make([]nsg.Result, 0, n)
	for _, c := range clusters {
		for j, id := range c.IDs {
			all = append(all, nsg.Result{ID: id, Distance: nsg.Distance(q, c.Vectors[j*dim:(j+1)*dim])})
		}
	}
	less := func(a, b nsg.Result) bool {
		return a.Distance < b.Distance || (a.Distance == b.Distance && a.ID < b.ID)
	}
	// Merge duplicate IDs, keeping the smallest distance of each.
	sort.Slice(all, func(i, j int) bool {
		if all[i].ID != all[j].ID {
			return all[i].ID < all[j].ID
		}
		return all[i].Distance < all[j].Distance
	})
	uniq := all[:0]
	for _, r := range all {
		if len(uniq) == 0 || r.ID != uniq[len(uniq)-1].ID {
			uniq = append(uniq, r)
		}
	}
	sort.Slice(uniq, func(i, j int) bool { return less(uniq[i], uniq[j]) })
	return uniq[:min(k, len(uniq))], len(uniq)
}
