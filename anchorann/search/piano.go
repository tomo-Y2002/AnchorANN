package search

import (
	"context"
	"fmt"

	"privatesearch/anchorann/clusterpir"
)

// Piano adapts the cluster PIR Fetcher to Retriever. The Fetcher
// rebuilds the hints at the start of a Fetch on the public schedule (and
// reports it as a refresh); Refresh forces one by rebuilding every partition.
type Piano struct {
	F *clusterpir.Fetcher
}

// Preprocess builds the Piano hints of every partition.
func (p Piano) Preprocess(ctx context.Context) error { return p.F.Preprocess(ctx) }

// Refresh rebuilds the hints of every partition, which also empties the
// pirclient cache. It is the same work as Preprocess.
func (p Piano) Refresh(ctx context.Context) error {
	if err := p.F.Preprocess(ctx); err != nil {
		return fmt.Errorf("refresh: %w", err)
	}
	return nil
}

// Fetch retrieves the clusters and converts the result.
func (p Piano) Fetch(ctx context.Context, ids []int) (*Fetched, error) {
	r, err := p.F.Fetch(ctx, ids)
	if err != nil {
		return nil, err
	}
	out := &Fetched{Clusters: make([]*Cluster, len(r.Clusters)), Missed: r.Missed}
	for i, c := range r.Clusters {
		if c != nil {
			out.Clusters[i] = &Cluster{ID: c.ID, IDs: c.IDs, Vectors: c.Vectors}
		}
	}
	st := r.Stats
	out.Stats = FetchStats{
		OnlineSeconds: st.Seconds, UpBytes: st.Online.QueryUpBytes, DownBytes: st.Online.QueryDownBytes,
		Requests: st.Requests, RPCs: st.RPCs, CacheHits: st.CacheHits, PIRMissed: st.PIRMissed,
		Refreshes: st.Refreshes, RefreshSeconds: st.RefreshSeconds,
		RefreshUpBytes: st.Refresh.PreprocUpBytes, RefreshDownBytes: st.Refresh.PreprocDownBytes,
	}
	return out, nil
}
