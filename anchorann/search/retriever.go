// Package search runs the AnchorANN client query pipeline: local NSG search
// over the distributed anchors (no server traffic), private retrieval of the
// top n_probe clusters through a Retriever, and exact reranking of the
// retrieved real vectors into the final top-k original IDs.
package search

import "context"

// Cluster is one retrieved cluster: point j has original ID IDs[j] and real
// vector Vectors[j·dim:(j+1)·dim].
type Cluster struct {
	ID      int
	IDs     []uint32
	Vectors []float32
}

// FetchStats is the cost of one Fetch as reported by a Retriever. Online
// fields exclude any hint refresh that ran inside the call; such refreshes are
// reported separately so maintenance is never mixed into the online cost.
type FetchStats struct {
	OnlineSeconds    float64 `json:"online_seconds"`
	UpBytes          int64   `json:"online_up_bytes"`
	DownBytes        int64   `json:"online_down_bytes"`
	Requests         int     `json:"requests"`   // retrieval requests actually sent
	RPCs             int     `json:"rpcs"`       // round trips that carried them
	CacheHits        int     `json:"cache_hits"` // clusters answered from a client cache (a retriever may still send a dummy request)
	PIRMissed        int     `json:"pir_missed"` // clusters the retriever could not answer for lack of hints (also in Missed)
	Refreshes        int     `json:"refreshes"`  // refreshes the retriever ran on its own during the call
	RefreshSeconds   float64 `json:"refresh_seconds"`
	RefreshUpBytes   int64   `json:"refresh_up_bytes"`
	RefreshDownBytes int64   `json:"refresh_down_bytes"`
}

// Fetched is the answer to one Fetch. Clusters is aligned with the requested
// IDs; Clusters[i] is nil exactly when ids[i] is listed in Missed.
type Fetched struct {
	Clusters []*Cluster
	Missed   []int
	Stats    FetchStats
}

// Retriever is the only way the pipeline talks to the PIR layer. Swapping the
// PIR scheme (e.g. client-side hint generation in phase 2) means providing
// another Retriever; the pipeline code does not change.
type Retriever interface {
	// Preprocess runs the initial offline phase (e.g. Piano hint construction).
	Preprocess(ctx context.Context) error
	// Refresh rebuilds the offline state from scratch, discarding any cache.
	Refresh(ctx context.Context) error
	// Fetch privately retrieves the given distinct cluster IDs as one batch.
	// It must not silently drop clusters: anything not retrieved is listed in
	// Fetched.Missed, and transport failures are returned as errors.
	Fetch(ctx context.Context, ids []int) (*Fetched, error)
}
