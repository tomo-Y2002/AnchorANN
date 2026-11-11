package clusterpir

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"example.com/private-search/pianopir"
	"example.com/private-search/pirclient"
	"example.com/private-search/rpc"

	"privatesearch/anchorann/bundle"
)

// BatchConfig fixes how one query's n_probe cluster fetches are sent. Every
// query issues exactly Rounds·PerPartition PrivateQuery requests to each of the
// server's partitions (P·PerPartition·Rounds in total), regardless of which
// clusters were selected. In each round a partition answers at most
// PerPartition of the still-pending clusters that fall in it; unused slots are
// dummy requests. Clusters still pending after Rounds rounds are reported as
// missed.
//
// RPC selects how the requests travel: RPCSequential (the default) sends one
// PrivateQuery round trip per request; RPCBatch sends the PerPartition
// requests of a partition and round in one BatchPrivateQuery round trip
// (P·Rounds round trips per query). Both send the same offset sets.
type BatchConfig struct {
	NProbe       int    `json:"n_probe"`
	PerPartition int    `json:"per_partition"`
	Rounds       int    `json:"rounds"`
	RPC          string `json:"rpc_mode,omitempty"`
}

// RPC modes of BatchConfig.
const (
	RPCSequential = "sequential" // one PrivateQuery round trip per request
	RPCBatch      = "batch"      // one BatchPrivateQuery round trip per partition and round (deferred parity)
)

// ParseRPCMode checks an RPC mode name; "" selects RPCSequential.
func ParseRPCMode(s string) (string, error) {
	switch s {
	case "", RPCSequential:
		return RPCSequential, nil
	case RPCBatch:
		return RPCBatch, nil
	}
	return "", fmt.Errorf("unknown PIR RPC mode %q (want %s or %s)", s, RPCSequential, RPCBatch)
}

// RPCsPerQuery is the fixed number of round trips per query.
func (b BatchConfig) RPCsPerQuery(partitions int) int {
	if b.RPC == RPCBatch {
		return partitions * b.Rounds
	}
	return b.RequestsPerQuery(partitions)
}

// DefaultBatch is the collision-free single round for one partition
// (PerPartition = n_probe), and ceil(n_probe/P) slots with 3 rounds otherwise.
func DefaultBatch(nProbe, partitions int) BatchConfig {
	if partitions <= 1 {
		return BatchConfig{NProbe: nProbe, PerPartition: nProbe, Rounds: 1}
	}
	return BatchConfig{NProbe: nProbe, PerPartition: (nProbe + partitions - 1) / partitions, Rounds: 3}
}

// RequestsPerQuery is the fixed number of PrivateQuery requests per query.
func (b BatchConfig) RequestsPerQuery(partitions int) int {
	return partitions * b.PerPartition * b.Rounds
}

// PIRParams returns the public retrieval parameters for the distribution
// bundle (anchorann/bundle): entries = C, entry bytes, and requests
// per query.
func (b BatchConfig) PIRParams(l Layout, partitions int) bundle.PIRParams {
	return bundle.PIRParams{EntryCount: uint64(l.Count), EntryBytes: uint64(l.EntryBytes()), BatchSize: uint32(b.RequestsPerQuery(partitions))}
}

// Validate checks n_probe ∈ 1..C, positive slots and rounds, and enough slots for n_probe.
func (b BatchConfig) Validate(count, partitions int) error {
	if b.NProbe <= 0 || b.NProbe > count {
		return fmt.Errorf("n_probe=%d must be in 1..C=%d", b.NProbe, count)
	}
	if b.PerPartition <= 0 || b.Rounds <= 0 {
		return fmt.Errorf("invalid batch: per-partition=%d rounds=%d (both must be positive)", b.PerPartition, b.Rounds)
	}
	if b.RequestsPerQuery(partitions) < b.NProbe {
		return fmt.Errorf("invalid batch: P·per-partition·rounds=%d < n_probe=%d", b.RequestsPerQuery(partitions), b.NProbe)
	}
	if _, err := ParseRPCMode(b.RPC); err != nil {
		return err
	}
	return nil
}

// OfflineStats describes the last full Piano hint construction of all
// partitions: the server streams every partition of the DB and the client
// builds the hints itself.
type OfflineStats struct {
	Seconds          float64 `json:"seconds"`
	UpBytes          int64   `json:"up_bytes"`
	DownBytes        int64   `json:"down_bytes"`
	HintStorageBytes int64   `json:"client_hint_storage_bytes"`
}

// FetchStats describes one Fetch call. Every slot sends exactly one PIR
// request, so Requests is always P·PerPartition·Rounds; RPCs is Requests in
// the sequential mode and P·Rounds in the batch mode.
type FetchStats struct {
	Requests  int `json:"requests"`   // PIR requests sent: P·PerPartition·Rounds
	RPCs      int `json:"rpcs"`       // round trips: PrivateQuery or BatchPrivateQuery calls
	Real      int `json:"real"`       // slots that requested a pending cluster (including cache hits and PIR misses)
	Dummy     int `json:"dummy"`      // unused slots, sent as dummy requests
	CacheHits int `json:"cache_hits"` // real slots answered from the pirclient cache; a dummy request was sent in their place
	// PIRMissed counts real slots that found no usable hint (the backup hints
	// of the chunk or the primary hints covering the cluster are used up). A
	// dummy request was sent in their place and the cluster is listed in
	// Result.Missed; it is not retried.
	PIRMissed int `json:"pir_missed"`
	Rounds    int `json:"rounds"`
	// PendingReuses counts batch-mode requests whose hint was replaced by an
	// earlier request of the same round trip (its parity waited for that
	// request's entry); always 0 in the sequential mode.
	PendingReuses int `json:"pending_reuses"`
	// Refreshes counts hint rebuilds run at the start of this call because
	// the call's PerPartition·Rounds requests would exceed MaxQueryNum
	// requests (of any kind) per partition since the last construction. The
	// schedule depends only on the number of Fetch calls.
	Refreshes      int     `json:"refreshes"`
	RefreshSeconds float64 `json:"refresh_seconds"` // wall time of those rebuilds, excluded from Seconds
	Seconds        float64 `json:"online_seconds"`  // online wall time without refreshes
	Online         Traffic `json:"online"`
	Refresh        Traffic `json:"refresh"`
}

// Result holds the clusters of one Fetch, aligned with the requested IDs.
// Clusters[i] is nil exactly when ids[i] is listed in Missed.
type Result struct {
	Clusters []*Cluster
	Missed   []int
	Stats    FetchStats
}

// Fetcher privately retrieves cluster entries from a PirService serving a
// cluster PIR DB, using one pirclient.RemotePIRClient per partition. The
// clients build their hints from the streamed DB, send one request per slot
// whatever its outcome, and rebuild only at the start of a Fetch on the
// public schedule. It is not safe for concurrent use.
type Fetcher struct {
	layout   Layout
	batch    BatchConfig
	partSize int
	hintSize int64
	dbBytes  int64
	parts    []*partition
	ready    bool
	offline  OfflineStats
}

type partition struct {
	cc *countingClient
	pc *pirclient.RemotePIRClient
}

// CacheStorageBytes returns logical cached key and entry bytes across partitions.
// Allocator overhead and query-local buffers are excluded.
func (f *Fetcher) CacheStorageBytes() int64 {
	var size int64
	for _, part := range f.parts {
		size += part.pc.CacheStorageBytes()
	}
	return size
}

// NewFetcher reads the server metadata, checks it describes a cluster PIR DB
// (N=C entries, M=s, entry bytes = Layout.EntryBytes), and creates one Piano
// client per partition. Zero PerPartition and Rounds select DefaultBatch; an
// empty RPC selects RPCSequential. Call Preprocess before Fetch.
func NewFetcher(ctx context.Context, client rpc.PirServiceClient, batch BatchConfig) (*Fetcher, error) {
	meta, err := client.GetMetadata(ctx, &rpc.MetadataRequest{})
	if err != nil {
		return nil, fmt.Errorf("get metadata: %w", err)
	}
	if meta.N > math.MaxInt32 || meta.M > math.MaxInt32 || meta.Dim > math.MaxInt32 || meta.PartitionNum > math.MaxInt32 {
		return nil, fmt.Errorf("metadata out of range: C=%d s=%d dimension=%d partitions=%d", meta.N, meta.M, meta.Dim, meta.PartitionNum)
	}
	l, err := NewLayout(int(meta.N), int(meta.M), int(meta.Dim))
	if err != nil {
		return nil, fmt.Errorf("metadata: %w", err)
	}
	if meta.DbEntryByteNum != uint64(l.EntryBytes()) {
		return nil, fmt.Errorf("metadata entry bytes=%d expected=%d for s=%d dimension=%d", meta.DbEntryByteNum, l.EntryBytes(), l.Size, l.Dim)
	}
	p := int(meta.PartitionNum)
	size, err := PartitionSize(l.Count, p)
	if err != nil {
		return nil, fmt.Errorf("metadata: %w", err)
	}
	if meta.PartitionSize != uint64(size) || meta.SetSize == 0 || meta.ChunkSize == 0 {
		return nil, fmt.Errorf("metadata partition size=%d expected=%d (chunk=%d set=%d)", meta.PartitionSize, size, meta.ChunkSize, meta.SetSize)
	}
	if batch.PerPartition == 0 && batch.Rounds == 0 {
		mode := batch.RPC
		batch = DefaultBatch(batch.NProbe, p)
		batch.RPC = mode
	}
	if batch.RPC, err = ParseRPCMode(batch.RPC); err != nil {
		return nil, err
	}
	if err := batch.Validate(l.Count, p); err != nil {
		return nil, err
	}
	f := &Fetcher{layout: l, batch: batch, partSize: size, parts: make([]*partition, p)}
	f.dbBytes = int64(meta.PartitionSize) * int64(meta.PartitionNum) * int64(l.EntryBytes())
	for i := range f.parts {
		cc := &countingClient{inner: client}
		pc, err := pirclient.NewRemotePIRClientFromMetadata(meta, uint64(i), cc)
		if err != nil {
			return nil, fmt.Errorf("partition=%d: %w", i, err)
		}
		f.parts[i] = &partition{cc: cc, pc: pc}
		f.hintSize += pc.HintStorageBytes()
	}
	if perPart := uint64(batch.PerPartition * batch.Rounds); perPart > f.parts[0].pc.MaxQueryNum() {
		return nil, fmt.Errorf("invalid batch: per-partition=%d x rounds=%d requests exceed MaxQueryNum=%d of partition size %d", batch.PerPartition, batch.Rounds, f.parts[0].pc.MaxQueryNum(), size)
	}
	return f, nil
}

// Layout returns the DB layout announced by the server.
func (f *Fetcher) Layout() Layout { return f.layout }

// Batch returns the batch configuration in use.
func (f *Fetcher) Batch() BatchConfig { return f.batch }

// Partitions returns the server's partition count P.
func (f *Fetcher) Partitions() int { return len(f.parts) }

// MaxQueryNum is the number of requests (real, dummy and cache-hit dummies)
// per partition one hint construction supports; a Fetch that would exceed it
// rebuilds the hints first.
func (f *Fetcher) MaxQueryNum() int { return int(f.parts[0].pc.MaxQueryNum()) }

// FetchesPerRefresh is the number of Fetch calls one hint construction
// serves: floor(MaxQueryNum / (PerPartition·Rounds)).
func (f *Fetcher) FetchesPerRefresh() int {
	return f.MaxQueryNum() / (f.batch.PerPartition * f.batch.Rounds)
}

// BackupPerChunk is the number of backup hints per chunk. Once a chunk has
// used them all, its further requests miss until the next construction.
func (f *Fetcher) BackupPerChunk() int { return int(f.parts[0].pc.BackupPerChunk()) }

// DBBytes is the PIR DB size over all partitions (PartitionSize entries
// each), i.e. the entry payload a hint construction downloads.
func (f *Fetcher) DBBytes() int64 { return f.dbBytes }

// Offline returns the statistics of the last Preprocess call.
func (f *Fetcher) Offline() OfflineStats { return f.offline }

func (f *Fetcher) traffic() Traffic {
	var t Traffic
	for _, p := range f.parts {
		t.Add(p.cc.t)
	}
	return t
}

// forEach runs fn for every partition concurrently and returns the first error.
func (f *Fetcher) forEach(fn func(i int, p *partition) error) error {
	errs := make([]error, len(f.parts))
	var wg sync.WaitGroup
	for i, p := range f.parts {
		wg.Add(1)
		go func(i int, p *partition) {
			defer wg.Done()
			errs[i] = fn(i, p)
		}(i, p)
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			return fmt.Errorf("partition=%d: %w", i, err)
		}
	}
	return nil
}

// Preprocess builds the Piano hints of every partition (offline phase).
func (f *Fetcher) Preprocess(ctx context.Context) error {
	before := f.traffic()
	start := time.Now()
	err := f.forEach(func(_ int, p *partition) error {
		if err := p.pc.Preprocess(ctx); err != nil {
			return fmt.Errorf("preprocess: %w", err)
		}
		return nil
	})
	if err != nil {
		f.ready = false
		return err
	}
	d := f.traffic().Sub(before)
	f.offline = OfflineStats{Seconds: time.Since(start).Seconds(), UpBytes: d.PreprocUpBytes, DownBytes: d.PreprocDownBytes, HintStorageBytes: f.hintSize}
	f.ready = true
	return nil
}

// Fetch privately retrieves the clusters ids (exactly n_probe distinct IDs in
// 0..C-1). Only PIR offsets and partition numbers reach the server; the IDs
// themselves are never sent.
func (f *Fetcher) Fetch(ctx context.Context, ids []int) (*Result, error) {
	if !f.ready {
		return nil, fmt.Errorf("fetch: Preprocess has not been run")
	}
	if len(ids) != f.batch.NProbe {
		return nil, fmt.Errorf("fetch: got %d cluster IDs, expected n_probe=%d", len(ids), f.batch.NProbe)
	}
	pending := make([][]int, len(f.parts)) // positions in ids, per partition
	seen := make(map[int]struct{}, len(ids))
	for i, c := range ids {
		if c < 0 || c >= f.layout.Count {
			return nil, fmt.Errorf("fetch: cluster ID=%d outside 0..%d", c, f.layout.Count-1)
		}
		if _, dup := seen[c]; dup {
			return nil, fmt.Errorf("fetch: duplicate cluster ID=%d (deduplicate before fetching)", c)
		}
		seen[c] = struct{}{}
		p := c / f.partSize
		pending[p] = append(pending[p], i)
	}

	res := &Result{Clusters: make([]*Cluster, len(ids)), Stats: FetchStats{Rounds: f.batch.Rounds}}
	before := f.traffic()
	// Rebuild before the batch if its requests would exceed MaxQueryNum. All
	// partitions send the same number of requests, so they rebuild together.
	start := time.Now()
	perPart := uint64(f.batch.PerPartition * f.batch.Rounds)
	err := f.forEach(func(_ int, p *partition) error {
		if _, err := p.pc.EnsureBudget(ctx, perPart); err != nil {
			return fmt.Errorf("refresh: %w", err)
		}
		return nil
	})
	if err != nil {
		f.ready = false
		return nil, fmt.Errorf("fetch: %w", err)
	}
	refreshed := f.traffic().Sub(before)
	if refreshed.Preprocesses > 0 {
		res.Stats.RefreshSeconds = time.Since(start).Seconds()
	}

	start = time.Now()
	counts := make([]FetchStats, len(f.parts))
	missed := make([][]int, len(f.parts)) // positions whose request missed
	for r := 0; r < f.batch.Rounds; r++ {
		err := f.forEach(func(i int, p *partition) error {
			if f.batch.RPC == RPCBatch {
				return f.roundBatch(ctx, p, i, ids, pending, missed, res, &counts[i])
			}
			taken := 0
			for slot := 0; slot < f.batch.PerPartition; slot++ {
				if taken == len(pending[i]) {
					counts[i].Dummy++
					if _, err := p.pc.Query(ctx, 0, false); err != nil {
						return fmt.Errorf("dummy query: %w", err)
					}
					continue
				}
				pos := pending[i][taken]
				taken++
				c, err := f.queryReal(ctx, p, ids[pos], i, &counts[i])
				if err != nil {
					return err
				}
				if c == nil {
					missed[i] = append(missed[i], pos)
				}
				res.Clusters[pos] = c
			}
			pending[i] = pending[i][taken:]
			return nil
		})
		if err != nil {
			return nil, fmt.Errorf("fetch round=%d: %w", r, err)
		}
	}
	res.Stats.Seconds = time.Since(start).Seconds()

	var lost []int
	for i := range pending {
		lost = append(lost, missed[i]...)
		lost = append(lost, pending[i]...)
	}
	sort.Ints(lost)
	for _, pos := range lost {
		res.Missed = append(res.Missed, ids[pos])
	}
	for _, c := range counts {
		res.Stats.Real += c.Real
		res.Stats.Dummy += c.Dummy
		res.Stats.CacheHits += c.CacheHits
		res.Stats.PIRMissed += c.PIRMissed
		res.Stats.PendingReuses += c.PendingReuses
	}
	d := f.traffic().Sub(before)
	res.Stats.Requests = d.Queries
	res.Stats.RPCs = d.QueryRPCs
	res.Stats.Refreshes = d.Preprocesses
	res.Stats.Online = Traffic{Queries: d.Queries, QueryRPCs: d.QueryRPCs, QueryUpBytes: d.QueryUpBytes, QueryDownBytes: d.QueryDownBytes}
	res.Stats.Refresh = Traffic{Preprocesses: d.Preprocesses, PreprocUpBytes: d.PreprocUpBytes, PreprocDownBytes: d.PreprocDownBytes}
	if want := f.batch.RequestsPerQuery(len(f.parts)); res.Stats.Requests != want {
		return nil, fmt.Errorf("fetch: sent %d PIR requests, expected %d slots", res.Stats.Requests, want)
	}
	if want := f.batch.RPCsPerQuery(len(f.parts)); res.Stats.RPCs != want {
		return nil, fmt.Errorf("fetch: made %d round trips, expected %d", res.Stats.RPCs, want)
	}
	return res, nil
}

// queryReal fetches cluster c through partition p. It returns a nil cluster
// when pirclient found no usable hint (pirclient.ErrMissed); pirclient sent a
// dummy request in its place, as it does for a cache hit.
func (f *Fetcher) queryReal(ctx context.Context, p *partition, c, i int, st *FetchStats) (*Cluster, error) {
	st.Real++
	hits := p.pc.Stats().Queries.CacheHits
	words, err := p.pc.Query(ctx, uint64(c-i*f.partSize), true)
	if errors.Is(err, pirclient.ErrMissed) {
		st.PIRMissed++
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("query cluster: %w", err)
	}
	if p.pc.Stats().Queries.CacheHits != hits {
		st.CacheHits++
	}
	cl, err := DecodeEntry(f.layout, words)
	if err != nil {
		return nil, fmt.Errorf("cluster=%d: %w", c, err)
	}
	cl.ID = c
	return &cl, nil
}

// roundBatch runs one round of partition i in the batch mode: the first
// PerPartition pending clusters, then dummies, in one BatchPrivateQuery. The
// slots and their offset sets are those of the sequential round.
func (f *Fetcher) roundBatch(ctx context.Context, p *partition, i int, ids []int, pending, missed [][]int, res *Result, st *FetchStats) error {
	taken := min(f.batch.PerPartition, len(pending[i]))
	qs := make([]pianopir.BatchQuery, f.batch.PerPartition)
	for slot := 0; slot < taken; slot++ {
		qs[slot] = pianopir.BatchQuery{Idx: uint64(ids[pending[i][slot]] - i*f.partSize), Real: true}
	}
	reuses := p.pc.Stats().Queries.PendingReuses
	answers, err := p.pc.QueryBatch(ctx, qs)
	if err != nil {
		return fmt.Errorf("batch query: %w", err)
	}
	st.Dummy += f.batch.PerPartition - taken
	st.PendingReuses += p.pc.Stats().Queries.PendingReuses - reuses
	for slot := 0; slot < taken; slot++ {
		pos := pending[i][slot]
		st.Real++
		a := answers[slot]
		if a.Outcome == pianopir.QueryMissed {
			st.PIRMissed++
			missed[i] = append(missed[i], pos)
			continue
		}
		if a.Outcome == pianopir.QueryCacheHit {
			st.CacheHits++
		}
		cl, err := DecodeEntry(f.layout, a.Entry)
		if err != nil {
			return fmt.Errorf("cluster=%d: %w", ids[pos], err)
		}
		cl.ID = ids[pos]
		res.Clusters[pos] = &cl
	}
	pending[i] = pending[i][taken:]
	return nil
}
