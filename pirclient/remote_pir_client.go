package pirclient

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"reflect"
	"time"

	"example.com/private-search/pianopir"
	"example.com/private-search/rpc"
	"google.golang.org/protobuf/proto"
)

// ErrMissed reports a real request that no hint could answer: the backup
// hints of its chunk or the primary hints covering it are used up. The client
// sent a dummy request in its place, so the server observes the same request
// as for any other outcome, and did not re-preprocess.
var ErrMissed = errors.New("pir request missed: no usable hint left")

// PreprocessStats accumulates hint constructions (StartPreprocess streams).
// Bytes are protobuf payload sizes (without gRPC framing); Up is
// client→server, Down is server→client.
type PreprocessStats struct {
	Count     int     `json:"count"`
	Seconds   float64 `json:"seconds"`
	UpBytes   int64   `json:"up_bytes"`
	DownBytes int64   `json:"down_bytes"`
}

// Add accumulates o into s.
func (s *PreprocessStats) Add(o PreprocessStats) {
	s.Count += o.Count
	s.Seconds += o.Seconds
	s.UpBytes += o.UpBytes
	s.DownBytes += o.DownBytes
}

// QueryStats counts the PrivateQuery requests of a client by outcome. Every
// request is one PrivateQuery RPC: Requests = Fetched + CacheHits + Missed +
// Dummy.
type QueryStats struct {
	Requests  int `json:"requests"`
	Fetched   int `json:"fetched"`
	CacheHits int `json:"cache_hits"`
	Missed    int `json:"missed"`
	Dummy     int `json:"dummy"`
	// BatchRPCs counts the BatchPrivateQuery round trips of QueryBatch; their
	// requests are counted in Requests like any other.
	BatchRPCs int `json:"batch_rpcs,omitempty"`
	// PendingReuses counts QueryBatch requests whose hint had been replaced
	// by an earlier request of the same batch (see pianopir.DeferredBatch).
	PendingReuses int `json:"pending_reuses,omitempty"`
}

// Add accumulates o into s.
func (s *QueryStats) Add(o QueryStats) {
	s.Requests += o.Requests
	s.Fetched += o.Fetched
	s.CacheHits += o.CacheHits
	s.Missed += o.Missed
	s.Dummy += o.Dummy
	s.BatchRPCs += o.BatchRPCs
	s.PendingReuses += o.PendingReuses
}

// ClientStats describes everything a client did so far. Initial is the first
// hint construction, Refresh every later one.
type ClientStats struct {
	Initial PreprocessStats `json:"initial_preprocess"`
	Refresh PreprocessStats `json:"refresh"`
	Queries QueryStats      `json:"queries"`
	// RefreshRequests counts all requests consumed by completed hint sets.
	// Batch boundaries can leave part of MaxQueryNum unused.
	RefreshRequests uint64 `json:"refresh_requests"`
}

// Add accumulates o into s.
func (s *ClientStats) Add(o ClientStats) {
	s.Initial.Add(o.Initial)
	s.Refresh.Add(o.Refresh)
	s.Queries.Add(o.Queries)
	s.RefreshRequests += o.RefreshRequests
}

// RemotePIRClient runs the Piano PIR client of one partition against a
// remote server. The server streams the partition DB (StartPreprocess) and
// the client builds its hints locally; the PRF key, hint tags and replacement
// positions never leave the client.
//
// Every Query sends exactly one PrivateQuery, whether the entry is fetched,
// answered from the cache or missed, and the hints are rebuilt only when the
// request count since the last preprocessing would exceed MaxQueryNum. The
// requests and the rebuilds the server observes therefore depend only on the
// number of Query calls, not on the requested indices.
type RemotePIRClient struct {
	partitionID uint64
	client      rpc.PirServiceClient
	protocol    uint32

	pir      *pianopir.PianoPIRClient
	ready    bool
	requests uint64 // PrivateQuery requests since the last preprocessing

	stats ClientStats
}

// NewRemotePIRClientFromMetadata builds the client of partitionID from the
// server metadata. Preprocess must run before Query.
func NewRemotePIRClientFromMetadata(meta *rpc.MetadataReply, partitionID uint64, client rpc.PirServiceClient) (*RemotePIRClient, error) {
	if meta.DbEntryByteNum == 0 || meta.DbEntryByteNum%8 != 0 {
		return nil, fmt.Errorf("db entry bytes must be a positive multiple of 8")
	}
	if meta.ChunkSize == 0 || meta.ChunkSize&(meta.ChunkSize-1) != 0 || meta.SetSize == 0 || meta.PartitionSize == 0 {
		return nil, fmt.Errorf("invalid partition layout: partition size %d chunk size %d set size %d", meta.PartitionSize, meta.ChunkSize, meta.SetSize)
	}
	if meta.PartitionSize > meta.ChunkSize*meta.SetSize {
		return nil, fmt.Errorf("partition size %d exceeds chunk size %d x set size %d", meta.PartitionSize, meta.ChunkSize, meta.SetSize)
	}
	cfg := &pianopir.PianoPIRConfig{
		DBEntryByteNum:  meta.DbEntryByteNum,
		DBEntrySize:     meta.DbEntryByteNum / 8,
		DBSize:          meta.PartitionSize,
		ChunkSize:       meta.ChunkSize,
		SetSize:         meta.SetSize,
		ThreadNum:       1,
		FailureProbLog2: meta.FailureProbLog2,
	}
	return &RemotePIRClient{
		partitionID: partitionID,
		client:      client,
		protocol:    meta.PreprocessProtocolVersion,
		pir:         pianopir.NewPianoPIRClient(cfg),
	}, nil
}

// MaxQueryNum is the number of requests (of any kind) the hints support; the
// client rebuilds them before the request that would exceed it.
func (c *RemotePIRClient) MaxQueryNum() uint64 { return c.pir.MaxQueryNum }

// BackupPerChunk is the number of backup hints per chunk. A chunk whose
// backups are used up misses its further requests until the next rebuild.
func (c *RemotePIRClient) BackupPerChunk() uint64 { return c.pir.MaxQueryPerChunk() }

// RequestsSincePreprocess is the number of requests sent since the last
// hint construction.
func (c *RemotePIRClient) RequestsSincePreprocess() uint64 { return c.requests }

// HintStorageBytes is the client hint storage of the partition.
func (c *RemotePIRClient) HintStorageBytes() int64 { return int64(c.pir.LocalStorageSize()) }

// CacheStorageBytes reports the keys and entry payloads retained in the cache,
// excluding Go map and slice allocation overhead.
func (c *RemotePIRClient) CacheStorageBytes() int64 { return c.pir.LocalCacheStorageBytes() }

// Stats returns the cumulative statistics of the client.
func (c *RemotePIRClient) Stats() ClientStats { return c.stats }

// Preprocess builds the hints from the streamed partition DB.
func (c *RemotePIRClient) Preprocess(ctx context.Context) error {
	return c.preprocessWithStep(ctx, "preprocess")
}

// EnsureBudget rebuilds the hints now if the next n requests would exceed
// MaxQueryNum, so that a batch of n requests runs without a rebuild in the
// middle. The decision depends only on public request counts. It reports
// whether a rebuild ran.
func (c *RemotePIRClient) EnsureBudget(ctx context.Context, n uint64) (bool, error) {
	if c.ready && c.requests+n <= c.pir.MaxQueryNum {
		return false, nil
	}
	return true, c.preprocessWithStep(ctx, "preprocess_refresh")
}

func (c *RemotePIRClient) preprocessWithStep(ctx context.Context, step string) error {
	if c.protocol != rpc.PreprocessProtocolClientHints {
		return fmt.Errorf("server announces preprocess protocol version %d, this client needs version %d (the server streams the DB and the client builds its hints); a version 0 server builds hints from a client PRF key, which this client never sends: upgrade the server",
			c.protocol, rpc.PreprocessProtocolClientHints)
	}
	requestID := rpc.NewRequestID()
	ctx = rpc.WithRequestID(ctx, requestID)
	trace := traceInfoFromContext(ctx)
	logStep(step, "start", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, requestID)
	defer logStep(step, "end", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, requestID)

	started := time.Now()
	c.ready = false
	builder := c.pir.NewStreamPreprocessor()
	req := &rpc.PreprocessRequest{
		PartitionId:     int32(c.partitionID),
		ProtocolVersion: rpc.PreprocessProtocolClientHints,
	}
	run := PreprocessStats{Count: 1, UpBytes: int64(proto.Size(req))}
	// cancel the stream if the client gives up before its end
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	stream, err := c.client.StartPreprocess(ctx, req)
	if err != nil {
		return fmt.Errorf("start preprocess: %w", err)
	}
	for {
		block, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("recv preprocess block: %w", err)
		}
		run.DownBytes += int64(proto.Size(block))
		if block.ProtocolVersion != rpc.PreprocessProtocolClientHints {
			return fmt.Errorf("server sent a preprocess message of protocol version %d, want %d (an old server sends hints, not DB blocks): upgrade the server",
				block.ProtocolVersion, rpc.PreprocessProtocolClientHints)
		}
		if block.PartitionId != c.partitionID {
			return fmt.Errorf("server sent a block of partition %d, want %d", block.PartitionId, c.partitionID)
		}
		words, err := decodeU64s(block.Entries)
		if err != nil {
			return err
		}
		if err := builder.AddBlock(block.BlockIndex, block.FirstEntry, words); err != nil {
			return err
		}
	}
	if err := builder.Finish(); err != nil {
		return err
	}
	c.ready = true
	completedRequests := c.requests
	c.requests = 0
	run.Seconds = time.Since(started).Seconds()
	if c.stats.Initial.Count == 0 {
		c.stats.Initial.Add(run)
	} else {
		c.stats.Refresh.Add(run)
		c.stats.RefreshRequests += completedRequests
	}
	return nil
}

// Query sends exactly one PrivateQuery. A dummy query (realQuery false)
// returns a zero entry. A real query returns the entry, from the server or
// from the cache, or ErrMissed when no hint is left for it. Before the
// request, the hints are rebuilt if MaxQueryNum requests were already sent.
func (c *RemotePIRClient) Query(ctx context.Context, idx uint64, realQuery bool) ([]uint64, error) {
	trace := traceInfoFromContext(ctx)
	logStep("remote_pir.query", "start", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, "")
	defer logStep("remote_pir.query", "end", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, "")

	if realQuery && idx >= c.dbSize() {
		return nil, fmt.Errorf("idx %d out of range", idx)
	}
	if _, err := c.EnsureBudget(ctx, 1); err != nil {
		return nil, err
	}

	requestID := rpc.NewRequestID()
	transport := &rpcTransport{ctx: rpc.WithRequestID(ctx, requestID), c: c, trace: trace, requestID: requestID}
	c.requests++
	c.stats.Queries.Requests++
	if !realQuery {
		c.stats.Queries.Dummy++
		if err := c.pir.DummyQuery(transport); err != nil {
			return nil, err
		}
		return make([]uint64, c.entryWords()), nil
	}

	entry, outcome, err := c.pir.QueryOnce(idx, transport)
	if err != nil {
		return nil, err
	}
	switch outcome {
	case pianopir.QueryCacheHit:
		c.stats.Queries.CacheHits++
		logStep("cache_hit", "start", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, requestID)
		logStep("cache_hit", "end", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, requestID)
	case pianopir.QueryMissed:
		c.stats.Queries.Missed++
		logStep("missed", "start", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, requestID)
		logStep("missed", "end", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, requestID)
		return nil, ErrMissed
	default:
		c.stats.Queries.Fetched++
	}
	return entry, nil
}

// BatchAnswer is the answer to one QueryBatch slot. Entry is nil for a dummy
// slot and for a real slot that missed (Outcome pianopir.QueryMissed).
type BatchAnswer struct {
	Entry   []uint64
	Outcome pianopir.QueryOutcome
}

// QueryBatch sends one request per query in a single BatchPrivateQuery
// round trip. The requests, their offset sets and the client state they leave
// are those of calling Query for each query in order (pianopir.DeferredBatch);
// only the number of round trips differs. Before the batch, the hints are
// rebuilt if its requests would exceed MaxQueryNum.
//
// If the round trip or the recovery fails, the hints already consumed by the
// batch have incomplete parities: the client marks itself as needing a
// rebuild, which the next EnsureBudget (or Query/QueryBatch) runs, and
// returns the error.
func (c *RemotePIRClient) QueryBatch(ctx context.Context, queries []pianopir.BatchQuery) ([]BatchAnswer, error) {
	trace := traceInfoFromContext(ctx)
	logStep("remote_pir.query_batch", "start", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, "")
	defer logStep("remote_pir.query_batch", "end", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, "")

	if len(queries) == 0 {
		return nil, fmt.Errorf("empty query batch")
	}
	if _, err := c.EnsureBudget(ctx, uint64(len(queries))); err != nil {
		return nil, err
	}
	batch, err := c.pir.PrepareDeferredBatch(queries)
	if err != nil {
		return nil, err
	}
	setSize := c.pir.Config().SetSize
	req := &rpc.BatchPrivateQueryRequest{PartitionId: c.partitionID, Count: uint32(len(queries)), Offsets: make([]uint32, 0, uint64(len(queries))*setSize)}
	for _, o := range batch.Offsets {
		req.Offsets = append(req.Offsets, o...)
	}
	c.requests += uint64(len(queries))
	c.stats.Queries.Requests += len(queries)
	c.stats.Queries.BatchRPCs++
	c.stats.Queries.PendingReuses += batch.PendingReuses

	requestID := rpc.NewRequestID()
	logStep("batch_private_query.rpc", "start", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, requestID)
	resp, err := c.client.BatchPrivateQuery(rpc.WithRequestID(ctx, requestID), req)
	logStep("batch_private_query.rpc", "end", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, requestID)
	if err == nil && len(resp.Entries) != len(queries) {
		err = fmt.Errorf("batch reply has %d entries for %d requests", len(resp.Entries), len(queries))
	}
	var entries [][]uint64
	if err == nil {
		responses := make([][]uint64, len(queries))
		for i, b := range resp.Entries {
			if responses[i], err = decodeU64s(b); err != nil {
				break
			}
		}
		if err == nil {
			entries, err = batch.Finish(responses)
		}
	}
	if err != nil {
		c.ready = false // the consumed hints lack their new parities
		return nil, fmt.Errorf("batch private query: %w", err)
	}

	out := make([]BatchAnswer, len(queries))
	for i, q := range queries {
		out[i] = BatchAnswer{Entry: entries[i], Outcome: batch.Outcomes[i]}
		switch {
		case !q.Real:
			c.stats.Queries.Dummy++
		case batch.Outcomes[i] == pianopir.QueryCacheHit:
			c.stats.Queries.CacheHits++
		case batch.Outcomes[i] == pianopir.QueryMissed:
			c.stats.Queries.Missed++
		default:
			c.stats.Queries.Fetched++
		}
	}
	return out, nil
}

// Clone returns a client of the same partition with a deep copy of the hint
// state and statistics that sends its requests through client. It answers
// every real query as c would; tests use it to run the same queries two ways.
func (c *RemotePIRClient) Clone(client rpc.PirServiceClient) *RemotePIRClient {
	d := *c
	d.client = client
	d.pir = c.pir.Clone()
	return &d
}

// SameHintState reports whether c and o hold the same hint state: key, hint
// tags, parities, programmed points, replacements, query counters and cache.
func (c *RemotePIRClient) SameHintState(o *RemotePIRClient) bool {
	return c.ready == o.ready && c.requests == o.requests && reflect.DeepEqual(c.pir.Clone(), o.pir.Clone())
}

// SetDummyOffsetSource makes dummy requests reproducible; tests only (see
// pianopir.PianoPIRClient.SetDummyOffsetSource).
func (c *RemotePIRClient) SetDummyOffsetSource(fn func(setSize, chunkSize uint64) []uint32) {
	c.pir.SetDummyOffsetSource(fn)
}

func (c *RemotePIRClient) dbSize() uint64     { return c.pir.Config().DBSize }
func (c *RemotePIRClient) entryWords() uint64 { return c.pir.Config().DBEntrySize }

// rpcTransport sends the offsets of one request to the partition.
type rpcTransport struct {
	ctx       context.Context
	c         *RemotePIRClient
	trace     rpc.TraceInfo
	requestID string
}

func (t *rpcTransport) PrivateQuery(offsets []uint32) ([]uint64, error) {
	logStep("private_query.rpc", "start", t.trace.QueryIndex, t.trace.StepIndex, t.trace.PartitionIndex, t.requestID)
	resp, err := t.c.client.PrivateQuery(t.ctx, &rpc.PrivateQueryRequest{
		PartitionId: t.c.partitionID,
		Offsets:     offsets,
	})
	logStep("private_query.rpc", "end", t.trace.QueryIndex, t.trace.StepIndex, t.trace.PartitionIndex, t.requestID)
	if err != nil {
		return nil, err
	}
	return decodeU64s(resp.Entry)
}

func decodeU64s(b []byte) ([]uint64, error) {
	if len(b)%8 != 0 {
		return nil, fmt.Errorf("invalid u64 slice size: %d", len(b))
	}
	out := make([]uint64, len(b)/8)
	for i := 0; i < len(out); i++ {
		out[i] = binary.LittleEndian.Uint64(b[i*8:])
	}
	return out, nil
}
