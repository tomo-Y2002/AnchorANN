package pirclient

import (
	"context"
	"errors"
	"fmt"
	"time"

	"example.com/private-search/rpc"
)

const defaultValue = 0xdeadbeef

// BatchConfig defines the partition layout.
type BatchConfig struct {
	PartitionNum   uint64
	PartitionSize  uint64
	DBEntryByteNum uint64
}

// PartitionClient is the per-partition PIR interface. Query sends one
// request; it returns ErrMissed for a real request no hint could answer.
type PartitionClient interface {
	Query(ctx context.Context, idx uint64, realQuery bool) ([]uint64, error)
}

// budgeter is implemented by partitions that rebuild their hints on a public
// schedule (RemotePIRClient).
type budgeter interface {
	EnsureBudget(ctx context.Context, n uint64) (bool, error)
}

// BatchStats summarizes a RemoteBatchPIR. Client aggregates the partition
// clients that report statistics (RemotePIRClient). Dropped counts real
// indices that got no request slot because their partition already had
// len(idx)/PartitionNum of them in the batch; Missed counts real requests
// that were sent (as dummies) but had no usable hint. Both are returned as
// zero entries. RefreshSeconds is the wall time of the hint rebuilds the
// batch ran before its requests.
type BatchStats struct {
	Client         ClientStats `json:"client"`
	Batches        int         `json:"batches"`
	Dropped        int         `json:"dropped"`
	Missed         int         `json:"missed"`
	RefreshSeconds float64     `json:"refresh_seconds"`
}

// RemoteBatchPIR dispatches queries to partition clients.
type RemoteBatchPIR struct {
	config     BatchConfig
	partitions []PartitionClient
	stats      BatchStats
}

// NewRemoteBatchPIRWithClients builds a batch client using provided partitions.
func NewRemoteBatchPIRWithClients(cfg BatchConfig, clients []PartitionClient) (*RemoteBatchPIR, error) {
	if cfg.PartitionNum == 0 {
		return nil, fmt.Errorf("partition num must be > 0")
	}
	if uint64(len(clients)) != cfg.PartitionNum {
		return nil, fmt.Errorf("partition count mismatch: want %d got %d", cfg.PartitionNum, len(clients))
	}
	return &RemoteBatchPIR{config: cfg, partitions: clients}, nil
}

// NewRemoteBatchPIRFromMetadata builds partition clients from server metadata.
func NewRemoteBatchPIRFromMetadata(meta *rpc.MetadataReply, client rpc.PirServiceClient) (*RemoteBatchPIR, error) {
	cfg := BatchConfig{
		PartitionNum:   meta.PartitionNum,
		PartitionSize:  meta.PartitionSize,
		DBEntryByteNum: meta.DbEntryByteNum,
	}
	partitions := make([]PartitionClient, int(meta.PartitionNum))
	for i := uint64(0); i < meta.PartitionNum; i++ {
		p, err := NewRemotePIRClientFromMetadata(meta, i, client)
		if err != nil {
			return nil, err
		}
		partitions[int(i)] = p
	}
	return NewRemoteBatchPIRWithClients(cfg, partitions)
}

// Preprocess invokes preprocessing on partitions that support it.
func (p *RemoteBatchPIR) Preprocess(ctx context.Context) error {
	trace := traceInfoFromContext(ctx)
	logStep("batch_pir.preprocess", "start", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, "")
	defer logStep("batch_pir.preprocess", "end", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, "")
	for _, part := range p.partitions {
		if pre, ok := part.(interface{ Preprocess(context.Context) error }); ok {
			if err := pre.Preprocess(ctx); err != nil {
				return err
			}
		}
	}
	return nil
}

// Stats returns the cumulative statistics of the batch client.
func (p *RemoteBatchPIR) Stats() BatchStats {
	out := p.stats
	for _, part := range p.partitions {
		if s, ok := part.(interface{ Stats() ClientStats }); ok {
			out.Client.Add(s.Stats())
		}
	}
	return out
}

// HintStorageBytes is the client hint storage summed over the partitions
// that report it (RemotePIRClient).
func (p *RemoteBatchPIR) HintStorageBytes() int64 {
	var total int64
	for _, part := range p.partitions {
		if s, ok := part.(interface{ HintStorageBytes() int64 }); ok {
			total += s.HintStorageBytes()
		}
	}
	return total
}

// CacheStorageBytes sums retained cache keys and entry payloads over partitions.
func (p *RemoteBatchPIR) CacheStorageBytes() int64 {
	var total int64
	for _, part := range p.partitions {
		if s, ok := part.(interface{ CacheStorageBytes() int64 }); ok {
			total += s.CacheStorageBytes()
		}
	}
	return total
}

// Query issues a batch PIR query and returns entries in the original order.
// Every partition receives exactly len(idx)/PartitionNum requests, so the
// server observes only the batch length. If those requests would exceed a
// partition's MaxQueryNum, all partitions rebuild their hints before the
// batch (never in its middle). Dropped and missed indices are returned as
// zero entries.
func (p *RemoteBatchPIR) Query(ctx context.Context, idx []uint64) ([][]uint64, error) {
	if len(idx) == 0 {
		return [][]uint64{}, nil
	}
	if p.config.PartitionNum == 0 {
		return nil, fmt.Errorf("partition num must be > 0")
	}

	trace := traceInfoFromContext(ctx)
	logStep("batch_pir.query", "start", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, "")
	defer logStep("batch_pir.query", "end", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, "")

	queryNumToMake := len(idx) / int(p.config.PartitionNum)
	partitionQueries := make([][]uint64, int(p.config.PartitionNum))
	for _, id := range idx {
		partitionIdx := id / p.config.PartitionSize
		if partitionIdx >= p.config.PartitionNum {
			return nil, fmt.Errorf("partition id out of range: %d", partitionIdx)
		}
		partitionQueries[int(partitionIdx)] = append(partitionQueries[int(partitionIdx)], id)
	}
	p.stats.Batches++

	if queryNumToMake > 0 {
		started := time.Now()
		refreshed := false
		for i, part := range p.partitions {
			if b, ok := part.(budgeter); ok {
				ctxPart := rpc.WithTrace(ctx, rpc.TraceInfo{QueryIndex: trace.QueryIndex, StepIndex: trace.StepIndex, PartitionIndex: i})
				ran, err := b.EnsureBudget(ctxPart, uint64(queryNumToMake))
				if err != nil {
					return nil, fmt.Errorf("refresh partition %d: %w", i, err)
				}
				refreshed = refreshed || ran
			}
		}
		if refreshed {
			p.stats.RefreshSeconds += time.Since(started).Seconds()
		}
	}

	responses := make(map[uint64][]uint64)
	for i := uint64(0); i < p.config.PartitionNum; i++ {
		tracePart := trace
		tracePart.PartitionIndex = int(i)
		logStep("partition.query", "start", tracePart.QueryIndex, tracePart.StepIndex, tracePart.PartitionIndex, "")

		if len(partitionQueries[int(i)]) > queryNumToMake {
			p.stats.Dropped += len(partitionQueries[int(i)]) - queryNumToMake
		}
		for j := len(partitionQueries[int(i)]); j < queryNumToMake; j++ {
			partitionQueries[int(i)] = append(partitionQueries[int(i)], defaultValue)
		}
		for j := 0; j < queryNumToMake; j++ {
			q := partitionQueries[int(i)][j]
			ctxPart := rpc.WithTrace(ctx, tracePart)
			if q == defaultValue {
				if _, err := p.partitions[int(i)].Query(ctxPart, 0, false); err != nil {
					return nil, err
				}
				continue
			}
			resp, err := p.partitions[int(i)].Query(ctxPart, q-i*p.config.PartitionSize, true)
			if errors.Is(err, ErrMissed) {
				p.stats.Missed++
				continue
			}
			if err != nil {
				return nil, err
			}
			responses[q] = resp
		}
		logStep("partition.query", "end", tracePart.QueryIndex, tracePart.StepIndex, tracePart.PartitionIndex, "")
	}

	entrySize := p.config.DBEntryByteNum / 8
	ret := make([][]uint64, len(idx))
	for i, id := range idx {
		if resp, ok := responses[id]; ok {
			ret[i] = resp
		} else {
			ret[i] = make([]uint64, entrySize)
		}
	}
	return ret, nil
}
