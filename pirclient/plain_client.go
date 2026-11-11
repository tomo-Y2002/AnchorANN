package pirclient

import (
	"context"
	"fmt"

	"example.com/private-search/rpc"
)

// PlainBatchClient implements the BatchClient interface using the
// non-private PlainQuery RPC. Preprocess is a no-op since no PIR hint
// material is required.
type PlainBatchClient struct {
	client      rpc.PirServiceClient
	dbEntrySize uint64
}

// NewPlainBatchClient builds a non-private batch client from server metadata.
func NewPlainBatchClient(meta *rpc.MetadataReply, client rpc.PirServiceClient) (*PlainBatchClient, error) {
	if meta.DbEntryByteNum%8 != 0 {
		return nil, fmt.Errorf("db entry bytes must be multiple of 8")
	}
	return &PlainBatchClient{
		client:      client,
		dbEntrySize: meta.DbEntryByteNum / 8,
	}, nil
}

// Preprocess is a no-op for the non-private client.
// It still emits a zero-duration trace span so trace_summary.py reports
// preprocess cost as 0 instead of treating it as missing data.
func (c *PlainBatchClient) Preprocess(ctx context.Context) error {
	trace := traceInfoFromContext(ctx)
	logStep("plain_batch.preprocess", "start", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, "")
	logStep("plain_batch.preprocess", "end", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, "")
	return nil
}

// Query fetches the requested entries from the server in a single PlainQuery RPC.
func (c *PlainBatchClient) Query(ctx context.Context, idx []uint64) ([][]uint64, error) {
	trace := traceInfoFromContext(ctx)
	logStep("plain_batch.query", "start", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, "")
	defer logStep("plain_batch.query", "end", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, "")

	if len(idx) == 0 {
		return [][]uint64{}, nil
	}

	requestID := rpc.NewRequestID()
	ctxRPC := rpc.WithRequestID(ctx, requestID)

	logStep("plain_query.rpc", "start", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, requestID)
	resp, err := c.client.PlainQuery(ctxRPC, &rpc.PlainQueryRequest{Ids: idx})
	logStep("plain_query.rpc", "end", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, requestID)
	if err != nil {
		return nil, err
	}
	if len(resp.Entries) != len(idx) {
		return nil, fmt.Errorf("plain query reply count mismatch: want %d got %d", len(idx), len(resp.Entries))
	}

	logStep("plain_query.decode", "start", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, requestID)
	defer logStep("plain_query.decode", "end", trace.QueryIndex, trace.StepIndex, trace.PartitionIndex, requestID)

	out := make([][]uint64, len(idx))
	for i, b := range resp.Entries {
		entry, err := decodeU64s(b)
		if err != nil {
			return nil, err
		}
		if uint64(len(entry)) != c.dbEntrySize {
			return nil, fmt.Errorf("plain query entry size mismatch: want %d got %d", c.dbEntrySize, len(entry))
		}
		out[i] = entry
	}
	return out, nil
}
