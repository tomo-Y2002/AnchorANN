package pirclient

import (
	"context"
	"reflect"
	"testing"
)

type queryCall struct {
	idx       uint64
	realQuery bool
}

type fakePartitionClient struct {
	partitionOffset uint64
	calls           []queryCall
}

func (f *fakePartitionClient) Query(ctx context.Context, idx uint64, realQuery bool) ([]uint64, error) {
	f.calls = append(f.calls, queryCall{idx: idx, realQuery: realQuery})
	if realQuery {
		return []uint64{idx + f.partitionOffset}, nil
	}
	return []uint64{0}, nil
}

func TestRemoteBatchPIRQueryOrder(t *testing.T) {
	cfg := BatchConfig{
		PartitionNum:   2,
		PartitionSize:  10,
		DBEntryByteNum: 8,
	}

	p0 := &fakePartitionClient{partitionOffset: 0}
	p1 := &fakePartitionClient{partitionOffset: 10}

	batch, err := NewRemoteBatchPIRWithClients(cfg, []PartitionClient{p0, p1})
	if err != nil {
		t.Fatalf("NewRemoteBatchPIRWithClients: %v", err)
	}

	idxs := []uint64{1, 12, 3, 15}
	got, err := batch.Query(context.Background(), idxs)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}

	want := [][]uint64{{1}, {12}, {3}, {15}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("output mismatch: want %v got %v", want, got)
	}

	wantCalls0 := []queryCall{{idx: 1, realQuery: true}, {idx: 3, realQuery: true}}
	wantCalls1 := []queryCall{{idx: 2, realQuery: true}, {idx: 5, realQuery: true}}

	if !reflect.DeepEqual(p0.calls, wantCalls0) {
		t.Fatalf("partition 0 calls mismatch: want %v got %v", wantCalls0, p0.calls)
	}
	if !reflect.DeepEqual(p1.calls, wantCalls1) {
		t.Fatalf("partition 1 calls mismatch: want %v got %v", wantCalls1, p1.calls)
	}
}

func TestRemoteBatchPIRPaddingAndDrop(t *testing.T) {
	cfg := BatchConfig{
		PartitionNum:   2,
		PartitionSize:  10,
		DBEntryByteNum: 8,
	}

	p0 := &fakePartitionClient{partitionOffset: 0}
	p1 := &fakePartitionClient{partitionOffset: 10}

	batch, err := NewRemoteBatchPIRWithClients(cfg, []PartitionClient{p0, p1})
	if err != nil {
		t.Fatalf("NewRemoteBatchPIRWithClients: %v", err)
	}

	idxs := []uint64{1, 2, 3, 4}
	got, err := batch.Query(context.Background(), idxs)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}

	want := [][]uint64{{1}, {2}, {0}, {0}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("output mismatch: want %v got %v", want, got)
	}

	wantCalls0 := []queryCall{{idx: 1, realQuery: true}, {idx: 2, realQuery: true}}
	wantCalls1 := []queryCall{{idx: 0, realQuery: false}, {idx: 0, realQuery: false}}

	if !reflect.DeepEqual(p0.calls, wantCalls0) {
		t.Fatalf("partition 0 calls mismatch: want %v got %v", wantCalls0, p0.calls)
	}
	if !reflect.DeepEqual(p1.calls, wantCalls1) {
		t.Fatalf("partition 1 calls mismatch: want %v got %v", wantCalls1, p1.calls)
	}
}
