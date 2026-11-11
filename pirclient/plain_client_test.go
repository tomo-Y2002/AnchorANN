package pirclient

import (
	"context"
	"reflect"
	"testing"

	"example.com/private-search/rpc"
	"google.golang.org/grpc"
)

type plainFakeClient struct {
	*fakeRPCClient
	plainCalls   int
	lastIds      []uint64
	plainReplies [][]byte
}

func (f *plainFakeClient) PlainQuery(ctx context.Context, in *rpc.PlainQueryRequest, opts ...grpc.CallOption) (*rpc.PlainQueryReply, error) {
	f.plainCalls++
	f.lastIds = append([]uint64(nil), in.Ids...)
	return &rpc.PlainQueryReply{Entries: f.plainReplies}, nil
}

func TestPlainBatchClientPreprocessIsNoop(t *testing.T) {
	meta := &rpc.MetadataReply{DbEntryByteNum: 8}
	c, err := NewPlainBatchClient(meta, &plainFakeClient{fakeRPCClient: &fakeRPCClient{}})
	if err != nil {
		t.Fatalf("NewPlainBatchClient: %v", err)
	}
	if err := c.Preprocess(context.Background()); err != nil {
		t.Fatalf("Preprocess: %v", err)
	}
}

func TestPlainBatchClientQueryReturnsEntries(t *testing.T) {
	dbEntrySize := uint64(2)
	dbEntryByteNum := dbEntrySize * 8
	meta := &rpc.MetadataReply{DbEntryByteNum: dbEntryByteNum}

	want := [][]uint64{{10, 11}, {20, 21}, {30, 31}}
	replies := make([][]byte, len(want))
	for i, w := range want {
		replies[i] = encodeU64s(w)
	}
	fake := &plainFakeClient{fakeRPCClient: &fakeRPCClient{}, plainReplies: replies}

	c, err := NewPlainBatchClient(meta, fake)
	if err != nil {
		t.Fatalf("NewPlainBatchClient: %v", err)
	}

	ids := []uint64{2, 5, 9}
	got, err := c.Query(context.Background(), ids)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if fake.plainCalls != 1 {
		t.Fatalf("expected 1 PlainQuery RPC, got %d", fake.plainCalls)
	}
	if !reflect.DeepEqual(fake.lastIds, ids) {
		t.Fatalf("ids mismatch: want %v got %v", ids, fake.lastIds)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("entries mismatch: want %v got %v", want, got)
	}
}

func TestPlainBatchClientQueryEmpty(t *testing.T) {
	meta := &rpc.MetadataReply{DbEntryByteNum: 8}
	fake := &plainFakeClient{fakeRPCClient: &fakeRPCClient{}}
	c, err := NewPlainBatchClient(meta, fake)
	if err != nil {
		t.Fatalf("NewPlainBatchClient: %v", err)
	}
	got, err := c.Query(context.Background(), nil)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected empty result, got %v", got)
	}
	if fake.plainCalls != 0 {
		t.Fatalf("expected no RPC for empty query, got %d", fake.plainCalls)
	}
}

func TestPlainBatchClientReplyCountMismatch(t *testing.T) {
	meta := &rpc.MetadataReply{DbEntryByteNum: 8}
	fake := &plainFakeClient{fakeRPCClient: &fakeRPCClient{}, plainReplies: [][]byte{encodeU64s([]uint64{1})}}
	c, _ := NewPlainBatchClient(meta, fake)
	if _, err := c.Query(context.Background(), []uint64{1, 2}); err == nil {
		t.Fatal("expected error on reply count mismatch")
	}
}
