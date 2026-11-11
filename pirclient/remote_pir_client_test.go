package pirclient

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"example.com/private-search/pianopir"
	"example.com/private-search/rpc"
	"example.com/private-search/rpc/server"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

// fakeRPCClient is a PirServiceClient whose calls all succeed with empty
// replies; tests embed it and override what they need.
type fakeRPCClient struct{}

func (f *fakeRPCClient) GetMetadata(ctx context.Context, in *rpc.MetadataRequest, opts ...grpc.CallOption) (*rpc.MetadataReply, error) {
	return &rpc.MetadataReply{}, nil
}

func (f *fakeRPCClient) StartPreprocess(ctx context.Context, in *rpc.PreprocessRequest, opts ...grpc.CallOption) (rpc.PirService_StartPreprocessClient, error) {
	return &blockStream{ctx: ctx, ch: closedBlocks()}, nil
}

func (f *fakeRPCClient) PrivateQuery(ctx context.Context, in *rpc.PrivateQueryRequest, opts ...grpc.CallOption) (*rpc.PrivateQueryReply, error) {
	return &rpc.PrivateQueryReply{}, nil
}

func (f *fakeRPCClient) BatchPrivateQuery(ctx context.Context, in *rpc.BatchPrivateQueryRequest, opts ...grpc.CallOption) (*rpc.BatchPrivateQueryReply, error) {
	return &rpc.BatchPrivateQueryReply{}, nil
}

func (f *fakeRPCClient) PlainQuery(ctx context.Context, in *rpc.PlainQueryRequest, opts ...grpc.CallOption) (*rpc.PlainQueryReply, error) {
	return &rpc.PlainQueryReply{}, nil
}

func (f *fakeRPCClient) GetStartVertices(ctx context.Context, in *rpc.GetStartVerticesRequest, opts ...grpc.CallOption) (*rpc.GetStartVerticesReply, error) {
	return &rpc.GetStartVerticesReply{}, nil
}

func (f *fakeRPCClient) Health(ctx context.Context, in *rpc.HealthRequest, opts ...grpc.CallOption) (*rpc.HealthReply, error) {
	return &rpc.HealthReply{}, nil
}

func encodeU64s(vals []uint64) []byte {
	out := make([]byte, len(vals)*8)
	for i, v := range vals {
		binary.LittleEndian.PutUint64(out[i*8:], v)
	}
	return out
}

func closedBlocks() chan streamItem {
	ch := make(chan streamItem)
	close(ch)
	return ch
}

type streamItem struct {
	block *rpc.PreprocessChunk
	err   error
}

// blockStream is the client end of an in-process StartPreprocess.
type blockStream struct {
	grpc.ClientStream
	ctx context.Context
	ch  chan streamItem
}

func (s *blockStream) Recv() (*rpc.PreprocessChunk, error) {
	item, ok := <-s.ch
	if !ok {
		return nil, io.EOF
	}
	return item.block, item.err
}
func (s *blockStream) Header() (metadata.MD, error) { return nil, nil }
func (s *blockStream) Trailer() metadata.MD         { return nil }
func (s *blockStream) CloseSend() error             { return nil }
func (s *blockStream) Context() context.Context     { return s.ctx }

// sendStream is the server end of an in-process StartPreprocess.
type sendStream struct {
	grpc.ServerStream
	ctx context.Context
	ch  chan streamItem
}

func (s *sendStream) Send(m *rpc.PreprocessChunk) error {
	s.ch <- streamItem{block: m}
	return nil
}
func (s *sendStream) Context() context.Context { return s.ctx }

// event is one call the server observes.
type event struct {
	Kind      string
	Partition uint64
	Offsets   int    // PrivateQuery: number of offsets
	Request   string // StartPreprocess: the serialized request
	Blocks    string // StartPreprocess: index/first entry/bytes of every block
}

// recordingClient runs a server.Server in-process and records what the
// server observes.
type recordingClient struct {
	fakeRPCClient
	srv    *server.Server
	events []event
}

func (c *recordingClient) GetMetadata(ctx context.Context, in *rpc.MetadataRequest, opts ...grpc.CallOption) (*rpc.MetadataReply, error) {
	return c.srv.GetMetadata(ctx, in)
}

func (c *recordingClient) StartPreprocess(ctx context.Context, in *rpc.PreprocessRequest, opts ...grpc.CallOption) (rpc.PirService_StartPreprocessClient, error) {
	raw, err := proto.MarshalOptions{Deterministic: true}.Marshal(in)
	if err != nil {
		return nil, err
	}
	c.events = append(c.events, event{Kind: "preprocess", Partition: uint64(in.PartitionId), Request: string(raw)})
	ev := &c.events[len(c.events)-1]
	src := make(chan streamItem)
	go func() {
		err := c.srv.StartPreprocess(in, &sendStream{ctx: ctx, ch: src})
		if err != nil {
			src <- streamItem{err: err}
		}
		close(src)
	}()
	out := make(chan streamItem)
	go func() {
		var shape strings.Builder
		for item := range src {
			if item.block != nil {
				fmt.Fprintf(&shape, "%d/%d/%d;", item.block.BlockIndex, item.block.FirstEntry, proto.Size(item.block))
			}
			out <- item
		}
		ev.Blocks = shape.String()
		close(out)
	}()
	return &blockStream{ctx: ctx, ch: out}, nil
}

func (c *recordingClient) PrivateQuery(ctx context.Context, in *rpc.PrivateQueryRequest, opts ...grpc.CallOption) (*rpc.PrivateQueryReply, error) {
	c.events = append(c.events, event{Kind: "private_query", Partition: in.PartitionId, Offsets: len(in.Offsets)})
	return c.srv.PrivateQuery(ctx, in)
}

func (c *recordingClient) BatchPrivateQuery(ctx context.Context, in *rpc.BatchPrivateQueryRequest, opts ...grpc.CallOption) (*rpc.BatchPrivateQueryReply, error) {
	c.events = append(c.events, event{Kind: "batch_private_query", Partition: in.PartitionId, Offsets: len(in.Offsets)})
	return c.srv.BatchPrivateQuery(ctx, in)
}

// testDB builds a server over n entries of entryWords non-zero words split
// into partitions of partitionSize.
func testDB(t *testing.T, n, partitionNum, partitionSize, entryWords uint64, failureProbLog2 uint64) (*recordingClient, *rpc.MetadataReply, [][]uint64) {
	t.Helper()
	rng := rand.New(rand.NewSource(7))
	entries := make([][]uint64, n)
	parts := make([][]uint64, partitionNum)
	for i := range entries {
		entries[i] = make([]uint64, entryWords)
		for w := range entries[i] {
			entries[i][w] = rng.Uint64() | 1
		}
		p := uint64(i) / partitionSize
		parts[p] = append(parts[p], entries[i]...)
	}
	srv, err := server.NewServer(server.Config{
		N: n, Dim: 1, M: 1, DBEntryByteNum: entryWords * 8,
		PartitionNum: partitionNum, PartitionSize: partitionSize, FailureProbLog2: failureProbLog2,
	}, parts)
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	client := &recordingClient{srv: srv}
	meta, err := client.GetMetadata(context.Background(), &rpc.MetadataRequest{})
	if err != nil {
		t.Fatalf("GetMetadata: %v", err)
	}
	return client, meta, entries
}

func TestPreprocessRequestCarriesNoSecrets(t *testing.T) {
	client, meta, _ := testDB(t, 64, 2, 32, 1, 10)
	pc, err := NewRemotePIRClientFromMetadata(meta, 1, client)
	if err != nil {
		t.Fatalf("NewRemotePIRClientFromMetadata: %v", err)
	}
	if err := pc.Preprocess(context.Background()); err != nil {
		t.Fatalf("Preprocess: %v", err)
	}
	if len(client.events) != 1 || client.events[0].Kind != "preprocess" {
		t.Fatalf("events = %+v, want one preprocess", client.events)
	}

	// The request schema has no field for a key, tags or replacements.
	fields := (&rpc.PreprocessRequest{}).ProtoReflect().Descriptor().Fields()
	var names []string
	for i := 0; i < fields.Len(); i++ {
		names = append(names, string(fields.Get(i).Name()))
	}
	if want := []string{"RequestId", "PartitionId", "ProtocolVersion"}; !reflect.DeepEqual(names, want) {
		t.Fatalf("PreprocessRequest fields = %v, want %v", names, want)
	}

	// The captured request sets only the public partition and version, and
	// no unknown (e.g. old key/tag) fields.
	var got rpc.PreprocessRequest
	if err := proto.Unmarshal([]byte(client.events[0].Request), &got); err != nil {
		t.Fatalf("unmarshal captured request: %v", err)
	}
	set := map[string]bool{}
	got.ProtoReflect().Range(func(fd protoreflect.FieldDescriptor, _ protoreflect.Value) bool {
		set[string(fd.Name())] = true
		return true
	})
	if want := map[string]bool{"PartitionId": true, "ProtocolVersion": true}; !reflect.DeepEqual(set, want) {
		t.Fatalf("captured request sets %v, want %v", set, want)
	}
	if len(got.ProtoReflect().GetUnknown()) != 0 {
		t.Fatalf("captured request has unknown fields")
	}
	if got.PartitionId != 1 || got.ProtocolVersion != rpc.PreprocessProtocolClientHints {
		t.Fatalf("captured request = %+v", &got)
	}
}

func TestStreamDependsOnlyOnPublicParameters(t *testing.T) {
	ctx := context.Background()
	run := func(queries []uint64) []event {
		client, meta, _ := testDB(t, 3000, 2, 1500, 3, 10)
		pc, err := NewRemotePIRClientFromMetadata(meta, 0, client)
		if err != nil {
			t.Fatalf("NewRemotePIRClientFromMetadata: %v", err)
		}
		if err := pc.Preprocess(ctx); err != nil {
			t.Fatalf("Preprocess: %v", err)
		}
		for _, q := range queries {
			if _, err := pc.Query(ctx, q, true); err != nil && !errors.Is(err, ErrMissed) {
				t.Fatalf("Query: %v", err)
			}
		}
		// a second construction after a different query history
		if err := pc.Preprocess(ctx); err != nil {
			t.Fatalf("Preprocess: %v", err)
		}
		var out []event
		for _, e := range client.events {
			if e.Kind == "preprocess" {
				out = append(out, e)
			}
		}
		return out
	}
	a := run([]uint64{1, 2, 3})
	b := run([]uint64{1400, 1400, 700, 12, 13})
	if len(a) != 2 || !reflect.DeepEqual(a, b) {
		t.Fatalf("preprocess streams differ:\n%+v\n%+v", a, b)
	}
	if a[0] != a[1] {
		t.Fatalf("first and second streams differ:\n%+v\n%+v", a[0], a[1])
	}
	// 1500 entries of 24 bytes: the whole partition, in blocks of B entries
	blockEntries := pianopir.StreamBlockEntries(128, 24)
	if want := int(pianopir.StreamBlockCount(1500, blockEntries)); strings.Count(a[0].Blocks, ";") != want {
		t.Fatalf("stream has %d blocks, want %d", strings.Count(a[0].Blocks, ";"), want)
	}
}

func TestRemotePIRClientMatchesPlainDB(t *testing.T) {
	ctx := context.Background()
	client, meta, entries := testDB(t, 1000, 1, 1000, 3, 40)
	pc, err := NewRemotePIRClientFromMetadata(meta, 0, client)
	if err != nil {
		t.Fatalf("NewRemotePIRClientFromMetadata: %v", err)
	}
	if err := pc.Preprocess(ctx); err != nil {
		t.Fatalf("Preprocess: %v", err)
	}
	rng := rand.New(rand.NewSource(3))
	for i := 0; i < 400; i++ {
		idx := uint64(rng.Intn(1000))
		got, err := pc.Query(ctx, idx, true)
		if errors.Is(err, ErrMissed) {
			continue
		}
		if err != nil {
			t.Fatalf("Query(%d): %v", idx, err)
		}
		if !reflect.DeepEqual(got, entries[idx]) {
			t.Fatalf("Query(%d) = %v, want %v", idx, got, entries[idx])
		}
	}
	st := pc.Stats()
	if st.Queries.Requests != 400 || st.Queries.Fetched == 0 || st.Queries.CacheHits == 0 {
		t.Fatalf("stats = %+v", st.Queries)
	}
	if st.Queries.Fetched+st.Queries.CacheHits+st.Queries.Missed != 400 {
		t.Fatalf("stats do not add up: %+v", st.Queries)
	}
	// MaxQueryNum = floor(sqrt(1000)·ln(1000)) = 218 requests per construction
	if pc.MaxQueryNum() != 218 || st.Initial.Count != 1 || st.Refresh.Count != 1 {
		t.Fatalf("MaxQueryNum=%d initial=%+v refresh=%+v, want 218 and one refresh", pc.MaxQueryNum(), st.Initial, st.Refresh)
	}
	if st.RefreshRequests != 218 {
		t.Fatalf("completed refresh requests=%d, want 218", st.RefreshRequests)
	}
	if st.Initial.DownBytes == 0 || st.Initial.UpBytes == 0 || st.Refresh.DownBytes != st.Initial.DownBytes {
		t.Fatalf("preprocess bytes initial=%+v refresh=%+v", st.Initial, st.Refresh)
	}
}

func TestBackupExhaustionMissesWithOneRequest(t *testing.T) {
	ctx := context.Background()
	// 4096 entries: chunk 128, 32 chunks, 48 backups per chunk.
	client, meta, entries := testDB(t, 4096, 1, 4096, 1, 40)
	pc, err := NewRemotePIRClientFromMetadata(meta, 0, client)
	if err != nil {
		t.Fatalf("NewRemotePIRClientFromMetadata: %v", err)
	}
	if pc.BackupPerChunk() != 48 {
		t.Fatalf("BackupPerChunk = %d, want 48", pc.BackupPerChunk())
	}
	if err := pc.Preprocess(ctx); err != nil {
		t.Fatalf("Preprocess: %v", err)
	}
	client.events = nil
	for idx := uint64(0); idx < 60; idx++ {
		got, err := pc.Query(ctx, idx, true)
		switch {
		case idx < 48 && err != nil:
			t.Fatalf("Query(%d): %v", idx, err)
		case idx < 48 && !reflect.DeepEqual(got, entries[idx]):
			t.Fatalf("Query(%d) = %v, want %v", idx, got, entries[idx])
		case idx >= 48 && !errors.Is(err, ErrMissed):
			t.Fatalf("Query(%d) error = %v, want ErrMissed", idx, err)
		}
	}
	// one PrivateQuery per call, no rebuild
	if len(client.events) != 60 {
		t.Fatalf("server observed %d calls, want 60", len(client.events))
	}
	for _, e := range client.events {
		if e.Kind != "private_query" {
			t.Fatalf("unexpected %s after backup exhaustion", e.Kind)
		}
	}
	if st := pc.Stats(); st.Queries.Missed != 12 || st.Refresh.Count != 0 {
		t.Fatalf("stats = %+v", st)
	}
}

// TestObservedTraceIndependentOfQueries runs concentrated, spread, repeated
// (cache hits) and backup-exhausting batches with the same batch length and
// checks the server observes the same calls in the same order, including the
// rebuilds.
func TestObservedTraceIndependentOfQueries(t *testing.T) {
	const (
		partitions = 2
		partSize   = 1024 // chunk 64, 16 chunks, MaxQueryNum 221, 39 backups per chunk
		perPart    = 8
		batches    = 60
	)
	n := uint64(partitions * partSize)
	rng := rand.New(rand.NewSource(11))
	// every batch holds distinct IDs
	distinct := func(limit int) []uint64 {
		ids := make([]uint64, partitions*perPart)
		for i, v := range rng.Perm(limit)[:len(ids)] {
			ids[i] = uint64(v)
		}
		return ids
	}
	repeated := distinct(int(n))
	inputs := map[string]func(b int) []uint64{
		"concentrated": func(b int) []uint64 { return distinct(64) },
		"spread":       func(b int) []uint64 { return distinct(int(n)) },
		"repeated":     func(b int) []uint64 { return repeated },
		"exhausting": func(b int) []uint64 {
			// distinct entries of chunk 0 in each partition
			ids := make([]uint64, 0, partitions*perPart)
			for p := 0; p < partitions; p++ {
				for j := 0; j < perPart; j++ {
					ids = append(ids, uint64(p*partSize+(b*perPart+j)%64))
				}
			}
			return ids
		},
	}

	var reference []event
	var referenceName string
	for _, name := range []string{"concentrated", "spread", "repeated", "exhausting"} {
		client, meta, entries := testDB(t, n, partitions, partSize, 1, 10)
		batch, err := NewRemoteBatchPIRFromMetadata(meta, client)
		if err != nil {
			t.Fatalf("NewRemoteBatchPIRFromMetadata: %v", err)
		}
		ctx := context.Background()
		if err := batch.Preprocess(ctx); err != nil {
			t.Fatalf("Preprocess: %v", err)
		}
		zeros := 0
		for b := 0; b < batches; b++ {
			ids := inputs[name](b)
			got, err := batch.Query(ctx, ids)
			if err != nil {
				t.Fatalf("%s batch %d: %v", name, b, err)
			}
			for i, id := range ids {
				if got[i][0] == 0 {
					zeros++
				} else if !reflect.DeepEqual(got[i], entries[id]) {
					t.Fatalf("%s batch %d: entry %d mismatch", name, b, id)
				}
			}
		}
		st := batch.Stats()
		if st.Client.Queries.Requests != partitions*perPart*batches {
			t.Fatalf("%s: %d requests, want %d", name, st.Client.Queries.Requests, partitions*perPart*batches)
		}
		// 221/8 = 27 batches per construction: rebuilds before batches 27 and 54
		if st.Client.Refresh.Count != 2*partitions {
			t.Fatalf("%s: %d rebuilds, want %d", name, st.Client.Refresh.Count, 2*partitions)
		}
		if st.Client.RefreshRequests != 2*partitions*27*perPart {
			t.Fatalf("%s: completed requests=%d, want batch-rounded interval", name, st.Client.RefreshRequests)
		}
		var hints, cache int64
		for _, part := range batch.partitions {
			pc := part.(*RemotePIRClient)
			hints += pc.HintStorageBytes()
			cache += pc.CacheStorageBytes()
		}
		if batch.HintStorageBytes() != hints || hints <= 0 || batch.CacheStorageBytes() != cache || cache <= 0 {
			t.Fatalf("%s: hint/cache storage mismatch (%d/%d)", name, hints, cache)
		}
		switch name {
		case "repeated":
			if st.Client.Queries.CacheHits == 0 {
				t.Fatalf("repeated: no cache hits")
			}
		case "exhausting":
			if st.Missed == 0 {
				t.Fatalf("exhausting: no misses: %+v", st)
			}
		case "concentrated":
			if st.Dropped == 0 {
				t.Fatalf("concentrated: nothing dropped: %+v", st)
			}
		}
		if zeros != st.Missed+st.Dropped {
			t.Fatalf("%s: %d zero entries, want missed %d + dropped %d", name, zeros, st.Missed, st.Dropped)
		}

		if reference == nil {
			reference, referenceName = client.events, name
			continue
		}
		if !reflect.DeepEqual(client.events, reference) {
			for i := range reference {
				if i >= len(client.events) || client.events[i] != reference[i] {
					t.Fatalf("%s and %s differ at call %d", referenceName, name, i)
				}
			}
			t.Fatalf("%s observed %d calls, %s %d", name, len(client.events), referenceName, len(reference))
		}
	}
	// rebuilds sit at batch boundaries: right after 27·16 requests
	var at []int
	queries := 0
	for _, e := range reference {
		if e.Kind == "private_query" {
			queries++
		} else if queries > 0 {
			at = append(at, queries)
		}
	}
	if want := []int{432, 432, 864, 864}; !reflect.DeepEqual(at, want) {
		t.Fatalf("rebuilds after %v requests, want %v", at, want)
	}
}

func TestOldServerFailsWithReason(t *testing.T) {
	ctx := context.Background()
	client, meta, _ := testDB(t, 64, 1, 64, 1, 10)

	// metadata of a server that predates the version field
	old := proto.Clone(meta).(*rpc.MetadataReply)
	old.PreprocessProtocolVersion = 0
	pc, err := NewRemotePIRClientFromMetadata(old, 0, client)
	if err != nil {
		t.Fatalf("NewRemotePIRClientFromMetadata: %v", err)
	}
	err = pc.Preprocess(ctx)
	if err == nil || !strings.Contains(err.Error(), "protocol version 0") || !strings.Contains(err.Error(), "upgrade the server") {
		t.Fatalf("Preprocess error = %v, want a protocol version error", err)
	}
	if len(client.events) != 0 {
		t.Fatalf("client contacted an incompatible server: %+v", client.events)
	}

	// a server that answers with a hint message (no protocol version)
	pc, err = NewRemotePIRClientFromMetadata(meta, 0, &oldStreamClient{})
	if err != nil {
		t.Fatalf("NewRemotePIRClientFromMetadata: %v", err)
	}
	err = pc.Preprocess(ctx)
	if err == nil || !strings.Contains(err.Error(), "protocol version 0") {
		t.Fatalf("Preprocess error = %v, want a protocol version error", err)
	}
}

type oldStreamClient struct{ fakeRPCClient }

func (c *oldStreamClient) StartPreprocess(ctx context.Context, in *rpc.PreprocessRequest, opts ...grpc.CallOption) (rpc.PirService_StartPreprocessClient, error) {
	ch := make(chan streamItem, 1)
	ch <- streamItem{block: &rpc.PreprocessChunk{PartitionId: 0}}
	close(ch)
	return &blockStream{ctx: ctx, ch: ch}, nil
}

func TestTruncatedStreamFails(t *testing.T) {
	ctx := context.Background()
	_, meta, _ := testDB(t, 64, 1, 64, 1, 10)
	pc, err := NewRemotePIRClientFromMetadata(meta, 0, &fakeRPCClient{})
	if err != nil {
		t.Fatalf("NewRemotePIRClientFromMetadata: %v", err)
	}
	if err := pc.Preprocess(ctx); err == nil || !strings.Contains(err.Error(), "stream ended") {
		t.Fatalf("Preprocess error = %v, want stream ended", err)
	}
	// a failed construction leaves the client unusable until the next one
	if _, err := pc.Query(ctx, 0, true); err == nil {
		t.Fatalf("Query after failed Preprocess succeeded")
	}
}
