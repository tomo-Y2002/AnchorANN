package server

import (
	"context"
	"encoding/binary"
	"io"
	"reflect"
	"strings"
	"testing"

	"example.com/private-search/pianopir"
	"example.com/private-search/rpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

type preprocessStream struct {
	ctx    context.Context
	chunks []*rpc.PreprocessChunk
}

func (s *preprocessStream) Send(chunk *rpc.PreprocessChunk) error {
	s.chunks = append(s.chunks, chunk)
	return nil
}

func (s *preprocessStream) SetHeader(md metadata.MD) error  { return nil }
func (s *preprocessStream) SendHeader(md metadata.MD) error { return nil }
func (s *preprocessStream) SetTrailer(md metadata.MD)       {}
func (s *preprocessStream) Context() context.Context        { return s.ctx }
func (s *preprocessStream) SendMsg(m interface{}) error     { return nil }
func (s *preprocessStream) RecvMsg(m interface{}) error     { return io.EOF }

func decodeU64s(b []byte) []uint64 {
	if len(b)%8 != 0 {
		return nil
	}
	out := make([]uint64, len(b)/8)
	for i := 0; i < len(out); i++ {
		out[i] = binary.LittleEndian.Uint64(b[i*8:])
	}
	return out
}

func TestPrivateQueryXor(t *testing.T) {
	dbSize := uint64(16)
	dbEntryByteNum := uint64(8)
	rawDB := make([]uint64, dbSize)
	for i := range rawDB {
		rawDB[i] = uint64(i)
	}

	cfg := Config{
		N:               dbSize,
		Dim:             1,
		M:               1,
		DBEntryByteNum:  dbEntryByteNum,
		PartitionNum:    1,
		PartitionSize:   dbSize,
		FailureProbLog2: 0,
	}

	srv, err := NewServer(cfg, [][]uint64{rawDB})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}

	_, setSize := pianopir.GenParams(uint32(dbSize))
	if setSize != 4 {
		t.Fatalf("unexpected set size: %d", setSize)
	}

	req := &rpc.PrivateQueryRequest{
		PartitionId: 0,
		Offsets:     []uint32{1, 3, 0, 0},
	}

	resp, err := srv.PrivateQuery(context.Background(), req)
	if err != nil {
		t.Fatalf("PrivateQuery: %v", err)
	}

	got := decodeU64s(resp.Entry)
	want := []uint64{rawDB[1] ^ rawDB[11]}

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("xor mismatch: want %v got %v", want, got)
	}
}

// BatchPrivateQuery answers every set exactly as PrivateQuery does.
func TestBatchPrivateQueryMatchesPrivateQuery(t *testing.T) {
	const dbSize = 40 // chunk 16, set size 4 (one padding chunk)
	rawDB := make([]uint64, 2*dbSize)
	for i := range rawDB {
		rawDB[i] = uint64(i)*0x9e3779b97f4a7c15 + 1
	}
	cfg := Config{N: 2 * dbSize, Dim: 1, M: 1, DBEntryByteNum: 8, PartitionNum: 2, PartitionSize: dbSize}
	srv, err := NewServer(cfg, [][]uint64{rawDB[:dbSize], rawDB[dbSize:]})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	sets := [][]uint32{{1, 3, 0, 15}, {15, 15, 7, 2}, {0, 0, 0, 0}}
	var flat []uint32
	for _, s := range sets {
		flat = append(flat, s...)
	}
	for pid := uint64(0); pid < 2; pid++ {
		resp, err := srv.BatchPrivateQuery(context.Background(), &rpc.BatchPrivateQueryRequest{PartitionId: pid, Count: uint32(len(sets)), Offsets: flat})
		if err != nil {
			t.Fatalf("BatchPrivateQuery: %v", err)
		}
		if len(resp.Entries) != len(sets) {
			t.Fatalf("got %d entries", len(resp.Entries))
		}
		for i, set := range sets {
			one, err := srv.PrivateQuery(context.Background(), &rpc.PrivateQueryRequest{PartitionId: pid, Offsets: set})
			if err != nil {
				t.Fatalf("PrivateQuery: %v", err)
			}
			if !reflect.DeepEqual(resp.Entries[i], one.Entry) {
				t.Fatalf("partition %d set %d: batch %v single %v", pid, i, resp.Entries[i], one.Entry)
			}
		}
	}
	for name, req := range map[string]*rpc.BatchPrivateQueryRequest{
		"partition":  {PartitionId: 2, Count: 1, Offsets: sets[0]},
		"zero count": {PartitionId: 0, Count: 0},
		"short":      {PartitionId: 0, Count: 2, Offsets: sets[0]},
		"long":       {PartitionId: 0, Count: 1, Offsets: flat},
	} {
		if _, err := srv.BatchPrivateQuery(context.Background(), req); err == nil {
			t.Fatalf("%s: accepted", name)
		}
	}
}

func TestStartPreprocessStreamsPartitionInPublicBlocks(t *testing.T) {
	// 3 partitions of size 300 with 2-word entries; the last one stores
	// only 100 entries and is padded with zeros in the stream.
	const partSize, words = 300, 2
	partitions := make([][]uint64, 3)
	for p := range partitions {
		stored := partSize
		if p == 2 {
			stored = 100
		}
		for i := 0; i < stored*words; i++ {
			partitions[p] = append(partitions[p], uint64(p*1000000+i+1))
		}
	}
	srv, err := NewServer(Config{N: 700, Dim: 1, M: 1, DBEntryByteNum: 8 * words, PartitionNum: 3, PartitionSize: partSize}, partitions)
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	meta, _ := srv.GetMetadata(context.Background(), &rpc.MetadataRequest{})
	if meta.PreprocessProtocolVersion != rpc.PreprocessProtocolClientHints {
		t.Fatalf("metadata protocol = %d", meta.PreprocessProtocolVersion)
	}
	blockEntries := pianopir.StreamBlockEntries(meta.ChunkSize, meta.DbEntryByteNum)
	blockCount := pianopir.StreamBlockCount(partSize, blockEntries)

	shape := func(chunks []*rpc.PreprocessChunk) []uint64 {
		var out []uint64
		for _, c := range chunks {
			out = append(out, c.BlockIndex, c.FirstEntry, uint64(len(c.Entries)))
		}
		return out
	}
	var first []uint64
	for _, pid := range []int32{1, 2} {
		stream := &preprocessStream{ctx: context.Background()}
		req := &rpc.PreprocessRequest{PartitionId: pid, ProtocolVersion: rpc.PreprocessProtocolClientHints}
		if err := srv.StartPreprocess(req, stream); err != nil {
			t.Fatalf("StartPreprocess(%d): %v", pid, err)
		}
		if uint64(len(stream.chunks)) != blockCount {
			t.Fatalf("partition %d: %d blocks, want %d", pid, len(stream.chunks), blockCount)
		}
		var got []uint64
		for b, c := range stream.chunks {
			if c.PartitionId != uint64(pid) || c.ProtocolVersion != rpc.PreprocessProtocolClientHints || c.BlockIndex != uint64(b) || c.FirstEntry != uint64(b)*blockEntries {
				t.Fatalf("partition %d block %d header = %+v", pid, b, c)
			}
			got = append(got, decodeU64s(c.Entries)...)
		}
		want := make([]uint64, partSize*words)
		copy(want, partitions[pid])
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("partition %d: streamed entries differ from the zero-padded partition", pid)
		}
		// the stream shape is the same for a full and a padded partition
		if first == nil {
			first = shape(stream.chunks)
		} else if !reflect.DeepEqual(shape(stream.chunks), first) {
			t.Fatalf("stream shape of partition %d differs: %v vs %v", pid, shape(stream.chunks), first)
		}
	}
}

func TestStartPreprocessRejectsOldClient(t *testing.T) {
	srv, err := NewServer(Config{N: 8, Dim: 1, M: 1, DBEntryByteNum: 8, PartitionNum: 1, PartitionSize: 8}, [][]uint64{make([]uint64, 8)})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	// an old client sends no version (and its key and tags arrive as
	// unknown fields, which the server ignores)
	stream := &preprocessStream{ctx: context.Background()}
	err = srv.StartPreprocess(&rpc.PreprocessRequest{PartitionId: 0}, stream)
	if status.Code(err) != codes.FailedPrecondition || !strings.Contains(err.Error(), "upgrade the client") {
		t.Fatalf("StartPreprocess error = %v, want FailedPrecondition asking to upgrade the client", err)
	}
	if len(stream.chunks) != 0 {
		t.Fatalf("server streamed %d blocks to an old client", len(stream.chunks))
	}
}

func TestNewServerRejectsOversizedPartition(t *testing.T) {
	_, err := NewServer(Config{N: 8, Dim: 1, M: 1, DBEntryByteNum: 8, PartitionNum: 1, PartitionSize: 4}, [][]uint64{make([]uint64, 8)})
	if err == nil {
		t.Fatal("NewServer accepted a partition larger than the partition size")
	}
}

func TestStartPreprocessPartitionOutOfRange(t *testing.T) {
	srv, err := NewServer(Config{N: 8, Dim: 1, M: 1, DBEntryByteNum: 8, PartitionNum: 1, PartitionSize: 8}, [][]uint64{make([]uint64, 8)})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	stream := &preprocessStream{ctx: context.Background()}
	err = srv.StartPreprocess(&rpc.PreprocessRequest{PartitionId: 5, ProtocolVersion: rpc.PreprocessProtocolClientHints}, stream)
	if err == nil {
		t.Fatal("expected error for out-of-range partition id")
	}
}

func TestGetStartVerticesRandomUnique(t *testing.T) {
	n := uint64(100)
	dbEntryByteNum := uint64(8)
	rawDB := make([]uint64, n)
	for i := range rawDB {
		rawDB[i] = uint64(i + 1)
	}

	cfg := Config{
		N:               n,
		Dim:             1,
		M:               1,
		DBEntryByteNum:  dbEntryByteNum,
		PartitionNum:    1,
		PartitionSize:   n,
		FailureProbLog2: 0,
	}

	srv, err := NewServer(cfg, [][]uint64{rawDB})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}

	req := &rpc.GetStartVerticesRequest{Num: 10}
	resp, err := srv.GetStartVertices(context.Background(), req)
	if err != nil {
		t.Fatalf("GetStartVertices: %v", err)
	}

	if len(resp.Vertices) != 10 {
		t.Fatalf("vertex count: want 10 got %d", len(resp.Vertices))
	}

	seen := make(map[uint64]bool)
	for _, v := range resp.Vertices {
		if v.Id >= n {
			t.Fatalf("vertex id out of range: %d", v.Id)
		}
		if seen[v.Id] {
			t.Fatalf("duplicate vertex id: %d", v.Id)
		}
		seen[v.Id] = true

		entry := decodeU64s(v.Entry)
		if len(entry) != 1 {
			t.Fatalf("entry length: want 1 got %d", len(entry))
		}
		if entry[0] != rawDB[v.Id] {
			t.Fatalf("entry mismatch for id %d: want %d got %d", v.Id, rawDB[v.Id], entry[0])
		}
	}
}

func TestGetStartVerticesNotDeterministic(t *testing.T) {
	n := uint64(1000)
	dbEntryByteNum := uint64(8)
	rawDB := make([]uint64, n)
	for i := range rawDB {
		rawDB[i] = uint64(i)
	}

	cfg := Config{
		N:               n,
		Dim:             1,
		M:               1,
		DBEntryByteNum:  dbEntryByteNum,
		PartitionNum:    1,
		PartitionSize:   n,
		FailureProbLog2: 0,
	}

	srv, err := NewServer(cfg, [][]uint64{rawDB})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}

	req := &rpc.GetStartVerticesRequest{Num: 10}
	resp1, _ := srv.GetStartVertices(context.Background(), req)
	resp2, _ := srv.GetStartVertices(context.Background(), req)

	same := 0
	for i := range resp1.Vertices {
		if resp1.Vertices[i].Id == resp2.Vertices[i].Id {
			same++
		}
	}
	if same == len(resp1.Vertices) {
		t.Fatal("two calls returned identical vertex IDs; expected random selection")
	}
}

func TestPlainQueryReturnsRawEntries(t *testing.T) {
	dbEntrySize := uint64(2)
	dbEntryByteNum := dbEntrySize * 8
	partitionSize := uint64(4)
	partitionNum := uint64(3)

	partitions := make([][]uint64, partitionNum)
	for p := uint64(0); p < partitionNum; p++ {
		part := make([]uint64, partitionSize*dbEntrySize)
		for i := uint64(0); i < partitionSize; i++ {
			part[i*dbEntrySize] = p*100 + i
			part[i*dbEntrySize+1] = p*100 + i + 1
		}
		partitions[p] = part
	}

	cfg := Config{
		N:               partitionSize * partitionNum,
		Dim:             1,
		M:               1,
		DBEntryByteNum:  dbEntryByteNum,
		PartitionNum:    partitionNum,
		PartitionSize:   partitionSize,
		FailureProbLog2: 0,
	}
	srv, err := NewServer(cfg, partitions)
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}

	ids := []uint64{0, 5, 9}
	resp, err := srv.PlainQuery(context.Background(), &rpc.PlainQueryRequest{Ids: ids})
	if err != nil {
		t.Fatalf("PlainQuery: %v", err)
	}
	if len(resp.Entries) != len(ids) {
		t.Fatalf("entry count: want %d got %d", len(ids), len(resp.Entries))
	}
	for i, id := range ids {
		entry := decodeU64s(resp.Entries[i])
		if uint64(len(entry)) != dbEntrySize {
			t.Fatalf("entry size mismatch: want %d got %d", dbEntrySize, len(entry))
		}
		pid := id / partitionSize
		off := id % partitionSize
		want := partitions[pid][off*dbEntrySize : (off+1)*dbEntrySize]
		if !reflect.DeepEqual(entry, want) {
			t.Fatalf("entry mismatch for id %d: want %v got %v", id, want, entry)
		}
	}
}

func TestPlainQueryOutOfRange(t *testing.T) {
	cfg := Config{
		N:               8,
		Dim:             1,
		M:               1,
		DBEntryByteNum:  8,
		PartitionNum:    1,
		PartitionSize:   8,
		FailureProbLog2: 0,
	}
	srv, err := NewServer(cfg, [][]uint64{make([]uint64, 8)})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	if _, err := srv.PlainQuery(context.Background(), &rpc.PlainQueryRequest{Ids: []uint64{99}}); err == nil {
		t.Fatal("expected error for out-of-range id (partition past end)")
	}
}

// When N < PartitionNum*PartitionSize, ids in the padding region of the
// last partition must also be rejected — they fall inside the partition
// slice but outside the real dataset.
func TestPlainQueryOutOfRangePadding(t *testing.T) {
	partitionSize := uint64(8)
	partitionNum := uint64(2)
	cfg := Config{
		N:               12, // last partition has 4 padding entries
		Dim:             1,
		M:               1,
		DBEntryByteNum:  8,
		PartitionNum:    partitionNum,
		PartitionSize:   partitionSize,
		FailureProbLog2: 0,
	}
	partitions := make([][]uint64, partitionNum)
	for p := range partitions {
		partitions[p] = make([]uint64, partitionSize)
	}
	srv, err := NewServer(cfg, partitions)
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	// id=12 is in partition 1, offset 4 — within the partition slice but past N.
	if _, err := srv.PlainQuery(context.Background(), &rpc.PlainQueryRequest{Ids: []uint64{12}}); err == nil {
		t.Fatal("expected error for id in padding region")
	}
}
