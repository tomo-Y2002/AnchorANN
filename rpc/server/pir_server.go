package server

import (
	context "context"
	"crypto/rand"
	"encoding/binary"
	"fmt"

	"example.com/private-search/pianopir"
	"example.com/private-search/rpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Config captures the server-side layout and PIR parameters.
type Config struct {
	N                   uint64
	Dim                 uint64
	M                   uint64
	DBEntryByteNum      uint64
	PartitionNum        uint64
	PartitionSize       uint64
	FailureProbLog2     uint64
	PqEnabled           bool
	PqM                 uint64
	PqKsub              uint64
	PqDsub              uint64
	PqNeighborCodeCount uint64
	PqCodebook          []byte
}

// Server implements rpc.PirServiceServer for a fixed dataset.
type Server struct {
	rpc.UnimplementedPirServiceServer
	cfg         Config
	chunkSize   uint64
	setSize     uint64
	dbEntrySize uint64
	partitions  [][]uint64
}

// NewServer constructs a PIR server with preloaded raw DB partitions.
func NewServer(cfg Config, partitions [][]uint64) (*Server, error) {
	if cfg.PartitionNum == 0 {
		return nil, fmt.Errorf("partition num must be > 0")
	}
	if uint64(len(partitions)) != cfg.PartitionNum {
		return nil, fmt.Errorf("partition count mismatch: want %d got %d", cfg.PartitionNum, len(partitions))
	}
	if cfg.DBEntryByteNum%8 != 0 {
		return nil, fmt.Errorf("DBEntryByteNum must be multiple of 8")
	}

	for i, part := range partitions {
		if uint64(len(part)) > cfg.PartitionSize*(cfg.DBEntryByteNum/8) {
			return nil, fmt.Errorf("partition %d holds %d words, more than partition size %d entries", i, len(part), cfg.PartitionSize)
		}
	}

	chunkSize32, setSize32 := pianopir.GenParams(uint32(cfg.PartitionSize))
	chunkSize := uint64(chunkSize32)
	setSize := uint64(setSize32)

	srv := &Server{
		cfg:         cfg,
		chunkSize:   chunkSize,
		setSize:     setSize,
		dbEntrySize: cfg.DBEntryByteNum / 8,
		partitions:  partitions,
	}

	return srv, nil
}

func (s *Server) GetMetadata(ctx context.Context, req *rpc.MetadataRequest) (*rpc.MetadataReply, error) {
	requestID := rpc.RequestIDFromContext(ctx)
	logStep("get_metadata", "start", requestID, -1, -1, -1)
	defer logStep("get_metadata", "end", requestID, -1, -1, -1)
	return &rpc.MetadataReply{
		N:                   s.cfg.N,
		Dim:                 s.cfg.Dim,
		M:                   s.cfg.M,
		DbEntryByteNum:      s.cfg.DBEntryByteNum,
		PartitionNum:        s.cfg.PartitionNum,
		PartitionSize:       s.cfg.PartitionSize,
		ChunkSize:           s.chunkSize,
		SetSize:             s.setSize,
		FailureProbLog2:     s.cfg.FailureProbLog2,
		PqEnabled:           s.cfg.PqEnabled,
		PqM:                 s.cfg.PqM,
		PqKsub:              s.cfg.PqKsub,
		PqDsub:              s.cfg.PqDsub,
		PqNeighborCodeCount: s.cfg.PqNeighborCodeCount,
		PqCodebook:          s.cfg.PqCodebook,

		PreprocessProtocolVersion: rpc.PreprocessProtocolClientHints,
	}, nil
}

func (s *Server) PrivateQuery(ctx context.Context, req *rpc.PrivateQueryRequest) (*rpc.PrivateQueryReply, error) {
	requestID := rpc.RequestIDFromContext(ctx)
	logStep("private_query", "start", requestID, -1, -1, int(req.PartitionId))
	defer logStep("private_query", "end", requestID, -1, -1, int(req.PartitionId))
	if req.PartitionId >= uint64(len(s.partitions)) {
		return nil, fmt.Errorf("partition id out of range: %d", req.PartitionId)
	}
	if len(req.Offsets) < int(s.setSize) {
		return nil, fmt.Errorf("offset size mismatch: want %d got %d", s.setSize, len(req.Offsets))
	}

	return &rpc.PrivateQueryReply{Entry: encodeU64s(s.answer(req.PartitionId, req.Offsets))}, nil
}

// answer XORs the entries of one offset set of partition pid: entry
// offsets[i] of chunk i, skipping the padding past the stored entries.
func (s *Server) answer(pid uint64, offsets []uint32) []uint64 {
	rawDB := s.partitions[pid]
	dbSize := uint64(len(rawDB)) / s.dbEntrySize
	ret := make([]uint64, s.dbEntrySize)

	for i := uint64(0); i < s.setSize; i++ {
		idx := uint64(offsets[int(i)]) + i*s.chunkSize
		if idx >= dbSize {
			continue
		}
		start := int(idx * s.dbEntrySize)
		end := start + int(s.dbEntrySize)
		entry := rawDB[start:end]
		xorEntry64(ret, entry)
	}
	return ret
}

// BatchPrivateQuery answers Count offset sets of one partition in one round
// trip. The sets are computed one after another, each exactly as
// PrivateQuery computes it, on a single goroutine.
func (s *Server) BatchPrivateQuery(ctx context.Context, req *rpc.BatchPrivateQueryRequest) (*rpc.BatchPrivateQueryReply, error) {
	requestID := rpc.RequestIDFromContext(ctx)
	logStep("batch_private_query", "start", requestID, -1, -1, int(req.PartitionId))
	defer logStep("batch_private_query", "end", requestID, -1, -1, int(req.PartitionId))
	if req.PartitionId >= uint64(len(s.partitions)) {
		return nil, fmt.Errorf("partition id out of range: %d", req.PartitionId)
	}
	if req.Count == 0 || uint64(len(req.Offsets)) != uint64(req.Count)*s.setSize {
		return nil, fmt.Errorf("batch offset size mismatch: want %d sets of %d, got %d offsets", req.Count, s.setSize, len(req.Offsets))
	}
	entries := make([][]byte, req.Count)
	for i := range entries {
		set := req.Offsets[uint64(i)*s.setSize : uint64(i+1)*s.setSize]
		entries[i] = encodeU64s(s.answer(req.PartitionId, set))
	}
	return &rpc.BatchPrivateQueryReply{Entries: entries}, nil
}

// StartPreprocess streams partition PartitionId of the DB to the client in
// blocks of pianopir.StreamBlockEntries entries, from entry 0 up to
// PartitionSize, zero padding the entries past the stored ones. The block
// count, order and sizes depend only on the public partition size, chunk size
// and entry length; the request carries no client secret. The client builds
// its hints from the stream (protocol rpc.PreprocessProtocolClientHints).
func (s *Server) StartPreprocess(req *rpc.PreprocessRequest, stream rpc.PirService_StartPreprocessServer) error {
	requestID := rpc.RequestIDFromContext(stream.Context())
	logStep("start_preprocess", "start", requestID, -1, -1, -1)
	defer logStep("start_preprocess", "end", requestID, -1, -1, -1)

	if req.ProtocolVersion != rpc.PreprocessProtocolClientHints {
		return status.Errorf(codes.FailedPrecondition,
			"preprocess protocol version %d is not supported: this server streams the DB so that the client builds its hints (version %d) and no longer builds hints from a client PRF key; upgrade the client",
			req.ProtocolVersion, rpc.PreprocessProtocolClientHints)
	}
	pid := int(req.PartitionId)
	if pid < 0 || pid >= len(s.partitions) {
		return fmt.Errorf("partition id out of range: %d", pid)
	}

	rawDB := s.partitions[pid]
	blockEntries := pianopir.StreamBlockEntries(s.chunkSize, s.cfg.DBEntryByteNum)
	blockCount := pianopir.StreamBlockCount(s.cfg.PartitionSize, blockEntries)
	logStep("stream_db", "start", requestID, -1, -1, pid)
	for b := uint64(0); b < blockCount; b++ {
		first := b * blockEntries
		n := pianopir.StreamBlockLen(s.cfg.PartitionSize, blockEntries, b)
		words := make([]uint64, n*s.dbEntrySize)
		lo := min(first*s.dbEntrySize, uint64(len(rawDB)))
		hi := min((first+n)*s.dbEntrySize, uint64(len(rawDB)))
		copy(words, rawDB[lo:hi])
		err := stream.Send(&rpc.PreprocessChunk{
			PartitionId:     uint64(pid),
			ProtocolVersion: rpc.PreprocessProtocolClientHints,
			BlockIndex:      b,
			FirstEntry:      first,
			Entries:         encodeU64s(words),
		})
		if err != nil {
			return fmt.Errorf("send preprocess block %d: %w", b, err)
		}
	}
	logStep("stream_db", "end", requestID, -1, -1, pid)
	return nil
}

// PlainQuery returns raw entries by vertex id without any PIR aggregation.
// It is the non-private counterpart of PrivateQuery used to measure the
// communication overhead introduced by privacy.
func (s *Server) PlainQuery(ctx context.Context, req *rpc.PlainQueryRequest) (*rpc.PlainQueryReply, error) {
	requestID := rpc.RequestIDFromContext(ctx)
	logStep("plain_query", "start", requestID, -1, -1, -1)
	defer logStep("plain_query", "end", requestID, -1, -1, -1)

	entries := make([][]byte, len(req.Ids))
	for i, id := range req.Ids {
		// PlainQuery accepts vertex IDs directly, so any out-of-range id
		// (whether the partition or the in-partition offset is past the
		// real dataset size N) is a client bug. Fail fast for easier
		// debugging — unlike PrivateQuery, there is no PIR padding region
		// that legitimately needs to be probed.
		if id >= s.cfg.N {
			return nil, fmt.Errorf("vertex id out of range: %d", id)
		}
		pid := id / s.cfg.PartitionSize
		offset := id % s.cfg.PartitionSize
		if pid >= uint64(len(s.partitions)) {
			return nil, fmt.Errorf("vertex id out of range: %d", id)
		}
		rawDB := s.partitions[pid]
		dbSize := uint64(len(rawDB)) / s.dbEntrySize
		if offset >= dbSize {
			return nil, fmt.Errorf("vertex id out of range: %d", id)
		}
		start := int(offset * s.dbEntrySize)
		end := start + int(s.dbEntrySize)
		entries[i] = encodeU64s(rawDB[start:end])
	}
	return &rpc.PlainQueryReply{Entries: entries}, nil
}

// GetStartVertices returns randomly selected start vertices without consuming PIR queries.
func (s *Server) GetStartVertices(ctx context.Context, req *rpc.GetStartVerticesRequest) (*rpc.GetStartVerticesReply, error) {
	requestID := rpc.RequestIDFromContext(ctx)
	logStep("get_start_vertices", "start", requestID, -1, -1, -1)
	defer logStep("get_start_vertices", "end", requestID, -1, -1, -1)

	num := req.Num
	n := s.cfg.N
	if num > n {
		num = n
	}

	added := make(map[uint64]bool, num)
	ids := make([]uint64, 0, num)
	buf := make([]byte, 8)
	for uint64(len(ids)) < num {
		if _, err := rand.Read(buf); err != nil {
			return nil, fmt.Errorf("crypto/rand.Read: %w", err)
		}
		x := binary.LittleEndian.Uint64(buf) % n
		if added[x] {
			continue
		}
		added[x] = true
		ids = append(ids, x)
	}

	vertices := make([]*rpc.StartVertexEntry, len(ids))
	for i, id := range ids {
		pid := id / s.cfg.PartitionSize
		offset := id % s.cfg.PartitionSize
		if int(pid) >= len(s.partitions) {
			continue
		}
		start := int(offset * s.dbEntrySize)
		end := start + int(s.dbEntrySize)
		entry := s.partitions[pid][start:end]
		vertices[i] = &rpc.StartVertexEntry{
			Id:    id,
			Entry: encodeU64s(entry),
		}
	}

	return &rpc.GetStartVerticesReply{Vertices: vertices}, nil
}

func (s *Server) Health(ctx context.Context, req *rpc.HealthRequest) (*rpc.HealthReply, error) {
	requestID := rpc.RequestIDFromContext(ctx)
	logStep("health", "start", requestID, -1, -1, -1)
	defer logStep("health", "end", requestID, -1, -1, -1)
	return &rpc.HealthReply{}, nil
}

func xorEntry64(dst, src []uint64) {
	for i := range dst {
		dst[i] ^= src[i]
	}
}

func encodeU64s(vals []uint64) []byte {
	out := make([]byte, len(vals)*8)
	for i, v := range vals {
		binary.LittleEndian.PutUint64(out[i*8:], v)
	}
	return out
}
