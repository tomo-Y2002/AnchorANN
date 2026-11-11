// Package bundlerpc distributes an encoded AnchorANN bundle over gRPC, in a
// service separate from the PIR service. The bundle is public data, so the
// transport adds no encryption or authentication.
package bundlerpc

import (
	"context"
	"fmt"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"privatesearch/anchorann/bundle"
)

// DefaultChunkBytes keeps each streamed message well below gRPC's default
// 4 MiB receive limit.
const DefaultChunkBytes = 1 << 20

// Server serves one immutable encoded bundle to any number of clients.
type Server struct {
	UnimplementedAnchorBundleServer
	data  []byte
	chunk int
	info  *Info
}

// NewServer decodes and validates data before serving it, so a corrupt
// bundle is never distributed. chunkBytes <= 0 selects DefaultChunkBytes.
func NewServer(data []byte, chunkBytes int) (*Server, error) {
	b, err := bundle.Decode(data, bundle.Expect{})
	if err != nil {
		return nil, fmt.Errorf("serve bundle: %w", err)
	}
	if chunkBytes <= 0 {
		chunkBytes = DefaultChunkBytes
	}
	return &Server{data: data, chunk: chunkBytes, info: &Info{
		FormatVersion: b.Version, TotalBytes: uint64(len(data)), Sha256: bundle.Checksum(data),
		ChunkBytes: uint32(chunkBytes), ClusterCount: uint64(b.Count), Dimension: uint64(b.Dim), ClusterSize: uint64(b.Size),
	}}, nil
}

// GetInfo returns the bundle digest, size and shape.
func (s *Server) GetInfo(context.Context, *InfoRequest) (*Info, error) {
	return s.info, nil
}

// Fetch streams the bundle in order if the requested digest matches.
func (s *Server) Fetch(req *FetchRequest, stream AnchorBundle_FetchServer) error {
	if req.GetSha256() != s.info.Sha256 {
		return status.Errorf(codes.FailedPrecondition, "bundle sha256=%s requested, serving %s", req.GetSha256(), s.info.Sha256)
	}
	for off := 0; off < len(s.data); off += s.chunk {
		end := min(off+s.chunk, len(s.data))
		if err := stream.Send(&Chunk{Offset: uint64(off), Data: s.data[off:end]}); err != nil {
			return err
		}
	}
	return nil
}
