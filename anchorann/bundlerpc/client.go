package bundlerpc

import (
	"context"
	"fmt"
	"io"
	"time"

	"privatesearch/anchorann/bundle"
)

// FetchStats records the distribution cost observed by the client.
type FetchStats struct {
	Bytes         int     `json:"bytes"`
	Chunks        int     `json:"chunks"`
	FetchSeconds  float64 `json:"fetch_seconds"`
	DecodeSeconds float64 `json:"decode_and_validate_seconds"`
	SHA256        string  `json:"sha256"`
}

// Fetch downloads the served bundle, checks that the chunks are contiguous,
// the size and SHA-256 match GetInfo and the header shape matches the info
// and want, then decodes and fully validates it. Any mismatch is an error.
func Fetch(ctx context.Context, c AnchorBundleClient, want bundle.Expect) (*bundle.Bundle, []byte, FetchStats, error) {
	var st FetchStats
	start := time.Now()
	info, err := c.GetInfo(ctx, &InfoRequest{})
	if err != nil {
		return nil, nil, st, fmt.Errorf("bundle info: %w", err)
	}
	if info.GetFormatVersion() != bundle.FormatVersion {
		return nil, nil, st, fmt.Errorf("server bundle version=%d, client supports %d", info.GetFormatVersion(), bundle.FormatVersion)
	}
	const maxBytes = 1 << 34
	if info.GetTotalBytes() == 0 || info.GetTotalBytes() > maxBytes {
		return nil, nil, st, fmt.Errorf("server bundle size=%d outside 1..%d", info.GetTotalBytes(), uint64(maxBytes))
	}
	stream, err := c.Fetch(ctx, &FetchRequest{Sha256: info.GetSha256()})
	if err != nil {
		return nil, nil, st, fmt.Errorf("fetch bundle: %w", err)
	}
	data := make([]byte, 0, info.GetTotalBytes())
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, st, fmt.Errorf("receive bundle: %w", err)
		}
		if chunk.GetOffset() != uint64(len(data)) || uint64(len(data)+len(chunk.GetData())) > info.GetTotalBytes() {
			return nil, nil, st, fmt.Errorf("bundle chunk offset=%d bytes=%d after %d of %d bytes", chunk.GetOffset(), len(chunk.GetData()), len(data), info.GetTotalBytes())
		}
		data = append(data, chunk.GetData()...)
		st.Chunks++
	}
	st.FetchSeconds = time.Since(start).Seconds()
	st.Bytes = len(data)
	if uint64(len(data)) != info.GetTotalBytes() {
		return nil, nil, st, fmt.Errorf("bundle truncated: received %d of %d bytes", len(data), info.GetTotalBytes())
	}
	start = time.Now()
	st.SHA256 = bundle.Checksum(data)
	if st.SHA256 != info.GetSha256() {
		return nil, nil, st, fmt.Errorf("bundle sha256=%s announced %s", st.SHA256, info.GetSha256())
	}
	b, err := bundle.Decode(data, want)
	if err != nil {
		return nil, nil, st, err
	}
	if uint64(b.Count) != info.GetClusterCount() || uint64(b.Dim) != info.GetDimension() || uint64(b.Size) != info.GetClusterSize() {
		return nil, nil, st, fmt.Errorf("bundle C=%d dimension=%d s=%d differ from info C=%d dimension=%d s=%d", b.Count, b.Dim, b.Size, info.GetClusterCount(), info.GetDimension(), info.GetClusterSize())
	}
	st.DecodeSeconds = time.Since(start).Seconds()
	return b, data, st, nil
}
