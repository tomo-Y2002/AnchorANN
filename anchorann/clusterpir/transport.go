package clusterpir

import (
	"context"
	"io"

	"example.com/private-search/rpc"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/proto"
)

// Traffic counts PirService messages and their protobuf payload bytes
// (excluding gRPC/HTTP2 framing). Up is client→server, Down is server→client.
type Traffic struct {
	Queries          int   `json:"private_queries"` // PIR requests (one per offset set)
	QueryRPCs        int   `json:"query_rpcs"`      // PrivateQuery / BatchPrivateQuery round trips
	QueryUpBytes     int64 `json:"query_up_bytes"`
	QueryDownBytes   int64 `json:"query_down_bytes"`
	Preprocesses     int   `json:"preprocesses"`
	PreprocUpBytes   int64 `json:"preprocess_up_bytes"`
	PreprocDownBytes int64 `json:"preprocess_down_bytes"`
}

// Add accumulates o into t.
func (t *Traffic) Add(o Traffic) {
	t.Queries += o.Queries
	t.QueryRPCs += o.QueryRPCs
	t.QueryUpBytes += o.QueryUpBytes
	t.QueryDownBytes += o.QueryDownBytes
	t.Preprocesses += o.Preprocesses
	t.PreprocUpBytes += o.PreprocUpBytes
	t.PreprocDownBytes += o.PreprocDownBytes
}

// Sub returns t - o.
func (t Traffic) Sub(o Traffic) Traffic {
	return Traffic{
		Queries: t.Queries - o.Queries, QueryRPCs: t.QueryRPCs - o.QueryRPCs, QueryUpBytes: t.QueryUpBytes - o.QueryUpBytes, QueryDownBytes: t.QueryDownBytes - o.QueryDownBytes,
		Preprocesses: t.Preprocesses - o.Preprocesses, PreprocUpBytes: t.PreprocUpBytes - o.PreprocUpBytes, PreprocDownBytes: t.PreprocDownBytes - o.PreprocDownBytes,
	}
}

// countingClient wraps one partition's view of the PirService. It is used by a
// single goroutine (one pirclient.RemotePIRClient), so counters are plain ints.
// The client only ever issues PrivateQuery, BatchPrivateQuery and
// StartPreprocess; PlainQuery and GetStartVertices are passed through but
// never called by the fetcher.
type countingClient struct {
	inner rpc.PirServiceClient
	t     Traffic
}

func (c *countingClient) GetMetadata(ctx context.Context, in *rpc.MetadataRequest, opts ...grpc.CallOption) (*rpc.MetadataReply, error) {
	return c.inner.GetMetadata(ctx, in, opts...)
}

func (c *countingClient) StartPreprocess(ctx context.Context, in *rpc.PreprocessRequest, opts ...grpc.CallOption) (rpc.PirService_StartPreprocessClient, error) {
	c.t.Preprocesses++
	c.t.PreprocUpBytes += int64(proto.Size(in))
	s, err := c.inner.StartPreprocess(ctx, in, opts...)
	if err != nil {
		return nil, err
	}
	return &countingStream{PirService_StartPreprocessClient: s, t: &c.t}, nil
}

func (c *countingClient) PrivateQuery(ctx context.Context, in *rpc.PrivateQueryRequest, opts ...grpc.CallOption) (*rpc.PrivateQueryReply, error) {
	c.t.Queries++
	c.t.QueryRPCs++
	c.t.QueryUpBytes += int64(proto.Size(in))
	r, err := c.inner.PrivateQuery(ctx, in, opts...)
	if err == nil {
		c.t.QueryDownBytes += int64(proto.Size(r))
	}
	return r, err
}

func (c *countingClient) BatchPrivateQuery(ctx context.Context, in *rpc.BatchPrivateQueryRequest, opts ...grpc.CallOption) (*rpc.BatchPrivateQueryReply, error) {
	c.t.Queries += int(in.Count)
	c.t.QueryRPCs++
	c.t.QueryUpBytes += int64(proto.Size(in))
	r, err := c.inner.BatchPrivateQuery(ctx, in, opts...)
	if err == nil {
		c.t.QueryDownBytes += int64(proto.Size(r))
	}
	return r, err
}

func (c *countingClient) PlainQuery(ctx context.Context, in *rpc.PlainQueryRequest, opts ...grpc.CallOption) (*rpc.PlainQueryReply, error) {
	return c.inner.PlainQuery(ctx, in, opts...)
}

func (c *countingClient) GetStartVertices(ctx context.Context, in *rpc.GetStartVerticesRequest, opts ...grpc.CallOption) (*rpc.GetStartVerticesReply, error) {
	return c.inner.GetStartVertices(ctx, in, opts...)
}

func (c *countingClient) Health(ctx context.Context, in *rpc.HealthRequest, opts ...grpc.CallOption) (*rpc.HealthReply, error) {
	return c.inner.Health(ctx, in, opts...)
}

type countingStream struct {
	rpc.PirService_StartPreprocessClient
	t *Traffic
}

func (s *countingStream) Recv() (*rpc.PreprocessChunk, error) {
	m, err := s.PirService_StartPreprocessClient.Recv()
	if err == nil {
		s.t.PreprocDownBytes += int64(proto.Size(m))
	}
	return m, err
}

// InProcess adapts a PirServiceServer (e.g. *server.Server from DB.NewServer)
// to the client interface without any network or serialization, for
// single-process experiments and tests. The server must be safe for concurrent
// calls; rpc/server.Server only reads its partitions.
func InProcess(srv rpc.PirServiceServer) rpc.PirServiceClient { return inProcess{srv} }

type inProcess struct{ srv rpc.PirServiceServer }

func (c inProcess) GetMetadata(ctx context.Context, in *rpc.MetadataRequest, _ ...grpc.CallOption) (*rpc.MetadataReply, error) {
	return c.srv.GetMetadata(ctx, in)
}

func (c inProcess) StartPreprocess(ctx context.Context, in *rpc.PreprocessRequest, _ ...grpc.CallOption) (rpc.PirService_StartPreprocessClient, error) {
	// The server runs in its own goroutine and hands over one block at a
	// time, so neither side holds the whole partition.
	ctx, cancel := context.WithCancel(ctx)
	s := &chunkStream{ctx: ctx, cancel: cancel, ch: make(chan *rpc.PreprocessChunk)}
	go func() {
		s.err = c.srv.StartPreprocess(in, &collectStream{ctx: ctx, s: s})
		close(s.ch)
	}()
	return s, nil
}

func (c inProcess) PrivateQuery(ctx context.Context, in *rpc.PrivateQueryRequest, _ ...grpc.CallOption) (*rpc.PrivateQueryReply, error) {
	return c.srv.PrivateQuery(ctx, in)
}

func (c inProcess) BatchPrivateQuery(ctx context.Context, in *rpc.BatchPrivateQueryRequest, _ ...grpc.CallOption) (*rpc.BatchPrivateQueryReply, error) {
	return c.srv.BatchPrivateQuery(ctx, in)
}

func (c inProcess) PlainQuery(ctx context.Context, in *rpc.PlainQueryRequest, _ ...grpc.CallOption) (*rpc.PlainQueryReply, error) {
	return c.srv.PlainQuery(ctx, in)
}

func (c inProcess) GetStartVertices(ctx context.Context, in *rpc.GetStartVerticesRequest, _ ...grpc.CallOption) (*rpc.GetStartVerticesReply, error) {
	return c.srv.GetStartVertices(ctx, in)
}

func (c inProcess) Health(ctx context.Context, in *rpc.HealthRequest, _ ...grpc.CallOption) (*rpc.HealthReply, error) {
	return c.srv.Health(ctx, in)
}

// collectStream is the server side of an in-process StartPreprocess.
type collectStream struct {
	ctx context.Context
	s   *chunkStream
}

func (c *collectStream) Send(m *rpc.PreprocessChunk) error {
	select {
	case c.s.ch <- m:
		return nil
	case <-c.ctx.Done():
		return c.ctx.Err()
	}
}
func (c *collectStream) SetHeader(metadata.MD) error  { return nil }
func (c *collectStream) SendHeader(metadata.MD) error { return nil }
func (c *collectStream) SetTrailer(metadata.MD)       {}
func (c *collectStream) Context() context.Context     { return c.ctx }
func (c *collectStream) SendMsg(any) error            { return nil }
func (c *collectStream) RecvMsg(any) error            { return io.EOF }

// chunkStream hands the server's blocks to the client. err is the server's
// result, read after ch is closed.
type chunkStream struct {
	ctx    context.Context
	cancel context.CancelFunc
	ch     chan *rpc.PreprocessChunk
	err    error
}

func (s *chunkStream) Recv() (*rpc.PreprocessChunk, error) {
	m, ok := <-s.ch
	if !ok {
		s.cancel()
		if s.err != nil {
			return nil, s.err
		}
		return nil, io.EOF
	}
	return m, nil
}
func (s *chunkStream) Header() (metadata.MD, error) { return nil, nil }
func (s *chunkStream) Trailer() metadata.MD         { return nil }
func (s *chunkStream) CloseSend() error             { return nil }
func (s *chunkStream) Context() context.Context     { return s.ctx }
func (s *chunkStream) SendMsg(any) error            { return nil }
func (s *chunkStream) RecvMsg(any) error            { return io.EOF }
