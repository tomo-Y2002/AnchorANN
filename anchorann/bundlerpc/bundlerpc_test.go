package bundlerpc

import (
	"bytes"
	"context"
	"crypto/sha256"
	"net"
	"strings"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	"privatesearch/anchorann"
	"privatesearch/anchorann/bundle"
	"privatesearch/anchorann/nsg"
)

func encodedBundle(t *testing.T) []byte {
	t.Helper()
	const count, size, dim = 64, 4, 8
	a := &anchorann.Artifact{Version: anchorann.FormatVersion, N: count * size, Size: size, Count: count, Dim: dim, Seed: 1, Method: anchorann.Common, Iterations: 1}
	for c := 0; c < count; c++ {
		ids := make([]uint32, size)
		for i := range ids {
			ids[i] = uint32(c*size + i)
		}
		anchor := make([]float64, dim)
		for j := range anchor {
			anchor[j] = float64((c*31 + j*17) % 97)
		}
		a.Members = append(a.Members, ids)
		a.Anchors = append(a.Anchors, anchor)
	}
	b, _, err := bundle.FromArtifact(a, sha256.Sum256([]byte("artifact")), nsg.Params{KNN: 8, L: 16, R: 6, CNSG: 32, Workers: 2}, bundle.PIRParams{})
	if err != nil {
		t.Fatal(err)
	}
	data, err := bundle.Encode(b)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// serve starts a real TCP loopback gRPC server and returns a client for it.
func serve(t *testing.T, srv AnchorBundleServer) AnchorBundleClient {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	gs := grpc.NewServer()
	RegisterAnchorBundleServer(gs, srv)
	go gs.Serve(lis)
	t.Cleanup(gs.Stop)
	conn, err := grpc.Dial(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	return NewAnchorBundleClient(conn)
}

func TestLoopbackFetch(t *testing.T) {
	data := encodedBundle(t)
	srv, err := NewServer(data, 1000) // force many chunks
	if err != nil {
		t.Fatal(err)
	}
	c := serve(t, srv)
	b, got, st, err := Fetch(context.Background(), c, bundle.Expect{Count: 64, Dim: 8, Size: 4})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, data) || st.Bytes != len(data) || st.Chunks != (len(data)+999)/1000 || st.SHA256 != bundle.Checksum(data) {
		t.Fatalf("fetch stats %+v", st)
	}
	want, _ := bundle.Decode(data, bundle.Expect{})
	if b.Count != want.Count || b.Graph.Entry != want.Graph.Entry || b.Graph.Edges() != want.Graph.Edges() {
		t.Fatal("fetched bundle differs")
	}
	// Two clients reuse the same served bundle.
	if _, _, _, err := Fetch(context.Background(), c, bundle.Expect{}); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := Fetch(context.Background(), c, bundle.Expect{Dim: 128}); err == nil || !strings.Contains(err.Error(), "expected") {
		t.Fatalf("dimension expectation not enforced: %v", err)
	}
}

func TestServerRejectsCorruptBundleAndWrongDigest(t *testing.T) {
	data := encodedBundle(t)
	bad := append([]byte(nil), data...)
	bad[100] ^= 1
	if _, err := NewServer(bad, 0); err == nil {
		t.Fatal("corrupt bundle served")
	}
	srv, err := NewServer(data, 0)
	if err != nil {
		t.Fatal(err)
	}
	if srv.chunk != DefaultChunkBytes {
		t.Fatalf("chunk=%d", srv.chunk)
	}
	c := serve(t, srv)
	stream, err := c.Fetch(context.Background(), &FetchRequest{Sha256: "stale"})
	if err == nil {
		_, err = stream.Recv()
	}
	if status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("stale digest err=%v", err)
	}
}

// fakeServer serves arbitrary info and chunks to exercise client checks.
type fakeServer struct {
	UnimplementedAnchorBundleServer
	info   *Info
	chunks []*Chunk
}

func (f *fakeServer) GetInfo(context.Context, *InfoRequest) (*Info, error) { return f.info, nil }

func (f *fakeServer) Fetch(_ *FetchRequest, s AnchorBundle_FetchServer) error {
	for _, c := range f.chunks {
		if err := s.Send(c); err != nil {
			return err
		}
	}
	return nil
}

func TestClientRejectsTamperedDistribution(t *testing.T) {
	data := encodedBundle(t)
	good, _ := NewServer(data, 0)
	info := func(mod func(i *Info)) *Info {
		i := proto.Clone(good.info).(*Info)
		mod(i)
		return i
	}
	tampered := append([]byte(nil), data...)
	tampered[len(tampered)/2] ^= 1
	half := len(data) / 2
	cases := []struct {
		name string
		srv  *fakeServer
		msg  string
	}{
		{"altered bytes", &fakeServer{info: good.info, chunks: []*Chunk{{Data: tampered}}}, "sha256"},
		{"altered bytes with matching announced digest", &fakeServer{info: info(func(i *Info) { i.Sha256 = bundle.Checksum(tampered) }), chunks: []*Chunk{{Data: tampered}}}, "checksum mismatch"},
		{"gap", &fakeServer{info: good.info, chunks: []*Chunk{{Data: data[:half]}, {Offset: uint64(half + 1), Data: data[half+1:]}}}, "offset"},
		{"oversize", &fakeServer{info: good.info, chunks: []*Chunk{{Data: append(append([]byte(nil), data...), 0)}}}, "offset"},
		{"truncated", &fakeServer{info: good.info, chunks: []*Chunk{{Data: data[:half]}}}, "truncated"},
		{"version", &fakeServer{info: info(func(i *Info) { i.FormatVersion = 9 })}, "version"},
		{"empty", &fakeServer{info: info(func(i *Info) { i.TotalBytes = 0 })}, "size"},
		{"info shape", &fakeServer{info: info(func(i *Info) { i.Dimension = 16 }), chunks: []*Chunk{{Data: data}}}, "differ"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, _, _, err := Fetch(context.Background(), serve(t, c.srv), bundle.Expect{}); err == nil || !strings.Contains(err.Error(), c.msg) {
				t.Fatalf("err=%v want %q", err, c.msg)
			}
		})
	}
}
