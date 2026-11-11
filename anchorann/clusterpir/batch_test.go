package clusterpir

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"net"
	"reflect"
	"sync"
	"testing"

	"example.com/private-search/rpc"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// sentSet is one offset set the server received.
type sentSet struct {
	offsets string
	dummy   bool
}

// recordServer records, per partition, every offset set it answers (split
// out of BatchPrivateQuery), the round trips and the batch sizes, and can
// fail the next BatchPrivateQuery.
type recordServer struct {
	rpc.PirServiceServer
	mu       sync.Mutex
	sets     map[uint64][]sentSet
	rpcs     int
	counts   map[uint32]int // BatchPrivateQuery Count -> calls
	dummies  map[string]bool
	failNext bool
}

func newRecordServer(srv rpc.PirServiceServer) *recordServer {
	return &recordServer{PirServiceServer: srv, sets: map[uint64][]sentSet{}, counts: map[uint32]int{}, dummies: map[string]bool{}}
}

func (s *recordServer) add(p uint64, offsets []uint32) {
	k := fmt.Sprint(offsets)
	s.sets[p] = append(s.sets[p], sentSet{offsets: k, dummy: s.dummies[k]})
}

func (s *recordServer) PrivateQuery(ctx context.Context, in *rpc.PrivateQueryRequest) (*rpc.PrivateQueryReply, error) {
	s.mu.Lock()
	s.rpcs++
	s.add(in.PartitionId, in.Offsets)
	s.mu.Unlock()
	return s.PirServiceServer.PrivateQuery(ctx, in)
}

func (s *recordServer) BatchPrivateQuery(ctx context.Context, in *rpc.BatchPrivateQueryRequest) (*rpc.BatchPrivateQueryReply, error) {
	s.mu.Lock()
	s.rpcs++
	s.counts[in.Count]++
	fail := s.failNext
	s.failNext = false
	if !fail && in.Count > 0 {
		set := len(in.Offsets) / int(in.Count)
		for i := 0; i < int(in.Count); i++ {
			s.add(in.PartitionId, in.Offsets[i*set:(i+1)*set])
		}
	}
	s.mu.Unlock()
	if fail {
		return nil, errors.New("injected BatchPrivateQuery failure")
	}
	return s.PirServiceServer.BatchPrivateQuery(ctx, in)
}

// seedDummies gives every partition client of f a reproducible dummy source
// whose sets are registered as dummies with srv.
func seedDummies(f *Fetcher, srv *recordServer, seed int64) {
	for i, p := range f.parts {
		rng := rand.New(rand.NewSource(seed + int64(i)))
		p.pc.SetDummyOffsetSource(func(setSize, chunkSize uint64) []uint32 {
			out := make([]uint32, setSize)
			for j := range out {
				out[j] = uint32(rng.Uint64() & (chunkSize - 1))
			}
			srv.mu.Lock()
			srv.dummies[fmt.Sprint(out)] = true
			srv.mu.Unlock()
			return out
		})
	}
}

// cloneFetcher copies f's client state (hint tables, replacements, cache)
// into a fetcher that talks to client in the given RPC mode.
func cloneFetcher(f *Fetcher, client rpc.PirServiceClient, mode string) *Fetcher {
	g := *f
	g.batch.RPC = mode
	g.parts = make([]*partition, len(f.parts))
	for i, p := range f.parts {
		cc := &countingClient{inner: client}
		g.parts[i] = &partition{cc: cc, pc: p.pc.Clone(cc)}
	}
	return &g
}

func sameFetcherState(a, b *Fetcher) bool {
	for i := range a.parts {
		if !a.parts[i].pc.SameHintState(b.parts[i].pc) {
			return false
		}
	}
	return true
}

func sameResult(a, b *Result) bool {
	return reflect.DeepEqual(a.Clusters, b.Clusters) && reflect.DeepEqual(a.Missed, b.Missed)
}

// In the batch mode every query makes P·Rounds round trips of PerPartition
// requests each, whatever clusters are selected, cached or missed.
func TestBatchModeRoundTripsFixed(t *testing.T) {
	const count = 2048 // partitions of 1024: chunk 64, 39 backups per chunk
	db, _, _ := buildDB(t, count, 2, 2)
	rng := rand.New(rand.NewSource(5))
	inputs := map[string]func(q int) []int{
		"concentrated": func(int) []int { return rng.Perm(16)[:4] },
		"spread":       func(int) []int { return rng.Perm(count)[:4] },
		"repeated":     func(int) []int { return []int{1, 2, 1100, 1101} },
		"exhausting":   func(q int) []int { return []int{(2 * q) % 64, (2*q + 1) % 64, 1024 + (2*q)%64, 1024 + (2*q+1)%64} },
	}
	for _, cfg := range []struct {
		parts int
		batch BatchConfig
	}{
		{1, BatchConfig{NProbe: 4, RPC: RPCBatch}},
		{2, BatchConfig{NProbe: 4, PerPartition: 2, Rounds: 2, RPC: RPCBatch}},
	} {
		for name, input := range inputs {
			srv := newRecordServer(mustServer(t, db, cfg.parts))
			f, err := NewFetcher(context.Background(), InProcess(srv), cfg.batch)
			if err != nil {
				t.Fatal(err)
			}
			if err := f.Preprocess(context.Background()); err != nil {
				t.Fatal(err)
			}
			b := f.Batch()
			var total FetchStats
			for q := 0; q < 100; q++ {
				ids := input(q)
				before := srv.rpcs
				res, err := f.Fetch(context.Background(), ids)
				if err != nil {
					t.Fatalf("P=%d %s query %d: %v", cfg.parts, name, q, err)
				}
				checkFetched(t, db, ids, res)
				st := res.Stats
				if srv.rpcs-before != cfg.parts*b.Rounds || st.RPCs != cfg.parts*b.Rounds || st.Requests != b.RequestsPerQuery(cfg.parts) {
					t.Fatalf("P=%d %s query %d: %d round trips, stats %+v", cfg.parts, name, q, srv.rpcs-before, st)
				}
				total.CacheHits += st.CacheHits
				total.PIRMissed += st.PIRMissed
				total.Refreshes += st.Refreshes
			}
			if len(srv.counts) != 1 || srv.counts[uint32(b.PerPartition)] == 0 {
				t.Fatalf("P=%d %s: batch sizes %v, want only %d", cfg.parts, name, srv.counts, b.PerPartition)
			}
			if total.Refreshes == 0 || name == "repeated" && total.CacheHits == 0 || name == "exhausting" && total.PIRMissed == 0 {
				t.Fatalf("P=%d %s: input did not exercise its case: %+v", cfg.parts, name, total)
			}
		}
	}
}

// equivalence counts what a runEquivalence run exercised.
type equivalence struct{ reuses, hits, missed, dummies, refreshes int }

// runEquivalence fetches pick(q) for queries q from the same client state in
// the sequential and the batch mode and checks, query by query, that both
// send the same offset sets with dummies at the same positions, return the
// same clusters and misses, and leave the same client state. Before a query
// that would exceed MaxQueryNum both sides are rebuilt together (a rebuild
// draws a fresh key).
func runEquivalence(t *testing.T, count, nProbe, queries int, pick func(q int, seq *Fetcher, inner rpc.PirServiceClient, chunk int) []int) equivalence {
	t.Helper()
	db, _, _ := buildDB(t, count, 2, 2)
	inner := mustServer(t, db, 1)
	base, err := NewFetcher(context.Background(), InProcess(inner), BatchConfig{NProbe: nProbe})
	if err != nil {
		t.Fatal(err)
	}
	if err := base.Preprocess(context.Background()); err != nil {
		t.Fatal(err)
	}
	meta, _ := inner.GetMetadata(context.Background(), &rpc.MetadataRequest{})
	seqSrv, batSrv := newRecordServer(inner), newRecordServer(inner)
	seq := cloneFetcher(base, InProcess(seqSrv), RPCSequential)
	bat := cloneFetcher(base, InProcess(batSrv), RPCBatch)
	var ev equivalence
	for q := 0; q < queries; q++ {
		if seq.parts[0].pc.RequestsSincePreprocess()+uint64(nProbe) > uint64(seq.MaxQueryNum()) {
			if err := seq.Preprocess(context.Background()); err != nil {
				t.Fatal(err)
			}
			bat = cloneFetcher(seq, InProcess(batSrv), RPCBatch)
			ev.refreshes++
		}
		ids := pick(q, seq, InProcess(inner), int(meta.ChunkSize))
		seedDummies(seq, seqSrv, int64(q))
		seedDummies(bat, batSrv, int64(q))

		sBefore, bBefore := len(seqSrv.sets[0]), len(batSrv.sets[0])
		sr, err := seq.Fetch(context.Background(), ids)
		if err != nil {
			t.Fatal(err)
		}
		br, err := bat.Fetch(context.Background(), ids)
		if err != nil {
			t.Fatal(err)
		}
		checkFetched(t, db, ids, sr)
		checkFetched(t, db, ids, br)
		if !reflect.DeepEqual(seqSrv.sets[0][sBefore:], batSrv.sets[0][bBefore:]) {
			t.Fatalf("query %d %v: offset sets differ\nsequential %v\nbatch      %v", q, ids, seqSrv.sets[0][sBefore:], batSrv.sets[0][bBefore:])
		}
		if !sameResult(sr, br) {
			t.Fatalf("query %d %v: results differ: missed %v vs %v", q, ids, sr.Missed, br.Missed)
		}
		if !sameFetcherState(seq, bat) {
			t.Fatalf("query %d %v: client state differs", q, ids)
		}
		if sr.Stats.RPCs != nProbe || br.Stats.RPCs != 1 || sr.Stats.Requests != br.Stats.Requests {
			t.Fatalf("query %d: round trips sequential %+v batch %+v", q, sr.Stats, br.Stats)
		}
		ev.reuses += br.Stats.PendingReuses
		ev.hits += br.Stats.CacheHits
		ev.missed += len(br.Missed)
	}
	for _, s := range batSrv.sets[0] {
		if s.dummy {
			ev.dummies++
		}
	}
	return ev
}

// (a) several clusters of one chunk until its backups run out, cache hits
// and random queries, across hint generations.
func TestBatchModeMatchesSequential(t *testing.T) {
	// one partition of 1024: chunk 64, 41 backups per chunk, MaxQueryNum 221
	rng := rand.New(rand.NewSource(2))
	ev := runEquivalence(t, 1024, 4, 160, func(q int, _ *Fetcher, _ rpc.PirServiceClient, chunk int) []int {
		switch {
		case q%5 == 3:
			return []int{0, 1, 2, 3} // cache hits once fetched
		case q%2 == 0: // distinct clusters of chunk 0
			b := q % chunk
			return []int{b, b + 1, chunk + b, 2*chunk + b}
		}
		return rng.Perm(1024)[:4]
	})
	if ev.hits == 0 || ev.missed == 0 || ev.dummies == 0 || ev.refreshes == 0 {
		t.Fatalf("cases not exercised: %+v", ev)
	}
}

// findReusePair returns two clusters of different chunks such that, fetched
// together in batch mode from f's state, the second reuses the hint the first
// replaced.
func findReusePair(t *testing.T, f *Fetcher, client rpc.PirServiceClient, chunk int) []int {
	t.Helper()
	for a := 0; a < f.partSize; a++ {
		for b := 0; b < f.partSize; b++ {
			if a/chunk == b/chunk {
				continue
			}
			g := cloneFetcher(f, client, RPCBatch)
			res, err := g.Fetch(context.Background(), []int{a, b})
			if err != nil {
				t.Fatal(err)
			}
			if res.Stats.PendingReuses > 0 {
				return []int{a, b}
			}
		}
	}
	t.Fatal("no pair reuses a hint replaced in the same batch")
	return nil
}

// (b) the second slot uses the hint the first slot of the same round trip
// replaced (forced in a small DB, where such pairs are easy to find).
func TestBatchModeHintReuseMatchesSequential(t *testing.T) {
	// one partition of 64: chunk 16, 4 chunks, MaxQueryNum 33
	rng := rand.New(rand.NewSource(3))
	cases := 0
	ev := runEquivalence(t, 64, 2, 60, func(q int, seq *Fetcher, inner rpc.PirServiceClient, chunk int) []int {
		if q%3 == 0 {
			cases++
			return findReusePair(t, seq, inner, chunk)
		}
		return rng.Perm(64)[:2]
	})
	if ev.reuses < cases || ev.refreshes == 0 {
		t.Fatalf("reuses=%d, want at least %d: %+v", ev.reuses, cases, ev)
	}
}

// Independent fetchers (each with its own keys) return the same clusters and
// misses in both modes, across hint rebuilds.
func TestBatchModeResultsAcrossRefreshes(t *testing.T) {
	const count, nProbe = 256, 4 // 22 queries per rebuild
	db, _, _ := buildDB(t, count, 2, 2)
	fs := map[string]*Fetcher{}
	for _, mode := range []string{RPCSequential, RPCBatch} {
		fs[mode], _ = newFetcher(t, db, 1, BatchConfig{NProbe: nProbe, RPC: mode})
	}
	rng := rand.New(rand.NewSource(6))
	refreshes := map[string]int{}
	for q := 0; q < 3*fs[RPCBatch].FetchesPerRefresh()+5; q++ {
		ids := rng.Perm(count)[:nProbe]
		if q%3 == 0 {
			for i := range ids {
				ids[i] = (4*q + i) % 32 // chunk 0: same chunk, backups run out
			}
		}
		var res [2]*Result
		for i, mode := range []string{RPCSequential, RPCBatch} {
			r, err := fs[mode].Fetch(context.Background(), ids)
			if err != nil {
				t.Fatalf("%s query %d: %v", mode, q, err)
			}
			checkFetched(t, db, ids, r)
			refreshes[mode] += r.Stats.Refreshes
			res[i] = r
		}
		if !sameResult(res[0], res[1]) {
			t.Fatalf("query %d: missed %v vs %v", q, res[0].Missed, res[1].Missed)
		}
	}
	if refreshes[RPCSequential] < 3 || refreshes[RPCSequential] != refreshes[RPCBatch] {
		t.Fatalf("refreshes %v", refreshes)
	}
}

// A failed round trip returns an error and the next Fetch starts with a
// rebuild, after which clusters are correct again.
func TestBatchModeFailureRebuilds(t *testing.T) {
	db, _, _ := buildDB(t, 256, 2, 2)
	srv := newRecordServer(mustServer(t, db, 1))
	f, err := NewFetcher(context.Background(), InProcess(srv), BatchConfig{NProbe: 4, RPC: RPCBatch})
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Preprocess(context.Background()); err != nil {
		t.Fatal(err)
	}
	ids := []int{3, 70, 140, 200}
	if res, err := f.Fetch(context.Background(), ids); err != nil || res.Stats.Refreshes != 0 {
		t.Fatalf("first fetch: %v", err)
	}
	srv.failNext = true
	if _, err := f.Fetch(context.Background(), []int{5, 6, 7, 8}); err == nil {
		t.Fatal("failed round trip not reported")
	}
	res, err := f.Fetch(context.Background(), ids)
	if err != nil {
		t.Fatal(err)
	}
	checkFetched(t, db, ids, res)
	if res.Stats.Refreshes != 1 || res.Stats.CacheHits != 0 || len(res.Missed) != 0 {
		t.Fatalf("fetch after the failure: %+v", res.Stats)
	}
}

func TestBatchModeConfig(t *testing.T) {
	db, _, _ := buildDB(t, 64, 2, 2)
	for _, mode := range []string{"", RPCSequential} {
		f, _ := newFetcher(t, db, 1, BatchConfig{NProbe: 2, RPC: mode})
		if f.Batch().RPC != RPCSequential || f.Batch().RPCsPerQuery(1) != 2 {
			t.Fatalf("mode %q: %+v", mode, f.Batch())
		}
	}
	f, _ := newFetcher(t, db, 2, BatchConfig{NProbe: 4, RPC: RPCBatch})
	if b := f.Batch(); b.RPC != RPCBatch || b.Rounds != 3 || b.RPCsPerQuery(2) != 6 || b.RequestsPerQuery(2) != 12 {
		t.Fatalf("default batch keeps the mode: %+v", b)
	}
	if _, err := NewFetcher(context.Background(), InProcess(mustServer(t, db, 1)), BatchConfig{NProbe: 2, RPC: "parallel"}); err == nil {
		t.Fatal("unknown mode accepted")
	}
}

func TestBatchModeOverGRPCLoopback(t *testing.T) {
	db, _, _ := buildDB(t, 30, 3, 4)
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	gs := grpc.NewServer()
	rpc.RegisterPirServiceServer(gs, mustServer(t, db, 2))
	go gs.Serve(lis)
	defer gs.Stop()
	conn, err := grpc.Dial(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	f, err := NewFetcher(context.Background(), rpc.NewPirServiceClient(conn), BatchConfig{NProbe: 4, PerPartition: 2, Rounds: 3, RPC: RPCBatch})
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Preprocess(context.Background()); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(11))
	for q := 0; q < 10; q++ {
		ids := rng.Perm(30)[:4]
		res, err := f.Fetch(context.Background(), ids)
		if err != nil {
			t.Fatal(err)
		}
		checkFetched(t, db, ids, res)
		st := res.Stats
		if st.Requests != 12 || st.RPCs != 6 || st.Online.QueryRPCs != 6 || st.Online.QueryUpBytes <= 0 || st.Online.QueryDownBytes < int64(12*db.Layout.EntryBytes()) {
			t.Fatalf("stats %+v", st)
		}
	}
}
