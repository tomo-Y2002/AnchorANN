package pirclient

import (
	"context"
	"errors"
	"math/rand"
	"reflect"
	"testing"

	"example.com/private-search/pianopir"
	"example.com/private-search/rpc"
	"google.golang.org/grpc"
)

// offsetLog records every offset set sent, split per request, and can fail
// the next BatchPrivateQuery.
type offsetLog struct {
	*recordingClient
	sets     [][]uint32
	rpcs     int
	failNext bool
}

func (c *offsetLog) PrivateQuery(ctx context.Context, in *rpc.PrivateQueryRequest, opts ...grpc.CallOption) (*rpc.PrivateQueryReply, error) {
	c.rpcs++
	c.sets = append(c.sets, append([]uint32(nil), in.Offsets...))
	return c.recordingClient.PrivateQuery(ctx, in, opts...)
}

func (c *offsetLog) BatchPrivateQuery(ctx context.Context, in *rpc.BatchPrivateQueryRequest, opts ...grpc.CallOption) (*rpc.BatchPrivateQueryReply, error) {
	c.rpcs++
	if c.failNext {
		c.failNext = false
		return nil, errors.New("injected failure")
	}
	set := len(in.Offsets) / int(in.Count)
	for i := 0; i < int(in.Count); i++ {
		c.sets = append(c.sets, append([]uint32(nil), in.Offsets[i*set:(i+1)*set]...))
	}
	return c.recordingClient.BatchPrivateQuery(ctx, in, opts...)
}

func seeded(seed int64) func(setSize, chunkSize uint64) []uint32 {
	rng := rand.New(rand.NewSource(seed))
	return func(setSize, chunkSize uint64) []uint32 {
		out := make([]uint32, setSize)
		for i := range out {
			out[i] = uint32(rng.Uint64() & (chunkSize - 1))
		}
		return out
	}
}

// QueryBatch sends the requests of Query in one round trip and leaves the
// same state, across hint rebuilds.
func TestQueryBatchMatchesQuery(t *testing.T) {
	const n, words = 256, 3 // one partition: chunk 32, 8 chunks; MaxQueryNum 88
	client, meta, entries := testDB(t, n, 1, n, words, 40)
	base, err := NewRemotePIRClientFromMetadata(meta, 0, client)
	if err != nil {
		t.Fatal(err)
	}
	if err := base.Preprocess(context.Background()); err != nil {
		t.Fatal(err)
	}
	seqLog, batLog := &offsetLog{recordingClient: client}, &offsetLog{recordingClient: client}
	seq, bat := base.Clone(seqLog), base.Clone(batLog)
	seq.SetDummyOffsetSource(seeded(3))
	bat.SetDummyOffsetSource(seeded(3))

	rng := rand.New(rand.NewSource(8))
	const q = 8
	for round := 0; round < 30; round++ { // 240 requests: two rebuilds
		qs := make([]pianopir.BatchQuery, q)
		for i := range qs {
			qs[i] = pianopir.BatchQuery{Idx: uint64(rng.Intn(n)), Real: i < 6}
			if round%4 == 0 {
				qs[i].Idx %= 32 // concentrate on chunk 0: same chunk, cache hits, misses
			}
		}
		rebuilds := seq.Stats().Refresh.Count
		if _, err := seq.EnsureBudget(context.Background(), q); err != nil {
			t.Fatal(err)
		}
		if seq.Stats().Refresh.Count != rebuilds {
			// a rebuild draws a fresh key: copy it to the batch side
			bat = seq.Clone(batLog)
			bat.SetDummyOffsetSource(seeded(int64(round)))
			seq.SetDummyOffsetSource(seeded(int64(round)))
		}
		before := len(seqLog.sets)
		var want []BatchAnswer
		for _, x := range qs {
			e, err := seq.Query(context.Background(), x.Idx, x.Real)
			a := BatchAnswer{Entry: e, Outcome: pianopir.QueryFetched}
			switch {
			case errors.Is(err, ErrMissed):
				a = BatchAnswer{Outcome: pianopir.QueryMissed}
			case err != nil:
				t.Fatal(err)
			case !x.Real:
				a = BatchAnswer{Outcome: pianopir.QueryMissed}
			}
			want = append(want, a)
		}
		rpcs := batLog.rpcs
		got, err := bat.QueryBatch(context.Background(), qs)
		if err != nil {
			t.Fatal(err)
		}
		if batLog.rpcs-rpcs != 1 {
			t.Fatalf("round %d: %d round trips", round, batLog.rpcs-rpcs)
		}
		if !reflect.DeepEqual(seqLog.sets[before:], batLog.sets[len(batLog.sets)-q:]) {
			t.Fatalf("round %d: offset sets differ", round)
		}
		for i, a := range got {
			if a.Outcome == pianopir.QueryCacheHit {
				a.Outcome = pianopir.QueryFetched // Query does not report cache hits
			}
			if !reflect.DeepEqual(a, want[i]) {
				t.Fatalf("round %d slot %d: batch %+v sequential %+v", round, i, a, want[i])
			}
			if a.Entry != nil && !reflect.DeepEqual(a.Entry, entries[qs[i].Idx]) {
				t.Fatalf("round %d slot %d: wrong entry", round, i)
			}
		}
		if !seq.SameHintState(bat) {
			t.Fatalf("round %d: hint state differs", round)
		}
	}
	ss, bs := seq.Stats().Queries, bat.Stats().Queries
	if ss.Requests != bs.Requests || ss.Dummy != bs.Dummy || ss.Missed != bs.Missed || ss.CacheHits != bs.CacheHits || ss.Fetched != bs.Fetched || ss.CacheHits == 0 {
		t.Fatalf("stats differ or cases not exercised: sequential %+v batch %+v", ss, bs)
	}
}

// A failed round trip leaves consumed hints without parity: the client
// rebuilds before its next batch and answers correctly again.
func TestQueryBatchFailureForcesRebuild(t *testing.T) {
	const n, words = 256, 2
	client, meta, entries := testDB(t, n, 1, n, words, 40)
	log := &offsetLog{recordingClient: client}
	pc, err := NewRemotePIRClientFromMetadata(meta, 0, log)
	if err != nil {
		t.Fatal(err)
	}
	if err := pc.Preprocess(context.Background()); err != nil {
		t.Fatal(err)
	}
	qs := []pianopir.BatchQuery{{Idx: 1, Real: true}, {Idx: 40, Real: true}, {Real: false}}
	log.failNext = true
	if _, err := pc.QueryBatch(context.Background(), qs); err == nil {
		t.Fatal("failed round trip not reported")
	}
	if pc.Stats().Refresh.Count != 0 {
		t.Fatal("rebuilt during the failed batch")
	}
	got, err := pc.QueryBatch(context.Background(), qs)
	if err != nil {
		t.Fatal(err)
	}
	if pc.Stats().Refresh.Count != 1 || pc.RequestsSincePreprocess() != uint64(len(qs)) {
		t.Fatalf("no rebuild before the next batch: %+v", pc.Stats())
	}
	for i, q := range qs[:2] {
		if got[i].Outcome != pianopir.QueryFetched || !reflect.DeepEqual(got[i].Entry, entries[q.Idx]) {
			t.Fatalf("slot %d after rebuild: %+v", i, got[i])
		}
	}
	if _, err := pc.QueryBatch(context.Background(), nil); err == nil {
		t.Fatal("empty batch accepted")
	}
}
