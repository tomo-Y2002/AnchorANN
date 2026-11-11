package pianopir

import (
	"math/rand"
	"reflect"
	"strings"
	"testing"
)

type countingTransport struct {
	server *PianoPIRServer
	calls  int
}

func (t *countingTransport) PrivateQuery(offsets []uint32) ([]uint64, error) {
	t.calls++
	return t.server.PrivateQuery(offsets)
}

// streamClient builds a client for dbSize entries of entryWords words and
// feeds it rawDB through a StreamPreprocessor.
func streamClient(t *testing.T, rawDB []uint64, dbSize, entryWords uint64) (*PianoPIRClient, *PianoPIRServer) {
	t.Helper()
	chunkSize, setSize := GenParams(uint32(dbSize))
	cfg := &PianoPIRConfig{
		DBEntryByteNum: entryWords * 8, DBEntrySize: entryWords, DBSize: dbSize,
		ChunkSize: uint64(chunkSize), SetSize: uint64(setSize), ThreadNum: 1, FailureProbLog2: 40,
	}
	c := NewPianoPIRClient(cfg)
	p := c.NewStreamPreprocessor()
	for b := uint64(0); b < p.BlockCount(); b++ {
		first := b * p.BlockEntries()
		n := StreamBlockLen(dbSize, p.BlockEntries(), b)
		if err := p.AddBlock(b, first, rawDB[first*entryWords:(first+n)*entryWords]); err != nil {
			t.Fatalf("AddBlock(%d): %v", b, err)
		}
	}
	if err := p.Finish(); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	return c, NewPianoPIRServer(cfg, rawDB)
}

func randomDB(dbSize, entryWords uint64) []uint64 {
	rng := rand.New(rand.NewSource(1))
	db := make([]uint64, dbSize*entryWords)
	for i := range db {
		db[i] = rng.Uint64() | 1
	}
	return db
}

func TestStreamBlockEntries(t *testing.T) {
	cases := []struct{ chunk, entryBytes, want uint64 }{
		{128, 8, 128},        // whole chunk fits
		{4096, 1024, 1024},   // 1 MiB / 1 KiB
		{4096, 3000, 256},    // largest power of two within 1 MiB
		{64, 2 << 20, 1},     // one entry larger than the target
		{1, 8, 1},            // chunk of one entry
		{1024, 640 * 8, 128}, // 5 KiB entries
		{2048, 1 << 20, 1},   // exactly the target
		{2048, (1 << 20) / 2, 2},
	}
	for _, c := range cases {
		if got := StreamBlockEntries(c.chunk, c.entryBytes); got != c.want {
			t.Errorf("StreamBlockEntries(%d, %d) = %d, want %d", c.chunk, c.entryBytes, got, c.want)
		}
	}
	if StreamBlockCount(1000, 128) != 8 || StreamBlockLen(1000, 128, 7) != 104 || StreamBlockLen(1000, 128, 8) != 0 {
		t.Fatalf("block count/len mismatch")
	}
}

func TestStreamPreprocessingAnswersQueries(t *testing.T) {
	// 1000 entries: chunk 64, 16 chunks; the last chunk is partial and two
	// chunks are padding only. 5-word entries exercise the EntryXor tail.
	const dbSize, words = 1000, 5
	db := randomDB(dbSize, words)
	c, srv := streamClient(t, db, dbSize, words)
	tr := &countingTransport{server: srv}
	fetched := 0
	for idx := uint64(0); idx < dbSize; idx += 7 {
		got, outcome, err := c.QueryOnce(idx, tr)
		if err != nil {
			t.Fatalf("QueryOnce(%d): %v", idx, err)
		}
		if outcome == QueryFetched {
			fetched++
			if !reflect.DeepEqual(got, db[idx*words:(idx+1)*words]) {
				t.Fatalf("QueryOnce(%d) = %v, want %v", idx, got, db[idx*words:(idx+1)*words])
			}
		}
	}
	if fetched < 140 || tr.calls != 143 {
		t.Fatalf("fetched %d of 143 with %d requests", fetched, tr.calls)
	}
}

func TestQueryOnceSendsOneRequestPerOutcome(t *testing.T) {
	// 4096 entries: chunk 128, 48 backups per chunk
	const dbSize = 4096
	db := randomDB(dbSize, 1)
	c, srv := streamClient(t, db, dbSize, 1)
	tr := &countingTransport{server: srv}
	outcomes := map[QueryOutcome]int{}
	for idx := uint64(0); idx < 50; idx++ {
		_, o, err := c.QueryOnce(idx, tr)
		if err != nil {
			t.Fatalf("QueryOnce(%d): %v", idx, err)
		}
		outcomes[o]++
	}
	got, o, err := c.QueryOnce(3, tr)
	if err != nil || o != QueryCacheHit || got[0] != db[3] {
		t.Fatalf("cached QueryOnce = %v %v %v", got, o, err)
	}
	outcomes[o]++
	if want := map[QueryOutcome]int{QueryFetched: 48, QueryMissed: 2, QueryCacheHit: 1}; !reflect.DeepEqual(outcomes, want) {
		t.Fatalf("outcomes = %v, want %v", outcomes, want)
	}
	if tr.calls != 51 {
		t.Fatalf("%d requests for 51 queries", tr.calls)
	}
	if size := c.LocalCacheStorageBytes(); size != 48*(8+8) {
		t.Fatalf("cache bytes=%d, want keys and payloads for 48 distinct entries", size)
	}
	if err := c.DummyQuery(tr); err != nil || tr.calls != 52 {
		t.Fatalf("DummyQuery: %v (%d calls)", err, tr.calls)
	}
}

func TestStreamPreprocessorRejectsBadStreams(t *testing.T) {
	cfg := &PianoPIRConfig{DBEntryByteNum: 8, DBEntrySize: 1, DBSize: 100, ChunkSize: 32, SetSize: 4, ThreadNum: 1, FailureProbLog2: 10}
	c := NewPianoPIRClient(cfg)
	p := c.NewStreamPreprocessor()
	if p.BlockEntries() != 32 || p.BlockCount() != 4 {
		t.Fatalf("layout %d x %d", p.BlockEntries(), p.BlockCount())
	}
	if err := p.AddBlock(1, 32, make([]uint64, 32)); err == nil || !strings.Contains(err.Error(), "out of order") {
		t.Fatalf("out-of-order block: %v", err)
	}
	if err := p.AddBlock(0, 1, make([]uint64, 32)); err == nil {
		t.Fatalf("wrong first entry accepted")
	}
	if err := p.AddBlock(0, 0, make([]uint64, 31)); err == nil {
		t.Fatalf("short block accepted")
	}
	if err := p.AddBlock(0, 0, make([]uint64, 32)); err != nil {
		t.Fatalf("AddBlock: %v", err)
	}
	if err := p.Finish(); err == nil || !strings.Contains(err.Error(), "after 1 of 4") {
		t.Fatalf("truncated stream: %v", err)
	}
}

func TestEntryXorTail(t *testing.T) {
	a := []uint64{1, 2, 3, 4, 5, 6, 7}
	b := []uint64{1, 1, 1, 1, 1, 1, 1, 99}
	EntryXor(a, b, 7)
	if want := []uint64{0, 3, 2, 5, 4, 7, 6}; !reflect.DeepEqual(a, want) {
		t.Fatalf("EntryXor = %v, want %v", a, want)
	}
}
