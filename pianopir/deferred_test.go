package pianopir

import (
	"math/rand"
	"reflect"
	"testing"
)

// recordingTransport answers from the server and records every offset set.
type recordingTransport struct {
	server  *PianoPIRServer
	offsets [][]uint32
}

func (t *recordingTransport) PrivateQuery(offsets []uint32) ([]uint64, error) {
	t.offsets = append(t.offsets, append([]uint32(nil), offsets...))
	return t.server.PrivateQuery(offsets)
}

// seededDummies returns a reproducible dummy offset source.
func seededDummies(seed int64) func(setSize, chunkSize uint64) []uint32 {
	rng := rand.New(rand.NewSource(seed))
	return func(setSize, chunkSize uint64) []uint32 {
		out := make([]uint32, setSize)
		for i := range out {
			out[i] = uint32(rng.Uint64() & (chunkSize - 1))
		}
		return out
	}
}

// slotRun is what one way of sending a batch produced.
type slotRun struct {
	offsets  [][]uint32
	entries  [][]uint64
	outcomes []QueryOutcome
	reuses   int
}

// runSequential sends the batch one QueryOnce / DummyQuery at a time.
func runSequential(t *testing.T, c *PianoPIRClient, srv *PianoPIRServer, qs []BatchQuery) slotRun {
	t.Helper()
	tr := &recordingTransport{server: srv}
	var r slotRun
	for _, q := range qs {
		if !q.Real {
			if err := c.DummyQuery(tr); err != nil {
				t.Fatal(err)
			}
			r.entries = append(r.entries, nil)
			r.outcomes = append(r.outcomes, QueryMissed)
			continue
		}
		e, o, err := c.QueryOnce(q.Idx, tr)
		if err != nil {
			t.Fatal(err)
		}
		r.entries = append(r.entries, e)
		r.outcomes = append(r.outcomes, o)
	}
	r.offsets = tr.offsets
	return r
}

// runDeferred prepares the batch, answers every offset set and finishes it.
func runDeferred(t *testing.T, c *PianoPIRClient, srv *PianoPIRServer, qs []BatchQuery) slotRun {
	t.Helper()
	b, err := c.PrepareDeferredBatch(qs)
	if err != nil {
		t.Fatal(err)
	}
	resp := make([][]uint64, len(b.Offsets))
	for i, o := range b.Offsets {
		if resp[i], err = srv.PrivateQuery(o); err != nil {
			t.Fatal(err)
		}
	}
	entries, err := b.Finish(resp)
	if err != nil {
		t.Fatal(err)
	}
	return slotRun{offsets: b.Offsets, entries: entries, outcomes: b.Outcomes, reuses: b.PendingReuses}
}

// sameState compares two clients, ignoring the dummy offset source.
func sameState(a, b *PianoPIRClient) bool {
	fa, fb := a.dummyOffsets, b.dummyOffsets
	a.dummyOffsets, b.dummyOffsets = nil, nil
	defer func() { a.dummyOffsets, b.dummyOffsets = fa, fb }()
	return reflect.DeepEqual(a, b)
}

// twins returns two copies of c with the same dummy offset sequence.
func twins(c *PianoPIRClient, seed int64) (*PianoPIRClient, *PianoPIRClient) {
	seq, bat := c.Clone(), c.Clone()
	seq.SetDummyOffsetSource(seededDummies(seed))
	bat.SetDummyOffsetSource(seededDummies(seed))
	return seq, bat
}

func compareRuns(t *testing.T, db []uint64, words uint64, qs []BatchQuery, seq, bat slotRun) {
	t.Helper()
	if !reflect.DeepEqual(seq.offsets, bat.offsets) {
		t.Fatalf("offset sets differ:\nsequential %v\ndeferred   %v", seq.offsets, bat.offsets)
	}
	if !reflect.DeepEqual(seq.outcomes, bat.outcomes) {
		t.Fatalf("outcomes differ: sequential %v deferred %v", seq.outcomes, bat.outcomes)
	}
	for i, q := range qs {
		if !q.Real || seq.outcomes[i] == QueryMissed {
			if bat.entries[i] != nil {
				t.Fatalf("slot %d: deferred returned an entry for a dummy/missed slot", i)
			}
			continue
		}
		want := db[q.Idx*words : (q.Idx+1)*words]
		if !reflect.DeepEqual(seq.entries[i], want) || !reflect.DeepEqual(bat.entries[i], want) {
			t.Fatalf("slot %d idx %d: sequential %v deferred %v want %v", i, q.Idx, seq.entries[i], bat.entries[i], want)
		}
	}
}

// findReuse returns x, y such that in a batch [x, y] the second slot uses
// the hint the first slot replaced.
func findReuse(t *testing.T, c *PianoPIRClient) (uint64, uint64) {
	t.Helper()
	for x := uint64(0); x < c.config.DBSize; x++ {
		for y := uint64(0); y < c.config.DBSize; y++ {
			if y/c.config.ChunkSize == x/c.config.ChunkSize {
				continue
			}
			b, err := c.Clone().PrepareDeferredBatch([]BatchQuery{{x, true}, {y, true}})
			if err != nil {
				t.Fatal(err)
			}
			if b.PendingReuses == 1 {
				return x, y
			}
		}
	}
	t.Fatal("no batch of two reuses a replaced hint")
	return 0, 0
}

func TestDeferredBatchMatchesSequential(t *testing.T) {
	// 200 entries: chunk 32, 8 chunks (one partial, one padding only);
	// 3-word entries exercise the EntryXor tail.
	const dbSize, words = 200, 3
	db := randomDB(dbSize, words)
	c, srv := streamClient(t, db, dbSize, words)
	chunk := c.config.ChunkSize

	batches := map[string][]BatchQuery{
		"same chunk":       {{1, true}, {2, true}, {3, true}, {4, true}},
		"in-batch reuse":   nil, // found against the state before the batch
		"repeat in batch":  {{7, true}, {chunk + 7, true}, {7, true}, {0, false}},
		"dummies and real": {{0, false}, {2*chunk + 1, true}, {0, false}, {3*chunk + 9, true}},
		"out of chunk":     {{dbSize - 1, true}, {0, false}},
	}
	order := []string{"same chunk", "in-batch reuse", "repeat in batch", "dummies and real", "out of chunk"}
	seq, bat := twins(c, 9)
	reuses := 0
	for _, name := range order {
		qs := batches[name]
		if qs == nil {
			x, y := findReuse(t, bat)
			qs = []BatchQuery{{x, true}, {y, true}}
		}
		s, b := runSequential(t, seq, srv, qs), runDeferred(t, bat, srv, qs)
		compareRuns(t, db, words, qs, s, b)
		if !sameState(seq, bat) {
			t.Fatalf("%s: client state differs after the batch", name)
		}
		reuses += b.reuses
	}
	if reuses == 0 {
		t.Fatal("no slot reused a hint replaced in the same batch")
	}
	// the repeated batch is all cache hits now
	qs := batches["same chunk"]
	s, b := runSequential(t, seq, srv, qs), runDeferred(t, bat, srv, qs)
	compareRuns(t, db, words, qs, s, b)
	for _, o := range b.outcomes {
		if o != QueryCacheHit {
			t.Fatalf("outcomes %v, want cache hits", b.outcomes)
		}
	}
}

// Random batches until the budget and some chunk backups are used up, so
// misses happen (in a tiny DB, where in-batch hint reuse is frequent).
func TestDeferredBatchRandomMatchesSequential(t *testing.T) {
	const dbSize, words = 64, 2 // chunk 16, 4 chunks
	db := randomDB(dbSize, words)
	c, srv := streamClient(t, db, dbSize, words)
	rng := rand.New(rand.NewSource(4))
	seq, bat := twins(c, 11)
	var reuses, missed int
	for round := 0; round < 20; round++ {
		qs := make([]BatchQuery, 6)
		for i := range qs {
			qs[i] = BatchQuery{Idx: uint64(rng.Intn(dbSize)), Real: rng.Intn(5) > 0}
			if round%3 == 0 { // concentrate on chunk 0 to exhaust its backups
				qs[i].Idx %= c.config.ChunkSize
			}
		}
		s, b := runSequential(t, seq, srv, qs), runDeferred(t, bat, srv, qs)
		compareRuns(t, db, words, qs, s, b)
		if !sameState(seq, bat) {
			t.Fatalf("round %d: client state differs", round)
		}
		reuses += b.reuses
		for i, o := range b.outcomes {
			if qs[i].Real && o == QueryMissed {
				missed++
			}
		}
	}
	if missed == 0 {
		t.Fatal("random batches did not miss")
	}
	t.Logf("in-batch hint reuses=%d missed=%d", reuses, missed)
}

func TestDeferredBatchRejects(t *testing.T) {
	const dbSize, words = 64, 2
	c, _ := streamClient(t, randomDB(dbSize, words), dbSize, words)
	before := c.Clone()
	if _, err := c.PrepareDeferredBatch([]BatchQuery{{1, true}, {dbSize, true}}); err == nil {
		t.Fatal("out-of-range index accepted")
	}
	if !sameState(c, before) {
		t.Fatal("rejected batch changed the client")
	}
	b, err := c.PrepareDeferredBatch([]BatchQuery{{1, true}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b.Finish(nil); err == nil {
		t.Fatal("missing responses accepted")
	}
	if _, err := b.Finish([][]uint64{make([]uint64, words)}); err == nil {
		t.Fatal("second Finish accepted")
	}
	b, _ = c.Clone().PrepareDeferredBatch([]BatchQuery{{2, true}})
	if _, err := b.Finish([][]uint64{make([]uint64, words+1)}); err == nil {
		t.Fatal("wrong response size accepted")
	}
}

func TestCloneIsIndependent(t *testing.T) {
	const dbSize, words = 64, 2
	db := randomDB(dbSize, words)
	c, srv := streamClient(t, db, dbSize, words)
	d := c.Clone()
	if !sameState(c, d) {
		t.Fatal("clone differs")
	}
	if _, _, err := d.QueryOnce(3, &recordingTransport{server: srv}); err != nil {
		t.Fatal(err)
	}
	if sameState(c, d) || c.FinishedQueryNum != 0 || len(c.localCache) != 0 {
		t.Fatal("query on the clone changed the original")
	}
}
