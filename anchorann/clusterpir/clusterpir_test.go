package clusterpir

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"net"
	"reflect"
	"strings"
	"sync"
	"testing"

	"example.com/private-search/rpc"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"

	"privatesearch/anchorann"
)

// fixture returns a valid artifact with C clusters of s points whose members
// are a seeded permutation of 0..N-1, and a matching random dataset.
func fixture(t testing.TB, count, size, dim int, seed int64) (*anchorann.Artifact, anchorann.Dataset) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	n := count * size
	perm := rng.Perm(n)
	a := &anchorann.Artifact{Version: anchorann.FormatVersion, N: n, Size: size, Count: count, Dim: dim, Seed: seed, Method: anchorann.Common, Iterations: 1}
	for c := 0; c < count; c++ {
		ids := make([]uint32, size)
		for j := range ids {
			ids[j] = uint32(perm[c*size+j])
		}
		a.Members = append(a.Members, ids)
		a.Anchors = append(a.Anchors, make([]float64, dim))
	}
	d := anchorann.Dataset{N: n, Dim: dim, Values: make([]float32, n*dim)}
	for i := range d.Values {
		d.Values[i] = float32(rng.Intn(256)) + 0.25
	}
	return a, d
}

func buildDB(t testing.TB, count, size, dim int) (*DB, *anchorann.Artifact, anchorann.Dataset) {
	t.Helper()
	a, d := fixture(t, count, size, dim, 7)
	db, err := Build(a, d)
	if err != nil {
		t.Fatal(err)
	}
	return db, a, d
}

func newFetcher(t testing.TB, db *DB, partitions int, b BatchConfig) (*Fetcher, rpc.PirServiceClient) {
	t.Helper()
	srv, err := db.NewServer(partitions, 40)
	if err != nil {
		t.Fatal(err)
	}
	client := InProcess(srv)
	f, err := NewFetcher(context.Background(), client, b)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Preprocess(context.Background()); err != nil {
		t.Fatal(err)
	}
	return f, client
}

func TestLayoutSIFT1M(t *testing.T) {
	l, err := NewLayout(31250, 32, 128)
	if err != nil {
		t.Fatal(err)
	}
	if l.PayloadBytes() != 16512 || l.EntryBytes() != 16512 || l.EntryWords() != 2064 || l.DBBytes() != 516_000_000 || l.N() != 1_000_000 {
		t.Fatalf("payload=%d entry=%d words=%d db=%d N=%d", l.PayloadBytes(), l.EntryBytes(), l.EntryWords(), l.DBBytes(), l.N())
	}
}

func TestLayoutPadding(t *testing.T) {
	for _, tc := range []struct{ s, d, payload, entry int }{{1, 1, 8, 8}, {1, 2, 12, 16}, {3, 3, 48, 48}, {3, 2, 36, 40}} {
		l, err := NewLayout(5, tc.s, tc.d)
		if err != nil {
			t.Fatal(err)
		}
		if l.PayloadBytes() != tc.payload || l.EntryBytes() != tc.entry || l.EntryBytes()%WordBytes != 0 {
			t.Fatalf("s=%d d=%d payload=%d entry=%d", tc.s, tc.d, l.PayloadBytes(), l.EntryBytes())
		}
	}
	for _, bad := range [][3]int{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {-1, 2, 2}} {
		if _, err := NewLayout(bad[0], bad[1], bad[2]); err == nil {
			t.Fatalf("NewLayout%v accepted", bad)
		}
	}
}

func TestEntryRoundTrip(t *testing.T) {
	l, _ := NewLayout(4, 3, 2) // N=12, payload 36 -> 40 bytes with padding
	ids := []uint32{11, 0, 5}
	vec := []float32{1.5, -2, 0, 3.25, 255, 1e-3}
	words := make([]uint64, l.EntryWords())
	if err := EncodeEntry(l, ids, vec, words); err != nil {
		t.Fatal(err)
	}
	c, err := DecodeEntry(l, words)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(c.IDs, ids) || !reflect.DeepEqual(c.Vectors, vec) {
		t.Fatalf("round trip: %v %v", c.IDs, c.Vectors)
	}
	again := make([]uint64, l.EntryWords())
	if err := EncodeEntry(l, c.IDs, c.Vectors, again); err != nil || !reflect.DeepEqual(again, words) {
		t.Fatalf("re-encode differs: %v %v", again, err)
	}
}

func TestDecodeEntryRejects(t *testing.T) {
	l, _ := NewLayout(4, 3, 2)
	good := make([]uint64, l.EntryWords())
	if err := EncodeEntry(l, []uint32{1, 2, 3}, []float32{1, 2, 3, 4, 5, 6}, good); err != nil {
		t.Fatal(err)
	}
	mutate := func(fn func(b []uint64)) []uint64 {
		w := append([]uint64(nil), good...)
		fn(w)
		return w
	}
	cases := map[string]struct {
		words []uint64
		want  string
	}{
		"short":         {good[:len(good)-1], "length"},
		"long":          {append(append([]uint64(nil), good...), 0), "length"},
		"id range":      {mutate(func(w []uint64) { w[0] = uint64(12) | uint64(2)<<32 }), "outside"},
		"duplicate id":  {mutate(func(w []uint64) { w[0] = uint64(2) | uint64(2)<<32 }), "duplicate"},
		"all zero":      {make([]uint64, l.EntryWords()), "duplicate"},
		"nan":           {mutate(func(w []uint64) { w[2] = uint64(math.Float32bits(float32(math.NaN()))) << 32 }), "nonfinite"},
		"padding":       {mutate(func(w []uint64) { w[len(w)-1] |= 1 << 63 }), "padding"},
		"encode length": {nil, ""},
	}
	for name, tc := range cases {
		if name == "encode length" {
			if err := EncodeEntry(l, []uint32{1, 2}, make([]float32, 6), good); err == nil {
				t.Fatal("EncodeEntry accepted s-1 IDs")
			}
			if err := EncodeEntry(l, []uint32{1, 2, 3}, make([]float32, 6), good[:1]); err == nil {
				t.Fatal("EncodeEntry accepted short dst")
			}
			continue
		}
		_, err := DecodeEntry(l, tc.words)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("%s: err=%v want %q", name, err, tc.want)
		}
	}
}

func TestBuildStoresEveryClusterOnce(t *testing.T) {
	db, a, d := buildDB(t, 9, 4, 3)
	if len(db.Words) != 9*db.Layout.EntryWords() {
		t.Fatalf("DB words=%d", len(db.Words))
	}
	seen := make([]bool, a.N)
	for c := 0; c < a.Count; c++ {
		cl, err := DecodeEntry(db.Layout, db.Entry(c))
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(cl.IDs, a.Members[c]) {
			t.Fatalf("cluster %d IDs %v want %v", c, cl.IDs, a.Members[c])
		}
		for j, id := range cl.IDs {
			if seen[id] {
				t.Fatalf("ID %d stored twice", id)
			}
			seen[id] = true
			if !reflect.DeepEqual(cl.Vectors[j*3:(j+1)*3], d.Values[int(id)*3:(int(id)+1)*3]) {
				t.Fatalf("cluster %d point %d vector mismatch", c, j)
			}
		}
	}
	for id, ok := range seen {
		if !ok {
			t.Fatalf("ID %d missing", id)
		}
	}
}

func TestBuildRejectsContractViolations(t *testing.T) {
	a, d := fixture(t, 4, 3, 2, 1)
	uneven := *a
	uneven.Members = append([][]uint32{a.Members[0][:2]}, a.Members[1:]...)
	dup := *a
	dup.Members = append([][]uint32(nil), a.Members...)
	dup.Members[1] = []uint32{a.Members[0][0], a.Members[1][1], a.Members[1][2]}
	cases := map[string]struct {
		a *anchorann.Artifact
		d anchorann.Dataset
	}{
		"uneven cluster":  {&uneven, d},
		"duplicate ID":    {&dup, d},
		"nil artifact":    {nil, d},
		"dataset N":       {a, anchorann.Dataset{N: 11, Dim: 2, Values: d.Values[:22]}},
		"dataset dim":     {a, anchorann.Dataset{N: 12, Dim: 1, Values: d.Values[:12]}},
		"short values":    {a, anchorann.Dataset{N: 12, Dim: 2, Values: d.Values[:20]}},
		"nonfinite value": {a, anchorann.Dataset{N: 12, Dim: 2, Values: append(append([]float32(nil), d.Values[:23]...), float32(math.Inf(1)))}},
	}
	for name, tc := range cases {
		if _, err := Build(tc.a, tc.d); err == nil {
			t.Fatalf("%s: Build accepted", name)
		}
	}
}

func TestPartitionSize(t *testing.T) {
	for _, tc := range []struct{ c, p, size int }{{10, 1, 10}, {10, 4, 3}, {31250, 8, 3907}, {1, 1, 1}} {
		got, err := PartitionSize(tc.c, tc.p)
		if err != nil || got != tc.size {
			t.Fatalf("PartitionSize(%d,%d)=%d,%v want %d", tc.c, tc.p, got, err, tc.size)
		}
	}
	for _, bad := range [][2]int{{10, 0}, {10, 11}, {10, 6}} {
		if _, err := PartitionSize(bad[0], bad[1]); err == nil {
			t.Fatalf("PartitionSize%v accepted", bad)
		}
	}
}

// checkFetched compares every retrieved cluster with the plaintext entry.
func checkFetched(t *testing.T, db *DB, ids []int, res *Result) {
	t.Helper()
	missed := map[int]bool{}
	for _, c := range res.Missed {
		missed[c] = true
	}
	for i, c := range ids {
		if missed[c] {
			if res.Clusters[i] != nil {
				t.Fatalf("missed cluster %d has data", c)
			}
			continue
		}
		got := res.Clusters[i]
		if got == nil || got.ID != c {
			t.Fatalf("cluster %d not returned: %+v", c, got)
		}
		words := make([]uint64, db.Layout.EntryWords())
		if err := EncodeEntry(db.Layout, got.IDs, got.Vectors, words); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(words, db.Entry(c)) {
			t.Fatalf("cluster %d: PIR entry differs from plaintext", c)
		}
	}
}

func TestFetchSingleCluster(t *testing.T) {
	db, _, _ := buildDB(t, 1, 2, 3)
	f, _ := newFetcher(t, db, 1, BatchConfig{NProbe: 1})
	if f.MaxQueryNum() != 1 {
		t.Fatalf("MaxQueryNum=%d, want 1 for C=1", f.MaxQueryNum())
	}
	for q := 0; q < 5; q++ {
		res, err := f.Fetch(context.Background(), []int{0})
		if err != nil {
			t.Fatal(err)
		}
		checkFetched(t, db, []int{0}, res)
		// One request per construction: every later query rebuilds the hints
		// first, which also empties the cache.
		want := FetchStats{Requests: 1, Real: 1, Rounds: 1}
		if q > 0 {
			want.Refreshes = 1
		}
		st := res.Stats
		if st.Requests != want.Requests || st.CacheHits != 0 || st.Real != 1 || len(res.Missed) != 0 || st.Refreshes != want.Refreshes {
			t.Fatalf("query %d stats %+v missed %v", q, res.Stats, res.Missed)
		}
	}
}

func TestFetchNProbeOneAndMax(t *testing.T) {
	const count = 20 // MaxQueryNum = floor(sqrt(20)·ln 20) = 13
	db, _, _ := buildDB(t, count, 4, 3)
	rng := rand.New(rand.NewSource(3))
	for _, nProbe := range []int{1, 13} {
		f, _ := newFetcher(t, db, 1, BatchConfig{NProbe: nProbe})
		for q := 0; q < 12; q++ {
			ids := rng.Perm(count)[:nProbe]
			res, err := f.Fetch(context.Background(), ids)
			if err != nil {
				t.Fatalf("n_probe=%d query %d: %v", nProbe, q, err)
			}
			checkFetched(t, db, ids, res)
			if len(res.Missed) != 0 || res.Stats.Requests != nProbe || res.Stats.Real != nProbe {
				t.Fatalf("n_probe=%d stats %+v missed %v", nProbe, res.Stats, res.Missed)
			}
		}
	}
}

func TestFetchRejectsInvalidIDs(t *testing.T) {
	db, _, _ := buildDB(t, 6, 2, 2)
	f, _ := newFetcher(t, db, 1, BatchConfig{NProbe: 2})
	for name, ids := range map[string][]int{
		"duplicate": {3, 3}, "negative": {-1, 2}, "too large": {0, 6}, "too few": {1}, "too many": {1, 2, 3},
	} {
		if _, err := f.Fetch(context.Background(), ids); err == nil {
			t.Fatalf("%s: Fetch(%v) accepted", name, ids)
		}
	}
	if _, err := NewFetcher(context.Background(), InProcess(mustServer(t, db, 1)), BatchConfig{NProbe: 7}); err == nil {
		t.Fatal("n_probe > C accepted")
	}
	if _, err := NewFetcher(context.Background(), InProcess(mustServer(t, db, 2)), BatchConfig{NProbe: 4, PerPartition: 1, Rounds: 1}); err == nil {
		t.Fatal("batch with fewer slots than n_probe accepted")
	}
	// one Fetch must fit in one hint construction (MaxQueryNum = 4 for C=6)
	if _, err := NewFetcher(context.Background(), InProcess(mustServer(t, db, 1)), BatchConfig{NProbe: 5}); err == nil || !strings.Contains(err.Error(), "MaxQueryNum") {
		t.Fatalf("batch beyond MaxQueryNum: err=%v", err)
	}
	unprepared, err := NewFetcher(context.Background(), InProcess(mustServer(t, db, 1)), BatchConfig{NProbe: 1})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := unprepared.Fetch(context.Background(), []int{0}); err == nil {
		t.Fatal("Fetch before Preprocess accepted")
	}
}

func mustServer(t *testing.T, db *DB, p int) rpc.PirServiceServer {
	t.Helper()
	srv, err := db.NewServer(p, 40)
	if err != nil {
		t.Fatal(err)
	}
	return srv
}

// entryBytesServer reports a wrong entry length in its metadata.
type entryBytesServer struct{ rpc.PirServiceServer }

func (s entryBytesServer) GetMetadata(ctx context.Context, in *rpc.MetadataRequest) (*rpc.MetadataReply, error) {
	m, err := s.PirServiceServer.GetMetadata(ctx, in)
	if err == nil {
		m.DbEntryByteNum += 8
	}
	return m, err
}

func TestFetcherRejectsEntryLengthMismatch(t *testing.T) {
	db, _, _ := buildDB(t, 6, 2, 2)
	_, err := NewFetcher(context.Background(), InProcess(entryBytesServer{mustServer(t, db, 1)}), BatchConfig{NProbe: 1})
	if err == nil || !strings.Contains(err.Error(), "entry bytes") {
		t.Fatalf("err=%v", err)
	}
}

func TestFetchCollisionMissesAreReported(t *testing.T) {
	db, _, _ := buildDB(t, 64, 2, 2) // P=2: partition 0 = clusters 0..31, partition 1 = 32..63
	ids := []int{0, 1}               // both in partition 0

	f, _ := newFetcher(t, db, 2, BatchConfig{NProbe: 2, PerPartition: 1, Rounds: 1})
	res, err := f.Fetch(context.Background(), ids)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(res.Missed, []int{1}) || res.Clusters[1] != nil || res.Stats.Requests != 2 || res.Stats.Real != 1 || res.Stats.Dummy != 1 {
		t.Fatalf("missed=%v stats=%+v", res.Missed, res.Stats)
	}
	checkFetched(t, db, ids, res)

	// A second round re-fetches the collided cluster with the same fixed count.
	f, _ = newFetcher(t, db, 2, BatchConfig{NProbe: 2, PerPartition: 1, Rounds: 2})
	for _, q := range [][]int{{0, 1}, {0, 40}, {32, 63}} {
		res, err := f.Fetch(context.Background(), q)
		if err != nil {
			t.Fatal(err)
		}
		if len(res.Missed) != 0 || res.Stats.Requests != 4 || res.Stats.Real+res.Stats.Dummy != 4 {
			t.Fatalf("ids=%v missed=%v stats=%+v", q, res.Missed, res.Stats)
		}
		checkFetched(t, db, q, res)
	}
}

// countingServer counts the PrivateQuery and StartPreprocess calls the
// server observes and records, per partition, their order and sizes.
type countingServer struct {
	rpc.PirServiceServer
	mu             sync.Mutex
	queries, preps int
	offsets        map[int]bool
	trace          map[uint64][]string
}

func (s *countingServer) PrivateQuery(ctx context.Context, in *rpc.PrivateQueryRequest) (*rpc.PrivateQueryReply, error) {
	s.mu.Lock()
	s.queries++
	s.offsets[len(in.Offsets)] = true
	s.record(in.PartitionId, fmt.Sprintf("query/%d", len(in.Offsets)))
	s.mu.Unlock()
	return s.PirServiceServer.PrivateQuery(ctx, in)
}

func (s *countingServer) StartPreprocess(in *rpc.PreprocessRequest, st rpc.PirService_StartPreprocessServer) error {
	s.mu.Lock()
	s.preps++
	s.record(uint64(in.PartitionId), fmt.Sprintf("preprocess/%d", proto.Size(in)))
	s.mu.Unlock()
	return s.PirServiceServer.StartPreprocess(in, st)
}

func (s *countingServer) record(p uint64, e string) {
	if s.trace == nil {
		s.trace = map[uint64][]string{}
	}
	s.trace[p] = append(s.trace[p], e)
}

// countingFixture serves db through a countingServer with one partition and
// n_probe=1, and returns the fetcher and the server metadata.
func countingFixture(t *testing.T, db *DB) (*Fetcher, *countingServer, *rpc.MetadataReply) {
	t.Helper()
	srv := &countingServer{PirServiceServer: mustServer(t, db, 1), offsets: map[int]bool{}}
	f, err := NewFetcher(context.Background(), InProcess(srv), BatchConfig{NProbe: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Preprocess(context.Background()); err != nil {
		t.Fatal(err)
	}
	meta, err := srv.GetMetadata(context.Background(), &rpc.MetadataRequest{})
	if err != nil {
		t.Fatal(err)
	}
	srv.queries, srv.preps = 0, 0
	return f, srv, meta
}

// fetchOne fetches cluster c and checks the server saw rpcs PrivateQuery and
// refreshes StartPreprocess calls.
func fetchOne(t *testing.T, db *DB, f *Fetcher, srv *countingServer, c, rpcs, refreshes int) *Result {
	t.Helper()
	beforeQ, beforeP := srv.queries, srv.preps
	res, err := f.Fetch(context.Background(), []int{c})
	if err != nil {
		t.Fatal(err)
	}
	checkFetched(t, db, []int{c}, res)
	if srv.queries-beforeQ != rpcs || res.Stats.Requests != rpcs || srv.preps-beforeP != refreshes || res.Stats.Refreshes != refreshes || len(res.Missed) != 0 {
		t.Fatalf("cluster %d: server saw %d requests / %d preprocesses, want %d / %d; stats %+v", c, srv.queries-beforeQ, srv.preps-beforeP, rpcs, refreshes, res.Stats)
	}
	return res
}

func checkRefreshSeparated(t *testing.T, res *Result, entryBytes int) {
	t.Helper()
	st := res.Stats
	if st.Refresh.Preprocesses != 1 || st.Refresh.PreprocDownBytes <= 0 || st.Refresh.PreprocUpBytes <= 0 || st.Refresh.Queries != 0 || st.RefreshSeconds <= 0 {
		t.Fatalf("refresh cost not recorded: %+v", st)
	}
	if st.Online.Preprocesses != 0 || st.Online.PreprocDownBytes != 0 || st.Online.QueryDownBytes < int64(entryBytes) || st.Seconds < 0 {
		t.Fatalf("refresh cost mixed into the online cost: %+v", st)
	}
}

// Refreshes follow the public schedule: after MaxQueryNum requests of any
// kind (cache hits send a dummy and count too), the next Fetch rebuilds the
// hints before its request.
func TestRefreshAfterMaxQueryNum(t *testing.T) {
	const count = 1024
	db, _, _ := buildDB(t, count, 2, 2)
	f, srv, meta := countingFixture(t, db)
	chunk, set := int(meta.ChunkSize), int(meta.SetSize)
	maxQ := f.MaxQueryNum()
	if maxQ+1 > count || f.FetchesPerRefresh() != maxQ {
		t.Fatalf("fixture: MaxQueryNum=%d fetches/refresh=%d", maxQ, f.FetchesPerRefresh())
	}
	// Spread the requests over the chunks round-robin so no chunk runs out of
	// backups; every fourth fetch repeats a cluster and hits the cache.
	id := func(k int) int { return (k%set)*chunk + k/set }
	for k := 0; k < maxQ; k++ {
		c := id(k)
		if k%4 == 3 {
			c = id(k - 1)
		}
		fetchOne(t, db, f, srv, c, 1, 0)
	}
	res := fetchOne(t, db, f, srv, id(maxQ), 1, 1)
	checkRefreshSeparated(t, res, db.Layout.EntryBytes())
	// The refresh cleared the cache: the same cluster is fetched again.
	if res = fetchOne(t, db, f, srv, id(0), 1, 0); res.Stats.CacheHits != 0 {
		t.Fatalf("cache survived the refresh: %+v", res.Stats)
	}
}

// When the queried chunk has used all its backup hints, the request misses:
// a dummy is sent, nothing is rebuilt and the cluster is reported missed.
func TestBackupExhaustionIsMissedNotRefreshed(t *testing.T) {
	const count = 1024
	db, _, _ := buildDB(t, count, 2, 2)
	f, srv, meta := countingFixture(t, db)
	per := f.BackupPerChunk()
	if per+1 > int(meta.ChunkSize) || per >= f.MaxQueryNum() {
		t.Fatalf("fixture: chunk=%d backup/chunk=%d MaxQueryNum=%d cannot isolate backup exhaustion", meta.ChunkSize, per, f.MaxQueryNum())
	}
	for c := 0; c < per; c++ { // distinct clusters of chunk 0
		fetchOne(t, db, f, srv, c, 1, 0)
	}
	fetchOne(t, db, f, srv, int(meta.ChunkSize), 1, 0) // another chunk still has backups
	beforeQ, beforeP := srv.queries, srv.preps
	res, err := f.Fetch(context.Background(), []int{per})
	if err != nil {
		t.Fatal(err)
	}
	st := res.Stats
	if !reflect.DeepEqual(res.Missed, []int{per}) || res.Clusters[0] != nil || st.PIRMissed != 1 || st.Requests != 1 || st.Refreshes != 0 {
		t.Fatalf("missed=%v stats=%+v", res.Missed, st)
	}
	if srv.queries-beforeQ != 1 || srv.preps != beforeP {
		t.Fatalf("server saw %d requests / %d preprocesses, want 1 / 0", srv.queries-beforeQ, srv.preps-beforeP)
	}
}

// The server observes the same calls, in the same order per partition, for
// concentrated, spread, repeated (cache hits) and backup-exhausting queries.
func TestObservedTraceIndependentOfQueries(t *testing.T) {
	// partitions of 1024: chunk 64, 39 backups per chunk, 55 queries per refresh
	const count, parts, nProbe, queries = 2048, 2, 4, 60
	db, _, _ := buildDB(t, count, 2, 2)
	rng := rand.New(rand.NewSource(5))
	inputs := map[string]func(q int) []int{
		"concentrated": func(int) []int { return rng.Perm(16)[:nProbe] },
		"spread":       func(int) []int { return rng.Perm(count)[:nProbe] },
		"repeated":     func(int) []int { return []int{1, 2, 1100, 1101} },
		// distinct clusters of chunk 0 of both partitions
		"exhausting": func(q int) []int { return []int{(2 * q) % 64, (2*q + 1) % 64, 1024 + (2*q)%64, 1024 + (2*q+1)%64} },
	}
	var ref map[uint64][]string
	for _, name := range []string{"concentrated", "spread", "repeated", "exhausting"} {
		srv := &countingServer{PirServiceServer: mustServer(t, db, parts), offsets: map[int]bool{}}
		f, err := NewFetcher(context.Background(), InProcess(srv), BatchConfig{NProbe: nProbe, PerPartition: 2, Rounds: 2})
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Preprocess(context.Background()); err != nil {
			t.Fatal(err)
		}
		var total FetchStats
		for q := 0; q < queries; q++ {
			ids := inputs[name](q)
			res, err := f.Fetch(context.Background(), ids)
			if err != nil {
				t.Fatalf("%s query %d: %v", name, q, err)
			}
			checkFetched(t, db, ids, res)
			total.Requests += res.Stats.Requests
			total.CacheHits += res.Stats.CacheHits
			total.PIRMissed += res.Stats.PIRMissed
			total.Refreshes += res.Stats.Refreshes
		}
		if total.Requests != queries*parts*4 || total.Refreshes == 0 {
			t.Fatalf("%s: %+v", name, total)
		}
		if name == "repeated" && total.CacheHits == 0 || name == "exhausting" && total.PIRMissed == 0 {
			t.Fatalf("%s: input did not exercise its case: %+v", name, total)
		}
		if ref == nil {
			ref = srv.trace
		} else if !reflect.DeepEqual(srv.trace, ref) {
			t.Fatalf("%s: server trace differs from concentrated", name)
		}
	}
}

func TestCacheHitSendsDummy(t *testing.T) {
	const count, nProbe = 64, 4
	db, _, _ := buildDB(t, count, 2, 2)
	srv := &countingServer{PirServiceServer: mustServer(t, db, 1), offsets: map[int]bool{}}
	f, err := NewFetcher(context.Background(), InProcess(srv), BatchConfig{NProbe: nProbe})
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Preprocess(context.Background()); err != nil {
		t.Fatal(err)
	}
	for q, tc := range []struct {
		ids  []int
		hits int
	}{{[]int{0, 1, 2, 3}, 0}, {[]int{0, 1, 2, 3}, 4}, {[]int{2, 3, 10, 20}, 2}} {
		before := srv.queries
		res, err := f.Fetch(context.Background(), tc.ids)
		if err != nil {
			t.Fatal(err)
		}
		checkFetched(t, db, tc.ids, res)
		st := res.Stats
		if st.CacheHits != tc.hits || st.Requests != nProbe || srv.queries-before != nProbe || st.Real != nProbe || st.Dummy != 0 {
			t.Fatalf("query %d: server saw %d requests, stats %+v", q, srv.queries-before, st)
		}
	}
	if len(srv.offsets) != 1 {
		t.Fatalf("request sizes vary: %v", srv.offsets)
	}
}

// With several partitions, unused slots are still sent as dummy requests.
func TestFetchPartitionedSendsDummies(t *testing.T) {
	const count = 96 // partitions of 32: 12 backups per chunk
	db, _, _ := buildDB(t, count, 2, 2)
	srv := &countingServer{PirServiceServer: mustServer(t, db, 3), offsets: map[int]bool{}}
	f, err := NewFetcher(context.Background(), InProcess(srv), BatchConfig{NProbe: 2, PerPartition: 1, Rounds: 2})
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Preprocess(context.Background()); err != nil {
		t.Fatal(err)
	}
	for q, ids := range [][]int{{0, 1}, {40, 70}, {33, 95}} {
		before := srv.queries
		res, err := f.Fetch(context.Background(), ids)
		if err != nil {
			t.Fatal(err)
		}
		checkFetched(t, db, ids, res)
		st := res.Stats
		if len(res.Missed) != 0 || st.Real != 2 || st.Dummy != 4 || st.Requests != 6 || srv.queries-before != st.Requests {
			t.Fatalf("query %d stats %+v missed %v", q, st, res.Missed)
		}
	}
}

func TestFetchOverGRPCLoopback(t *testing.T) {
	db, _, _ := buildDB(t, 30, 3, 4)
	srv := mustServer(t, db, 2)
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	gs := grpc.NewServer()
	rpc.RegisterPirServiceServer(gs, srv)
	go gs.Serve(lis)
	defer gs.Stop()
	conn, err := grpc.Dial(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	f, err := NewFetcher(context.Background(), rpc.NewPirServiceClient(conn), BatchConfig{NProbe: 4, PerPartition: 2, Rounds: 3})
	if err != nil {
		t.Fatal(err)
	}
	if f.CacheStorageBytes() != 0 {
		t.Fatal("cache nonempty before queries")
	}
	if err := f.Preprocess(context.Background()); err != nil {
		t.Fatal(err)
	}
	if off := f.Offline(); off.DownBytes <= 0 || off.UpBytes <= 0 || off.HintStorageBytes <= 0 {
		t.Fatalf("offline stats %+v", off)
	}
	rng := rand.New(rand.NewSource(11))
	for q := 0; q < 10; q++ {
		ids := rng.Perm(30)[:4]
		res, err := f.Fetch(context.Background(), ids)
		if err != nil {
			t.Fatal(err)
		}
		checkFetched(t, db, ids, res)
		if f.CacheStorageBytes() <= 0 {
			t.Fatal("cache measurement excludes retrieved clusters")
		}
		up := res.Stats.Online.QueryUpBytes
		down := res.Stats.Online.QueryDownBytes
		if res.Stats.Requests != 12 || up <= 0 || down < int64(res.Stats.Requests*db.Layout.EntryBytes()) {
			t.Fatalf("stats %+v", res.Stats)
		}
	}
}

func TestPIRParams(t *testing.T) {
	l, _ := NewLayout(31250, 32, 128)
	p := DefaultBatch(8, 1).PIRParams(l, 1)
	if p.EntryCount != 31250 || p.EntryBytes != 16512 || p.BatchSize != 8 {
		t.Fatalf("%+v", p)
	}
	if b := DefaultBatch(8, 4); b.PerPartition != 2 || b.Rounds != 3 || b.RequestsPerQuery(4) != 24 {
		t.Fatalf("%+v", b)
	}
}
