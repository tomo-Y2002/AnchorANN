// Command search-client runs AnchorANN queries against search-server: it
// receives and validates the anchor NSG bundle (or loads it from a file),
// builds the Piano hints for each n_probe, answers every query of a query
// file through the anchorann/search pipeline for each L_search (and the exact
// anchor diagnostic mode), and reports Recall@k (with a ground-truth file),
// the latency breakdown, online traffic, misses and refreshes as JSON.
// -per-query additionally writes one JSON line per query.
package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"example.com/private-search/rpc"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"privatesearch/anchorann"
	"privatesearch/anchorann/bundle"
	"privatesearch/anchorann/bundlerpc"
	"privatesearch/anchorann/clusterpir"
	"privatesearch/anchorann/search"
	"privatesearch/internal/annmetric"
)

type latency struct {
	Mean float64 `json:"mean"`
	P50  float64 `json:"p50"`
	P95  float64 `json:"p95"`
	P99  float64 `json:"p99"`
	Max  float64 `json:"max"`
}

// configReport summarizes one (mode, L_search, n_probe) run over all queries.
type configReport struct {
	Mode    string `json:"mode"` // "nsg" (local graph search) or "exact" (diagnostic)
	LSearch int    `json:"l_search"`
	NProbe  int    `json:"n_probe"`
	K       int    `json:"k"`
	Queries int    `json:"queries"`
	// Historical kRecall@k is retained for comparisons with phase 1.
	Recall      *float64 `json:"recall_at_k,omitempty"`
	RecallHits  int      `json:"recall_hits,omitempty"`
	OneRecall   *float64 `json:"one_recall_at_k,omitempty"`
	NearestHits int      `json:"nearest_hits,omitempty"`
	// Share of the ground truth top-k found among the retrieved candidates
	// (all points of the fetched clusters), before reranking.
	CandidateContainment    *float64 `json:"candidate_gt_containment,omitempty"`
	CandidateHits           int      `json:"candidate_gt_hits,omitempty"`
	CandidateOneContainment *float64 `json:"candidate_gt_one_containment,omitempty"`
	CandidateNearestHits    int      `json:"candidate_nearest_hits,omitempty"`
	// Refresh forced before this run (all but the first run of an n_probe),
	// so clusters cached by an earlier run cannot become cache hits here.
	ForcedRefreshSeconds float64 `json:"forced_refresh_seconds,omitempty"`
	Online               latency `json:"online_seconds"`
	AnchorSeconds        float64 `json:"mean_anchor_seconds"`
	FetchSeconds         float64 `json:"mean_fetch_seconds"`
	RerankSeconds        float64 `json:"mean_rerank_seconds"`
	UpBytes              float64 `json:"online_up_bytes_per_query"`
	DownBytes            float64 `json:"online_down_bytes_per_query"`
	Requests             int     `json:"requests"`
	RPCs                 int     `json:"rpcs"` // PIR round trips (PrivateQuery or BatchPrivateQuery)
	CacheHits            int     `json:"cache_hits"`
	CacheStorageBytes    int64   `json:"client_cache_storage_bytes"`
	AnchorEvaluations    float64 `json:"mean_anchor_distance_evaluations"`
	Candidates           float64 `json:"mean_candidates"`
	MissedClusters       int     `json:"missed_clusters"`
	PIRMissedClusters    int     `json:"pir_missed_clusters"` // missed for lack of hints (part of missed_clusters)
	IncompleteQueries    int     `json:"incomplete_queries"`
	Shortfall            int     `json:"shortfall_ids"`
	Refreshes            int     `json:"refreshes"` // hint rebuilds on the public schedule, excluded from the online time
	RefreshSeconds       float64 `json:"refresh_seconds"`
	RefreshDownBytes     int64   `json:"refresh_down_bytes"`
	RefreshUpBytes       int64   `json:"refresh_up_bytes"`
	// Head summarizes the first -head-queries queries alone (comparable with
	// runs of that many queries).
	Head *headReport `json:"first_queries,omitempty"`
}

// headReport is the online cost and accuracy of the first Queries queries of
// a run.
type headReport struct {
	Queries        int      `json:"queries"`
	Online         latency  `json:"online_seconds"`
	FetchSeconds   float64  `json:"mean_fetch_seconds"`
	UpBytes        float64  `json:"online_up_bytes_per_query"`
	DownBytes      float64  `json:"online_down_bytes_per_query"`
	Requests       int      `json:"requests"`
	RPCs           int      `json:"rpcs"`
	MissedClusters int      `json:"missed_clusters"`
	Refreshes      int      `json:"refreshes"`
	OneRecall      *float64 `json:"one_recall_at_k,omitempty"`
	Recall         *float64 `json:"recall_at_k,omitempty"`
}

type probeReport struct {
	NProbe          int                     `json:"n_probe"`
	Batch           clusterpir.BatchConfig  `json:"batch"`
	Offline         clusterpir.OfflineStats `json:"offline"`
	ClientHeapBytes int64                   `json:"client_heap_after_preprocess_bytes"`
	// The hints are rebuilt before the query whose requests (of any kind)
	// would exceed MaxQueryNum per partition, i.e. every QueriesPerRefresh
	// queries; a chunk that used its BackupPerChunk backups misses.
	MaxQueryNum       int            `json:"max_query_num"`
	QueriesPerRefresh int            `json:"queries_per_refresh"`
	BackupPerChunk    int            `json:"backup_per_chunk"`
	DBBytes           int64          `json:"db_bytes"`
	Runs              []configReport `json:"runs"`
}

type report struct {
	Addr         string                `json:"addr"`
	BundleSource string                `json:"bundle_source"`
	BundleFetch  *bundlerpc.FetchStats `json:"bundle_fetch,omitempty"`
	BundleBytes  int                   `json:"bundle_bytes"`
	BundleSHA256 string                `json:"bundle_sha256"`
	SourceSHA256 string                `json:"artifact_sha256"`
	LoadSeconds  float64               `json:"bundle_load_and_validate_seconds"`
	N            int                   `json:"n"`
	Count        int                   `json:"c"`
	Size         int                   `json:"s"`
	Dimension    int                   `json:"dimension"`
	Partitions   int                   `json:"partitions"`
	QueryFile    string                `json:"query_file"`
	Queries      int                   `json:"queries"`
	GroundTruth  string                `json:"ground_truth,omitempty"`
	K            int                   `json:"k"`
	Probes       []probeReport         `json:"n_probe"`
	PeakRSSBytes int64                 `json:"peak_rss_bytes"`
	GoVersion    string                `json:"go_version"`
	Platform     string                `json:"platform"`
}

// queryRecord is one line of -per-query output. Clusters are client-local.
type queryRecord struct {
	Mode       string         `json:"mode"`
	LSearch    int            `json:"l_search"`
	NProbe     int            `json:"n_probe"`
	Query      int            `json:"query"`
	Hits       *int           `json:"recall_hits,omitempty"`
	NearestHit *int           `json:"nearest_hit,omitempty"`
	Result     *search.Result `json:"result"`
}

func ints(s string) ([]int, error) {
	var out []int
	for _, part := range strings.Split(s, ",") {
		v, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil || v <= 0 {
			return nil, fmt.Errorf("invalid positive integer list %q", s)
		}
		out = append(out, v)
	}
	return out, nil
}

func peakRSS() int64 {
	var u syscall.Rusage
	if syscall.Getrusage(syscall.RUSAGE_SELF, &u) != nil {
		return 0
	}
	if runtime.GOOS == "darwin" {
		return u.Maxrss
	}
	return u.Maxrss * 1024
}

func heapAlloc() int64 {
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return int64(m.HeapAlloc)
}

func summarize(xs []float64) latency {
	if len(xs) == 0 {
		return latency{}
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	sum := 0.0
	for _, x := range s {
		sum += x
	}
	at := func(q float64) float64 { return s[min(len(s)-1, int(q*float64(len(s))))] }
	return latency{Mean: sum / float64(len(s)), P50: at(0.5), P95: at(0.95), P99: at(0.99), Max: s[len(s)-1]}
}

// readIvecs reads the first rows of an ivecs ground-truth file; every row
// must hold at least k IDs.
func readIvecs(path string, rows, k int) ([][]uint32, error) {
	if strings.HasSuffix(strings.ToLower(path), ".npy") {
		matrix, err := loadIntMatrix(path, rows, k)
		if err != nil {
			return nil, fmt.Errorf("load ground truth: %w", err)
		}
		if len(matrix) != rows {
			return nil, fmt.Errorf("ground truth has %d rows, need %d", len(matrix), rows)
		}
		out := make([][]uint32, rows)
		for i, row := range matrix {
			if len(row) < k {
				return nil, fmt.Errorf("ground truth row=%d has %d IDs, need k=%d", i, len(row), k)
			}
			for _, id := range row[:k] {
				if id < 0 {
					return nil, fmt.Errorf("negative ground truth ID=%d row=%d", id, i)
				}
				out[i] = append(out[i], uint32(id))
			}
		}
		return out, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open ground truth: %w", err)
	}
	defer f.Close()
	r := bufio.NewReader(f)
	out := make([][]uint32, 0, rows)
	for len(out) < rows {
		var dim uint32
		if err := binary.Read(r, binary.LittleEndian, &dim); err != nil {
			return nil, fmt.Errorf("ground truth has %d rows, need %d: %w", len(out), rows, err)
		}
		if dim < uint32(k) || dim > 1<<20 {
			return nil, fmt.Errorf("ground truth row=%d has %d IDs, need k=%d", len(out), dim, k)
		}
		row := make([]uint32, dim)
		if err := binary.Read(r, binary.LittleEndian, row); err != nil {
			return nil, fmt.Errorf("ground truth row=%d: %w", len(out), err)
		}
		out = append(out, row[:k])
	}
	return out, nil
}

func hits(got []uint32, gt []uint32) int {
	want := make(map[uint32]bool, len(gt))
	for _, id := range gt {
		want[id] = true
	}
	n := 0
	for _, id := range got {
		if want[id] {
			n++
		}
	}
	return n
}

type options struct {
	lists     []int
	modes     []string
	k         int
	refresh   bool
	gt        [][]uint32
	recorder  *recorder
	perQuery  *json.Encoder
	queries   anchorann.Dataset
	batchOpts clusterpir.BatchConfig
	head      int
}

func run(ctx context.Context, args []string, out io.Writer) error {
	f := flag.NewFlagSet("search-client", flag.ContinueOnError)
	addr := f.String("addr", "127.0.0.1:50091", "search-server address (PirService and, unless -bundle/-bundle-addr, AnchorBundle)")
	bundleAddr := f.String("bundle-addr", "", "separate AnchorBundle address (default -addr)")
	bundlePath := f.String("bundle", "", "load the bundle from this file instead of fetching it")
	wantC := f.Int("expect-c", 0, "required cluster count C (0 = any)")
	wantDim := f.Int("expect-dim", 0, "required dimension (0 = any)")
	wantS := f.Int("expect-s", 0, "required cluster size s (0 = any)")
	queryFile := f.String("queries", "", "query vectors, fvecs or bvecs (required)")
	queryLimit := f.Int("query-limit", 0, "explicit query prefix size; 0 reads all")
	gtFile := f.String("gt", "", "optional ground truth (nearest first) for 1Recall@k and kRecall@k")
	k := f.Int("k", 10, "final answer count k")
	lList := f.String("l-search", "32", "comma-separated L_search values for the nsg mode")
	pList := f.String("n-probe", "8", "comma-separated n_probe values (one Piano preprocessing each)")
	modeList := f.String("modes", "nsg", "comma-separated modes: nsg (local graph search), exact (diagnostic exact anchor ranking)")
	perPartition := f.Int("per-partition", 0, "PIR requests per partition per round (0 = default batch)")
	rounds := f.Int("rounds", 0, "PIR rounds per query (0 = default batch)")
	pirRPC := f.String("pir-rpc", clusterpir.RPCSequential, "PIR round trips: sequential (one PrivateQuery per request) or batch (one BatchPrivateQuery per partition and round)")
	head := f.Int("head-queries", 10, "also summarize the first N queries of every run (0 = off)")
	refresh := f.Bool("refresh-between-runs", true, "rebuild hints between runs of the same n_probe so cached clusters do not carry over")
	perQuery := f.String("per-query", "", "optional JSONL output with one record per query")
	maxMsgMB := f.Int("max-msg-mb", 1024, "gRPC max receive message MiB")
	wantBundleSHA := f.String("expect-bundle-sha256", "", "required SHA-256 of the received bundle (hex; empty = any)")
	wantArtifactSHA := f.String("expect-artifact-sha256", "", "required SHA-256 of the clustering artifact the bundle was built from (hex; empty = any)")
	if err := f.Parse(args); err != nil {
		return err
	}
	if *queryFile == "" {
		return fmt.Errorf("-queries is required")
	}
	if *k <= 0 || *maxMsgMB <= 0 {
		return fmt.Errorf("-k and -max-msg-mb must be positive")
	}
	if *head < 0 {
		return fmt.Errorf("-head-queries must not be negative")
	}
	if *bundlePath != "" && *bundleAddr != "" {
		return fmt.Errorf("-bundle and -bundle-addr are exclusive")
	}
	mode, err := clusterpir.ParseRPCMode(*pirRPC)
	if err != nil {
		return fmt.Errorf("-pir-rpc: %w", err)
	}
	opt := options{k: *k, refresh: *refresh, head: *head, batchOpts: clusterpir.BatchConfig{PerPartition: *perPartition, Rounds: *rounds, RPC: mode}}
	ls, err := ints(*lList)
	if err != nil {
		return fmt.Errorf("-l-search: %w", err)
	}
	ps, err := ints(*pList)
	if err != nil {
		return fmt.Errorf("-n-probe: %w", err)
	}
	opt.lists = ls
	for _, m := range strings.Split(*modeList, ",") {
		m = strings.TrimSpace(m)
		if m != "nsg" && m != "exact" {
			return fmt.Errorf("-modes: unknown mode %q", m)
		}
		opt.modes = append(opt.modes, m)
	}

	opts := []grpc.DialOption{grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(*maxMsgMB<<20), grpc.MaxCallSendMsgSize(*maxMsgMB<<20))}
	conn, err := grpc.Dial(*addr, opts...)
	if err != nil {
		return fmt.Errorf("dial: %w", err)
	}
	defer conn.Close()

	rep := report{Addr: *addr, QueryFile: *queryFile, GroundTruth: *gtFile, K: *k, GoVersion: runtime.Version(), Platform: runtime.GOOS + "/" + runtime.GOARCH}
	b, err := loadBundle(ctx, conn, *bundleAddr, *bundlePath, bundle.Expect{Count: *wantC, Dim: *wantDim, Size: *wantS}, opts, &rep)
	if err != nil {
		return err
	}
	rep.SourceSHA256 = fmt.Sprintf("%x", b.SourceSHA256)
	if err := checkPins(rep.BundleSHA256, rep.SourceSHA256, *wantBundleSHA, *wantArtifactSHA); err != nil {
		return err
	}
	ix, err := bundle.NewIndex(b)
	if err != nil {
		return err
	}
	rep.N, rep.Count, rep.Size, rep.Dimension = b.N, b.Count, b.Size, b.Dim

	if opt.queries, err = anchorann.ReadVectors(*queryFile, "auto", *queryLimit); err != nil {
		return err
	}
	if opt.queries.Dim != b.Dim {
		return fmt.Errorf("query dimension=%d bundle dimension=%d", opt.queries.Dim, b.Dim)
	}
	rep.Queries = opt.queries.N
	if *gtFile != "" {
		if opt.gt, err = readIvecs(*gtFile, opt.queries.N, *k); err != nil {
			return err
		}
	}
	if *perQuery != "" {
		pf, err := os.Create(*perQuery)
		if err != nil {
			return fmt.Errorf("create per-query output: %w", err)
		}
		w := bufio.NewWriter(pf)
		defer func() {
			w.Flush()
			pf.Close()
		}()
		opt.perQuery = json.NewEncoder(w)
	}

	client := rpc.NewPirServiceClient(conn)
	for _, n := range ps {
		pr, parts, err := measure(ctx, client, ix, n, &opt)
		if err != nil {
			return fmt.Errorf("n_probe=%d: %w", n, err)
		}
		rep.Partitions = parts
		rep.Probes = append(rep.Probes, *pr)
	}
	rep.PeakRSSBytes = peakRSS()
	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	if err := enc.Encode(rep); err != nil {
		return fmt.Errorf("write report: %w", err)
	}
	return nil
}

// loadBundle reads the bundle from a file or fetches it over AnchorBundle.
func loadBundle(ctx context.Context, conn *grpc.ClientConn, bundleAddr, path string, want bundle.Expect, opts []grpc.DialOption, rep *report) (*bundle.Bundle, error) {
	if path != "" {
		start := time.Now()
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read bundle: %w", err)
		}
		b, err := bundle.Decode(data, want)
		if err != nil {
			return nil, err
		}
		rep.BundleSource, rep.BundleBytes, rep.BundleSHA256 = path, len(data), bundle.Checksum(data)
		rep.LoadSeconds = time.Since(start).Seconds()
		return b, nil
	}
	bc := conn
	rep.BundleSource = rep.Addr
	if bundleAddr != "" {
		c, err := grpc.Dial(bundleAddr, opts...)
		if err != nil {
			return nil, fmt.Errorf("dial bundle server: %w", err)
		}
		defer c.Close()
		bc, rep.BundleSource = c, bundleAddr
	}
	b, data, st, err := bundlerpc.Fetch(ctx, bundlerpc.NewAnchorBundleClient(bc), want)
	if err != nil {
		return nil, err
	}
	rep.BundleFetch, rep.BundleBytes, rep.BundleSHA256, rep.LoadSeconds = &st, len(data), st.SHA256, st.DecodeSeconds
	return b, nil
}

// measure preprocesses Piano for n_probe and runs every mode / L_search.
func measure(ctx context.Context, client rpc.PirServiceClient, ix *bundle.Index, n int, opt *options) (*probeReport, int, error) {
	heap0 := heapAlloc()
	batch := opt.batchOpts
	batch.NProbe = n
	fe, err := clusterpir.NewFetcher(ctx, client, batch)
	if err != nil {
		return nil, 0, err
	}
	var r search.Retriever = search.Piano{F: fe}
	if opt.gt != nil {
		opt.recorder = &recorder{Retriever: r}
		r = opt.recorder
	}
	sc, err := search.New(ix, r)
	if err != nil {
		return nil, 0, err
	}
	if err := sc.Preprocess(ctx); err != nil {
		return nil, 0, err
	}
	pr := &probeReport{NProbe: n, Batch: fe.Batch(), Offline: fe.Offline(), ClientHeapBytes: heapAlloc() - heap0,
		MaxQueryNum: fe.MaxQueryNum(), QueriesPerRefresh: fe.FetchesPerRefresh(), BackupPerChunk: fe.BackupPerChunk(), DBBytes: fe.DBBytes()}
	var runs []search.Params
	for _, m := range opt.modes {
		if m == "exact" {
			runs = append(runs, search.Params{NProbe: n, K: opt.k, ExactAnchors: true})
			continue
		}
		for _, l := range opt.lists {
			if l >= n {
				runs = append(runs, search.Params{LSearch: l, NProbe: n, K: opt.k})
			}
		}
	}
	for i, p := range runs {
		forced := 0.0
		if i > 0 && opt.refresh {
			start := time.Now()
			if err := sc.Refresh(ctx); err != nil {
				return nil, 0, err
			}
			forced = time.Since(start).Seconds()
		}
		cr, err := runQueries(ctx, sc, p, opt)
		if err != nil {
			return nil, 0, fmt.Errorf("%s L_search=%d: %w", mode(p), p.LSearch, err)
		}
		cr.ForcedRefreshSeconds = forced
		cr.CacheStorageBytes = fe.CacheStorageBytes()
		pr.Runs = append(pr.Runs, *cr)
	}
	return pr, fe.Partitions(), nil
}

// recorder keeps the original IDs of the clusters retrieved by the last Fetch,
// so the candidate set can be compared with the ground truth.
type recorder struct {
	search.Retriever
	ids []uint32
}

// Fetch forwards to the wrapped Retriever and records the retrieved IDs.
func (r *recorder) Fetch(ctx context.Context, ids []int) (*search.Fetched, error) {
	r.ids = r.ids[:0]
	f, err := r.Retriever.Fetch(ctx, ids)
	if f != nil {
		for _, c := range f.Clusters {
			if c != nil {
				r.ids = append(r.ids, c.IDs...)
			}
		}
	}
	return f, err
}

// checkPins rejects a bundle whose SHA-256 or source artifact SHA-256 differs
// from the pinned values.
func checkPins(bundleSHA, artifactSHA, wantBundle, wantArtifact string) error {
	if wantBundle != "" && !strings.EqualFold(bundleSHA, wantBundle) {
		return fmt.Errorf("bundle sha256=%s, expected %s", bundleSHA, wantBundle)
	}
	if wantArtifact != "" && !strings.EqualFold(artifactSHA, wantArtifact) {
		return fmt.Errorf("bundle was built from artifact sha256=%s, expected %s", artifactSHA, wantArtifact)
	}
	return nil
}

func mode(p search.Params) string {
	if p.ExactAnchors {
		return "exact"
	}
	return "nsg"
}

// runQueries answers every query with p. Incomplete answers (PIR misses) are
// counted and kept; any other error, including transport failures, aborts.
func runQueries(ctx context.Context, sc *search.Client, p search.Params, opt *options) (*configReport, error) {
	q := opt.queries
	cr := &configReport{Mode: mode(p), LSearch: p.LSearch, NProbe: p.NProbe, K: p.K, Queries: q.N}
	online := make([]float64, 0, q.N)
	var up, down int64
	evals, cands := 0, 0
	var head *headReport
	if opt.head > 0 && opt.head < q.N {
		head = &headReport{Queries: opt.head}
	}
	var headHits, headNearest int
	for i := 0; i < q.N; i++ {
		if head != nil && i == head.Queries {
			n := float64(head.Queries)
			head.Online = summarize(online)
			head.FetchSeconds = cr.FetchSeconds / n
			head.UpBytes, head.DownBytes = float64(up)/n, float64(down)/n
			head.Requests, head.RPCs, head.MissedClusters, head.Refreshes = cr.Requests, cr.RPCs, cr.MissedClusters, cr.Refreshes
			headHits, headNearest = cr.RecallHits, cr.NearestHits
		}
		res, err := sc.Search(ctx, q.Values[i*q.Dim:(i+1)*q.Dim], p)
		if err != nil && !errors.Is(err, search.ErrIncomplete) {
			return nil, fmt.Errorf("query %d: %w", i, err)
		}
		if !res.Complete {
			cr.IncompleteQueries++
		}
		cr.MissedClusters += len(res.MissedClusters)
		cr.PIRMissedClusters += res.PIRMissed
		cr.Shortfall += res.Shortfall
		online = append(online, res.Timing.OnlineSeconds)
		cr.AnchorSeconds += res.Timing.AnchorSeconds
		cr.FetchSeconds += res.Timing.FetchSeconds
		cr.RerankSeconds += res.Timing.RerankSeconds
		up += res.UpBytes
		down += res.DownBytes
		cr.Requests += res.Requests
		cr.RPCs += res.RPCs
		cr.CacheHits += res.CacheHits
		evals += res.AnchorDistanceEvaluations
		cands += res.Candidates
		cr.Refreshes += res.Refresh.Count
		cr.RefreshSeconds += res.Refresh.Seconds
		cr.RefreshDownBytes += res.Refresh.DownBytes
		cr.RefreshUpBytes += res.Refresh.UpBytes
		var h, nearest *int
		if opt.gt != nil {
			v := hits(res.IDs, opt.gt[i])
			cr.RecallHits += v
			h = &v
			one := annmetric.NearestHit(res.IDs, opt.gt[i], p.K)
			cr.NearestHits += one
			nearest = &one
			if opt.recorder != nil {
				cr.CandidateHits += hits(opt.recorder.ids, opt.gt[i])
				cr.CandidateNearestHits += annmetric.NearestHit(opt.recorder.ids, opt.gt[i], len(opt.recorder.ids))
			}
		}
		if opt.perQuery != nil {
			if err := opt.perQuery.Encode(queryRecord{Mode: cr.Mode, LSearch: p.LSearch, NProbe: p.NProbe, Query: i, Hits: h, NearestHit: nearest, Result: res}); err != nil {
				return nil, fmt.Errorf("write per-query output: %w", err)
			}
		}
	}
	n := float64(q.N)
	cr.Online = summarize(online)
	cr.AnchorSeconds /= n
	cr.FetchSeconds /= n
	cr.RerankSeconds /= n
	cr.UpBytes, cr.DownBytes = float64(up)/n, float64(down)/n
	cr.AnchorEvaluations, cr.Candidates = float64(evals)/n, float64(cands)/n
	if head != nil && opt.gt != nil {
		one := float64(headNearest) / float64(head.Queries)
		r := float64(headHits) / float64(head.Queries*p.K)
		head.OneRecall, head.Recall = &one, &r
	}
	cr.Head = head
	if opt.gt != nil {
		one := float64(cr.NearestHits) / n
		cr.OneRecall = &one
		r := float64(cr.RecallHits) / (n * float64(p.K))
		cr.Recall = &r
		if opt.recorder != nil {
			cOne := float64(cr.CandidateNearestHits) / n
			cr.CandidateOneContainment = &cOne
			c := float64(cr.CandidateHits) / (n * float64(p.K))
			cr.CandidateContainment = &c
		}
	}
	return cr, nil
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 24*time.Hour)
	defer cancel()
	if err := run(ctx, os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "anchorann search-client:", err)
		os.Exit(1)
	}
}
