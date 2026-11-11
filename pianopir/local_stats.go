package pianopir

// LocalPIRStats measures the legacy in-process path without changing its
// scheduling or adding compensation requests. Wire bytes are modeled payload
// bytes at the private-query boundary; an in-process run sends no network bytes.
type LocalPIRStats struct {
	Requests              uint64
	DummyRequests         uint64
	CacheHits             uint64
	Missed                uint64
	PrivateQueries        uint64
	QueryUpBytes          uint64
	QueryDownBytes        uint64
	Preprocesses          uint64
	PreprocessSeconds     float64
	PreprocessDBBytes     uint64
	CompletedRequests     uint64
	AutomaticBuildSeconds float64
}

type localCountingTransport struct {
	server *PianoPIRServer
	stats  *LocalPIRStats
}

func (t localCountingTransport) PrivateQuery(offsets []uint32) ([]uint64, error) {
	t.stats.PrivateQueries++
	t.stats.QueryUpBytes += uint64(len(offsets)) * 4
	reply, err := t.server.PrivateQuery(offsets)
	if err == nil {
		t.stats.QueryDownBytes += uint64(len(reply)) * 8
	}
	return reply, err
}

// LocalBatchStats includes all scheduled requests (real, dummy and cache hits)
// and unscheduled input positions discarded by the existing partition limit.
type LocalBatchStats struct {
	Partitions        []LocalPIRStats
	Dropped           uint64
	PreprocessSeconds float64 // bulk rebuild wall time plus automatic partition rebuilds
	HintBytes         uint64
	CacheBytes        uint64
}

// LocalStats returns a snapshot of the legacy in-process measurements.
func (p *SimpleBatchPianoPIR) LocalStats() LocalBatchStats {
	s := LocalBatchStats{Dropped: p.dropped, PreprocessSeconds: p.bulkPreprocessSeconds}
	for _, part := range p.subPIR {
		s.Partitions = append(s.Partitions, part.stats)
		s.PreprocessSeconds += part.stats.AutomaticBuildSeconds
		s.HintBytes += uint64(part.client.LocalStorageSize())
		s.CacheBytes += uint64(part.client.LocalCacheStorageBytes())
	}
	return s
}

// LocalMaxQueryNums returns the nominal per-partition query budgets. Cache hits
// and dummies do not consume that budget in this unchanged legacy path.
func (p *SimpleBatchPianoPIR) LocalMaxQueryNums() []uint64 {
	var budgets []uint64
	for _, part := range p.subPIR {
		budgets = append(budgets, part.client.MaxQueryNum)
	}
	return budgets
}
