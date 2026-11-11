package pianopir

import "testing"

func TestLocalStatsPreserveCacheDummyDropAndRefresh(t *testing.T) {
	db := randomDB(128, 2)
	p := NewSimpleBatchPianoPIR(128, 16, 4, db, 40)
	p.Preprocessing()
	initial := p.LocalStats()
	if initial.CacheBytes != 0 || initial.HintBytes == 0 || initial.PreprocessSeconds <= 0 {
		t.Fatalf("initial %+v", initial)
	}
	// Four inputs in one partition: two dropped and two dummies in the other.
	for pass := 0; pass < 2; pass++ {
		got, err := p.Query([]uint64{1, 2, 3, 4})
		if err != nil {
			t.Fatal(err)
		}
		if got[0][0] != db[2] || got[1][0] != db[4] || got[2][0] != 0 || got[3][0] != 0 {
			t.Fatalf("legacy results changed: %v", got)
		}
	}
	s := p.LocalStats()
	if s.Dropped != 4 || s.CacheBytes != 48 || s.Partitions[0].CacheHits != 2 || s.Partitions[1].DummyRequests != 4 {
		t.Fatalf("outcomes %+v", s)
	}
	if s.Partitions[0].Requests != 4 || s.Partitions[1].Requests != 4 || s.Partitions[0].PrivateQueries != 2 || s.Partitions[1].PrivateQueries != 4 {
		t.Fatalf("legacy cache hits unexpectedly compensated: %+v", s)
	}
	if s.Partitions[0].QueryDownBytes != 32 || s.Partitions[1].QueryDownBytes != 64 || s.Partitions[0].QueryUpBytes == 0 {
		t.Fatalf("payload counters %+v", s)
	}
	p.Preprocessing()
	s = p.LocalStats()
	if s.CacheBytes != 0 || s.Partitions[0].CompletedRequests != 4 || s.Partitions[1].CompletedRequests != 4 || s.Partitions[0].Preprocesses != 2 {
		t.Fatalf("refresh accounting %+v", s)
	}
	// Force an automatic partition rebuild using the existing budget trigger.
	p.subPIR[0].client.FinishedQueryNum = p.subPIR[0].client.MaxQueryNum
	if _, err := p.Query([]uint64{1, 2, 65, 66}); err != nil {
		t.Fatal(err)
	}
	if after := p.LocalStats(); after.Partitions[0].AutomaticBuildSeconds <= 0 || after.PreprocessSeconds <= s.PreprocessSeconds {
		t.Fatalf("automatic build missing %+v", after)
	}
}

func TestLocalStatsCountMissWithoutAddingTransportCalls(t *testing.T) {
	p := NewSimpleBatchPianoPIR(128, 16, 4, randomDB(128, 2), 40)
	p.Preprocessing()
	p.subPIR[0].client.QueryHistogram[0] = p.subPIR[0].client.maxQueryPerChunk
	if _, err := p.Query([]uint64{1, 2, 65, 66}); err != nil {
		t.Fatal(err)
	}
	s := p.LocalStats().Partitions[0]
	if s.Requests != 2 || s.Missed != 2 || s.PrivateQueries != 0 {
		t.Fatalf("miss changed legacy transport: %+v", s)
	}
}
