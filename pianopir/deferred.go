package pianopir

import (
	"fmt"
	"maps"
	"slices"
)

// BatchQuery is one request slot of a deferred batch: a real request for
// index Idx, or a dummy request when Real is false.
type BatchQuery struct {
	Idx  uint64
	Real bool
}

// DeferredBatch is a batch of requests prepared by PrepareDeferredBatch: the
// offset sets to send in one round trip and what is needed to recover the
// entries once the responses arrive.
//
// Every slot is prepared exactly as QueryOnce (or DummyQuery) would prepare
// it, in slot order, against the same client state: the hint is selected, the
// offset set is built and the hint is replaced by the next backup hint on the
// spot (tag, programmed point, QueryHistogram, FinishedQueryNum). Only the
// part of the replacement that needs the retrieved entry DB[x] — XORing it
// into the new parity — is deferred to Finish. A slot whose selected hint was
// replaced earlier in the same batch therefore records the hint parity
// without that entry, together with the earlier slot that will supply it.
// The offset sets, and the client state after Finish, are those of running
// QueryOnce slot by slot.
type DeferredBatch struct {
	c *PianoPIRClient
	// Offsets holds one offset set per slot, in slot order.
	Offsets [][]uint32
	// Outcomes is the outcome of each real slot (dummy slots report
	// QueryMissed and return no entry).
	Outcomes []QueryOutcome
	// PendingReuses counts slots whose hint was replaced by an earlier slot
	// of the batch, so that its parity waits for that slot's entry.
	PendingReuses int

	slots []deferredSlot
	done  bool
}

type deferredSlot struct {
	real   bool
	idx    uint64
	hint   uint64
	replV  []uint64 // replacement value of the slot (replacements stay fixed until the next construction)
	parity []uint64 // hint parity when the hint was selected
	prev   int      // earlier slot whose entry the recorded parity lacks, -1 if none
	apply  bool     // XOR the entry into the replaced hint (false once a later slot replaced it again)
	cached []uint64 // cache hit answered from the cache before the batch
	from   int      // cache hit answered by an earlier slot of the batch, -1 if none
}

// PrepareDeferredBatch prepares one request per query and updates the hint
// table as QueryOnce would, except for the parities, which Finish completes
// from the responses. The client must not run other queries before Finish;
// if the responses never arrive the client must be rebuilt (Initialization or
// a new preprocessing) before further use.
func (c *PianoPIRClient) PrepareDeferredBatch(queries []BatchQuery) (*DeferredBatch, error) {
	for _, q := range queries {
		if q.Real && q.Idx >= c.config.DBSize {
			return nil, fmt.Errorf("idx %v is out of range", q.Idx)
		}
	}
	entrySize := c.config.DBEntrySize
	b := &DeferredBatch{
		c:        c,
		Offsets:  make([][]uint32, len(queries)),
		Outcomes: make([]QueryOutcome, len(queries)),
		slots:    make([]deferredSlot, len(queries)),
	}
	owner := map[uint64]int{}    // hint -> last slot that replaced it in this batch
	inflight := map[uint64]int{} // idx -> slot that fetches it in this batch
	for i, q := range queries {
		s := &b.slots[i]
		*s = deferredSlot{real: q.Real, idx: q.Idx, prev: -1, from: -1}
		b.Outcomes[i] = QueryMissed
		if !q.Real {
			b.Offsets[i] = c.dummyOffsetSet()
			continue
		}
		// the cache, then the in-batch cache that QueryOnce would have filled
		if v, ok := c.localCache[q.Idx]; ok {
			s.cached = append([]uint64(nil), v...)
			b.Outcomes[i] = QueryCacheHit
			b.Offsets[i] = c.dummyOffsetSet()
			continue
		}
		if k, ok := inflight[q.Idx]; ok {
			s.from = k
			b.Outcomes[i] = QueryCacheHit
			b.Offsets[i] = c.dummyOffsetSet()
			continue
		}
		chunkId := q.Idx / c.config.ChunkSize
		offset := q.Idx % c.config.ChunkSize
		if c.FinishedQueryNum >= c.MaxQueryNum || c.QueryHistogram[chunkId] >= c.maxQueryPerChunk {
			b.Offsets[i] = c.dummyOffsetSet()
			continue
		}
		hitId := uint64(DefaultProgramPoint)
		for h := uint64(0); h < c.primaryHintNum; h++ {
			hintOffset := PRFEvalWithLongKeyAndTag(c.longKey, c.primaryShortTag[h], chunkId) & (c.config.ChunkSize - 1)
			if hintOffset == offset {
				// a set already programmed in this chunk does not cover idx
				if c.primaryProgramPoint[h] == DefaultProgramPoint || (c.primaryProgramPoint[h]/c.config.ChunkSize != chunkId) {
					hitId = h
					break
				}
			}
		}
		if hitId == DefaultProgramPoint {
			b.Offsets[i] = c.dummyOffsetSet()
			continue
		}

		querySet := make([]uint64, c.config.SetSize)
		for j := uint64(0); j < c.config.SetSize; j++ {
			hintOffset := PRFEvalWithLongKeyAndTag(c.longKey, c.primaryShortTag[hitId], j) & (c.config.ChunkSize - 1)
			querySet[j] = j*c.config.ChunkSize + hintOffset
		}
		if c.primaryProgramPoint[hitId] != DefaultProgramPoint {
			querySet[c.primaryProgramPoint[hitId]/c.config.ChunkSize] = c.primaryProgramPoint[hitId]
		}
		inGroupIdx := c.QueryHistogram[chunkId]
		querySet[chunkId] = c.replacementIdx[chunkId][inGroupIdx]
		offsets := make([]uint32, c.config.SetSize)
		for j := uint64(0); j < c.config.SetSize; j++ {
			offsets[j] = uint32(querySet[j] & (c.config.ChunkSize - 1))
		}
		b.Offsets[i] = offsets
		b.Outcomes[i] = QueryFetched

		parity := c.primaryParity[hitId*entrySize : (hitId+1)*entrySize]
		s.hint = hitId
		s.replV = c.replacementVal[chunkId][inGroupIdx*entrySize : (inGroupIdx+1)*entrySize]
		s.parity = append([]uint64(nil), parity...)
		s.apply = true
		if k, ok := owner[hitId]; ok {
			// the recorded parity lacks slot k's entry, and slot k's entry no
			// longer belongs in this hint once it is replaced again below
			s.prev = k
			b.slots[k].apply = false
			b.PendingReuses++
		}

		// replace the hint now; its parity misses DB[idx] until Finish
		c.primaryShortTag[hitId] = c.backupShortTag[chunkId][inGroupIdx]
		copy(parity, c.backupParity[chunkId][inGroupIdx*entrySize:(inGroupIdx+1)*entrySize])
		c.primaryProgramPoint[hitId] = q.Idx
		owner[hitId] = i
		inflight[q.Idx] = i
		c.FinishedQueryNum++
		c.QueryHistogram[chunkId]++
	}
	return b, nil
}

// Finish recovers the entries from the responses (one per slot, in slot
// order) and completes the deferred parity updates. It returns one entry per
// slot: the entry of a fetched or cache-hit slot, nil for missed and dummy
// slots. A dummy slot's response is ignored. After an error the hint table
// is incomplete and the client must be rebuilt before further use.
func (b *DeferredBatch) Finish(responses [][]uint64) ([][]uint64, error) {
	if b.done {
		return nil, fmt.Errorf("deferred batch already finished")
	}
	b.done = true
	if len(responses) != len(b.slots) {
		return nil, fmt.Errorf("got %d responses for %d requests", len(responses), len(b.slots))
	}
	c := b.c
	entrySize := c.config.DBEntrySize
	entries := make([][]uint64, len(b.slots))
	for i := range b.slots {
		s := &b.slots[i]
		switch {
		case !s.real || b.Outcomes[i] == QueryMissed:
		case b.Outcomes[i] == QueryCacheHit && s.from >= 0:
			entries[i] = slices.Clone(entries[s.from])
		case b.Outcomes[i] == QueryCacheHit:
			entries[i] = s.cached
		default:
			resp := responses[i]
			if uint64(len(resp)) != entrySize {
				return nil, fmt.Errorf("response %d size mismatch: want %d got %d", i, entrySize, len(resp))
			}
			// entry = response ⊕ replacement ⊕ parity at selection ⊕ pending entry
			e := slices.Clone(resp)
			EntryXor(e, s.replV, entrySize)
			EntryXor(e, s.parity, entrySize)
			if s.prev >= 0 {
				EntryXor(e, entries[s.prev], entrySize)
			}
			if s.apply {
				EntryXor(c.primaryParity[s.hint*entrySize:(s.hint+1)*entrySize], e, entrySize)
			}
			c.localCache[s.idx] = e
			entries[i] = slices.Clone(e)
		}
	}
	return entries, nil
}

// Clone returns a deep copy of the client: key, hint tables, replacements,
// programmed points, query counters and cache. The configuration is shared
// (it is never modified); a dummy offset source set by SetDummyOffsetSource
// is not copied (the copy draws dummies from crypto/rand). The copy answers
// every real query exactly as the original would, so two copies can run the
// same queries side by side.
func (c *PianoPIRClient) Clone() *PianoPIRClient {
	d := *c
	d.dummyOffsets = nil
	d.longKey = slices.Clone(c.longKey)
	d.QueryHistogram = slices.Clone(c.QueryHistogram)
	d.primaryShortTag = slices.Clone(c.primaryShortTag)
	d.primaryParity = slices.Clone(c.primaryParity)
	d.primaryProgramPoint = slices.Clone(c.primaryProgramPoint)
	d.replacementIdx = cloneRows(c.replacementIdx)
	d.replacementVal = cloneRows(c.replacementVal)
	d.backupShortTag = cloneRows(c.backupShortTag)
	d.backupParity = cloneRows(c.backupParity)
	d.localCache = maps.Clone(c.localCache)
	for k, v := range d.localCache {
		d.localCache[k] = slices.Clone(v)
	}
	return &d
}

func cloneRows(rows [][]uint64) [][]uint64 {
	out := make([][]uint64, len(rows))
	for i, r := range rows {
		out[i] = slices.Clone(r)
	}
	return out
}
