package pianopir

import (
	crand "crypto/rand"
	"encoding/binary"
	"fmt"
)

// StreamBlockTargetBytes bounds the payload of one block of the
// preprocessing DB stream (see StreamBlockEntries).
const StreamBlockTargetBytes = 1 << 20

// StreamBlockEntries returns B, the number of entries in one block of the
// preprocessing DB stream: the largest power of two that is at most chunkSize
// and keeps B·entryBytes within StreamBlockTargetBytes (at least 1). B depends
// only on public parameters, and since chunkSize is a power of two a block
// never spans two chunks.
func StreamBlockEntries(chunkSize uint64, entryBytes uint64) uint64 {
	b := uint64(1)
	for b*2 <= chunkSize && b*2*entryBytes <= StreamBlockTargetBytes {
		b *= 2
	}
	return b
}

// StreamBlockCount returns the number of blocks that cover dbSize entries.
func StreamBlockCount(dbSize uint64, blockEntries uint64) uint64 {
	return (dbSize + blockEntries - 1) / blockEntries
}

// StreamBlockLen returns the number of entries in block index of a dbSize
// entry stream: blockEntries for every block but the last one.
func StreamBlockLen(dbSize uint64, blockEntries uint64, index uint64) uint64 {
	first := index * blockEntries
	if first >= dbSize {
		return 0
	}
	return min(blockEntries, dbSize-first)
}

// StreamPreprocessor builds the client hints from the DB streamed block by
// block in index order (the offline phase of Piano with a client-side
// preprocessing). It keeps one chunk of the DB at a time; the PRF key, the
// hint tags and the replacement positions never leave the client.
type StreamPreprocessor struct {
	c            *PianoPIRClient
	blockEntries uint64
	blockCount   uint64
	next         uint64
	chunkId      uint64
	chunk        []uint64
}

// NewStreamPreprocessor resets the client (fresh key, empty hints and cache)
// and returns a preprocessor for the stream of its DBSize entries.
func (c *PianoPIRClient) NewStreamPreprocessor() *StreamPreprocessor {
	c.Initialization()
	b := StreamBlockEntries(c.config.ChunkSize, c.config.DBEntryByteNum)
	return &StreamPreprocessor{
		c:            c,
		blockEntries: b,
		blockCount:   StreamBlockCount(c.config.DBSize, b),
		chunk:        make([]uint64, c.config.ChunkSize*c.config.DBEntrySize),
	}
}

// BlockEntries is B, the entries per block.
func (p *StreamPreprocessor) BlockEntries() uint64 { return p.blockEntries }

// BlockCount is the number of blocks the stream must contain.
func (p *StreamPreprocessor) BlockCount() uint64 { return p.blockCount }

// AddBlock consumes block index, which holds entries [firstEntry,
// firstEntry+len(entries)/DBEntrySize). Blocks must arrive in index order
// with the layout of StreamBlockEntries.
func (p *StreamPreprocessor) AddBlock(index uint64, firstEntry uint64, entries []uint64) error {
	cfg := p.c.config
	if index != p.next || index >= p.blockCount {
		return fmt.Errorf("stream block %d out of order: want block %d of %d", index, p.next, p.blockCount)
	}
	if firstEntry != index*p.blockEntries {
		return fmt.Errorf("stream block %d starts at entry %d, want %d", index, firstEntry, index*p.blockEntries)
	}
	n := StreamBlockLen(cfg.DBSize, p.blockEntries, index)
	if uint64(len(entries)) != n*cfg.DBEntrySize {
		return fmt.Errorf("stream block %d has %d words, want %d", index, len(entries), n*cfg.DBEntrySize)
	}
	inChunk := firstEntry % cfg.ChunkSize
	copy(p.chunk[inChunk*cfg.DBEntrySize:], entries)
	p.next++
	if (inChunk+n) == cfg.ChunkSize || p.next == p.blockCount {
		p.flush()
	}
	return nil
}

// flush folds the current chunk into the hints and starts the next one.
func (p *StreamPreprocessor) flush() {
	p.c.UpdatePreprocessing(p.chunkId, p.chunk)
	clear(p.chunk)
	p.chunkId++
}

// Finish checks that the whole stream arrived and folds the all-zero padding
// chunks up to SetSize into the hints.
func (p *StreamPreprocessor) Finish() error {
	if p.next != p.blockCount {
		return fmt.Errorf("stream ended after %d of %d blocks", p.next, p.blockCount)
	}
	for p.chunkId < p.c.config.SetSize {
		p.flush()
	}
	return nil
}

// QueryOutcome tells how QueryOnce answered a request.
type QueryOutcome int

const (
	// QueryFetched means the entry was retrieved with a real PIR request.
	QueryFetched QueryOutcome = iota
	// QueryCacheHit means the entry came from the local cache; a dummy
	// request was sent in place of the real one.
	QueryCacheHit
	// QueryMissed means no usable hint was left (the backup hints of the
	// chunk, a primary hint covering the index, or the query budget were used
	// up); a dummy request was sent and no entry is returned.
	QueryMissed
)

// String returns the outcome name.
func (o QueryOutcome) String() string {
	switch o {
	case QueryFetched:
		return "fetched"
	case QueryCacheHit:
		return "cache_hit"
	case QueryMissed:
		return "missed"
	}
	return fmt.Sprintf("QueryOutcome(%d)", int(o))
}

// MaxQueryPerChunk is the number of backup hints (and replacements) per chunk.
func (c *PianoPIRClient) MaxQueryPerChunk() uint64 { return c.maxQueryPerChunk }

// Config returns the client configuration.
func (c *PianoPIRClient) Config() *PianoPIRConfig { return c.config }

// PrimaryHintNum is the number of primary hints.
func (c *PianoPIRClient) PrimaryHintNum() uint64 { return c.primaryHintNum }

// DummyQuery sends one request with uniformly random offsets.
func (c *PianoPIRClient) DummyQuery(transport PrivateQueryTransport) error {
	_, err := transport.PrivateQuery(c.dummyOffsetSet())
	return err
}

// dummyOffsetSet draws the offsets of one dummy request: SetSize uniform
// offsets from crypto/rand, or from the source set by SetDummyOffsetSource.
func (c *PianoPIRClient) dummyOffsetSet() []uint32 {
	if c.dummyOffsets != nil {
		return c.dummyOffsets(c.config.SetSize, c.config.ChunkSize)
	}
	return randomOffsets32(c.config.SetSize, c.config.ChunkSize)
}

// SetDummyOffsetSource replaces the crypto/rand offsets of dummy requests
// with fn(SetSize, ChunkSize). It exists only so that tests can reproduce a
// request sequence exactly; a predictable source makes dummies
// distinguishable and must never be used against a real server. nil
// restores crypto/rand.
func (c *PianoPIRClient) SetDummyOffsetSource(fn func(setSize, chunkSize uint64) []uint32) {
	c.dummyOffsets = fn
}

// QueryOnce answers idx and sends exactly one PrivateQuery whatever the
// outcome, so the server observes the same request whether the entry was
// fetched, cached or missed. A miss never triggers a re-preprocessing; the
// caller decides when to preprocess again from the public request count.
func (c *PianoPIRClient) QueryOnce(idx uint64, transport PrivateQueryTransport) ([]uint64, QueryOutcome, error) {
	if idx >= c.config.DBSize {
		return nil, QueryMissed, fmt.Errorf("idx %v is out of range", idx)
	}
	if v, ok := c.localCache[idx]; ok {
		return append([]uint64(nil), v...), QueryCacheHit, c.DummyQuery(transport)
	}
	chunkId := idx / c.config.ChunkSize
	offset := idx % c.config.ChunkSize
	if c.FinishedQueryNum >= c.MaxQueryNum || c.QueryHistogram[chunkId] >= c.maxQueryPerChunk {
		return nil, QueryMissed, c.DummyQuery(transport)
	}

	hitId := uint64(DefaultProgramPoint)
	for i := uint64(0); i < c.primaryHintNum; i++ {
		hintOffset := PRFEvalWithLongKeyAndTag(c.longKey, c.primaryShortTag[i], chunkId) & (c.config.ChunkSize - 1)
		if hintOffset == offset {
			// a set already programmed in this chunk does not cover idx
			if c.primaryProgramPoint[i] == DefaultProgramPoint || (c.primaryProgramPoint[i]/c.config.ChunkSize != chunkId) {
				hitId = i
				break
			}
		}
	}
	if hitId == DefaultProgramPoint {
		return nil, QueryMissed, c.DummyQuery(transport)
	}

	querySet := make([]uint64, c.config.SetSize)
	for i := uint64(0); i < c.config.SetSize; i++ {
		hintOffset := PRFEvalWithLongKeyAndTag(c.longKey, c.primaryShortTag[hitId], i) & (c.config.ChunkSize - 1)
		querySet[i] = i*c.config.ChunkSize + hintOffset
	}
	if c.primaryProgramPoint[hitId] != DefaultProgramPoint {
		querySet[c.primaryProgramPoint[hitId]/c.config.ChunkSize] = c.primaryProgramPoint[hitId]
	}
	inGroupIdx := c.QueryHistogram[chunkId]
	entrySize := c.config.DBEntrySize
	replVal := c.replacementVal[chunkId][inGroupIdx*entrySize : (inGroupIdx+1)*entrySize]
	querySet[chunkId] = c.replacementIdx[chunkId][inGroupIdx]

	offsets := make([]uint32, c.config.SetSize)
	for i := uint64(0); i < c.config.SetSize; i++ {
		offsets[i] = uint32(querySet[i] & (c.config.ChunkSize - 1))
	}
	response, err := transport.PrivateQuery(offsets)
	if err != nil {
		return nil, QueryFetched, err
	}
	if uint64(len(response)) != entrySize {
		return nil, QueryFetched, fmt.Errorf("response size mismatch: want %d got %d", entrySize, len(response))
	}

	// revert the replacement and the hint parity to recover the entry
	EntryXor(response, replVal, entrySize)
	EntryXor(response, c.primaryParity[hitId*entrySize:(hitId+1)*entrySize], entrySize)

	// refresh the used primary hint with the next backup hint of the chunk
	c.primaryShortTag[hitId] = c.backupShortTag[chunkId][inGroupIdx]
	copy(c.primaryParity[hitId*entrySize:(hitId+1)*entrySize], c.backupParity[chunkId][inGroupIdx*entrySize:(inGroupIdx+1)*entrySize])
	c.primaryProgramPoint[hitId] = idx
	EntryXor(c.primaryParity[hitId*entrySize:(hitId+1)*entrySize], response, entrySize)

	c.FinishedQueryNum++
	c.QueryHistogram[chunkId]++
	c.localCache[idx] = response
	return append([]uint64(nil), response...), QueryFetched, nil
}

// cryptoKey128 draws a PRF key from crypto/rand.
func cryptoKey128() PrfKey128 {
	var key PrfKey128
	if _, err := crand.Read(key[:]); err != nil {
		panic(fmt.Sprintf("crypto/rand.Read failed: %v", err))
	}
	return key
}

// cryptoOffsets draws n uniform offsets in [0, chunkSize) from crypto/rand.
// chunkSize must be a power of two.
func cryptoOffsets(n uint64, chunkSize uint64) []uint64 {
	buf := make([]byte, n*8)
	if _, err := crand.Read(buf); err != nil {
		panic(fmt.Sprintf("crypto/rand.Read failed: %v", err))
	}
	out := make([]uint64, n)
	for i := range out {
		out[i] = binary.LittleEndian.Uint64(buf[i*8:]) & (chunkSize - 1)
	}
	return out
}

func randomOffsets32(n uint64, chunkSize uint64) []uint32 {
	offsets := cryptoOffsets(n, chunkSize)
	out := make([]uint32, n)
	for i, o := range offsets {
		out[i] = uint32(o)
	}
	return out
}
