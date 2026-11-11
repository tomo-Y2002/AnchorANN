package clusterpir

import (
	"fmt"

	"example.com/private-search/rpc/server"

	"privatesearch/anchorann"
)

// DB is the server-side cluster PIR DB: entry c (cluster c) occupies
// Words[c·EntryWords:(c+1)·EntryWords].
type DB struct {
	Layout Layout
	Words  []uint64
}

// Build validates the clustering artifact contract (each cluster exactly s
// points, every original ID 0..N-1 exactly once) and that the dataset matches
// its N and dimension, then encodes cluster c's members and vectors as entry c.
func Build(a *anchorann.Artifact, d anchorann.Dataset) (*DB, error) {
	if err := a.Validate(); err != nil {
		return nil, fmt.Errorf("build cluster PIR DB: %w", err)
	}
	if d.N != a.N || d.Dim != a.Dim || len(d.Values) != d.N*d.Dim {
		return nil, fmt.Errorf("build cluster PIR DB: dataset N=%d dimension=%d does not match artifact N=%d dimension=%d", d.N, d.Dim, a.N, a.Dim)
	}
	if err := d.Validate(); err != nil {
		return nil, fmt.Errorf("build cluster PIR DB: %w", err)
	}
	l, err := NewLayout(a.Count, a.Size, a.Dim)
	if err != nil {
		return nil, err
	}
	db := &DB{Layout: l, Words: make([]uint64, l.Count*l.EntryWords())}
	vectors := make([]float32, l.Size*l.Dim)
	for c, ids := range a.Members {
		for j, id := range ids {
			copy(vectors[j*l.Dim:(j+1)*l.Dim], d.Values[int(id)*l.Dim:(int(id)+1)*l.Dim])
		}
		if err := EncodeEntry(l, ids, vectors, db.Entry(c)); err != nil {
			return nil, fmt.Errorf("build cluster PIR DB: cluster=%d: %w", c, err)
		}
	}
	return db, nil
}

// Entry returns the plaintext words of entry c (a view, not a copy).
func (db *DB) Entry(c int) []uint64 {
	w := db.Layout.EntryWords()
	return db.Words[c*w : (c+1)*w]
}

// PartitionSize is ceil(C/partitions), the entries per Piano partition; the
// last partition may be shorter.
func PartitionSize(count, partitions int) (int, error) {
	if partitions <= 0 || partitions > count {
		return 0, fmt.Errorf("partition count=%d must be in 1..C=%d", partitions, count)
	}
	size := (count + partitions - 1) / partitions
	if (partitions-1)*size >= count {
		return 0, fmt.Errorf("partition count=%d leaves an empty partition for C=%d (partition size=%d)", partitions, count, size)
	}
	return size, nil
}

// NewServer splits the DB into contiguous partitions (partition p holds
// clusters [p·size, min((p+1)·size, C))) and wraps them in the existing
// rpc/server PirService. Metadata carries N=C (entries), M=s, Dim=dimension.
func (db *DB) NewServer(partitions int, failureProbLog2 uint64) (*server.Server, error) {
	l := db.Layout
	size, err := PartitionSize(l.Count, partitions)
	if err != nil {
		return nil, err
	}
	w := l.EntryWords()
	parts := make([][]uint64, partitions)
	for p := range parts {
		end := min((p+1)*size, l.Count)
		parts[p] = db.Words[p*size*w : end*w]
	}
	cfg := server.Config{
		N:               uint64(l.Count),
		Dim:             uint64(l.Dim),
		M:               uint64(l.Size),
		DBEntryByteNum:  uint64(l.EntryBytes()),
		PartitionNum:    uint64(partitions),
		PartitionSize:   uint64(size),
		FailureProbLog2: failureProbLog2,
	}
	srv, err := server.NewServer(cfg, parts)
	if err != nil {
		return nil, fmt.Errorf("new cluster PIR server: %w", err)
	}
	return srv, nil
}
