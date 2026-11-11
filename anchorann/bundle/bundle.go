// Package bundle defines the AnchorANN distribution bundle: anchor vectors,
// the server-built anchor NSG, the anchor -> cluster mapping and public
// cluster-retrieval parameters. Clients load and validate a bundle and search
// it locally; they never build the graph.
package bundle

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"privatesearch/anchorann"
	"privatesearch/anchorann/nsg"
)

// FormatVersion is the bundle format version.
const FormatVersion uint32 = 1

var magic = [8]byte{'A', 'N', 'C', 'N', 'S', 'G', '0', '1'}

// PIRParams are the public cluster-retrieval parameters a client needs. They
// are finalized by the cluster PIR DB; all-zero means not yet set.
type PIRParams struct {
	EntryCount uint64 `json:"entry_count"` // PIR DB entries; must equal the cluster count when set
	EntryBytes uint64 `json:"entry_bytes"` // bytes per entry; positive multiple of 8 when set
	BatchSize  uint32 `json:"batch_size"`  // PIR requests per query batch; 0 if unset
}

// IsSet reports whether the PIR parameters have been filled in.
func (p PIRParams) IsSet() bool { return p != PIRParams{} }

// Bundle is the decoded distribution bundle. Anchor a's vector is
// Vectors[a*Dim:(a+1)*Dim] and its cluster is ClusterOf[a].
type Bundle struct {
	Version    uint32
	N          int
	Size       int
	Count      int
	Dim        int
	Seed       int64
	Method     anchorann.Method
	Iterations int
	// SourceSHA256 identifies the clustering artifact the bundle was built
	// from; a re-run of preprocessing requires regenerating the bundle.
	SourceSHA256 [32]byte
	Params       nsg.Params // Workers is not stored and is always 0
	Graph        nsg.Graph
	ClusterOf    []uint32
	Vectors      []float32
	PIR          PIRParams
}

type diskHeader struct {
	Magic         [8]byte
	Version       uint32
	N             uint64
	Size          uint32
	Count         uint32
	Dim           uint32
	Seed          int64
	Method        uint32
	Iterations    uint32
	SourceSHA256  [32]byte
	KNN           uint32
	L             uint32
	R             uint32
	CNSG          uint32
	Entry         uint32
	Edges         uint64
	PIREntryCount uint64
	PIREntryBytes uint64
	PIRBatchSize  uint32
	Reserved      uint32
}

// HeaderBytes is the fixed header length on disk.
var HeaderBytes = binary.Size(diskHeader{})

func methodCode(m anchorann.Method) uint32 {
	switch m {
	case anchorann.Recursive:
		return 1
	case anchorann.Global:
		return 2
	case anchorann.Common:
		return 3
	}
	return 0
}

func methodOf(c uint32) (anchorann.Method, bool) {
	switch c {
	case 1:
		return anchorann.Recursive, true
	case 2:
		return anchorann.Global, true
	case 3:
		return anchorann.Common, true
	}
	return "", false
}

// FromArtifact builds the anchor NSG over float32-rounded anchors of a
// validated clustering artifact. Anchor ID a maps to cluster ID a (the
// artifact's shared index). sourceSHA256 is the artifact file checksum.
func FromArtifact(a *anchorann.Artifact, sourceSHA256 [32]byte, p nsg.Params, pir PIRParams) (*Bundle, nsg.Stats, error) {
	if err := a.Validate(); err != nil {
		return nil, nsg.Stats{}, fmt.Errorf("bundle source: %w", err)
	}
	vectors := make([]float32, a.Count*a.Dim)
	for c, anchor := range a.Anchors {
		for j, x := range anchor {
			vectors[c*a.Dim+j] = float32(x)
		}
	}
	g, st, err := nsg.Build(vectors, a.Dim, p)
	if err != nil {
		return nil, st, fmt.Errorf("build anchor NSG: %w", err)
	}
	clusterOf := make([]uint32, a.Count)
	for c := range clusterOf {
		clusterOf[c] = uint32(c)
	}
	p.Workers = 0
	b := &Bundle{
		Version: FormatVersion, N: a.N, Size: a.Size, Count: a.Count, Dim: a.Dim,
		Seed: a.Seed, Method: a.Method, Iterations: a.Iterations, SourceSHA256: sourceSHA256,
		Params: p, Graph: *g, ClusterOf: clusterOf, Vectors: vectors, PIR: pir,
	}
	if err := b.Validate(Expect{}); err != nil {
		return nil, st, err
	}
	return b, st, nil
}

// Expect lists values a client requires of a bundle; zero fields are ignored.
type Expect struct {
	Count int
	Dim   int
	Size  int
}

// Validate checks the shape (node count = C = N/s, dimension), parameters,
// navigating node range, neighbor ID range, absence of self-loops and
// duplicate edges, finite vectors, the anchor -> cluster permutation, PIR
// parameter consistency, and that every node is reachable from the entry.
func (b *Bundle) Validate(want Expect) error {
	if b == nil {
		return fmt.Errorf("nil bundle")
	}
	if b.Version != FormatVersion {
		return fmt.Errorf("unsupported bundle version=%d", b.Version)
	}
	if b.Count <= 0 || b.Size <= 0 || b.Dim <= 0 || b.N != b.Count*b.Size || uint64(b.Count) > math.MaxUint32 {
		return fmt.Errorf("invalid bundle shape: N=%d s=%d C=%d dimension=%d", b.N, b.Size, b.Count, b.Dim)
	}
	if (want.Count != 0 && want.Count != b.Count) || (want.Dim != 0 && want.Dim != b.Dim) || (want.Size != 0 && want.Size != b.Size) {
		return fmt.Errorf("bundle C=%d dimension=%d s=%d, expected C=%d dimension=%d s=%d (0 = any)", b.Count, b.Dim, b.Size, want.Count, want.Dim, want.Size)
	}
	if _, ok := methodOf(methodCode(b.Method)); !ok || b.Iterations <= 0 {
		return fmt.Errorf("invalid bundle source: method=%q iterations=%d", b.Method, b.Iterations)
	}
	if err := b.Params.Validate(); err != nil {
		return err
	}
	if len(b.Vectors) != b.Count*b.Dim || len(b.Graph.Neighbors) != b.Count || len(b.ClusterOf) != b.Count {
		return fmt.Errorf("bundle sections: vectors=%d neighbor lists=%d mapping=%d for C=%d dimension=%d", len(b.Vectors), len(b.Graph.Neighbors), len(b.ClusterOf), b.Count, b.Dim)
	}
	if int(b.Graph.Entry) >= b.Count {
		return fmt.Errorf("entry=%d outside 0..%d", b.Graph.Entry, b.Count-1)
	}
	for i, x := range b.Vectors {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return fmt.Errorf("nonfinite anchor=%d coordinate=%d", i/b.Dim, i%b.Dim)
		}
	}
	seen := make([]bool, b.Count)
	for a, c := range b.ClusterOf {
		if int(c) >= b.Count || seen[c] {
			return fmt.Errorf("anchor=%d maps to cluster=%d: mapping must be a permutation of 0..%d", a, c, b.Count-1)
		}
		seen[c] = true
	}
	last := make([]int, b.Count)
	for i := range last {
		last[i] = -1
	}
	for v, adj := range b.Graph.Neighbors {
		for _, u := range adj {
			if int(u) >= b.Count {
				return fmt.Errorf("node=%d neighbor=%d outside 0..%d", v, u, b.Count-1)
			}
			if int(u) == v || last[u] == v {
				return fmt.Errorf("node=%d has self-loop or duplicate neighbor=%d", v, u)
			}
			last[u] = v
		}
	}
	if u := b.Graph.Unreachable(); u != 0 {
		return fmt.Errorf("%d of %d nodes unreachable from entry=%d", u, b.Count, b.Graph.Entry)
	}
	if b.PIR.IsSet() {
		if b.PIR.EntryCount != uint64(b.Count) || b.PIR.EntryBytes == 0 || b.PIR.EntryBytes%8 != 0 {
			return fmt.Errorf("invalid PIR params: entries=%d (C=%d) entry bytes=%d (positive multiple of 8)", b.PIR.EntryCount, b.Count, b.PIR.EntryBytes)
		}
	}
	return nil
}

// Encode serializes a validated bundle as little-endian sections followed by
// the SHA-256 of all preceding bytes. Equal bundles encode to equal bytes.
func Encode(b *Bundle) ([]byte, error) {
	if err := b.Validate(Expect{}); err != nil {
		return nil, fmt.Errorf("encode bundle: %w", err)
	}
	h := diskHeader{
		Magic: magic, Version: b.Version, N: uint64(b.N), Size: uint32(b.Size), Count: uint32(b.Count),
		Dim: uint32(b.Dim), Seed: b.Seed, Method: methodCode(b.Method), Iterations: uint32(b.Iterations),
		SourceSHA256: b.SourceSHA256, KNN: uint32(b.Params.KNN), L: uint32(b.Params.L), R: uint32(b.Params.R),
		CNSG: uint32(b.Params.CNSG), Entry: b.Graph.Entry, Edges: uint64(b.Graph.Edges()),
		PIREntryCount: b.PIR.EntryCount, PIREntryBytes: b.PIR.EntryBytes, PIRBatchSize: b.PIR.BatchSize,
	}
	var buf bytes.Buffer
	buf.Grow(int(encodedBytes(uint64(b.Count), uint64(b.Dim), h.Edges)))
	degrees := make([]uint32, b.Count)
	for v, adj := range b.Graph.Neighbors {
		degrees[v] = uint32(len(adj))
	}
	for _, part := range []any{h, b.ClusterOf, b.Vectors, degrees} {
		if err := binary.Write(&buf, binary.LittleEndian, part); err != nil {
			return nil, fmt.Errorf("encode bundle: %w", err)
		}
	}
	for _, adj := range b.Graph.Neighbors {
		if err := binary.Write(&buf, binary.LittleEndian, adj); err != nil {
			return nil, fmt.Errorf("encode bundle: %w", err)
		}
	}
	sum := sha256.Sum256(buf.Bytes())
	buf.Write(sum[:])
	return buf.Bytes(), nil
}

func encodedBytes(count, dim, edges uint64) uint64 {
	return uint64(HeaderBytes) + 4*count + 4*count*dim + 4*count + 4*edges + sha256.Size
}

// Checksum returns the hex SHA-256 of the whole encoded bundle, which is what
// the distribution service announces.
func Checksum(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// Decode verifies the trailing SHA-256 before parsing, checks the exact
// encoded length implied by the header, and then runs Validate(want).
func Decode(data []byte, want Expect) (*Bundle, error) {
	if len(data) < HeaderBytes+sha256.Size {
		return nil, fmt.Errorf("bundle length=%d shorter than header and checksum", len(data))
	}
	body, trailer := data[:len(data)-sha256.Size], data[len(data)-sha256.Size:]
	if sum := sha256.Sum256(body); !bytes.Equal(sum[:], trailer) {
		return nil, fmt.Errorf("bundle checksum mismatch: stored=%x computed=%x", trailer, sum)
	}
	var h diskHeader
	if err := binary.Read(bytes.NewReader(body[:HeaderBytes]), binary.LittleEndian, &h); err != nil {
		return nil, fmt.Errorf("read bundle header: %w", err)
	}
	if h.Magic != magic || h.Version != FormatVersion {
		return nil, fmt.Errorf("unsupported bundle magic=%q version=%d", h.Magic, h.Version)
	}
	if h.Count == 0 || h.Dim == 0 || h.Edges > math.MaxInt32 || uint64(h.Dim) > math.MaxInt32/uint64(h.Count) {
		return nil, fmt.Errorf("invalid bundle header: C=%d dimension=%d edges=%d", h.Count, h.Dim, h.Edges)
	}
	if want := encodedBytes(uint64(h.Count), uint64(h.Dim), h.Edges); uint64(len(data)) != want {
		return nil, fmt.Errorf("bundle length=%d expected=%d for C=%d dimension=%d edges=%d", len(data), want, h.Count, h.Dim, h.Edges)
	}
	m, ok := methodOf(h.Method)
	if !ok {
		return nil, fmt.Errorf("invalid bundle method code=%d", h.Method)
	}
	count, dim := int(h.Count), int(h.Dim)
	b := &Bundle{
		Version: h.Version, N: int(h.N), Size: int(h.Size), Count: count, Dim: dim, Seed: h.Seed,
		Method: m, Iterations: int(h.Iterations), SourceSHA256: h.SourceSHA256,
		Params: nsg.Params{KNN: int(h.KNN), L: int(h.L), R: int(h.R), CNSG: int(h.CNSG)},
		Graph:  nsg.Graph{Entry: h.Entry, Neighbors: make([][]uint32, count)},
		PIR:    PIRParams{EntryCount: h.PIREntryCount, EntryBytes: h.PIREntryBytes, BatchSize: h.PIRBatchSize},
	}
	if h.N != uint64(b.N) {
		return nil, fmt.Errorf("bundle N=%d exceeds address space", h.N)
	}
	b.ClusterOf = make([]uint32, count)
	b.Vectors = make([]float32, count*dim)
	degrees := make([]uint32, count)
	edges := make([]uint32, h.Edges)
	r := bytes.NewReader(body[HeaderBytes:])
	for _, part := range []any{b.ClusterOf, b.Vectors, degrees, edges} {
		if err := binary.Read(r, binary.LittleEndian, part); err != nil {
			return nil, fmt.Errorf("read bundle sections: %w", err)
		}
	}
	var off uint64
	for v, d := range degrees {
		if uint64(d) > h.Edges-off {
			return nil, fmt.Errorf("node=%d degree=%d exceeds remaining edges=%d", v, d, h.Edges-off)
		}
		b.Graph.Neighbors[v] = edges[off : off+uint64(d) : off+uint64(d)]
		off += uint64(d)
	}
	if off != h.Edges {
		return nil, fmt.Errorf("degree sum=%d does not match edges=%d", off, h.Edges)
	}
	if err := b.Validate(want); err != nil {
		return nil, fmt.Errorf("validate bundle: %w", err)
	}
	return b, nil
}

// Save encodes b and atomically replaces path. It returns the encoded bytes.
func Save(path string, b *Bundle) ([]byte, error) {
	data, err := Encode(b)
	if err != nil {
		return nil, err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".anchor-bundle-*")
	if err != nil {
		return nil, fmt.Errorf("create bundle: %w", err)
	}
	defer os.Remove(f.Name())
	defer f.Close()
	if _, err := f.Write(data); err != nil {
		return nil, fmt.Errorf("write bundle: %w", err)
	}
	if err := f.Sync(); err != nil {
		return nil, fmt.Errorf("sync bundle: %w", err)
	}
	if err := f.Close(); err != nil {
		return nil, fmt.Errorf("close bundle: %w", err)
	}
	if err := os.Rename(f.Name(), path); err != nil {
		return nil, fmt.Errorf("replace bundle: %w", err)
	}
	return data, nil
}

// Load reads and decodes a bundle file.
func Load(path string, want Expect) (*Bundle, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read bundle: %w", err)
	}
	return Decode(data, want)
}
