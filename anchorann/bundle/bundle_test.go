package bundle

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"go/parser"
	"go/token"
	"math/rand"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"privatesearch/anchorann"
	"privatesearch/anchorann/nsg"
)

func testArtifact(count, size, dim int, seed int64) *anchorann.Artifact {
	rng := rand.New(rand.NewSource(seed))
	a := &anchorann.Artifact{Version: anchorann.FormatVersion, N: count * size, Size: size, Count: count, Dim: dim, Seed: seed, Method: anchorann.Common, Iterations: 2}
	perm := rng.Perm(a.N)
	for c := 0; c < count; c++ {
		ids := make([]uint32, size)
		for i := range ids {
			ids[i] = uint32(perm[c*size+i])
		}
		anchor := make([]float64, dim)
		for j := range anchor {
			anchor[j] = rng.Float64() * 255
		}
		a.Members = append(a.Members, ids)
		a.Anchors = append(a.Anchors, anchor)
	}
	return a
}

func testParams() nsg.Params { return nsg.Params{KNN: 10, L: 20, R: 8, CNSG: 40, Workers: 3} }

func testBundle(t *testing.T, count int) (*Bundle, []byte) {
	t.Helper()
	b, _, err := FromArtifact(testArtifact(count, 4, 6, 1), sha256.Sum256([]byte("src")), testParams(), PIRParams{})
	if err != nil {
		t.Fatal(err)
	}
	data, err := Encode(b)
	if err != nil {
		t.Fatal(err)
	}
	return b, data
}

// resign replaces the trailing checksum so tests reach the structural checks.
func resign(data []byte) []byte {
	out := append([]byte(nil), data...)
	sum := sha256.Sum256(out[:len(out)-sha256.Size])
	copy(out[len(out)-sha256.Size:], sum[:])
	return out
}

// fieldOffset returns a header field's packed on-disk offset.
func fieldOffset(t *testing.T, name string) int {
	t.Helper()
	typ := reflect.TypeOf(diskHeader{})
	off := 0
	for i := 0; i < typ.NumField(); i++ {
		f := typ.Field(i)
		if f.Name == name {
			return off
		}
		off += int(f.Type.Size())
	}
	t.Fatalf("no header field %s", name)
	return 0
}

func TestRoundTripSaveLoad(t *testing.T) {
	b, data := testBundle(t, 50)
	if HeaderBytes != 132 {
		t.Fatalf("header bytes=%d; the documented version 1 layout is 132", HeaderBytes)
	}
	if len(data) != int(encodedBytes(50, 6, uint64(b.Graph.Edges()))) {
		t.Fatalf("encoded length %d", len(data))
	}
	path := filepath.Join(t.TempDir(), "anchors.bundle")
	saved, err := Save(path, b)
	if err != nil || !bytes.Equal(saved, data) {
		t.Fatalf("save err=%v equal=%v", err, bytes.Equal(saved, data))
	}
	got, err := Load(path, Expect{Count: 50, Dim: 6, Size: 4})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, b) {
		t.Fatalf("round trip differs:\n%+v\n%+v", got, b)
	}
	if _, err := Load(filepath.Join(t.TempDir(), "missing"), Expect{}); err == nil {
		t.Fatal("missing file accepted")
	}
}

func TestDeterministicBundleBytes(t *testing.T) {
	_, a := testBundle(t, 60)
	p := testParams()
	p.Workers = 1
	b, _, err := FromArtifact(testArtifact(60, 4, 6, 1), sha256.Sum256([]byte("src")), p, PIRParams{})
	if err != nil {
		t.Fatal(err)
	}
	data, _ := Encode(b)
	if !bytes.Equal(a, data) {
		t.Fatal("same artifact produced different bundle bytes")
	}
}

func TestBundleContents(t *testing.T) {
	art := testArtifact(30, 4, 6, 2)
	pir := PIRParams{EntryCount: 30, EntryBytes: 128, BatchSize: 4}
	b, st, err := FromArtifact(art, [32]byte{9}, testParams(), pir)
	if err != nil {
		t.Fatal(err)
	}
	if b.N != 120 || b.Count != 30 || b.Size != 4 || b.Dim != 6 || b.Seed != 2 || b.Method != anchorann.Common || b.Iterations != 2 || b.SourceSHA256 != [32]byte{9} || b.PIR != pir || b.Params.Workers != 0 {
		t.Fatalf("metadata %+v", b)
	}
	for a, c := range b.ClusterOf {
		if c != uint32(a) {
			t.Fatalf("anchor %d -> cluster %d", a, c)
		}
		for j := 0; j < 6; j++ {
			if b.Vectors[a*6+j] != float32(art.Anchors[a][j]) {
				t.Fatal("anchor vectors not float32-rounded artifact anchors")
			}
		}
	}
	if st.Nodes != 30 || st.Edges != b.Graph.Edges() {
		t.Fatalf("stats %+v", st)
	}
	data, _ := Encode(b)
	got, err := Decode(data, Expect{})
	if err != nil || got.PIR != pir {
		t.Fatalf("PIR params lost: %+v %v", got, err)
	}
	if _, _, err := FromArtifact(&anchorann.Artifact{}, [32]byte{}, testParams(), PIRParams{}); err == nil {
		t.Fatal("invalid artifact accepted")
	}
}

func TestSingleCluster(t *testing.T) {
	b, data := testBundle(t, 1)
	if b.Graph.Entry != 0 || b.Graph.Edges() != 0 {
		t.Fatalf("C=1 graph %+v", b.Graph)
	}
	got, err := Decode(data, Expect{Count: 1})
	if err != nil {
		t.Fatal(err)
	}
	ix, err := NewIndex(got)
	if err != nil {
		t.Fatal(err)
	}
	ps, err := ix.Probe(make([]float32, 6), SearchParams{LSearch: 1, NProbe: 1, K: 4})
	if err != nil || len(ps) != 1 || ps[0].Anchor != 0 || ps[0].Cluster != 0 {
		t.Fatalf("C=1 probe %v %v", ps, err)
	}
	if _, err := ix.Probe(make([]float32, 6), SearchParams{LSearch: 2, NProbe: 2, K: 1}); err == nil {
		t.Fatal("n_probe > C accepted")
	}
}

func TestDecodeRejectsTampering(t *testing.T) {
	b, data := testBundle(t, 40)
	vecOff := HeaderBytes + 4*b.Count
	degOff := vecOff + 4*b.Count*b.Dim
	edgeOff := degOff + 4*b.Count
	put32 := func(off int, v uint32) []byte {
		out := append([]byte(nil), data...)
		binary.LittleEndian.PutUint32(out[off:], v)
		return resign(out)
	}
	flipped := append([]byte(nil), data...)
	flipped[vecOff+3] ^= 0x40
	cases := []struct {
		name string
		data []byte
		want Expect
		msg  string
	}{
		{"checksum mismatch", flipped, Expect{}, "checksum mismatch"},
		{"truncated", data[:len(data)-1], Expect{}, "checksum mismatch"},
		{"too short", data[:10], Expect{}, "shorter"},
		{"bad magic", func() []byte { d := append([]byte(nil), data...); d[0] = 'X'; return resign(d) }(), Expect{}, "magic"},
		{"neighbor out of range", put32(edgeOff, 40), Expect{}, "outside 0..39"},
		{"self loop", put32(edgeOff, 0), Expect{}, "self-loop"},
		{"entry out of range", put32(fieldOffset(t, "Entry"), 40), Expect{}, "entry=40"},
		{"header dimension", put32(fieldOffset(t, "Dim"), 7), Expect{}, "length="},
		{"header count", put32(fieldOffset(t, "Count"), 41), Expect{}, "length="},
		{"mapping not permutation", put32(HeaderBytes, 1), Expect{}, "permutation"},
		{"degree sum", put32(degOff, binary.LittleEndian.Uint32(data[degOff:])+1), Expect{}, "degree"},
		{"bad method", put32(fieldOffset(t, "Method"), 9), Expect{}, "method"},
		{"bad params", put32(fieldOffset(t, "R"), 0), Expect{}, "positive"},
		{"bad PIR", put32(fieldOffset(t, "PIREntryCount"), 39), Expect{}, "PIR"},
		{"expected dimension", data, Expect{Dim: 8}, "expected"},
		{"expected count", data, Expect{Count: 41}, "expected"},
		{"expected size", data, Expect{Size: 32}, "expected"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := Decode(c.data, c.want); err == nil || !strings.Contains(err.Error(), c.msg) {
				t.Fatalf("err=%v want %q", err, c.msg)
			}
		})
	}
}

func TestValidateRejectsStructuralErrors(t *testing.T) {
	mutate := func(f func(b *Bundle)) error {
		b, _ := testBundle(t, 20)
		f(b)
		return b.Validate(Expect{})
	}
	checks := map[string]func(b *Bundle){
		"duplicate edge": func(b *Bundle) {
			adj := b.Graph.Neighbors[0]
			b.Graph.Neighbors[0] = append(append([]uint32(nil), adj...), adj[0])
		},
		"unreachable": func(b *Bundle) {
			for v := range b.Graph.Neighbors {
				b.Graph.Neighbors[v] = nil
			}
		},
		"nonfinite": func(b *Bundle) { b.Vectors[5] = float32(inf()) },
		"version":   func(b *Bundle) { b.Version = 2 },
		"shape":     func(b *Bundle) { b.N++ },
		"sections":  func(b *Bundle) { b.Vectors = b.Vectors[:len(b.Vectors)-1] },
		"PIR bytes": func(b *Bundle) { b.PIR = PIRParams{EntryCount: 20, EntryBytes: 12} },
		"source":    func(b *Bundle) { b.Iterations = 0 },
	}
	for name, f := range checks {
		if err := mutate(f); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
	var nilBundle *Bundle
	if nilBundle.Validate(Expect{}) == nil {
		t.Error("nil bundle accepted")
	}
	if _, err := Encode(&Bundle{}); err == nil {
		t.Error("invalid bundle encoded")
	}
}

func inf() float64 { v, _ := strconv.ParseFloat("+Inf", 64); return v }

func TestSearchParams(t *testing.T) {
	cases := []struct {
		p  SearchParams
		ok bool
	}{
		{SearchParams{LSearch: 8, NProbe: 8, K: 10}, true},
		{SearchParams{LSearch: 4, NProbe: 8, K: 10}, false}, // L_search < n_probe
		{SearchParams{LSearch: 64, NProbe: 51, K: 1}, false},
		{SearchParams{LSearch: 8, NProbe: 2, K: 9}, false}, // k > n_probe*s
		{SearchParams{LSearch: 8, NProbe: 2, K: 0}, false},
		{SearchParams{LSearch: 0, NProbe: 0, K: 1}, false},
	}
	for _, c := range cases {
		if err := c.p.Validate(50, 4); (err == nil) != c.ok {
			t.Errorf("%+v err=%v", c.p, err)
		}
	}
}

func TestProbeMatchesExactWhenSearchCoversAll(t *testing.T) {
	b, _ := testBundle(t, 80)
	ix, err := NewIndex(b)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(5))
	for q := 0; q < 10; q++ {
		query := make([]float32, b.Dim)
		for j := range query {
			query[j] = rng.Float32() * 255
		}
		ps, err := ix.Probe(query, SearchParams{LSearch: b.Count, NProbe: 5, K: 3})
		if err != nil {
			t.Fatal(err)
		}
		exact := nsg.Exact(b.Vectors, b.Dim, query, 5)
		for i, p := range ps {
			if p.Anchor != exact[i].ID || p.Cluster != b.ClusterOf[p.Anchor] || p.Distance != exact[i].Distance {
				t.Fatalf("query %d probe %d = %+v want %+v", q, i, p, exact[i])
			}
		}
		rs, _, err := ix.Search(query, 7)
		if err != nil || len(rs) != 7 {
			t.Fatalf("search %d results err=%v", len(rs), err)
		}
	}
	if _, err := ix.Probe([]float32{1}, SearchParams{LSearch: 5, NProbe: 5, K: 1}); err == nil {
		t.Fatal("query dimension mismatch accepted")
	}
	if _, err := NewIndex(&Bundle{}); err == nil {
		t.Fatal("invalid bundle indexed")
	}
}

// Local search must not be able to talk to the server: the client-side
// packages may not import networking or the distribution RPC.
func TestLocalSearchHasNoNetworkImports(t *testing.T) {
	forbidden := []string{"net", "google.golang.org/grpc", "privatesearch/anchorann/bundlerpc", "privatesearch/rpc", "privatesearch/pirclient"}
	for _, dir := range []string{".", "../nsg"} {
		pkgs, err := parser.ParseDir(token.NewFileSet(), dir, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		for _, pkg := range pkgs {
			for name, f := range pkg.Files {
				if strings.HasSuffix(name, "_test.go") {
					continue
				}
				for _, imp := range f.Imports {
					path, _ := strconv.Unquote(imp.Path.Value)
					for _, bad := range forbidden {
						if path == bad || strings.HasPrefix(path, bad+"/") {
							t.Errorf("%s imports %s", name, path)
						}
					}
				}
			}
		}
	}
}
