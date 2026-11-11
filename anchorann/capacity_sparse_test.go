package anchorann

import (
	"context"
	"math"
	"reflect"
	"testing"
)

func TestSparseTransportAgainstFullSAP(t *testing.T) {
	for trial := 0; trial < 15; trial++ {
		d := testDataset(64, 3)
		centers := make([][]float64, 32)
		for c := range centers {
			centers[c] = mean(d, []uint32{uint32((c + trial) % d.N)})
		}
		_, _, want, err := OptimalAssignment(context.Background(), d, centers, 2)
		if err != nil {
			t.Fatal(err)
		}
		owners, p, got, err := pricedAssignment(context.Background(), d, centers, 2, 4)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(got.Objective-want.Objective) > 1e-9 {
			t.Fatalf("objective got=%g want=%g", got.Objective, want.Objective)
		}
		if _, err := VerifyAssignment(context.Background(), d, centers, 2, owners, p); err != nil {
			t.Fatal(err)
		}
	}
}

func TestSparseNeedsDisplacementAndInfeasibleGraph(t *testing.T) {
	g := transportGraph{{{0, 16}, {1, 36}}, {{0, 0}, {1, 100}}}
	owners, p, err := sparseTransport(context.Background(), g, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if owners[0] != 1 || owners[1] != 0 {
		t.Fatalf("no residual displacement: %v", owners)
	}
	d := Dataset{N: 2, Dim: 1, Values: []float32{4, 0}}
	if _, err := VerifyAssignment(context.Background(), d, [][]float64{{0}, {10}}, 1, owners, p); err != nil {
		t.Fatal(err)
	}
	if _, _, err := sparseTransport(context.Background(), transportGraph{{{0, 0}}, {{0, 1}}}, 2, 1); err == nil {
		t.Fatal("accepted infeasible network")
	}
}

func TestParallelDistancePaddingAndRepeatedVectors(t *testing.T) {
	for _, dim := range []int{1, 2, 128} {
		for _, c := range []int{1, 3, 5, 31} {
			d := testDataset(3, dim)
			centers := make([][]float64, c)
			for i := range centers {
				centers[i] = mean(d, []uint32{uint32(i % d.N)})
			}
			ids := []int{0, 2, 1}
			table := newDistanceTable(centers)
			got := table.rows(context.Background(), d, ids, 20)
			for row, id := range ids {
				for i, m := range centers {
					want := distance(d.Values[id*dim:(id+1)*dim], m)
					if got[row*table.padded+i] != want {
						t.Fatalf("dim=%d count=%d row=%d c=%d mismatch", dim, c, row, i)
					}
				}
			}
		}
	}
}

func TestPricedAssignmentSmallExhaustiveAndDegenerate(t *testing.T) {
	for _, d := range []Dataset{testDataset(6, 2), {N: 6, Dim: 2, Values: make([]float32, 12)}} {
		centers := [][]float64{{0, 1}, {2, 0}, {-1, 1}}
		owners, p, cert, err := pricedAssignment(context.Background(), d, centers, 2, 20)
		if err != nil {
			t.Fatal(err)
		}
		want := exhaustiveCost(d, centers, 2)
		if math.Abs(cert.Objective-want) > 1e-9 {
			t.Fatalf("got=%g want=%g", cert.Objective, want)
		}
		if cert.CheckedEdges != 18 {
			t.Fatal("not all edges audited")
		}
		if _, err := VerifyAssignment(context.Background(), d, centers, 2, owners, p); err != nil {
			t.Fatal(err)
		}
	}
}

func TestWarmSparseRetainsCertifiedRows(t *testing.T) {
	d := testDataset(120, 4)
	centers := make([][]float64, 30)
	for c := range centers {
		centers[c] = mean(d, []uint32{uint32(c * 2)})
	}
	g := make(transportGraph, d.N)
	for row := range g {
		g[row] = []transportEdge{{int32(row % 30), distance(d.Values[row*4:(row+1)*4], centers[row%30])}}
	}
	owners, p, err := sparseTransport(context.Background(), g, 30, 4)
	if err != nil {
		t.Fatal(err)
	}
	warm := append([]int(nil), owners...)
	for row := range g {
		base := g[row][0].cost
		for c, m := range centers {
			if c == owners[row] {
				continue
			}
			cost := distance(d.Values[row*4:(row+1)*4], m)
			g[row] = append(g[row], transportEdge{int32(c), cost})
			if cost-base+p[owners[row]]-p[c] < 0 {
				warm[row] = -1
			}
		}
	}
	got, dual, err := sparseTransportWarm(context.Background(), g, 30, 4, warm, p)
	if err != nil {
		t.Fatal(err)
	}
	cert, err := VerifyAssignment(context.Background(), d, centers, 4, got, dual)
	if err != nil {
		t.Fatal(err)
	}
	_, _, want, err := OptimalAssignment(context.Background(), d, centers, 4)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(cert.Objective-want.Objective) > 1e-8 {
		t.Fatalf("warm objective=%g optimum=%g", cert.Objective, want.Objective)
	}
}

func TestDualInitializerThenSAP(t *testing.T) {
	d := testDataset(64, 3)
	centers := make([][]float64, 16)
	g := make(transportGraph, d.N)
	for c := range centers {
		centers[c] = mean(d, []uint32{uint32(c)})
	}
	for row := range g {
		for c, m := range centers {
			g[row] = append(g[row], transportEdge{int32(c), distance(d.Values[row*3:(row+1)*3], m)})
		}
	}
	warm, p, err := auctionWarmStart(context.Background(), g, 16, 4)
	if err != nil {
		t.Fatal(err)
	}
	owners, p, err := sparseTransportWarm(context.Background(), g, 16, 4, warm, p)
	if err != nil {
		t.Fatal(err)
	}
	cert, err := VerifyAssignment(context.Background(), d, centers, 4, owners, p)
	if err != nil {
		t.Fatal(err)
	}
	_, _, want, err := OptimalAssignment(context.Background(), d, centers, 4)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(cert.Objective-want.Objective) > 1e-8 {
		t.Fatalf("got=%g optimum=%g", cert.Objective, want.Objective)
	}
}

func TestLargeParallelLloydSSEAndFullCertificate(t *testing.T) {
	d := testDataset(320, 5)
	o := DefaultOptions()
	o.Size = 16
	o.Method = Common
	o.Workers = 4
	o.Iterations = 3
	previous := math.Inf(1)
	steps := 0
	a, summary, err := ClusterWithProgress(context.Background(), d, o, func(s Step) error {
		steps++
		if s.SSE > previous+1e-8 {
			t.Fatalf("SSE increase=%g -> %g", previous, s.SSE)
		}
		previous = s.SSE
		if s.Certificate == nil || s.Certificate.CheckedEdges != 6400 || s.Certificate.MaxRelativeViolation > assignmentTolerance {
			t.Fatalf("invalid certificate %+v", s.Certificate)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if summary.CompletedIterations != steps || a.Validate() != nil {
		t.Fatal("invalid result")
	}
	b, err := Cluster(d, o)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatal("nonreproducible parallel optimized Lloyd")
	}
}
