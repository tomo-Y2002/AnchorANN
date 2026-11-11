package anchorann

import (
	"context"
	"math"
	"math/rand"
	"path/filepath"
	"reflect"
	"testing"
)

func exhaustiveCost(d Dataset, centers [][]float64, size int) float64 {
	counts := make([]int, len(centers))
	best := math.Inf(1)
	var search func(int, float64)
	search = func(row int, total float64) {
		if row == d.N {
			best = math.Min(best, total)
			return
		}
		for c, m := range centers {
			if counts[c] < size {
				counts[c]++
				search(row+1, total+distance(d.Values[row*d.Dim:(row+1)*d.Dim], m))
				counts[c]--
			}
		}
	}
	search(0, 0)
	return best
}

func TestOptimalAssignmentExhaustive(t *testing.T) {
	rng := rand.New(rand.NewSource(711))
	for trial := 0; trial < 100; trial++ {
		d := testDataset(6, 2)
		for i := range d.Values {
			d.Values[i] = float32(rng.Intn(9)-4) / 3
		}
		centers := make([][]float64, 3)
		for c := range centers {
			centers[c] = []float64{rng.Float64()*4 - 2, rng.Float64()*4 - 2}
		}
		owners, potentials, cert, err := OptimalAssignment(context.Background(), d, centers, 2)
		if err != nil {
			t.Fatalf("trial=%d: %v", trial, err)
		}
		best := exhaustiveCost(d, centers, 2)
		if math.Abs(cert.Objective-best) > 1e-9 {
			t.Fatalf("trial=%d got=%g optimum=%g", trial, cert.Objective, best)
		}
		if cert.CheckedEdges != 18 || cert.MaxRelativeViolation > assignmentTolerance {
			t.Fatalf("unchecked certificate: %+v", cert)
		}
		if _, err := VerifyAssignment(context.Background(), d, centers, 2, owners, potentials); err != nil {
			t.Fatal(err)
		}
	}
}

func TestGreedyCapacityCompetition(t *testing.T) {
	// Greedy sends x=4 to center 0, forcing x=0 to center 10: cost 116.
	// The augmenting path moves x=4 to center 10, reaching optimum 36.
	d := Dataset{N: 2, Dim: 1, Values: []float32{4, 0}}
	centers := [][]float64{{0}, {10}}
	greedy, err := greedyAssignment(context.Background(), d, centers, 1, []int{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	_, _, cert, err := OptimalAssignment(context.Background(), d, centers, 1)
	if err != nil {
		t.Fatal(err)
	}
	var cost float64
	for id, c := range greedy {
		cost += distance(d.Values[id:id+1], centers[c])
	}
	if cost != 116 || cert.Objective != 36 {
		t.Fatalf("greedy=%g optimum=%g", cost, cert.Objective)
	}
	if _, err := VerifyAssignment(context.Background(), d, centers, 1, greedy, []float64{0, 0}); err == nil {
		t.Fatal("certified suboptimal greedy")
	}
}

func TestAssignmentCertificateToleranceAndErrors(t *testing.T) {
	d := Dataset{N: 2, Dim: 1, Values: []float32{0, 1}}
	centers := [][]float64{{0}, {1}}
	owners := []int{0, 1}
	if _, err := VerifyAssignment(context.Background(), d, centers, 1, owners, []float64{0, 1 + 1e-11}); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyAssignment(context.Background(), d, centers, 1, owners, []float64{0, 1 + 1e-6}); err == nil {
		t.Fatal("accepted dual violation")
	}
	for _, bad := range [][]int{nil, {-1, 1}, {0, 0}, {0, 2}} {
		if _, err := VerifyAssignment(context.Background(), d, centers, 1, bad, []float64{0, 0}); err == nil {
			t.Fatalf("accepted owners=%v", bad)
		}
	}
	if _, err := VerifyAssignment(context.Background(), d, centers, 1, owners, []float64{math.NaN(), 0}); err == nil {
		t.Fatal("accepted NaN dual")
	}
	for _, bad := range [][][]float64{nil, {{0}, {1, 2}}, {{0}, {math.Inf(1)}}} {
		if _, _, _, err := OptimalAssignment(context.Background(), d, bad, 1); err == nil {
			t.Fatal("accepted bad centers")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, _, err := OptimalAssignment(ctx, d, centers, 1); err == nil {
		t.Fatal("ignored cancellation")
	}
}

func TestCommonLloydProgressAndMonotonicity(t *testing.T) {
	for seed := int64(1); seed <= 10; seed++ {
		d := testDataset(24, 3)
		o := DefaultOptions()
		o.Method = Common
		o.Size = 4
		o.Seed = seed
		var steps []Step
		a, summary, err := ClusterWithProgress(context.Background(), d, o, func(s Step) error { steps = append(steps, s); return nil })
		if err != nil {
			t.Fatal(err)
		}
		if summary.CompletedIterations != len(steps) || len(steps) == 0 {
			t.Fatalf("summary=%+v history=%d", summary, len(steps))
		}
		for i, s := range steps {
			if s.Iteration != i+1 || s.Certificate == nil || s.Certificate.CheckedEdges != 144 {
				t.Fatalf("invalid step=%+v", s)
			}
			if i > 0 && s.SSE > steps[i-1].SSE+1e-8 {
				t.Fatalf("SSE increased %g -> %g", steps[i-1].SSE, s.SSE)
			}
		}
		b, err := Cluster(d, o)
		if err != nil || !reflect.DeepEqual(a, b) {
			t.Fatalf("observer changed output: %v", err)
		}
		path := filepath.Join(t.TempDir(), "common.anchor")
		if err := Save(path, a); err != nil {
			t.Fatal(err)
		}
		loaded, err := Load(path)
		if err != nil || !reflect.DeepEqual(a, loaded) {
			t.Fatalf("roundtrip: %v", err)
		}
	}
}

func TestLloydStopsAndCompletedHistory(t *testing.T) {
	for _, method := range []Method{Global, Common} {
		o := DefaultOptions()
		o.Method = method
		o.Size = 2
		o.Iterations = 1
		_, summary, err := ClusterWithProgress(context.Background(), testDataset(4, 1), o, nil)
		if err != nil || summary.StopReason != "iteration_limit" || summary.CompletedIterations != 1 {
			t.Fatalf("limit: %+v %v", summary, err)
		}
		d := Dataset{N: 8, Dim: 1, Values: make([]float32, 8)}
		o.Iterations = 6
		o.Size = 8
		_, summary, err = ClusterWithProgress(context.Background(), d, o, nil)
		if err != nil || summary.StopReason != "unchanged_membership" || summary.CompletedIterations != 2 {
			t.Fatalf("convergence: %+v %v", summary, err)
		}
		o.Size = 2
		ctx, cancel := context.WithCancel(context.Background())
		n := 0
		_, summary, err = ClusterWithProgress(ctx, testDataset(12, 2), o, func(s Step) error { n++; cancel(); return nil })
		if err == nil || n != 1 || summary.CompletedIterations != 1 || summary.StopReason != "cancelled" {
			t.Fatalf("partial: %+v steps=%d %v", summary, n, err)
		}
	}
}

func TestAssignmentScalesAndDuplicateCenters(t *testing.T) {
	for _, scale := range []float32{1e-9, 1, 1e9} {
		d := Dataset{N: 6, Dim: 1, Values: []float32{0, scale, 2 * scale, 3 * scale, 4 * scale, 5 * scale}}
		for _, centers := range [][][]float64{{{0}, {0}, {0}}, {{0}, {float64(scale)}, {5 * float64(scale)}}} {
			owners, dual, cert, err := OptimalAssignment(context.Background(), d, centers, 2)
			if err != nil {
				t.Fatal(err)
			}
			best := exhaustiveCost(d, centers, 2)
			if math.Abs(cert.Objective-best) > 1e-10*math.Max(1, best) {
				t.Fatalf("scale=%g got=%g best=%g", scale, cert.Objective, best)
			}
			owners2, dual2, cert2, err := OptimalAssignment(context.Background(), d, centers, 2)
			if err != nil || !reflect.DeepEqual(owners, owners2) || !reflect.DeepEqual(dual, dual2) || cert != cert2 {
				t.Fatal("not reproducible")
			}
		}
	}
}
