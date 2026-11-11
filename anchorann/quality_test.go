package anchorann

import (
	"math"
	"sort"
	"testing"
)

func TestQualityMetrics(t *testing.T) {
	d := Dataset{N: 4, Dim: 1, Values: []float32{0, 2, 10, 12}}
	a := &Artifact{Version: FormatVersion, N: 4, Size: 2, Count: 2, Dim: 1, Method: Recursive, Iterations: 1, Members: [][]uint32{{0, 1}, {2, 3}}, Anchors: [][]float64{{1}, {11}}}
	if sse, err := SSE(d, a); err != nil || sse != 4 {
		t.Fatalf("SSE=%g %v; want 4", sse, err)
	}
	q := Dataset{N: 2, Dim: 1, Values: []float32{0, 12}}
	for _, tc := range []struct {
		k    int
		want float64
	}{{1, 1}, {2, 1}, {3, 2.0 / 3}, {4, 0.5}} {
		got, err := TopKContainment(d, a, q, tc.k)
		if err != nil || got != tc.want {
			t.Fatalf("k=%d containment=%g error=%v want=%g", tc.k, got, err, tc.want)
		}
	}
}

func TestContainmentMatchesSortedGroundTruth(t *testing.T) {
	d, queries := testDataset(30, 3), testDataset(5, 3)
	a := testCluster(t, d, 2, Recursive, 2)
	const k = 10
	owners := make([]int, d.N)
	for c, ids := range a.Members {
		for _, id := range ids {
			owners[id] = c
		}
	}
	hits := 0
	for i := 0; i < queries.N; i++ {
		q := queries.Values[i*queries.Dim : (i+1)*queries.Dim]
		nearest, best := 0, math.Inf(1)
		for c, center := range a.Anchors {
			if cost := distance(q, center); cost < best {
				nearest, best = c, cost
			}
		}
		ids := make([]scoredID, d.N)
		for id := range ids {
			var cost float64
			for j, x := range q {
				diff := float64(x) - float64(d.Values[id*d.Dim+j])
				cost += diff * diff
			}
			ids[id] = scoredID{uint32(id), cost}
		}
		sort.Slice(ids, func(i, j int) bool { return less(ids[i], ids[j]) })
		for _, x := range ids[:k] {
			if owners[x.id] == nearest {
				hits++
			}
		}
	}
	want := float64(hits) / float64(queries.N*k)
	if got, err := TopKContainment(d, a, queries, k); err != nil || got != want {
		t.Fatalf("containment=%g error=%v want=%g", got, err, want)
	}
}

func TestContainmentTieBreaks(t *testing.T) {
	d := Dataset{N: 4, Dim: 1, Values: make([]float32, 4)}
	a := &Artifact{Version: FormatVersion, N: 4, Size: 2, Count: 2, Dim: 1, Method: Recursive, Iterations: 1, Members: [][]uint32{{2, 3}, {0, 1}}, Anchors: [][]float64{{0}, {0}}}
	q := Dataset{N: 1, Dim: 1, Values: []float32{0}}
	// Equal centroid distances choose cluster 0; equal point distances choose IDs 0,1.
	if got, err := TopKContainment(d, a, q, 2); err != nil || got != 0 {
		t.Fatalf("tie-break containment=%g %v", got, err)
	}
}

func TestQualityValidation(t *testing.T) {
	d := testDataset(4, 2)
	a := testCluster(t, d, 2, Recursive, 1)
	q := testDataset(1, 2)
	for _, k := range []int{0, -1, 5} {
		if _, err := TopKContainment(d, a, q, k); err == nil {
			t.Fatalf("accepted k=%d", k)
		}
	}
	if _, err := TopKContainment(d, a, testDataset(1, 3), 1); err == nil {
		t.Fatal("accepted query dimension mismatch")
	}
	if _, err := TopKContainment(d, a, Dataset{}, 1); err == nil {
		t.Fatal("accepted empty queries")
	}
	for _, bad := range []Dataset{Dataset{}, testDataset(6, 2), testDataset(4, 3)} {
		if _, err := SSE(bad, a); err == nil {
			t.Fatal("SSE accepted mismatched dataset")
		}
	}
	if _, err := SSE(d, nil); err == nil {
		t.Fatal("SSE accepted nil artifact")
	}
	if _, err := TopKContainment(d, nil, q, 1); err == nil {
		t.Fatal("containment accepted nil artifact")
	}
}

func TestContainmentUnionsAndCachedGT(t *testing.T) {
	// Nine equal centroids rank strictly by cluster ID. Reversed GT order must
	// still count the union, not just one cluster or repeated point IDs.
	d := Dataset{N: 18, Dim: 1, Values: make([]float32, 18)}
	a := &Artifact{Version: FormatVersion, N: 18, Size: 2, Count: 9, Dim: 1, Method: Common, Iterations: 1, Members: make([][]uint32, 9), Anchors: make([][]float64, 9)}
	for c := 0; c < 9; c++ {
		a.Members[c] = []uint32{uint32(c * 2), uint32(c*2 + 1)}
		a.Anchors[c] = []float64{0}
	}
	q := Dataset{N: 1, Dim: 1, Values: []float32{0}}
	gt, err := ExactGroundTruth(d, q, 18)
	if err != nil {
		t.Fatal(err)
	}
	for i, j := 0, 17; i < j; i, j = i+1, j-1 {
		gt.IDs[0][i], gt.IDs[0][j] = gt.IDs[0][j], gt.IDs[0][i]
	}
	if err := ValidateGroundTruth(d, q, gt); err != nil {
		t.Fatal(err)
	}
	rates, err := Containment(a, q, gt)
	if err != nil || rates.At1 != 2.0/18 || rates.At4 != 8.0/18 || rates.At8 != 16.0/18 {
		t.Fatalf("unions=%+v %v", rates, err)
	}
	changed := d
	changed.Values = append([]float32(nil), d.Values...)
	changed.Values[0] = 1
	if err := ValidateGroundTruth(changed, q, gt); err == nil {
		t.Fatal("accepted changed input")
	}
	if err := ValidateGroundTruth(d, Dataset{N: 1, Dim: 1, Values: []float32{1}}, gt); err == nil {
		t.Fatal("accepted changed query")
	}
	if err := ValidateGroundTruth(Dataset{N: 16, Dim: 1, Values: d.Values[:16]}, q, gt); err == nil {
		t.Fatal("accepted subset GT")
	}
	gt.IDs[0][0] = gt.IDs[0][1]
	if _, err := Containment(a, q, gt); err == nil {
		t.Fatal("accepted duplicate GT")
	}
}
