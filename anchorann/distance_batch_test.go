package anchorann

import (
	"context"
	"reflect"
	"testing"
)

func TestParallelDistancesAndGreedyIdentity(t *testing.T) {
	d := testDataset(320, 7)
	centers := make([][]float64, 20)
	for c := range centers {
		centers[c] = mean(d, []uint32{uint32(c)})
	}
	ids := make([]int, d.N)
	for i := range ids {
		ids[i] = i
	}
	table := newDistanceTable(centers)
	costs := table.rows(context.Background(), d, ids, 20)
	for i := range ids {
		for c := range centers {
			want := distance(d.Values[i*d.Dim:(i+1)*d.Dim], centers[c])
			if costs[i*table.padded+c] != want {
				t.Fatalf("distance i=%d c=%d got=%g want=%g", i, c, costs[i*table.padded+c], want)
			}
		}
	}
	a, err := greedyAssignment(context.Background(), d, centers, 16, ids)
	if err != nil {
		t.Fatal(err)
	}
	b, err := greedyParallel(context.Background(), d, centers, 16, ids, 20)
	if err != nil || !reflect.DeepEqual(a, b) {
		t.Fatalf("parallel changed decisions %v", err)
	}
}
