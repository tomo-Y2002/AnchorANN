package anchorann

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func testDataset(n, dim int) Dataset {
	rng := rand.New(rand.NewSource(17))
	d := Dataset{N: n, Dim: dim, Values: make([]float32, n*dim)}
	for i := range d.Values {
		d.Values[i] = float32(rng.NormFloat64())
	}
	return d
}

func testCluster(t *testing.T, d Dataset, s int, method Method, workers int) *Artifact {
	t.Helper()
	o := DefaultOptions()
	o.Size, o.Method, o.Workers = s, method, workers
	a, err := Cluster(d, o)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestClusterInputValidation(t *testing.T) {
	for _, tc := range []struct {
		name string
		d    Dataset
		s    int
		want []string
	}{
		{"remainder", testDataset(7, 2), 3, []string{"N=7", "s=3", "remainder=1"}},
		{"zero size", testDataset(4, 2), 0, []string{"N=4", "s=0"}},
		{"negative size", testDataset(4, 2), -2, []string{"N=4", "s=-2"}},
		{"empty", Dataset{Dim: 2}, 32, []string{"N=0", "s=32"}},
		{"dimension mismatch", Dataset{N: 4, Dim: 2, Values: make([]float32, 7)}, 2, []string{"N=4", "s=2", "dimension=2", "coordinates=7"}},
		{"zero dimension", Dataset{N: 4}, 2, []string{"N=4", "s=2", "dimension=0"}},
		{"overflow shape", Dataset{N: int(^uint(0) >> 1), Dim: 2}, 1, []string{"invalid dataset"}},
		{"nan", Dataset{N: 2, Dim: 1, Values: []float32{0, float32(math.NaN())}}, 1, []string{"nonfinite", "point=1"}},
		{"infinity", Dataset{N: 2, Dim: 1, Values: []float32{float32(math.Inf(1)), 0}}, 1, []string{"nonfinite", "point=0"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			o := DefaultOptions()
			o.Size = tc.s
			a, err := Cluster(tc.d, o)
			if err == nil || a != nil {
				t.Fatalf("invalid input produced artifact: %v, %v", a, err)
			}
			for _, want := range tc.want {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q lacks %q", err, want)
				}
			}
		})
	}
	for _, change := range []func(*Options){
		func(o *Options) { o.Method = "unknown" },
		func(o *Options) { o.Iterations = 0 },
		func(o *Options) { o.Workers = 0 },
		func(o *Options) { o.Workers = -1 },
	} {
		o := DefaultOptions()
		o.Size = 2
		change(&o)
		if _, err := Cluster(testDataset(4, 2), o); err == nil {
			t.Fatalf("accepted invalid options: %+v", o)
		}
	}
}

func TestEqualSizesAndCentroids(t *testing.T) {
	for _, method := range []Method{Recursive, Global, Common} {
		for _, s := range []int{1, 2, 4, 12} {
			t.Run(fmt.Sprintf("%s/s=%d", method, s), func(t *testing.T) {
				d := testDataset(12, 3) // C=3 and C=6 exercise non-power-of-two splits.
				before := append([]float32(nil), d.Values...)
				a := testCluster(t, d, s, method, 4)
				if a.Count != d.N/s || len(a.Members) != a.Count || len(a.Anchors) != a.Count {
					t.Fatalf("wrong cluster count: %+v", a)
				}
				seen := make([]int, d.N)
				for c, ids := range a.Members {
					if len(ids) != s {
						t.Fatalf("cluster %d has %d real points, expected %d", c, len(ids), s)
					}
					for _, id := range ids {
						seen[id]++
					}
					for j, got := range a.Anchors[c] {
						var sum float64
						for _, id := range ids {
							sum += float64(d.Values[int(id)*d.Dim+j])
						}
						if got != sum/float64(s) {
							t.Fatalf("anchor=%d coordinate=%d got=%g want=%g", c, j, got, sum/float64(s))
						}
					}
				}
				for id, count := range seen {
					if count != 1 {
						t.Fatalf("ID=%d appears %d times", id, count)
					}
				}
				if !reflect.DeepEqual(d.Values, before) {
					t.Fatal("clustering modified source vectors")
				}
				if s == 1 {
					if sse, err := SSE(d, a); err != nil || sse != 0 {
						t.Fatalf("singleton SSE=%g, %v", sse, err)
					}
				}
			})
		}
	}
}

func TestCentroidFloat64Accumulation(t *testing.T) {
	d := Dataset{N: 4, Dim: 1, Values: []float32{1e8, 1, -1e8, 3}}
	for _, method := range []Method{Recursive, Global, Common} {
		a := testCluster(t, d, d.N, method, 1)
		if a.Anchors[0][0] != 1 {
			t.Fatalf("%s centroid=%g; float64 accumulation should retain small terms", method, a.Anchors[0][0])
		}
	}
}

func TestReproducibilityAndDegenerateInput(t *testing.T) {
	for _, d := range []Dataset{testDataset(96, 5), {N: 96, Dim: 5, Values: make([]float32, 96*5)}} {
		for _, method := range []Method{Recursive, Global, Common} {
			a := testCluster(t, d, 32, method, 1)
			for _, workers := range []int{1, 2, 8} {
				b := testCluster(t, d, 32, method, workers)
				if !reflect.DeepEqual(a, b) {
					t.Fatalf("%s is not reproducible with workers=%d", method, workers)
				}
			}
		}
	}
}

func TestPartitionMatchesSortedCapacities(t *testing.T) {
	rng := rand.New(rand.NewSource(711))
	for n := 2; n < 100; n++ {
		for k := 1; k < n; k++ {
			a := make([]scoredID, n)
			for i := range a {
				a[i] = scoredID{uint32(i), float64(rng.Intn(5))}
			}
			rng.Shuffle(n, func(i, j int) { a[i], a[j] = a[j], a[i] })
			want := append([]scoredID(nil), a...)
			sort.Slice(want, func(i, j int) bool { return less(want[i], want[j]) })
			partition(a, k)
			sort.Slice(a[:k], func(i, j int) bool { return less(a[i], a[j]) })
			sort.Slice(a[k:], func(i, j int) bool { return less(a[k+i], a[k+j]) })
			if !reflect.DeepEqual(a, want) {
				t.Fatalf("partition n=%d k=%d differs from exact sorting", n, k)
			}
		}
	}
}

func TestFixedTwoCenterAssignmentOptimality(t *testing.T) {
	// Enumerate every feasible 3:4 assignment and compare the cost against
	// the selected score differences, including ties and unequal capacities.
	d := testDataset(7, 3)
	a, b := []float64{1, -2, 3}, []float64{-1, 0, 1}
	scores := make([]scoredID, d.N)
	for id := 0; id < d.N; id++ {
		v := d.Values[id*d.Dim : (id+1)*d.Dim]
		scores[id] = scoredID{uint32(id), distance(v, a) - distance(v, b)}
	}
	partition(scores, 3)
	var got float64
	for i, x := range scores {
		center := b
		if i < 3 {
			center = a
		}
		got += distance(d.Values[int(x.id)*d.Dim:(int(x.id)+1)*d.Dim], center)
	}
	best := math.Inf(1)
	for mask := 0; mask < 1<<d.N; mask++ {
		count, cost := 0, float64(0)
		for id := 0; id < d.N; id++ {
			center := b
			if mask&(1<<id) != 0 {
				count++
				center = a
			}
			cost += distance(d.Values[id*d.Dim:(id+1)*d.Dim], center)
		}
		if count == 3 && cost < best {
			best = cost
		}
	}
	if math.Abs(got-best) > 1e-10 {
		t.Fatalf("fixed-center cost=%g optimal=%g", got, best)
	}
}
