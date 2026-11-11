package anchorann

import (
	"context"
	"fmt"
	"math/rand"
	"runtime"
	"sort"
	"sync"
)

// Method names the capacity-constrained clustering algorithm.
type Method string

const (
	// Recursive uses optimal fixed-two-center assignments at each tree split.
	Recursive Method = "recursive"
	// Global uses greedy capacity-constrained Lloyd assignments over all centers.
	Global Method = "global"
	// Common uses certified optimal fixed-all-centroid capacity assignments.
	Common Method = "common"
)

// Options controls clustering. Workers parallelizes tree branches or bounded
// full-centroid distance blocks. Greedy decisions are independent of scheduling.
// Common may choose a different tied optimum on the certified sparse path.
type Options struct {
	Size       int
	Seed       int64
	Method     Method
	Iterations int
	Workers    int
}

// DefaultOptions returns certified all-centroid Lloyd with size 32 and up to
// 20 parallel distance workers, selected from full-scale quality/cost results.
func DefaultOptions() Options {
	return Options{Size: 32, Seed: 711, Method: Common, Iterations: 6, Workers: min(20, runtime.GOMAXPROCS(0))}
}

// Cluster partitions every real ID exactly once and computes float64 centroids.
func Cluster(d Dataset, o Options) (*Artifact, error) {
	a, _, err := ClusterWithProgress(context.Background(), d, o, nil)
	return a, err
}

// ClusterWithProgress reports only completed assignment and mean-update steps.
func ClusterWithProgress(ctx context.Context, d Dataset, o Options, observer Observer) (*Artifact, RunSummary, error) {
	summary := RunSummary{StopReason: "iteration_limit"}
	if err := validateSize(d.N, o.Size); err != nil {
		return nil, summary, err
	}
	if err := d.Validate(); err != nil {
		return nil, summary, fmt.Errorf("clustering input N=%d s=%d: %w", d.N, o.Size, err)
	}
	if o.Iterations <= 0 || o.Workers <= 0 || (o.Method != Recursive && o.Method != Global && o.Method != Common) {
		return nil, summary, fmt.Errorf("invalid clustering options: method=%q iterations=%d workers=%d", o.Method, o.Iterations, o.Workers)
	}
	if uint64(d.N) > uint64(^uint32(0)) {
		return nil, summary, fmt.Errorf("N=%d exceeds uint32 original ID format", d.N)
	}
	ids := make([]uint32, d.N)
	for i := range ids {
		ids[i] = uint32(i)
	}
	a := &Artifact{Version: FormatVersion, N: d.N, Size: o.Size, Count: d.N / o.Size, Dim: d.Dim, Seed: o.Seed, Method: o.Method, Iterations: o.Iterations}
	if o.Method == Recursive {
		sem := make(chan struct{}, min(o.Workers, a.Count)-1)
		var divide func([]uint32, int64)
		divide = func(part []uint32, seed int64) {
			if ctx.Err() != nil {
				return
			}
			c := len(part) / o.Size
			if c == 1 {
				return
			}
			rng := rand.New(rand.NewSource(seed))
			left := (c / 2) * o.Size
			split(d, part, left, o.Iterations, rng)
			leftSeed, rightSeed := rng.Int63(), rng.Int63()
			select {
			case sem <- struct{}{}:
				var wg sync.WaitGroup
				wg.Add(1)
				go func() {
					defer wg.Done()
					defer func() { <-sem }()
					divide(part[:left], leftSeed)
				}()
				divide(part[left:], rightSeed)
				wg.Wait()
			default:
				divide(part[:left], leftSeed)
				divide(part[left:], rightSeed)
			}
		}
		divide(ids, o.Seed)
		if err := ctx.Err(); err != nil {
			summary.StopReason = "cancelled"
			return nil, summary, err
		}
	} else {
		var err error
		ids, summary, err = lloyd(ctx, d, o, observer)
		if err != nil {
			return nil, summary, err
		}
	}
	return artifactFromIDs(d, o, ids, summary)
}

func artifactFromIDs(d Dataset, o Options, ids []uint32, summary RunSummary) (*Artifact, RunSummary, error) {
	a := &Artifact{Version: FormatVersion, N: d.N, Size: o.Size, Count: d.N / o.Size, Dim: d.Dim, Seed: o.Seed, Method: o.Method, Iterations: o.Iterations}
	a.Members = make([][]uint32, a.Count)
	a.Anchors = make([][]float64, a.Count)
	for c := 0; c < a.Count; c++ {
		part := ids[c*o.Size : (c+1)*o.Size]
		sort.Slice(part, func(i, j int) bool { return part[i] < part[j] })
		a.Members[c] = part
		a.Anchors[c] = mean(d, part)
	}
	if err := a.Validate(); err != nil {
		return nil, summary, fmt.Errorf("clustering output: %w", err)
	}
	return a, summary, nil
}

func mean(d Dataset, ids []uint32) []float64 {
	m := make([]float64, d.Dim)
	for _, id := range ids {
		for j, x := range d.Values[int(id)*d.Dim : (int(id)+1)*d.Dim] {
			m[j] += float64(x)
		}
	}
	for j := range m {
		m[j] /= float64(len(ids))
	}
	return m
}

func distance(v []float32, m []float64) float64 {
	var sum float64
	for j, x := range v {
		diff := float64(x) - m[j]
		sum += diff * diff
	}
	return sum
}

type scoredID struct {
	id    uint32
	score float64
}

func less(a, b scoredID) bool {
	return a.score < b.score || (a.score == b.score && a.id < b.id)
}

// partition selects the k lowest scores using in-place quickselect. The midpoint
// pivot and ID tie-break make even identical-vector input deterministic.
func partition(a []scoredID, k int) {
	lo, hi := 0, len(a)-1
	for lo < hi {
		pivot := a[lo+(hi-lo)/2]
		i, j := lo, hi
		for i <= j {
			for less(a[i], pivot) {
				i++
			}
			for less(pivot, a[j]) {
				j--
			}
			if i <= j {
				a[i], a[j] = a[j], a[i]
				i++
				j--
			}
		}
		if k-1 <= j {
			hi = j
		} else if k-1 >= i {
			lo = i
		} else {
			return
		}
	}
}

func split(d Dataset, ids []uint32, left, iterations int, rng *rand.Rand) {
	first := ids[rng.Intn(len(ids))]
	a := mean(d, []uint32{first})
	// Farthest-point second seed avoids coincident initial centers where possible.
	second, farthest := first, float64(-1)
	for _, id := range ids {
		cost := distance(d.Values[int(id)*d.Dim:(int(id)+1)*d.Dim], a)
		if cost > farthest || (cost == farthest && id < second) {
			second, farthest = id, cost
		}
	}
	b := mean(d, []uint32{second})
	scores := make([]scoredID, len(ids))
	direction := make([]float64, d.Dim)
	for iter := 0; iter < iterations; iter++ {
		for j := range direction {
			direction[j] = 2 * (b[j] - a[j])
		}
		// ||x-a||² - ||x-b||² differs from this dot product only by a
		// point-independent constant. The lowest left scores minimize cost
		// for these fixed centers subject to the exact left/right capacities.
		for i, id := range ids {
			var score float64
			for j, x := range d.Values[int(id)*d.Dim : (int(id)+1)*d.Dim] {
				score += float64(x) * direction[j]
			}
			scores[i] = scoredID{id, score}
		}
		partition(scores, left)
		for i := range ids {
			ids[i] = scores[i].id
		}
		a, b = mean(d, ids[:left]), mean(d, ids[left:])
	}
}
