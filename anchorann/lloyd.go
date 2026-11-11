package anchorann

import (
	"context"
	"fmt"
	"math/rand"
	"reflect"
	"sort"
	"time"
)

// Step is emitted after a complete assignment, certification (Common), mean
// update and SSE. The synchronous observer's time is excluded from Seconds.
type Step struct {
	Iteration   int                    `json:"iteration"`
	SSE         float64                `json:"sse"`
	Seconds     float64                `json:"cluster_seconds"`
	Changed     bool                   `json:"changed"`
	Certificate *AssignmentCertificate `json:"certificate,omitempty"`
	Artifact    *Artifact              `json:"-"`
}

// Observer must not mutate the supplied artifact. Errors abort the run.
type Observer func(Step) error

// RunSummary distinguishes unchanged membership from a limit or cancellation.
type RunSummary struct {
	CompletedIterations int     `json:"completed_iterations"`
	StopReason          string  `json:"stop_reason"`
	ClusterSeconds      float64 `json:"cluster_seconds"`
}

func lloyd(ctx context.Context, d Dataset, o Options, observer Observer) ([]uint32, RunSummary, error) {
	return lloydFrom(ctx, d, o, observer, nil, 0)
}

func lloydFrom(ctx context.Context, d Dataset, o Options, observer Observer, checkpoint *Artifact, completed int) ([]uint32, RunSummary, error) {
	summary := RunSummary{StopReason: "iteration_limit", CompletedIterations: completed}
	setupStart := time.Now()
	rng := rand.New(rand.NewSource(o.Seed))
	order := rng.Perm(d.N)
	c := d.N / o.Size
	centers := make([][]float64, c)
	var previous []int
	if checkpoint == nil {
		for i := range centers {
			centers[i] = mean(d, []uint32{uint32(order[i])})
		}
	} else {
		previous = make([]int, d.N)
		for i, members := range checkpoint.Members {
			centers[i] = append([]float64(nil), checkpoint.Anchors[i]...)
			for _, id := range members {
				previous[id] = i
			}
		}
		// Replay only RNG draws and permutation mutations, never assignments.
		for iteration := 1; iteration <= completed; iteration++ {
			if err := ctx.Err(); err != nil {
				summary.StopReason = "cancelled"
				return nil, summary, err
			}
			rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
		}
	}
	summary.ClusterSeconds = time.Since(setupStart).Seconds()
	var ids []uint32
	for iteration := completed + 1; iteration <= o.Iterations; iteration++ {
		start := time.Now()
		var owners []int
		var certificate *AssignmentCertificate
		var err error
		if o.Method == Common {
			var cert AssignmentCertificate
			if d.N >= 256 && o.Workers > 1 {
				owners, _, cert, err = pricedAssignment(ctx, d, centers, o.Size, o.Workers)
			} else {
				owners, _, cert, err = OptimalAssignment(ctx, d, centers, o.Size)
			}
			certificate = &cert
		} else {
			rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
			owners, err = greedyParallel(ctx, d, centers, o.Size, order, o.Workers)
		}
		if err != nil {
			summary.ClusterSeconds += time.Since(start).Seconds()
			summary.StopReason = "error"
			if ctx.Err() != nil {
				summary.StopReason = "cancelled"
			}
			return nil, summary, err
		}
		ids = make([]uint32, d.N)
		counts := make([]int, c)
		// Ascending IDs give deterministic means and stable persisted membership.
		for id, owner := range owners {
			ids[owner*o.Size+counts[owner]] = uint32(id)
			counts[owner]++
		}
		a := &Artifact{Version: FormatVersion, N: d.N, Size: o.Size, Count: c, Dim: d.Dim, Seed: o.Seed, Method: o.Method, Iterations: o.Iterations, Members: make([][]uint32, c), Anchors: make([][]float64, c)}
		for i := range centers {
			a.Members[i] = ids[i*o.Size : (i+1)*o.Size]
			centers[i] = mean(d, a.Members[i])
			a.Anchors[i] = centers[i]
		}
		sse, err := SSE(d, a)
		if err != nil {
			return nil, summary, err
		}
		changed := !reflect.DeepEqual(previous, owners)
		seconds := time.Since(start).Seconds()
		summary.ClusterSeconds += seconds
		summary.CompletedIterations = iteration
		if observer != nil {
			if err := observer(Step{iteration, sse, seconds, changed, certificate, a}); err != nil {
				summary.StopReason = "observer_error"
				return nil, summary, fmt.Errorf("iteration observer: %w", err)
			}
		}
		if !changed {
			summary.StopReason = "unchanged_membership"
			break
		}
		previous = owners
	}
	// final membership already sorted; keep explicit requirement at caller too.
	for i := 0; i < c; i++ {
		sort.Slice(ids[i*o.Size:(i+1)*o.Size], func(a, b int) bool { return ids[i*o.Size+a] < ids[i*o.Size+b] })
	}
	return ids, summary, nil
}

func greedyAssignment(ctx context.Context, d Dataset, centers [][]float64, size int, order []int) ([]int, error) {
	c := len(centers)
	counts := make([]int, c)
	active := make([]int, c)
	for i := range active {
		active[i] = i
	}
	available := c
	owners := make([]int, d.N)
	for row, id := range order {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("greedy assigned=%d/%d: %w", row, d.N, err)
		}
		v := d.Values[id*d.Dim : (id+1)*d.Dim]
		bestSlot, best := 0, active[0]
		bestCost := distance(v, centers[best])
		for slot := 1; slot < available; slot++ {
			cluster := active[slot]
			cost := distance(v, centers[cluster])
			if cost < bestCost || (cost == bestCost && cluster < best) {
				bestSlot, best, bestCost = slot, cluster, cost
			}
		}
		owners[id] = best
		counts[best]++
		if counts[best] == size {
			available--
			active[bestSlot], active[available] = active[available], active[bestSlot]
		}
	}
	return owners, nil
}

// greedyParallel computes full fixed-centroid distances in bounded blocks, then
// applies the exact same seeded row order and capacity/tie decisions serially.
func greedyParallel(ctx context.Context, d Dataset, centers [][]float64, size int, order []int, workers int) ([]int, error) {
	if workers <= 1 || d.N < 256 {
		return greedyAssignment(ctx, d, centers, size, order)
	}
	table := newDistanceTable(centers)
	counts := make([]int, len(centers))
	owners := make([]int, d.N)
	for start := 0; start < len(order); start += 512 {
		end := min(start+512, len(order))
		costs := table.rows(ctx, d, order[start:end], workers)
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for row, id := range order[start:end] {
			best := -1
			bestCost := 0.0
			for c, cost := range costs[row*table.padded : row*table.padded+table.count] {
				if counts[c] < size && (best < 0 || cost < bestCost) {
					best, bestCost = c, cost
				}
			}
			owners[id] = best
			counts[best]++
		}
	}
	return owners, nil
}
