package anchorann

import (
	"context"
	"runtime"
	"sync"
)

// distanceTable keeps only a bounded row block, never an N*C matrix.
type distanceTable struct {
	centers            []float64
	count, padded, dim int
}

func newDistanceTable(centers [][]float64) *distanceTable {
	t := &distanceTable{count: len(centers), padded: (len(centers) + 3) / 4 * 4, dim: len(centers[0])}
	t.centers = make([]float64, t.padded*t.dim)
	for c, m := range centers {
		for j, x := range m {
			t.centers[(c/4*t.dim+j)*4+c%4] = x
		}
	}
	return t
}
func (t *distanceTable) rows(ctx context.Context, d Dataset, ids []int, workers int) []float64 {
	costs := make([]float64, len(ids)*t.padded)
	workers = min(max(1, workers), len(ids))
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for start := w * 16; start < len(ids); start += workers * 16 {
				if ctx.Err() != nil {
					return
				}
				end := min(start+16, len(ids))
				values := make([]float32, (end-start)*t.dim)
				for j, id := range ids[start:end] {
					copy(values[j*t.dim:], d.Values[id*t.dim:(id+1)*t.dim])
				}
				batchDistances(values, t.centers, costs[start*t.padded:end*t.padded], end-start, t.padded, t.dim)
			}
		}(w)
	}
	wg.Wait()
	return costs
}
func parallelWorkers(workers int) int { return min(max(1, workers), runtime.GOMAXPROCS(0)) }
