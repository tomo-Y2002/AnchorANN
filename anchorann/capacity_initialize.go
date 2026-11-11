package anchorann

import (
	"context"
	"fmt"
	"math"
	"os"
	"time"
)

// auctionWarmStart accelerates sparse SAP with epsilon-scaling dual prices.
// It is an initializer only: violating rows are released and SAP completes the
// same sparse min-cost network before full-network certification/pricing.
func auctionWarmStart(ctx context.Context, g transportGraph, count, size int) ([]int, []float64, error) {
	n := len(g)
	prices := make([]float64, count)
	owners := make([]int, n)
	bids := make([]float64, n)
	members := make([][]int, count)
	maxCost := 1.0
	for _, edges := range g {
		for _, e := range edges {
			maxCost = math.Max(maxCost, math.Abs(e.cost))
		}
	}
	epsilon := maxCost / 16
	final := 1e-8
	// At extreme finite float32 scales, use an initializer epsilon larger than
	// a price ULP. SAP and full certification retain the same final tolerance.
	if maxCost > 1e12 {
		final = maxCost * 1e-12
	}
	start := time.Now()
	for stage := 0; ; stage++ {
		for i := range owners {
			owners[i] = -1
		}
		for c := range members {
			members[c] = members[c][:0]
		}
		queue := make([]int, n)
		for i := range queue {
			queue[i] = n - 1 - i
		}
		operations := 0
		for len(queue) > 0 {
			if operations%100000 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, nil, err
				}
			}
			row := queue[len(queue)-1]
			queue = queue[:len(queue)-1]
			best := -1
			first, second := math.Inf(1), math.Inf(1)
			for _, edge := range g[row] {
				c := int(edge.cluster)
				value := edge.cost + prices[c]
				if value < first || (value == first && c < best) {
					second = first
					first = value
					best = c
				} else if value < second {
					second = value
				}
			}
			if best < 0 {
				return nil, nil, fmt.Errorf("initializer missing edges row=%d", row)
			}
			increment := epsilon
			if !math.IsInf(second, 1) {
				increment += second - first
			}
			bids[row] = prices[best] + increment
			owners[row] = best
			// Direct capacity heap of retained row IDs, minimum bid at the root.
			h := members[best]
			if len(h) == size {
				evicted := h[0]
				owners[evicted] = -1
				queue = append(queue, evicted)
				h[0] = row
				bidDown(h, bids, 0)
			} else {
				h = append(h, row)
				bidUp(h, bids, len(h)-1)
			}
			members[best] = h
			if len(h) == size {
				prices[best] = bids[h[0]]
			}
			operations++
			if operations%10000000 == 0 && n >= 100000 {
				fmt.Fprintf(os.Stderr, "capacity initialize stage=%d epsilon=%g operations=%d pending=%d elapsed=%.3f\n", stage, epsilon, operations, len(queue), time.Since(start).Seconds())
			}
		}
		// Normalize common dual price offset after each stage.
		offset := prices[0]
		for c := range prices {
			prices[c] -= offset
		}
		if n >= 100000 {
			fmt.Fprintf(os.Stderr, "capacity initialize stage=%d epsilon=%g operations=%d elapsed=%.3f\n", stage, epsilon, operations, time.Since(start).Seconds())
		}
		if epsilon == final {
			break
		}
		epsilon = math.Max(final, epsilon/8)
	}
	p := make([]float64, count)
	for c, price := range prices {
		p[c] = -price
	}
	// Preserve only rows dual-feasible for every represented edge. Partial
	// capacities remain feasible; SAP augments every released row.
	released := 0
	for row, edges := range g {
		a := owners[row]
		base := 0.0
		for _, e := range edges {
			if int(e.cluster) == a {
				base = e.cost
				break
			}
		}
		for _, e := range edges {
			delta := p[a] - p[e.cluster]
			reduced := e.cost - base + delta
			scale := math.Max(1, math.Max(math.Abs(e.cost), math.Max(math.Abs(base), math.Abs(delta))))
			if reduced < -assignmentTolerance*scale {
				owners[row] = -1
				released++
				break
			}
		}
	}
	if n >= 100000 {
		fmt.Fprintf(os.Stderr, "capacity initialize released=%d/%d elapsed=%.3f\n", released, n, time.Since(start).Seconds())
	}
	return owners, p, nil
}
func bidLess(a, b int, bids []float64) bool {
	return bids[a] < bids[b] || (bids[a] == bids[b] && a < b)
}
func bidUp(h []int, bids []float64, i int) {
	x := h[i]
	for i > 0 {
		p := (i - 1) / 2
		if !bidLess(x, h[p], bids) {
			break
		}
		h[i] = h[p]
		i = p
	}
	h[i] = x
}
func bidDown(h []int, bids []float64, i int) {
	x := h[i]
	for i*2+1 < len(h) {
		child := i*2 + 1
		if child+1 < len(h) && bidLess(h[child+1], h[child], bids) {
			child++
		}
		if !bidLess(h[child], x, bids) {
			break
		}
		h[i] = h[child]
		i = child
	}
	h[i] = x
}
