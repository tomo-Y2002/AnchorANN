package anchorann

import (
	"context"
	"fmt"
	"math"
	"os"
	"sync"
	"time"
)

type transportEdge struct {
	cluster int32
	cost    float64
}
type transportGraph [][]transportEdge

type pathNode struct {
	cluster int
	dist    float64
}
type pathHeap []pathNode

// sparseTransport solves the capacity network on a subset of arcs exactly by
// shortest augmenting paths. Full-network column pricing follows each solve.
func sparseTransport(ctx context.Context, g transportGraph, count, size int) ([]int, []float64, error) {
	return sparseTransportWarm(ctx, g, count, size, nil, nil)
}

func sparseTransportWarm(ctx context.Context, g transportGraph, count, size int, initialOwners []int, initialPotentials []float64) ([]int, []float64, error) {
	n := len(g)
	owners := make([]int, n)
	assignedCost := make([]float64, n)
	for i := range owners {
		owners[i] = -1
	}
	members := make([][]int, count)
	for c := range members {
		members[c] = make([]int, 0, size)
	}
	p := make([]float64, count)
	if initialOwners != nil {
		copy(owners, initialOwners)
		copy(p, initialPotentials)
		for row, c := range owners {
			if c < 0 {
				continue
			}
			members[c] = append(members[c], row)
			found := false
			for _, edge := range g[row] {
				if int(edge.cluster) == c {
					assignedCost[row] = edge.cost
					found = true
					break
				}
			}
			if !found {
				return nil, nil, fmt.Errorf("warm assigned edge missing row=%d", row)
			}
		}
	}

	dist := make([]float64, count)
	prev := make([]int, count)
	moving := make([]int, count)
	moveCost := make([]float64, count)
	generation := make([]int, count)
	settled := make([]int, count)
	touched := make([]int, 0, count)
	queue := make(pathHeap, 0, 1024)
	start := time.Now()
	for row := 0; row < n; row++ {
		if owners[row] >= 0 {
			continue
		}
		if row%10000 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, nil, err
			}
			if n >= 100000 {
				fmt.Fprintf(os.Stderr, "capacity sparse augmented=%d/%d elapsed=%.3f\n", row, n, time.Since(start).Seconds())
			}
		}
		gen := row + 1
		touched = touched[:0]
		queue = queue[:0]
		for _, edge := range g[row] {
			c := int(edge.cluster)
			v := edge.cost - p[c]
			generation[c] = gen
			dist[c] = v
			prev[c] = -1
			moving[c] = row
			moveCost[c] = edge.cost
			queue.push(pathNode{c, v})
		}
		end := -1
		for len(queue) > 0 {
			node := queue.pop()
			a := node.cluster
			if settled[a] == gen || node.dist != dist[a] {
				continue
			}
			settled[a] = gen
			touched = append(touched, a)
			if len(members[a]) < size {
				end = a
				break
			}
			for _, id := range members[a] {
				base := assignedCost[id]
				for _, edge := range g[id] {
					b := int(edge.cluster)
					if settled[b] == gen {
						continue
					}
					reduced := edge.cost - base + p[a] - p[b]
					if reduced < 0 {
						scale := math.Max(1, math.Max(math.Abs(edge.cost), math.Max(math.Abs(base), math.Abs(p[a]-p[b]))))
						if reduced < -assignmentTolerance*scale {
							return nil, nil, fmt.Errorf("sparse residual violation row=%d edge=%d cost=%g", id, b, reduced)
						}
						reduced = 0
					}
					v := dist[a] + math.Max(0, reduced)
					if generation[b] != gen || v < dist[b] {
						generation[b] = gen
						dist[b] = v
						prev[b] = a
						moving[b] = id
						moveCost[b] = edge.cost
						queue.push(pathNode{b, v})
					}
				}
			}
		}
		if end < 0 {
			return nil, nil, fmt.Errorf("sparse capacity network infeasible at row=%d", row)
		}
		// p += min(dist,dist[end]), with common dist[end] offset omitted. Only
		// settled vertices can have distances strictly below the terminal distance.
		for _, b := range touched {
			p[b] += dist[b] - dist[end]
		}
		for b := end; b >= 0; {
			id := moving[b]
			a := prev[b]
			if a >= 0 {
				for j, x := range members[a] {
					if x == id {
						members[a][j] = members[a][len(members[a])-1]
						members[a] = members[a][:len(members[a])-1]
						break
					}
				}
			}
			members[b] = append(members[b], id)
			owners[id] = b
			assignedCost[id] = moveCost[b]
			b = a
		}
	}
	return owners, p, nil
}

// pricedAssignment uses full squared-distance pricing to add every violating
// omitted edge. Termination requires the same full N*C certificate as dense SAP.
// No fixed candidate cutoff is accepted as optimal.
func pricedAssignment(ctx context.Context, d Dataset, centers [][]float64, size, workers int) ([]int, []float64, AssignmentCertificate, error) {
	if err := validateAssignmentInput(d, centers, size); err != nil {
		return nil, nil, AssignmentCertificate{}, err
	}
	table := newDistanceTable(centers)
	g := make(transportGraph, d.N)
	start := time.Now()
	for offset := 0; offset < d.N; offset += 256 {
		end := min(offset+256, d.N)
		ids := make([]int, end-offset)
		for i := range ids {
			ids[i] = offset + i
		}
		costs := table.rows(ctx, d, ids, workers)
		if err := ctx.Err(); err != nil {
			return nil, nil, AssignmentCertificate{}, err
		}
		for local, row := range ids {
			edges := make([]transportEdge, 0, 25)
			rowCosts := costs[local*table.padded : local*table.padded+table.count]
			for c, cost := range rowCosts {
				if len(edges) == 24 && cost >= edges[len(edges)-1].cost {
					continue
				}
				edge := transportEdge{int32(c), cost}
				position := len(edges)
				if position == 24 {
					position--
				} else {
					edges = append(edges, edge)
				}
				for position > 0 && (cost < edges[position-1].cost || (cost == edges[position-1].cost && c < int(edges[position-1].cluster))) {
					edges[position] = edges[position-1]
					position--
				}
				edges[position] = edge
			}
			// This explicitly feasible assignment guarantees the sparse graph admits
			// exact capacities even before any omitted arcs are priced.
			feasible := row % len(centers)
			found := false
			for _, edge := range edges {
				if int(edge.cluster) == feasible {
					found = true
					break
				}
			}
			if !found {
				edges = append(edges, transportEdge{int32(feasible), rowCosts[feasible]})
			}
			g[row] = edges
		}
		if offset%32768 == 0 && d.N >= 100000 {
			fmt.Fprintf(os.Stderr, "capacity initial pricing rows=%d/%d elapsed=%.3f\n", end, d.N, time.Since(start).Seconds())
		}
	}
	var warmOwners []int
	var warmPotentials []float64
	for pass := 1; ; pass++ {
		if err := ctx.Err(); err != nil {
			return nil, nil, AssignmentCertificate{}, err
		}
		fmt.Fprintf(os.Stderr, "capacity solve pass=%d\n", pass)
		// Reinitialize dual prices on the expanded graph; epsilon initialization
		// replaces repeated long sparse reaugmentation, then SAP verifies the block.
		if d.N >= 256 {
			var err error
			warmOwners, warmPotentials, err = auctionWarmStart(ctx, g, len(centers), size)
			if err != nil {
				return nil, nil, AssignmentCertificate{}, err
			}
		}
		owners, p, err := sparseTransportWarm(ctx, g, len(centers), size, warmOwners, warmPotentials)
		if err != nil {
			return nil, nil, AssignmentCertificate{}, err
		}
		cert := AssignmentCertificate{}
		added := 0
		warmOwners = append([]int(nil), owners...)
		warmPotentials = p
		for offset := 0; offset < d.N; offset += 256 {
			end := min(offset+256, d.N)
			ids := make([]int, end-offset)
			for i := range ids {
				ids[i] = offset + i
			}
			costs := table.rows(ctx, d, ids, workers)
			if err := ctx.Err(); err != nil {
				return nil, nil, cert, err
			}
			results := make([]pricingResult, len(ids))
			workerCount := min(workers, len(ids))
			var wg sync.WaitGroup
			for worker := 0; worker < workerCount; worker++ {
				wg.Add(1)
				go func(worker int) {
					defer wg.Done()
					for local := worker; local < len(ids); local += workerCount {
						row := ids[local]
						a := owners[row]
						base := costs[local*table.padded+a]
						result := pricingResult{objective: base}
						for b, cost := range costs[local*table.padded : local*table.padded+table.count] {
							delta := p[a] - p[b]
							reduced := cost - base + delta
							if reduced >= 0 {
								continue
							}
							scale := math.Max(1, math.Max(math.Abs(cost), math.Max(math.Abs(base), math.Abs(delta))))
							violation := -reduced
							result.maxViolation = math.Max(result.maxViolation, violation)
							result.maxRelative = math.Max(result.maxRelative, violation/scale)
							if violation > assignmentTolerance*scale {
								found := false
								for _, edge := range g[row] {
									if int(edge.cluster) == b {
										found = true
										break
									}
								}
								if found {
									result.err = fmt.Errorf("existing sparse edge failed certificate row=%d cluster=%d", row, b)
									break
								}
								g[row] = append(g[row], transportEdge{int32(b), cost})
								result.added++
							}
						}
						results[local] = result
					}
				}(worker)
			}
			wg.Wait()
			for local, result := range results {
				if result.err != nil {
					return nil, nil, cert, result.err
				}
				cert.Objective += result.objective
				cert.CheckedEdges += uint64(table.count)
				cert.MaxViolation = math.Max(cert.MaxViolation, result.maxViolation)
				cert.MaxRelativeViolation = math.Max(cert.MaxRelativeViolation, result.maxRelative)
				if result.added > 0 {
					warmOwners[ids[local]] = -1
				}
				added += result.added
			}

			if offset%131072 == 0 && d.N >= 100000 {
				fmt.Fprintf(os.Stderr, "capacity certificate pass=%d rows=%d/%d added=%d\n", pass, end, d.N, added)
			}
		}
		fmt.Fprintf(os.Stderr, "capacity priced pass=%d added=%d elapsed=%.3f\n", pass, added, time.Since(start).Seconds())
		if added == 0 {
			counts := make([]int, len(centers))
			for row, c := range owners {
				if c < 0 || c >= len(centers) {
					return nil, nil, cert, fmt.Errorf("invalid final owner row=%d", row)
				}
				counts[c]++
			}
			for c, n := range counts {
				if n != size {
					return nil, nil, cert, fmt.Errorf("final capacity c=%d n=%d expected=%d", c, n, size)
				}
			}
			return owners, p, cert, nil
		}
	}
}

func (h *pathHeap) push(x pathNode) {
	a := append(*h, x)
	i := len(a) - 1
	for i > 0 {
		p := (i - 1) / 2
		if !(x.dist < a[p].dist || (x.dist == a[p].dist && x.cluster < a[p].cluster)) {
			break
		}
		a[i] = a[p]
		i = p
	}
	a[i] = x
	*h = a
}
func (h *pathHeap) pop() pathNode {
	a := *h
	result := a[0]
	x := a[len(a)-1]
	a = a[:len(a)-1]
	i := 0
	for i*2+1 < len(a) {
		child := i*2 + 1
		if child+1 < len(a) && (a[child+1].dist < a[child].dist || (a[child+1].dist == a[child].dist && a[child+1].cluster < a[child].cluster)) {
			child++
		}
		if !(a[child].dist < x.dist || (a[child].dist == x.dist && a[child].cluster < x.cluster)) {
			break
		}
		a[i] = a[child]
		i = child
	}
	if len(a) > 0 {
		a[i] = x
	}
	*h = a
	return result
}

type pricingResult struct {
	objective, maxViolation, maxRelative float64
	added                                int
	err                                  error
}
