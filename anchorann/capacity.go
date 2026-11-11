package anchorann

import (
	"context"
	"fmt"
	"math"
)

// AssignmentCertificate audits every edge of the full fixed-centroid problem.
// Potentials are centroid dual variables; each row's dual is cost(i,owner)-p(owner).
// Thus nonnegative reduced costs certify an optimal feasible assignment.
type AssignmentCertificate struct {
	Objective            float64 `json:"objective"`
	MaxViolation         float64 `json:"max_violation"`
	MaxRelativeViolation float64 `json:"max_relative_violation"`
	CheckedEdges         uint64  `json:"checked_edges"`
}

const assignmentTolerance = 1e-10

// VerifyAssignment checks exact capacity and dual feasibility on ALL N*C edges.
// The tolerance per edge is 1e-10*max(1,abs(cost),abs(assigned cost),abs(potential
// difference)). This is a floating-point certificate, not joint Lloyd optimality.
func VerifyAssignment(ctx context.Context, d Dataset, centers [][]float64, size int, owners []int, potentials []float64) (AssignmentCertificate, error) {
	var cert AssignmentCertificate
	if err := validateAssignmentInput(d, centers, size); err != nil {
		return cert, err
	}
	if len(owners) != d.N || len(potentials) != len(centers) {
		return cert, fmt.Errorf("invalid assignment certificate shape")
	}
	counts := make([]int, len(centers))
	for c, p := range potentials {
		if math.IsNaN(p) || math.IsInf(p, 0) {
			return cert, fmt.Errorf("nonfinite potential c=%d", c)
		}
	}
	for i, a := range owners {
		if err := ctx.Err(); err != nil {
			return cert, err
		}
		if a < 0 || a >= len(centers) {
			return cert, fmt.Errorf("invalid owner row=%d cluster=%d", i, a)
		}
		counts[a]++
		v := d.Values[i*d.Dim : (i+1)*d.Dim]
		base := distance(v, centers[a])
		cert.Objective += base
		for b := range centers {
			cost := distance(v, centers[b])
			delta := potentials[a] - potentials[b]
			reduced := cost - base + delta
			scale := math.Max(1, math.Max(math.Abs(cost), math.Max(math.Abs(base), math.Abs(delta))))
			violation := math.Max(0, -reduced)
			cert.MaxViolation = math.Max(cert.MaxViolation, violation)
			cert.MaxRelativeViolation = math.Max(cert.MaxRelativeViolation, violation/scale)
			cert.CheckedEdges++
			if math.IsNaN(reduced) || math.IsInf(reduced, 0) || violation > assignmentTolerance*scale {
				return cert, fmt.Errorf("assignment dual violation row=%d cluster=%d reduced=%g tolerance=%g", i, b, reduced, assignmentTolerance*scale)
			}
		}
	}
	for c, n := range counts {
		if n != size {
			return cert, fmt.Errorf("assignment cluster=%d size=%d expected=%d", c, n, size)
		}
	}
	return cert, nil
}

func validateAssignmentInput(d Dataset, centers [][]float64, size int) error {
	if err := validateSize(d.N, size); err != nil {
		return err
	}
	if err := d.Validate(); err != nil {
		return err
	}
	if len(centers) != d.N/size {
		return fmt.Errorf("fixed centers=%d expected=%d", len(centers), d.N/size)
	}
	for c, m := range centers {
		if len(m) != d.Dim {
			return fmt.Errorf("fixed center=%d dimension=%d expected=%d", c, len(m), d.Dim)
		}
		for _, x := range m {
			if math.IsNaN(x) || math.IsInf(x, 0) {
				return fmt.Errorf("nonfinite fixed center=%d", c)
			}
		}
	}
	return nil
}

// OptimalAssignment solves fixed-centroid squared L2 transport with direct
// cluster capacities using successive shortest augmenting paths. Full residual
// edges are evaluated lazily from the rows currently assigned to a reached
// centroid. No slots, N*C table, candidate restriction, or greedy fallback is
// used. Storage is O(N+C+Cd); worst-case time is O(N*C*C*s*d).
func OptimalAssignment(ctx context.Context, d Dataset, centers [][]float64, size int) ([]int, []float64, AssignmentCertificate, error) {
	var cert AssignmentCertificate
	if err := validateAssignmentInput(d, centers, size); err != nil {
		return nil, nil, cert, err
	}
	c := len(centers)
	owners := make([]int, d.N)
	for i := range owners {
		owners[i] = -1
	}
	members := make([][]int, c)
	potentials := make([]float64, c)
	dist := make([]float64, c)
	visited := make([]bool, c)
	previous := make([]int, c)
	moving := make([]int, c)
	for row := 0; row < d.N; row++ {
		if err := ctx.Err(); err != nil {
			return nil, nil, cert, fmt.Errorf("assignment augmented=%d/%d: %w", row, d.N, err)
		}
		clear(visited)
		v := d.Values[row*d.Dim : (row+1)*d.Dim]
		for b := 0; b < c; b++ {
			dist[b] = distance(v, centers[b]) - potentials[b]
			previous[b] = -1
			moving[b] = row
		}
		end := -1
		for {
			a := -1
			for b := 0; b < c; b++ {
				if !visited[b] && (a < 0 || dist[b] < dist[a]) {
					a = b
				}
			}
			if a < 0 || math.IsNaN(dist[a]) || math.IsInf(dist[a], 0) {
				return nil, nil, cert, fmt.Errorf("invalid residual shortest path row=%d", row)
			}
			visited[a] = true
			if len(members[a]) < size {
				end = a
				break
			}
			for _, id := range members[a] {
				if err := ctx.Err(); err != nil {
					return nil, nil, cert, fmt.Errorf("assignment augmented=%d/%d: %w", row, d.N, err)
				}
				x := d.Values[id*d.Dim : (id+1)*d.Dim]
				base := distance(x, centers[a])
				for b := 0; b < c; b++ {
					if visited[b] {
						continue
					}
					cost := distance(x, centers[b])
					delta := potentials[a] - potentials[b]
					reduced := cost - base + delta
					scale := math.Max(1, math.Max(math.Abs(cost), math.Max(math.Abs(base), math.Abs(delta))))
					if reduced < -assignmentTolerance*scale || math.IsNaN(reduced) || math.IsInf(reduced, 0) {
						return nil, nil, cert, fmt.Errorf("negative residual row=%d from=%d to=%d cost=%g", id, a, b, reduced)
					}
					candidate := dist[a] + math.Max(0, reduced)
					if candidate < dist[b] {
						dist[b] = candidate
						previous[b] = a
						moving[b] = id
					}
				}
			}
		}
		for b := 0; b < c; b++ {
			potentials[b] += math.Min(dist[b], dist[end])
		}
		// Remove a displaced row from its source before inserting its replacement.
		for b := end; b >= 0; {
			id := moving[b]
			a := previous[b]
			if a >= 0 {
				for j, x := range members[a] {
					if x == id {
						members[a] = append(members[a][:j], members[a][j+1:]...)
						break
					}
				}
			}
			members[b] = append(members[b], id)
			owners[id] = b
			b = a
		}
		// Remove a common offset to limit potential magnitude without changing costs.
		offset := potentials[0]
		for b := range potentials {
			potentials[b] -= offset
		}
	}
	cert, err := VerifyAssignment(ctx, d, centers, size, owners, potentials)
	if err != nil {
		return nil, nil, cert, fmt.Errorf("fixed-centroid certification: %w", err)
	}
	return owners, potentials, cert, nil
}
