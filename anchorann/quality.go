package anchorann

import (
	"container/heap"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
)

// SSE returns total squared L2 error to each point's own cluster centroid.
func SSE(d Dataset, a *Artifact) (float64, error) {
	if err := validatePair(d, a); err != nil {
		return 0, err
	}
	var total float64
	for c, ids := range a.Members {
		for _, id := range ids {
			total += distance(d.Values[int(id)*d.Dim:(int(id)+1)*d.Dim], a.Anchors[c])
		}
	}
	return total, nil
}

func validatePair(d Dataset, a *Artifact) error {
	if err := d.Validate(); err != nil {
		return err
	}
	if err := a.Validate(); err != nil {
		return err
	}
	if d.N != a.N || d.Dim != a.Dim {
		return fmt.Errorf("dataset/artifact mismatch: dataset N=%d dimension=%d artifact N=%d dimension=%d", d.N, d.Dim, a.N, a.Dim)
	}
	return nil
}

type worstHeap []scoredID

func (h worstHeap) Len() int           { return len(h) }
func (h worstHeap) Less(i, j int) bool { return less(h[j], h[i]) }
func (h worstHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *worstHeap) Push(x any)        { *h = append(*h, x.(scoredID)) }
func (h *worstHeap) Pop() any          { a := *h; x := a[len(a)-1]; *h = a[:len(a)-1]; return x }

// GroundTruth stores exact top-k IDs and fingerprints the selected input and
// query bytes. It can be reused across methods/iterations, never across subsets.
type GroundTruth struct {
	N         int        `json:"n"`
	Dim       int        `json:"dimension"`
	K         int        `json:"k"`
	InputHash string     `json:"input_sha256"`
	QueryHash string     `json:"query_sha256"`
	IDs       [][]uint32 `json:"ids"`
}

func datasetHash(d Dataset) string {
	h := sha256.New()
	var b [4]byte
	for _, x := range d.Values {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(x))
		h.Write(b[:])
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

// ExactGroundTruth searches every selected input row, breaking ties by row ID.
func ExactGroundTruth(d, queries Dataset, k int) (*GroundTruth, error) {
	if err := d.Validate(); err != nil {
		return nil, err
	}
	if err := queries.Validate(); err != nil {
		return nil, err
	}
	if queries.Dim != d.Dim || k <= 0 || k > d.N {
		return nil, fmt.Errorf("invalid containment inputs: k=%d N=%d query dimension=%d expected=%d", k, d.N, queries.Dim, d.Dim)
	}
	gt := &GroundTruth{N: d.N, Dim: d.Dim, K: k, InputHash: datasetHash(d), QueryHash: datasetHash(queries), IDs: make([][]uint32, queries.N)}
	query := make([]float64, d.Dim)
	for q := 0; q < queries.N; q++ {
		for j, x := range queries.Values[q*d.Dim : (q+1)*d.Dim] {
			query[j] = float64(x)
		}
		h := make(worstHeap, 0, k)
		for id := 0; id < d.N; id++ {
			x := scoredID{uint32(id), distance(d.Values[id*d.Dim:(id+1)*d.Dim], query)}
			if len(h) < k {
				heap.Push(&h, x)
			} else if less(x, h[0]) {
				h[0] = x
				heap.Fix(&h, 0)
			}
		}
		sort.Slice(h, func(i, j int) bool { return less(h[i], h[j]) })
		gt.IDs[q] = make([]uint32, k)
		for j, x := range h {
			gt.IDs[q][j] = x.id
		}
	}
	return gt, nil
}

// ValidateGroundTruth rejects a cached GT for different input/query contents.
func ValidateGroundTruth(d, queries Dataset, gt *GroundTruth) error {
	if err := d.Validate(); err != nil {
		return err
	}
	if err := queries.Validate(); err != nil {
		return err
	}
	if gt == nil || gt.N != d.N || gt.Dim != d.Dim || queries.Dim != d.Dim || gt.K <= 0 || gt.K > d.N || len(gt.IDs) != queries.N || gt.InputHash != datasetHash(d) || gt.QueryHash != datasetHash(queries) {
		return fmt.Errorf("ground truth dataset/query mismatch")
	}
	for q, ids := range gt.IDs {
		if len(ids) != gt.K {
			return fmt.Errorf("ground truth query=%d length=%d expected=%d", q, len(ids), gt.K)
		}
		seen := make(map[uint32]bool, len(ids))
		for _, id := range ids {
			if uint64(id) >= uint64(d.N) || seen[id] {
				return fmt.Errorf("invalid ground truth ID=%d query=%d", id, q)
			}
			seen[id] = true
		}
	}
	return nil
}

// TopOneGroundTruth validates a cached distance/ID-sorted top-10 GT and copies
// the first ID of each row. The original cache and its fingerprints are preserved.
func TopOneGroundTruth(d, queries Dataset, gt *GroundTruth) (*GroundTruth, error) {
	if err := ValidateGroundTruth(d, queries, gt); err != nil {
		return nil, err
	}
	if gt.K != 10 {
		return nil, fmt.Errorf("expected distance-sorted GT top-10 cache, got k=%d", gt.K)
	}
	one := *gt
	one.K = 1
	one.IDs = make([][]uint32, len(gt.IDs))
	for row, ids := range gt.IDs {
		query := make([]float64, d.Dim)
		for j, x := range queries.Values[row*d.Dim : (row+1)*d.Dim] {
			query[j] = float64(x)
		}
		previous := -1.0
		var previousID uint32
		for _, id := range ids {
			cost := distance(d.Values[int(id)*d.Dim:(int(id)+1)*d.Dim], query)
			if cost < previous || (cost == previous && id < previousID) {
				return nil, fmt.Errorf("GT not distance/ID sorted at query=%d id=%d", row, id)
			}
			previous, previousID = cost, id
		}
		one.IDs[row] = []uint32{ids[0]}
	}
	return &one, nil
}

// ContainmentRates measures GT top-k membership in top 1/4/8 centroid unions.
type ContainmentRates struct {
	At1 float64 `json:"at_1"`
	At4 float64 `json:"at_4"`
	At8 float64 `json:"at_8"`
}

// Containment uses an already validated GT for the exact same queries/input.
// It ranks ALL centroids by squared L2, with cluster-ID ties. The caller should
// ValidateGroundTruth once before reusing it across completed iterations.
func Containment(a *Artifact, queries Dataset, gt *GroundTruth) (ContainmentRates, error) {
	var rates ContainmentRates
	if err := a.Validate(); err != nil {
		return rates, err
	}
	if err := queries.Validate(); err != nil {
		return rates, err
	}
	if gt == nil || gt.N != a.N || gt.Dim != a.Dim || queries.Dim != a.Dim || gt.K <= 0 || gt.K > a.N || len(gt.IDs) != queries.N || gt.QueryHash != datasetHash(queries) {
		return rates, fmt.Errorf("containment shape mismatch")
	}
	owner := make([]int, a.N)
	for c, ids := range a.Members {
		for _, id := range ids {
			owner[id] = c
		}
	}
	rank := make([]int, a.Count)
	scores := make([]scoredID, a.Count)
	hits := [3]int{}
	for q := 0; q < queries.N; q++ {
		v := queries.Values[q*a.Dim : (q+1)*a.Dim]
		for c, m := range a.Anchors {
			scores[c] = scoredID{uint32(c), distance(v, m)}
		}
		sort.Slice(scores, func(i, j int) bool { return less(scores[i], scores[j]) })
		for i, x := range scores {
			rank[x.id] = i
		}
		if len(gt.IDs[q]) != gt.K {
			return rates, fmt.Errorf("invalid GT row=%d", q)
		}
		seen := make(map[uint32]bool, gt.K)
		for _, id := range gt.IDs[q] {
			if uint64(id) >= uint64(a.N) || seen[id] {
				return rates, fmt.Errorf("invalid GT ID=%d", id)
			}
			seen[id] = true
			r := rank[owner[id]]
			for i, p := range []int{1, 4, 8} {
				if r < p {
					hits[i]++
				}
			}
		}
	}
	denom := float64(queries.N) * float64(gt.K)
	rates = ContainmentRates{float64(hits[0]) / denom, float64(hits[1]) / denom, float64(hits[2]) / denom}
	if rates.At1 > rates.At4 || rates.At4 > rates.At8 {
		return rates, fmt.Errorf("nonmonotone containment")
	}
	return rates, nil
}

// TopKContainment retains the original top-one-cluster convenience API.
func TopKContainment(d Dataset, a *Artifact, queries Dataset, k int) (float64, error) {
	if err := validatePair(d, a); err != nil {
		return 0, err
	}
	gt, err := ExactGroundTruth(d, queries, k)
	if err != nil {
		return 0, err
	}
	rates, err := Containment(a, queries, gt)
	return rates.At1, err
}
