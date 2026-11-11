// Package annmetric implements accuracy metrics without changing ANN search.
package annmetric

import (
	"fmt"
	"io"
	"strconv"
)

// NearestHit returns 1 when the first ground-truth ID occurs in the first k
// answers, and 0 otherwise. Only the GT file's first ID counts, including ties;
// duplicate answers never contribute more than one hit.
func NearestHit[T ~int | ~int32 | ~uint32](answers, gt []T, k int) int {
	if len(gt) == 0 || k <= 0 {
		return 0
	}
	for _, id := range answers[:min(k, len(answers))] {
		if id == gt[0] {
			return 1
		}
	}
	return 0
}

// OneRecall returns 1Recall@k: the fraction of queries whose true nearest
// neighbor is among the first k answers. Missing answer rows count as misses.
func OneRecall[T ~int | ~int32 | ~uint32](gt, answers [][]T, k int) float64 {
	if len(gt) == 0 {
		return 0
	}
	hits := 0
	for i, row := range gt {
		if i < len(answers) {
			hits += NearestHit(answers[i], row, k)
		}
	}
	return float64(hits) / float64(len(gt))
}

// Accuracy carries explicit top-ten metrics without changing legacy recall fields.
// Nil values mean that the ground truth or answers cannot be verified.
type Accuracy struct {
	One           *float64 `json:"one_recall_at_10"`
	Ten           *float64 `json:"ten_recall_at_10"`
	Metric        string   `json:"recall_metric"`
	MissingReason string   `json:"recall_missing_reason,omitempty"`
}

// Evaluate uses the existing GT order and returned answer order for q queries.
// Missing GT makes the batch unverifiable; missing answer rows are misses.
func Evaluate[T ~int | ~int32 | ~uint32](gt, answers [][]T, q, returnedK int) Accuracy {
	a := Accuracy{Metric: "1Recall@10"}
	if q <= 0 || len(gt) < q {
		a.MissingReason = "ground truth rows missing"
		return a
	}
	if answers == nil {
		a.MissingReason = "returned answer IDs unavailable"
		return a
	}
	if returnedK < 10 {
		a.MissingReason = "fewer than ten answers requested"
		return a
	}
	one, ten := 0.0, 0.0
	for i := 0; i < q; i++ {
		if len(gt[i]) < 10 {
			a.MissingReason = "ground truth has fewer than ten IDs"
			return a
		}
		if i >= len(answers) {
			continue
		}
		one += float64(NearestHit(answers[i], gt[i], 10))
		expected := make(map[T]bool, 10)
		for _, id := range gt[i][:10] {
			expected[id] = true
		}
		seen := make(map[T]bool, 10)
		for _, id := range answers[i][:min(10, len(answers[i]))] {
			if expected[id] && !seen[id] {
				ten += 0.1
				seen[id] = true
			}
		}
	}
	one /= float64(q)
	ten /= float64(q)
	a.One, a.Ten = &one, &ten
	return a
}

// Columns returns explicit CSV fields, keeping unavailable values blank.
func (a Accuracy) Columns() map[string]string {
	metric := a.Metric
	if metric == "" {
		metric = "1Recall@10"
	}
	out := map[string]string{"recall_metric": metric, "1Recall@10": "", "10Recall@10": "", "recall_missing_reason": a.MissingReason}
	if a.One != nil {
		out["1Recall@10"] = strconv.FormatFloat(*a.One, 'f', 6, 64)
	}
	if a.Ten != nil {
		out["10Recall@10"] = strconv.FormatFloat(*a.Ten, 'f', 6, 64)
	}
	if a.One == nil && out["recall_missing_reason"] == "" {
		out["recall_missing_reason"] = "GT/answer IDs not supplied to evaluation"
	}
	return out
}

// WriteText appends formal metrics to a report; legacy Recall is left intact.
func (a Accuracy) WriteText(w io.Writer) {
	c := a.Columns()
	fmt.Fprintf(w, "Recall Metric: %s\n", c["recall_metric"])
	fmt.Fprintf(w, "1Recall@10: %s\n10Recall@10: %s\n", c["1Recall@10"], c["10Recall@10"])
	if c["recall_missing_reason"] != "" {
		fmt.Fprintf(w, "Recall Missing Reason: %s\n", c["recall_missing_reason"])
	}
}
