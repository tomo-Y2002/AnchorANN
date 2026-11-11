package annmetric

import (
	"math"
	"testing"
)

func TestNearestHit(t *testing.T) {
	for _, tc := range []struct {
		name        string
		answers, gt []int
		k, want     int
	}{
		{"in tenth position", []int{2, 3, 4, 5, 6, 7, 8, 9, 10, 1}, []int{1, 2}, 10, 1},
		{"outside top ten", []int{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1}, []int{1}, 10, 0},
		{"other GT IDs only", []int{2, 3}, []int{1, 2, 3}, 10, 0},
		{"duplicate nearest", []int{1, 1, 1}, []int{1, 2}, 10, 1},
		{"empty GT", []int{1}, nil, 10, 0},
		{"empty answers", nil, []int{1}, 10, 0},
		{"zero k", []int{1}, []int{1}, 0, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := NearestHit(tc.answers, tc.gt, tc.k); got != tc.want {
				t.Fatalf("got %d, want %d", got, tc.want)
			}
		})
	}
}

func TestAccuracyTopTenAndMissingInputs(t *testing.T) {
	truth := make([][]int, 100)
	answers := make([][]int, 100)
	for i := range truth {
		truth[i] = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
		switch i % 5 {
		case 0:
			answers[i] = []int{20, 21, 22, 23, 24, 25, 26, 27, 28, 1} // tenth
		case 1:
			answers[i] = []int{20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1} // eleventh
		case 2:
			answers[i] = []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1} // duplicates
		case 3:
			answers[i] = []int{2} // GT2, even if tied with GT1
		}
	}
	a := Evaluate(truth, answers, 100, 11)
	if a.One == nil || *a.One != .4 || math.Abs(*a.Ten-.06) > 1e-12 {
		t.Fatalf("accuracy = %+v columns=%v", a, a.Columns())
	}
	for _, tc := range []struct {
		gt, ans [][]int
		q, k    int
	}{
		{nil, answers, 100, 10}, {[][]int{{1}}, answers, 1, 10}, {truth, answers, 100, 9},
	} {
		a := Evaluate(tc.gt, tc.ans, tc.q, tc.k)
		if a.One != nil || a.Ten != nil || a.MissingReason == "" {
			t.Fatalf("invented accuracy: %+v", a)
		}
	}
	missingAnswers := Evaluate(truth, nil, 100, 10)
	if missingAnswers.One != nil || missingAnswers.MissingReason == "" {
		t.Fatal("missing answers are unverifiable")
	}
}

func TestOneRecall(t *testing.T) {
	gt := [][]uint32{{1, 2}, {3, 4}, {5, 6}, {7, 8}}
	answers := [][]uint32{{1, 1}, {4}, {6, 5}}
	if got := OneRecall(gt, answers, 10); got != 0.5 {
		t.Fatalf("got %v, want 0.5 (two hits, one miss, one missing row)", got)
	}
	if got := OneRecall[int](nil, nil, 10); got != 0 {
		t.Fatalf("empty batch = %v", got)
	}
}
