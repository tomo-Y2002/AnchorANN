package pianopir

import (
	"math/rand"
	"testing"
)

// xorSlicesReference is a pure-Go reference implementation of xorSlices.
// It mirrors the corrected fallback (xor_fallback.go) logic and serves as
// a cross-platform oracle so that the assembly and fallback paths can both
// be validated against the same expected behaviour.
func xorSlicesReference(dst, src []uint64, n int) {
	count := n
	if count > len(dst) {
		count = len(dst)
	}
	if count > len(src) {
		count = len(src)
	}
	for i := 0; i < count; i++ {
		dst[i] ^= src[i]
	}
}

// TestXorSlicesFallbackMatchesReference verifies that xorSlices (whichever
// build-tag variant is active) produces the same result as the pure-Go
// reference for several representative sizes, including the SIFT1M entry
// size that triggered the original bug.
func TestXorSlicesFallbackMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	// All sizes are multiples of 4 because both the AVX2 assembly (amd64)
	// and production entry sizes are always aligned to 4 uint64 elements.
	// 80 is the SIFT1M entry size that exposed the original n/8 bug.
	sizes := []int{4, 8, 16, 32, 64, 80, 112, 128, 256}

	for _, n := range sizes {
		// Build identical random slices for both implementations.
		dstActual := make([]uint64, n)
		dstExpect := make([]uint64, n)
		src := make([]uint64, n)
		for i := 0; i < n; i++ {
			v := rng.Uint64()
			dstActual[i] = v
			dstExpect[i] = v
			src[i] = rng.Uint64()
		}

		xorSlices(dstActual, src, n)
		xorSlicesReference(dstExpect, src, n)

		for i := 0; i < n; i++ {
			if dstActual[i] != dstExpect[i] {
				t.Errorf("n=%d: mismatch at index %d: got %x, want %x", n, i, dstActual[i], dstExpect[i])
			}
		}
	}
}

// TestXorSlicesFallbackAllElementsProcessed ensures that every element in
// the slice is XOR-ed, not just the first n/8 elements (the original bug).
func TestXorSlicesFallbackAllElementsProcessed(t *testing.T) {
	// Use n=80 (SIFT1M entry size). The old buggy code would only process
	// 80/8 = 10 elements, leaving indices 10..79 unchanged.
	n := 80
	dst := make([]uint64, n)
	src := make([]uint64, n)
	for i := 0; i < n; i++ {
		dst[i] = 0
		src[i] = uint64(i + 1) // non-zero so we can detect missing XOR
	}

	xorSlices(dst, src, n)

	for i := 0; i < n; i++ {
		if dst[i] != uint64(i+1) {
			t.Errorf("element %d not XOR-ed: got %d, want %d", i, dst[i], i+1)
		}
	}
}
