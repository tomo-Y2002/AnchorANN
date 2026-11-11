//go:build !amd64

package pianopir

func xorSlices(dst, src []uint64, n int) {
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
