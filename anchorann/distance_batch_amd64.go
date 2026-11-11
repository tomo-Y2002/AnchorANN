//go:build amd64

package anchorann

import "golang.org/x/sys/cpu"

//go:noescape
func distancesAVX2(points, centers, costs []float64, rows, count, dim int)

func batchDistances(points []float32, centers, costs []float64, rows, count, dim int) {
	if !cpu.X86.HasAVX2 {
		batchDistancesScalar(points, centers, costs, rows, count, dim)
		return
	}
	doubles := make([]float64, len(points))
	for i, x := range points {
		doubles[i] = float64(x)
	}
	distancesAVX2(doubles, centers, costs, rows, count, dim)
}
