//go:build !amd64

package anchorann

func batchDistances(points []float32, centers, costs []float64, rows, count, dim int) {
	batchDistancesScalar(points, centers, costs, rows, count, dim)
}
