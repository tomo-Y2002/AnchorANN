package anchorann

func batchDistancesScalar(points []float32, centers, costs []float64, rows, count, dim int) {
	for group := 0; group < count/4; group++ {
		for row := 0; row < rows; row++ {
			for lane := 0; lane < 4; lane++ {
				sum := 0.0
				for j := 0; j < dim; j++ {
					delta := float64(points[row*dim+j]) - centers[(group*dim+j)*4+lane]
					sum += delta * delta
				}
				costs[row*count+group*4+lane] = sum
			}
		}
	}
}
