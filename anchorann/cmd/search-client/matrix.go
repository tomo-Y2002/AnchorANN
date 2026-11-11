package main

import (
	"fmt"
	"os"

	"github.com/kshedden/gonpy"
)

// loadIntMatrix reads an int32 NumPy matrix prefix without native ANN libraries.
func loadIntMatrix(path string, rows, cols int) ([][]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open matrix: %w", err)
	}
	defer f.Close()
	r, err := gonpy.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("read matrix header: %w", err)
	}
	if rows <= 0 || cols <= 0 || len(r.Shape) != 2 || r.Shape[0] < rows || r.Shape[1] < cols || r.ColumnMajor {
		return nil, fmt.Errorf("invalid matrix shape %v for %dx%d row-major prefix", r.Shape, rows, cols)
	}
	data, err := r.GetInt32()
	if err != nil {
		return nil, fmt.Errorf("read matrix values: %w", err)
	}
	out := make([][]int, rows)
	for i := range out {
		out[i] = make([]int, cols)
		for j := range out[i] {
			out[i][j] = int(data[i*r.Shape[1]+j])
		}
	}
	return out, nil
}
