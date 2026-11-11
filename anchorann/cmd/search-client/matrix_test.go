package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/kshedden/gonpy"
)

func saveIntMatrix(path string, matrix [][]int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w, err := gonpy.NewWriter(f)
	if err != nil {
		return err
	}
	w.Shape = []int{len(matrix), len(matrix[0])}
	data := make([]int32, 0, len(matrix)*len(matrix[0]))
	for _, row := range matrix {
		for _, value := range row {
			data = append(data, int32(value))
		}
	}
	return w.WriteInt32(data)
}

func TestMatrixShapeAndTruncation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "matrix.npy")
	if err := saveIntMatrix(path, [][]int{{1, 2}, {3, 4}}); err != nil {
		t.Fatal(err)
	}
	for _, shape := range [][2]int{{3, 2}, {2, 3}, {0, 2}, {2, 0}} {
		if _, err := loadIntMatrix(path, shape[0], shape[1]); err == nil {
			t.Fatalf("accepted invalid prefix %v", shape)
		}
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(path, info.Size()-1); err != nil {
		t.Fatal(err)
	}
	if _, err := loadIntMatrix(path, 2, 2); err == nil {
		t.Fatal("accepted truncated payload")
	}
}
