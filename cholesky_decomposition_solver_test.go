package linear

import (
	"math"
	"testing"
)

var (
	testData_cs = [][]float64{
		{1, 2, 4, 7, 11},
		{2, 13, 23, 38, 58},
		{4, 23, 77, 122, 182},
		{7, 38, 122, 294, 430},
		{11, 58, 182, 430, 855},
	}
)

func TestCholeskyDecompositionSolverSolveDimensionErrors(t *testing.T) {
	cd, _ := NewDefaultCholeskyDecomposition(createRealMatrixFromSlices(t, testData_cs))
	solver := cd.Solver()
	m := make([][]float64, 2)
	for i := 0; i < 2; i++ {
		m[i] = make([]float64, 2)
	}
	b := createRealMatrixFromSlices(t, m)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	solver.SolveMatrix(b)
	solver.SolveVector(b.ColumnVectorAt(0))
	solver.SolveVector(createArrayRealVectorFromSlice(t, b.ColumnAt(0)))

}

func TestCholeskyDecompositionSolverSolve(t *testing.T) {
	cd, _ := NewDefaultCholeskyDecomposition(createRealMatrixFromSlices(t, testData_cs))
	solver := cd.Solver()
	b := createRealMatrixFromSlices(t, [][]float64{
		{78, -13, 1},
		{414, -62, -1},
		{1312, -202, -37},
		{2989, -542, 145},
		{5510, -1465, 201},
	})
	xRef := createRealMatrixFromSlices(t, [][]float64{
		{1, 0, 1},
		{0, 1, 1},
		{2, 1, -4},
		{2, 2, 2},
		{5, -3, 0},
	})

	// using RealMatrix
	res := solver.SolveMatrix(b).Subtract(xRef)
	if math.Abs(MatLInfNorm(res)-0) > 1.0e-13 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(res))
	}

	// using ArrayRealVector
	for i := 0; i < b.ColumnDimension(); i++ {
		resVec := solver.SolveVector(b.ColumnVectorAt(i)).Subtract(xRef.ColumnVectorAt(i))
		if math.Abs(VecNorm(resVec)-0) > 1.0e-13 {
			t.Errorf("Mismatch. want: %v, got: %v", 0, VecNorm(resVec))
		}

	}

}

func TestCholeskyDecompositionSolverDeterminant(t *testing.T) {
	m, _ := NewDefaultCholeskyDecomposition(createRealMatrixFromSlices(t, testData_cs))
	if math.Abs(m.Determinant()-7290000.0) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 7290000.0, m.Determinant())
	}

}
