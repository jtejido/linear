package linear

import (
	"math"
	"testing"
)

var (
	testData_ls = [][]float64{
		{1.0, 2.0, 3.0},
		{2.0, 5.0, 3.0},
		{1.0, 0.0, 8.0},
	}
	luData_ls = [][]float64{
		{2.0, 3.0, 3.0},
		{0.0, 5.0, 7.0},
		{6.0, 9.0, 8.0},
	}

	// singular matrices
	singular_ls = [][]float64{
		{2.0, 3.0},
		{2.0, 3.0},
	}

	bigSingular_ls = [][]float64{
		{1.0, 2.0, 3.0, 4.0},
		{2.0, 5.0, 3.0, 4.0},
		{7.0, 3.0, 256.0, 1930.0},
		{3.0, 7.0, 6.0, 8.0},
	} // 4th row = 1st + 2nd
)

func TestLUDecompositionSolverThreshold(t *testing.T) {
	m := createRealMatrixFromSlices(t, [][]float64{
		{1.0, 2.0, 3.0},
		{2.0, 5.0, 3.0},
		{4.000001, 9.0, 9.0},
	})
	lud, _ := NewLUDecompositionWithThreshold(m, 1.0e-5)
	if lud.Solver().IsNonSingular() {
		t.Errorf("Singular matrix not detected.")
	}
	lud, _ = NewLUDecompositionWithThreshold(m, 1.0e-10)
	if !lud.Solver().IsNonSingular() {
		t.Errorf("Singular matrix detected.")
	}

}

func TestLUDecompositionSolverSingular(t *testing.T) {
	lud, _ := NewLUDecomposition(createRealMatrixFromSlices(t, testData_ls))
	solver := lud.Solver()
	if !solver.IsNonSingular() {
		t.Errorf("Singular matrix detected.")
	}

	lud, _ = NewLUDecomposition(createRealMatrixFromSlices(t, singular_ls))
	solver = lud.Solver()
	if solver.IsNonSingular() {
		t.Errorf("Singular matrix not detected.")
	}

	lud, _ = NewLUDecomposition(createRealMatrixFromSlices(t, bigSingular_ls))
	solver = lud.Solver()
	if solver.IsNonSingular() {
		t.Errorf("Singular matrix not detected.")
	}
}

func TestLUDecompositionSolverSolveDimensionErrors(t *testing.T) {
	lud, _ := NewLUDecomposition(createRealMatrixFromSlices(t, testData_ls))
	solver := lud.Solver()
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

func TestLUDecompositionSolverSolveSingularityErrors(t *testing.T) {
	lud, _ := NewLUDecomposition(createRealMatrixFromSlices(t, singular_ls))
	solver := lud.Solver()
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

func TestLUDecompositionSolverSolve(t *testing.T) {
	lud, _ := NewLUDecomposition(createRealMatrixFromSlices(t, testData_ls))
	solver := lud.Solver()
	b := createRealMatrixFromSlices(t, [][]float64{
		{1, 0}, {2, -5}, {3, 1},
	})
	xRef := createRealMatrixFromSlices(t, [][]float64{
		{19, -71}, {-6, 22}, {-2, 9},
	})

	// using RealMatrix
	diffMatNorm := MatLInfNorm(solver.SolveMatrix(b).Subtract(xRef))
	if math.Abs(0.0-diffMatNorm) > 1.0e-13 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, diffMatNorm)
	}

	// using ArrayRealVector

	for i := 0; i < b.ColumnDimension(); i++ {
		diffVecNorm := VecNorm(solver.SolveVector(b.ColumnVectorAt(i)).Subtract(xRef.ColumnVectorAt(i)))
		if math.Abs(0.0-diffVecNorm) > 1.0e-13 {
			t.Errorf("Mismatch. want: %v, got: %v", 0.0, diffVecNorm)
		}

	}

}

func TestLUDecompositionSolveDeterminant(t *testing.T) {
	if math.Abs(-1-getDeterminant(createRealMatrixFromSlices(t, testData_ls))) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", -1, getDeterminant(createRealMatrixFromSlices(t, testData_ls)))
	}
	if math.Abs(-10-getDeterminant(createRealMatrixFromSlices(t, luData_ls))) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", -10, getDeterminant(createRealMatrixFromSlices(t, luData_ls)))
	}
	if math.Abs(0-getDeterminant(createRealMatrixFromSlices(t, singular_ls))) > 1.0e-17 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, getDeterminant(createRealMatrixFromSlices(t, singular_ls)))
	}
	if math.Abs(0-getDeterminant(createRealMatrixFromSlices(t, bigSingular_ls))) > 1.0e-10 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, getDeterminant(createRealMatrixFromSlices(t, bigSingular_ls)))
	}

}
func getDeterminant(m RealMatrix) float64 {
	lud, _ := NewLUDecomposition(m)
	return lud.Determinant()
}
