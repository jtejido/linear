package linear

import (
	"math"
	"testing"
)

var (
	testSquare_svs = [][]float64{
		{24.0 / 25.0, 43.0 / 25.0},
		{57.0 / 25.0, 24.0 / 25.0},
	}

	bigSingular_svs = [][]float64{
		{1.0, 2.0, 3.0, 4.0},
		{2.0, 5.0, 3.0, 4.0},
		{7.0, 3.0, 256.0, 1930.0},
		{3.0, 7.0, 6.0, 8.0},
	} // 4th row = 1st + 2nd
)

func createSingularValueDecomposition(t *testing.T, mat RealMatrix) *SingularValueDecomposition {
	svd, err := NewSingularValueDecomposition(mat)
	if err != nil {
		t.Errorf("Error while creating SVD %s", err)
	}

	return svd
}

func TestSingularValueDecompositionSolverSolveDimensionErrors(t *testing.T) {
	solver := createSingularValueDecomposition(t, createRealMatrixFromSlices(t, testSquare_svs)).Solver()
	m := make([][]float64, 3)
	for i := 0; i < 3; i++ {
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

func TestSingularValueDecompositionSolverLeastSquareSolve(t *testing.T) {
	m := createRealMatrixFromSlices(t, [][]float64{
		{1.0, 0.0},
		{0.0, 0.0},
	})
	solver := createSingularValueDecomposition(t, m).Solver()
	b := createRealMatrixFromSlices(t, [][]float64{
		{11, 12}, {21, 22},
	})
	xMatrix := solver.SolveMatrix(b)
	if math.Abs(11-xMatrix.At(0, 0)) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 11, xMatrix.At(0, 0))
	}

	if math.Abs(12-xMatrix.At(0, 1)) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 12, xMatrix.At(0, 1))
	}

	if math.Abs(0-xMatrix.At(1, 0)) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, xMatrix.At(1, 0))
	}

	if math.Abs(0-xMatrix.At(1, 1)) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, xMatrix.At(1, 1))
	}

	xColVec := solver.SolveVector(b.ColumnVectorAt(0))
	if math.Abs(11-xColVec.At(0)) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 11, xColVec.At(0))
	}

	if math.Abs(0-xColVec.At(1)) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, xColVec.At(1))
	}
}

func TestSingularValueDecompositionSolverSolve(t *testing.T) {
	solver := createSingularValueDecomposition(t, createRealMatrixFromSlices(t, testSquare_svs)).Solver()
	b := createRealMatrixFromSlices(t, [][]float64{
		{1, 2, 3}, {0, -5, 1},
	})
	xRef := createRealMatrixFromSlices(t, [][]float64{
		{-8.0 / 25.0, -263.0 / 75.0, -29.0 / 75.0},
		{19.0 / 25.0, 78.0 / 25.0, 49.0 / 25.0},
	})

	// using RealMatrix
	if math.Abs(0-MatLInfNorm(solver.SolveMatrix(b).Subtract(xRef))) > normTolerance {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(solver.SolveMatrix(b).Subtract(xRef)))
	}

	// using ArrayRealVector

	for i := 0; i < b.ColumnDimension(); i++ {
		diffVecNorm := VecNorm(solver.SolveVector(b.ColumnVectorAt(i)).Subtract(xRef.ColumnVectorAt(i)))
		if math.Abs(0.0-diffVecNorm) > 1.0e-13 {
			t.Errorf("Mismatch. want: %v, got: %v", 0.0, diffVecNorm)
		}

	}

}

func TestSingularValueDecompositionSolverConditionNumber(t *testing.T) {
	svd := createSingularValueDecomposition(t, createRealMatrixFromSlices(t, testSquare_svs))
	// replace 1.0e-15 with 1.5e-15
	if math.Abs(3.0-svd.ConditionNumber()) > 1.5e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 3.0, svd.ConditionNumber())
	}

}

func TestSingularValueDecompositionSolverMath320B(t *testing.T) {
	rm := createArray2DRowRealMatrixFromSlices(t, [][]float64{
		{1.0, 2.0}, {1.0, 2.0},
	})
	svd := createSingularValueDecomposition(t, rm)
	recomposed := svd.U().Multiply(svd.S()).Multiply(svd.VT())
	if math.Abs(0.0-MatLInfNorm(recomposed.Subtract(rm))) > 2.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, MatLInfNorm(recomposed.Subtract(rm)))
	}

}

func TestSingularValueDecompositionSolverSingular(t *testing.T) {
	svd := createSingularValueDecomposition(t, createRealMatrixFromSlices(t, bigSingular))
	pseudoInverse := svd.Solver().Inverse()
	expected := createArray2DRowRealMatrixFromSlices(t, [][]float64{
		{-0.0355022687, 0.0512742236, -0.0001045523, 0.0157719549},
		{-0.3214992438, 0.3162419255, 0.0000348508, -0.0052573183},
		{0.5437098346, -0.4107754586, -0.0008256918, 0.132934376},
		{-0.0714905202, 0.053808742, 0.0006279816, -0.0176817782},
	})
	if math.Abs(0.0-MatLInfNorm(expected.Subtract(pseudoInverse))) > 1.0e-9 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, MatLInfNorm(expected.Subtract(pseudoInverse)))
	}

}
