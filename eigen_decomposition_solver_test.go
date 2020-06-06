package linear

import (
	"math"
	"math/rand"
	"testing"
)

var (
	bigSingular_es = [][]float64{
		{1.0, 2.0, 3.0, 4.0},
		{2.0, 5.0, 3.0, 4.0},
		{7.0, 3.0, 256.0, 1930.0},
		{3.0, 7.0, 6.0, 8.0},
	}
)

func TestEigenDecompositionSolverNonInvertible(t *testing.T) {
	r := rand.New(rand.NewSource(9994100315209))
	m := createTestMatrix(r, []float64{1.0, 0.0, -1.0, -2.0, -3.0})
	ed, _ := NewEigenDecomposition(m)
	es := ed.Solver()
	if es.IsNonSingular() {
		t.Errorf("Should be singular")
	}
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	es.Inverse()

}

func TestEigenDecompositionSolverInvertible(t *testing.T) {
	r := rand.New(rand.NewSource(9994100315209))
	m := createTestMatrix(r, []float64{1.0, 0.5, -1.0, -2.0, -3.0})
	ed, _ := NewEigenDecomposition(m)
	es := ed.Solver()
	if !es.IsNonSingular() {
		t.Errorf("Should not be singular")
	}

	inverse := es.Inverse()
	error := m.Multiply(inverse).Subtract(createRealIdentityMatrix(t, m.RowDimension()))
	if math.Abs(0.0-MatLInfNorm(error)) > 4.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, MatLInfNorm(error))
	}

}

func TestEigenDecompositionSolverInvertibleTinyValues(t *testing.T) {
	tiny := 1e-100
	m := createRealMatrixFromSlices(t, [][]float64{
		{3, 2, 4},
		{2, 0, 2},
		{4, 2, 3},
	})
	m = m.ScalarMultiply(tiny)

	ed, _ := NewEigenDecomposition(m)
	inv := ed.Solver().Inverse()

	id := m.Multiply(inv)
	for i := 0; i < m.RowDimension(); i++ {
		for j := 0; j < m.ColumnDimension(); j++ {
			if i == j {
				if !equalsWithError(1, id.At(i, j), 1e-15) {
					t.Errorf("Mismatch. want: %v, got: %v", 1, id.At(i, j))
				}
			} else {
				if !equalsWithError(0, id.At(i, j), 1e-15) {
					t.Errorf("Mismatch. want: %v, got: %v", 1, id.At(i, j))
				}
			}
		}
	}
}

func TestEigenDecompositionSolverNonInvertibleMath1045(t *testing.T) {
	eigen, _ := NewEigenDecomposition(createRealMatrixFromSlices(t, bigSingular_es))
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	eigen.Solver().Inverse()
}

func TestEigenDecompositionSolverZeroMatrix(t *testing.T) {
	eigen, _ := NewEigenDecomposition(createRealMatrixFromSlices(t, [][]float64{{0}}))
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	eigen.Solver().Inverse()
}

func TestEigenDecompositionSolverIsNonSingularTinyOutOfOrderEigenvalue(t *testing.T) {
	eigen, _ := NewEigenDecomposition(createRealMatrixFromSlices(t, [][]float64{
		{1e-13, 0},
		{1, 1},
	}))

	if eigen.Solver().IsNonSingular() {
		t.Errorf("Singular matrix not detected")
	}

}

func TestEigenDecompositionSolverSolveDimensionErrors(t *testing.T) {
	refValues := []float64{
		2.003, 2.002, 2.001, 1.001, 1.000, 0.001,
	}
	m := make([][]float64, 2)
	for i := 0; i < 2; i++ {
		m[i] = make([]float64, 2)
	}

	m2 := createTestMatrix(rand.New(rand.NewSource(35992629946426)), refValues)

	ed, _ := NewEigenDecomposition(m2)
	es := ed.Solver()
	b := createRealMatrixFromSlices(t, m)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	es.SolveMatrix(b)
	es.SolveVector(b.ColumnVectorAt(0))
	es.SolveVector(createArrayRealVectorFromSlice(t, b.ColumnAt(0)))

}

func TestEigenDecompositionSolverSolve(t *testing.T) {
	m := createRealMatrixFromSlices(t, [][]float64{
		{91, 5, 29, 32, 40, 14},
		{5, 34, -1, 0, 2, -1},
		{29, -1, 12, 9, 21, 8},
		{32, 0, 9, 14, 9, 0},
		{40, 2, 21, 9, 51, 19},
		{14, -1, 8, 0, 19, 14},
	})
	ed, _ := NewEigenDecomposition(m)
	es := ed.Solver()
	b := createRealMatrixFromSlices(t, [][]float64{
		{1561, 269, 188},
		{69, -21, 70},
		{739, 108, 63},
		{324, 86, 59},
		{1624, 194, 107},
		{796, 69, 36},
	})

	xRef := createRealMatrixFromSlices(t, [][]float64{
		{1, 2, 1},
		{2, -1, 2},
		{4, 2, 3},
		{8, -1, 0},
		{16, 2, 0},
		{32, -1, 0},
	})

	// using RealMatrix
	solution := es.SolveMatrix(b)
	if math.Abs(0.0-MatLInfNorm(solution.Subtract(xRef))) > 2.5e-12 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, MatLInfNorm(solution.Subtract(xRef)))
	}

	for i := 0; i < b.ColumnDimension(); i++ {
		// using RealVector
		vecNorm := VecNorm(es.SolveVector(b.ColumnVectorAt(i)).Subtract(xRef.ColumnVectorAt(i)))
		if math.Abs(0.0-vecNorm) > 2.0e-11 {
			t.Errorf("Mismatch. want: %v, got: %v", 0.0, vecNorm)
		}

	}

}
