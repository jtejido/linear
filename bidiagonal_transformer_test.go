package linear

import (
	"math"
	"testing"
)

var (
	testSquare_bdt = [][]float64{
		{24.0 / 25.0, 43.0 / 25.0},
		{57.0 / 25.0, 24.0 / 25.0},
	}

	testNonSquare_bdt = [][]float64{
		{-540.0 / 625.0, 963.0 / 625.0, -216.0 / 625.0},
		{-1730.0 / 625.0, -744.0 / 625.0, 1008.0 / 625.0},
		{-720.0 / 625.0, 1284.0 / 625.0, -288.0 / 625.0},
		{-360.0 / 625.0, 192.0 / 625.0, 1756.0 / 625.0},
	}
)

func createBiDiagonalTransformer(t *testing.T, mat RealMatrix) *BiDiagonalTransformer {
	bdt, err := NewBiDiagonalTransformer(mat)
	if err != nil {
		t.Errorf("Error while creating Transformer %s", err)
	}

	return bdt
}

func checkdimensions(mat RealMatrix, t *testing.T) {
	m := mat.RowDimension()
	n := mat.ColumnDimension()
	transformer, _ := NewBiDiagonalTransformer(mat)
	if m != transformer.U().RowDimension() {
		t.Errorf("Mismatch. want: %v, got: %v", m, transformer.U().RowDimension())
	}

	if m != transformer.U().ColumnDimension() {
		t.Errorf("Mismatch. want: %v, got: %v", m, transformer.U().ColumnDimension())
	}

	if m != transformer.B().RowDimension() {
		t.Errorf("Mismatch. want: %v, got: %v", m, transformer.B().RowDimension())
	}

	if n != transformer.B().ColumnDimension() {
		t.Errorf("Mismatch. want: %v, got: %v", n, transformer.B().ColumnDimension())
	}

	if n != transformer.V().RowDimension() {
		t.Errorf("Mismatch. want: %v, got: %v", n, transformer.V().RowDimension())
	}

	if n != transformer.V().ColumnDimension() {
		t.Errorf("Mismatch. want: %v, got: %v", n, transformer.V().ColumnDimension())
	}
}

func checkAEqualUSVtBDT(m RealMatrix, t *testing.T) {
	transformer, _ := NewBiDiagonalTransformer(m)
	u := transformer.U()
	b := transformer.B()
	v := transformer.V()
	norm := MatLInfNorm(u.Multiply(b).Multiply(v.Transpose()).Subtract(m))
	if math.Abs(norm-0) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, norm)
	}
}

func checkOrthogonalBDT(m RealMatrix, t *testing.T) {
	mTm := m.Transpose().Multiply(m)
	id := createRealIdentityMatrix(t, mTm.RowDimension())
	if math.Abs(MatLInfNorm(mTm.Subtract(id))-0) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(mTm.Subtract(id)))
	}
}

func checkBiDiagonal(m RealMatrix, t *testing.T) {
	rows := m.RowDimension()
	cols := m.ColumnDimension()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if rows < cols {
				if (i < j) || (i > j+1) {
					if math.Abs(m.At(i, j)-0) > 1.0e-16 {
						t.Errorf("Mismatch. want: %v, got: %v", 0, m.At(i, j))
					}
				}
			} else {
				if (i < j-1) || (i > j) {
					if math.Abs(m.At(i, j)-0) > 1.0e-16 {
						t.Errorf("Mismatch. want: %v, got: %v", 0, m.At(i, j))
					}
				}
			}
		}
	}
}

func TestDimensions(t *testing.T) {
	checkdimensions(createRealMatrixFromSlices(t, testSquare_bdt), t)
	checkdimensions(createRealMatrixFromSlices(t, testNonSquare_bdt), t)
	checkdimensions(createRealMatrixFromSlices(t, testNonSquare_bdt).Transpose(), t)
}

func TestAEqualUSVt(t *testing.T) {
	checkAEqualUSVtBDT(createRealMatrixFromSlices(t, testSquare_bdt), t)
	checkAEqualUSVtBDT(createRealMatrixFromSlices(t, testNonSquare_bdt), t)
	checkAEqualUSVtBDT(createRealMatrixFromSlices(t, testNonSquare_bdt).Transpose(), t)
}

func TestUOrthogonal(t *testing.T) {
	checkOrthogonalBDT(createBiDiagonalTransformer(t, createRealMatrixFromSlices(t, testSquare_bdt)).U(), t)
	checkOrthogonalBDT(createBiDiagonalTransformer(t, createRealMatrixFromSlices(t, testNonSquare_bdt)).U(), t)
	checkOrthogonalBDT(createBiDiagonalTransformer(t, createRealMatrixFromSlices(t, testNonSquare_bdt).Transpose()).U(), t)
}

func TestVOrthogonal(t *testing.T) {
	checkOrthogonalBDT(createBiDiagonalTransformer(t, createRealMatrixFromSlices(t, testSquare_bdt)).V(), t)
	checkOrthogonalBDT(createBiDiagonalTransformer(t, createRealMatrixFromSlices(t, testNonSquare_bdt)).V(), t)
	checkOrthogonalBDT(createBiDiagonalTransformer(t, createRealMatrixFromSlices(t, testNonSquare_bdt).Transpose()).V(), t)
}

func TestBBiDiagonal(t *testing.T) {
	checkBiDiagonal(createBiDiagonalTransformer(t, createRealMatrixFromSlices(t, testSquare_bdt)).B(), t)
	checkBiDiagonal(createBiDiagonalTransformer(t, createRealMatrixFromSlices(t, testNonSquare_bdt)).B(), t)
	checkBiDiagonal(createBiDiagonalTransformer(t, createRealMatrixFromSlices(t, testNonSquare_bdt).Transpose()).B(), t)
}

func TestSingularMatrix(t *testing.T) {
	transformer, _ := NewBiDiagonalTransformer(createRealMatrixFromSlices(t, [][]float64{
		{1.0, 2.0, 3.0},
		{2.0, 3.0, 4.0},
		{3.0, 5.0, 7.0},
	}))
	s3 := math.Sqrt(3.0)
	s14 := math.Sqrt(14.0)
	s1553 := math.Sqrt(1553.0)
	uRef := createRealMatrixFromSlices(t, [][]float64{
		{-1.0 / s14, 5.0 / (s3 * s14), 1.0 / s3},
		{-2.0 / s14, -4.0 / (s3 * s14), 1.0 / s3},
		{-3.0 / s14, 1.0 / (s3 * s14), -1.0 / s3},
	})
	bRef := createRealMatrixFromSlices(t, [][]float64{
		{-s14, s1553 / s14, 0.0},
		{0.0, -87 * s3 / (s14 * s1553), -s3 * s14 / s1553},
		{0.0, 0.0, 0.0},
	})
	vRef := createRealMatrixFromSlices(t, [][]float64{
		{1.0, 0.0, 0.0},
		{0.0, -23 / s1553, 32 / s1553},
		{0.0, -32 / s1553, -23 / s1553},
	})

	// check values against known references
	u := transformer.U()
	if math.Abs(MatLInfNorm(u.Subtract(uRef))-0) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(u.Subtract(uRef)))
	}

	b := transformer.B()
	if math.Abs(MatLInfNorm(b.Subtract(bRef))-0) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(b.Subtract(bRef)))
	}

	v := transformer.V()
	if math.Abs(MatLInfNorm(v.Subtract(vRef))-0) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(v.Subtract(vRef)))
	}

	// check the same cached instance is returned the second time
	if !u.Equals(transformer.U()) {
		t.Errorf("Mismatch. Matrix not equal.")
	}
	if !b.Equals(transformer.B()) {
		t.Errorf("Mismatch. Matrix not equal.")
	}
	if !v.Equals(transformer.V()) {
		t.Errorf("Mismatch. Matrix not equal.")
	}
}

func TestMatricesValues(t *testing.T) {
	transformer, _ := NewBiDiagonalTransformer(createRealMatrixFromSlices(t, testSquare_bdt))
	s17 := math.Sqrt(17.0)
	uRef := createRealMatrixFromSlices(t, [][]float64{
		{-8 / (5 * s17), 19 / (5 * s17)},
		{-19 / (5 * s17), -8 / (5 * s17)},
	})
	bRef := createRealMatrixFromSlices(t, [][]float64{
		{-3 * s17 / 5, 32 * s17 / 85},
		{0.0, -5 * s17 / 17},
	})
	vRef := createRealMatrixFromSlices(t, [][]float64{
		{1.0, 0.0},
		{0.0, -1.0},
	})

	// check values against known references
	u := transformer.U()
	if math.Abs(MatLInfNorm(u.Subtract(uRef))-0) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(u.Subtract(uRef)))
	}

	b := transformer.B()
	if math.Abs(MatLInfNorm(b.Subtract(bRef))-0) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(b.Subtract(bRef)))
	}
	v := transformer.V()
	if math.Abs(MatLInfNorm(v.Subtract(vRef))-0) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(v.Subtract(vRef)))
	}

	// check the same cached instance is returned the second time
	if !u.Equals(transformer.U()) {
		t.Errorf("Mismatch. Matrix not equal.")
	}
	if !b.Equals(transformer.B()) {
		t.Errorf("Mismatch. Matrix not equal.")
	}
	if !v.Equals(transformer.V()) {
		t.Errorf("Mismatch. Matrix not equal.")
	}

}

func TestUpperOrLower(t *testing.T) {
	if !createBiDiagonalTransformer(t, createRealMatrixFromSlices(t, testSquare_bdt)).IsUpperBiDiagonal() {
		t.Errorf("Mismatch. Matrix not Upper Bi-Diagonal.")
	}

	if !createBiDiagonalTransformer(t, createRealMatrixFromSlices(t, testNonSquare_bdt)).IsUpperBiDiagonal() {
		t.Errorf("Mismatch. Matrix not Upper Bi-Diagonal.")
	}

	if createBiDiagonalTransformer(t, createRealMatrixFromSlices(t, testNonSquare_bdt).Transpose()).IsUpperBiDiagonal() {
		t.Errorf("Mismatch. Matrix Upper Bi-Diagonal.")
	}

}
