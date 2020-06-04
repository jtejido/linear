package linear

import (
	"math"
	"testing"
)

func TestLUDecompositionDimensions(t *testing.T) {
	m := createRealMatrixFromSlices(t, testData)
	LU, _ := NewLUDecomposition(m)

	if LU.L().RowDimension() != len(testData) {
		t.Errorf("Mismatch. testData row dimension. want: %v, got: %v", len(testData), LU.L().RowDimension())
	}

	if LU.L().ColumnDimension() != len(testData) {
		t.Errorf("Mismatch. testData column dimension. want: %v, got: %v", len(testData), LU.L().ColumnDimension())
	}

	if LU.U().RowDimension() != len(testData) {
		t.Errorf("Mismatch. testData row dimension. want: %v, got: %v", len(testData), LU.U().RowDimension())
	}

	if LU.U().ColumnDimension() != len(testData) {
		t.Errorf("Mismatch. testData column dimension. want: %v, got: %v", len(testData), LU.U().ColumnDimension())
	}
	if LU.P().RowDimension() != len(testData) {
		t.Errorf("Mismatch. testData row dimension. want: %v, got: %v", len(testData), LU.P().RowDimension())
	}

	if LU.P().ColumnDimension() != len(testData) {
		t.Errorf("Mismatch. testData column dimension. want: %v, got: %v", len(testData), LU.P().ColumnDimension())
	}

}

func TestLUDecompositionNonSquare(t *testing.T) {
	m := make([][]float64, 3)
	for i := 0; i < 3; i++ {
		m[i] = make([]float64, 2)
	}

	_, err := NewLUDecomposition(createRealMatrixFromSlices(t, m))
	if err == nil {
		t.Errorf("error expected.")
	}
}

func TestLUDecompositionPAEqualLU(t *testing.T) {
	m := createRealMatrixFromSlices(t, testData)
	lu, _ := NewLUDecomposition(m)
	l := lu.L()
	u := lu.U()
	p := lu.P()
	norm := MatLInfNorm(l.Multiply(u).Subtract(p.Multiply(m)))

	if math.Abs(norm-0) > normTolerance {
		t.Errorf("Mismatch. want: %v, got: %v", 0, norm)
	}

	m = createRealMatrixFromSlices(t, testDataMinus)
	lu, _ = NewLUDecomposition(m)
	l = lu.L()
	u = lu.U()
	p = lu.P()
	norm = MatLInfNorm(l.Multiply(u).Subtract(p.Multiply(m)))
	if math.Abs(norm-0) > normTolerance {
		t.Errorf("Mismatch. want: %v, got: %v", 0, norm)
	}

	m = createRealIdentityMatrix(t, 17)
	lu, _ = NewLUDecomposition(m)
	l = lu.L()
	u = lu.U()
	p = lu.P()
	norm = MatLInfNorm(l.Multiply(u).Subtract(p.Multiply(m)))
	if math.Abs(norm-0) > normTolerance {
		t.Errorf("Mismatch. want: %v, got: %v", 0, norm)
	}

	m = createRealMatrixFromSlices(t, singular)
	lu, _ = NewLUDecomposition(m)

	if lu.Solver().IsNonSingular() {
		t.Errorf("Mismatch. should be singular")
	}

	if lu.L() != nil {
		t.Errorf("Mismatch. L should be nil")
	}

	if lu.U() != nil {
		t.Errorf("Mismatch. L should be nil")
	}

	if lu.P() != nil {
		t.Errorf("Mismatch. L should be nil")
	}

	m = createRealMatrixFromSlices(t, bigSingular)
	lu, _ = NewLUDecomposition(m)

	if lu.Solver().IsNonSingular() {
		t.Errorf("Mismatch. should be singular")
	}

	if lu.L() != nil {
		t.Errorf("Mismatch. L should be nil")
	}

	if lu.U() != nil {
		t.Errorf("Mismatch. L should be nil")
	}

	if lu.P() != nil {
		t.Errorf("Mismatch. L should be nil")
	}

}

/** test that L is lower triangular with unit diagonal */
func TestLUDecompositionLLowerTriangular(t *testing.T) {
	m := createRealMatrixFromSlices(t, testData)
	lud, _ := NewLUDecomposition(m)
	l := lud.L()
	for i := 0; i < l.RowDimension(); i++ {
		if math.Abs(l.At(i, i)-1) > entryTolerance {
			t.Errorf("Mismatch. want: %v, got: %v", 1, l.At(i, i))
		}

		for j := i + 1; j < l.ColumnDimension(); j++ {
			if math.Abs(l.At(i, j)-0) > entryTolerance {
				t.Errorf("Mismatch. want: %v, got: %v", 0, l.At(i, j))
			}
		}
	}
}

/** test that U is upper triangular */
func TestLUDecompositionUUpperTriangular(t *testing.T) {
	m := createRealMatrixFromSlices(t, testData)
	lud, _ := NewLUDecomposition(m)
	u := lud.U()
	for i := 0; i < u.RowDimension(); i++ {
		for j := 0; j < i; j++ {
			if math.Abs(u.At(i, j)-0) > entryTolerance {
				t.Errorf("Mismatch. want: %v, got: %v", 0, u.At(i, j))
			}
		}
	}
}

func TestLUDecompositionPPermutation(t *testing.T) {
	m := createRealMatrixFromSlices(t, testData)
	lud, _ := NewLUDecomposition(m)
	p := lud.P()

	ppT := p.Multiply(p.Transpose())
	id := createRealIdentityMatrix(t, p.RowDimension())
	if math.Abs(MatLInfNorm(ppT.Subtract(id))-0) > normTolerance {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(ppT.Subtract(id)))
	}

	for i := 0; i < p.RowDimension(); i++ {
		zeroCount := 0
		oneCount := 0
		otherCount := 0
		for j := 0; j < p.ColumnDimension(); j++ {
			e := p.At(i, j)
			if e == 0 {
				zeroCount++
			} else if e == 1 {
				oneCount++
			} else {
				otherCount++
			}
		}
		if zeroCount != p.ColumnDimension()-1 {
			t.Errorf("Mismatch. want: %v, got: %v", p.ColumnDimension()-1, zeroCount)
		}

		if oneCount != 1 {
			t.Errorf("Mismatch. want: %v, got: %v", 1, oneCount)
		}
		if otherCount != 0 {
			t.Errorf("Mismatch. want: %v, got: %v", 0, otherCount)
		}

	}

	for j := 0; j < p.ColumnDimension(); j++ {
		zeroCount := 0
		oneCount := 0
		otherCount := 0
		for i := 0; i < p.RowDimension(); i++ {
			e := p.At(i, j)
			if e == 0 {
				zeroCount++
			} else if e == 1 {
				oneCount++
			} else {
				otherCount++
			}
		}
		if zeroCount != p.RowDimension()-1 {
			t.Errorf("Mismatch. want: %v, got: %v", p.RowDimension()-1, zeroCount)
		}

		if oneCount != 1 {
			t.Errorf("Mismatch. want: %v, got: %v", 1, oneCount)
		}
		if otherCount != 0 {
			t.Errorf("Mismatch. want: %v, got: %v", 0, otherCount)
		}

	}

}

func TestLUDecompositionSingular(t *testing.T) {
	lu, _ := NewLUDecomposition(createRealMatrixFromSlices(t, testData))
	if !lu.Solver().IsNonSingular() {
		t.Errorf("Mismatch. should not be singular.")
	}

	lu, _ = NewLUDecomposition(createRealMatrixFromSlices(t, singular))
	if lu.Solver().IsNonSingular() {
		t.Errorf("Mismatch. should be singular.")
	}

	lu, _ = NewLUDecomposition(createRealMatrixFromSlices(t, bigSingular))
	if lu.Solver().IsNonSingular() {
		t.Errorf("Mismatch. should be singular.")
	}
}

func TestLUDecompositionMatricesValues1(t *testing.T) {
	lu, _ := NewLUDecomposition(createRealMatrixFromSlices(t, testData))
	lRef := createRealMatrixFromSlices(t, [][]float64{
		{1.0, 0.0, 0.0},
		{0.5, 1.0, 0.0},
		{0.5, 0.2, 1.0},
	})
	uRef := createRealMatrixFromSlices(t, [][]float64{
		{2.0, 5.0, 3.0},
		{0.0, -2.5, 6.5},
		{0.0, 0.0, 0.2},
	})
	pRef := createRealMatrixFromSlices(t, [][]float64{
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
		{1.0, 0.0, 0.0},
	})
	pivotRef := []int{1, 2, 0}

	// check values against known references
	l := lu.L()
	if math.Abs(MatLInfNorm(l.Subtract(lRef))-0) > 1.0e-13 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(l.Subtract(lRef)))
	}

	u := lu.U()
	if math.Abs(MatLInfNorm(u.Subtract(uRef))-0) > 1.0e-13 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(u.Subtract(uRef)))
	}

	p := lu.P()
	if math.Abs(MatLInfNorm(p.Subtract(pRef))-0) > 1.0e-13 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(p.Subtract(pRef)))
	}

	pivot := lu.Pivot()
	for i := 0; i < len(pivotRef); i++ {
		if pivotRef[i] != pivot[i] {
			t.Errorf("Mismatch. want: %v, got: %v", pivotRef[i], pivot[i])
		}

	}

	// check the same cached instance is returned the second time

	if !lu.L().Equals(l) {
		t.Errorf("Mismatch. Cache failed for L()")
	}
	if !lu.U().Equals(u) {
		t.Errorf("Mismatch. Cache failed for U()")
	}
	if !lu.P().Equals(p) {
		t.Errorf("Mismatch. Cache failed for P()")
	}

}

func TestLUDecompositionMatricesValues2(t *testing.T) {
	lu, _ := NewLUDecomposition(createRealMatrixFromSlices(t, luData))
	lRef := createRealMatrixFromSlices(t, [][]float64{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{1.0 / 3.0, 0.0, 1.0},
	})
	uRef := createRealMatrixFromSlices(t, [][]float64{
		{6.0, 9.0, 8.0},
		{0.0, 5.0, 7.0},
		{0.0, 0.0, 1.0 / 3.0},
	})
	pRef := createRealMatrixFromSlices(t, [][]float64{
		{0.0, 0.0, 1.0},
		{0.0, 1.0, 0.0},
		{1.0, 0.0, 0.0},
	})
	pivotRef := []int{2, 1, 0}

	// check values against known references
	l := lu.L()
	if math.Abs(MatLInfNorm(l.Subtract(lRef))-0) > 1.0e-13 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(l.Subtract(lRef)))
	}

	u := lu.U()
	if math.Abs(MatLInfNorm(u.Subtract(uRef))-0) > 1.0e-13 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(u.Subtract(uRef)))
	}

	p := lu.P()
	if math.Abs(MatLInfNorm(p.Subtract(pRef))-0) > 1.0e-13 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(p.Subtract(pRef)))
	}
	pivot := lu.Pivot()
	for i := 0; i < len(pivotRef); i++ {
		if pivotRef[i] != pivot[i] {
			t.Errorf("Mismatch. want: %v, got: %v", pivotRef[i], pivot[i])
		}

	}

	// check the same cached instance is returned the second time

	if !lu.L().Equals(l) {
		t.Errorf("Mismatch. Cache failed for L()")
	}
	if !lu.U().Equals(u) {
		t.Errorf("Mismatch. Cache failed for U()")
	}
	if !lu.P().Equals(p) {
		t.Errorf("Mismatch. Cache failed for P()")
	}
}
