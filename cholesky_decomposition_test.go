package linear

import (
	"math"
	"testing"
)

var (
	testData_cd = [][]float64{
		{1, 2, 4, 7, 11},
		{2, 13, 23, 38, 58},
		{4, 23, 77, 122, 182},
		{7, 38, 122, 294, 430},
		{11, 58, 182, 430, 855},
	}
)

func TestCholeskyDecompositionDimensions(t *testing.T) {
	m := createRealMatrixFromSlices(t, testData_cd)
	CD, _ := NewDefaultCholeskyDecomposition(m)

	if CD.L().RowDimension() != len(testData_cd) {
		t.Errorf("Mismatch. testData row dimension. want: %v, got: %v", len(testData_cd), CD.L().RowDimension())
	}

	if CD.L().ColumnDimension() != len(testData_cd) {
		t.Errorf("Mismatch. testData column dimension. want: %v, got: %v", len(testData_cd), CD.L().ColumnDimension())
	}

	if CD.LT().RowDimension() != len(testData_cd) {
		t.Errorf("Mismatch. testData row dimension. want: %v, got: %v", len(testData_cd), CD.LT().RowDimension())
	}

	if CD.LT().ColumnDimension() != len(testData_cd) {
		t.Errorf("Mismatch. testData column dimension. want: %v, got: %v", len(testData_cd), CD.LT().ColumnDimension())
	}

}

func TestCholeskyDecompositionNonSquare(t *testing.T) {
	m := make([][]float64, 3)
	for i := 0; i < 3; i++ {
		m[i] = make([]float64, 2)
	}
	_, err := NewDefaultCholeskyDecomposition(createRealMatrixFromSlices(t, m))
	if err == nil {
		t.Errorf("error expected.")
	}
}

func TestCholeskyDecompositionNotSymmetricMatrixException(t *testing.T) {
	changed := make([][]float64, len(testData_cd))
	for i := 0; i < len(testData_cd[0]); i++ {
		changed[i] = append([]float64{}, testData_cd[i]...)
	}

	changed[0][len(changed[0])-1] += 1.0e-5
	_, err := NewDefaultCholeskyDecomposition(createRealMatrixFromSlices(t, changed))
	if err == nil {
		t.Errorf("error expected.")
	}
}

func TestCholeskyDecompositionNotPositiveDefinite(t *testing.T) {
	m := [][]float64{
		{14, 11, 13, 15, 24},
		{11, 34, 13, 8, 25},
		{13, 13, 14, 15, 21},
		{15, 8, 15, 18, 23},
		{24, 25, 21, 23, 45},
	}

	_, err := NewDefaultCholeskyDecomposition(createRealMatrixFromSlices(t, m))
	if err == nil {
		t.Errorf("error expected.")
	}
}

func TestCholeskyDecompositionAEqualLLT(t *testing.T) {
	m := createRealMatrixFromSlices(t, testData_cd)
	CD, _ := NewDefaultCholeskyDecomposition(m)
	CDl := CD.L()
	CDlt := CD.LT()
	norm := MatLInfNorm(CDl.Multiply(CDlt).Subtract(m))

	if math.Abs(norm-0) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, norm)
	}
}

func TestCholeskyDecompositionLLowerTriangular(t *testing.T) {
	m := createRealMatrixFromSlices(t, testData_cd)
	cd, _ := NewDefaultCholeskyDecomposition(m)
	l := cd.L()
	for i := 0; i < l.RowDimension(); i++ {
		for j := i + 1; j < l.ColumnDimension(); j++ {
			if 0 != l.At(i, j) {
				t.Errorf("Mismatch. want: %v, got: %v", 0, l.At(i, j))
			}
		}
	}
}

func TestCholeskyDecompositionLTTransposed(t *testing.T) {
	m := createRealMatrixFromSlices(t, testData_cd)
	llt, _ := NewDefaultCholeskyDecomposition(m)
	l := llt.L()
	lt := llt.LT()
	norm := MatLInfNorm(l.Subtract(lt.Transpose()))
	if math.Abs(norm-0) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, norm)
	}

}

func TestCholeskyDecompositionMatricesValues(t *testing.T) {
	lRef := createRealMatrixFromSlices(t, [][]float64{
		{1, 0, 0, 0, 0},
		{2, 3, 0, 0, 0},
		{4, 5, 6, 0, 0},
		{7, 8, 9, 10, 0},
		{11, 12, 13, 14, 15},
	})
	llt, _ := NewDefaultCholeskyDecomposition(createRealMatrixFromSlices(t, testData_cd))

	// check values against known references
	l := llt.L()
	if math.Abs(MatLInfNorm(l.Subtract(lRef))-0) > 1.0e-13 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(l.Subtract(lRef)))
	}

	lt := llt.LT()
	if math.Abs(MatLInfNorm(lt.Subtract(lRef.Transpose()))-0) > 1.0e-13 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(lt.Subtract(lRef.Transpose())))
	}

	// check the same cached instance is returned the second time
	if !l.Equals(llt.L()) {
		t.Errorf("Mismatch. Lower triangular matrix doesn't match")
	}
	if !lt.Equals(llt.LT()) {
		t.Errorf("Mismatch. Lower triangular transposition doesn't match")
	}

}
