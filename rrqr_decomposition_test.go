package linear

import (
	"math"
	"math/rand"
	"testing"
)

var (
	testData3x3NonSingular_rrqr = [][]float64{
		{12, -51, 4},
		{6, 167, -68},
		{-4, 24, -41},
	}

	testData3x3Singular_rrqr = [][]float64{
		{1, 4, 7},
		{2, 5, 8},
		{3, 6, 9},
	}

	testData3x4_rrqr = [][]float64{
		{12, -51, 4, 1},
		{6, 167, -68, 2},
		{-4, 24, -41, 3},
	}

	testData4x3_rrqr = [][]float64{
		{12, -51, 4},
		{6, 167, -68},
		{-4, 24, -41},
		{-5, 34, 7},
	}
)

func TestRRQRDecompositionDimensions(t *testing.T) {

	checkRRQRDimension(t, createRealMatrixFromSlices(t, testData3x3NonSingular_rrqr))

	checkRRQRDimension(t, createRealMatrixFromSlices(t, testData4x3_rrqr))

	checkRRQRDimension(t, createRealMatrixFromSlices(t, testData3x4_rrqr))

	r := rand.New(rand.NewSource(643895747384642))
	p := (5 * BLOCK_SIZE) / 4
	q := (7 * BLOCK_SIZE) / 4
	checkRRQRDimension(t, createQRTestMatrix(r, p, q))
	checkRRQRDimension(t, createQRTestMatrix(r, q, p))
}

func TestRRQRDecompositionAPEqualQR(t *testing.T) {
	checkAPEqualQR(t, createRealMatrixFromSlices(t, testData3x3NonSingular_rrqr))

	checkAPEqualQR(t, createRealMatrixFromSlices(t, testData3x3Singular_rrqr))

	checkAPEqualQR(t, createRealMatrixFromSlices(t, testData3x4_rrqr))

	checkAPEqualQR(t, createRealMatrixFromSlices(t, testData4x3_rrqr))

	r := rand.New(rand.NewSource(643895747384642))
	p := (5 * BLOCK_SIZE) / 4
	q := (7 * BLOCK_SIZE) / 4
	checkAPEqualQR(t, createQRTestMatrix(r, p, q))

	checkAPEqualQR(t, createQRTestMatrix(r, q, p))
}

func TestRRQRDecompositionQOrthogonal(t *testing.T) {
	checkRRQRQOrthogonal(t, createRealMatrixFromSlices(t, testData3x3NonSingular_rrqr))

	checkRRQRQOrthogonal(t, createRealMatrixFromSlices(t, testData3x3Singular_rrqr))

	checkRRQRQOrthogonal(t, createRealMatrixFromSlices(t, testData3x4_rrqr))

	checkRRQRQOrthogonal(t, createRealMatrixFromSlices(t, testData4x3_rrqr))

	r := rand.New(rand.NewSource(643895747384642))
	p := (5 * BLOCK_SIZE) / 4
	q := (7 * BLOCK_SIZE) / 4
	checkRRQRQOrthogonal(t, createQRTestMatrix(r, p, q))

	checkRRQRQOrthogonal(t, createQRTestMatrix(r, q, p))
}

func TestRRQRDecompositionRUpperTriangular(t *testing.T) {
	m := createRealMatrixFromSlices(t, testData3x3NonSingular_rrqr)
	checkUpperTriangular(t, createRRQRDecomposition(t, m).R())

	m = createRealMatrixFromSlices(t, testData3x3Singular_rrqr)
	checkUpperTriangular(t, createRRQRDecomposition(t, m).R())

	m = createRealMatrixFromSlices(t, testData3x4_rrqr)
	checkUpperTriangular(t, createRRQRDecomposition(t, m).R())

	m = createRealMatrixFromSlices(t, testData4x3_rrqr)
	checkUpperTriangular(t, createRRQRDecomposition(t, m).R())

	r := rand.New(rand.NewSource(643895747384642))
	p := (5 * BLOCK_SIZE) / 4
	q := (7 * BLOCK_SIZE) / 4
	m = createQRTestMatrix(r, p, q)
	checkUpperTriangular(t, createRRQRDecomposition(t, m).R())

	m = createQRTestMatrix(r, p, q)
	checkUpperTriangular(t, createRRQRDecomposition(t, m).R())
}

func TestRRQRDecompositionHTrapezoidal(t *testing.T) {
	m := createRealMatrixFromSlices(t, testData3x3NonSingular_rrqr)
	checkTrapezoidal(t, createRRQRDecomposition(t, m).H())

	m = createRealMatrixFromSlices(t, testData3x3Singular_rrqr)
	checkTrapezoidal(t, createRRQRDecomposition(t, m).H())

	m = createRealMatrixFromSlices(t, testData3x4_rrqr)
	checkTrapezoidal(t, createRRQRDecomposition(t, m).H())

	m = createRealMatrixFromSlices(t, testData4x3_rrqr)
	checkTrapezoidal(t, createRRQRDecomposition(t, m).H())

	r := rand.New(rand.NewSource(643895747384642))
	p := (5 * BLOCK_SIZE) / 4
	q := (7 * BLOCK_SIZE) / 4
	m = createQRTestMatrix(r, p, q)
	checkTrapezoidal(t, createRRQRDecomposition(t, m).H())

	m = createQRTestMatrix(r, p, q)
	checkTrapezoidal(t, createRRQRDecomposition(t, m).H())

}

func TestRRQRDecompositionNonInvertible(t *testing.T) {
	qr := createRRQRDecompositionWithThreshold(t, createRealMatrixFromSlices(t, testData3x3Singular_rrqr), 3.0e-16)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	qr.Solver().Inverse()
}

/** test the rank is returned correctly */
func TestRRQRDecompositionRank(t *testing.T) {
	d := [][]float64{{1, 1, 1}, {0, 0, 0}, {1, 2, 3}}
	m := createArray2DRowRealMatrixFromSlices(t, d)
	qr := createRRQRDecomposition(t, m)

	if 2 != qr.Rank(0) {
		t.Errorf("Mismatch. want %v, got: %v", 2, qr.Rank(0))
	}
}
func TestRRQRDecompositionRank2(t *testing.T) {
	d := [][]float64{{1, 1, 1}, {2, 3, 4}, {1, 2, 3}}
	m := createArray2DRowRealMatrixFromSlices(t, d)
	qr := createRRQRDecomposition(t, m)
	if 2 != qr.Rank(1e-14) {
		t.Errorf("Mismatch. want %v, got: %v", 2, qr.Rank(1e-14))
	}

}

func TestRRQRDecompositionRank3(t *testing.T) {
	d := [][]float64{
		{0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 1, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 1, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0},
	}
	m := createArray2DRowRealMatrixFromSlices(t, d)
	qr := createRRQRDecomposition(t, m.Transpose())
	if 4 != qr.Rank(1e-14) {
		t.Errorf("Mismatch. want %v, got: %v", 4, qr.Rank(1e-14))
	}

}

func checkRRQRQOrthogonal(t *testing.T, m RealMatrix) {
	qr := createRRQRDecomposition(t, m)
	eye := createRealIdentityMatrix(t, m.RowDimension())
	norm := MatLInfNorm(qr.QT().Multiply(qr.Q()).Subtract(eye))
	if math.Abs(0-norm) > normTolerance {
		t.Errorf("Mismatch. want %v, got: %v", 0, norm)
	}
}

func checkAPEqualQR(t *testing.T, m RealMatrix) {
	rrqr := createRRQRDecomposition(t, m)
	norm := MatLInfNorm(rrqr.Q().Multiply(rrqr.R()).Subtract(m.Multiply(rrqr.P())))
	if math.Abs(0-norm) > normTolerance {
		t.Errorf("Mismatch. want %v, got: %v", 0, norm)
	}

}
func checkRRQRDimension(t *testing.T, m RealMatrix) {
	rows := m.RowDimension()
	columns := m.ColumnDimension()
	qr := createRRQRDecomposition(t, m)
	if rows != qr.Q().RowDimension() {
		t.Errorf("Mismatch. want: %v, got: %v", rows, qr.Q().RowDimension())
	}
	if rows != qr.Q().ColumnDimension() {
		t.Errorf("Mismatch. want: %v, got: %v", rows, qr.Q().ColumnDimension())
	}
	if rows != qr.R().RowDimension() {
		t.Errorf("Mismatch. want: %v, got: %v", rows, qr.R().RowDimension())
	}
	if columns != qr.R().ColumnDimension() {
		t.Errorf("Mismatch. want: %v, got: %v", columns, qr.R().ColumnDimension())
	}
}
