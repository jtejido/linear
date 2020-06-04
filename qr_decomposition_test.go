package linear

import (
	"math"
	"math/rand"
	"testing"
)

var (
	testData3x3NonSingular_qr = [][]float64{
		{12, -51, 4},
		{6, 167, -68},
		{-4, 24, -41},
	}

	testData3x3Singular_qr = [][]float64{
		{1, 4, 7},
		{2, 5, 8},
		{3, 6, 9},
	}

	testData3x4_qr = [][]float64{
		{12, -51, 4, 1},
		{6, 167, -68, 2},
		{-4, 24, -41, 3},
	}

	testData4x3_qr = [][]float64{
		{12, -51, 4},
		{6, 167, -68},
		{-4, 24, -41},
		{-5, 34, 7},
	}
)

func createQRDecomposition(t *testing.T, mat RealMatrix) *QRDecomposition {
	qrd, err := NewQRDecomposition(mat)
	if err != nil {
		t.Errorf("Error while creating QRD %s", err)
	}

	return qrd
}

func TestQRDecompositionDimensions(t *testing.T) {

	checkDimension(t, createRealMatrixFromSlices(t, testData3x3NonSingular_qr))

	checkDimension(t, createRealMatrixFromSlices(t, testData4x3_qr))

	checkDimension(t, createRealMatrixFromSlices(t, testData3x4_qr))

	r := rand.New(rand.NewSource(643895747384642))
	p := (5 * BLOCK_SIZE) / 4
	q := (7 * BLOCK_SIZE) / 4
	checkDimension(t, createQRTestMatrix(r, p, q))
	checkDimension(t, createQRTestMatrix(r, q, p))
}

func TestQRDecompositionAEqualQR(t *testing.T) {
	checkAEqualQR(t, createRealMatrixFromSlices(t, testData3x3NonSingular_qr))

	checkAEqualQR(t, createRealMatrixFromSlices(t, testData3x3Singular_qr))

	checkAEqualQR(t, createRealMatrixFromSlices(t, testData3x4_qr))

	checkAEqualQR(t, createRealMatrixFromSlices(t, testData4x3_qr))

	r := rand.New(rand.NewSource(643895747384642))
	p := (5 * BLOCK_SIZE) / 4
	q := (7 * BLOCK_SIZE) / 4
	checkAEqualQR(t, createQRTestMatrix(r, p, q))

	checkAEqualQR(t, createQRTestMatrix(r, q, p))
}

func TestQRDecompositionQOrthogonal(t *testing.T) {
	checkQOrthogonal(t, createRealMatrixFromSlices(t, testData3x3NonSingular_qr))

	checkQOrthogonal(t, createRealMatrixFromSlices(t, testData3x3Singular_qr))

	checkQOrthogonal(t, createRealMatrixFromSlices(t, testData3x4_qr))

	checkQOrthogonal(t, createRealMatrixFromSlices(t, testData4x3_qr))

	r := rand.New(rand.NewSource(643895747384642))
	p := (5 * BLOCK_SIZE) / 4
	q := (7 * BLOCK_SIZE) / 4
	checkQOrthogonal(t, createQRTestMatrix(r, p, q))

	checkQOrthogonal(t, createQRTestMatrix(r, q, p))
}

func TestQRDecompositionRUpperTriangular(t *testing.T) {
	m := createRealMatrixFromSlices(t, testData3x3NonSingular_qr)
	checkUpperTriangular(t, createQRDecomposition(t, m).R())

	m = createRealMatrixFromSlices(t, testData3x3Singular_qr)
	checkUpperTriangular(t, createQRDecomposition(t, m).R())

	m = createRealMatrixFromSlices(t, testData3x4_qr)
	checkUpperTriangular(t, createQRDecomposition(t, m).R())

	m = createRealMatrixFromSlices(t, testData4x3_qr)
	checkUpperTriangular(t, createQRDecomposition(t, m).R())

	r := rand.New(rand.NewSource(643895747384642))
	p := (5 * BLOCK_SIZE) / 4
	q := (7 * BLOCK_SIZE) / 4
	m = createQRTestMatrix(r, p, q)
	checkUpperTriangular(t, createQRDecomposition(t, m).R())

	m = createQRTestMatrix(r, p, q)
	checkUpperTriangular(t, createQRDecomposition(t, m).R())
}

func TestQRDecompositionHTrapezoidal(t *testing.T) {
	m := createRealMatrixFromSlices(t, testData3x3NonSingular_qr)
	checkTrapezoidal(t, createQRDecomposition(t, m).H())

	m = createRealMatrixFromSlices(t, testData3x3Singular_qr)
	checkTrapezoidal(t, createQRDecomposition(t, m).H())

	m = createRealMatrixFromSlices(t, testData3x4_qr)
	checkTrapezoidal(t, createQRDecomposition(t, m).H())

	m = createRealMatrixFromSlices(t, testData4x3_qr)
	checkTrapezoidal(t, createQRDecomposition(t, m).H())

	r := rand.New(rand.NewSource(643895747384642))
	p := (5 * BLOCK_SIZE) / 4
	q := (7 * BLOCK_SIZE) / 4
	m = createQRTestMatrix(r, p, q)
	checkTrapezoidal(t, createQRDecomposition(t, m).H())

	m = createQRTestMatrix(r, p, q)
	checkTrapezoidal(t, createQRDecomposition(t, m).H())
}

func TestQRDecompositionMatricesValues(t *testing.T) {
	qr := createQRDecomposition(t, createRealMatrixFromSlices(t, testData3x3NonSingular_qr))
	qRef := createRealMatrixFromSlices(t, [][]float64{
		{-12.0 / 14.0, 69.0 / 175.0, -58.0 / 175.0},
		{-6.0 / 14.0, -158.0 / 175.0, 6.0 / 175.0},
		{4.0 / 14.0, -30.0 / 175.0, -165.0 / 175.0},
	})
	rRef := createRealMatrixFromSlices(t, [][]float64{
		{-14.0, -21.0, 14.0},
		{0.0, -175.0, 70.0},
		{0.0, 0.0, 35.0},
	})
	hRef := createRealMatrixFromSlices(t, [][]float64{
		{26.0 / 14.0, 0.0, 0.0},
		{6.0 / 14.0, 648.0 / 325.0, 0.0},
		{-4.0 / 14.0, 36.0 / 325.0, 2.0},
	})

	// check values against known references
	q := qr.Q()
	if math.Abs(0.0-MatLInfNorm(q.Subtract(qRef))) > 1.0e-13 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, MatLInfNorm(q.Subtract(qRef)))
	}

	qT := qr.QT()
	if math.Abs(0.0-MatLInfNorm(qT.Subtract(qRef.Transpose()))) > 1.0e-13 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, MatLInfNorm(qT.Subtract(qRef.Transpose())))
	}

	r := qr.R()
	if math.Abs(0.0-MatLInfNorm(r.Subtract(rRef))) > 1.0e-13 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, MatLInfNorm(r.Subtract(rRef)))
	}

	h := qr.H()
	if math.Abs(0.0-MatLInfNorm(h.Subtract(hRef))) > 1.0e-13 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, MatLInfNorm(h.Subtract(hRef)))
	}

	// check the same cached instance is returned the second time (same pointer)
	if q != qr.Q() {
		t.Errorf("Mismatch. Q() wasn't cached.")
	}
	if r != qr.R() {
		t.Errorf("Mismatch. R() wasn't cached.")
	}
	if h != qr.H() {
		t.Errorf("Mismatch. H() wasn't cached.")
	}
}

func TestQRDecompositionNonInvertible(t *testing.T) {
	qr := createQRDecomposition(t, createRealMatrixFromSlices(t, testData3x3Singular_qr))
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	qr.Solver().Inverse()
}

func TestQRDecompositionInvertTallSkinny(t *testing.T) {
	a := createRealMatrixFromSlices(t, testData4x3_qr)
	pinv := createQRDecomposition(t, a).Solver().Inverse()
	norm := MatLInfNorm(pinv.Multiply(a).Subtract(createRealIdentityMatrix(t, 3)))
	if math.Abs(0.0-norm) > 1.0e-6 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, norm)
	}
}

func TestQRDecompositionInvertShortWide(t *testing.T) {
	a := createRealMatrixFromSlices(t, testData3x4_qr)
	pinv := createQRDecomposition(t, a).Solver().Inverse()
	norm := MatLInfNorm(a.Multiply(pinv).Subtract(createRealIdentityMatrix(t, 3)))
	if math.Abs(0.0-norm) > 1.0e-6 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, norm)
	}
	norm = MatLInfNorm(pinv.Multiply(a).SubMatrix(0, 2, 0, 2).Subtract(createRealIdentityMatrix(t, 3)))
	if math.Abs(0.0-norm) > 1.0e-6 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, norm)
	}
}

func TestQRDecompositionQRSingular(t *testing.T) {
	a := createRealMatrixFromSlices(t, [][]float64{
		{1, 6, 4}, {2, 4, -1}, {-1, 2, 5},
	})
	b := createArrayRealVectorFromSlice(t, []float64{5, 6, 1})
	qrd, _ := NewQRDecompositionWithThreshold(a, 1.0e-15)

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	qrd.Solver().SolveVector(b)

}

type testQRMatrixPreservingVisitor struct {
	visit func(row, column int, value float64)
}

func (qrmcv *testQRMatrixPreservingVisitor) Start(rows, columns, startRow, endRow, startColumn, endColumn int) {
}
func (qrmcv *testQRMatrixPreservingVisitor) Visit(row, column int, value float64) {
	qrmcv.visit(row, column, value)
}
func (qrmcv *testQRMatrixPreservingVisitor) End() float64 { return 0.0 }

func checkTrapezoidal(t *testing.T, m RealMatrix) {
	visitor := new(testQRMatrixPreservingVisitor)
	visitor.visit = func(row, column int, value float64) {
		if column > row {
			if math.Abs(0.0-value) > entryTolerance {
				t.Errorf("Mismatch. want: %v, got: %v", 0.0, value)
			}
		}
	}
	m.WalkInOptimizedOrder(visitor)
}

func checkUpperTriangular(t *testing.T, m RealMatrix) {
	visitor := new(testQRMatrixPreservingVisitor)
	visitor.visit = func(row, column int, value float64) {
		if column < row {
			if math.Abs(0.0-value) > entryTolerance {
				t.Errorf("Mismatch. want: %v, got: %v", 0.0, value)
			}
		}
	}

	m.WalkInOptimizedOrder(visitor)
}

func checkQOrthogonal(t *testing.T, m RealMatrix) {
	qr := createQRDecomposition(t, m)
	eye := createRealIdentityMatrix(t, m.RowDimension())
	norm := MatLInfNorm(qr.QT().Multiply(qr.Q()).Subtract(eye))
	if math.Abs(0.0-norm) > normTolerance {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, norm)
	}

}

func checkAEqualQR(t *testing.T, m RealMatrix) {
	qr := createQRDecomposition(t, m)
	norm := MatLInfNorm(qr.Q().Multiply(qr.R()).Subtract(m))
	if math.Abs(0.0-norm) > normTolerance {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, norm)
	}

}

type testQRMatrixChangingVisitor struct {
	visit func(row, column int, value float64) float64
}

func (qrmcv *testQRMatrixChangingVisitor) Start(rows, columns, startRow, endRow, startColumn, endColumn int) {
}
func (qrmcv *testQRMatrixChangingVisitor) Visit(row, column int, value float64) float64 {
	return qrmcv.visit(row, column, value)
}
func (qrmcv *testQRMatrixChangingVisitor) End() float64 { return 0.0 }

func createQRTestMatrix(r *rand.Rand, rows, columns int) RealMatrix {
	m, _ := NewRealMatrixWithDimension(rows, columns)
	visitor := new(testQRMatrixChangingVisitor)
	visitor.visit = func(row, column int, value float64) float64 {
		return 2.0*r.Float64() - 1.0
	}
	m.WalkInUpdateOptimizedOrder(visitor)
	return m
}

func checkDimension(t *testing.T, m RealMatrix) {
	rows := m.RowDimension()
	columns := m.ColumnDimension()
	qr := createQRDecomposition(t, m)
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
