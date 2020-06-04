package linear

import (
	"math"
	"math/rand"
	"testing"
)

var (
	testData3x3NonSingular_qrs = [][]float64{
		{12, -51, 4},
		{6, 167, -68},
		{-4, 24, -41},
	}

	testData3x3Singular_qrs = [][]float64{
		{1, 2, 2},
		{2, 4, 6},
		{4, 8, 12},
	}

	testData3x4_qrs = [][]float64{
		{12, -51, 4, 1},
		{6, 167, -68, 2},
		{-4, 24, -41, 3},
	}

	testData4x3_qrs = [][]float64{
		{12, -51, 4},
		{6, 167, -68},
		{-4, 24, -41},
		{-5, 34, 7},
	}
)

func TestQRDecompositionSolverRank(t *testing.T) {
	qrd, _ := NewQRDecomposition(createRealMatrixFromSlices(t, testData3x3NonSingular_qrs))
	solver := qrd.Solver()
	if !solver.IsNonSingular() {
		t.Errorf("Mismatch. Should be singular.")
	}
	qrd, _ = NewQRDecomposition(createRealMatrixFromSlices(t, testData3x3Singular_qrs))
	solver = qrd.Solver()
	if solver.IsNonSingular() {
		t.Errorf("Mismatch. Should not be singular.")
	}
	qrd, _ = NewQRDecomposition(createRealMatrixFromSlices(t, testData3x4_qrs))
	solver = qrd.Solver()
	if !solver.IsNonSingular() {
		t.Errorf("Mismatch. Should be singular.")
	}
	qrd, _ = NewQRDecomposition(createRealMatrixFromSlices(t, testData4x3_qrs))
	solver = qrd.Solver()
	if !solver.IsNonSingular() {
		t.Errorf("Mismatch. Should be singular.")
	}

}

func TestQRDecompositionSolverDimensionErrors(t *testing.T) {
	qrd, _ := NewQRDecomposition(createRealMatrixFromSlices(t, testData3x3NonSingular_qrs))
	solver := qrd.Solver()
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
}

func TestQRDecompositionSolverSolveRankErrors(t *testing.T) {
	qrd, _ := NewQRDecomposition(createRealMatrixFromSlices(t, testData3x3Singular_qrs))
	solver := qrd.Solver()
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
}

func TestQRDecompositionSolverSolve(t *testing.T) {
	decomposition, _ := NewQRDecomposition(createRealMatrixFromSlices(t, testData3x3NonSingular_qrs))
	solver := decomposition.Solver()
	b := createRealMatrixFromSlices(t, [][]float64{
		{-102, 12250}, {544, 24500}, {167, -36750},
	})
	xRef := createRealMatrixFromSlices(t, [][]float64{
		{1, 2515}, {2, 422}, {-3, 898},
	})

	// using RealMatrix
	if math.Abs(0-MatLInfNorm(solver.SolveMatrix(b).Subtract(xRef))) > 2.0e-16*MatLInfNorm(xRef) {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(solver.SolveMatrix(b).Subtract(xRef)))
	}

	// using ArrayRealVector

	for i := 0; i < b.ColumnDimension(); i++ {
		diffVecNorm := VecNorm(solver.SolveVector(b.ColumnVectorAt(i)).Subtract(xRef.ColumnVectorAt(i)))
		if math.Abs(0.0-diffVecNorm) > 3.0e-16*VecNorm(xRef.ColumnVectorAt(i)) {
			t.Errorf("Mismatch. want: %v, got: %v", 0.0, diffVecNorm)
		}

	}

}

type testQRDecompositionSolverChangingVisitor struct {
	visit func(row, column int, value float64) float64
}

func (qrmcv *testQRDecompositionSolverChangingVisitor) Start(rows, columns, startRow, endRow, startColumn, endColumn int) {
}
func (qrmcv *testQRDecompositionSolverChangingVisitor) Visit(row, column int, value float64) float64 {
	return qrmcv.visit(row, column, value)
}
func (qrmcv *testQRDecompositionSolverChangingVisitor) End() float64 { return 0.0 }

func TestQRDecompositionSolverOverdetermined(t *testing.T) {
	r := rand.New(rand.NewSource(5559252868205245))
	p := (7 * BLOCK_SIZE) / 4
	q := (5 * BLOCK_SIZE) / 4
	a := createQRTestMatrix(r, p, q)
	xRef := createQRTestMatrix(r, q, BLOCK_SIZE+3)

	// build a perturbed system: A.X + noise = B
	b := a.Multiply(xRef)
	noise := 0.001
	visitor := new(testQRDecompositionSolverChangingVisitor)
	visitor.visit = func(row, column int, value float64) float64 {
		return value * (1.0 + noise*(2*r.Float64()-1))
	}

	b.WalkInUpdateOptimizedOrder(visitor)

	// despite perturbation, the least square solution should be pretty good
	qrd, _ := NewQRDecomposition(a)
	x := qrd.Solver().SolveMatrix(b)
	if math.Abs(0.0-MatLInfNorm(x.Subtract(xRef))) > 0.01*noise*float64(p)*float64(q) {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, MatLInfNorm(x.Subtract(xRef)))
	}
}

func TestQRDecompositionSolverUnderdetermined(t *testing.T) {
	r := rand.New(rand.NewSource(42185006424567123))
	p := (5 * BLOCK_SIZE) / 4
	q := (7 * BLOCK_SIZE) / 4
	a := createQRTestMatrix(r, p, q)
	xRef := createQRTestMatrix(r, q, BLOCK_SIZE+3)
	b := a.Multiply(xRef)
	qrd, _ := NewQRDecomposition(a)
	x := qrd.Solver().SolveMatrix(b)

	// too many equations, the system cannot be solved at all
	if MatLInfNorm(x.Subtract(xRef))/(float64(p)*float64(q)) <= 0.01 {
		t.Errorf("Mismatch. %v should be > %v", MatLInfNorm(x.Subtract(xRef))/(float64(p)*float64(q)), 0.01)
	}

	// the last unknown should have been set to 0
	if math.Abs(0.0-MatLInfNorm(x.SubMatrix(p, q-1, 0, x.ColumnDimension()-1))) > 0 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, MatLInfNorm(x.SubMatrix(p, q-1, 0, x.ColumnDimension()-1)))
	}

}
