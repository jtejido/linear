package linear

import (
	"math"
	"testing"
)

func TestDecomposition3x3(t *testing.T) {

	m := createRealMatrixFromSlices(t, [][]float64{
		{1, 9, 9},
		{9, 225, 225},
		{9, 225, 625},
	})

	d, _ := NewRectangularCholeskyDecompositionWithThreshold(m, 1.0e-6)

	// as this decomposition permutes lines and columns, the root is NOT triangular
	// (in fact here it is the lower right part of the matrix which is zero and
	//  the upper left non-zero)
	if math.Abs(0.8-d.RootMatrix().At(0, 2)) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.8, d.RootMatrix().At(0, 2))
	}

	if math.Abs(25.0-d.RootMatrix().At(2, 0)) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 25.0, d.RootMatrix().At(2, 0))
	}

	if math.Abs(0.0-d.RootMatrix().At(2, 2)) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, d.RootMatrix().At(2, 2))
	}

	root := d.RootMatrix()
	rebuiltM := root.Multiply(root.Transpose())

	if math.Abs(0.0-MatLInfNorm(m.Subtract(rebuiltM))) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, MatLInfNorm(m.Subtract(rebuiltM)))
	}

}

func TestFullRank(t *testing.T) {

	base := createRealMatrixFromSlices(t, [][]float64{
		{0.1159548705, 0., 0., 0.},
		{0.0896442724, 0.1223540781, 0., 0.},
		{0.0852155322, 4.558668e-3, 0.1083577299, 0.},
		{0.0905486674, 0.0213768077, 0.0128878333, 0.1014155693},
	})

	m := base.Multiply(base.Transpose())

	d, _ := NewRectangularCholeskyDecompositionWithThreshold(m, 1.0e-10)

	root := d.RootMatrix()
	rebuiltM := root.Multiply(root.Transpose())

	if math.Abs(0.0-MatLInfNorm(m.Subtract(rebuiltM))) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, MatLInfNorm(m.Subtract(rebuiltM)))
	}

	// the pivoted Cholesky decomposition is *not* unique. Here, the root is
	// not equal to the original trianbular base matrix
	if MatLInfNorm(root.Subtract(base)) <= 0.3 {
		t.Errorf("Mismatch. Norm should be greater than 0.3")
	}

}

func TestMath789(t *testing.T) {

	m1 := createRealMatrixFromSlices(t, [][]float64{
		{0.013445532, 0.010394690, 0.009881156, 0.010499559},
		{0.010394690, 0.023006616, 0.008196856, 0.010732709},
		{0.009881156, 0.008196856, 0.019023866, 0.009210099},
		{0.010499559, 0.010732709, 0.009210099, 0.019107243},
	})
	composeAndTest(t, m1, 4)

	m2 := createRealMatrixFromSlices(t, [][]float64{
		{0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 0.013445532, 0.010394690, 0.009881156, 0.010499559},
		{0.0, 0.010394690, 0.023006616, 0.008196856, 0.010732709},
		{0.0, 0.009881156, 0.008196856, 0.019023866, 0.009210099},
		{0.0, 0.010499559, 0.010732709, 0.009210099, 0.019107243},
	})
	composeAndTest(t, m2, 4)

	m3 := createRealMatrixFromSlices(t, [][]float64{
		{0.013445532, 0.010394690, 0.0, 0.009881156, 0.010499559},
		{0.010394690, 0.023006616, 0.0, 0.008196856, 0.010732709},
		{0.0, 0.0, 0.0, 0.0, 0.0},
		{0.009881156, 0.008196856, 0.0, 0.019023866, 0.009210099},
		{0.010499559, 0.010732709, 0.0, 0.009210099, 0.019107243},
	})
	composeAndTest(t, m3, 4)

}

func composeAndTest(t *testing.T, m RealMatrix, expectedRank int) {
	r, err := NewRectangularCholeskyDecomposition(m)
	if err != nil {
		t.Errorf("Error while creating RectangularCholeskyDecomposition %s", err)
	}
	if expectedRank != r.Rank() {
		t.Errorf("Mismatch. want %v, got: %v", expectedRank, r.Rank())
	}

	root := r.RootMatrix()
	rebuiltMatrix := root.Multiply(root.Transpose())
	if math.Abs(0.0-MatLInfNorm(m.Subtract(rebuiltMatrix))) > 1.0e-16 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, MatLInfNorm(m.Subtract(rebuiltMatrix)))
	}

}
