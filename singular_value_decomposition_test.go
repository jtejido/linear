package linear

import (
	"math"
	"math/rand"
	"testing"
)

var (
	testSquare_svd = [][]float64{
		{24.0 / 25.0, 43.0 / 25.0},
		{57.0 / 25.0, 24.0 / 25.0},
	}

	testNonSquare_svd = [][]float64{
		{-540.0 / 625.0, 963.0 / 625.0, -216.0 / 625.0},
		{-1730.0 / 625.0, -744.0 / 625.0, 1008.0 / 625.0},
		{-720.0 / 625.0, 1284.0 / 625.0, -288.0 / 625.0},
		{-360.0 / 625.0, 192.0 / 625.0, 1756.0 / 625.0},
	}
)

func TestSingularValueDecompositionMoreRows(t *testing.T) {
	singularValues := []float64{123.456, 2.3, 1.001, 0.999}
	rows := len(singularValues) + 2
	columns := len(singularValues)
	r := rand.New(rand.NewSource(15338437322523))
	svd := createSingularValueDecomposition(t, createTestMatrixForSVD(t, r, rows, columns, singularValues))
	computedSV := svd.SingularValues()
	if len(singularValues) != len(computedSV) {
		t.Errorf("Mismatch. want: %v, got: %v", len(singularValues), len(computedSV))
	}

	for i := 0; i < len(singularValues); i++ {
		if math.Abs(singularValues[i]-computedSV[i]) > 1.0e-10 {
			t.Errorf("Mismatch. want: %v, got: %v", singularValues[i], computedSV[i])
		}

	}
}

func TestSingularValueDecompositionMoreColumns(t *testing.T) {
	singularValues := []float64{123.456, 2.3, 1.001, 0.999}
	rows := len(singularValues)
	columns := len(singularValues) + 2
	r := rand.New(rand.NewSource(732763225836210))
	svd := createSingularValueDecomposition(t, createTestMatrixForSVD(t, r, rows, columns, singularValues))
	computedSV := svd.SingularValues()
	if len(singularValues) != len(computedSV) {
		t.Errorf("Mismatch. want: %v, got: %v", len(singularValues), len(computedSV))
	}

	for i := 0; i < len(singularValues); i++ {
		if math.Abs(singularValues[i]-computedSV[i]) > 1.0e-10 {
			t.Errorf("Mismatch. want: %v, got: %v", singularValues[i], computedSV[i])
		}
	}
}

func TestSingularValueDecompositionDimensions(t *testing.T) {
	mm := createRealMatrixFromSlices(t, testSquare_svd)
	m := mm.RowDimension()
	n := mm.ColumnDimension()
	svd := createSingularValueDecomposition(t, mm)

	if svd.U().RowDimension() != m {
		t.Errorf("Mismatch. testData row dimension. want: %v, got: %v", m, svd.U().RowDimension())
	}

	if svd.U().ColumnDimension() != m {
		t.Errorf("Mismatch. testData column dimension. want: %v, got: %v", m, svd.U().ColumnDimension())
	}

	if svd.S().ColumnDimension() != m {
		t.Errorf("Mismatch. testData row dimension. want: %v, got: %v", m, svd.S().ColumnDimension())
	}

	if svd.S().ColumnDimension() != n {
		t.Errorf("Mismatch. testData column dimension. want: %v, got: %v", n, svd.S().ColumnDimension())
	}
	if svd.V().RowDimension() != n {
		t.Errorf("Mismatch. testData row dimension. want: %v, got: %v", n, svd.V().RowDimension())
	}

	if svd.V().ColumnDimension() != n {
		t.Errorf("Mismatch. testData column dimension. want: %v, got: %v", n, svd.V().ColumnDimension())
	}

}

func TestSingularValueDecompositionHadamard(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, [][]float64{
		{15.0 / 2.0, 5.0 / 2.0, 9.0 / 2.0, 3.0 / 2.0},
		{5.0 / 2.0, 15.0 / 2.0, 3.0 / 2.0, 9.0 / 2.0},
		{9.0 / 2.0, 3.0 / 2.0, 15.0 / 2.0, 5.0 / 2.0},
		{3.0 / 2.0, 9.0 / 2.0, 5.0 / 2.0, 15.0 / 2.0},
	})
	svd := createSingularValueDecomposition(t, m)
	if math.Abs(16.0-svd.SingularValues()[0]) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 16.0, svd.SingularValues()[0])
	}
	if math.Abs(8.0-svd.SingularValues()[1]) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 8.0, svd.SingularValues()[1])
	}
	if math.Abs(4.0-svd.SingularValues()[2]) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 4.0, svd.SingularValues()[2])
	}
	if math.Abs(2.0-svd.SingularValues()[3]) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 2.0, svd.SingularValues()[3])
	}

	fullCovariance := createArray2DRowRealMatrixFromSlices(t, [][]float64{
		{85.0 / 1024, -51.0 / 1024, -75.0 / 1024, 45.0 / 1024},
		{-51.0 / 1024, 85.0 / 1024, 45.0 / 1024, -75.0 / 1024},
		{-75.0 / 1024, 45.0 / 1024, 85.0 / 1024, -51.0 / 1024},
		{45.0 / 1024, -75.0 / 1024, -51.0 / 1024, 85.0 / 1024},
	})

	if math.Abs(0.0-MatLInfNorm(fullCovariance.Subtract(svd.Covariance(0.0)))) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, MatLInfNorm(fullCovariance.Subtract(svd.Covariance(0.0))))
	}

	halfCovariance := createArray2DRowRealMatrixFromSlices(t, [][]float64{
		{5.0 / 1024, -3.0 / 1024, 5.0 / 1024, -3.0 / 1024},
		{-3.0 / 1024, 5.0 / 1024, -3.0 / 1024, 5.0 / 1024},
		{5.0 / 1024, -3.0 / 1024, 5.0 / 1024, -3.0 / 1024},
		{-3.0 / 1024, 5.0 / 1024, -3.0 / 1024, 5.0 / 1024},
	})

	if math.Abs(0.0-MatLInfNorm(halfCovariance.Subtract(svd.Covariance(6.0)))) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, MatLInfNorm(halfCovariance.Subtract(svd.Covariance(6.0))))
	}

}

func TestSingularValueDecompositionAEqualUSVt(t *testing.T) {
	checkAEqualUSVt(t, createRealMatrixFromSlices(t, testSquare_svd))
	checkAEqualUSVt(t, createRealMatrixFromSlices(t, testNonSquare_svd))
	checkAEqualUSVt(t, createRealMatrixFromSlices(t, testNonSquare_svd).Transpose())
}

func TestSingularValueDecompositionUOrthogonal(t *testing.T) {
	checkOrthogonal(t, createSingularValueDecomposition(t, createRealMatrixFromSlices(t, testSquare_svd)).U())
	checkOrthogonal(t, createSingularValueDecomposition(t, createRealMatrixFromSlices(t, testNonSquare_svd)).U())
	checkOrthogonal(t, createSingularValueDecomposition(t, createRealMatrixFromSlices(t, testNonSquare_svd).Transpose()).U())
}

/** test that V is orthogonal */
func TestSingularValueDecompositionVOrthogonal(t *testing.T) {
	checkOrthogonal(t, createSingularValueDecomposition(t, createRealMatrixFromSlices(t, testSquare_svd)).V())
	checkOrthogonal(t, createSingularValueDecomposition(t, createRealMatrixFromSlices(t, testNonSquare_svd)).V())
	checkOrthogonal(t, createSingularValueDecomposition(t, createRealMatrixFromSlices(t, testNonSquare_svd).Transpose()).V())
}

func TestSingularValueDecompositionRank(t *testing.T) {
	d := [][]float64{{1, 1, 1}, {0, 0, 0}, {1, 2, 3}}
	m := createArray2DRowRealMatrixFromSlices(t, d)
	svd := createSingularValueDecomposition(t, m)
	if 2 != svd.Rank() {
		t.Errorf("Mismatch. want: %v, got: %v", 2, svd.Rank())
	}

}

func TestSingularValueDecompositionConditionNumber(t *testing.T) {
	svd := createSingularValueDecomposition(t, createRealMatrixFromSlices(t, testSquare_svd))
	if math.Abs(3.0-svd.ConditionNumber()) > 1.5e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 3.0, svd.ConditionNumber())
	}

}

func TestSingularValueDecompositionInverseConditionNumber(t *testing.T) {
	svd := createSingularValueDecomposition(t, createRealMatrixFromSlices(t, testSquare_svd))
	if math.Abs((1.0/3.0)-svd.InverseConditionNumber()) > 1.5e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 1.0/3.0, svd.InverseConditionNumber())
	}
}

func TestSingularValueDecompositionIssue947(t *testing.T) {
	nans := [][]float64{
		{math.NaN(), math.NaN()},
		{math.NaN(), math.NaN()},
	}
	m := createArray2DRowRealMatrixFromSlices(t, nans)
	svd := createSingularValueDecomposition(t, m)
	if !math.IsNaN(svd.SingularValues()[0]) {
		t.Errorf("Mismatch. want: %v, got: %v", math.NaN(), svd.SingularValues()[0])
	}
	if !math.IsNaN(svd.SingularValues()[1]) {
		t.Errorf("Mismatch. want: %v, got: %v", math.NaN(), svd.SingularValues()[1])
	}
}

func checkAEqualUSVt(t *testing.T, m RealMatrix) {
	svd := createSingularValueDecomposition(t, m)
	u := svd.U()
	s := svd.S()
	v := svd.V()
	norm := MatLInfNorm(u.Multiply(s).Multiply(v.Transpose()).Subtract(m))
	if math.Abs(0.0-norm) > normTolerance {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, norm)
	}

}

func createTestMatrixForSVD(t *testing.T, r *rand.Rand, rows, columns int, singularValues []float64) RealMatrix {
	u := createOrthogonalMatrix(r, rows)
	d := createArray2DRowRealMatrix(t, rows, columns)
	d.SetSubMatrix(createRealDiagonalMatrix(t, singularValues).Data(), 0, 0)
	v := createOrthogonalMatrix(r, columns)
	return u.Multiply(d).Multiply(v)
}
