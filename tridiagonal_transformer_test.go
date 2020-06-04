package linear

import (
	"math"
	"testing"
)

var (
	testSquare5_tdt = [][]float64{
		{1, 2, 3, 1, 1},
		{2, 1, 1, 3, 1},
		{3, 1, 1, 1, 2},
		{1, 3, 1, 2, 1},
		{1, 1, 2, 1, 3},
	}

	testSquare3_tdt = [][]float64{
		{1, 3, 4},
		{3, 2, 2},
		{4, 2, 0},
	}
)

func createTriDiagonalTransformer(t *testing.T, mat RealMatrix) *TriDiagonalTransformer {
	tdt, err := NewTriDiagonalTransformer(mat)
	if err != nil {
		t.Errorf("Error while creating TDT %s", err)
	}

	return tdt
}

func TestTriDiagonalTransformerNonSquare(t *testing.T) {

	m := make([][]float64, 3)
	for i := 0; i < 3; i++ {
		m[i] = make([]float64, 2)
	}

	_, err := NewTriDiagonalTransformer(createRealMatrixFromSlices(t, m))
	if err == nil {
		t.Errorf("panic expected.")
	}

}

func TestTriDiagonalTransformerAEqualQTQt(t *testing.T) {
	checkAEqualQTQt(t, createRealMatrixFromSlices(t, testSquare5_tdt))
	checkAEqualQTQt(t, createRealMatrixFromSlices(t, testSquare3_tdt))
}

func TestTriDiagonalTransformerNoAccessBelowDiagonal(t *testing.T) {
	checkNoAccessBelowDiagonal(t, testSquare5_tdt)
	checkNoAccessBelowDiagonal(t, testSquare3_tdt)
}

func TestTriDiagonalTransformerQOrthogonal(t *testing.T) {
	checkOrthogonal(t, createTriDiagonalTransformer(t, createRealMatrixFromSlices(t, testSquare5_tdt)).Q())
	checkOrthogonal(t, createTriDiagonalTransformer(t, createRealMatrixFromSlices(t, testSquare3_tdt)).Q())
}

func TestTriDiagonalTransformerQTOrthogonal(t *testing.T) {
	checkOrthogonal(t, createTriDiagonalTransformer(t, createRealMatrixFromSlices(t, testSquare5_tdt)).QT())
	checkOrthogonal(t, createTriDiagonalTransformer(t, createRealMatrixFromSlices(t, testSquare3_tdt)).QT())
}

func TestTriDiagonalTransformerTTriDiagonal(t *testing.T) {
	checkTransformedMatrix(t, createTriDiagonalTransformer(t, createRealMatrixFromSlices(t, testSquare5_tdt)).T())
	checkTransformedMatrix(t, createTriDiagonalTransformer(t, createRealMatrixFromSlices(t, testSquare3_tdt)).T())
}

func TestTriDiagonalTransformerMatricesValues5(t *testing.T) {
	checkTriDiagonalMatricesValues(t, testSquare5_tdt,
		[][]float64{
			{1.0, 0.0, 0.0, 0.0, 0.0},
			{0.0, -0.5163977794943222, 0.016748280772542083, 0.839800693771262, 0.16669620021405473},
			{0.0, -0.7745966692414833, -0.4354553000860955, -0.44989322880603355, -0.08930153582895772},
			{0.0, -0.2581988897471611, 0.6364346693566014, -0.30263204032131164, 0.6608313651342882},
			{0.0, -0.2581988897471611, 0.6364346693566009, -0.027289660803112598, -0.7263191580755246},
		},
		[]float64{1, 4.4, 1.433099579242636, -0.89537362758743, 2.062274048344794},
		[]float64{-math.Sqrt(15), -3.0832882879592476, 0.6082710842351517, 1.1786086405912128})
}

func TestTriDiagonalTransformerMatricesValues3(t *testing.T) {
	checkTriDiagonalMatricesValues(t, testSquare3_tdt,
		[][]float64{
			{1.0, 0.0, 0.0},
			{0.0, -0.6, 0.8},
			{0.0, -0.8, -0.6},
		},
		[]float64{1, 2.64, -0.64},
		[]float64{-5, -1.52})
}
