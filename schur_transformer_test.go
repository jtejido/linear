package linear

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

var (
	testSquare5_st = [][]float64{
		{5, 4, 3, 2, 1},
		{1, 4, 0, 3, 3},
		{2, 0, 3, 0, 0},
		{3, 2, 1, 2, 5},
		{4, 2, 1, 4, 1},
	}

	testSquare3_st = [][]float64{
		{2, -1, 1},
		{-1, 2, 1},
		{1, -1, 2},
	}

	testRandom_st = [][]float64{
		{0.680, -0.3300, -0.2700, -0.717, -0.687, 0.0259},
		{-0.211, 0.5360, 0.0268, 0.214, -0.198, 0.6780},
		{0.566, -0.4440, 0.9040, -0.967, -0.740, 0.2250},
		{0.597, 0.1080, 0.8320, -0.514, -0.782, -0.4080},
		{0.823, -0.0452, 0.2710, -0.726, 0.998, 0.2750},
		{-0.605, 0.2580, 0.4350, 0.608, -0.563, 0.0486},
	}
)

func createSchurTransformer(t *testing.T, mat RealMatrix) *SchurTransformer {
	st, err := NewSchurTransformer(mat)
	if err != nil {
		t.Errorf("Error while creating ST %s", err)
	}

	return st
}

func TestSchurTransformerNonSquare(t *testing.T) {
	m := make([][]float64, 3)
	for i := 0; i < 3; i++ {
		m[i] = make([]float64, 2)
	}

	_, err := NewSchurTransformer(createRealMatrixFromSlices(t, m))
	if err == nil {
		t.Errorf("error expected.")
	}

}

func TestSchurTransformerAEqualPTPt(t *testing.T) {
	checkAEqualPTPt(t, createRealMatrixFromSlices(t, testSquare5_st))
	checkAEqualPTPt(t, createRealMatrixFromSlices(t, testSquare3_st))
	checkAEqualPTPt(t, createRealMatrixFromSlices(t, testRandom_st))
}

func TestSchurTransformerPOrthogonal(t *testing.T) {
	checkOrthogonal(t, createSchurTransformer(t, createRealMatrixFromSlices(t, testSquare5_st)).P())
	checkOrthogonal(t, createSchurTransformer(t, createRealMatrixFromSlices(t, testSquare3_st)).P())
	checkOrthogonal(t, createSchurTransformer(t, createRealMatrixFromSlices(t, testRandom_st)).P())
}

func TestSchurTransformerPTOrthogonal(t *testing.T) {
	checkOrthogonal(t, createSchurTransformer(t, createRealMatrixFromSlices(t, testSquare5_st)).PT())
	checkOrthogonal(t, createSchurTransformer(t, createRealMatrixFromSlices(t, testSquare3_st)).PT())
	checkOrthogonal(t, createSchurTransformer(t, createRealMatrixFromSlices(t, testRandom_st)).PT())
}

func TestSchurTransformerSchurForm(t *testing.T) {
	checkTransformedMatrix(t, createSchurTransformer(t, createRealMatrixFromSlices(t, testSquare5_st)).T())
	checkTransformedMatrix(t, createSchurTransformer(t, createRealMatrixFromSlices(t, testSquare3_st)).T())
	checkTransformedMatrix(t, createSchurTransformer(t, createRealMatrixFromSlices(t, testRandom_st)).T())
}

func TestSchurTransformerRandomData(t *testing.T) {
	for run := 0; run < 100; run++ {
		rs := rand.NewSource(time.Now().UnixNano())
		r := rand.New(rs)

		// matrix size
		size := r.Intn(20) + 4

		data := make([][]float64, size)
		for i := 0; i < size; i++ {
			data[i] = make([]float64, size)
			for j := 0; j < size; j++ {
				data[i][j] = float64(r.Intn(100))
			}
		}

		m := createRealMatrixFromSlices(t, data)
		s := checkAEqualPTPt(t, m)
		checkTransformedMatrix(t, s)
	}
}

func TestSchurTransformerRandomDataNormalDistribution(t *testing.T) {
	for run := 0; run < 100; run++ {
		rs := rand.NewSource(time.Now().UnixNano())
		r := rand.New(rs)

		// matrix size
		size := r.Intn(20) + 4
		scale := math.Sqrt(r.Float64() * 5)
		rnd := rand.New(rand.NewSource(64925784252))
		data := make([][]float64, size)
		for i := 0; i < size; i++ {
			data[i] = make([]float64, size)
			for j := 0; j < size; j++ {
				data[i][j] = rnd.NormFloat64()*scale + 0.0
			}
		}

		m := createRealMatrixFromSlices(t, data)
		s := checkAEqualPTPt(t, m)
		checkTransformedMatrix(t, s)
	}
}
