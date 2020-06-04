package linear

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

var (
	testSquare5_ht = [][]float64{
		{5, 4, 3, 2, 1},
		{1, 4, 0, 3, 3},
		{2, 0, 3, 0, 0},
		{3, 2, 1, 2, 5},
		{4, 2, 1, 4, 1},
	}

	testSquare3_ht = [][]float64{
		{2, -1, 1},
		{-1, 2, 1},
		{1, -1, 2},
	}

	testRandom_ht = [][]float64{
		{0.680, 0.823, -0.4440, -0.2700},
		{-0.211, -0.605, 0.1080, 0.0268},
		{0.566, -0.330, -0.0452, 0.9040},
		{0.597, 0.536, 0.2580, 0.8320},
	}
)

func createHessenbergTransformer(t *testing.T, mat RealMatrix) *HessenbergTransformer {
	ht, err := NewHessenbergTransformer(mat)
	if err != nil {
		t.Errorf("Error while creating Transformer %s", err)
	}

	return ht
}

func TestHessenbergTransformerNonSquare(t *testing.T) {
	m := make([][]float64, 3)
	for i := 0; i < 3; i++ {
		m[i] = make([]float64, 2)
	}

	_, err := NewHessenbergTransformer(createRealMatrixFromSlices(t, m))
	if err == nil {
		t.Errorf("error expected.")
	}

}

func TestHessenbergTransformerRandomData(t *testing.T) {
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
		s := checkAEqualPHPt(t, m)
		checkTransformedMatrix(t, s)
	}
}

func TestHessenbergTransformerRandomDataNormalDistribution(t *testing.T) {
	for run := 0; run < 100; run++ {
		rs := rand.NewSource(time.Now().UnixNano())
		r := rand.New(rs)

		// matrix size
		size := r.Intn(20) + 4
		rnd := rand.New(rand.NewSource(64925784252))
		scale := math.Sqrt(r.Float64() * 5)
		data := make([][]float64, size)
		for i := 0; i < size; i++ {
			data[i] = make([]float64, size)
			for j := 0; j < size; j++ {
				data[i][j] = rnd.NormFloat64()*scale + 0.0
			}
		}

		m := createRealMatrixFromSlices(t, data)
		s := checkAEqualPHPt(t, m)
		checkTransformedMatrix(t, s)
	}
}

func TestHessenbergTransformerMatricesValues5(t *testing.T) {
	checkHessenbergMatricesValues(t, testSquare5_ht,
		[][]float64{
			{1.0, 0.0, 0.0, 0.0, 0.0},
			{0.0, -0.182574185835055, 0.784218758628863, 0.395029040913988, -0.442289115981669},
			{0.0, -0.365148371670111, -0.337950625265477, -0.374110794088820, -0.782621974707823},
			{0.0, -0.547722557505166, 0.402941130124223, -0.626468266309003, 0.381019628053472},
			{0.0, -0.730296743340221, -0.329285224617644, 0.558149336547665, 0.216118545309225},
		},
		[][]float64{
			{5.0, -3.65148371670111, 2.59962019434982, -0.237003414680848, -3.13886458663398},
			{-5.47722557505166, 6.9, -2.29164066120599, 0.207283564429169, 0.703858369151728},
			{0.0, -4.21386600008432, 2.30555659846067, 2.74935928725112, 0.857569835914113},
			{0.0, 0.0, 2.86406180891882, -1.11582249161595, 0.817995267184158},
			{0.0, 0.0, 0.0, 0.683518597386085, 1.91026589315528},
		})
}

func TestHessenbergTransformerMatricesValues3(t *testing.T) {
	checkHessenbergMatricesValues(t, testSquare3_ht,
		[][]float64{
			{1.0, 0.0, 0.0},
			{0.0, -0.707106781186547, 0.707106781186547},
			{0.0, 0.707106781186547, 0.707106781186548},
		},
		[][]float64{
			{2.0, 1.41421356237309, 0.0},
			{1.41421356237310, 2.0, -1.0},
			{0.0, 1.0, 2.0},
		})
}

func TestHessenbergTransformerAEqualPTPt(t *testing.T) {
	checkAEqualPHPt(t, createRealMatrixFromSlices(t, testSquare5_ht))
	checkAEqualPHPt(t, createRealMatrixFromSlices(t, testSquare3_ht))
	checkAEqualPHPt(t, createRealMatrixFromSlices(t, testRandom_ht))
}

func TestHessenbergTransformerPOrthogonal(t *testing.T) {
	checkOrthogonal(t, createHessenbergTransformer(t, createRealMatrixFromSlices(t, testSquare5_ht)).P())
	checkOrthogonal(t, createHessenbergTransformer(t, createRealMatrixFromSlices(t, testSquare3_ht)).P())
}

func TestHessenbergTransformerPTOrthogonal(t *testing.T) {
	checkOrthogonal(t, createHessenbergTransformer(t, createRealMatrixFromSlices(t, testSquare5_ht)).PT())
	checkOrthogonal(t, createHessenbergTransformer(t, createRealMatrixFromSlices(t, testSquare3_ht)).PT())
}

func TestHessenbergTransformerSchurForm(t *testing.T) {
	checkTransformedMatrix(t, createHessenbergTransformer(t, createRealMatrixFromSlices(t, testSquare5_ht)).H())
	checkTransformedMatrix(t, createHessenbergTransformer(t, createRealMatrixFromSlices(t, testSquare3_ht)).H())
}
