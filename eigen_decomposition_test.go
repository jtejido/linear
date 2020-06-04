package linear

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

var (
	refValues = []float64{2.003, 2.002, 2.001, 1.001, 1.000, 0.001}
	mat_ed    = createTestMatrix(rand.New(rand.NewSource(35992629946426)), refValues)
)

func TestEigenDecompositionDimension1(t *testing.T) {
	m := createRealMatrixFromSlices(t, [][]float64{{1.5}})
	ed, _ := NewEigenDecomposition(m)

	if math.Abs(ed.RealEigenvalueAt(0)-1.5) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 1.5, ed.RealEigenvalueAt(0))
	}

}

func TestEigenDecompositionDimension2(t *testing.T) {
	m := createRealMatrixFromSlices(t, [][]float64{{59.0, 12.0}, {12.0, 66.0}})
	ed, _ := NewEigenDecomposition(m)

	if math.Abs(ed.RealEigenvalueAt(0)-75.0) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 75.0, ed.RealEigenvalueAt(0))
	}
	if math.Abs(ed.RealEigenvalueAt(1)-50.0) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 50.0, ed.RealEigenvalueAt(1))
	}
}

func TestEigenDecompositionDimension3(t *testing.T) {
	m := createRealMatrixFromSlices(t, [][]float64{
		{39632.0, -4824.0, -16560.0},
		{-4824.0, 8693.0, 7920.0},
		{-16560.0, 7920.0, 17300.0},
	})
	ed, _ := NewEigenDecomposition(m)

	if math.Abs(ed.RealEigenvalueAt(0)-50000.0) > 3.0e-11 {
		t.Errorf("Mismatch. want: %v, got: %v", 50000.0, ed.RealEigenvalueAt(0))
	}
	if math.Abs(ed.RealEigenvalueAt(1)-12500.0) > 3.0e-11 {
		t.Errorf("Mismatch. want: %v, got: %v", 12500.0, ed.RealEigenvalueAt(1))
	}
	if math.Abs(ed.RealEigenvalueAt(2)-3125.0) > 3.0e-11 {
		t.Errorf("Mismatch. want: %v, got: %v", 3125.0, ed.RealEigenvalueAt(2))
	}

}

func TestEigenDecompositionDimension3MultipleRoot(t *testing.T) {
	m := createRealMatrixFromSlices(t, [][]float64{
		{5, 10, 15},
		{10, 20, 30},
		{15, 30, 45},
	})
	ed, _ := NewEigenDecomposition(m)

	if math.Abs(ed.RealEigenvalueAt(0)-70.0) > 3.0e-11 {
		t.Errorf("Mismatch. want: %v, got: %v", 70.0, ed.RealEigenvalueAt(0))
	}
	if math.Abs(ed.RealEigenvalueAt(1)-0.0) > 3.0e-11 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, ed.RealEigenvalueAt(1))
	}
	if math.Abs(ed.RealEigenvalueAt(2)-0.0) > 3.0e-11 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, ed.RealEigenvalueAt(2))
	}

}

func TestEigenDecompositionDimension4WithSplit(t *testing.T) {
	m := createRealMatrixFromSlices(t, [][]float64{
		{0.784, -0.288, 0.000, 0.000},
		{-0.288, 0.616, 0.000, 0.000},
		{0.000, 0.000, 0.164, -0.048},
		{0.000, 0.000, -0.048, 0.136},
	})
	ed, _ := NewEigenDecomposition(m)

	if math.Abs(ed.RealEigenvalueAt(0)-1.0) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 1.0, ed.RealEigenvalueAt(0))
	}
	if math.Abs(ed.RealEigenvalueAt(1)-0.4) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.4, ed.RealEigenvalueAt(1))
	}
	if math.Abs(ed.RealEigenvalueAt(2)-0.2) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.2, ed.RealEigenvalueAt(2))
	}
	if math.Abs(ed.RealEigenvalueAt(3)-0.1) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.1, ed.RealEigenvalueAt(3))
	}

}

func TestEigenDecompositionDimension4WithoutSplit(t *testing.T) {
	m := createRealMatrixFromSlices(t, [][]float64{
		{0.5608, -0.2016, 0.1152, -0.2976},
		{-0.2016, 0.4432, -0.2304, 0.1152},
		{0.1152, -0.2304, 0.3088, -0.1344},
		{-0.2976, 0.1152, -0.1344, 0.3872},
	})
	ed, _ := NewEigenDecomposition(m)

	if math.Abs(ed.RealEigenvalueAt(0)-1.0) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 1.0, ed.RealEigenvalueAt(0))
	}
	if math.Abs(ed.RealEigenvalueAt(1)-0.4) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.4, ed.RealEigenvalueAt(1))
	}
	if math.Abs(ed.RealEigenvalueAt(2)-0.2) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.2, ed.RealEigenvalueAt(2))
	}
	if math.Abs(ed.RealEigenvalueAt(3)-0.1) > 1.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.1, ed.RealEigenvalueAt(3))
	}

}

func TestEigenDecompositionMathpbx02(t *testing.T) {

	mainTridiagonal := []float64{
		7484.860960227216, 18405.28129035345, 13855.225609560746,
		10016.708722343366, 559.8117399576674, 6750.190788301587,
		71.21428769782159,
	}
	secondaryTridiagonal := []float64{
		-4175.088570476366, 1975.7955858241994, 5193.178422374075,
		1995.286659169179, 75.34535882933804, -234.0808002076056,
	}

	// the reference values have been computed using routine DSTEMR
	// from the fortran library LAPACK version 3.2.1
	refEigenValues := []float64{
		20654.744890306974412, 16828.208208485466457,
		6893.155912634994820, 6757.083016675340332,
		5887.799885688558788, 64.309089923240379,
		57.992628792736340,
	}
	refEigenVectors := []RealVector{
		createArrayRealVectorFromSlice(t, []float64{-0.270356342026904, 0.852811091326997, 0.399639490702077, 0.198794657813990, 0.019739323307666, 0.000106983022327, -0.000001216636321}),
		createArrayRealVectorFromSlice(t, []float64{0.179995273578326, -0.402807848153042, 0.701870993525734, 0.555058211014888, 0.068079148898236, 0.000509139115227, -0.000007112235617}),
		createArrayRealVectorFromSlice(t, []float64{-0.399582721284727, -0.056629954519333, -0.514406488522827, 0.711168164518580, 0.225548081276367, 0.125943999652923, -0.004321507456014}),
		createArrayRealVectorFromSlice(t, []float64{0.058515721572821, 0.010200130057739, 0.063516274916536, -0.090696087449378, -0.017148420432597, 0.991318870265707, -0.034707338554096}),
		createArrayRealVectorFromSlice(t, []float64{0.855205995537564, 0.327134656629775, -0.265382397060548, 0.282690729026706, 0.105736068025572, -0.009138126622039, 0.000367751821196}),
		createArrayRealVectorFromSlice(t, []float64{-0.002913069901144, -0.005177515777101, 0.041906334478672, -0.109315918416258, 0.436192305456741, 0.026307315639535, 0.891797507436344}),
		createArrayRealVectorFromSlice(t, []float64{-0.005738311176435, -0.010207611670378, 0.082662420517928, -0.215733886094368, 0.861606487840411, -0.025478530652759, -0.451080697503958}),
	}

	// the following line triggers the exception
	decomposition, _ := NewEigenDecompositionFromTridiagonal(mainTridiagonal, secondaryTridiagonal)

	eigenValues := decomposition.RealEigenvalues()
	for i := 0; i < len(refEigenValues); i++ {
		var expEigenVecNorm float64
		if math.Abs(eigenValues[i]-refEigenValues[i]) > 1.0e-3 {
			t.Errorf("Mismatch. want: %v, got: %v", eigenValues[i], eigenValues[i])
		}

		if VecDotProduct(refEigenVectors[i], decomposition.EigenvectorAt(i)) < 0 {
			expEigenVecNorm = VecNorm(refEigenVectors[i].Add(decomposition.EigenvectorAt(i)))
			if math.Abs(0-expEigenVecNorm) > 1.0e-5 {
				t.Errorf("Mismatch. want: %v, got: %v", 0, expEigenVecNorm)
			}
		} else {
			expEigenVecNorm = VecNorm(refEigenVectors[i].Subtract(decomposition.EigenvectorAt(i)))
			if math.Abs(0-expEigenVecNorm) > 1.0e-5 {
				t.Errorf("Mismatch. want: %v, got: %v", 0, expEigenVecNorm)
			}
		}
	}

}

func TestEigenDecompositionMathpbx03(t *testing.T) {

	mainTridiagonal := []float64{
		1809.0978259647177, 3395.4763425956166, 1832.1894584712693, 3804.364873592377,
		806.0482458637571, 2403.656427234185, 28.48691431556015,
	}
	secondaryTridiagonal := []float64{
		-656.8932064545833, -469.30804108920734, -1021.7714889369421,
		-1152.540497328983, -939.9765163817368, -12.885877015422391,
	}

	// the reference values have been computed using routine DSTEMR
	// from the fortran library LAPACK version 3.2.1
	refEigenValues := []float64{
		4603.121913685183245, 3691.195818048970978, 2743.442955402465032, 1657.596442107321764,
		1336.797819095331306, 30.129865209677519, 17.035352085224986,
	}

	refEigenVectors := []RealVector{
		createArrayRealVectorFromSlice(t, []float64{-0.036249830202337, 0.154184732411519, -0.346016328392363, 0.867540105133093, -0.294483395433451, 0.125854235969548, -0.000354507444044}),
		createArrayRealVectorFromSlice(t, []float64{-0.318654191697157, 0.912992309960507, -0.129270874079777, -0.184150038178035, 0.096521712579439, -0.070468788536461, 0.000247918177736}),
		createArrayRealVectorFromSlice(t, []float64{-0.051394668681147, 0.073102235876933, 0.173502042943743, -0.188311980310942, -0.327158794289386, 0.905206581432676, -0.004296342252659}),
		createArrayRealVectorFromSlice(t, []float64{0.838150199198361, 0.193305209055716, -0.457341242126146, -0.166933875895419, 0.094512811358535, 0.119062381338757, -0.000941755685226}),
		createArrayRealVectorFromSlice(t, []float64{0.438071395458547, 0.314969169786246, 0.768480630802146, 0.227919171600705, -0.193317045298647, -0.170305467485594, 0.001677380536009}),
		createArrayRealVectorFromSlice(t, []float64{-0.003726503878741, -0.010091946369146, -0.067152015137611, -0.113798146542187, -0.313123000097908, -0.118940107954918, 0.932862311396062}),
		createArrayRealVectorFromSlice(t, []float64{0.009373003194332, 0.025570377559400, 0.170955836081348, 0.291954519805750, 0.807824267665706, 0.320108347088646, 0.360202112392266}),
	}

	// the following line triggers the exception
	decomposition, _ := NewEigenDecompositionFromTridiagonal(mainTridiagonal, secondaryTridiagonal)

	eigenValues := decomposition.RealEigenvalues()
	for i := 0; i < len(refEigenValues); i++ {
		var expEigenVecNorm float64
		if math.Abs(eigenValues[i]-refEigenValues[i]) > 1.0e-4 {
			t.Errorf("Mismatch. want: %v, got: %v", eigenValues[i], eigenValues[i])
		}
		if VecDotProduct(refEigenVectors[i], decomposition.EigenvectorAt(i)) < 0 {
			expEigenVecNorm = VecNorm(refEigenVectors[i].Add(decomposition.EigenvectorAt(i)))
			if math.Abs(0-expEigenVecNorm) > 1.0e-5 {
				t.Errorf("Mismatch. want: %v, got: %v", 0, expEigenVecNorm)
			}
		} else {
			expEigenVecNorm = VecNorm(refEigenVectors[i].Subtract(decomposition.EigenvectorAt(i)))
			if math.Abs(0-expEigenVecNorm) > 1.0e-5 {
				t.Errorf("Mismatch. want: %v, got: %v", 0, expEigenVecNorm)
			}
		}

	}

}

func TestEigenDecompositionTridiagonal(t *testing.T) {
	s := rand.NewSource(4366663527842)
	rg := rand.New(s)
	ref := make([]float64, 30)
	for i := 0; i < len(ref); i++ {
		if i < 5 {
			ref[i] = 2*rg.Float64() - 1
		} else {
			ref[i] = 0.0001*rg.Float64() + 6
		}
	}
	sort.Float64s(ref)

	tt, _ := NewTriDiagonalTransformer(createTestMatrix(rg, ref))

	ed, _ := NewEigenDecompositionFromTridiagonal(tt.MainDiagonalRef(), tt.SecondaryDiagonalRef())
	eigenValues := ed.RealEigenvalues()
	if len(ref) != len(eigenValues) {
		t.Errorf("Mismatch. want: %v, got: %v", len(ref), len(eigenValues))
	}

	for i := 0; i < len(ref); i++ {
		if math.Abs(ref[len(ref)-i-1]-eigenValues[i]) > 2.0e-14 {
			t.Errorf("Mismatch. want: %v, got: %v", ref[len(ref)-i-1], eigenValues[i])
		}

	}

}

func TestEigenDecompositionDimensions(t *testing.T) {
	m := mat_ed.RowDimension()
	ed, _ := NewEigenDecomposition(mat_ed)

	if ed.V().RowDimension() != m {
		t.Errorf("Mismatch. testData row dimension. want: %v, got: %v", m, ed.V().RowDimension())
	}

	if ed.V().ColumnDimension() != m {
		t.Errorf("Mismatch. testData column dimension. want: %v, got: %v", m, ed.V().ColumnDimension())
	}

	if ed.D().RowDimension() != m {
		t.Errorf("Mismatch. testData row dimension. want: %v, got: %v", m, ed.D().RowDimension())
	}

	if ed.D().ColumnDimension() != m {
		t.Errorf("Mismatch. testData column dimension. want: %v, got: %v", m, ed.D().ColumnDimension())
	}

	if ed.VT().RowDimension() != m {
		t.Errorf("Mismatch. testData row dimension. want: %v, got: %v", m, ed.VT().RowDimension())
	}

	if ed.VT().ColumnDimension() != m {
		t.Errorf("Mismatch. testData column dimension. want: %v, got: %v", m, ed.VT().ColumnDimension())
	}
}

func TestEigenDecompositionEigenvalues(t *testing.T) {
	ed, _ := NewEigenDecomposition(mat_ed)
	eigenValues := ed.RealEigenvalues()
	if len(refValues) != len(eigenValues) {
		t.Errorf("Mismatch. want: %v, got: %v", len(refValues), len(eigenValues))
	}
	for i := 0; i < len(refValues); i++ {
		if math.Abs(refValues[i]-eigenValues[i]) > 3.0e-15 {
			t.Errorf("Mismatch. want: %v, got: %v", refValues[i], eigenValues[i])
		}
	}
}

func TestEigenDecompositionBigMatrix(t *testing.T) {
	s := rand.NewSource(4366663527842)
	r := rand.New(s)
	bigValues := make([]float64, 200)
	for i := 0; i < len(bigValues); i++ {
		bigValues[i] = 2*r.Float64() - 1
	}
	sort.Float64s(bigValues)
	ed, _ := NewEigenDecomposition(createTestMatrix(r, bigValues))
	eigenValues := ed.RealEigenvalues()
	if len(bigValues) != len(eigenValues) {
		t.Errorf("Mismatch. want: %v, got: %v", len(bigValues), len(eigenValues))
	}
	for i := 0; i < len(bigValues); i++ {
		if math.Abs(bigValues[len(bigValues)-i-1]-eigenValues[i]) > 2.0e-14 {
			t.Errorf("Mismatch. want: %v, got: %v", bigValues[len(bigValues)-i-1], eigenValues[i])
		}

	}
}

func TestEigenDecompositionSymmetric(t *testing.T) {
	symmetric := createRealMatrixFromSlices(t, [][]float64{
		{4, 1, 1},
		{1, 2, 3},
		{1, 3, 6},
	})

	ed, _ := NewEigenDecomposition(symmetric)

	d := ed.D()
	v := ed.V()
	vT := ed.VT()

	norm := MatLInfNorm(v.Multiply(d).Multiply(vT).Subtract(symmetric))
	if math.Abs(0-norm) > 6.0e-13 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, norm)
	}

}

func TestEigenDecompositionSquareRoot(t *testing.T) {
	data := [][]float64{
		{33, 24, 7},
		{24, 57, 11},
		{7, 11, 9},
	}

	dec, _ := NewEigenDecomposition(createRealMatrixFromSlices(t, data))
	sqrtM := dec.SquareRoot()

	// Reconstruct initial
	m := sqrtM.Multiply(sqrtM)

	dim := len(data)
	for r := 0; r < dim; r++ {
		for c := 0; c < dim; c++ {
			if math.Abs(data[r][c]-m.At(r, c)) > 6.0e-13 {
				t.Errorf("Mismatch. m[ %v ][ %v ]. want: %v, got: %v", r, c, data[r][c], m.At(r, c))
			}

		}
	}
}

func TestEigenDecompositionSquareRootNonSymmetric(t *testing.T) {
	data := [][]float64{
		{1, 2, 4},
		{2, 3, 5},
		{11, 5, 9},
	}

	dec, _ := NewEigenDecomposition(createRealMatrixFromSlices(t, data))
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	dec.SquareRoot()
}

func TestEigenDecompositionSquareRootNonPositiveDefinite(t *testing.T) {
	data := [][]float64{
		{1, 2, 4},
		{2, 3, 5},
		{4, 5, -9},
	}

	dec, _ := NewEigenDecomposition(createRealMatrixFromSlices(t, data))
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	dec.SquareRoot()
}

func TestEigenDecompositionUnsymmetric(t *testing.T) {
	// Vandermonde matrix V(x;i,j) = x_i^{n - j} with x = (-1,-2,3,4)
	vData := [][]float64{{-1.0, 1.0, -1.0, 1.0},
		{-8.0, 4.0, -2.0, 1.0},
		{27.0, 9.0, 3.0, 1.0},
		{64.0, 16.0, 4.0, 1.0}}
	checkUnsymmetricMatrix(t, createRealMatrixFromSlices(t, vData))

	randMatrix := createRealMatrixFromSlices(t, [][]float64{
		{0, 1, 0, 0},
		{1, 0, 2.e-7, 0},
		{0, -2.e-7, 0, 1},
		{0, 0, 1, 0},
	})
	checkUnsymmetricMatrix(t, randMatrix)

	// from http://eigen.tuxfamily.org/dox/classEigen_1_1RealSchur.html
	randData2 := [][]float64{
		{0.680, -0.3300, -0.2700, -0.717, -0.687, 0.0259},
		{-0.211, 0.5360, 0.0268, 0.214, -0.198, 0.6780},
		{0.566, -0.4440, 0.9040, -0.967, -0.740, 0.2250},
		{0.597, 0.1080, 0.8320, -0.514, -0.782, -0.4080},
		{0.823, -0.0452, 0.2710, -0.726, 0.998, 0.2750},
		{-0.605, 0.2580, 0.4350, 0.608, -0.563, 0.0486},
	}
	checkUnsymmetricMatrix(t, createRealMatrixFromSlices(t, randData2))
}

func TestEigenDecompositionEigenvectors(t *testing.T) {
	ed, _ := NewEigenDecomposition(mat_ed)
	for i := 0; i < mat_ed.RowDimension(); i++ {
		lambda := ed.RealEigenvalueAt(i)
		v := ed.EigenvectorAt(i)
		mV := mat_ed.OperateVector(v)
		v.MapMultiply(lambda)
		if math.Abs(0.0-VecNorm(mV.Subtract(v))) > 1.0e-13 {
			t.Errorf("Mismatch. want: %v, got: %v", 0.0, VecNorm(mV.Subtract(v)))
		}

	}
}

func TestEigenDecompositionAEqualVDVt(t *testing.T) {
	ed, _ := NewEigenDecomposition(mat_ed)
	v := ed.V()
	d := ed.D()
	vT := ed.VT()
	norm := MatLInfNorm(v.Multiply(d).Multiply(vT).Subtract(mat_ed))
	if math.Abs(0.0-norm) > 6.0e-13 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, norm)
	}

}

func TestEigenDecompositionVOrthogonal(t *testing.T) {
	mat, _ := NewEigenDecomposition(mat_ed)
	v := mat.V()
	vTv := v.Transpose().Multiply(v)
	id, _ := NewRealIdentityMatrix(vTv.RowDimension())
	if math.Abs(0.0-MatLInfNorm(vTv.Subtract(id))) > 2.0e-13 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, MatLInfNorm(vTv.Subtract(id)))
	}
}

func TestEigenDecompositionDiagonal(t *testing.T) {
	diagonal := []float64{-3.0, -2.0, 2.0, 5.0}
	m, _ := NewRealDiagonalMatrix(diagonal)
	ed, _ := NewEigenDecomposition(m)
	if math.Abs(diagonal[0]-ed.RealEigenvalueAt(3)) > 2.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", diagonal[0], ed.RealEigenvalueAt(3))
	}
	if math.Abs(diagonal[1]-ed.RealEigenvalueAt(2)) > 2.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", diagonal[1], ed.RealEigenvalueAt(2))
	}
	if math.Abs(diagonal[2]-ed.RealEigenvalueAt(1)) > 2.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", diagonal[2], ed.RealEigenvalueAt(2))
	}
	if math.Abs(diagonal[3]-ed.RealEigenvalueAt(0)) > 2.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", diagonal[3], ed.RealEigenvalueAt(2))
	}

}

func TestEigenDecompositionRepeatedEigenvalue(t *testing.T) {
	repeated := createRealMatrixFromSlices(t, [][]float64{
		{3, 2, 4},
		{2, 0, 2},
		{4, 2, 3},
	})
	ed, _ := NewEigenDecomposition(repeated)
	checkEigenValues(t, []float64{8, -1, -1}, ed, 1e-12)
	checkEigenVector(t, []float64{2, 1, 2}, ed, 1e-12)
}

func TestEigenDecompositionZeroDivide(t *testing.T) {
	indefinite := createRealMatrixFromSlices(t, [][]float64{
		{0.0, 1.0, -1.0},
		{1.0, 1.0, 0.0},
		{-1.0, 0.0, 1.0},
	})
	ed, _ := NewEigenDecomposition(indefinite)
	checkEigenValues(t, []float64{2, 1, -1}, ed, 1e-12)
	isqrt3 := 1 / math.Sqrt(3.0)
	checkEigenVector(t, []float64{isqrt3, isqrt3, -isqrt3}, ed, 1e-12)
	isqrt2 := 1 / math.Sqrt(2.0)
	checkEigenVector(t, []float64{0.0, -isqrt2, -isqrt2}, ed, 1e-12)
	isqrt6 := 1 / math.Sqrt(6.0)
	checkEigenVector(t, []float64{2 * isqrt6, -isqrt6, isqrt6}, ed, 1e-12)
}

func TestEigenDecompositionTinyValues(t *testing.T) {
	tiny := 1e-100
	distinct := createRealMatrixFromSlices(t, [][]float64{
		{3, 1, -4},
		{1, 3, -4},
		{-4, -4, 8},
	})
	distinct = distinct.ScalarMultiply(tiny)

	ed, _ := NewEigenDecomposition(distinct)
	s := []float64{2, 0, 12}
	for i := 0; i < len(s); i++ {
		s[i] *= tiny
	}

	checkEigenValues(t, s, ed, 1e-12*tiny)
	checkEigenVector(t, []float64{1, -1, 0}, ed, 1e-12)
	checkEigenVector(t, []float64{1, 1, 1}, ed, 1e-12)
	checkEigenVector(t, []float64{-1, -1, 2}, ed, 1e-12)
}

func checkEigenVector(t *testing.T, eigenVector []float64, ed *EigenDecomposition, tolerance float64) {
	if !isIncludedColumn(eigenVector, ed.V(), tolerance) {
		t.Errorf("Mismatch. vector not included.")
	}

}

func checkEigenValues(t *testing.T, targetValues []float64, ed *EigenDecomposition, tolerance float64) {
	observed := ed.RealEigenvalues()
	for i := 0; i < len(observed); i++ {
		if !isIncludedValue(observed[i], targetValues, tolerance) {
			t.Errorf("Mismatch. %v not included.", observed[i])
		}
		if !isIncludedValue(targetValues[i], observed, tolerance) {
			t.Errorf("Mismatch. %v not included.", targetValues[i])
		}

	}
}

func isIncludedColumn(column []float64, searchMatrix RealMatrix, tolerance float64) bool {
	var found bool
	i := 0
	for !found && i < searchMatrix.ColumnDimension() {
		multiplier := 1.0
		matching := true
		j := 0
		for matching && j < searchMatrix.RowDimension() {
			colEntry := searchMatrix.At(j, i)
			// Use the first entry where both are non-zero as scalar
			if math.Abs(multiplier-1.0) <= LSBFloat64(1.) && math.Abs(colEntry) > 1e-14 && math.Abs(column[j]) > 1e-14 {
				multiplier = colEntry / column[j]
			}
			if math.Abs(column[j]*multiplier-colEntry) > tolerance {
				matching = false
			}
			j++
		}
		found = matching
		i++
	}
	return found
}

func LSBFloat64(x float64) float64 {
	if math.IsInf(x, -1) || math.IsInf(x, 1) {
		return math.Inf(1)
	}
	return math.Abs(x - math.Float64frombits(math.Float64bits(x)^1))
}

func isIncludedValue(value float64, searchArray []float64, tolerance float64) bool {
	var found bool
	i := 0
	for !found && i < len(searchArray) {
		if math.Abs(value-searchArray[i]) < tolerance {
			found = true
		}
		i++
	}
	return found
}

func checkUnsymmetricMatrix(t *testing.T, m RealMatrix) {

	ed, _ := NewEigenDecomposition(m)

	d := ed.D()
	v := ed.V()
	//RealMatrix vT = ed.getVT();

	x := m.Multiply(v)
	y := v.Multiply(d)

	diffNorm := MatLInfNorm(x.Subtract(y))
	if diffNorm >= 1000*doubleeps*math.Max(MatLInfNorm(x), MatLInfNorm(y)) {
		t.Errorf("The norm of (X-Y) is too large.")
	}

	lud, _ := NewLUDecomposition(v)
	invV := lud.Solver().Inverse()
	norm := MatLInfNorm(v.Multiply(d).Multiply(invV).Subtract(m))
	if math.Abs(0.0-norm) > 1.0e-10 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, norm)
	}

}

func createTestMatrix(r *rand.Rand, eigenValues []float64) RealMatrix {
	n := len(eigenValues)
	v := createOrthogonalMatrix(r, n)
	d, _ := NewRealMatrixWithDiagonal(eigenValues)
	return v.Multiply(d).Multiply(v.Transpose())
}

func createOrthogonalMatrix(r *rand.Rand, size int) RealMatrix {

	data := make([][]float64, size)

	for i := 0; i < size; i++ {
		data[i] = make([]float64, size)
		dataI := data[i]
		var norm2 float64
		for ok := true; ok; ok = norm2*float64(size) < 0.01 {

			// generate randomly row I
			for j := 0; j < size; j++ {
				dataI[j] = 2*r.Float64() - 1
			}

			// project the row in the subspace orthogonal to previous rows
			for k := 0; k < i; k++ {
				dataK := data[k]
				var dotProduct float64
				for j := 0; j < size; j++ {
					dotProduct += dataI[j] * dataK[j]
				}
				for j := 0; j < size; j++ {
					dataI[j] -= dotProduct * dataK[j]
				}
			}

			// normalize the row
			norm2 = 0
			for _, dataIJ := range dataI {
				norm2 += dataIJ * dataIJ
			}
			inv := 1.0 / math.Sqrt(norm2)
			for j := 0; j < size; j++ {
				dataI[j] *= inv
			}

		}
	}

	mat, _ := NewRealMatrixFromSlices(data)
	return mat
}
