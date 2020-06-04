package linear

import (
	"math"
	"math/rand"
	"testing"
)

var (
	// 3 x 3 identity matrix
	id = [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}

	// Test data for group operations
	testData        = [][]float64{{1, 2, 3}, {2, 5, 3}, {1, 0, 8}}
	testDataLU      = [][]float64{{2, 5, 3}, {.5, -2.5, 6.5}, {0.5, 0.2, .2}}
	testDataPlus2   = [][]float64{{3, 4, 5}, {4, 7, 5}, {3, 2, 10}}
	testDataMinus   = [][]float64{{-1, -2, -3}, {-2, -5, -3}, {-1, 0, -8}}
	testDataRow1    = []float64{1, 2, 3}
	testDataCol3    = []float64{3, 3, 8}
	testDataInv     = [][]float64{{-40, 16, 9}, {13, -5, -3}, {5, -2, -1}}
	preMultTest     = []float64{8, 12, 33}
	testData2       = [][]float64{{1, 2, 3}, {2, 5, 3}}
	testData2T      = [][]float64{{1, 2}, {2, 5}, {3, 3}}
	testDataPlusInv = [][]float64{{-39, 18, 12}, {15, 0, 0}, {6, -2, 7}}
	d3              = [][]float64{{1, 2, 3, 4}, {5, 6, 7, 8}}
	d4              = [][]float64{{1}, {2}, {3}, {4}}
	d5              = [][]float64{{30}, {70}}

	// lu decomposition tests
	luData                = [][]float64{[]float64{2, 3, 3}, []float64{0, 5, 7}, []float64{6, 9, 8}}
	luDataLUDecomposition = [][]float64{[]float64{6, 9, 8}, []float64{0, 5, 7}, []float64{0.33333333333333, 0, 0.33333333333333}}

	// singular matrices
	singular    = [][]float64{[]float64{2, 3}, []float64{2, 3}}
	bigSingular = [][]float64{[]float64{1, 2, 3, 4}, []float64{2, 5, 3, 4}, []float64{7, 3, 256, 1930}, []float64{3, 7, 6, 8}} // 4th row = 1st + 2nd
	detData     = [][]float64{[]float64{1, 2, 3}, []float64{4, 5, 6}, []float64{7, 8, 10}}
	detData2    = [][]float64{[]float64{1, 3}, []float64{2, 4}}

	// vectors
	testVector  = []float64{1, 2, 3}
	testVector2 = []float64{1, 2, 3, 4}

	// submatrix accessor tests
	subTestData = [][]float64{{1, 2, 3, 4}, {1.5, 2.5, 3.5, 4.5}, {2, 4, 6, 8}, {4, 5, 6, 7}}
	// array selections
	subRows02Cols13  = [][]float64{{2, 4}, {4, 8}}
	subRows03Cols12  = [][]float64{{2, 3}, {5, 6}}
	subRows03Cols123 = [][]float64{{2, 3, 4}, {5, 6, 7}}
	// effective permutations
	subRows20Cols123 = [][]float64{{4, 6, 8}, {2, 3, 4}}
	subRows31Cols31  = [][]float64{{7, 5}, {4.5, 2.5}}
	// contiguous ranges
	subRows01Cols23 = [][]float64{{3, 4}, {3.5, 4.5}}
	subRows23Cols00 = [][]float64{{2}, {4}}
	subRows00Cols33 = [][]float64{{4}}
	// row matrices
	subRow0 = [][]float64{{1, 2, 3, 4}}
	subRow3 = [][]float64{{4, 5, 6, 7}}
	// column matrices
	subColumn1 = [][]float64{{2}, {2.5}, {4}, {5}}
	subColumn3 = [][]float64{{4}, {4.5}, {8}, {7}}

	// tolerances
	entryTolerance = 10e-16
	normTolerance  = 10e-14
	powerTolerance = 10e-16
)

func createBlockRealMatrixFromSlices(t *testing.T, data [][]float64) *BlockRealMatrix {
	m, err := NewBlockRealMatrixFromSlices(data)

	if err != nil {
		t.Errorf("Error while creating Matrix %s", err)
	}

	return m
}

func createBlockRealMatrix(t *testing.T, rowDimension, columnDimension int) *BlockRealMatrix {
	m, err := NewBlockRealMatrix(rowDimension, columnDimension)

	if err != nil {
		t.Errorf("Error while creating Matrix %s", err)
	}

	return m
}

func TestBlockRealMatrixDimensions(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, testData)
	m2 := createBlockRealMatrixFromSlices(t, testData2)

	if m.RowDimension() != 3 {
		t.Errorf("Mismatch. testData row dimension. want: %v, got: %v", 3, m.RowDimension())
	}

	if m.ColumnDimension() != 3 {
		t.Errorf("Mismatch. testData column dimension. want: %v, got: %v", 3, m.ColumnDimension())
	}

	if !IsSquare(m) {
		t.Errorf("Mismatch. testData is not square. want: true, got: %v", IsSquare(m))
	}

	if m2.RowDimension() != 2 {
		t.Errorf("Mismatch. testData2 row dimension. want: %v, got: %v", 2, m2.RowDimension())
	}

	if m2.ColumnDimension() != 3 {
		t.Errorf("Mismatch. testData2 column dimension. want: %v, got: %v", 3, m2.ColumnDimension())
	}

	if IsSquare(m2) {
		t.Errorf("Mismatch. testData2 is square. want: false, got: %v", IsSquare(m2))
	}

}

func TestBlockRealMatrixCopy(t *testing.T) {
	rs := rand.NewSource(66636328996002)
	r := rand.New(rs)
	m1 := createRandomMatrix(r, 47, 83)
	m2 := createBlockRealMatrixFromSlices(t, m1.Data())

	if !m1.Equals(m2) {
		t.Errorf("Mismatch. m1 not equal m2.")
	}

	m3 := createBlockRealMatrixFromSlices(t, testData)
	m4 := createBlockRealMatrixFromSlices(t, m3.Data())
	if !m3.Equals(m4) {
		t.Errorf("Mismatch. m3 not equal m4.")
	}

}

func TestBlockRealMatrixAdd(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, testData)
	mInv := createBlockRealMatrixFromSlices(t, testDataInv)
	mPlusMInv := m.Add(mInv)
	sumEntries := mPlusMInv.Data()
	for row := 0; row < m.RowDimension(); row++ {
		for col := 0; col < m.ColumnDimension(); col++ {
			if math.Abs(testDataPlusInv[row][col]-sumEntries[row][col]) > entryTolerance {
				t.Errorf("Mismatch. sum entry not equal testDataPlusInv entry, want: %v, got: %v", sumEntries[row][col], testDataPlusInv[row][col])
			}
		}
	}
}

func TestBlockRealMatrixAddFail(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, testData)
	m2 := createBlockRealMatrixFromSlices(t, testData2)

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	m.Add(m2)

}

func TestBlockRealMatrixNorm(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, testData)
	m2 := createBlockRealMatrixFromSlices(t, testData2)
	if math.Abs(MatLInfNorm(m)-14) > entryTolerance {
		t.Errorf("Mismatch. testData norm, want: %v, got: %v", 14, MatLInfNorm(m))
	}

	if math.Abs(MatLInfNorm(m2)-7) > entryTolerance {
		t.Errorf("Mismatch. testData2 norm, want: %v, got: %v", 7, MatLInfNorm(m2))
	}
}

func TestBlockRealMatrixMatFrobeniusNorm(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, testData)
	mExp := math.Sqrt(117.0)
	m2 := createBlockRealMatrixFromSlices(t, testData2)
	m2Exp := math.Sqrt(52.0)
	if math.Abs(MatFrobeniusNorm(m)-mExp) > entryTolerance {
		t.Errorf("Mismatch. testData norm, want: %v, got: %v", mExp, MatFrobeniusNorm(m))
	}

	if math.Abs(MatFrobeniusNorm(m2)-m2Exp) > entryTolerance {
		t.Errorf("Mismatch. testData2 norm, want: %v, got: %v", m2Exp, MatFrobeniusNorm(m2))
	}
}

func TestBlockRealMatrixPlusMinus(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, testData)
	m2 := createBlockRealMatrixFromSlices(t, testDataInv)
	assertCloseMatrix(t, m.Subtract(m2), m2.ScalarMultiply(-1).Add(m), entryTolerance)

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	m.Subtract(createBlockRealMatrixFromSlices(t, testData2))

}

func TestBlockRealMatrixMultiply(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, testData)
	mInv := createBlockRealMatrixFromSlices(t, testDataInv)
	identity := createBlockRealMatrixFromSlices(t, id)
	m2 := createBlockRealMatrixFromSlices(t, testData2)
	assertCloseMatrix(t, m.Multiply(mInv), identity, entryTolerance)
	assertCloseMatrix(t, mInv.Multiply(m), identity, entryTolerance)
	assertCloseMatrix(t, m.Multiply(identity), m, entryTolerance)
	assertCloseMatrix(t, identity.Multiply(mInv), mInv, entryTolerance)
	assertCloseMatrix(t, m2.Multiply(identity), m2, entryTolerance)

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	m.Multiply(createBlockRealMatrixFromSlices(t, bigSingular))

}

func TestBlockRealMatrixMultiply2(t *testing.T) {
	m3 := createBlockRealMatrixFromSlices(t, d3)
	m4 := createBlockRealMatrixFromSlices(t, d4)
	m5 := createBlockRealMatrixFromSlices(t, d5)
	assertCloseMatrix(t, m3.Multiply(m4), m5, entryTolerance)
}

func TestBlockRealMatrixSeveralBlocks(t *testing.T) {
	m := createBlockRealMatrix(t, 35, 71)
	for i := 0; i < m.RowDimension(); i++ {
		for j := 0; j < m.ColumnDimension(); j++ {
			m.SetEntry(i, j, float64(i)+float64(j)/1024.0)
		}
	}

	mT := m.Transpose()
	if m.RowDimension() != mT.ColumnDimension() {
		t.Errorf("Mismatch. row dimension not equal transposed column dimension. m: %v, mT: %v", m.RowDimension(), mT.ColumnDimension())
	}

	if m.ColumnDimension() != mT.RowDimension() {
		t.Errorf("Mismatch. column dimension not equal transposed row dimension. m: %v, mT: %v", m.ColumnDimension(), mT.RowDimension())
	}

	mPm := m.Add(m)
	for i := 0; i < mPm.RowDimension(); i++ {
		for j := 0; j < mPm.ColumnDimension(); j++ {
			if 2*m.At(i, j) != mPm.At(i, j) {
				t.Errorf("Mismatch. want: %v, got: %v", 2*m.At(i, j), mPm.At(i, j))
			}
		}
	}

	mPmMm := mPm.Subtract(m)
	for i := 0; i < mPmMm.RowDimension(); i++ {
		for j := 0; j < mPmMm.ColumnDimension(); j++ {
			if m.At(i, j) != mPmMm.At(i, j) {
				t.Errorf("Mismatch. want: %v, got: %v", m.At(i, j), mPmMm.At(i, j))
			}

		}
	}

	mTm := mT.Multiply(m)
	for i := 0; i < mTm.RowDimension(); i++ {
		for j := 0; j < mTm.ColumnDimension(); j++ {
			var sum float64
			for k := 0; k < mT.ColumnDimension(); k++ {
				sum += (float64(k) + float64(i)/1024.0) * (float64(k) + float64(j)/1024.0)
			}
			if sum != mTm.At(i, j) {
				t.Errorf("Mismatch. want: %v, got: %v", sum, mTm.At(i, j))
			}

		}
	}

	mmT := m.Multiply(mT)
	for i := 0; i < mmT.RowDimension(); i++ {
		for j := 0; j < mmT.ColumnDimension(); j++ {
			var sum float64
			for k := 0; k < m.ColumnDimension(); k++ {
				sum += (float64(i) + float64(k)/1024.0) * (float64(j) + float64(k)/1024.0)
			}
			if sum != mmT.At(i, j) {
				t.Errorf("Mismatch. want: %v, got: %v", sum, mmT.At(i, j))
			}

		}
	}

	sub1 := m.SubMatrix(2, 9, 5, 20)
	for i := 0; i < sub1.RowDimension(); i++ {
		for j := 0; j < sub1.ColumnDimension(); j++ {
			if (float64(i)+2)+(float64(j)+5)/1024.0 != sub1.At(i, j) {
				t.Errorf("Mismatch. want: %v, got: %v", (float64(i)+2)+(float64(j)+5)/1024.0, sub1.At(i, j))
			}

		}
	}

	sub2 := m.SubMatrix(10, 12, 3, 70)
	for i := 0; i < sub2.RowDimension(); i++ {
		for j := 0; j < sub2.ColumnDimension(); j++ {
			if (float64(i)+10)+(float64(j)+3)/1024.0 != sub2.At(i, j) {
				t.Errorf("Mismatch. want: %v, got: %v", (float64(i)+10)+(float64(j)+3)/1024.0, sub2.At(i, j))
			}

		}
	}

	sub3 := m.SubMatrix(30, 34, 0, 5)
	for i := 0; i < sub3.RowDimension(); i++ {
		for j := 0; j < sub3.ColumnDimension(); j++ {
			if (float64(i)+30)+(float64(j)+0)/1024.0 != sub3.At(i, j) {
				t.Errorf("Mismatch. want: %v, got: %v", (float64(i)+30)+(float64(j)+0)/1024.0, sub3.At(i, j))
			}

		}
	}

	sub4 := m.SubMatrix(30, 32, 62, 65)
	for i := 0; i < sub4.RowDimension(); i++ {
		for j := 0; j < sub4.ColumnDimension(); j++ {
			if (float64(i)+30)+(float64(j)+62)/1024.0 != sub4.At(i, j) {
				t.Errorf("Mismatch. want: %v, got: %v", (float64(i)+30)+(float64(j)+62)/1024.0, sub4.At(i, j))
			}

		}
	}

}

func TestBlockRealMatrixTrace(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, id)
	if math.Abs(m.Trace()-3) > entryTolerance {
		t.Errorf("Mismatch. identity trace. want: %v, got: %v", 3, m.Trace())
	}

	m = createBlockRealMatrixFromSlices(t, testData2)

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	m.Trace()
}

func TestBlockRealMatrixScalarAdd(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, testData)
	assertCloseMatrix(t, createBlockRealMatrixFromSlices(t, testDataPlus2), m.ScalarAdd(2), entryTolerance)
}

func TestBlockRealMatrixOperateVector(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, id)
	assertClose(t, testVector, m.Operate(testVector), entryTolerance)
	assertClose(t, testVector, m.OperateVector(createArrayRealVectorFromSlice(t, testVector)).ToArray(), entryTolerance)
	m = createBlockRealMatrixFromSlices(t, bigSingular)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	m.Operate(testVector)
}

func TestBlockRealMatrixOperateLarge(t *testing.T) {
	p := (7 * BLOCK_SIZE) / 2
	q := (5 * BLOCK_SIZE) / 2
	r := 3 * BLOCK_SIZE
	rs := rand.NewSource(111007463902334)
	rg := rand.New(rs)

	m1 := createRandomMatrix(rg, p, q)
	m2 := createRandomMatrix(rg, q, r)
	m1m2 := m1.Multiply(m2)
	for i := 0; i < r; i++ {
		checkArrays(t, m1m2.ColumnAt(i), m1.Operate(m2.ColumnAt(i)))
	}
}

func TestBlockRealMatrixOperatePremultiplyLarge(t *testing.T) {
	p := (7 * BLOCK_SIZE) / 2
	q := (5 * BLOCK_SIZE) / 2
	r := 3 * BLOCK_SIZE
	rs := rand.NewSource(111007463902334)
	rg := rand.New(rs)
	m1 := createRandomMatrix(rg, p, q)
	m2 := createRandomMatrix(rg, q, r)
	m1m2 := m1.Multiply(m2)
	for i := 0; i < r; i++ {
		checkArrays(t, m1m2.RowAt(i), m2.PreMultiply(m1.RowAt(i)))
	}
}

func TestBlockRealMatrixTranspose(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, testData)
	mI, _ := NewLUDecomposition(m)
	mIT := mI.Solver().Inverse().Transpose()
	mT, _ := NewLUDecomposition(m.Transpose())
	mTI := mT.Solver().Inverse()
	assertCloseMatrix(t, mIT, mTI, normTolerance)
	m = createBlockRealMatrixFromSlices(t, testData2)
	mt := createBlockRealMatrixFromSlices(t, testData2T)
	assertCloseMatrix(t, mt, m.Transpose(), normTolerance)
}

func TestBlockRealMatrixPremultiplyVector(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, testData)
	assertClose(t, m.PreMultiply(testVector), preMultTest, normTolerance)
	assertClose(t, m.PreMultiply(createArrayRealVectorFromSlice(t, testVector).ToArray()), preMultTest, normTolerance)
	m = createBlockRealMatrixFromSlices(t, bigSingular)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	m.PreMultiply(testVector)
}

func TestBlockRealMatrixPremultiplyMatrix(t *testing.T) {
	m3 := createBlockRealMatrixFromSlices(t, d3)
	m4 := createBlockRealMatrixFromSlices(t, d4)
	m5 := createBlockRealMatrixFromSlices(t, d5)
	assertCloseMatrix(t, m4.PreMultiplyMatrix(m3), m5, entryTolerance)

	m := createBlockRealMatrixFromSlices(t, testData)
	mInv := createBlockRealMatrixFromSlices(t, testDataInv)
	identity := createBlockRealMatrixFromSlices(t, id)
	assertCloseMatrix(t, m.PreMultiplyMatrix(mInv), identity, entryTolerance)
	assertCloseMatrix(t, mInv.PreMultiplyMatrix(m), identity, entryTolerance)
	assertCloseMatrix(t, m.PreMultiplyMatrix(identity), m, entryTolerance)
	assertCloseMatrix(t, identity.PreMultiplyMatrix(mInv), mInv, entryTolerance)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	m.PreMultiplyMatrix(createBlockRealMatrixFromSlices(t, bigSingular))
}

func TestBlockRealMatrixVectorAt(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, testData)
	assertClose(t, m.RowAt(0), testDataRow1, entryTolerance)
	assertClose(t, m.ColumnAt(2), testDataCol3, entryTolerance)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	m.RowAt(10)
	m.ColumnAt(-1)
}

func TestBlockRealMatrixAt(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, testData)
	if math.Abs(m.At(0, 1)-2) > entryTolerance {
		t.Errorf("Mismatch. entry at. want: %v, got: %v", 2, m.At(0, 1))
	}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	m.At(10, 4)
}

func TestBlockRealMatrixSubMatrix(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, subTestData)
	checkSubMatrix(t, m, subRows23Cols00, 2, 3, 0, 0)
	checkSubMatrix(t, m, subRows00Cols33, 0, 0, 3, 3)
	checkSubMatrix(t, m, subRows01Cols23, 0, 1, 2, 3)
	checkSubMatrixFromIndices(t, m, subRows02Cols13, []int{0, 2}, []int{1, 3})
	checkSubMatrixFromIndices(t, m, subRows03Cols12, []int{0, 3}, []int{1, 2})
	checkSubMatrixFromIndices(t, m, subRows03Cols123, []int{0, 3}, []int{1, 2, 3})
	checkSubMatrixFromIndices(t, m, subRows20Cols123, []int{2, 0}, []int{1, 2, 3})
	checkSubMatrixFromIndices(t, m, subRows31Cols31, []int{3, 1}, []int{3, 1})
	checkSubMatrixFromIndices(t, m, subRows31Cols31, []int{3, 1}, []int{3, 1})
	checkSubMatrix(t, m, nil, 1, 0, 2, 4)
	checkSubMatrix(t, m, nil, -1, 1, 2, 2)
	checkSubMatrix(t, m, nil, 1, 0, 2, 2)
	checkSubMatrix(t, m, nil, 1, 0, 2, 4)
	checkSubMatrixFromIndices(t, m, nil, []int{}, []int{0})
	checkSubMatrixFromIndices(t, m, nil, []int{0}, []int{4})
}

func TestBlockRealMatrixSetMatrixLarge(t *testing.T) {
	n := 3 * BLOCK_SIZE
	m := createBlockRealMatrix(t, n, n)
	sub := createBlockRealMatrix(t, n-4, n-4).ScalarAdd(1)

	m.SetSubMatrix(sub.Data(), 2, 2)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if (i < 2) || (i > n-3) || (j < 2) || (j > n-3) {
				if m.At(i, j) != 0. {
					t.Errorf("Mismatch. entry at. want: %v, got: %v", 0., m.At(i, j))
				}

			} else {
				if m.At(i, j) != 1.0 {
					t.Errorf("Mismatch. entry at. want: %v, got: %v", 1.0, m.At(i, j))
				}

			}
		}
	}

	if !sub.Equals(m.SubMatrix(2, n-3, 2, n-3)) {
		t.Errorf("Mismatch. sub mat not equal expected mat.")
	}
}

func TestBlockRealMatrixRowMatrixAt(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, subTestData)
	mRow0 := createBlockRealMatrixFromSlices(t, subRow0)
	mRow3 := createBlockRealMatrixFromSlices(t, subRow3)
	if !mRow0.Equals(m.RowMatrixAt(0)) {
		t.Errorf("Mismatch. mat not equal to row0 mat.")
	}

	if !mRow3.Equals(m.RowMatrixAt(3)) {
		t.Errorf("Mismatch. mat not equal to row3 mat.")
	}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	m.RowMatrixAt(-1)

}

func TestBlockRealMatrixSetRowMatrix(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, subTestData)
	mRow3 := createBlockRealMatrixFromSlices(t, subRow3)
	if mRow3.Equals(m.RowMatrixAt(0)) {
		t.Errorf("Mismatch. mat equal to row3 mat.")
	}

	m.SetRowMatrix(0, mRow3)
	if !mRow3.Equals(m.RowMatrixAt(0)) {
		t.Errorf("Mismatch. mat not equal to row3 mat.")
	}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	m.SetRowMatrix(-1, mRow3)
	m.SetRowMatrix(0, m)

}

func TestBlockRealMatrixSetRowMatrixLarge(t *testing.T) {
	n := 3 * BLOCK_SIZE
	m := createBlockRealMatrix(t, n, n)
	sub := createBlockRealMatrix(t, 1, n).ScalarAdd(1)

	m.SetRowMatrix(2, sub)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i != 2 {
				if m.At(i, j) != 0. {
					t.Errorf("Mismatch.")
				}

			} else {
				if m.At(i, j) != 1.0 {
					t.Errorf("Mismatch.")
				}
			}
		}
	}

	if !sub.Equals(m.RowMatrixAt(2)) {
		t.Errorf("Mismatch. sub not equal row mat at 2.")
	}
}

func TestBlockRealMatrixColumnMatrixAt(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, subTestData)
	mColumn1 := createBlockRealMatrixFromSlices(t, subColumn1)
	mColumn3 := createBlockRealMatrixFromSlices(t, subColumn3)
	if !mColumn1.Equals(m.ColumnMatrixAt(1)) {
		t.Errorf("Mismatch. mColumn1 not equal column mat at 1.")
	}

	if !mColumn3.Equals(m.ColumnMatrixAt(3)) {
		t.Errorf("Mismatch. mColumn3 not equal column mat at 3.")
	}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	m.ColumnMatrixAt(-1)
	m.ColumnMatrixAt(4)

}

func TestBlockRealMatrixSetColumnMatrix(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, subTestData)
	mColumn3 := createBlockRealMatrixFromSlices(t, subColumn3)
	if mColumn3.Equals(m.ColumnMatrixAt(1)) {
		t.Errorf("Mismatch. mColumn3  equal column mat at 1.")
	}

	m.SetColumnMatrix(1, mColumn3)

	if !mColumn3.Equals(m.ColumnMatrixAt(1)) {
		t.Errorf("Mismatch. mColumn3 not equal column mat at 1.")
	}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	m.SetColumnMatrix(-1, mColumn3)
	m.SetColumnMatrix(0, m)
}

func TestBlockRealMatrixSetColumnMatrixLarge(t *testing.T) {
	n := 3 * BLOCK_SIZE
	m := createBlockRealMatrix(t, n, n)
	sub := createBlockRealMatrix(t, n, 1).ScalarAdd(1)

	m.SetColumnMatrix(2, sub)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if j != 2 {
				if m.At(i, j) != 0. {
					t.Errorf("Mismatch. want: %v, got: %v", 0., m.At(i, j))
				}

			} else {
				if m.At(i, j) != 1.0 {
					t.Errorf("Mismatch. want: %v, got: %v", 1.0, m.At(i, j))
				}
			}
		}
	}

	if !sub.Equals(m.ColumnMatrixAt(2)) {
		t.Errorf("Mismatch. sub not equal column mat at 2.")
	}
}

func TestBlockRealMatrixRowVectorAt(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, subTestData)
	mRow0 := createArrayRealVectorFromSlice(t, subRow0[0])
	mRow3 := createArrayRealVectorFromSlice(t, subRow3[0])
	if !mRow0.Equals(m.RowVectorAt(0)) {
		t.Errorf("Mismatch. mRow0 not equal row vector at 0.")
	}
	if !mRow3.Equals(m.RowVectorAt(3)) {
		t.Errorf("Mismatch. mRow3 not equal row vector at 3.")
	}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	m.RowVectorAt(-1)
	m.RowVectorAt(4)
}

func TestBlockRealMatrixSetRowVector(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, subTestData)
	mRow3 := createArrayRealVectorFromSlice(t, subRow3[0])
	if mRow3.Equals(m.RowMatrixAt(0)) {
		t.Errorf("Mismatch. mRow3  equal row vector at 0.")
	}

	m.SetRowVector(0, mRow3)
	if !mRow3.Equals(m.RowVectorAt(0)) {
		t.Errorf("Mismatch. mRow3  equal row vector at 0.")
	}
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	m.SetRowVector(-1, mRow3)
	m.SetRowVector(0, createSizedArrayRealVector(t, 5))

}

func TestBlockRealMatrixSetRowVectorLarge(t *testing.T) {
	n := 3 * BLOCK_SIZE
	m := createBlockRealMatrix(t, n, n)
	sub := createSizedArrayRealVectorWithPreset(t, n, 1.0)

	m.SetRowVector(2, sub)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i != 2 {
				if m.At(i, j) != 0. {
					t.Errorf("Mismatch. want: %v, got: %v", 0., m.At(i, j))
				}

			} else {
				if m.At(i, j) != 1.0 {
					t.Errorf("Mismatch. want: %v, got: %v", 1.0, m.At(i, j))
				}

			}
		}
	}
	if !sub.Equals(m.RowVectorAt(2)) {
		t.Errorf("Mismatch. sub not equal row vector at 2.")
	}

}

func TestBlockRealMatrixColumnVectorAt(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, subTestData)
	mColumn1 := columnToVector(subColumn1)
	mColumn3 := columnToVector(subColumn3)
	if !mColumn1.Equals(m.ColumnVectorAt(1)) {
		t.Errorf("Mismatch. mColumn1 not equal column vector at 1.")
	}

	if !mColumn3.Equals(m.ColumnVectorAt(3)) {
		t.Errorf("Mismatch. mColumn1 not equal column vector at 3.")
	}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	m.ColumnVectorAt(-1)
	m.ColumnVectorAt(4)

}

func TestBlockRealMatrixSetColumnVector(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, subTestData)
	mColumn3 := columnToVector(subColumn3)

	if mColumn3.Equals(m.ColumnVectorAt(1)) {
		t.Errorf("Mismatch. mColumn3 equal column vector at 1.")
	}

	m.SetColumnVector(1, mColumn3)

	if !mColumn3.Equals(m.ColumnVectorAt(1)) {
		t.Errorf("Mismatch. mColumn3 not equal column vector at 1.")
	}
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	m.SetColumnVector(-1, mColumn3)
	m.SetColumnVector(0, createSizedArrayRealVector(t, 5))

}

func TestBlockRealMatrixSetColumnVectorLarge(t *testing.T) {
	n := 3 * BLOCK_SIZE
	m := createBlockRealMatrix(t, n, n)
	sub := createSizedArrayRealVectorWithPreset(t, n, 1.0)

	m.SetColumnVector(2, sub)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if j != 2 {
				if m.At(i, j) != 0. {
					t.Errorf("Mismatch. want: %v, got: %v", 0., m.At(i, j))
				}

			} else {
				if m.At(i, j) != 1.0 {
					t.Errorf("Mismatch. want: %v, got: %v", 1.0, m.At(i, j))
				}
			}
		}
	}
	if !sub.Equals(m.ColumnVectorAt(2)) {
		t.Errorf("Mismatch. sub not equal column vector at 2.")
	}

}

func TestBlockRealMatrixRowAt(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, subTestData)
	checkArrays(t, subRow0[0], m.RowAt(0))
	checkArrays(t, subRow3[0], m.RowAt(3))
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	m.RowAt(-1)
	m.RowAt(4)

}

func TestBlockRealMatrixSetRow(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, subTestData)
	if subRow3[0][0] == m.RowAt(0)[0] {
		t.Errorf("Mismatch. subRow3[0][0] equal mat row at 0,0.")
	}

	m.SetRow(0, subRow3[0])
	checkArrays(t, subRow3[0], m.RowAt(0))
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	m.SetRow(-1, subRow3[0])
	m.SetRow(0, make([]float64, 5))

}

func TestBlockRealMatrixSetRowLarge(t *testing.T) {
	n := 3 * BLOCK_SIZE
	m := createBlockRealMatrix(t, n, n)
	sub := make([]float64, n)
	for i := 0; i < n; i++ {
		sub[i] = 1.
	}

	m.SetRow(2, sub)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i != 2 {
				if m.At(i, j) != 0. {
					t.Errorf("Mismatch. want: %v, got: %v", 0., m.At(i, j))
				}
			} else {
				if m.At(i, j) != 1.0 {
					t.Errorf("Mismatch. want: %v, got: %v", 1.0, m.At(i, j))
				}
			}
		}
	}
	checkArrays(t, sub, m.RowAt(2))
}

func TestBlockRealMatrixColumnAt(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, subTestData)
	mColumn1 := columnToArray(subColumn1)
	mColumn3 := columnToArray(subColumn3)
	checkArrays(t, mColumn1, m.ColumnAt(1))
	checkArrays(t, mColumn3, m.ColumnAt(3))
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	m.ColumnAt(-1)
	m.ColumnAt(4)

}

func TestBlockRealMatrixSetColumn(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, subTestData)
	mColumn3 := columnToArray(subColumn3)

	if m.ColumnAt(1)[0] == mColumn3[0] {
		t.Errorf("Mismatch. want: %v, got: %v", mColumn3[0], m.ColumnAt(1)[0])
	}
	m.SetColumn(1, mColumn3)
	checkArrays(t, mColumn3, m.ColumnAt(1))
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	m.SetColumn(-1, mColumn3)
	m.SetColumn(0, make([]float64, 5))

}

func TestBlockRealMatrixGetSetColumnLarge(t *testing.T) {
	n := 3 * BLOCK_SIZE
	m := createBlockRealMatrix(t, n, n)
	sub := make([]float64, n)
	for i := 0; i < n; i++ {
		sub[i] = 1.
	}

	m.SetColumn(2, sub)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {

			if j != 2 {
				if m.At(i, j) != 0. {
					t.Errorf("Mismatch. want: %v, got: %v", 0., m.At(i, j))
				}
			} else {
				if m.At(i, j) != 1.0 {
					t.Errorf("Mismatch. want: %v, got: %v", 1.0, m.At(i, j))
				}
			}
		}
	}
	checkArrays(t, sub, m.ColumnAt(2))
}

func TestBlockRealMatrixEquals(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, testData)
	m1 := m.Copy().(*BlockRealMatrix)
	mt := m.Transpose().(*BlockRealMatrix)
	if !m.Equals(m) {
		t.Errorf("Mismatch. matrix m should be equal to itself")
	}
	if !m.Equals(m1) {
		t.Errorf("Mismatch. matrix m should be equal to deep copy m1")
	}

	if m.Equals(nil) {
		t.Errorf("Mismatch. matrix m should not be equal to nil")
	}

	if m.Equals(mt) {
		t.Errorf("Mismatch. matrix m should not be equal to transposed matrix of itself")
	}

	if m.Equals(createBlockRealMatrixFromSlices(t, bigSingular)) {
		t.Errorf("Mismatch. matrix m should not be equal to anything other than copy of itself")
	}
}

func TestBlockRealMatrixSetSubMatrix(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, testData)
	m.SetSubMatrix(detData2, 1, 1)
	expected := createBlockRealMatrixFromSlices(t, [][]float64{{1.0, 2.0, 3.0}, {2.0, 1.0, 3.0}, {1.0, 2.0, 4.0}})
	if !m.Equals(expected) {
		t.Errorf("Mismatch. two mats should be the same. want: true, got: false")
	}

	m.SetSubMatrix(detData2, 0, 0)
	expected = createBlockRealMatrixFromSlices(t, [][]float64{{1.0, 3.0, 3.0}, {2.0, 4.0, 3.0}, {1.0, 2.0, 4.0}})
	if !m.Equals(expected) {
		t.Errorf("Mismatch. two mats should be the same. want: true, got: false")
	}

	m.SetSubMatrix(testDataPlus2, 0, 0)
	expected = createBlockRealMatrixFromSlices(t, [][]float64{{3.0, 4.0, 5.0}, {4.0, 7.0, 5.0}, {3.0, 2.0, 10.0}})
	if !m.Equals(expected) {
		t.Errorf("Mismatch. two mats should be the same. want: true, got: false")
	}

	m2 := createBlockRealMatrixFromSlices(t, [][]float64{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 0, 1, 2}})
	m2.SetSubMatrix([][]float64{{3, 4}, {5, 6}}, 1, 1)
	expected = createBlockRealMatrixFromSlices(t, [][]float64{{1, 2, 3, 4}, {5, 3, 4, 8}, {9, 5, 6, 2}})
	if !m2.Equals(expected) {
		t.Errorf("Mismatch. two mats should be the same. want: true, got: false")
	}
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	m.SetSubMatrix(testData, 1, 1)
	m.SetSubMatrix(testData, -1, 1)
	m.SetSubMatrix(testData, 1, -1)
	m.SetSubMatrix(nil, 1, 1)
	m.SetSubMatrix([][]float64{{1}, {2, 3}}, 0, 0)
	m.SetSubMatrix([][]float64{{}}, 0, 0)

}

func TestBlockRealMatrixCopySubMatrix(t *testing.T) {
	m := createBlockRealMatrixFromSlices(t, subTestData)
	checkBlockRealMatrixCopy(t, m, subRows23Cols00, 2, 3, 0, 0)
	checkBlockRealMatrixCopy(t, m, subRows00Cols33, 0, 0, 3, 3)
	checkBlockRealMatrixCopy(t, m, subRows01Cols23, 0, 1, 2, 3)
	checkBlockRealMatrixCopyFromIndices(t, m, subRows02Cols13, []int{0, 2}, []int{1, 3})
	checkBlockRealMatrixCopyFromIndices(t, m, subRows03Cols12, []int{0, 3}, []int{1, 2})
	checkBlockRealMatrixCopyFromIndices(t, m, subRows03Cols123, []int{0, 3}, []int{1, 2, 3})
	checkBlockRealMatrixCopyFromIndices(t, m, subRows20Cols123, []int{2, 0}, []int{1, 2, 3})
	checkBlockRealMatrixCopyFromIndices(t, m, subRows31Cols31, []int{3, 1}, []int{3, 1})
	checkBlockRealMatrixCopyFromIndices(t, m, subRows31Cols31, []int{3, 1}, []int{3, 1})

	checkBlockRealMatrixCopy(t, m, nil, 1, 0, 2, 4)
	checkBlockRealMatrixCopy(t, m, nil, -1, 1, 2, 2)
	checkBlockRealMatrixCopy(t, m, nil, 1, 0, 2, 2)
	checkBlockRealMatrixCopy(t, m, nil, 1, 0, 2, 4)
	checkBlockRealMatrixCopyFromIndices(t, m, nil, []int{}, []int{0})
	checkBlockRealMatrixCopyFromIndices(t, m, nil, []int{0}, []int{4})
}

func checkBlockRealMatrixCopy(t *testing.T, m *BlockRealMatrix, reference [][]float64, startRow, endRow, startColumn, endColumn int) {

	var sub [][]float64
	if reference == nil {
		sub = make([][]float64, 1)
		for i := 0; i < 1; i++ {
			sub[i] = make([]float64, 1)
		}
	} else {
		sub = createIdenticalCopy(reference)
	}

	if reference == nil {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("panic expected.")
			}
		}()
	}

	CopySubMatrix(m, startRow, endRow, startColumn, endColumn, sub)
	if !createArray2DRowRealMatrixFromSlices(t, reference).Equals(createArray2DRowRealMatrixFromSlices(t, sub)) {
		t.Errorf("Mismatch. Sub must equal reference ")
	}

}

func checkBlockRealMatrixCopyFromIndices(t *testing.T, m *BlockRealMatrix, reference [][]float64, selectedRows, selectedColumns []int) {

	var sub [][]float64
	if reference == nil {
		sub = make([][]float64, 1)
		for i := 0; i < 1; i++ {
			sub[i] = make([]float64, 1)
		}
	} else {
		sub = createIdenticalCopy(reference)
	}

	if reference == nil {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("panic expected.")
			}
		}()
	}

	CopySubMatrixFromIndices(m, selectedRows, selectedColumns, sub)
	if !createArray2DRowRealMatrixFromSlices(t, reference).Equals(createArray2DRowRealMatrixFromSlices(t, sub)) {
		t.Errorf("Mismatch. Sub must equal reference ")
	}

}

func TestBlockRealMatrixWalk(t *testing.T) {
	rows := 150
	columns := 75

	m := createBlockRealMatrix(t, rows, columns)
	m.WalkInUpdateRowOrder(new(SetVisitor))
	getVisitor := new(GetVisitor)
	getVisitor.t = t
	m.WalkInOptimizedOrder(getVisitor)

	if rows*columns != getVisitor.count {
		t.Errorf("Mismatch. visitor counter doesn't match rows and column.")
	}

	m = createBlockRealMatrix(t, rows, columns)
	m.WalkInUpdateRowOrderBounded(new(SetVisitor), 1, rows-2, 1, columns-2)
	getVisitor = new(GetVisitor)
	getVisitor.t = t
	m.WalkInOptimizedOrderBounded(getVisitor, 1, rows-2, 1, columns-2)
	if (rows-2)*(columns-2) != getVisitor.count {
		t.Errorf("Mismatch. visitor counter doesn't match rows and column.")
	}

	for i := 0; i < rows; i++ {
		if 0.0 != m.At(i, 0) {
			t.Errorf("Mismatch. want: %v, got: %v.", 0, m.At(i, 0))
		}
		if 0.0 != m.At(i, columns-1) {
			t.Errorf("Mismatch. want: %v, got: %v.", 0, m.At(i, columns-1))
		}
	}

	for j := 0; j < columns; j++ {
		if 0.0 != m.At(0, j) {
			t.Errorf("Mismatch. want: %v, got: %v.", 0, m.At(0, j))
		}
		if 0.0 != m.At(rows-1, j) {
			t.Errorf("Mismatch. want: %v, got: %v.", 0, m.At(rows-1, j))
		}
	}

	m = createBlockRealMatrix(t, rows, columns)
	m.WalkInUpdateColumnOrder(new(SetVisitor))
	getVisitor = new(GetVisitor)
	getVisitor.t = t
	m.WalkInOptimizedOrder(getVisitor)
	if rows*columns != getVisitor.count {
		t.Errorf("Mismatch. visitor counter doesn't match rows and column.")
	}

	m = createBlockRealMatrix(t, rows, columns)
	m.WalkInUpdateColumnOrderBounded(new(SetVisitor), 1, rows-2, 1, columns-2)
	getVisitor = new(GetVisitor)
	getVisitor.t = t
	m.WalkInOptimizedOrderBounded(getVisitor, 1, rows-2, 1, columns-2)
	if (rows-2)*(columns-2) != getVisitor.count {
		t.Errorf("Mismatch. visitor counter doesn't match rows and column.")
	}

	for i := 0; i < rows; i++ {
		if 0.0 != m.At(i, 0) {
			t.Errorf("Mismatch. want: %v, got: %v.", 0, m.At(i, 0))
		}
		if 0.0 != m.At(i, columns-1) {
			t.Errorf("Mismatch. want: %v, got: %v.", 0, m.At(i, columns-1))
		}
	}
	for j := 0; j < columns; j++ {
		if 0.0 != m.At(0, j) {
			t.Errorf("Mismatch. want: %v, got: %v.", 0, m.At(0, j))
		}
		if 0.0 != m.At(rows-1, j) {
			t.Errorf("Mismatch. want: %v, got: %v.", 0, m.At(rows-1, j))
		}
	}

	m = createBlockRealMatrix(t, rows, columns)
	m.WalkInUpdateOptimizedOrder(new(SetVisitor))
	getVisitor = new(GetVisitor)
	getVisitor.t = t
	m.WalkInRowOrder(getVisitor)
	if rows*columns != getVisitor.count {
		t.Errorf("Mismatch. visitor counter doesn't match rows and column.")
	}

	m = createBlockRealMatrix(t, rows, columns)
	m.WalkInUpdateOptimizedOrderBounded(new(SetVisitor), 1, rows-2, 1, columns-2)
	getVisitor = new(GetVisitor)
	getVisitor.t = t
	m.WalkInRowOrderBounded(getVisitor, 1, rows-2, 1, columns-2)
	if (rows-2)*(columns-2) != getVisitor.count {
		t.Errorf("Mismatch. visitor counter doesn't match rows and column.")
	}
	for i := 0; i < rows; i++ {
		if 0.0 != m.At(i, 0) {
			t.Errorf("Mismatch. want: %v, got: %v.", 0, m.At(i, 0))
		}
		if 0.0 != m.At(i, columns-1) {
			t.Errorf("Mismatch. want: %v, got: %v.", 0, m.At(i, columns-1))
		}
	}
	for j := 0; j < columns; j++ {
		if 0.0 != m.At(0, j) {
			t.Errorf("Mismatch. want: %v, got: %v.", 0, m.At(0, j))
		}
		if 0.0 != m.At(rows-1, j) {
			t.Errorf("Mismatch. want: %v, got: %v.", 0, m.At(rows-1, j))
		}
	}

	m = createBlockRealMatrix(t, rows, columns)
	m.WalkInUpdateOptimizedOrder(new(SetVisitor))
	getVisitor = new(GetVisitor)
	getVisitor.t = t
	m.WalkInColumnOrder(getVisitor)
	if rows*columns != getVisitor.count {
		t.Errorf("Mismatch. visitor counter doesn't match rows and column.")
	}

	m = createBlockRealMatrix(t, rows, columns)
	m.WalkInUpdateOptimizedOrderBounded(new(SetVisitor), 1, rows-2, 1, columns-2)
	getVisitor = new(GetVisitor)
	getVisitor.t = t
	m.WalkInColumnOrderBounded(getVisitor, 1, rows-2, 1, columns-2)
	if (rows-2)*(columns-2) != getVisitor.count {
		t.Errorf("Mismatch. visitor counter doesn't match rows and column.")
	}
	for i := 0; i < rows; i++ {
		if 0.0 != m.At(i, 0) {
			t.Errorf("Mismatch. want: %v, got: %v.", 0, m.At(i, 0))
		}
		if 0.0 != m.At(i, columns-1) {
			t.Errorf("Mismatch. want: %v, got: %v.", 0, m.At(i, columns-1))
		}
	}
	for j := 0; j < columns; j++ {
		if 0.0 != m.At(0, j) {
			t.Errorf("Mismatch. want: %v, got: %v.", 0, m.At(0, j))
		}
		if 0.0 != m.At(rows-1, j) {
			t.Errorf("Mismatch. want: %v, got: %v.", 0, m.At(rows-1, j))
		}
	}

}
