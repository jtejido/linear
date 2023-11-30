package linear

import (
	"math"
	"testing"
)

func createArray2DRowRealMatrixFromSlices(t *testing.T, data [][]float64) *Array2DRowRealMatrix {
	m, err := NewArray2DRowRealMatrixFromSlices(data, true)

	if err != nil {
		t.Errorf("Error while creating Matrix %s", err)
	}

	return m
}

func createArray2DRowRealMatrix(t *testing.T, rowDimension, columnDimension int) *Array2DRowRealMatrix {
	m, err := NewArray2DRowRealMatrix(rowDimension, columnDimension)

	if err != nil {
		t.Errorf("Error while creating Matrix %s", err)
	}

	return m
}

func TestArray2DRowRealMatrixDimensions(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, testData)
	m2 := createArray2DRowRealMatrixFromSlices(t, testData2)

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

func TestArray2DRowRealMatrixCopyFunctions(t *testing.T) {
	m1 := createArray2DRowRealMatrixFromSlices(t, testData)
	m2 := createArray2DRowRealMatrixFromSlices(t, m1.Data())

	if !m1.Equals(m2) {
		t.Errorf("Mismatch. m1 not equal m2.")
	}

	m3 := createArray2DRowRealMatrixFromSlices(t, testData)
	m4 := createArray2DRowRealMatrixFromSlices(t, m3.Data())

	if !m3.Equals(m4) {
		t.Errorf("Mismatch. m3 not equal m4.")
	}
}

func TestArray2DRowRealMatrixAdd(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, testData)
	mInv := createArray2DRowRealMatrixFromSlices(t, testDataInv)
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

func TestArray2DRowRealMatrixAddFail(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, testData)
	m2 := createArray2DRowRealMatrixFromSlices(t, testData2)

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	m.Add(m2)
}

func TestArray2DRowRealMatrixMatLInfNorm(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, testData)
	m2 := createArray2DRowRealMatrixFromSlices(t, testData2)
	if math.Abs(MatLInfNorm(m)-14) > entryTolerance {
		t.Errorf("Mismatch. testData norm, want: %v, got: %v", 14, MatLInfNorm(m))
	}

	if math.Abs(MatLInfNorm(m2)-7) > entryTolerance {
		t.Errorf("Mismatch. testData2 norm, want: %v, got: %v", 7, MatLInfNorm(m2))
	}
}

func TestArray2DRowRealMatrixMatFrobeniusNorm(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, testData)
	mExp := math.Sqrt(117.0)
	m2 := createArray2DRowRealMatrixFromSlices(t, testData2)
	m2Exp := math.Sqrt(52.0)
	if math.Abs(MatFrobeniusNorm(m)-mExp) > entryTolerance {
		t.Errorf("Mismatch. testData norm, want: %v, got: %v", mExp, MatFrobeniusNorm(m))
	}

	if math.Abs(MatFrobeniusNorm(m2)-m2Exp) > entryTolerance {
		t.Errorf("Mismatch. testData2 norm, want: %v, got: %v", m2Exp, MatFrobeniusNorm(m2))
	}
}

func TestArray2DRowRealMatrixPlusMinus(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, testData)
	m2 := createArray2DRowRealMatrixFromSlices(t, testDataInv)
	assertCloseMatrix(t, m.Subtract(m2), m2.ScalarMultiply(-1).Add(m), entryTolerance)

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	m.Subtract(createArray2DRowRealMatrixFromSlices(t, testData2))

}

func TestArray2DRowRealMatrixMultiply(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, testData)
	mInv := createArray2DRowRealMatrixFromSlices(t, testDataInv)
	identity := createArray2DRowRealMatrixFromSlices(t, id)
	m2 := createArray2DRowRealMatrixFromSlices(t, testData2)
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

	m.Multiply(createArray2DRowRealMatrixFromSlices(t, bigSingular))

}

func TestArray2DRowRealMatrixMultiply2(t *testing.T) {
	m3 := createArray2DRowRealMatrixFromSlices(t, d3)
	m4 := createArray2DRowRealMatrixFromSlices(t, d4)
	m5 := createArray2DRowRealMatrixFromSlices(t, d5)
	assertCloseMatrix(t, m3.Multiply(m4), m5, entryTolerance)
}

func TestArray2DRowRealMatrixPower(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, testData)
	mInv := createArray2DRowRealMatrixFromSlices(t, testDataInv)
	mPlusInv := createArray2DRowRealMatrixFromSlices(t, testDataPlusInv)
	identity := createArray2DRowRealMatrixFromSlices(t, id)

	if !Power(m, 0).Equals(identity) {
		t.Errorf("Mismatch. m^0 not equal identity ")
	}

	if !Power(mInv, 0).Equals(identity) {
		t.Errorf("Mismatch. mInv^0 not equal identity ")
	}

	if !Power(mPlusInv, 0).Equals(identity) {
		t.Errorf("Mismatch. mPlusInv^0 not equal identity ")
	}

	if !Power(m, 1).Equals(m) {
		t.Errorf("Mismatch. m^1 not equal m ")
	}

	if !Power(mInv, 1).Equals(mInv) {
		t.Errorf("Mismatch.mInv^1 not equal mInv ")
	}

	if !Power(mPlusInv, 1).Equals(mPlusInv) {
		t.Errorf("Mismatch.mPlusInv^1 not equal mInv ")
	}

	C1 := m.Copy()
	C2 := mInv.Copy()
	C3 := mPlusInv.Copy()

	for i := 2; i <= 10; i++ {
		C1 = C1.Multiply(m)
		C2 = C2.Multiply(mInv)
		C3 = C3.Multiply(mPlusInv)
		if !Power(m, i).Equals(C1) {
			t.Errorf("Mismatch. m^%d not equal C1 ", i)
		}

		if !Power(mInv, i).Equals(C2) {
			t.Errorf("Mismatch. mInv^%d not equal C2 ", i)
		}

		if !Power(mPlusInv, i).Equals(C3) {
			t.Errorf("Mismatch. mPlusInv^%d not equal C3 ", i)
		}
	}

	mNotSquare := createArray2DRowRealMatrixFromSlices(t, testData2T)

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	Power(mNotSquare, 2)
	Power(m, -1)

}

func TestArray2DRowRealMatrixTrace(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, id)
	if math.Abs(m.Trace()-3) > entryTolerance {
		t.Errorf("Mismatch. identity trace. want: %v, got: %v", 3, m.Trace())
	}

	m = createArray2DRowRealMatrixFromSlices(t, testData2)

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	m.Trace()
}

func TestArray2DRowRealMatrixScalarAdd(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, testData)
	mat := createArray2DRowRealMatrixFromSlices(t, testDataPlus2)
	assertCloseMatrix(t, mat, m.ScalarAdd(2), entryTolerance)
}

func TestArray2DRowRealMatrixOperateVector(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, id)
	assertClose(t, testVector, m.Operate(testVector), entryTolerance)
	assertClose(t, testVector, m.OperateVector(createArrayRealVectorFromSlice(t, testVector)).ToArray(), entryTolerance)
	m = createArray2DRowRealMatrixFromSlices(t, bigSingular)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	m.Operate(testVector)
}

func TestArray2DRowRealMatrixTranspose(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, testData)
	mI, _ := NewLUDecomposition(m)
	mT, _ := NewLUDecomposition(m.Transpose())
	mIT := mI.Solver().Inverse().Transpose()
	mTI := mT.Solver().Inverse()

	assertCloseMatrix(t, mIT, mTI, normTolerance)

	m = createArray2DRowRealMatrixFromSlices(t, testData2)
	mt := createArray2DRowRealMatrixFromSlices(t, testData2T)
	assertCloseMatrix(t, mt, m.Transpose(), normTolerance)
}

func TestArray2DRowRealMatrixPremultiplyVector(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, testData)
	assertClose(t, m.PreMultiply(testVector), preMultTest, normTolerance)
	assertClose(t, m.PreMultiply(createArrayRealVectorFromSlice(t, testVector).ToArray()), preMultTest, normTolerance)
	m = createArray2DRowRealMatrixFromSlices(t, bigSingular)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	m.PreMultiply(testVector)
}

func TestArray2DRowRealMatrixPremultiplyMatrix(t *testing.T) {
	m3 := createArray2DRowRealMatrixFromSlices(t, d3)
	m4 := createArray2DRowRealMatrixFromSlices(t, d4)
	m5 := createArray2DRowRealMatrixFromSlices(t, d5)
	assertCloseMatrix(t, m4.PreMultiplyMatrix(m3), m5, entryTolerance)

	m := createArray2DRowRealMatrixFromSlices(t, testData)
	mInv := createArray2DRowRealMatrixFromSlices(t, testDataInv)
	identity := createArray2DRowRealMatrixFromSlices(t, id)
	assertCloseMatrix(t, m.PreMultiplyMatrix(mInv), identity, entryTolerance)
	assertCloseMatrix(t, mInv.PreMultiplyMatrix(m), identity, entryTolerance)
	assertCloseMatrix(t, m.PreMultiplyMatrix(identity), m, entryTolerance)
	assertCloseMatrix(t, identity.PreMultiplyMatrix(mInv), mInv, entryTolerance)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	mat := createArray2DRowRealMatrixFromSlices(t, bigSingular)
	m.PreMultiplyMatrix(mat)
}

func TestArray2DRowRealMatrixVectorAt(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, testData)
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

func TestArray2DRowRealMatrixAt(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, testData)
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

func TestArray2DRowRealMatrixSubMatrix(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, subTestData)
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

func TestArray2DRowRealMatrixCopySubMatrix(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, subTestData)
	checkCopy(t, m, subRows23Cols00, 2, 3, 0, 0, false)
	checkCopy(t, m, subRows00Cols33, 0, 0, 3, 3, false)
	checkCopy(t, m, subRows01Cols23, 0, 1, 2, 3, false)
	checkCopyFromIndices(t, m, subRows02Cols13, []int{0, 2}, []int{1, 3}, false)
	checkCopyFromIndices(t, m, subRows03Cols12, []int{0, 3}, []int{1, 2}, false)
	checkCopyFromIndices(t, m, subRows03Cols123, []int{0, 3}, []int{1, 2, 3}, false)
	checkCopyFromIndices(t, m, subRows20Cols123, []int{2, 0}, []int{1, 2, 3}, false)
	checkCopyFromIndices(t, m, subRows31Cols31, []int{3, 1}, []int{3, 1}, false)
	checkCopyFromIndices(t, m, subRows31Cols31, []int{3, 1}, []int{3, 1}, false)

	checkCopy(t, m, nil, 1, 0, 2, 4, true)
	checkCopy(t, m, nil, -1, 1, 2, 2, true)
	checkCopy(t, m, nil, 1, 0, 2, 2, true)
	checkCopy(t, m, nil, 1, 0, 2, 4, true)
	checkCopyFromIndices(t, m, nil, []int{}, []int{0}, true)
	checkCopyFromIndices(t, m, nil, []int{0}, []int{4}, true)

	// rectangular check
	copy := [][]float64{{0, 0, 0}, {0, 0}}
	checkCopy(t, m, copy, 0, 1, 0, 2, true)
	checkCopyFromIndices(t, m, copy, []int{0, 1}, []int{0, 1, 2}, true)
}

func checkCopy(t *testing.T, m RealMatrix, reference [][]float64, startRow, endRow, startColumn, endColumn int, mustFail bool) {

	var sub [][]float64
	if reference == nil {
		sub = make([][]float64, 1)
		for i := 0; i < 1; i++ {
			sub[i] = make([]float64, 1)
		}
	} else {
		sub = createIdenticalCopy(reference)
	}

	if mustFail {
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

func checkCopyFromIndices(t *testing.T, m RealMatrix, reference [][]float64, selectedRows, selectedColumns []int, mustFail bool) {

	var sub [][]float64
	if reference == nil {
		sub = make([][]float64, 1)
		for i := 0; i < 1; i++ {
			sub[i] = make([]float64, 1)
		}
	} else {
		sub = createIdenticalCopy(reference)
	}
	if mustFail {
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

func createIdenticalCopy(matrix [][]float64) [][]float64 {
	matrixCopy := make([][]float64, len(matrix))
	for i := 0; i < len(matrixCopy); i++ {
		matrixCopy[i] = make([]float64, len(matrix[i]))
	}

	return matrixCopy
}

func TestArray2DRowRealMatrixRowMatrixAt(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, subTestData)
	mRow0 := createArray2DRowRealMatrixFromSlices(t, subRow0)
	mRow3 := createArray2DRowRealMatrixFromSlices(t, subRow3)
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
	m.RowMatrixAt(4)

}

func TestArray2DRowRealMatrixSetRowMatrix(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, subTestData)
	mRow3 := createArray2DRowRealMatrixFromSlices(t, subRow3)
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

func TestArray2DRowRealMatrixColumnMatrixAt(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, subTestData)
	mColumn1 := createArray2DRowRealMatrixFromSlices(t, subColumn1)
	mColumn3 := createArray2DRowRealMatrixFromSlices(t, subColumn3)
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

func TestArray2DRowRealMatrixSetColumnMatrix(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, subTestData)
	mColumn3 := createArray2DRowRealMatrixFromSlices(t, subColumn3)
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

func TestArray2DRowRealMatrixRowVectorAt(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, subTestData)
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

func TestArray2DRowRealMatrixSetRowVector(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, subTestData)
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

func TestArray2DRowRealMatrixColumnVectorAt(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, subTestData)
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

func TestArray2DRowRealMatrixSetColumnVector(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, subTestData)
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

func TestArray2DRowRealMatrixRowAt(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, subTestData)
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

func TestArray2DRowRealMatrixSetRow(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, subTestData)
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

func TestArray2DRowRealMatrixColumnAt(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, subTestData)
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

func TestArray2DRowRealMatrixSetColumn(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, subTestData)
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

func TestArray2DRowRealMatrixEquals(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, testData)
	m1 := m.Copy().(*Array2DRowRealMatrix)
	mt := m.Transpose().(*Array2DRowRealMatrix)
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

	if m.Equals(createArray2DRowRealMatrixFromSlices(t, bigSingular)) {
		t.Errorf("Mismatch. matrix m should not be equal to anything other than copy of itself")
	}
}

func TestArray2DRowRealMatrixSetSubMatrix(t *testing.T) {
	m := createArray2DRowRealMatrixFromSlices(t, testData)
	m.SetSubMatrix(detData2, 1, 1)
	expected := createArray2DRowRealMatrixFromSlices(t, [][]float64{{1.0, 2.0, 3.0}, {2.0, 1.0, 3.0}, {1.0, 2.0, 4.0}})
	if !m.Equals(expected) {
		t.Errorf("Mismatch. two mats should be the same. want: true, got: false")
	}

	m.SetSubMatrix(detData2, 0, 0)
	expected = createArray2DRowRealMatrixFromSlices(t, [][]float64{{1.0, 3.0, 3.0}, {2.0, 4.0, 3.0}, {1.0, 2.0, 4.0}})
	if !m.Equals(expected) {
		t.Errorf("Mismatch. two mats should be the same. want: true, got: false")
	}

	m.SetSubMatrix(testDataPlus2, 0, 0)
	expected = createArray2DRowRealMatrixFromSlices(t, [][]float64{{3.0, 4.0, 5.0}, {4.0, 7.0, 5.0}, {3.0, 2.0, 10.0}})
	if !m.Equals(expected) {
		t.Errorf("Mismatch. two mats should be the same. want: true, got: false")
	}

	m2 := createArray2DRowRealMatrixFromSlices(t, [][]float64{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 0, 1, 2}})
	m2.SetSubMatrix([][]float64{{3, 4}, {5, 6}}, 1, 1)
	expected = createArray2DRowRealMatrixFromSlices(t, [][]float64{{1, 2, 3, 4}, {5, 3, 4, 8}, {9, 5, 6, 2}})
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

func TestArray2DRowRealMatrixWalk(t *testing.T) {
	rows := 150
	columns := 75

	m := createArray2DRowRealMatrix(t, rows, columns)
	m.WalkInUpdateRowOrder(new(SetVisitor))
	getVisitor := new(GetVisitor)
	getVisitor.t = t
	m.WalkInOptimizedOrder(getVisitor)

	if rows*columns != getVisitor.count {
		t.Errorf("Mismatch. visitor counter doesn't match rows and column.")
	}

	m = createArray2DRowRealMatrix(t, rows, columns)
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

	m = createArray2DRowRealMatrix(t, rows, columns)
	m.WalkInUpdateColumnOrder(new(SetVisitor))
	getVisitor = new(GetVisitor)
	getVisitor.t = t
	m.WalkInOptimizedOrder(getVisitor)
	if rows*columns != getVisitor.count {
		t.Errorf("Mismatch. visitor counter doesn't match rows and column.")
	}

	m = createArray2DRowRealMatrix(t, rows, columns)
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

	m = createArray2DRowRealMatrix(t, rows, columns)
	m.WalkInUpdateOptimizedOrder(new(SetVisitor))
	getVisitor = new(GetVisitor)
	getVisitor.t = t
	m.WalkInRowOrder(getVisitor)
	if rows*columns != getVisitor.count {
		t.Errorf("Mismatch. visitor counter doesn't match rows and column.")
	}

	m = createArray2DRowRealMatrix(t, rows, columns)
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

	m = createArray2DRowRealMatrix(t, rows, columns)
	m.WalkInUpdateOptimizedOrder(new(SetVisitor))
	getVisitor = new(GetVisitor)
	getVisitor.t = t
	m.WalkInColumnOrder(getVisitor)
	if rows*columns != getVisitor.count {
		t.Errorf("Mismatch. visitor counter doesn't match rows and column.")
	}

	m = createArray2DRowRealMatrix(t, rows, columns)
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
