/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package linear

// Implementation of RealMatrix using a [][]float64 to store entries.
type Array2DRowRealMatrix struct {
	data [][]float64
}

func NewArray2DRowRealMatrix(rowDimension, columnDimension int) (*Array2DRowRealMatrix, error) {
	if rowDimension < 1 {
		return nil, notStrictlyPositiveErrorf(float64(rowDimension))
	}
	if columnDimension < 1 {
		return nil, notStrictlyPositiveErrorf(float64(columnDimension))
	}

	data := make([][]float64, rowDimension)
	for i := 0; i < len(data); i++ {
		data[i] = make([]float64, columnDimension)
	}

	return &Array2DRowRealMatrix{data: data}, nil
}

func NewArray2DRowRealMatrixFromSlices(data [][]float64, copyArray bool) (*Array2DRowRealMatrix, error) {
	ans := new(Array2DRowRealMatrix)
	if copyArray {
		ans.copyIn(data)
	} else {
		if data == nil {
			return nil, invalidArgumentSimpleErrorf()
		}
		nRows := len(data)
		if nRows == 0 {
			return nil, noDataErrorf(at_least_one_row)
		}
		nCols := len(data[0])
		if nCols == 0 {
			return nil, noDataErrorf(at_least_one_column)
		}
		for r := 1; r < nRows; r++ {
			if len(data[r]) != nCols {
				return nil, dimensionsMismatchSimpleErrorf(len(data[r]), nCols)
			}
		}
		ans.data = data
	}

	return ans, nil
}

func NewArray2DRowRealMatrixFromSlice(v []float64) (*Array2DRowRealMatrix, error) {
	if v == nil {
		return nil, invalidArgumentSimpleErrorf()
	}

	ans := new(Array2DRowRealMatrix)
	nRows := len(v)
	ans.data = make([][]float64, nRows)

	for i := 0; i < nRows; i++ {
		ans.data[i] = make([]float64, 1)
	}

	for row := 0; row < nRows; row++ {
		ans.data[row][0] = v[row]
	}

	return ans, nil
}

func (a2drrm *Array2DRowRealMatrix) Copy() RealMatrix {
	ans := new(Array2DRowRealMatrix)
	ans.copyIn(a2drrm.copyOut())
	return ans
}

func (a2drrm *Array2DRowRealMatrix) Add(mat RealMatrix) RealMatrix {
	if err := checkAdditionCompatible(a2drrm, mat); err != nil {
		panic(err)
	}

	var outData [][]float64

	if m, ok := mat.(*Array2DRowRealMatrix); ok {
		rowCount := a2drrm.RowDimension()
		columnCount := a2drrm.ColumnDimension()
		outData = make([][]float64, rowCount)
		for row := 0; row < rowCount; row++ {
			dataRow := a2drrm.data[row]
			mRow := m.data[row]
			outData[row] = make([]float64, columnCount)
			outDataRow := outData[row]
			for col := 0; col < columnCount; col++ {
				outDataRow[col] = dataRow[col] + mRow[col]
			}
		}

	} else {
		rowCount := a2drrm.RowDimension()
		columnCount := a2drrm.ColumnDimension()
		outData = make([][]float64, rowCount)
		for row := 0; row < rowCount; row++ {
			dataRow := a2drrm.data[row]
			outData[row] = make([]float64, columnCount)
			outDataRow := outData[row]
			for col := 0; col < columnCount; col++ {
				outDataRow[col] = dataRow[col] + mat.At(row, col)
			}
		}

	}

	m := new(Array2DRowRealMatrix)
	m.data = outData
	return m
}

func (a2drrm *Array2DRowRealMatrix) Subtract(mat RealMatrix) RealMatrix {
	// Safety check.
	if err := checkAdditionCompatible(a2drrm, mat); err != nil {
		panic(err)
	}

	var outData [][]float64

	if m, ok := mat.(*Array2DRowRealMatrix); ok {
		rowCount := a2drrm.RowDimension()
		columnCount := a2drrm.ColumnDimension()
		outData = make([][]float64, rowCount)
		for row := 0; row < rowCount; row++ {
			dataRow := a2drrm.data[row]
			mRow := m.data[row]
			outData[row] = make([]float64, columnCount)
			outDataRow := outData[row]
			for col := 0; col < columnCount; col++ {
				outDataRow[col] = dataRow[col] - mRow[col]
			}
		}
	} else {
		rowCount := a2drrm.RowDimension()
		columnCount := a2drrm.ColumnDimension()
		outData = make([][]float64, rowCount)
		for row := 0; row < rowCount; row++ {
			dataRow := a2drrm.data[row]
			outData[row] = make([]float64, columnCount)
			outDataRow := outData[row]
			for col := 0; col < columnCount; col++ {
				outDataRow[col] = dataRow[col] - mat.At(row, col)
			}
		}
	}

	m := new(Array2DRowRealMatrix)
	m.data = outData
	return m
}

func (a2drrm *Array2DRowRealMatrix) Multiply(mat RealMatrix) RealMatrix {
	if err := checkMultiplicationCompatible(a2drrm, mat); err != nil {
		panic(err)
	}

	var outData [][]float64

	if m, ok := mat.(*Array2DRowRealMatrix); ok {
		nRows := a2drrm.RowDimension()
		nCols := m.ColumnDimension()
		nSum := a2drrm.ColumnDimension()

		outData = make([][]float64, nRows)
		mCol := make([]float64, nSum)

		for col := 0; col < nCols; col++ {
			for mRow := 0; mRow < nSum; mRow++ {
				mCol[mRow] = m.data[mRow][col]
			}

			for row := 0; row < nRows; row++ {
				var sum float64
				for i := 0; i < nSum; i++ {
					sum += a2drrm.data[row][i] * mCol[i]
				}
				if outData[row] == nil {
					outData[row] = make([]float64, nCols)
				}

				outData[row][col] = sum
			}
		}

	} else {
		nRows := a2drrm.RowDimension()
		nCols := mat.ColumnDimension()
		nSum := a2drrm.ColumnDimension()
		outData = make([][]float64, nRows)

		for row := 0; row < nRows; row++ {
			outData[row] = make([]float64, nCols)
			dataRow := a2drrm.data[row]
			for col := 0; col < nCols; col++ {
				var sum float64
				for i := 0; i < nSum; i++ {
					sum += dataRow[i] * mat.At(i, col)
				}
				outData[row][col] = sum
			}
		}

	}

	m := new(Array2DRowRealMatrix)
	m.data = outData
	return m
}

func (a2drrm *Array2DRowRealMatrix) ScalarAdd(d float64) RealMatrix {
	rowCount := a2drrm.RowDimension()
	columnCount := a2drrm.ColumnDimension()

	outData := make([][]float64, rowCount)
	for row := 0; row < rowCount; row++ {
		outData[row] = make([]float64, columnCount)
		for col := 0; col < columnCount; col++ {
			outData[row][col] = a2drrm.data[row][col] + d
		}
	}

	mat := new(Array2DRowRealMatrix)
	mat.data = outData
	return mat
}

func (a2drrm *Array2DRowRealMatrix) ScalarMultiply(d float64) RealMatrix {
	rowCount := a2drrm.RowDimension()
	columnCount := a2drrm.ColumnDimension()

	outData := make([][]float64, rowCount)
	for row := 0; row < rowCount; row++ {
		outData[row] = make([]float64, columnCount)
		for col := 0; col < columnCount; col++ {
			outData[row][col] = a2drrm.data[row][col] * d
		}
	}

	mat := new(Array2DRowRealMatrix)
	mat.data = outData
	return mat
}

func (a2drrm *Array2DRowRealMatrix) Data() [][]float64 {
	return a2drrm.copyOut()
}

func (a2drrm *Array2DRowRealMatrix) DataRef() [][]float64 {
	return a2drrm.data
}

func (a2drrm *Array2DRowRealMatrix) copyIn(in [][]float64) {
	a2drrm.SetSubMatrix(in, 0, 0)
}

func (a2drrm *Array2DRowRealMatrix) SetSubMatrix(subMatrix [][]float64, row, column int) {

	if subMatrix == nil {
		panic(invalidArgumentSimpleErrorf())
	}
	nRows := len(subMatrix)
	if nRows == 0 {
		panic(noDataErrorf(at_least_one_row))
	}

	nCols := len(subMatrix[0])
	if nCols == 0 {
		panic(noDataErrorf(at_least_one_row))
	}

	if a2drrm.data == nil {
		if row > 0 {
			panic(mathIllegalStateErrorf(first_rows_not_initialized_yet, row))
		}

		if column > 0 {
			panic(mathIllegalStateErrorf(first_columns_not_initialized_yet, column))
		}

		a2drrm.data = make([][]float64, len(subMatrix))

		for i := 0; i < nRows; i++ {
			a2drrm.data[i] = make([]float64, nCols)
		}

		for i := 0; i < len(a2drrm.data); i++ {
			if len(subMatrix[i]) != nCols {
				panic(dimensionsMismatchSimpleErrorf(len(subMatrix[i]), nCols))
			}

			copy(a2drrm.data[i+row][column:column+nCols], subMatrix[i][0:nCols])
		}
	} else {

		for r := 1; r < nRows; r++ {
			if len(subMatrix[r]) != nCols {
				panic(dimensionsMismatchSimpleErrorf(nCols, len(subMatrix[r])))
			}
		}

		if err := checkRowIndex(a2drrm, row); err != nil {
			panic(err)
		}
		if err := checkColumnIndex(a2drrm, column); err != nil {
			panic(err)
		}
		if err := checkRowIndex(a2drrm, nRows+row-1); err != nil {
			panic(err)
		}
		if err := checkColumnIndex(a2drrm, nCols+column-1); err != nil {
			panic(err)
		}

		for i := 0; i < nRows; i++ {
			for j := 0; j < nCols; j++ {
				a2drrm.SetEntry(row+i, column+j, subMatrix[i][j])
			}
		}
	}

}

func (a2drrm *Array2DRowRealMatrix) At(row, column int) float64 {
	if err := checkMatrixIndex(a2drrm, row, column); err != nil {
		panic(err)
	}
	return a2drrm.data[row][column]
}

func (a2drrm *Array2DRowRealMatrix) ColumnAt(column int) []float64 {
	if err := checkColumnIndex(a2drrm, column); err != nil {
		panic(err)
	}
	nRows := a2drrm.RowDimension()
	out := make([]float64, nRows)
	for i := 0; i < nRows; i++ {
		out[i] = a2drrm.data[i][column]
	}

	return out
}

func (a2drrm *Array2DRowRealMatrix) SetColumn(column int, array []float64) {
	if err := checkColumnIndex(a2drrm, column); err != nil {
		panic(err)
	}
	nRows := a2drrm.RowDimension()
	if len(array) != nRows {
		panic(matrixDimensionMismatchErrorf(len(array), 1, nRows, 1))
	}
	for i := 0; i < nRows; i++ {
		a2drrm.data[i][column] = array[i]
	}
}

func (a2drrm *Array2DRowRealMatrix) SetEntry(row, column int, value float64) {
	if err := checkMatrixIndex(a2drrm, row, column); err != nil {
		panic(err)
	}

	a2drrm.data[row][column] = value
}

func (a2drrm *Array2DRowRealMatrix) AddToEntry(row, column int, increment float64) {
	if err := checkMatrixIndex(a2drrm, row, column); err != nil {
		panic(err)
	}

	a2drrm.data[row][column] += increment
}

func (a2drrm *Array2DRowRealMatrix) MultiplyEntry(row, column int, factor float64) {
	if err := checkMatrixIndex(a2drrm, row, column); err != nil {
		panic(err)
	}

	a2drrm.data[row][column] *= factor
}

func (a2drrm *Array2DRowRealMatrix) RowDimension() int {
	if a2drrm.data == nil {
		return 0
	}

	return len(a2drrm.data)

}

func (a2drrm *Array2DRowRealMatrix) ColumnDimension() int {
	if a2drrm.data == nil || a2drrm.data[0] == nil {
		return 0
	}
	return len(a2drrm.data[0])
}

func (a2drrm *Array2DRowRealMatrix) Transpose() RealMatrix {
	nRows := a2drrm.RowDimension()
	nCols := a2drrm.ColumnDimension()

	data := make([][]float64, nCols)
	for i := 0; i < len(data); i++ {
		data[i] = make([]float64, nRows)
	}

	for i := 0; i < nRows; i++ {
		for j := 0; j < nCols; j++ {
			data[j][i] = a2drrm.data[i][j]
		}
	}

	return &Array2DRowRealMatrix{data: data}
}

func (a2drrm *Array2DRowRealMatrix) Trace() float64 {
	nRows := a2drrm.RowDimension()
	nCols := a2drrm.ColumnDimension()
	if nRows != nCols {
		panic(nonSquareMatrixSimpleErrorf(nRows, nCols))
	}
	trace := 0.
	for i := 0; i < nRows; i++ {
		trace += a2drrm.data[i][i]
	}
	return trace
}

func (a2drrm *Array2DRowRealMatrix) Operate(v []float64) []float64 {
	nRows := a2drrm.RowDimension()
	nCols := a2drrm.ColumnDimension()
	if len(v) != nCols {
		panic(dimensionsMismatchSimpleErrorf(len(v), nCols))
	}
	out := make([]float64, nRows)
	for row := 0; row < nRows; row++ {
		var sum float64
		for i := 0; i < nCols; i++ {
			sum += a2drrm.data[row][i] * v[i]
		}
		out[row] = sum
	}
	return out
}

func (a2drrm *Array2DRowRealMatrix) OperateVector(vec RealVector) RealVector {
	var out []float64
	if v, ok := vec.(*ArrayRealVector); ok {
		out = a2drrm.Operate(v.DataRef())
	} else {
		nRows := a2drrm.RowDimension()
		nCols := a2drrm.ColumnDimension()
		if v.Dimension() != nCols {
			panic(dimensionsMismatchSimpleErrorf(v.Dimension(), nCols))
		}

		out = make([]float64, nRows)
		for row := 0; row < nRows; row++ {
			var sum float64
			for i := 0; i < nCols; i++ {
				sum += a2drrm.data[row][i] * v.At(i)
			}
			out[row] = sum
		}
	}

	v := new(ArrayRealVector)
	v.data = out
	return v
}

func (a2drrm *Array2DRowRealMatrix) PreMultiplyMatrix(m RealMatrix) RealMatrix {
	return m.Multiply(a2drrm)
}

func (a2drrm *Array2DRowRealMatrix) PreMultiplyVector(vec RealVector) RealVector {
	var out []float64
	if v, ok := vec.(*ArrayRealVector); ok {
		out = a2drrm.PreMultiply(v.DataRef())
	} else {
		nRows := a2drrm.RowDimension()
		nCols := a2drrm.ColumnDimension()
		if v.Dimension() != nRows {
			panic(dimensionsMismatchSimpleErrorf(v.Dimension(), nRows))
		}

		out = make([]float64, nCols)
		for col := 0; col < nCols; col++ {
			var sum float64
			for i := 0; i < nRows; i++ {
				sum += a2drrm.data[i][col] * v.At(i)
			}
			out[col] = sum
		}
	}

	v := new(ArrayRealVector)
	v.data = append([]float64{}, out...)
	return v
}

func (a2drrm *Array2DRowRealMatrix) PreMultiply(v []float64) []float64 {
	nRows := a2drrm.RowDimension()
	nCols := a2drrm.ColumnDimension()
	if len(v) != nRows {
		panic(dimensionsMismatchSimpleErrorf(len(v), nRows))
	}

	out := make([]float64, nCols)
	for col := 0; col < nCols; col++ {
		var sum float64
		for i := 0; i < nRows; i++ {
			sum += a2drrm.data[i][col] * v[i]
		}
		out[col] = sum
	}

	return out

}

func (a2drrm *Array2DRowRealMatrix) SubMatrix(startRow, endRow, startColumn, endColumn int) RealMatrix {
	if err := checkSubMatrixIndex(a2drrm, startRow, endRow, startColumn, endColumn); err != nil {
		panic(err)
	}
	rowCount := endRow - startRow + 1
	columnCount := endColumn - startColumn + 1
	outData := make([][]float64, rowCount)
	for i := 0; i < rowCount; i++ {
		outData[i] = make([]float64, columnCount)
		copy(outData[i][0:columnCount], a2drrm.data[startRow+i][startColumn:startColumn+columnCount])
	}

	subMatrix := new(Array2DRowRealMatrix)
	subMatrix.data = outData
	return subMatrix
}

func (a2drrm *Array2DRowRealMatrix) WalkInRowOrder(visitor RealMatrixPreservingVisitor) float64 {
	rows := a2drrm.RowDimension()
	columns := a2drrm.ColumnDimension()
	visitor.Start(rows, columns, 0, rows-1, 0, columns-1)
	for i := 0; i < rows; i++ {
		rowI := a2drrm.data[i]
		for j := 0; j < columns; j++ {
			visitor.Visit(i, j, rowI[j])
		}
	}
	return visitor.End()
}

func (a2drrm *Array2DRowRealMatrix) WalkInRowOrderBounded(visitor RealMatrixPreservingVisitor, startRow, endRow, startColumn, endColumn int) float64 {
	if err := checkSubMatrixIndex(a2drrm, startRow, endRow, startColumn, endColumn); err != nil {
		panic(err)
	}
	visitor.Start(a2drrm.RowDimension(), a2drrm.ColumnDimension(), startRow, endRow, startColumn, endColumn)
	for i := startRow; i <= endRow; i++ {
		rowI := a2drrm.data[i]
		for j := startColumn; j <= endColumn; j++ {
			visitor.Visit(i, j, rowI[j])
		}
	}
	return visitor.End()
}

func (a2drrm *Array2DRowRealMatrix) WalkInUpdateRowOrder(visitor RealMatrixChangingVisitor) float64 {
	rows := a2drrm.RowDimension()
	columns := a2drrm.ColumnDimension()
	visitor.Start(rows, columns, 0, rows-1, 0, columns-1)
	for i := 0; i < rows; i++ {
		rowI := a2drrm.data[i]
		for j := 0; j < columns; j++ {
			rowI[j] = visitor.Visit(i, j, rowI[j])
		}
	}
	return visitor.End()
}

func (a2drrm *Array2DRowRealMatrix) WalkInUpdateRowOrderBounded(visitor RealMatrixChangingVisitor, startRow, endRow, startColumn, endColumn int) float64 {
	if err := checkSubMatrixIndex(a2drrm, startRow, endRow, startColumn, endColumn); err != nil {
		panic(err)
	}
	visitor.Start(a2drrm.RowDimension(), a2drrm.ColumnDimension(), startRow, endRow, startColumn, endColumn)
	for i := startRow; i <= endRow; i++ {
		rowI := a2drrm.data[i]
		for j := startColumn; j <= endColumn; j++ {
			rowI[j] = visitor.Visit(i, j, rowI[j])
		}
	}
	return visitor.End()
}

func (a2drrm *Array2DRowRealMatrix) WalkInUpdateColumnOrder(visitor RealMatrixChangingVisitor) float64 {
	rows := a2drrm.RowDimension()
	columns := a2drrm.ColumnDimension()
	visitor.Start(rows, columns, 0, rows-1, 0, columns-1)
	for j := 0; j < columns; j++ {
		for i := 0; i < rows; i++ {
			rowI := a2drrm.data[i]
			rowI[j] = visitor.Visit(i, j, rowI[j])
		}
	}
	return visitor.End()
}

func (a2drrm *Array2DRowRealMatrix) WalkInColumnOrder(visitor RealMatrixPreservingVisitor) float64 {
	rows := a2drrm.RowDimension()
	columns := a2drrm.ColumnDimension()
	visitor.Start(rows, columns, 0, rows-1, 0, columns-1)
	for j := 0; j < columns; j++ {
		for i := 0; i < rows; i++ {
			visitor.Visit(i, j, a2drrm.data[i][j])
		}
	}
	return visitor.End()
}

func (a2drrm *Array2DRowRealMatrix) WalkInUpdateColumnOrderBounded(visitor RealMatrixChangingVisitor, startRow, endRow, startColumn, endColumn int) float64 {
	if err := checkSubMatrixIndex(a2drrm, startRow, endRow, startColumn, endColumn); err != nil {
		panic(err)
	}
	visitor.Start(a2drrm.RowDimension(), a2drrm.ColumnDimension(), startRow, endRow, startColumn, endColumn)
	for j := startColumn; j <= endColumn; j++ {
		for i := startRow; i <= endRow; i++ {
			rowI := a2drrm.data[i]
			rowI[j] = visitor.Visit(i, j, rowI[j])
		}
	}
	return visitor.End()
}

func (a2drrm *Array2DRowRealMatrix) WalkInColumnOrderBounded(visitor RealMatrixPreservingVisitor, startRow, endRow, startColumn, endColumn int) float64 {
	if err := checkSubMatrixIndex(a2drrm, startRow, endRow, startColumn, endColumn); err != nil {
		panic(err)
	}
	visitor.Start(a2drrm.RowDimension(), a2drrm.ColumnDimension(), startRow, endRow, startColumn, endColumn)
	for j := startColumn; j <= endColumn; j++ {
		for i := startRow; i <= endRow; i++ {
			visitor.Visit(i, j, a2drrm.data[i][j])
		}
	}
	return visitor.End()
}

func (a2drrm *Array2DRowRealMatrix) WalkInUpdateOptimizedOrder(visitor RealMatrixChangingVisitor) float64 {
	return a2drrm.WalkInUpdateRowOrder(visitor)
}

func (a2drrm *Array2DRowRealMatrix) WalkInOptimizedOrder(visitor RealMatrixPreservingVisitor) float64 {
	return a2drrm.WalkInRowOrder(visitor)
}

func (a2drrm *Array2DRowRealMatrix) WalkInUpdateOptimizedOrderBounded(visitor RealMatrixChangingVisitor, startRow, endRow, startColumn, endColumn int) float64 {
	return a2drrm.WalkInUpdateRowOrderBounded(visitor, startRow, endRow, startColumn, endColumn)
}

func (a2drrm *Array2DRowRealMatrix) WalkInOptimizedOrderBounded(visitor RealMatrixPreservingVisitor, startRow, endRow, startColumn, endColumn int) float64 {
	return a2drrm.WalkInRowOrderBounded(visitor, startRow, endRow, startColumn, endColumn)
}

func (a2drrm *Array2DRowRealMatrix) copyOut() [][]float64 {
	nRows := a2drrm.RowDimension()
	out := make([][]float64, nRows)
	for i := 0; i < nRows; i++ {
		out[i] = make([]float64, a2drrm.ColumnDimension())
		copy(out[i][0:len(a2drrm.data[i])], a2drrm.data[i][0:len(a2drrm.data[i])])
	}
	return out
}

func (a2drrm *Array2DRowRealMatrix) RowAt(row int) []float64 {
	if err := checkRowIndex(a2drrm, row); err != nil {
		panic(err)
	}
	nCols := a2drrm.ColumnDimension()
	out := make([]float64, nCols)
	copy(out[0:nCols], a2drrm.data[row][0:nCols])

	return out
}

func (a2drrm *Array2DRowRealMatrix) SetRow(row int, array []float64) {
	if err := checkRowIndex(a2drrm, row); err != nil {
		panic(err)
	}
	nCols := a2drrm.ColumnDimension()
	if len(array) != nCols {
		panic(matrixDimensionMismatchErrorf(1, len(array), 1, nCols))
	}
	copy(a2drrm.data[row][0:row+nCols], array[0:nCols])
}

func (a2drrm *Array2DRowRealMatrix) RowMatrixAt(row int) RealMatrix {
	if err := checkRowIndex(a2drrm, row); err != nil {
		panic(err)
	}
	nCols := a2drrm.ColumnDimension()
	out, err := NewArray2DRowRealMatrix(1, nCols)
	if err != nil {
		panic(err)
	}
	for i := 0; i < nCols; i++ {
		out.data[0][i] = a2drrm.data[row][i]
	}

	return out
}

func (a2drrm *Array2DRowRealMatrix) SetRowMatrix(row int, matrix RealMatrix) {
	if err := checkRowIndex(a2drrm, row); err != nil {
		panic(err)
	}
	nCols := a2drrm.ColumnDimension()
	if (matrix.RowDimension() != 1) || (matrix.ColumnDimension() != nCols) {
		panic(matrixDimensionMismatchErrorf(matrix.RowDimension(), matrix.ColumnDimension(), 1, nCols))
	}
	for i := 0; i < nCols; i++ {
		a2drrm.data[row][i] = matrix.At(0, i)
	}
}

func (a2drrm *Array2DRowRealMatrix) ColumnMatrixAt(column int) RealMatrix {
	if err := checkColumnIndex(a2drrm, column); err != nil {
		panic(err)
	}
	nRows := a2drrm.RowDimension()
	out, err := NewArray2DRowRealMatrix(nRows, 1)
	if err != nil {
		panic(err)
	}
	for i := 0; i < nRows; i++ {
		out.data[i][0] = a2drrm.data[i][column]
	}

	return out
}

func (a2drrm *Array2DRowRealMatrix) SetColumnMatrix(column int, matrix RealMatrix) {
	if err := checkColumnIndex(a2drrm, column); err != nil {
		panic(err)
	}
	nRows := a2drrm.RowDimension()
	if (matrix.RowDimension() != nRows) || (matrix.ColumnDimension() != 1) {
		panic(matrixDimensionMismatchErrorf(matrix.RowDimension(), matrix.ColumnDimension(), nRows, 1))
	}
	for i := 0; i < nRows; i++ {
		a2drrm.data[i][column] = matrix.At(i, 0)
	}
}

func (a2drrm *Array2DRowRealMatrix) RowVectorAt(row int) RealVector {
	v := new(ArrayRealVector)
	v.data = append([]float64{}, a2drrm.RowAt(row)...)
	return v
}

func (a2drrm *Array2DRowRealMatrix) SetRowVector(row int, vec RealVector) {
	if err := checkRowIndex(a2drrm, row); err != nil {
		panic(err)
	}

	nCols := a2drrm.ColumnDimension()
	if vec.Dimension() != nCols {
		panic(matrixDimensionMismatchErrorf(1, vec.Dimension(), 1, nCols))
	}
	for i := 0; i < nCols; i++ {
		a2drrm.data[row][i] = vec.At(i)
	}
}

func (a2drrm *Array2DRowRealMatrix) ColumnVectorAt(column int) RealVector {
	v := new(ArrayRealVector)
	v.data = append([]float64{}, a2drrm.ColumnAt(column)...)
	return v
}

func (a2drrm *Array2DRowRealMatrix) SetColumnVector(column int, vec RealVector) {
	checkColumnIndex(a2drrm, column)
	nRows := a2drrm.RowDimension()
	if vec.Dimension() != nRows {
		panic(matrixDimensionMismatchErrorf(1, vec.Dimension(), nRows, 1))

	}
	for i := 0; i < nRows; i++ {
		a2drrm.data[i][column] = vec.At(i)
	}
}

func (a2drrm *Array2DRowRealMatrix) Equals(object interface{}) bool {
	if object == a2drrm {
		return true
	}
	if _, ok := object.(RealMatrix); !ok {
		return false
	}

	m := object.(RealMatrix)
	nRows := a2drrm.RowDimension()
	nCols := a2drrm.ColumnDimension()
	if m.ColumnDimension() != nCols || m.RowDimension() != nRows {
		return false
	}
	for row := 0; row < nRows; row++ {
		for col := 0; col < nCols; col++ {
			if a2drrm.At(row, col) != m.At(row, col) {
				return false
			}
		}
	}
	return true
}

func checkAdditionCompatible(left, right Matrix) error {
	if !canAdd(left, right) {
		return matrixDimensionMismatchErrorf(left.RowDimension(), left.ColumnDimension(), right.RowDimension(), right.ColumnDimension())
	}

	return nil
}

func canAdd(left, right Matrix) bool {
	return left.RowDimension() == right.RowDimension() && left.ColumnDimension() == right.ColumnDimension()
}

func checkMultiplicationCompatible(left, right Matrix) error {
	if !canMultiply(left, right) {
		return dimensionsMismatchSimpleErrorf(left.ColumnDimension(), right.RowDimension())
	}

	return nil
}

func canMultiply(left, right Matrix) bool {
	return left.ColumnDimension() == right.RowDimension()
}

func checkMatrixIndex(m Matrix, row, column int) error {
	if err := checkRowIndex(m, row); err != nil {
		return err
	}
	if err := checkColumnIndex(m, column); err != nil {
		return err
	}

	return nil
}

func checkRowIndex(m Matrix, row int) error {
	if row < 0 || row >= m.RowDimension() {
		return outOfRangeErrorf(row_index, float64(row), 0, float64(m.RowDimension()-1))
	}

	return nil
}

func checkColumnIndex(m Matrix, column int) error {
	if column < 0 || column >= m.ColumnDimension() {
		return outOfRangeErrorf(column_index, float64(column), 0, float64(m.ColumnDimension()-1))
	}

	return nil
}

func checkSubMatrixIndex(m Matrix, startRow, endRow, startColumn, endColumn int) error {
	if err := checkRowIndex(m, startRow); err != nil {
		return err
	}

	if err := checkRowIndex(m, endRow); err != nil {
		return err
	}

	if endRow < startRow {
		return numberIsTooSmallErrorf(initial_row_after_final_row, float64(endRow), float64(startRow), false)
	}

	if err := checkColumnIndex(m, startColumn); err != nil {
		return err
	}

	if err := checkColumnIndex(m, endColumn); err != nil {
		return err
	}

	if endColumn < startColumn {
		return numberIsTooSmallErrorf(initial_column_after_final_column, float64(endColumn), float64(startColumn), false)
	}

	return nil
}

func checkSubMatrixIndexFromIndices(m Matrix, selectedRows, selectedColumns []int) error {
	if selectedRows == nil {
		return invalidArgumentSimpleErrorf()
	}
	if selectedColumns == nil {
		return invalidArgumentSimpleErrorf()
	}
	if len(selectedRows) == 0 {
		return noDataErrorf(empty_selected_row_index_array)
	}
	if len(selectedColumns) == 0 {
		return noDataErrorf(empty_selected_column_index_array)
	}

	for _, row := range selectedRows {
		if err := checkRowIndex(m, row); err != nil {
			return err
		}
	}

	for _, column := range selectedColumns {
		if err := checkColumnIndex(m, column); err != nil {
			return err
		}
	}

	return nil
}
