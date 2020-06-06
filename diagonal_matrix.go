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

import (
	"math"
)

// Implementation of a diagonal matrix.
type DiagonalMatrix struct {
	data []float64
}

func NewDiagonalMatrixWithDimension(dimension int) (*DiagonalMatrix, error) {
	if dimension < 1 {
		return nil, notStrictlyPositiveErrorf(float64(dimension))
	}

	ans := new(DiagonalMatrix)
	ans.data = make([]float64, dimension)
	return ans, nil
}

func NewDiagonalMatrixFromSlice(data []float64) (*DiagonalMatrix, error) {
	return NewDiagonalMatrix(data, true)
}

func NewDiagonalMatrix(data []float64, copyArray bool) (*DiagonalMatrix, error) {
	if data == nil {
		return nil, invalidArgumentSimpleErrorf()
	}

	ans := new(DiagonalMatrix)
	if copyArray {
		ans.data = append([]float64{}, data...)
	} else {
		ans.data = data
	}

	return ans, nil
}

func (dm *DiagonalMatrix) Copy() RealMatrix {
	ans := new(DiagonalMatrix)
	ans.data = append([]float64{}, dm.data...)
	return ans
}

func (dm *DiagonalMatrix) Add(mat RealMatrix) RealMatrix {
	// Safety check.
	if err := checkAdditionCompatible(dm, mat); err != nil {
		panic(err)
	}

	if m, ok := mat.(*DiagonalMatrix); ok {
		dim := dm.RowDimension()
		outData := make([]float64, dim)
		for i := 0; i < dim; i++ {
			outData[i] = dm.data[i] + m.data[i]
		}

		mat := new(DiagonalMatrix)
		mat.data = outData
		return mat
	} else {
		rowCount := mat.RowDimension()
		columnCount := mat.ColumnDimension()
		if rowCount != columnCount {
			panic(dimensionsMismatchSimpleErrorf(rowCount, columnCount))
		}

		out, err := NewDiagonalMatrixWithDimension(rowCount)
		if err != nil {
			panic(err)
		}

		for row := 0; row < rowCount; row++ {
			for col := 0; col < columnCount; col++ {
				out.SetEntry(row, col, dm.At(row, col)+mat.At(row, col))
			}
		}

		return out
	}

}

func (dm *DiagonalMatrix) Subtract(mat RealMatrix) RealMatrix {
	// Safety check.
	if err := checkAdditionCompatible(dm, mat); err != nil {
		panic(err)
	}
	if m, ok := mat.(*DiagonalMatrix); ok {
		dim := dm.RowDimension()
		outData := make([]float64, dim)
		for i := 0; i < dim; i++ {
			outData[i] = dm.data[i] - m.data[i]
		}

		mat := new(DiagonalMatrix)
		mat.data = outData
		return mat
	} else {

		rowCount := mat.RowDimension()
		columnCount := mat.ColumnDimension()
		if rowCount != columnCount {
			panic(dimensionsMismatchSimpleErrorf(rowCount, columnCount))
		}

		out, err := NewDiagonalMatrixWithDimension(rowCount)
		if err != nil {
			panic(err)
		}
		for row := 0; row < rowCount; row++ {
			for col := 0; col < columnCount; col++ {
				out.SetEntry(row, col, dm.At(row, col)-mat.At(row, col))
			}
		}

		return out
	}
}

func (dm *DiagonalMatrix) Multiply(mat RealMatrix) RealMatrix {
	if err := checkMultiplicationCompatible(dm, mat); err != nil {
		panic(err)
	}

	if m, ok := mat.(*DiagonalMatrix); ok {
		dim := dm.RowDimension()
		outData := make([]float64, dim)
		for i := 0; i < dim; i++ {
			outData[i] = dm.data[i] * m.data[i]
		}

		mat := new(DiagonalMatrix)
		mat.data = outData
		return mat

	} else {
		nRows := mat.RowDimension()
		nCols := mat.ColumnDimension()
		product := make([][]float64, nRows)
		for r := 0; r < nRows; r++ {
			product[r] = make([]float64, nCols)
			for c := 0; c < nCols; c++ {
				product[r][c] = dm.data[r] * mat.At(r, c)
			}
		}

		rm := new(Array2DRowRealMatrix)
		rm.copyIn(product)
		return rm
	}
}

func (dm *DiagonalMatrix) Data() [][]float64 {
	dim := dm.RowDimension()
	out := make([][]float64, dim)

	for i := 0; i < dim; i++ {
		if out[i] == nil {
			out[i] = make([]float64, dim)
		}

		out[i][i] = dm.data[i]
	}

	return out
}

func (dm *DiagonalMatrix) DataRef() []float64 {
	return dm.data
}

func (dm *DiagonalMatrix) At(row, column int) float64 {
	if err := checkMatrixIndex(dm, row, column); err != nil {
		panic(err)
	}

	if row == column {
		return dm.data[row]
	}
	return 0
}

func (dm *DiagonalMatrix) SetEntry(row, column int, value float64) {
	if row == column {
		if err := checkRowIndex(dm, row); err != nil {
			panic(err)
		}
		dm.data[row] = value
	} else {
		if err := dm.ensureZero(value); err != nil {
			panic(err)
		}
	}
}

func (dm *DiagonalMatrix) AddToEntry(row, column int, increment float64) {
	if row == column {
		if err := checkRowIndex(dm, row); err != nil {
			panic(err)
		}
		dm.data[row] += increment
	} else {
		if err := dm.ensureZero(increment); err != nil {
			panic(err)
		}
	}
}

func (dm *DiagonalMatrix) ensureZero(value float64) error {
	if !equalsWithULP(0.0, value, 1) {
		return numberIsTooLargeBoundedErrorf(math.Abs(value), 0, true)
	}

	return nil
}

func (dm *DiagonalMatrix) MultiplyEntry(row, column int, factor float64) {
	// we don't care about non-diagonal elements for multiplication
	if row == column {
		if err := checkRowIndex(dm, row); err != nil {
			panic(err)
		}
		dm.data[row] *= factor
	}
}

func (dm *DiagonalMatrix) RowDimension() int {
	return len(dm.data)
}

func (dm *DiagonalMatrix) ColumnDimension() int {
	return len(dm.data)
}

func (dm *DiagonalMatrix) Operate(v []float64) []float64 {
	diag := new(DiagonalMatrix)
	diag.data = v

	return dm.Multiply(diag).(*DiagonalMatrix).DataRef()
}

func (dm *DiagonalMatrix) OperateVector(vec RealVector) RealVector {
	if v, ok := vec.(*ArrayRealVector); ok {
		vec := new(ArrayRealVector)
		vec.data = append([]float64{}, dm.Operate(v.DataRef())...)
		return vec
	} else {
		nRows := dm.RowDimension()
		nCols := dm.ColumnDimension()
		if vec.Dimension() != nCols {
			panic(dimensionsMismatchSimpleErrorf(vec.Dimension(), nCols))
		}

		out := make([]float64, nRows)
		for row := 0; row < nRows; row++ {
			var sum float64
			for i := 0; i < nCols; i++ {
				sum += dm.At(row, i) * vec.At(i)
			}
			out[row] = sum
		}

		vec := new(ArrayRealVector)
		vec.data = append([]float64{}, out...)
		return vec
	}
}

func (dm *DiagonalMatrix) PreMultiply(v []float64) []float64 {
	return dm.Operate(v)
}

func (dm *DiagonalMatrix) PreMultiplyVector(vec RealVector) RealVector {
	var vectorData []float64
	if v, ok := vec.(*ArrayRealVector); ok {
		vectorData = v.DataRef()
	} else {
		vectorData = vec.ToArray()
	}
	rv, err := NewRealVector(dm.PreMultiply(vectorData))
	if err != nil {
		panic(err)
	}

	return rv
}

func (dm *DiagonalMatrix) PreMultiplyMatrix(m RealMatrix) RealMatrix {
	return m.Multiply(dm)
}

func (dm *DiagonalMatrix) Inverse() *DiagonalMatrix {
	return dm.InverseWithThreshold(0)
}

func (dm *DiagonalMatrix) InverseWithThreshold(threshold float64) *DiagonalMatrix {
	if dm.IsSingular(threshold) {
		panic(singularMatrixSimpleErrorf())
	}

	result := make([]float64, len(dm.data))
	for i := 0; i < len(dm.data); i++ {
		result[i] = 1.0 / dm.data[i]
	}

	mat := new(DiagonalMatrix)
	mat.data = result
	return mat
}

func (dm *DiagonalMatrix) IsSingular(threshold float64) bool {
	for i := 0; i < len(dm.data); i++ {
		if equalsWithError(dm.data[i], 0.0, threshold) {
			return true
		}
	}
	return false
}

func (dm *DiagonalMatrix) ColumnAt(column int) []float64 {
	if err := checkColumnIndex(dm, column); err != nil {
		panic(err)
	}
	nRows := dm.RowDimension()
	out := make([]float64, nRows)
	for i := 0; i < nRows; i++ {
		out[i] = dm.At(i, column)
	}

	return out
}

func (dm *DiagonalMatrix) ColumnMatrixAt(column int) RealMatrix {
	if err := checkColumnIndex(dm, column); err != nil {
		panic(err)
	}
	nRows := dm.RowDimension()
	if 1 != nRows {
		panic(dimensionsMismatchSimpleErrorf(1, nRows))
	}
	out, err := NewDiagonalMatrixWithDimension(1)
	if err != nil {
		panic(err)
	}
	for i := 0; i < nRows; i++ {
		out.SetEntry(i, 0, dm.At(i, column))
	}

	return out
}

func (dm *DiagonalMatrix) SetColumnMatrix(column int, matrix RealMatrix) {
	if err := checkColumnIndex(dm, column); err != nil {
		panic(err)
	}
	nRows := dm.RowDimension()
	if (matrix.RowDimension() != nRows) || (matrix.ColumnDimension() != 1) {
		panic(matrixDimensionMismatchErrorf(matrix.RowDimension(), matrix.ColumnDimension(), nRows, 1))
	}
	for i := 0; i < nRows; i++ {
		dm.SetEntry(i, column, matrix.At(i, 0))
	}
}

func (dm *DiagonalMatrix) ColumnVectorAt(column int) RealVector {
	mat := new(ArrayRealVector)
	mat.data = dm.ColumnAt(column)
	return mat
}

func (dm *DiagonalMatrix) SetColumnVector(column int, vec RealVector) {
	if err := checkColumnIndex(dm, column); err != nil {
		panic(err)
	}
	nRows := dm.RowDimension()
	if vec.Dimension() != nRows {
		panic(matrixDimensionMismatchErrorf(vec.Dimension(), 1, nRows, 1))
	}
	for i := 0; i < nRows; i++ {
		dm.SetEntry(i, column, vec.At(i))
	}
}

func (dm *DiagonalMatrix) Equals(object interface{}) bool {
	if object == dm {
		return true
	}
	if _, ok := object.(RealMatrix); !ok {
		return false
	}

	m := object.(RealMatrix)
	nRows := dm.RowDimension()
	nCols := dm.ColumnDimension()
	if m.ColumnDimension() != nCols || m.RowDimension() != nRows {
		return false
	}
	for row := 0; row < nRows; row++ {
		for col := 0; col < nCols; col++ {
			if dm.At(row, col) != m.At(row, col) {
				return false
			}
		}
	}
	return true
}

func (dm *DiagonalMatrix) RowAt(row int) []float64 {
	if err := checkRowIndex(dm, row); err != nil {
		panic(err)
	}
	nCols := dm.ColumnDimension()
	out := make([]float64, nCols)
	for i := 0; i < nCols; i++ {
		out[i] = dm.At(row, i)
	}

	return out
}

func (dm *DiagonalMatrix) RowMatrixAt(row int) RealMatrix {
	if err := checkRowIndex(dm, row); err != nil {
		panic(err)
	}
	nCols := dm.ColumnDimension()
	if 1 != nCols {
		panic(dimensionsMismatchSimpleErrorf(1, nCols))
	}
	out, err := NewDiagonalMatrixWithDimension(1)
	if err != nil {
		panic(err)
	}
	for i := 0; i < nCols; i++ {
		out.SetEntry(0, i, dm.At(row, i))
	}

	return out
}

func (dm *DiagonalMatrix) RowVectorAt(row int) RealVector {
	mat := new(ArrayRealVector)
	mat.data = append([]float64{}, dm.RowAt(row)...)
	return mat
}

func (dm *DiagonalMatrix) ScalarAdd(d float64) RealMatrix {
	rowCount := dm.RowDimension()
	columnCount := dm.ColumnDimension()
	if rowCount != columnCount {
		panic(dimensionsMismatchSimpleErrorf(rowCount, columnCount))
	}
	out, err := NewDiagonalMatrixWithDimension(rowCount)
	if err != nil {
		panic(err)
	}
	for row := 0; row < rowCount; row++ {
		for col := 0; col < columnCount; col++ {
			out.SetEntry(row, col, dm.At(row, col)+d)
		}
	}

	return out
}

func (dm *DiagonalMatrix) ScalarMultiply(d float64) RealMatrix {
	rowCount := dm.RowDimension()
	columnCount := dm.ColumnDimension()
	if rowCount != columnCount {
		panic(dimensionsMismatchSimpleErrorf(rowCount, columnCount))
	}
	out, err := NewDiagonalMatrixWithDimension(rowCount)
	if err != nil {
		panic(err)
	}
	for row := 0; row < rowCount; row++ {
		for col := 0; col < columnCount; col++ {
			out.SetEntry(row, col, dm.At(row, col)*d)
		}
	}

	return out
}

func (dm *DiagonalMatrix) SetColumn(column int, array []float64) {
	if err := checkColumnIndex(dm, column); err != nil {
		panic(err)
	}

	nRows := dm.RowDimension()
	if len(array) != nRows {
		panic(matrixDimensionMismatchErrorf(len(array), 1, nRows, 1))

	}
	for i := 0; i < nRows; i++ {
		dm.SetEntry(i, column, array[i])
	}
}

func (dm *DiagonalMatrix) SetRow(row int, array []float64) {
	if err := checkRowIndex(dm, row); err != nil {
		panic(err)
	}
	nCols := dm.ColumnDimension()
	if len(array) != nCols {
		panic(matrixDimensionMismatchErrorf(1, len(array), 1, nCols))
	}
	for i := 0; i < nCols; i++ {
		dm.SetEntry(row, i, array[i])
	}
}

func (dm *DiagonalMatrix) SetRowMatrix(row int, matrix RealMatrix) {
	if err := checkRowIndex(dm, row); err != nil {
		panic(err)
	}
	nCols := dm.ColumnDimension()
	if (matrix.RowDimension() != 1) || (matrix.ColumnDimension() != nCols) {
		panic(matrixDimensionMismatchErrorf(matrix.RowDimension(), matrix.ColumnDimension(), 1, nCols))
	}
	for i := 0; i < nCols; i++ {
		dm.SetEntry(row, i, matrix.At(0, i))
	}
}

func (dm *DiagonalMatrix) SetRowVector(row int, vec RealVector) {
	if err := checkRowIndex(dm, row); err != nil {
		panic(err)
	}
	nCols := dm.ColumnDimension()
	if vec.Dimension() != nCols {
		panic(matrixDimensionMismatchErrorf(1, vec.Dimension(), 1, nCols))
	}

	for i := 0; i < nCols; i++ {
		dm.SetEntry(row, i, vec.At(i))
	}
}

func (dm *DiagonalMatrix) SetSubMatrix(subMatrix [][]float64, row, column int) {
	if subMatrix == nil {
		panic(invalidArgumentSimpleErrorf())
	}
	nRows := len(subMatrix)
	if nRows == 0 {
		panic(noDataErrorf(at_least_one_row))
	}

	nCols := len(subMatrix[0])
	if nCols == 0 {
		panic(noDataErrorf(at_least_one_column))
	}

	for r := 1; r < nRows; r++ {
		if len(subMatrix[r]) != nCols {
			panic(dimensionsMismatchSimpleErrorf(nCols, len(subMatrix[r])))
		}
	}

	checkRowIndex(dm, row)
	checkColumnIndex(dm, column)
	checkRowIndex(dm, nRows+row-1)
	checkColumnIndex(dm, nCols+column-1)

	for i := 0; i < nRows; i++ {
		for j := 0; j < nCols; j++ {
			dm.SetEntry(row+i, column+j, subMatrix[i][j])
		}
	}
}

func (dm *DiagonalMatrix) SubMatrix(startRow, endRow, startColumn, endColumn int) RealMatrix {
	checkSubMatrixIndex(dm, startRow, endRow, startColumn, endColumn)
	if (endRow - startRow + 1) != (endColumn - startColumn + 1) {
		panic(dimensionsMismatchSimpleErrorf((endRow - startRow + 1), (endColumn - startColumn + 1)))
	}

	subMatrix, err := NewDiagonalMatrixWithDimension(endRow - startRow + 1)
	if err != nil {
		panic(err)
	}

	for i := startRow; i <= endRow; i++ {
		for j := startColumn; j <= endColumn; j++ {
			subMatrix.SetEntry(i-startRow, j-startColumn, dm.At(i, j))
		}
	}

	return subMatrix
}

func (dm *DiagonalMatrix) SubMatrixFromIndices(selectedRows, selectedColumns []int) RealMatrix {
	if err := checkSubMatrixIndexFromIndices(dm, selectedRows, selectedColumns); err != nil {
		panic(err)
	}
	if len(selectedRows) != len(selectedColumns) {
		panic(dimensionsMismatchSimpleErrorf(len(selectedRows), len(selectedColumns)))
	}

	subMatrix, err := NewDiagonalMatrixWithDimension(len(selectedRows))
	if err != nil {
		panic(err)
	}
	drmcv := new(RealMatrixChangingVisitorImpl)

	drmcv.s = func(int, int, int, int, int, int) {

	}

	drmcv.v = func(row, column int, value float64) float64 {
		return dm.At(selectedRows[row], selectedColumns[column])
	}

	drmcv.e = func() float64 {
		return 0
	}

	subMatrix.WalkInUpdateRowOrder(drmcv)

	return subMatrix
}

func (dm *DiagonalMatrix) Trace() float64 {
	nRows := dm.RowDimension()
	nCols := dm.ColumnDimension()
	if nRows != nCols {
		panic(nonSquareMatrixSimpleErrorf(nRows, nCols))
	}
	trace := 0.
	for i := 0; i < nRows; i++ {
		trace += dm.At(i, i)
	}
	return trace
}

type RealMatrixPreservingVisitorImpl struct {
	s func(int, int, int, int, int, int)
	v func(int, int, float64)
	e func() float64
}

func (drmpv *RealMatrixPreservingVisitorImpl) Start(rows, columns, startRow, endRow, startColumn, endColumn int) {
	drmpv.s(rows, columns, startRow, endRow, startColumn, endColumn)
}

func (drmpv *RealMatrixPreservingVisitorImpl) Visit(row, column int, value float64) {
	drmpv.v(row, column, value)
}

func (drmpv *RealMatrixPreservingVisitorImpl) End() float64 {
	return drmpv.e()
}

func (dm *DiagonalMatrix) Transpose() RealMatrix {
	nRows := dm.RowDimension()
	nCols := dm.ColumnDimension()
	if nRows != nCols {
		panic(nonSquareMatrixSimpleErrorf(nRows, nCols))
	}

	copy := new(DiagonalMatrix)
	copy.data = make([]float64, nCols)

	drmpv := new(RealMatrixPreservingVisitorImpl)
	drmpv.s = func(int, int, int, int, int, int) {}
	drmpv.v = func(row, column int, value float64) {
		copy.SetEntry(column, row, value)
	}

	drmpv.e = func() float64 {
		return 0
	}

	dm.WalkInOptimizedOrder(drmpv)

	return copy
}

func (dm *DiagonalMatrix) WalkInRowOrder(visitor RealMatrixPreservingVisitor) float64 {
	rows := dm.RowDimension()
	columns := dm.ColumnDimension()
	visitor.Start(rows, columns, 0, rows-1, 0, columns-1)
	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			visitor.Visit(i, j, dm.At(i, j))
		}
	}
	return visitor.End()
}

func (dm *DiagonalMatrix) WalkInRowOrderBounded(visitor RealMatrixPreservingVisitor, startRow, endRow, startColumn, endColumn int) float64 {
	checkSubMatrixIndex(dm, startRow, endRow, startColumn, endColumn)
	visitor.Start(dm.RowDimension(), dm.ColumnDimension(), startRow, endRow, startColumn, endColumn)
	for i := startRow; i <= endRow; i++ {
		for j := startColumn; j <= endColumn; j++ {
			visitor.Visit(i, j, dm.At(i, j))
		}
	}
	return visitor.End()
}

func (dm *DiagonalMatrix) WalkInUpdateRowOrder(visitor RealMatrixChangingVisitor) float64 {
	rows := dm.RowDimension()
	columns := dm.ColumnDimension()
	visitor.Start(rows, columns, 0, rows-1, 0, columns-1)
	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			dm.SetEntry(i, j, visitor.Visit(i, j, dm.At(i, j)))
		}
	}
	return visitor.End()
}

func (dm *DiagonalMatrix) WalkInUpdateRowOrderBounded(visitor RealMatrixChangingVisitor, startRow, endRow, startColumn, endColumn int) float64 {
	checkSubMatrixIndex(dm, startRow, endRow, startColumn, endColumn)
	visitor.Start(dm.RowDimension(), dm.ColumnDimension(), startRow, endRow, startColumn, endColumn)
	for i := startRow; i <= endRow; i++ {
		for j := startColumn; j <= endColumn; j++ {
			dm.SetEntry(i, j, visitor.Visit(i, j, dm.At(i, j)))
		}
	}
	return visitor.End()
}

func (dm *DiagonalMatrix) WalkInUpdateColumnOrder(visitor RealMatrixChangingVisitor) float64 {
	rows := dm.RowDimension()
	columns := dm.ColumnDimension()
	visitor.Start(rows, columns, 0, rows-1, 0, columns-1)
	for j := 0; j < columns; j++ {
		for i := 0; i < rows; i++ {
			dm.SetEntry(i, j, visitor.Visit(i, j, dm.At(i, j)))
		}
	}
	return visitor.End()
}

func (dm *DiagonalMatrix) WalkInColumnOrder(visitor RealMatrixPreservingVisitor) float64 {
	rows := dm.RowDimension()
	columns := dm.ColumnDimension()
	visitor.Start(rows, columns, 0, rows-1, 0, columns-1)
	for j := 0; j < columns; j++ {
		for i := 0; i < rows; i++ {
			visitor.Visit(i, j, dm.At(i, j))
		}
	}
	return visitor.End()
}

func (dm *DiagonalMatrix) WalkInUpdateColumnOrderBounded(visitor RealMatrixChangingVisitor, startRow, endRow, startColumn, endColumn int) float64 {
	checkSubMatrixIndex(dm, startRow, endRow, startColumn, endColumn)
	visitor.Start(dm.RowDimension(), dm.ColumnDimension(), startRow, endRow, startColumn, endColumn)
	for j := startColumn; j <= endColumn; j++ {
		for i := startRow; i <= endRow; i++ {
			dm.SetEntry(i, j, visitor.Visit(i, j, dm.At(i, j)))
		}
	}
	return visitor.End()
}

func (dm *DiagonalMatrix) WalkInColumnOrderBounded(visitor RealMatrixPreservingVisitor, startRow, endRow, startColumn, endColumn int) float64 {
	checkSubMatrixIndex(dm, startRow, endRow, startColumn, endColumn)
	visitor.Start(dm.RowDimension(), dm.ColumnDimension(), startRow, endRow, startColumn, endColumn)
	for j := startColumn; j <= endColumn; j++ {
		for i := startRow; i <= endRow; i++ {
			visitor.Visit(i, j, dm.At(i, j))
		}
	}
	return visitor.End()
}

func (dm *DiagonalMatrix) WalkInUpdateOptimizedOrder(visitor RealMatrixChangingVisitor) float64 {
	return dm.WalkInUpdateRowOrder(visitor)
}

func (dm *DiagonalMatrix) WalkInOptimizedOrder(visitor RealMatrixPreservingVisitor) float64 {
	return dm.WalkInRowOrder(visitor)
}

func (dm *DiagonalMatrix) WalkInUpdateOptimizedOrderBounded(visitor RealMatrixChangingVisitor, startRow, endRow, startColumn, endColumn int) float64 {
	return dm.WalkInUpdateRowOrderBounded(visitor, startRow, endRow, startColumn, endColumn)
}

func (dm *DiagonalMatrix) WalkInOptimizedOrderBounded(visitor RealMatrixPreservingVisitor, startRow, endRow, startColumn, endColumn int) float64 {
	return dm.WalkInRowOrderBounded(visitor, startRow, endRow, startColumn, endColumn)
}
