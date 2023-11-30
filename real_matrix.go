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
	"strconv"
)

type Matrix interface {
	/**
	 * Returns the number of rows in the matrix.
	 */
	RowDimension() int

	/**
	 * Returns the number of columns in the matrix.
	 */
	ColumnDimension() int
}

/**
 * Interface defining a real-valued matrix with basic algebraic operations.
 *
 * Matrix element indexing is 0-based -- e.g., At(0, 0)
 * returns the element in the first row, first column of the matrix.
 *
 */
type RealMatrix interface {
	Matrix
	/**
	 * Returns a (deep) copy of this.
	 */
	Copy() RealMatrix
	/**
	 * Returns the sum of this matrix and m.
	 */
	Add(m RealMatrix) RealMatrix
	/**
	 * Returns this matrix minus m.
	 */
	Subtract(m RealMatrix) RealMatrix
	/**
	 * Returns the result of adding d to each entry.
	 */
	ScalarAdd(d float64) RealMatrix

	/**
	 * Returns the result of multiplying each entry by d
	 */
	ScalarMultiply(d float64) RealMatrix
	/**
	 * Returns the result of postmultiplying by m.
	 */
	Multiply(m RealMatrix) RealMatrix
	/**
	 * Returns the result of premultiplying by m.
	 */
	PreMultiplyMatrix(m RealMatrix) RealMatrix

	/**
	 * Returns matrix entries as a two-dimensional array.
	 */
	Data() [][]float64

	/**
	 * Gets a submatrix. Rows and columns are indicated
	 * counting from 0 to n-1.
	 */
	SubMatrix(startRow, endRow, startColumn, endColumn int) RealMatrix

	/**
	 * Replace the submatrix starting at row, column using data in the
	 * input subMatrix array. Indexes are 0-based.
	 *
	 * Example:
	 * Starting with
	 * 1  2  3  4
	 * 5  6  7  8
	 * 9  0  1  2
	 *
	 * and subMatrix = {{3, 4} {5,6}},
	 * invoking setSubMatrix(subMatrix,1,1) will result in
	 * 1  2  3  4
	 * 5  3  4  8
	 * 9  5  6  2
	 */
	SetSubMatrix(subMatrix [][]float64, row, column int)

	/**
	 * Get the entries at the given row index as a row matrix.  Row indices start
	 * at 0.
	 */
	RowMatrixAt(row int) RealMatrix

	/**
	 * Sets the specified row of this matrix to the entries of
	 * the specified row matrix. Row indices start at 0.
	 */
	SetRowMatrix(row int, matrix RealMatrix)

	/**
	 * Get the entries at the given column index as a column matrix. Column
	 * indices start at 0.
	 */
	ColumnMatrixAt(column int) RealMatrix

	/**
	 * Sets the specified column of this matrix to the entries
	 * of the specified column matrix. Column indices start at 0.
	 */
	SetColumnMatrix(column int, matrix RealMatrix)

	/**
	 * Returns the entries at the given row index as a Row indices
	 * start at 0.
	 */
	RowVectorAt(row int) RealVector

	/**
	 * Sets the specified row of this matrix to the entries of
	 * the specified vector. Row indices start at 0.
	 */
	SetRowVector(row int, vector RealVector)

	/**
	 * Get the entries at the given column index as a  Column indices
	 * start at 0.
	 */
	ColumnVectorAt(column int) RealVector

	/**
	 * Sets the specified column of this matrix to the entries
	 * of the specified vector. Column indices start at 0.
	 */
	SetColumnVector(column int, vector RealVector)

	/**
	 * Get the entries at the given row index. Row indices start at 0.
	 */
	RowAt(row int) []float64

	/**
	 * Sets the specified row of this matrix to the entries
	 * of the specified array. Row indices start at 0.
	 */
	SetRow(row int, array []float64)

	/**
	 * Get the entries at the given column index as an array. Column indices
	 * start at 0.
	 */
	ColumnAt(column int) []float64

	/**
	 * Sets the specified column of this matrix to the entries
	 * of the specified array. Column indices start at 0.
	 */
	SetColumn(column int, array []float64)

	/**
	 * Get the entry in the specified row and column. Row and column indices
	 * start at 0.
	 */
	At(row, column int) float64

	/**
	 * Set the entry in the specified row and column. Row and column indices
	 * start at 0.
	 */
	SetEntry(row, column int, value float64)

	/**
	 * Adds (in place) the specified value to the specified entry of
	 * this matrix. Row and column indices start at 0.
	 */
	AddToEntry(row, column int, increment float64)

	/**
	 * Multiplies (in place) the specified entry of this matrix by the
	 * specified value. Row and column indices start at 0.
	 */
	MultiplyEntry(row, column int, factor float64)
	/**
	 * Returns the transpose of this matrix.
	 */
	Transpose() RealMatrix

	/**
	 * Returns the  trace of the matrix (the sum of the elements on the main diagonal).
	 */
	Trace() float64

	/**
	 * Returns the result of multiplying this by the slice v.
	 */
	Operate(v []float64) []float64

	/**
	 * Returns the result of multiplying this by the vector v.
	 */
	OperateVector(v RealVector) RealVector

	/**
	 * Returns the (row) vector result of premultiplying this by the slice v.
	 */
	PreMultiply(v []float64) []float64

	/**
	 * Returns the (row) vector result of premultiplying this by the vector v.
	 *
	 */
	PreMultiplyVector(v RealVector) RealVector

	/**
	 * Checks equality between this and any interface.
	 */
	Equals(object interface{}) bool

	/**
	 * Visit (and possibly change) all matrix entries in row order.
	 * Row order starts at upper left and iterating through all elements
	 * of a row from left to right before going to the leftmost element
	 * of the next row.
	 */
	WalkInUpdateRowOrder(visitor RealMatrixChangingVisitor) float64

	/**
	 * Visit (but don't change) all matrix entries in row order.
	 * Row order starts at upper left and iterating through all elements
	 * of a row from left to right before going to the leftmost element
	 * of the next row.
	 */
	WalkInRowOrder(visitor RealMatrixPreservingVisitor) float64

	/**
	 * Visit (and possibly change) some matrix entries in row order.
	 * Row order starts at upper left and iterating through all elements
	 * of a row from left to right before going to the leftmost element
	 * of the next row.
	 */
	WalkInUpdateRowOrderBounded(visitor RealMatrixChangingVisitor, startRow, endRow, startColumn, endColumn int) float64

	/**
	 * Visit (but don't change) some matrix entries in row order.
	 * Row order starts at upper left and iterating through all elements
	 * of a row from left to right before going to the leftmost element
	 * of the next row.
	 */
	WalkInRowOrderBounded(visitor RealMatrixPreservingVisitor, startRow, endRow, startColumn, endColumn int) float64

	/**
	 * Visit (and possibly change) all matrix entries in column order.
	 * Column order starts at upper left and iterating through all elements
	 * of a column from top to bottom before going to the topmost element
	 * of the next column.
	 */
	WalkInUpdateColumnOrder(visitor RealMatrixChangingVisitor) float64

	/**
	 * Visit (but don't change) all matrix entries in column order.
	 * Column order starts at upper left and iterating through all elements
	 * of a column from top to bottom before going to the topmost element
	 * of the next column.
	 */
	WalkInColumnOrder(visitor RealMatrixPreservingVisitor) float64

	/**
	 * Visit (and possibly change) some matrix entries in column order.
	 * Column order starts at upper left and iterating through all elements
	 * of a column from top to bottom before going to the topmost element
	 * of the next column.
	 */
	WalkInUpdateColumnOrderBounded(visitor RealMatrixChangingVisitor, startRow, endRow, startColumn, endColumn int) float64

	/**
	 * Visit (but don't change) some matrix entries in column order.
	 * Column order starts at upper left and iterating through all elements
	 * of a column from top to bottom before going to the topmost element
	 * of the next column.
	 */
	WalkInColumnOrderBounded(visitor RealMatrixPreservingVisitor, startRow, endRow, startColumn, endColumn int) float64

	/**
	 * Visit (and possibly change) all matrix entries using the fastest possible order.
	 * The fastest walking order depends on the exact matrix class. It may be
	 * different from traditional row or column orders.
	 */
	WalkInUpdateOptimizedOrder(visitor RealMatrixChangingVisitor) float64

	/**
	 * Visit (but don't change) all matrix entries using the fastest possible order.
	 * The fastest walking order depends on the exact matrix class. It may be
	 * different from traditional row or column orders.
	 */
	WalkInOptimizedOrder(visitor RealMatrixPreservingVisitor) float64

	/**
	 * Visit (and possibly change) some matrix entries using the fastest possible order.
	 * The fastest walking order depends on the exact matrix class. It may be
	 * different from traditional row or column orders.
	 */
	WalkInUpdateOptimizedOrderBounded(visitor RealMatrixChangingVisitor, startRow, endRow, startColumn, endColumn int) float64

	/**
	 * Visit (but don't change) some matrix entries using the fastest possible order.
	 * The fastest walking order depends on the exact matrix class. It may be
	 * different from traditional row or column orders.
	 */
	WalkInOptimizedOrderBounded(visitor RealMatrixPreservingVisitor, startRow, endRow, startColumn, endColumn int) float64
}

func NewRealMatrixWithDimension(rows, columns int) (RealMatrix, error) {
	if rows*columns <= 4096 {
		return NewArray2DRowRealMatrix(rows, columns)
	}

	return NewBlockRealMatrix(rows, columns)
}

func NewRealMatrixFromSlices(data [][]float64) (RealMatrix, error) {
	if data == nil || data[0] == nil {
		return nil, invalidArgumentSimpleErrorf()
	}

	if len(data)*len(data[0]) <= 4096 {
		return NewArray2DRowRealMatrixFromSlices(data, true)
	}

	return NewBlockRealMatrixFromSlices(data)
}

func NewRealIdentityMatrix(dimension int) (RealMatrix, error) {
	m, err := NewRealMatrixWithDimension(dimension, dimension)
	if err != nil {
		return nil, err
	}

	for i := 0; i < dimension; i++ {
		m.SetEntry(i, i, 1.0)
	}

	return m, nil
}

func NewRealMatrixWithDiagonal(diagonal []float64) (RealMatrix, error) {
	size := len(diagonal)
	m, err := NewRealMatrixWithDimension(size, size)
	if err != nil {
		return nil, err
	}
	for i := 0; i < size; i++ {
		m.SetEntry(i, i, diagonal[i])
	}

	return m, nil
}

func NewRealDiagonalMatrix(diagonal []float64) (RealMatrix, error) {
	return NewDiagonalMatrix(diagonal, true)
}

type matrixCopierPreservingVisitor struct {
	s func(int, int, int, int, int, int)
	v func(int, int, float64)
	e func() float64
}

func (mcpv *matrixCopierPreservingVisitor) Start(rows, columns, startRow, endRow, startColumn, endColumn int) {
	mcpv.s(rows, columns, startRow, endRow, startColumn, endColumn)
}

func (mcpv *matrixCopierPreservingVisitor) Visit(row, column int, value float64) {
	mcpv.v(row, column, value)
}

func (mcpv *matrixCopierPreservingVisitor) End() float64 {
	return mcpv.e()
}

/**
 * Copy a submatrix. Rows and columns are indicated counting from 0 to n-1.
 */
func CopySubMatrix(m RealMatrix, startRow, endRow, startColumn, endColumn int, destination [][]float64) {
	if err := checkSubMatrixIndex(m, startRow, endRow, startColumn, endColumn); err != nil {
		panic(err)
	}

	rowsCount := endRow + 1 - startRow
	columnsCount := endColumn + 1 - startColumn
	if (len(destination) < rowsCount) || (len(destination[0]) < columnsCount) {
		panic(matrixDimensionMismatchErrorf(len(destination), len(destination[0]), rowsCount, columnsCount))
	}

	for i := 1; i < rowsCount; i++ {
		if len(destination[i]) < columnsCount {
			panic(matrixDimensionMismatchErrorf(len(destination), len(destination[i]), rowsCount, columnsCount))
		}
	}

	for i := startRow; i <= endRow; i++ {
		for j := startColumn; j <= endColumn; j++ {
			destination[i-startRow][j-startColumn] = m.At(i, j)
		}
	}
}

/**
 * Copy a submatrix. Rows and columns are indicated counting from 0 to n-1.
 */
func CopySubMatrixFromIndices(m RealMatrix, selectedRows, selectedColumns []int, destination [][]float64) {
	if err := checkSubMatrixIndexFromIndices(m, selectedRows, selectedColumns); err != nil {
		panic(err)
	}

	nCols := len(selectedColumns)
	if (len(destination) < len(selectedRows)) || (len(destination[0]) < nCols) {
		panic(matrixDimensionMismatchErrorf(len(destination), len(destination[0]), len(selectedRows), len(selectedColumns)))
	}

	for i := 0; i < len(selectedRows); i++ {
		destinationI := destination[i]
		if len(destinationI) < nCols {
			panic(matrixDimensionMismatchErrorf(len(destination), len(destination[0]), len(selectedRows), len(selectedColumns)))
		}
		for j := 0; j < len(selectedColumns); j++ {
			destinationI[j] = m.At(selectedRows[i], selectedColumns[j])
		}
	}
}

type realMatrixChangingVisitorImpl struct {
	v func(int, int, float64) float64
}

func (drmcv realMatrixChangingVisitorImpl) Start(rows, columns, startRow, endRow, startColumn, endColumn int) {
}

func (drmcv realMatrixChangingVisitorImpl) Visit(row, column int, value float64) float64 {
	return drmcv.v(row, column, value)
}

func (drmcv realMatrixChangingVisitorImpl) End() float64 { return 0 }

/**
 * Gets a submatrix. Rows and columns are indicated counting from 0 to n-1.
 */
func SubMatrix(m RealMatrix, selectedRows, selectedColumns []int) RealMatrix {
	checkSubMatrixIndexFromIndices(m, selectedRows, selectedColumns)

	subMatrix, err := NewRealMatrixWithDimension(len(selectedRows), len(selectedColumns))
	if err != nil {
		panic(err)
	}

	var drmcv realMatrixChangingVisitorImpl
	drmcv.v = func(row, column int, value float64) float64 {
		return m.At(selectedRows[row], selectedColumns[column])
	}

	subMatrix.WalkInUpdateRowOrder(drmcv)
	return subMatrix
}

/**
 * Is this a square matrix?
 */
func IsSquare(m Matrix) bool {
	return m.ColumnDimension() == m.RowDimension()
}

/**
 * Returns the result of multiplying this with itself p times. Depending on the underlying storage, instability for high powers
 * might occur.
 */
func Power(m RealMatrix, p int) RealMatrix {
	if p < 0 {
		panic(notPositiveErrorf(not_positive_exponent, float64(p)))
	}

	if !IsSquare(m) {
		panic(nonSquareMatrixSimpleErrorf(m.RowDimension(), m.ColumnDimension()))
	}

	if p == 0 {
		mat, err := NewRealIdentityMatrix(m.RowDimension())
		if err != nil {
			panic(err)
		}
		return mat
	}

	if p == 1 {
		mat := m.Copy()
		return mat
	}

	power := p - 1

	/*
	 * Only log_2(p) operations is used by doing as follows:
	 * 5^214 = 5^128 * 5^64 * 5^16 * 5^4 * 5^2
	 *
	 * In general, the same approach is used for A^p.
	 */

	binaryRepresentation := strconv.FormatInt(int64(power), 2)
	nonZeroPositions := make([]int, 0)
	maxI := -1

	for i := 0; i < len(binaryRepresentation); i++ {
		if string(binaryRepresentation[i]) == `1` {
			pos := len(binaryRepresentation) - i - 1
			nonZeroPositions = append(nonZeroPositions, pos)

			// The positions are taken in turn, so maxI is only changed once
			if maxI == -1 {
				maxI = pos
			}
		}
	}

	results := make([]RealMatrix, maxI+1)

	results[0] = m.Copy()
	for i := 1; i <= maxI; i++ {
		results[i] = results[i-1].Multiply(results[i-1])
	}

	result := m.Copy()

	for _, i := range nonZeroPositions {
		result = result.Multiply(results[i])
	}

	return result
}

type normPreservingVisitorImpl struct {
	endRow               int
	columnSum, maxColSum float64
}

func (rmpvi *normPreservingVisitorImpl) Start(rows, columns, startRow, endRow, startColumn, endColumn int) {
	rmpvi.endRow = endRow
	rmpvi.columnSum = 0
	rmpvi.maxColSum = 0
}

func (rmpvi *normPreservingVisitorImpl) Visit(row, column int, value float64) {
	rmpvi.columnSum += math.Abs(value)
	if row == rmpvi.endRow {
		rmpvi.maxColSum = math.Max(rmpvi.maxColSum, rmpvi.columnSum)
		rmpvi.columnSum = 0
	}
}

func (rmpvi *normPreservingVisitorImpl) End() float64 {
	return rmpvi.maxColSum
}

/**
 * Returns the maximum absolute row sum norm of the matrix.
 */
func MatLInfNorm(m RealMatrix) float64 {
	return m.WalkInColumnOrder(new(normPreservingVisitorImpl))
}

type frobeniusNormPreservingVisitorImpl struct {
	sum float64
}

func (rmpvi *frobeniusNormPreservingVisitorImpl) Start(rows, columns, startRow, endRow, startColumn, endColumn int) {
	rmpvi.sum = 0
}

func (rmpvi *frobeniusNormPreservingVisitorImpl) Visit(row, column int, value float64) {
	rmpvi.sum += value * value
}

func (rmpvi *frobeniusNormPreservingVisitorImpl) End() float64 {
	return math.Sqrt(rmpvi.sum)
}

/**
 * Returns the Frobenius norm of the matrix.
 */
func MatFrobeniusNorm(m RealMatrix) float64 {
	return m.WalkInOptimizedOrder(new(frobeniusNormPreservingVisitorImpl))
}

/**
 * Compute the outer product between two vectors.
 */
func OuterProduct(vec1, vec2 RealVector) RealMatrix {

	m := vec1.Dimension()
	n := vec2.Dimension()
	// if (v instanceof SparseRealVector || this instanceof SparseRealVector) {
	//     product = new OpenMapRealMatrix(m, n);
	// }
	out, err := NewRealMatrixWithDimension(m, n)
	if err != nil {
		panic(err)
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			out.SetEntry(i, j, vec1.At(i)*vec2.At(j))
		}
	}

	return out
}
