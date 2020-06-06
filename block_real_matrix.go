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

const (
	BLOCK_SIZE int = 52
)

/**
 * Cache-friendly implementation of RealMatrix using a flat arrays to store
 * square blocks of the matrix.
 *
 * This implementation is specially designed to be cache-friendly. Square blocks are
 * stored as small arrays and allow efficient traversal of data both in row major direction
 * and columns major direction, one block at a time. This greatly increases performances
 * for algorithms that use crossed directions loops like multiplication or transposition.
 *
 * The size of square blocks is a static parameter. It may be tuned according to the cache
 * size of the target computer processor. As a rule of thumbs, it should be the largest
 * value that allows three blocks to be simultaneously cached (this is necessary for example
 * for matrix multiplication). The default value is to use 52x52 blocks which is well suited
 * for processors with 64k L1 cache (one block holds 2704 values or 21632 bytes). This value
 * could be lowered to 36x36 for processors with 32k L1 cache.
 *
 * The regular blocks represent BLOCK_SIZE x BLOCK_SIZE squares. Blocks
 * at right hand side and bottom side which may be smaller to fit matrix dimensions. The square
 * blocks are flattened in row major order in single dimension arrays which are therefore
 * BLOCK_SIZE<sup>2</sup> elements long for regular blocks. The blocks are themselves
 * organized in row major order.
 *
 * As an example, for a block size of 52x52, a 100x60 matrix would be stored in 4 blocks.
 * Block 0 would be a double[2704] array holding the upper left 52x52 square, block 1 would be
 * a double[416] array holding the upper right 52x8 rectangle, block 2 would be a double[2496]
 * array holding the lower left 48x52 rectangle and block 3 would be a double[384] array
 * holding the lower right 48x8 rectangle.
 *
 * The layout complexity overhead versus simple mapping of matrices to java
 * arrays is negligible for small matrices (about 1%). The gain from cache efficiency leads
 * to up to 3-fold improvements for matrices of moderate to large size.
 */
type BlockRealMatrix struct {
	blocks                                 [][]float64
	rows, columns, blockRows, blockColumns int
}

func NewBlockRealMatrix(rowDimension, columnDimension int) (*BlockRealMatrix, error) {
	if rowDimension < 1 {
		return nil, notStrictlyPositiveErrorf(float64(rowDimension))
	}
	if columnDimension < 1 {
		return nil, notStrictlyPositiveErrorf(float64(columnDimension))
	}

	ans := new(BlockRealMatrix)
	ans.rows = rowDimension
	ans.columns = columnDimension
	ans.blockRows = (rowDimension + BLOCK_SIZE - 1) / BLOCK_SIZE
	ans.blockColumns = (columnDimension + BLOCK_SIZE - 1) / BLOCK_SIZE
	ans.blocks = createBlocksLayout(rowDimension, columnDimension)
	return ans, nil
}

func NewBlockRealMatrixFromSlices(rawData [][]float64) (*BlockRealMatrix, error) {
	return NewBlockRealMatrixFromBlockData(len(rawData), len(rawData[0]), toBlocksLayout(rawData))
}

func NewBlockRealMatrixFromBlockData(rows, columns int, blockData [][]float64) (*BlockRealMatrix, error) {
	if rows < 1 {
		return nil, notStrictlyPositiveErrorf(float64(rows))
	}
	if columns < 1 {
		return nil, notStrictlyPositiveErrorf(float64(columns))
	}

	ans := new(BlockRealMatrix)
	ans.rows = rows
	ans.columns = columns

	// number of blocks
	ans.blockRows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE
	ans.blockColumns = (columns + BLOCK_SIZE - 1) / BLOCK_SIZE

	ans.blocks = blockData

	var index int
	for iBlock := 0; iBlock < ans.blockRows; iBlock++ {
		iHeight := ans.blockHeight(iBlock)
		for jBlock := 0; jBlock < ans.blockColumns; jBlock++ {
			if len(ans.blocks[index]) != iHeight*ans.blockWidth(jBlock) {
				return nil, dimensionsMismatchSimpleErrorf(len(ans.blocks[index]), iHeight*ans.blockWidth(jBlock))
			}

			index++
		}
	}

	return ans, nil
}

func toBlocksLayout(rawData [][]float64) [][]float64 {
	rows := len(rawData)
	columns := len(rawData[0])
	blockRows := (rows + BLOCK_SIZE - 1) / BLOCK_SIZE
	blockColumns := (columns + BLOCK_SIZE - 1) / BLOCK_SIZE

	// safety checks
	for i := 0; i < len(rawData); i++ {
		length := len(rawData[i])
		if length != columns {
			panic(dimensionsMismatchSimpleErrorf(columns, length))
		}
	}

	// convert array
	blocks := make([][]float64, blockRows*blockColumns)
	var blockIndex int
	for iBlock := 0; iBlock < blockRows; iBlock++ {
		pStart := iBlock * BLOCK_SIZE
		pEnd := int(math.Min(float64(pStart+BLOCK_SIZE), float64(rows)))
		iHeight := pEnd - pStart
		for jBlock := 0; jBlock < blockColumns; jBlock++ {
			qStart := jBlock * BLOCK_SIZE
			qEnd := int(math.Min(float64(qStart+BLOCK_SIZE), float64(columns)))
			jWidth := qEnd - qStart

			// allocate new block
			blocks[blockIndex] = make([]float64, iHeight*jWidth)

			// copy data
			var index int
			for p := pStart; p < pEnd; p++ {
				copy(blocks[blockIndex][index:index+jWidth], rawData[p][qStart:qStart+jWidth])
				index += jWidth
			}
			blockIndex++
		}
	}

	return blocks
}

func createBlocksLayout(rows, columns int) [][]float64 {
	blockRows := (rows + BLOCK_SIZE - 1) / BLOCK_SIZE
	blockColumns := (columns + BLOCK_SIZE - 1) / BLOCK_SIZE

	blocks := make([][]float64, blockRows*blockColumns)
	var blockIndex int
	for iBlock := 0; iBlock < blockRows; iBlock++ {
		pStart := iBlock * BLOCK_SIZE
		pEnd := int(math.Min(float64(pStart+BLOCK_SIZE), float64(rows)))
		iHeight := pEnd - pStart
		for jBlock := 0; jBlock < blockColumns; jBlock++ {
			qStart := jBlock * BLOCK_SIZE
			qEnd := int(math.Min(float64(qStart+BLOCK_SIZE), float64(columns)))
			jWidth := qEnd - qStart
			blocks[blockIndex] = make([]float64, iHeight*jWidth)
			blockIndex++
		}
	}

	return blocks
}

func (brm *BlockRealMatrix) Copy() RealMatrix {
	c := new(BlockRealMatrix)
	c.rows = brm.rows
	c.columns = brm.columns
	c.blockRows = (brm.rows + BLOCK_SIZE - 1) / BLOCK_SIZE
	c.blockColumns = (brm.columns + BLOCK_SIZE - 1) / BLOCK_SIZE
	c.blocks = createBlocksLayout(brm.rows, brm.columns)
	for i := 0; i < len(brm.blocks); i++ {
		copy(c.blocks[i], brm.blocks[i])
	}

	return c
}

func (brm *BlockRealMatrix) Add(mat RealMatrix) RealMatrix {
	if err := checkAdditionCompatible(brm, mat); err != nil {
		panic(err)
	}
	out, err := NewBlockRealMatrix(brm.rows, brm.columns)
	if err != nil {
		panic(err)
	}
	if m, ok := mat.(*BlockRealMatrix); ok {
		// perform addition block-wise, to ensure good cache behavior
		for blockIndex := 0; blockIndex < len(out.blocks); blockIndex++ {
			tBlock := brm.blocks[blockIndex]
			mBlock := m.blocks[blockIndex]
			for k := 0; k < len(out.blocks[blockIndex]); k++ {
				out.blocks[blockIndex][k] = tBlock[k] + mBlock[k]
			}
		}

	} else {
		// perform addition block-wise, to ensure good cache behavior
		blockIndex := 0
		for iBlock := 0; iBlock < out.blockRows; iBlock++ {
			for jBlock := 0; jBlock < out.blockColumns; jBlock++ {
				// perform addition on the current block
				tBlock := brm.blocks[blockIndex]
				pStart := iBlock * BLOCK_SIZE
				pEnd := int(math.Min(float64(pStart+BLOCK_SIZE), float64(brm.rows)))
				qStart := jBlock * BLOCK_SIZE
				qEnd := int(math.Min(float64(qStart+BLOCK_SIZE), float64(brm.columns)))
				k := 0
				for p := pStart; p < pEnd; p++ {
					for q := qStart; q < qEnd; q++ {
						out.blocks[blockIndex][k] = tBlock[k] + mat.At(p, q)
						k++
					}
				}
				// go to next block
				blockIndex++
			}
		}
	}

	return out
}

func (brm *BlockRealMatrix) Subtract(mat RealMatrix) RealMatrix {
	if err := checkAdditionCompatible(brm, mat); err != nil {
		panic(err)
	}
	out, err := NewBlockRealMatrix(brm.rows, brm.columns)
	if err != nil {
		panic(err)
	}

	if m, ok := mat.(*BlockRealMatrix); ok {
		// perform addition block-wise, to ensure good cache behavior
		for blockIndex := 0; blockIndex < len(out.blocks); blockIndex++ {
			tBlock := brm.blocks[blockIndex]
			mBlock := m.blocks[blockIndex]
			for k := 0; k < len(out.blocks[blockIndex]); k++ {
				out.blocks[blockIndex][k] = tBlock[k] - mBlock[k]
			}
		}
	} else {
		// perform addition block-wise, to ensure good cache behavior
		blockIndex := 0
		for iBlock := 0; iBlock < out.blockRows; iBlock++ {
			for jBlock := 0; jBlock < out.blockColumns; jBlock++ {
				// perform addition on the current block
				tBlock := brm.blocks[blockIndex]
				pStart := iBlock * BLOCK_SIZE
				pEnd := int(math.Min(float64(pStart+BLOCK_SIZE), float64(brm.rows)))
				qStart := jBlock * BLOCK_SIZE
				qEnd := int(math.Min(float64(qStart+BLOCK_SIZE), float64(brm.columns)))
				k := 0
				for p := pStart; p < pEnd; p++ {
					for q := qStart; q < qEnd; q++ {
						out.blocks[blockIndex][k] = tBlock[k] - mat.At(p, q)
						k++
					}
				}
				// go to next block
				blockIndex++
			}
		}
	}

	return out
}

func (brm *BlockRealMatrix) ScalarAdd(d float64) RealMatrix {

	out, err := NewBlockRealMatrix(brm.rows, brm.columns)
	if err != nil {
		panic(err)
	}

	// perform subtraction block-wise, to ensure good cache behavior
	for blockIndex := 0; blockIndex < len(out.blocks); blockIndex++ {
		tBlock := brm.blocks[blockIndex]
		for k := 0; k < len(out.blocks[blockIndex]); k++ {
			out.blocks[blockIndex][k] = tBlock[k] + d
		}
	}

	return out
}

func (brm *BlockRealMatrix) ScalarMultiply(d float64) RealMatrix {

	out, err := NewBlockRealMatrix(brm.rows, brm.columns)
	if err != nil {
		panic(err)
	}
	// perform subtraction block-wise, to ensure good cache behavior
	for blockIndex := 0; blockIndex < len(out.blocks); blockIndex++ {
		tBlock := brm.blocks[blockIndex]
		for k := 0; k < len(out.blocks[blockIndex]); k++ {
			out.blocks[blockIndex][k] = tBlock[k] * d
		}
	}

	return out
}

func (brm *BlockRealMatrix) Multiply(mat RealMatrix) RealMatrix {
	if err := checkMultiplicationCompatible(brm, mat); err != nil {
		panic(err)
	}

	if m, ok := mat.(*BlockRealMatrix); ok {
		out, err := NewBlockRealMatrix(brm.rows, m.columns)
		if err != nil {
			panic(err)
		}
		// perform multiplication block-wise, to ensure good cache behavior
		var blockIndex int
		for iBlock := 0; iBlock < out.blockRows; iBlock++ {

			pStart := iBlock * BLOCK_SIZE

			pEnd := int(math.Min(float64(pStart+BLOCK_SIZE), float64(brm.rows)))

			for jBlock := 0; jBlock < out.blockColumns; jBlock++ {
				jWidth := out.blockWidth(jBlock)
				jWidth2 := jWidth + jWidth
				jWidth3 := jWidth2 + jWidth
				jWidth4 := jWidth3 + jWidth

				for kBlock := 0; kBlock < brm.blockColumns; kBlock++ {
					kWidth := brm.blockWidth(kBlock)
					tBlock := brm.blocks[iBlock*brm.blockColumns+kBlock]
					mBlock := m.blocks[kBlock*m.blockColumns+jBlock]
					var k int
					for p := pStart; p < pEnd; p++ {
						lStart := (p - pStart) * kWidth
						lEnd := lStart + kWidth
						for nStart := 0; nStart < jWidth; nStart++ {
							var sum float64
							l := lStart
							n := nStart
							for l < lEnd-3 {
								sum += tBlock[l]*mBlock[n] +
									tBlock[l+1]*mBlock[n+jWidth] +
									tBlock[l+2]*mBlock[n+jWidth2] +
									tBlock[l+3]*mBlock[n+jWidth3]
								l += 4
								n += jWidth4
							}
							for l < lEnd {
								sum += tBlock[l] * mBlock[n]
								l++
								n += jWidth
							}
							out.blocks[blockIndex][k] += sum
							k++
						}
					}
				}
				blockIndex++
			}
		}

		return out
	} else {

		out, err := NewBlockRealMatrix(brm.rows, mat.ColumnDimension())
		if err != nil {
			panic(err)
		}

		var blockIndex int
		for iBlock := 0; iBlock < out.blockRows; iBlock++ {
			pStart := iBlock * BLOCK_SIZE
			pEnd := int(math.Min(float64(pStart+BLOCK_SIZE), float64(brm.rows)))

			for jBlock := 0; jBlock < out.blockColumns; jBlock++ {
				qStart := jBlock * BLOCK_SIZE
				qEnd := int(math.Min(float64(qStart+BLOCK_SIZE), float64(mat.ColumnDimension())))

				// perform multiplication on current block
				for kBlock := 0; kBlock < brm.blockColumns; kBlock++ {
					kWidth := brm.blockWidth(kBlock)
					tBlock := brm.blocks[iBlock*brm.blockColumns+kBlock]
					rStart := kBlock * BLOCK_SIZE
					var k int
					for p := pStart; p < pEnd; p++ {
						lStart := (p - pStart) * kWidth
						lEnd := lStart + kWidth
						for q := qStart; q < qEnd; q++ {
							var sum float64
							r := rStart
							for l := lStart; l < lEnd; l++ {
								sum += tBlock[l] * mat.At(r, q)
								r++
							}
							out.blocks[blockIndex][k] += sum
							k++
						}
					}
				}
				// go to next block
				blockIndex++
			}
		}

		return out
	}
}

func (brm *BlockRealMatrix) Data() [][]float64 {
	data := make([][]float64, brm.RowDimension())
	lastColumns := brm.columns - (brm.blockColumns-1)*BLOCK_SIZE

	for iBlock := 0; iBlock < brm.blockRows; iBlock++ {
		pStart := iBlock * BLOCK_SIZE
		pEnd := int(math.Min(float64(pStart+BLOCK_SIZE), float64(brm.rows)))
		var regularPos, lastPos int
		for p := pStart; p < pEnd; p++ {
			data[p] = make([]float64, brm.ColumnDimension())
			blockIndex := iBlock * brm.blockColumns
			dataPos := 0
			for jBlock := 0; jBlock < brm.blockColumns-1; jBlock++ {
				copy(data[p][dataPos:dataPos+BLOCK_SIZE], brm.blocks[blockIndex][regularPos:regularPos+BLOCK_SIZE])
				blockIndex++
				dataPos += BLOCK_SIZE
			}
			copy(data[p][dataPos:dataPos+lastColumns], brm.blocks[blockIndex][lastPos:lastPos+lastColumns])
			regularPos += BLOCK_SIZE
			lastPos += lastColumns
		}
	}

	return data
}

func (brm *BlockRealMatrix) Trace() float64 {
	nRows := brm.RowDimension()
	nCols := brm.ColumnDimension()
	if nRows != nCols {
		panic(nonSquareMatrixSimpleErrorf(nRows, nCols))
	}
	trace := 0.
	for i := 0; i < nRows; i++ {
		trace += brm.At(i, i)
	}
	return trace
}

func (brm *BlockRealMatrix) SubMatrix(startRow, endRow, startColumn, endColumn int) RealMatrix {
	if err := checkSubMatrixIndex(brm, startRow, endRow, startColumn, endColumn); err != nil {
		panic(err)
	}

	// create the output matrix
	out, err := NewBlockRealMatrix(endRow-startRow+1, endColumn-startColumn+1)
	if err != nil {
		panic(err)
	}

	// compute blocks shifts
	blockStartRow := startRow / BLOCK_SIZE
	rowsShift := startRow % BLOCK_SIZE
	blockStartColumn := startColumn / BLOCK_SIZE
	columnsShift := startColumn % BLOCK_SIZE

	// perform extraction block-wise, to ensure good cache behavior
	pBlock := blockStartRow
	for iBlock := 0; iBlock < out.blockRows; iBlock++ {
		iHeight := out.blockHeight(iBlock)
		qBlock := blockStartColumn
		for jBlock := 0; jBlock < out.blockColumns; jBlock++ {
			jWidth := out.blockWidth(jBlock)

			// handle one block of the output matrix
			outIndex := iBlock*out.blockColumns + jBlock
			index := pBlock*brm.blockColumns + qBlock
			width := brm.blockWidth(qBlock)

			heightExcess := iHeight + rowsShift - BLOCK_SIZE
			widthExcess := jWidth + columnsShift - BLOCK_SIZE
			if heightExcess > 0 {
				// the submatrix block spans on two blocks rows from the original matrix
				if widthExcess > 0 {
					// the submatrix block spans on two blocks columns from the original matrix
					width2 := brm.blockWidth(qBlock + 1)
					brm.copyBlockPart(brm.blocks[index], width, rowsShift, BLOCK_SIZE, columnsShift, BLOCK_SIZE, out.blocks[outIndex], jWidth, 0, 0)
					brm.copyBlockPart(brm.blocks[index+1], width2, rowsShift, BLOCK_SIZE, 0, widthExcess, out.blocks[outIndex], jWidth, 0, jWidth-widthExcess)
					brm.copyBlockPart(brm.blocks[index+brm.blockColumns], width, 0, heightExcess, columnsShift, BLOCK_SIZE, out.blocks[outIndex], jWidth, iHeight-heightExcess, 0)
					brm.copyBlockPart(brm.blocks[index+brm.blockColumns+1], width2, 0, heightExcess, 0, widthExcess, out.blocks[outIndex], jWidth, iHeight-heightExcess, jWidth-widthExcess)
				} else {
					// the submatrix block spans on one block column from the original matrix
					brm.copyBlockPart(brm.blocks[index], width, rowsShift, BLOCK_SIZE, columnsShift, jWidth+columnsShift, out.blocks[outIndex], jWidth, 0, 0)
					brm.copyBlockPart(brm.blocks[index+brm.blockColumns], width, 0, heightExcess, columnsShift, jWidth+columnsShift, out.blocks[outIndex], jWidth, iHeight-heightExcess, 0)
				}
			} else {
				// the submatrix block spans on one block row from the original matrix
				if widthExcess > 0 {
					// the submatrix block spans on two blocks columns from the original matrix
					width2 := brm.blockWidth(qBlock + 1)
					brm.copyBlockPart(brm.blocks[index], width, rowsShift, iHeight+rowsShift, columnsShift, BLOCK_SIZE, out.blocks[outIndex], jWidth, 0, 0)
					brm.copyBlockPart(brm.blocks[index+1], width2, rowsShift, iHeight+rowsShift, 0, widthExcess, out.blocks[outIndex], jWidth, 0, jWidth-widthExcess)
				} else {
					// the submatrix block spans on one block column from the original matrix
					brm.copyBlockPart(brm.blocks[index], width, rowsShift, iHeight+rowsShift, columnsShift, jWidth+columnsShift, out.blocks[outIndex], jWidth, 0, 0)
				}
			}
			qBlock++
		}
		pBlock++
	}

	return out
}

func (brm *BlockRealMatrix) copyBlockPart(srcBlock []float64, srcWidth, srcStartRow, srcEndRow, srcStartColumn, srcEndColumn int, dstBlock []float64, dstWidth, dstStartRow, dstStartColumn int) {
	length := srcEndColumn - srcStartColumn
	srcPos := srcStartRow*srcWidth + srcStartColumn
	dstPos := dstStartRow*dstWidth + dstStartColumn
	for srcRow := srcStartRow; srcRow < srcEndRow; srcRow++ {
		copy(dstBlock[dstPos:dstPos+length], srcBlock[srcPos:srcPos+length])
		srcPos += srcWidth
		dstPos += dstWidth
	}
}

func (brm *BlockRealMatrix) SetSubMatrix(subMatrix [][]float64, row, column int) {

	if subMatrix == nil {
		panic(invalidArgumentSimpleErrorf())
	}

	refLength := len(subMatrix[0])
	if refLength == 0 {
		panic(noDataErrorf(at_least_one_column))
	}
	endRow := row + len(subMatrix) - 1
	endColumn := column + refLength - 1
	if err := checkSubMatrixIndex(brm, row, endRow, column, endColumn); err != nil {
		panic(err)
	}
	for _, subRow := range subMatrix {
		if len(subRow) != refLength {
			panic(dimensionsMismatchSimpleErrorf(refLength, len(subRow)))
		}
	}

	// compute blocks bounds
	blockStartRow := row / BLOCK_SIZE
	blockEndRow := (endRow + BLOCK_SIZE) / BLOCK_SIZE
	blockStartColumn := column / BLOCK_SIZE
	blockEndColumn := (endColumn + BLOCK_SIZE) / BLOCK_SIZE

	// perform copy block-wise, to ensure good cache behavior
	for iBlock := blockStartRow; iBlock < blockEndRow; iBlock++ {
		iHeight := brm.blockHeight(iBlock)
		firstRow := iBlock * BLOCK_SIZE
		iStart := int(math.Max(float64(row), float64(firstRow)))
		iEnd := int(math.Min(float64(endRow+1), float64(firstRow+iHeight)))

		for jBlock := blockStartColumn; jBlock < blockEndColumn; jBlock++ {
			jWidth := brm.blockWidth(jBlock)
			firstColumn := jBlock * BLOCK_SIZE
			jStart := int(math.Max(float64(column), float64(firstColumn)))
			jEnd := int(math.Min(float64(endColumn+1), float64(firstColumn+jWidth)))
			jLength := jEnd - jStart

			// handle one block, row by row
			for i := iStart; i < iEnd; i++ {
				pos := (i-firstRow)*jWidth + (jStart - firstColumn)
				copy(brm.blocks[iBlock*brm.blockColumns+jBlock][pos:pos+jLength], subMatrix[i-row][jStart-column:(jStart-column)+jLength])
			}

		}
	}
}

func (brm *BlockRealMatrix) RowMatrixAt(row int) RealMatrix {
	if err := checkRowIndex(brm, row); err != nil {
		panic(err)
	}
	out, err := NewBlockRealMatrix(1, brm.columns)
	if err != nil {
		panic(err)
	}

	// perform copy block-wise, to ensure good cache behavior
	iBlock := row / BLOCK_SIZE
	iRow := row - iBlock*BLOCK_SIZE
	outBlockIndex := 0
	outIndex := 0
	for jBlock := 0; jBlock < brm.blockColumns; jBlock++ {
		jWidth := brm.blockWidth(jBlock)
		available := len(out.blocks[outBlockIndex]) - outIndex
		if jWidth > available {
			copy(out.blocks[outBlockIndex][outIndex:outIndex+available], brm.blocks[iBlock*brm.blockColumns+jBlock][iRow*jWidth:(iRow*jWidth)+available])
			outBlockIndex++
			copy(out.blocks[outBlockIndex][0:jWidth-available], brm.blocks[iBlock*brm.blockColumns+jBlock][iRow*jWidth:(iRow*jWidth)+(jWidth-available)])
			outIndex = jWidth - available
		} else {
			copy(out.blocks[outBlockIndex][outIndex:outIndex+jWidth], brm.blocks[iBlock*brm.blockColumns+jBlock][iRow*jWidth:(iRow*jWidth)+jWidth])
			outIndex += jWidth
		}
	}

	return out
}

func (brm *BlockRealMatrix) SetRowMatrix(row int, mat RealMatrix) {
	if err := checkRowIndex(brm, row); err != nil {
		panic(err)
	}
	nCols := brm.ColumnDimension()
	if (mat.RowDimension() != 1) || (mat.ColumnDimension() != nCols) {
		panic(matrixDimensionMismatchErrorf(mat.RowDimension(), mat.ColumnDimension(), 1, nCols))
	}
	if m, ok := mat.(*BlockRealMatrix); ok {
		// perform copy block-wise, to ensure good cache behavior
		iBlock := row / BLOCK_SIZE
		iRow := row - iBlock*BLOCK_SIZE
		mBlockIndex := 0
		mIndex := 0
		for jBlock := 0; jBlock < brm.blockColumns; jBlock++ {
			jWidth := brm.blockWidth(jBlock)

			available := len(m.blocks[mBlockIndex]) - mIndex
			if jWidth > available {
				copy(brm.blocks[iBlock*brm.blockColumns+jBlock][iRow*jWidth:(iRow*jWidth)+available], m.blocks[mBlockIndex][mIndex:mIndex+available])
				mBlockIndex++
				copy(brm.blocks[iBlock*brm.blockColumns+jBlock][iRow*jWidth:(iRow*jWidth)+(jWidth-available)], m.blocks[mBlockIndex][0:jWidth-available])
				mIndex = jWidth - available
			} else {
				copy(brm.blocks[iBlock*brm.blockColumns+jBlock][iRow*jWidth:(iRow*jWidth)+jWidth], m.blocks[mBlockIndex][mIndex:mIndex+jWidth])
				mIndex += jWidth
			}
		}
	} else {

		for i := 0; i < nCols; i++ {
			brm.SetEntry(row, i, mat.At(0, i))
		}
	}
}

func (brm *BlockRealMatrix) ColumnMatrixAt(column int) RealMatrix {
	if err := checkColumnIndex(brm, column); err != nil {
		panic(err)
	}
	out, err := NewBlockRealMatrix(brm.rows, 1)
	if err != nil {
		panic(err)
	}

	// perform copy block-wise, to ensure good cache behavior
	jBlock := column / BLOCK_SIZE
	jColumn := column - jBlock*BLOCK_SIZE
	jWidth := brm.blockWidth(jBlock)
	outBlockIndex := 0
	outIndex := 0
	for iBlock := 0; iBlock < brm.blockRows; iBlock++ {
		iHeight := brm.blockHeight(iBlock)
		for i := 0; i < iHeight; i++ {
			if outIndex >= len(out.blocks[outBlockIndex]) {
				outBlockIndex++
				outIndex = 0
			}
			out.blocks[outBlockIndex][outIndex] = brm.blocks[iBlock*brm.blockColumns+jBlock][i*jWidth+jColumn]
			outIndex++
		}
	}

	return out
}

func (brm *BlockRealMatrix) SetColumnMatrix(column int, mat RealMatrix) {
	if err := checkColumnIndex(brm, column); err != nil {
		panic(err)
	}

	nRows := brm.RowDimension()
	if (mat.RowDimension() != nRows) || (mat.ColumnDimension() != 1) {
		panic(matrixDimensionMismatchErrorf(mat.RowDimension(), mat.ColumnDimension(), nRows, 1))
	}
	if m, ok := mat.(*BlockRealMatrix); ok {
		// perform copy block-wise, to ensure good cache behavior
		jBlock := column / BLOCK_SIZE
		jColumn := column - jBlock*BLOCK_SIZE
		jWidth := brm.blockWidth(jBlock)
		mBlockIndex := 0
		mIndex := 0
		for iBlock := 0; iBlock < brm.blockRows; iBlock++ {
			iHeight := brm.blockHeight(iBlock)
			for i := 0; i < iHeight; i++ {
				if mIndex >= len(m.blocks[mBlockIndex]) {
					mBlockIndex++
					mIndex = 0
				}
				brm.blocks[iBlock*brm.blockColumns+jBlock][i*jWidth+jColumn] = m.blocks[mBlockIndex][mIndex]
				mIndex++
			}
		}
	} else {
		for i := 0; i < nRows; i++ {
			brm.SetEntry(i, column, mat.At(i, 0))
		}
	}
}

func (brm *BlockRealMatrix) RowVectorAt(row int) RealVector {
	if err := checkRowIndex(brm, row); err != nil {
		panic(err)
	}
	outData := make([]float64, brm.columns)

	// perform copy block-wise, to ensure good cache behavior
	iBlock := row / BLOCK_SIZE
	iRow := row - iBlock*BLOCK_SIZE
	outIndex := 0
	for jBlock := 0; jBlock < brm.blockColumns; jBlock++ {
		jWidth := brm.blockWidth(jBlock)
		copy(outData[outIndex:outIndex+jWidth], brm.blocks[iBlock*brm.blockColumns+jBlock][iRow*jWidth:(iRow*jWidth)+jWidth])
		outIndex += jWidth
	}

	v := new(ArrayRealVector)
	v.data = outData
	return v
}

func (brm *BlockRealMatrix) SetRowVector(row int, vec RealVector) {
	if vec, ok := vec.(*ArrayRealVector); ok {
		brm.SetRow(row, vec.DataRef())
	} else {
		if err := checkRowIndex(brm, row); err != nil {
			panic(err)
		}
		nCols := brm.ColumnDimension()
		if vec.Dimension() != nCols {
			panic(matrixDimensionMismatchErrorf(1, vec.Dimension(), 1, nCols))
		}
		for i := 0; i < nCols; i++ {
			brm.SetEntry(row, i, vec.At(i))
		}
	}
}

func (brm *BlockRealMatrix) ColumnVectorAt(column int) RealVector {
	if err := checkColumnIndex(brm, column); err != nil {
		panic(err)
	}
	outData := make([]float64, brm.rows)

	// perform copy block-wise, to ensure good cache behavior
	jBlock := column / BLOCK_SIZE
	jColumn := column - jBlock*BLOCK_SIZE
	jWidth := brm.blockWidth(jBlock)
	outIndex := 0
	for iBlock := 0; iBlock < brm.blockRows; iBlock++ {
		iHeight := brm.blockHeight(iBlock)
		for i := 0; i < iHeight; i++ {
			outData[outIndex] = brm.blocks[iBlock*brm.blockColumns+jBlock][i*jWidth+jColumn]
			outIndex++
		}
	}

	v := new(ArrayRealVector)
	v.data = outData
	return v
}

func (brm *BlockRealMatrix) SetColumnVector(column int, vec RealVector) {
	if vec, ok := vec.(*ArrayRealVector); ok {
		brm.SetColumn(column, vec.DataRef())
	} else {
		if err := checkColumnIndex(brm, column); err != nil {
			panic(err)
		}
		nRows := brm.RowDimension()
		if vec.Dimension() != nRows {
			panic(matrixDimensionMismatchErrorf(vec.Dimension(), 1, nRows, 1))
		}
		for i := 0; i < nRows; i++ {
			brm.SetEntry(i, column, vec.At(i))
		}
	}
}

func (brm *BlockRealMatrix) RowAt(row int) []float64 {
	if err := checkRowIndex(brm, row); err != nil {
		panic(err)
	}
	out := make([]float64, brm.columns)

	// perform copy block-wise, to ensure good cache behavior
	iBlock := row / BLOCK_SIZE
	iRow := row - iBlock*BLOCK_SIZE
	outIndex := 0
	for jBlock := 0; jBlock < brm.blockColumns; jBlock++ {
		jWidth := brm.blockWidth(jBlock)
		copy(out[outIndex:outIndex+jWidth], brm.blocks[iBlock*brm.blockColumns+jBlock][iRow*jWidth:(iRow*jWidth)+jWidth])
		outIndex += jWidth
	}

	return out
}

func (brm *BlockRealMatrix) SetRow(row int, array []float64) {
	if err := checkRowIndex(brm, row); err != nil {
		panic(err)
	}
	nCols := brm.ColumnDimension()
	if len(array) != nCols {
		panic(matrixDimensionMismatchErrorf(1, len(array), 1, nCols))
	}

	// perform copy block-wise, to ensure good cache behavior
	iBlock := row / BLOCK_SIZE
	iRow := row - iBlock*BLOCK_SIZE
	outIndex := 0
	for jBlock := 0; jBlock < brm.blockColumns; jBlock++ {
		jWidth := brm.blockWidth(jBlock)
		copy(brm.blocks[iBlock*brm.blockColumns+jBlock][iRow*jWidth:(iRow*jWidth)+jWidth], array[outIndex:outIndex+jWidth])
		outIndex += jWidth
	}
}

func (brm *BlockRealMatrix) ColumnAt(column int) []float64 {
	if err := checkColumnIndex(brm, column); err != nil {
		panic(err)
	}
	out := make([]float64, brm.rows)

	// perform copy block-wise, to ensure good cache behavior
	jBlock := column / BLOCK_SIZE
	jColumn := column - jBlock*BLOCK_SIZE
	jWidth := brm.blockWidth(jBlock)
	outIndex := 0
	for iBlock := 0; iBlock < brm.blockRows; iBlock++ {
		iHeight := brm.blockHeight(iBlock)
		for i := 0; i < iHeight; i++ {
			out[outIndex] = brm.blocks[iBlock*brm.blockColumns+jBlock][i*jWidth+jColumn]
			outIndex++
		}
	}

	return out
}

func (brm *BlockRealMatrix) SetColumn(column int, array []float64) {
	if err := checkColumnIndex(brm, column); err != nil {
		panic(err)
	}
	nRows := brm.RowDimension()
	if len(array) != nRows {
		panic(matrixDimensionMismatchErrorf(len(array), 1, nRows, 1))
	}

	// perform copy block-wise, to ensure good cache behavior
	jBlock := column / BLOCK_SIZE
	jColumn := column - jBlock*BLOCK_SIZE
	jWidth := brm.blockWidth(jBlock)
	outIndex := 0
	for iBlock := 0; iBlock < brm.blockRows; iBlock++ {
		iHeight := brm.blockHeight(iBlock)
		for i := 0; i < iHeight; i++ {
			brm.blocks[iBlock*brm.blockColumns+jBlock][i*jWidth+jColumn] = array[outIndex]
			outIndex++
		}
	}
}

func (brm *BlockRealMatrix) At(row, column int) float64 {
	if err := checkMatrixIndex(brm, row, column); err != nil {
		panic(err)
	}
	iBlock := row / BLOCK_SIZE
	jBlock := column / BLOCK_SIZE
	k := (row-iBlock*BLOCK_SIZE)*brm.blockWidth(jBlock) + (column - jBlock*BLOCK_SIZE)
	return brm.blocks[iBlock*brm.blockColumns+jBlock][k]
}

func (brm *BlockRealMatrix) SetEntry(row, column int, value float64) {
	if err := checkMatrixIndex(brm, row, column); err != nil {
		panic(err)
	}
	iBlock := row / BLOCK_SIZE
	jBlock := column / BLOCK_SIZE
	k := (row-iBlock*BLOCK_SIZE)*brm.blockWidth(jBlock) + (column - jBlock*BLOCK_SIZE)
	brm.blocks[iBlock*brm.blockColumns+jBlock][k] = value
}

func (brm *BlockRealMatrix) AddToEntry(row, column int, increment float64) {
	if err := checkMatrixIndex(brm, row, column); err != nil {
		panic(err)
	}
	iBlock := row / BLOCK_SIZE
	jBlock := column / BLOCK_SIZE
	k := (row-iBlock*BLOCK_SIZE)*brm.blockWidth(jBlock) + (column - jBlock*BLOCK_SIZE)
	brm.blocks[iBlock*brm.blockColumns+jBlock][k] += increment
}

func (brm *BlockRealMatrix) MultiplyEntry(row, column int, factor float64) {
	if err := checkMatrixIndex(brm, row, column); err != nil {
		panic(err)
	}
	iBlock := row / BLOCK_SIZE
	jBlock := column / BLOCK_SIZE
	k := (row-iBlock*BLOCK_SIZE)*brm.blockWidth(jBlock) + (column - jBlock*BLOCK_SIZE)
	brm.blocks[iBlock*brm.blockColumns+jBlock][k] *= factor
}

func (brm *BlockRealMatrix) Transpose() RealMatrix {
	nRows := brm.RowDimension()
	nCols := brm.ColumnDimension()
	copy := new(BlockRealMatrix)
	copy.rows = nCols
	copy.columns = nRows
	copy.blockRows = (nCols + BLOCK_SIZE - 1) / BLOCK_SIZE
	copy.blockColumns = (nRows + BLOCK_SIZE - 1) / BLOCK_SIZE
	copy.blocks = createBlocksLayout(nCols, nRows)

	// perform transpose block-wise, to ensure good cache behavior
	blockIndex := 0
	for iBlock := 0; iBlock < brm.blockColumns; iBlock++ {
		for jBlock := 0; jBlock < brm.blockRows; jBlock++ {
			// transpose current block

			pStart := iBlock * BLOCK_SIZE
			pEnd := int(math.Min(float64(pStart+BLOCK_SIZE), float64(brm.columns)))
			qStart := jBlock * BLOCK_SIZE
			qEnd := int(math.Min(float64(qStart+BLOCK_SIZE), float64(brm.rows)))
			k := 0
			for p := pStart; p < pEnd; p++ {
				lInc := pEnd - pStart
				l := p - pStart
				for q := qStart; q < qEnd; q++ {
					copy.blocks[blockIndex][k] = brm.blocks[jBlock*brm.blockColumns+iBlock][l]
					k++
					l += lInc
				}
			}
			// go to next block
			blockIndex++
		}
	}

	return copy
}

func (brm *BlockRealMatrix) RowDimension() int {
	return brm.rows
}

func (brm *BlockRealMatrix) ColumnDimension() int {
	return brm.columns
}

func (brm *BlockRealMatrix) OperateVector(vec RealVector) RealVector {
	var out []float64
	if v, ok := vec.(*ArrayRealVector); ok {
		out = brm.Operate(v.DataRef())
	} else {
		nRows := brm.RowDimension()
		nCols := brm.ColumnDimension()
		if v.Dimension() != nCols {
			panic(dimensionsMismatchSimpleErrorf(v.Dimension(), nCols))
		}

		out = make([]float64, nRows)
		for row := 0; row < nRows; row++ {
			var sum float64
			for i := 0; i < nCols; i++ {
				sum += brm.At(row, i) * vec.At(i)
			}
			out[row] = sum
		}

	}

	v := new(ArrayRealVector)
	v.data = out
	return v
}

func (brm *BlockRealMatrix) Operate(v []float64) []float64 {
	if len(v) != brm.columns {
		panic(dimensionsMismatchSimpleErrorf(len(v), brm.columns))
	}
	out := make([]float64, brm.rows)

	// perform multiplication block-wise, to ensure good cache behavior
	for iBlock := 0; iBlock < brm.blockRows; iBlock++ {
		pStart := iBlock * BLOCK_SIZE
		pEnd := int(math.Min(float64(pStart+BLOCK_SIZE), float64(brm.rows)))
		for jBlock := 0; jBlock < brm.blockColumns; jBlock++ {
			qStart := jBlock * BLOCK_SIZE
			qEnd := int(math.Min(float64(qStart+BLOCK_SIZE), float64(brm.columns)))
			k := 0
			for p := pStart; p < pEnd; p++ {
				sum := 0.
				q := qStart
				for q < qEnd-3 {
					sum += brm.blocks[iBlock*brm.blockColumns+jBlock][k]*v[q] +
						brm.blocks[iBlock*brm.blockColumns+jBlock][k+1]*v[q+1] +
						brm.blocks[iBlock*brm.blockColumns+jBlock][k+2]*v[q+2] +
						brm.blocks[iBlock*brm.blockColumns+jBlock][k+3]*v[q+3]
					k += 4
					q += 4
				}
				for q < qEnd {
					sum += brm.blocks[iBlock*brm.blockColumns+jBlock][k] * v[q]
					k++
					q++
				}
				out[p] += sum
			}
		}
	}

	return out
}

func (brm *BlockRealMatrix) PreMultiplyMatrix(m RealMatrix) RealMatrix {
	return m.Multiply(brm)
}

func (brm *BlockRealMatrix) PreMultiplyVector(vec RealVector) RealVector {
	var out []float64
	if v, ok := vec.(*ArrayRealVector); ok {
		out = brm.PreMultiply(v.DataRef())
	} else {
		nRows := brm.RowDimension()
		nCols := brm.ColumnDimension()
		if v.Dimension() != nRows {
			panic(dimensionsMismatchSimpleErrorf(v.Dimension(), nRows))
		}

		out := make([]float64, nCols)
		for col := 0; col < nCols; col++ {
			var sum float64
			for i := 0; i < nRows; i++ {
				sum += brm.At(i, col) * v.At(i)
			}
			out[col] = sum
		}
	}

	v := new(ArrayRealVector)
	v.data = out
	return v
}

func (brm *BlockRealMatrix) PreMultiply(v []float64) []float64 {
	if len(v) != brm.rows {
		panic(dimensionsMismatchSimpleErrorf(len(v), brm.rows))
	}
	out := make([]float64, brm.columns)

	// perform multiplication block-wise, to ensure good cache behavior
	for jBlock := 0; jBlock < brm.blockColumns; jBlock++ {
		jWidth := brm.blockWidth(jBlock)
		jWidth2 := jWidth + jWidth
		jWidth3 := jWidth2 + jWidth
		jWidth4 := jWidth3 + jWidth
		qStart := jBlock * BLOCK_SIZE
		qEnd := int(math.Min(float64(qStart+BLOCK_SIZE), float64(brm.columns)))
		for iBlock := 0; iBlock < brm.blockRows; iBlock++ {
			pStart := iBlock * BLOCK_SIZE
			pEnd := int(math.Min(float64(pStart+BLOCK_SIZE), float64(brm.rows)))
			for q := qStart; q < qEnd; q++ {
				k := q - qStart
				sum := 0.
				p := pStart
				for p < pEnd-3 {
					sum += brm.blocks[iBlock*brm.blockColumns+jBlock][k]*v[p] +
						brm.blocks[iBlock*brm.blockColumns+jBlock][k+jWidth]*v[p+1] +
						brm.blocks[iBlock*brm.blockColumns+jBlock][k+jWidth2]*v[p+2] +
						brm.blocks[iBlock*brm.blockColumns+jBlock][k+jWidth3]*v[p+3]
					k += jWidth4
					p += 4
				}
				for p < pEnd {
					sum += brm.blocks[iBlock*brm.blockColumns+jBlock][k] * v[p]
					p++
					k += jWidth
				}
				out[q] += sum
			}
		}
	}

	return out
}

func (brm *BlockRealMatrix) WalkInUpdateRowOrder(visitor RealMatrixChangingVisitor) float64 {
	visitor.Start(brm.rows, brm.columns, 0, brm.rows-1, 0, brm.columns-1)
	for iBlock := 0; iBlock < brm.blockRows; iBlock++ {
		pStart := iBlock * BLOCK_SIZE
		pEnd := int(math.Min(float64(pStart+BLOCK_SIZE), float64(brm.rows)))
		for p := pStart; p < pEnd; p++ {
			for jBlock := 0; jBlock < brm.blockColumns; jBlock++ {
				jWidth := brm.blockWidth(jBlock)
				qStart := jBlock * BLOCK_SIZE
				qEnd := int(math.Min(float64(qStart+BLOCK_SIZE), float64(brm.columns)))
				k := (p - pStart) * jWidth
				for q := qStart; q < qEnd; q++ {
					brm.blocks[iBlock*brm.blockColumns+jBlock][k] = visitor.Visit(p, q, brm.blocks[iBlock*brm.blockColumns+jBlock][k])
					k++
				}
			}
		}
	}
	return visitor.End()
}

func (brm *BlockRealMatrix) WalkInRowOrder(visitor RealMatrixPreservingVisitor) float64 {
	visitor.Start(brm.rows, brm.columns, 0, brm.rows-1, 0, brm.columns-1)
	for iBlock := 0; iBlock < brm.blockRows; iBlock++ {
		pStart := iBlock * BLOCK_SIZE
		pEnd := int(math.Min(float64(pStart+BLOCK_SIZE), float64(brm.rows)))
		for p := pStart; p < pEnd; p++ {
			for jBlock := 0; jBlock < brm.blockColumns; jBlock++ {
				jWidth := brm.blockWidth(jBlock)
				qStart := jBlock * BLOCK_SIZE
				qEnd := int(math.Min(float64(qStart+BLOCK_SIZE), float64(brm.columns)))
				k := (p - pStart) * jWidth
				for q := qStart; q < qEnd; q++ {
					visitor.Visit(p, q, brm.blocks[iBlock*brm.blockColumns+jBlock][k])
					k++
				}
			}
		}
	}
	return visitor.End()
}

func (brm *BlockRealMatrix) WalkInUpdateRowOrderBounded(visitor RealMatrixChangingVisitor, startRow, endRow, startColumn, endColumn int) float64 {
	if err := checkSubMatrixIndex(brm, startRow, endRow, startColumn, endColumn); err != nil {
		panic(err)
	}
	visitor.Start(brm.rows, brm.columns, startRow, endRow, startColumn, endColumn)
	for iBlock := startRow / BLOCK_SIZE; iBlock < 1+endRow/BLOCK_SIZE; iBlock++ {
		p0 := iBlock * BLOCK_SIZE
		pStart := int(math.Max(float64(startRow), float64(p0)))
		pEnd := int(math.Min(float64((iBlock+1)*BLOCK_SIZE), float64(1+endRow)))
		for p := pStart; p < pEnd; p++ {
			for jBlock := startColumn / BLOCK_SIZE; jBlock < 1+endColumn/BLOCK_SIZE; jBlock++ {
				jWidth := brm.blockWidth(jBlock)
				q0 := jBlock * BLOCK_SIZE
				qStart := int(math.Max(float64(startColumn), float64(q0)))
				qEnd := int(math.Min(float64((jBlock+1)*BLOCK_SIZE), float64(1+endColumn)))
				k := (p-p0)*jWidth + qStart - q0
				for q := qStart; q < qEnd; q++ {
					brm.blocks[iBlock*brm.blockColumns+jBlock][k] = visitor.Visit(p, q, brm.blocks[iBlock*brm.blockColumns+jBlock][k])
					k++
				}
			}
		}
	}
	return visitor.End()
}

func (brm *BlockRealMatrix) WalkInRowOrderBounded(visitor RealMatrixPreservingVisitor, startRow, endRow, startColumn, endColumn int) float64 {
	if err := checkSubMatrixIndex(brm, startRow, endRow, startColumn, endColumn); err != nil {
		panic(err)
	}
	visitor.Start(brm.rows, brm.columns, startRow, endRow, startColumn, endColumn)
	for iBlock := startRow / BLOCK_SIZE; iBlock < 1+endRow/BLOCK_SIZE; iBlock++ {
		p0 := iBlock * BLOCK_SIZE
		pStart := int(math.Max(float64(startRow), float64(p0)))
		pEnd := int(math.Min(float64((iBlock+1)*BLOCK_SIZE), float64(1+endRow)))
		for p := pStart; p < pEnd; p++ {
			for jBlock := startColumn / BLOCK_SIZE; jBlock < 1+endColumn/BLOCK_SIZE; jBlock++ {
				jWidth := brm.blockWidth(jBlock)
				q0 := jBlock * BLOCK_SIZE
				qStart := int(math.Max(float64(startColumn), float64(q0)))
				qEnd := int(math.Min(float64((jBlock+1)*BLOCK_SIZE), float64(1+endColumn)))
				k := (p-p0)*jWidth + qStart - q0
				for q := qStart; q < qEnd; q++ {
					visitor.Visit(p, q, brm.blocks[iBlock*brm.blockColumns+jBlock][k])
					k++
				}
			}
		}
	}
	return visitor.End()
}

func (brm *BlockRealMatrix) WalkInUpdateColumnOrder(visitor RealMatrixChangingVisitor) float64 {
	rows := brm.RowDimension()
	columns := brm.ColumnDimension()
	visitor.Start(rows, columns, 0, rows-1, 0, columns-1)
	for column := 0; column < columns; column++ {
		for row := 0; row < rows; row++ {
			oldValue := brm.At(row, column)
			newValue := visitor.Visit(row, column, oldValue)
			brm.SetEntry(row, column, newValue)
		}
	}
	return visitor.End()
}

func (brm *BlockRealMatrix) WalkInColumnOrder(visitor RealMatrixPreservingVisitor) float64 {
	rows := brm.RowDimension()
	columns := brm.ColumnDimension()
	visitor.Start(rows, columns, 0, rows-1, 0, columns-1)
	for column := 0; column < columns; column++ {
		for row := 0; row < rows; row++ {
			visitor.Visit(row, column, brm.At(row, column))
		}
	}
	return visitor.End()
}

func (brm *BlockRealMatrix) WalkInUpdateColumnOrderBounded(visitor RealMatrixChangingVisitor, startRow, endRow, startColumn, endColumn int) float64 {
	if err := checkSubMatrixIndex(brm, startRow, endRow, startColumn, endColumn); err != nil {
		panic(err)
	}
	visitor.Start(brm.RowDimension(), brm.ColumnDimension(), startRow, endRow, startColumn, endColumn)
	for column := startColumn; column <= endColumn; column++ {
		for row := startRow; row <= endRow; row++ {
			oldValue := brm.At(row, column)
			newValue := visitor.Visit(row, column, oldValue)
			brm.SetEntry(row, column, newValue)
		}
	}
	return visitor.End()
}

func (brm *BlockRealMatrix) WalkInColumnOrderBounded(visitor RealMatrixPreservingVisitor, startRow, endRow, startColumn, endColumn int) float64 {
	if err := checkSubMatrixIndex(brm, startRow, endRow, startColumn, endColumn); err != nil {
		panic(err)
	}
	visitor.Start(brm.RowDimension(), brm.ColumnDimension(), startRow, endRow, startColumn, endColumn)
	for column := startColumn; column <= endColumn; column++ {
		for row := startRow; row <= endRow; row++ {
			visitor.Visit(row, column, brm.At(row, column))
		}
	}
	return visitor.End()
}

func (brm *BlockRealMatrix) WalkInUpdateOptimizedOrder(visitor RealMatrixChangingVisitor) float64 {
	visitor.Start(brm.rows, brm.columns, 0, brm.rows-1, 0, brm.columns-1)
	blockIndex := 0
	for iBlock := 0; iBlock < brm.blockRows; iBlock++ {
		pStart := iBlock * BLOCK_SIZE
		pEnd := int(math.Min(float64(pStart+BLOCK_SIZE), float64(brm.rows)))
		for jBlock := 0; jBlock < brm.blockColumns; jBlock++ {
			qStart := jBlock * BLOCK_SIZE
			qEnd := int(math.Min(float64(qStart+BLOCK_SIZE), float64(brm.columns)))
			k := 0
			for p := pStart; p < pEnd; p++ {
				for q := qStart; q < qEnd; q++ {
					brm.blocks[blockIndex][k] = visitor.Visit(p, q, brm.blocks[blockIndex][k])
					k++
				}
			}
			blockIndex++
		}
	}
	return visitor.End()
}

func (brm *BlockRealMatrix) WalkInOptimizedOrder(visitor RealMatrixPreservingVisitor) float64 {
	visitor.Start(brm.rows, brm.columns, 0, brm.rows-1, 0, brm.columns-1)
	blockIndex := 0
	for iBlock := 0; iBlock < brm.blockRows; iBlock++ {
		pStart := iBlock * BLOCK_SIZE
		pEnd := int(math.Min(float64(pStart+BLOCK_SIZE), float64(brm.rows)))
		for jBlock := 0; jBlock < brm.blockColumns; jBlock++ {
			qStart := jBlock * BLOCK_SIZE
			qEnd := int(math.Min(float64(qStart+BLOCK_SIZE), float64(brm.columns)))
			k := 0
			for p := pStart; p < pEnd; p++ {
				for q := qStart; q < qEnd; q++ {
					visitor.Visit(p, q, brm.blocks[blockIndex][k])
					k++
				}
			}
			blockIndex++
		}
	}
	return visitor.End()
}

func (brm *BlockRealMatrix) WalkInUpdateOptimizedOrderBounded(visitor RealMatrixChangingVisitor, startRow, endRow, startColumn, endColumn int) float64 {
	if err := checkSubMatrixIndex(brm, startRow, endRow, startColumn, endColumn); err != nil {
		panic(err)
	}
	visitor.Start(brm.rows, brm.columns, startRow, endRow, startColumn, endColumn)
	for iBlock := startRow / BLOCK_SIZE; iBlock < 1+endRow/BLOCK_SIZE; iBlock++ {
		p0 := iBlock * BLOCK_SIZE
		pStart := int(math.Max(float64(startRow), float64(p0)))
		pEnd := int(math.Min(float64((iBlock+1)*BLOCK_SIZE), float64(1+endRow)))
		for jBlock := startColumn / BLOCK_SIZE; jBlock < 1+endColumn/BLOCK_SIZE; jBlock++ {
			jWidth := brm.blockWidth(jBlock)
			q0 := jBlock * BLOCK_SIZE
			qStart := int(math.Max(float64(startColumn), float64(q0)))
			qEnd := int(math.Min(float64((jBlock+1)*BLOCK_SIZE), float64(1+endColumn)))
			for p := pStart; p < pEnd; p++ {
				k := (p-p0)*jWidth + qStart - q0
				for q := qStart; q < qEnd; q++ {
					brm.blocks[iBlock*brm.blockColumns+jBlock][k] = visitor.Visit(p, q, brm.blocks[iBlock*brm.blockColumns+jBlock][k])
					k++
				}
			}
		}
	}
	return visitor.End()
}

func (brm *BlockRealMatrix) WalkInOptimizedOrderBounded(visitor RealMatrixPreservingVisitor, startRow, endRow, startColumn, endColumn int) float64 {
	if err := checkSubMatrixIndex(brm, startRow, endRow, startColumn, endColumn); err != nil {
		panic(err)
	}
	visitor.Start(brm.rows, brm.columns, startRow, endRow, startColumn, endColumn)
	for iBlock := startRow / BLOCK_SIZE; iBlock < 1+endRow/BLOCK_SIZE; iBlock++ {
		p0 := iBlock * BLOCK_SIZE
		pStart := int(math.Max(float64(startRow), float64(p0)))
		pEnd := int(math.Min(float64((iBlock+1)*BLOCK_SIZE), float64(1+endRow)))
		for jBlock := startColumn / BLOCK_SIZE; jBlock < 1+endColumn/BLOCK_SIZE; jBlock++ {
			jWidth := brm.blockWidth(jBlock)
			q0 := jBlock * BLOCK_SIZE
			qStart := int(math.Max(float64(startColumn), float64(q0)))
			qEnd := int(math.Min(float64((jBlock+1)*BLOCK_SIZE), float64(1+endColumn)))
			for p := pStart; p < pEnd; p++ {
				k := (p-p0)*jWidth + qStart - q0
				for q := qStart; q < qEnd; q++ {
					visitor.Visit(p, q, brm.blocks[iBlock*brm.blockColumns+jBlock][k])
					k++
				}
			}
		}
	}
	return visitor.End()
}

func (brm *BlockRealMatrix) blockHeight(blockRow int) int {
	if blockRow == brm.blockRows-1 {
		return brm.rows - blockRow*BLOCK_SIZE
	}

	return BLOCK_SIZE
}

func (brm *BlockRealMatrix) blockWidth(blockColumn int) int {
	if blockColumn == brm.blockColumns-1 {
		return brm.columns - blockColumn*BLOCK_SIZE
	}

	return BLOCK_SIZE
}

func (brm *BlockRealMatrix) Equals(object interface{}) bool {
	if object == brm {
		return true
	}
	if _, ok := object.(RealMatrix); !ok {
		return false
	}

	m := object.(RealMatrix)
	nRows := brm.RowDimension()
	nCols := brm.ColumnDimension()
	if m.ColumnDimension() != nCols || m.RowDimension() != nRows {
		return false
	}
	for row := 0; row < nRows; row++ {
		for col := 0; col < nCols; col++ {
			if brm.At(row, col) != m.At(row, col) {
				return false
			}
		}
	}
	return true
}
