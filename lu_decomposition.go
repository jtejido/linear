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
	default_too_small_lud = 1e-11
)

/**
 * Calculates the LUP-decomposition of a square matrix.
 * The LUP-decomposition of a matrix A consists of three matrices L, U and
 * P that satisfy: P&times;A = L&times;U. L is lower triangular (with unit
 * diagonal terms), U is upper triangular and P is a permutation matrix. All
 * matrices are m&times;m.
 * As shown by the presence of the P matrix, this decomposition is
 * implemented using partial pivoting.
 * This class is based on the class with similar name from the
 * JAMA library.
 */
type LUDecomposition struct {
	lu                        [][]float64
	pivot                     []int
	even, singular            bool
	cachedL, cachedU, cachedP RealMatrix
}

/**
 * Calculates the LU-decomposition of the given matrix.
 * This constructor uses 1e-11 as default value for the singularity
 * threshold.
 */
func NewLUDecomposition(matrix RealMatrix) (*LUDecomposition, error) {
	return NewLUDecompositionWithThreshold(matrix, default_too_small_lud)
}

/**
 * Calculates the LU-decomposition of the given matrix.
 */
func NewLUDecompositionWithThreshold(matrix RealMatrix, singularityThreshold float64) (*LUDecomposition, error) {
	if !IsSquare(matrix) {
		return nil, nonSquareMatrixSimpleErrorf(matrix.RowDimension(), matrix.ColumnDimension())
	}

	ans := new(LUDecomposition)
	m := matrix.ColumnDimension()
	ans.lu = matrix.Data()
	ans.pivot = make([]int, m)

	// Initialize permutation array and parity
	for row := 0; row < m; row++ {
		ans.pivot[row] = row
	}
	ans.even = true
	ans.singular = false

	// Loop over columns
	for col := 0; col < m; col++ {

		// upper
		for row := 0; row < col; row++ {
			luRow := ans.lu[row]
			sum := luRow[col]
			for i := 0; i < row; i++ {
				sum -= luRow[i] * ans.lu[i][col]
			}
			luRow[col] = sum
		}

		// lower
		max := col // permutation row
		largest := math.Inf(-1)
		for row := col; row < m; row++ {
			luRow := ans.lu[row]
			sum := luRow[col]
			for i := 0; i < col; i++ {
				sum -= luRow[i] * ans.lu[i][col]
			}
			luRow[col] = sum

			// maintain best permutation choice
			if math.Abs(sum) > largest {
				largest = math.Abs(sum)
				max = row
			}
		}

		// Singularity check
		if math.Abs(ans.lu[max][col]) < singularityThreshold {
			ans.singular = true
			return ans, nil
		}

		// Pivot if necessary
		if max != col {
			tmp := 0.
			luMax := ans.lu[max]
			luCol := ans.lu[col]
			for i := 0; i < m; i++ {
				tmp = luMax[i]
				luMax[i] = luCol[i]
				luCol[i] = tmp
			}
			temp := ans.pivot[max]
			ans.pivot[max] = ans.pivot[col]
			ans.pivot[col] = temp
			ans.even = !ans.even
		}

		// Divide the lower elements by the "winning" diagonal elt.
		luDiag := ans.lu[col][col]
		for row := col + 1; row < m; row++ {
			ans.lu[row][col] /= luDiag
		}
	}

	return ans, nil
}

/**
 * Returns the matrix L of the decomposition.
 * L is a lower-triangular matrix
 */
func (lud *LUDecomposition) L() RealMatrix {
	if (lud.cachedL == nil) && !lud.singular {
		m := len(lud.pivot)
		var err error
		lud.cachedL, err = NewRealMatrixWithDimension(m, m)
		if err != nil {
			panic(err)
		}

		for i := 0; i < m; i++ {
			luI := lud.lu[i]
			for j := 0; j < i; j++ {
				lud.cachedL.SetEntry(i, j, luI[j])
			}
			lud.cachedL.SetEntry(i, i, 1.0)
		}
	}
	return lud.cachedL
}

/**
 * Returns the matrix U of the decomposition.
 * <p>U is an upper-triangular matrix</p>
 * @return the U matrix (or null if decomposed matrix is singular)
 */
func (lud *LUDecomposition) U() RealMatrix {
	if (lud.cachedU == nil) && !lud.singular {
		m := len(lud.pivot)
		var err error
		lud.cachedU, err = NewRealMatrixWithDimension(m, m)
		if err != nil {
			panic(err)
		}
		for i := 0; i < m; i++ {
			luI := lud.lu[i]
			for j := i; j < m; j++ {
				lud.cachedU.SetEntry(i, j, luI[j])
			}
		}
	}
	return lud.cachedU
}

/**
 * Returns the P rows permutation matrix.
 * P is a sparse matrix with exactly one element set to 1.0 in
 * each row and each column, all other elements being set to 0.0.
 * The positions of the 1 elements are given by the {@link #getPivot()
 * pivot permutation vector}.
 */
func (lud *LUDecomposition) P() RealMatrix {
	if (lud.cachedP == nil) && !lud.singular {
		m := len(lud.pivot)
		var err error
		lud.cachedP, err = NewRealMatrixWithDimension(m, m)
		if err != nil {
			panic(err)
		}
		for i := 0; i < m; i++ {
			lud.cachedP.SetEntry(i, lud.pivot[i], 1.0)
		}
	}
	return lud.cachedP
}

/**
 * Returns the pivot permutation
 */
func (lud *LUDecomposition) Pivot() []int {
	return append([]int{}, lud.pivot...)
}

/**
 * Return the determinant of the matrix
 */
func (lud *LUDecomposition) Determinant() float64 {
	if lud.singular {
		return 0
	} else {
		m := len(lud.pivot)
		determinant := -1.
		if lud.even {
			determinant = 1.
		}
		for i := 0; i < m; i++ {
			determinant *= lud.lu[i][i]
		}

		return determinant
	}
}

/**
 * Get a solver for finding the A &times; X = B solution in exact linear
 * sense.
 */
func (lud *LUDecomposition) Solver() DecompositionSolver {
	return newLUDecompositionSolver(lud)
}

type luDecompositionSolver struct {
	lud *LUDecomposition
}

func newLUDecompositionSolver(lud *LUDecomposition) *luDecompositionSolver {
	return &luDecompositionSolver{lud: lud}
}

func (s *luDecompositionSolver) IsNonSingular() bool {
	return !s.lud.singular
}

func (s *luDecompositionSolver) SolveVector(b RealVector) RealVector {
	m := len(s.lud.pivot)
	if b.Dimension() != m {
		panic(dimensionsMismatchSimpleErrorf(b.Dimension(), m))
	}
	if s.lud.singular {
		panic(singularMatrixSimpleErrorf())
	}

	bp := make([]float64, m)

	// Apply permutations to b
	for row := 0; row < m; row++ {
		bp[row] = b.At(s.lud.pivot[row])
	}

	// Solve LY = b
	for col := 0; col < m; col++ {
		bpCol := bp[col]
		for i := col + 1; i < m; i++ {
			bp[i] -= bpCol * s.lud.lu[i][col]
		}
	}

	// Solve UX = Y
	for col := m - 1; col >= 0; col-- {
		bp[col] /= s.lud.lu[col][col]
		bpCol := bp[col]
		for i := 0; i < col; i++ {
			bp[i] -= bpCol * s.lud.lu[i][col]
		}
	}

	vec := new(ArrayRealVector)
	vec.data = append([]float64{}, bp...)
	return vec
}

func (s *luDecompositionSolver) SolveMatrix(b RealMatrix) RealMatrix {
	m := len(s.lud.pivot)
	if b.RowDimension() != m {
		panic(dimensionsMismatchSimpleErrorf(b.RowDimension(), m))
	}
	if s.lud.singular {
		panic(singularMatrixSimpleErrorf())
	}

	nColB := b.ColumnDimension()

	// Apply permutations to b
	bp := make([][]float64, m)
	for row := 0; row < m; row++ {
		bp[row] = make([]float64, nColB)
		bpRow := bp[row]
		pRow := s.lud.pivot[row]
		for col := 0; col < nColB; col++ {
			bpRow[col] = b.At(pRow, col)
		}
	}

	// Solve LY = b
	for col := 0; col < m; col++ {
		bpCol := bp[col]
		for i := col + 1; i < m; i++ {
			bpI := bp[i]
			luICol := s.lud.lu[i][col]
			for j := 0; j < nColB; j++ {
				bpI[j] -= bpCol[j] * luICol
			}
		}
	}

	// Solve UX = Y
	for col := m - 1; col >= 0; col-- {
		bpCol := bp[col]
		luDiag := s.lud.lu[col][col]
		for j := 0; j < nColB; j++ {
			bpCol[j] /= luDiag
		}
		for i := 0; i < col; i++ {
			bpI := bp[i]
			luICol := s.lud.lu[i][col]
			for j := 0; j < nColB; j++ {
				bpI[j] -= bpCol[j] * luICol
			}
		}
	}

	mat := new(Array2DRowRealMatrix)
	mat.copyIn(bp)
	return mat
}

func (s *luDecompositionSolver) Inverse() RealMatrix {
	m, err := NewRealIdentityMatrix(len(s.lud.pivot))
	if err != nil {
		panic(err)
	}

	return s.SolveMatrix(m)
}
