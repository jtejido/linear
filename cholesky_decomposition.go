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
	DEFAULT_RELATIVE_SYMMETRY_THRESHOLD   float64 = 1.0e-15
	DEFAULT_ABSOLUTE_POSITIVITY_THRESHOLD float64 = 1.0e-10
)

/**
 * Calculates the Cholesky decomposition of a matrix.
 * <p>The Cholesky decomposition of a real symmetric positive-definite
 * matrix A consists of a lower triangular matrix L with same size such
 * that: A = LL<sup>T</sup>. In a sense, this is the square root of A.</p>
 * <p>This class is based on the class with similar name from the
 * <a href="http://math.nist.gov/javanumerics/jama/">JAMA</a> library, with the
 * following changes:</p>
 * <ul>
 *   <li>a {@link #getLT() getLT} method has been added,</li>
 *   <li>the {@code isspd} method has been removed, since the constructor of
 *   this class throws a {@link NonPositiveDefiniteMatrixException} when a
 *   matrix cannot be decomposed,</li>
 *   <li>a {@link #getDeterminant() getDeterminant} method has been added,</li>
 *   <li>the {@code solve} method has been replaced by a {@link #getSolver()
 *   getSolver} method and the equivalent method provided by the returned
 *   {@link DecompositionSolver}.</li>
 * </ul>
 *
 * @see <a href="http://mathworld.wolfram.com/CholeskyDecomposition.html">MathWorld</a>
 * @see <a href="http://en.wikipedia.org/wiki/Cholesky_decomposition">Wikipedia</a>
 */
type CholeskyDecomposition struct {
	lTData            [][]float64
	cachedL, cachedLT RealMatrix
	m                 int
}

/**
 * Calculates the Cholesky decomposition of the given matrix.
 *
 * Calling this constructor is equivalent to call NewCholeskyDecompositionWithThreshold with the
 * thresholds set to the default values DEFAULT_RELATIVE_SYMMETRY_THRESHOLD DEFAULT_ABSOLUTE_POSITIVITY_THRESHOLD
 */
func NewDefaultCholeskyDecomposition(matrix RealMatrix) (*CholeskyDecomposition, error) {
	return NewCholeskyDecomposition(matrix, DEFAULT_RELATIVE_SYMMETRY_THRESHOLD, DEFAULT_ABSOLUTE_POSITIVITY_THRESHOLD)
}

/**
 * Calculates the Cholesky decomposition of the given matrix.
 */
func NewCholeskyDecomposition(matrix RealMatrix, relativeSymmetryThreshold, absolutePositivityThreshold float64) (*CholeskyDecomposition, error) {
	if !IsSquare(matrix) {
		return nil, nonSquareMatrixSimpleErrorf(matrix.RowDimension(), matrix.ColumnDimension())
	}

	ans := new(CholeskyDecomposition)
	order := matrix.RowDimension()
	ans.lTData = matrix.Data()
	ans.m = len(ans.lTData)

	// check the matrix before transformation
	for i := 0; i < order; i++ {
		lI := ans.lTData[i]

		// check off-diagonal elements (and reset them to 0)
		for j := i + 1; j < order; j++ {
			lJ := ans.lTData[j]
			lIJ := lI[j]
			lJI := lJ[i]
			maxDelta := relativeSymmetryThreshold * math.Max(math.Abs(lIJ), math.Abs(lJI))
			if math.Abs(lIJ-lJI) > maxDelta {
				return nil, nonSymmetricMatrixSimpleErrorf(i, j, relativeSymmetryThreshold)
			}
			lJ[i] = 0
		}
	}

	// transform the matrix
	for i := 0; i < order; i++ {

		ltI := ans.lTData[i]

		// check diagonal element
		if ltI[i] <= absolutePositivityThreshold {
			return nil, nonPositiveDefiniteMatrixErrorf(ltI[i], i, absolutePositivityThreshold)
		}

		ltI[i] = math.Sqrt(ltI[i])
		inverse := 1.0 / ltI[i]

		for q := order - 1; q > i; q-- {
			ltI[q] *= inverse
			ltQ := ans.lTData[q]
			for p := q; p < order; p++ {
				ltQ[p] -= ltI[q] * ltI[p]
			}
		}
	}

	return ans, nil

}

/**
 * Returns the matrix L of the decomposition. L is an lower-triangular matrix.
 */
func (cd *CholeskyDecomposition) L() RealMatrix {
	if cd.cachedL == nil {
		cd.cachedL = cd.LT().Transpose()
	}
	return cd.cachedL
}

/**
 * Returns the transpose of the matrix L of the decomposition. LT is an upper-triangular matrix.
 */
func (cd *CholeskyDecomposition) LT() RealMatrix {
	if cd.cachedLT == nil {
		var err error
		cd.cachedLT, err = NewRealMatrixFromSlices(cd.lTData)
		if err != nil {
			panic(err)
		}
	}
	return cd.cachedLT
}

/**
 * Return the determinant of the matrix.
 */
func (cd *CholeskyDecomposition) Determinant() float64 {
	determinant := 1.0
	for i := 0; i < len(cd.lTData); i++ {
		lTii := cd.lTData[i][i]
		determinant *= lTii * lTii
	}
	return determinant
}

/**
 * Get a solver for finding the A &times; X = B solution in least square sense.
 */
func (cd *CholeskyDecomposition) Solver() DecompositionSolver {
	return newCholeskyDecompositionSolver(cd)
}

type choleskyDecompositionSolver struct {
	cd *CholeskyDecomposition
}

func newCholeskyDecompositionSolver(cd *CholeskyDecomposition) *choleskyDecompositionSolver {
	return &choleskyDecompositionSolver{cd: cd}
}

func (s *choleskyDecompositionSolver) IsNonSingular() bool {
	// if we get this far, the matrix was positive definite, hence non-singular
	return true
}

func (s *choleskyDecompositionSolver) SolveVector(b RealVector) RealVector {
	m := s.cd.m
	if b.Dimension() != m {
		panic(dimensionsMismatchSimpleErrorf(b.Dimension(), m))
	}

	x := b.ToArray()

	// Solve LY = b
	for j := 0; j < m; j++ {
		lJ := s.cd.lTData[j]
		x[j] /= lJ[j]
		xJ := x[j]
		for i := j + 1; i < m; i++ {
			x[i] -= xJ * lJ[i]
		}
	}

	// Solve LTX = Y
	for j := m - 1; j >= 0; j-- {
		x[j] /= s.cd.lTData[j][j]
		xJ := x[j]
		for i := 0; i < j; i++ {
			x[i] -= xJ * s.cd.lTData[i][j]
		}
	}

	mat := new(ArrayRealVector)
	mat.data = append([]float64{}, x...)
	return mat
}

func (s *choleskyDecompositionSolver) SolveMatrix(b RealMatrix) RealMatrix {
	m := s.cd.m
	if b.RowDimension() != m {
		panic(dimensionsMismatchSimpleErrorf(b.RowDimension(), m))
	}

	nColB := b.ColumnDimension()
	x := b.Data()

	// Solve LY = b
	for j := 0; j < m; j++ {
		lJ := s.cd.lTData[j]
		lJJ := lJ[j]
		xJ := x[j]
		for k := 0; k < nColB; k++ {
			xJ[k] /= lJJ
		}
		for i := j + 1; i < m; i++ {
			xI := x[i]
			lJI := lJ[i]
			for k := 0; k < nColB; k++ {
				xI[k] -= xJ[k] * lJI
			}
		}
	}

	// Solve LTX = Y
	for j := m - 1; j >= 0; j-- {
		lJJ := s.cd.lTData[j][j]
		xJ := x[j]
		for k := 0; k < nColB; k++ {
			xJ[k] /= lJJ
		}
		for i := 0; i < j; i++ {
			xI := x[i]
			lIJ := s.cd.lTData[i][j]
			for k := 0; k < nColB; k++ {
				xI[k] -= xJ[k] * lIJ
			}
		}
	}

	mat := new(Array2DRowRealMatrix)
	mat.copyIn(x)
	return mat
}

func (s *choleskyDecompositionSolver) Inverse() RealMatrix {
	mat, err := NewRealIdentityMatrix(s.cd.m)
	if err != nil {
		panic(err)
	}

	return s.SolveMatrix(mat)
}
