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

/**
 * Calculates the rank-revealing QR-decomposition of a matrix, with column pivoting.
 * The rank-revealing QR-decomposition of a matrix A consists of three matrices Q,
 * R and P such that AP=QR.  Q is orthogonal (Q<sup>T</sup>Q = I), and R is upper triangular.
 * If A is m&times;n, Q is m&times;m and R is m&times;n and P is n&times;n.
 * QR decomposition with column pivoting produces a rank-revealing QR
 * decomposition and the Rank(float64) method may be used to return the rank of the
 * input matrix A.
 * This class compute the decomposition using Householder reflectors.
 * For efficiency purposes, the decomposition in packed form is transposed.
 * This allows inner loop to iterate inside rows.
 */
type RRQRDecomposition struct {
	QRDecomposition
	p       []int
	cachedP RealMatrix
}

func NewRRQRDecomposition(matrix RealMatrix) (*RRQRDecomposition, error) {
	return NewRRQRDecompositionWithThreshold(matrix, 0.)
}

func NewRRQRDecompositionWithThreshold(matrix RealMatrix, threshold float64) (*RRQRDecomposition, error) {
	ans := new(RRQRDecomposition)
	ans.threshold = threshold

	m := matrix.RowDimension()
	n := matrix.ColumnDimension()
	ans.qrt = matrix.Transpose().Data()
	ans.rDiag = make([]float64, int(math.Min(float64(m), float64(n))))

	ans.decompose(ans.qrt)
	return ans, nil
}

/** Decompose matrix.
 */
func (rrqrd *RRQRDecomposition) decompose(qrt [][]float64) {
	rrqrd.p = make([]int, len(qrt))
	for i := 0; i < len(rrqrd.p); i++ {
		rrqrd.p[i] = i
	}

	for minor := 0; minor < int(math.Min(float64(len(qrt)), float64(len(qrt[0])))); minor++ {
		rrqrd.performHouseholderReflection(minor, qrt)
	}
}

/**
 * Perform Householder reflection for a minor A(minor, minor) of A.
 */
func (rrqrd *RRQRDecomposition) performHouseholderReflection(minor int, qrt [][]float64) {
	var l2NormSquaredMax float64
	// Find the unreduced column with the greatest L2-Norm
	l2NormSquaredMaxIndex := minor
	for i := minor; i < len(qrt); i++ {
		var l2NormSquared float64
		for j := minor; j < len(qrt[i]); j++ {
			l2NormSquared += qrt[i][j] * qrt[i][j]
		}
		if l2NormSquared > l2NormSquaredMax {
			l2NormSquaredMax = l2NormSquared
			l2NormSquaredMaxIndex = i
		}
	}
	// swap the current column with that with the greated L2-Norm and record in p
	if l2NormSquaredMaxIndex != minor {
		qrt[minor], qrt[l2NormSquaredMaxIndex] = qrt[l2NormSquaredMaxIndex], qrt[minor]
		rrqrd.p[minor], rrqrd.p[l2NormSquaredMaxIndex] = rrqrd.p[l2NormSquaredMaxIndex], rrqrd.p[minor]
	}

	rrqrd.QRDecomposition.performHouseholderReflection(minor, qrt)
}

/**
 * Returns the pivot matrix, P, used in the QR Decomposition of matrix A such that AP = QR.
 *
 * If no pivoting is used in this decomposition then P is equal to the identity matrix.
 */
func (rrqrd *RRQRDecomposition) P() RealMatrix {
	if rrqrd.cachedP == nil {
		n := len(rrqrd.p)
		var err error
		rrqrd.cachedP, err = NewRealMatrixWithDimension(n, n)
		if err != nil {
			panic(err)
		}
		for i := 0; i < n; i++ {
			rrqrd.cachedP.SetEntry(rrqrd.p[i], i, 1)
		}
	}
	return rrqrd.cachedP
}

/**
 * Return the effective numerical matrix rank.
 * The effective numerical rank is the number of non-negligible
 * singular values.
 * This implementation looks at Frobenius norms of the sequence of
 * bottom right submatrices.  When a large fall in norm is seen,
 * the rank is returned. The drop is computed as:
 *
 *   (thisNorm/lastNorm) * rNorm &lt; dropThreshold
 *
 * where thisNorm is the Frobenius norm of the current submatrix,
 * lastNorm is the Frobenius norm of the previous submatrix,
 * rNorm is is the Frobenius norm of the complete matrix
 */
func (rrqrd *RRQRDecomposition) Rank(dropThreshold float64) int {
	r := rrqrd.R()
	rows := r.RowDimension()
	columns := r.ColumnDimension()
	rank := 1
	lastNorm := MatFrobeniusNorm(r)
	rNorm := lastNorm
	for rank < int(math.Min(float64(rows), float64(columns))) {
		thisNorm := MatFrobeniusNorm(r.SubMatrix(rank, rows-1, rank, columns-1))
		if thisNorm == 0 || (thisNorm/lastNorm)*rNorm < dropThreshold {
			break
		}
		lastNorm = thisNorm
		rank++
	}
	return rank
}

/**
 * Get a solver for finding the A &times; X = B solution in least square sense.
 *
 * Least Square sense means a solver can be computed for an overdetermined system,
 * (i.e. a system with more equations than unknowns, which corresponds to a tall A
 * matrix with more rows than columns). In any case, if the matrix is singular
 * within the tolerance set at RRQRDecomposition(RealMatrix, float64) construction, an error will be triggered when
 * the DecompositionSolver.Solve() method will be called.
 */
func (rrqrd *RRQRDecomposition) Solver() DecompositionSolver {
	return newRRQRDecompositionSolver(rrqrd)
}

type rrqrDecompositionSolver struct {
	rrqrd *RRQRDecomposition
	qrds  *qrDecompositionSolver
}

func newRRQRDecompositionSolver(rrqrd *RRQRDecomposition) *rrqrDecompositionSolver {
	return &rrqrDecompositionSolver{rrqrd: rrqrd, qrds: rrqrd.QRDecomposition.Solver().(*qrDecompositionSolver)}
}

func (rrqrds *rrqrDecompositionSolver) IsNonSingular() bool {
	return rrqrds.qrds.IsNonSingular()
}

func (rrqrds *rrqrDecompositionSolver) SolveVector(b RealVector) RealVector {
	return rrqrds.rrqrd.P().OperateVector(rrqrds.qrds.SolveVector(b))
}

func (rrqrds *rrqrDecompositionSolver) SolveMatrix(b RealMatrix) RealMatrix {
	return rrqrds.rrqrd.P().Multiply(rrqrds.qrds.SolveMatrix(b))
}

func (rrqrds *rrqrDecompositionSolver) Inverse() RealMatrix {
	m, err := NewRealIdentityMatrix(rrqrds.rrqrd.P().RowDimension())
	if err != nil {
		panic(err)
	}

	return rrqrds.SolveMatrix(m)
}
