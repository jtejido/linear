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
 * Calculates the QR-decomposition of a matrix.
 * The QR-decomposition of a matrix A consists of two matrices Q and R
 * that satisfy: A = QR, Q is orthogonal (Q<sup>T</sup>Q = I), and R is
 * upper triangular. If A is m&times;n, Q is m&times;m and R m&times;n.
 * This class compute the decomposition using Householder reflectors.
 * For efficiency purposes, the decomposition in packed form is transposed.
 * This allows inner loop to iterate inside rows, which is much more cache-efficient
 * in Java.
 */
type QRDecomposition struct {
	/**
	 * A packed TRANSPOSED representation of the QR decomposition.
	 * The elements BELOW the diagonal are the elements of the UPPER triangular
	 * matrix R, and the rows ABOVE the diagonal are the Householder reflector vectors
	 * from which an explicit form of Q can be recomputed if desired.
	 */
	qrt                                 [][]float64
	rDiag                               []float64
	cachedQ, cachedQT, cachedR, cachedH RealMatrix
	threshold                           float64
}

/**
 * Calculates the QR-decomposition of the given matrix.
 * The singularity threshold defaults to zero.
 */
func NewQRDecomposition(matrix RealMatrix) (*QRDecomposition, error) {
	return NewQRDecompositionWithThreshold(matrix, 0)
}

func NewQRDecompositionWithThreshold(matrix RealMatrix, threshold float64) (*QRDecomposition, error) {
	ans := new(QRDecomposition)
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
func (qrd *QRDecomposition) decompose(matrix [][]float64) {
	for minor := 0; minor < int(math.Min(float64(len(matrix)), float64(len(matrix[0])))); minor++ {
		qrd.performHouseholderReflection(minor, matrix)
	}
}

/** Perform Householder reflection for a minor A(minor, minor) of A.
 */
func (qrd *QRDecomposition) performHouseholderReflection(minor int, matrix [][]float64) {

	qrtMinor := matrix[minor]

	/*
	 * Let x be the first column of the minor, and a^2 = |x|^2.
	 * x will be in the positions qr[minor][minor] through qr[m][minor].
	 * The first column of the transformed minor will be (a,0,0,..)'
	 * The sign of a is chosen to be opposite to the sign of the first
	 * component of x. Let's find a:
	 */
	var xNormSqr float64
	for row := minor; row < len(qrtMinor); row++ {
		c := qrtMinor[row]
		xNormSqr += c * c
	}
	a := math.Sqrt(xNormSqr)
	if qrtMinor[minor] > 0 {
		a *= -1
	}

	qrd.rDiag[minor] = a

	if a != 0.0 {

		/*
		 * Calculate the normalized reflection vector v and transform
		 * the first column. We know the norm of v beforehand: v = x-ae
		 * so |v|^2 = <x-ae,x-ae> = <x,x>-2a<x,e>+a^2<e,e> =
		 * a^2+a^2-2a<x,e> = 2a*(a - <x,e>).
		 * Here <x, e> is now qr[minor][minor].
		 * v = x-ae is stored in the column at qr:
		 */
		qrtMinor[minor] -= a // now |v|^2 = -2a*(qr[minor][minor])

		/*
		 * Transform the rest of the columns of the minor:
		 * They will be transformed by the matrix H = I-2vv'/|v|^2.
		 * If x is a column vector of the minor, then
		 * Hx = (I-2vv'/|v|^2)x = x-2vv'x/|v|^2 = x - 2<x,v>/|v|^2 v.
		 * Therefore the transformation is easily calculated by
		 * subtracting the column vector (2<x,v>/|v|^2)v from x.
		 *
		 * Let 2<x,v>/|v|^2 = alpha. From above we have
		 * |v|^2 = -2a*(qr[minor][minor]), so
		 * alpha = -<x,v>/(a*qr[minor][minor])
		 */
		for col := minor + 1; col < len(matrix); col++ {
			qrtCol := matrix[col]
			alpha := 0.
			for row := minor; row < len(qrtCol); row++ {
				alpha -= qrtCol[row] * qrtMinor[row]
			}
			alpha /= a * qrtMinor[minor]

			// Subtract the column vector alpha*v from x.
			for row := minor; row < len(qrtCol); row++ {
				qrtCol[row] -= alpha * qrtMinor[row]
			}
		}
	}
}

/**
 * Returns the matrix R of the decomposition.
 * R is an upper-triangular matrix
 */
func (qrd *QRDecomposition) R() RealMatrix {

	if qrd.cachedR == nil {

		// R is supposed to be m x n
		n := len(qrd.qrt)
		m := len(qrd.qrt[0])
		ra := make([][]float64, m)
		for i := 0; i < m; i++ {
			ra[i] = make([]float64, n)
		}
		// copy the diagonal from rDiag and the upper triangle of qr
		for row := int(math.Min(float64(m), float64(n))) - 1; row >= 0; row-- {
			ra[row][row] = qrd.rDiag[row]
			for col := row + 1; col < n; col++ {
				ra[row][col] = qrd.qrt[col][row]
			}
		}
		var err error
		qrd.cachedR, err = NewRealMatrixFromSlices(ra)
		if err != nil {
			panic(err)
		}
	}

	// return the cached matrix
	return qrd.cachedR
}

/**
 * Returns the matrix Q of the decomposition.
 * Q is an orthogonal matrix
 */
func (qrd *QRDecomposition) Q() RealMatrix {
	if qrd.cachedQ == nil {
		qrd.cachedQ = qrd.QT().Transpose()
	}

	return qrd.cachedQ
}

/**
 * Returns the transpose of the matrix Q of the decomposition.
 * Q is an orthogonal matrix
 */
func (qrd *QRDecomposition) QT() RealMatrix {
	if qrd.cachedQT == nil {

		// QT is supposed to be m x m
		n := len(qrd.qrt)
		m := len(qrd.qrt[0])
		qta := make([][]float64, m)
		for i := 0; i < m; i++ {
			qta[i] = make([]float64, m)
		}

		/*
		 * Q = Q1 Q2 ... Q_m, so Q is formed by first constructing Q_m and then
		 * applying the Householder transformations Q_(m-1),Q_(m-2),...,Q1 in
		 * succession to the result
		 */
		for minor := m - 1; minor >= int(math.Min(float64(m), float64(n))); minor-- {
			qta[minor][minor] = 1.0
		}

		for minor := int(math.Min(float64(m), float64(n))) - 1; minor >= 0; minor-- {
			qrtMinor := qrd.qrt[minor]
			qta[minor][minor] = 1.0
			if qrtMinor[minor] != 0.0 {
				for col := minor; col < m; col++ {
					var alpha float64
					for row := minor; row < m; row++ {
						alpha -= qta[col][row] * qrtMinor[row]
					}
					alpha /= qrd.rDiag[minor] * qrtMinor[minor]

					for row := minor; row < m; row++ {
						qta[col][row] += -alpha * qrtMinor[row]
					}
				}
			}
		}
		var err error
		qrd.cachedQT, err = NewRealMatrixFromSlices(qta)
		if err != nil {
			panic(err)
		}
	}

	// return the cached matrix
	return qrd.cachedQT
}

/**
 * Returns the Householder reflector vectors.
 * H is a lower trapezoidal matrix whose columns represent
 * each successive Householder reflector  This matrix is used
 * to compute Q.
 */
func (qrd *QRDecomposition) H() RealMatrix {
	if qrd.cachedH == nil {

		n := len(qrd.qrt)
		m := len(qrd.qrt[0])
		ha := make([][]float64, m)
		for i := 0; i < m; i++ {
			ha[i] = make([]float64, n)
			for j := 0; j < int(math.Min(float64(i+1), float64(n))); j++ {
				ha[i][j] = qrd.qrt[j][i] / -qrd.rDiag[j]
			}
		}
		var err error
		qrd.cachedH, err = NewRealMatrixFromSlices(ha)
		if err != nil {
			panic(err)
		}
	}

	// return the cached matrix
	return qrd.cachedH
}

/**
 * Get a solver for finding the A &times; X = B solution in least square sense.
 *
 * Least Square sense means a solver can be computed for an overdetermined system,
 * (i.e. a system with more equations than unknowns, which corresponds to a tall A
 * matrix with more rows than columns). In any case, if the matrix is singular
 * within the tolerance set at QRDecomposition(RealMatrix, float64) construction, an error will be triggered when
 * the DecompositionSolver.Solve() method will be called.
 */
func (qrd *QRDecomposition) Solver() DecompositionSolver {
	return newQRDecompositionSolver(qrd)
}

type qrDecompositionSolver struct {
	qrd *QRDecomposition
}

func newQRDecompositionSolver(qrd *QRDecomposition) *qrDecompositionSolver {
	return &qrDecompositionSolver{qrd: qrd}
}

func (qrds *qrDecompositionSolver) IsNonSingular() bool {
	return !checkSingular(qrds.qrd.rDiag, qrds.qrd.threshold, false)
}

func checkSingular(diag []float64, min float64, raise bool) bool {
	len := len(diag)
	for i := 0; i < len; i++ {
		d := diag[i]
		if math.Abs(d) <= min {
			if raise {
				panic(singularMatrixErrorf(singular_matrix+" : "+number_too_small+" : "+index, d, min, i))
			} else {
				return true
			}
		}
	}
	return false
}

func (qrds *qrDecompositionSolver) SolveVector(b RealVector) RealVector {
	n := len(qrds.qrd.qrt)
	m := len(qrds.qrd.qrt[0])
	if b.Dimension() != m {
		panic(dimensionsMismatchSimpleErrorf(b.Dimension(), m))
	}

	checkSingular(qrds.qrd.rDiag, qrds.qrd.threshold, true)

	x := make([]float64, n)
	y := b.ToArray()

	// apply Householder transforms to solve Q.y = b
	for minor := 0; minor < int(math.Min(float64(m), float64(n))); minor++ {

		qrtMinor := qrds.qrd.qrt[minor]
		var dotProduct float64
		for row := minor; row < m; row++ {
			dotProduct += y[row] * qrtMinor[row]
		}
		dotProduct /= qrds.qrd.rDiag[minor] * qrtMinor[minor]

		for row := minor; row < m; row++ {
			y[row] += dotProduct * qrtMinor[row]
		}
	}

	// solve triangular system R.x = y
	for row := len(qrds.qrd.rDiag) - 1; row >= 0; row-- {
		y[row] /= qrds.qrd.rDiag[row]
		yRow := y[row]
		qrtRow := qrds.qrd.qrt[row]
		x[row] = yRow
		for i := 0; i < row; i++ {
			y[i] -= yRow * qrtRow[i]
		}
	}

	vec := new(ArrayRealVector)
	vec.data = x
	return vec
}

func (qrds *qrDecompositionSolver) SolveMatrix(b RealMatrix) RealMatrix {
	n := len(qrds.qrd.qrt)
	m := len(qrds.qrd.qrt[0])
	if b.RowDimension() != m {
		panic(dimensionsMismatchSimpleErrorf(b.RowDimension(), m))
	}
	checkSingular(qrds.qrd.rDiag, qrds.qrd.threshold, true)

	columns := b.ColumnDimension()
	blockSize := BLOCK_SIZE
	cBlocks := (columns + blockSize - 1) / blockSize
	xBlocks := createBlocksLayout(n, columns)
	y := make([][]float64, b.RowDimension())
	for i := 0; i < b.RowDimension(); i++ {
		y[i] = make([]float64, blockSize)
	}

	alpha := make([]float64, blockSize)

	for kBlock := 0; kBlock < cBlocks; kBlock++ {
		kStart := kBlock * blockSize
		kEnd := int(math.Min(float64(kStart+blockSize), float64(columns)))
		kWidth := kEnd - kStart

		// get the right hand side vector
		CopySubMatrix(b, 0, m-1, kStart, kEnd-1, y)

		// apply Householder transforms to solve Q.y = b
		for minor := 0; minor < int(math.Min(float64(m), float64(n))); minor++ {
			qrtMinor := qrds.qrd.qrt[minor]
			factor := 1.0 / (qrds.qrd.rDiag[minor] * qrtMinor[minor])

			for i := 0; i < kWidth; i++ {
				alpha[i] = 0.
			}

			for row := minor; row < m; row++ {
				d := qrtMinor[row]
				yRow := y[row]
				for k := 0; k < kWidth; k++ {
					alpha[k] += d * yRow[k]
				}
			}
			for k := 0; k < kWidth; k++ {
				alpha[k] *= factor
			}

			for row := minor; row < m; row++ {
				d := qrtMinor[row]
				yRow := y[row]
				for k := 0; k < kWidth; k++ {
					yRow[k] += alpha[k] * d
				}
			}
		}

		// solve triangular system R.x = y
		for j := len(qrds.qrd.rDiag) - 1; j >= 0; j-- {
			jBlock := j / blockSize
			jStart := jBlock * blockSize
			factor := 1.0 / qrds.qrd.rDiag[j]
			yJ := y[j]
			xBlock := xBlocks[jBlock*cBlocks+kBlock]
			index := (j - jStart) * kWidth
			for k := 0; k < kWidth; k++ {
				yJ[k] *= factor
				xBlock[index] = yJ[k]
				index++
			}

			qrtJ := qrds.qrd.qrt[j]
			for i := 0; i < j; i++ {
				rIJ := qrtJ[i]
				yI := y[i]
				for k := 0; k < kWidth; k++ {
					yI[k] -= yJ[k] * rIJ
				}
			}
		}
	}

	mat, err := NewBlockRealMatrixFromBlockData(n, columns, xBlocks)
	if err != nil {
		panic(err)
	}

	return mat
}

func (qrds *qrDecompositionSolver) Inverse() RealMatrix {
	mat, err := NewRealIdentityMatrix(len(qrds.qrd.qrt[0]))
	if err != nil {
		panic(err)
	}

	return qrds.SolveMatrix(mat)
}
