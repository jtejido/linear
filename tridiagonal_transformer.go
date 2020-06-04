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
 * Class transforming a symmetrical matrix to tridiagonal shape.
 * <p>A symmetrical m &times; m matrix A can be written as the product of three matrices:
 * A = Q &times; T &times; Q<sup>T</sup> with Q an orthogonal matrix and T a symmetrical
 * tridiagonal matrix. Both Q and T are m &times; m matrices.</p>
 * <p>This implementation only uses the upper part of the matrix, the part below the
 * diagonal is not accessed at all.</p>
 * <p>Transformation to tridiagonal shape is often not a goal by itself, but it is
 * an intermediate step in more general decomposition algorithms like {@link
 * EigenDecomposition eigen decomposition}. This class is therefore intended for internal
 * use by the library and is not public. As a consequence of this explicitly limited scope,
 * many methods directly returns references to internal arrays, not copies.</p>
 */
type TriDiagonalTransformer struct {
	householderVectors         [][]float64
	main, secondary            []float64
	cachedQ, cachedQt, cachedT RealMatrix
}

/**
 * Build the transformation to tridiagonal shape of a symmetrical matrix.
 * The specified matrix is assumed to be symmetrical without any check.
 * Only the upper triangular part of the matrix is used.
 */
func NewTriDiagonalTransformer(matrix RealMatrix) (*TriDiagonalTransformer, error) {
	if !IsSquare(matrix) {
		return nil, nonSquareMatrixSimpleErrorf(matrix.RowDimension(), matrix.ColumnDimension())
	}

	ans := new(TriDiagonalTransformer)
	m := matrix.RowDimension()
	ans.householderVectors = matrix.Data()
	ans.main = make([]float64, m)
	ans.secondary = make([]float64, m-1)

	// transform matrix
	ans.transform()
	return ans, nil
}

/**
 * Returns the matrix Q of the transform.
 * Q is an orthogonal matrix, i.e. its transpose is also its inverse.
 */
func (tdf *TriDiagonalTransformer) Q() RealMatrix {
	if tdf.cachedQ == nil {
		tdf.cachedQ = tdf.QT().Transpose()
	}
	return tdf.cachedQ
}

/**
 * Returns the transpose of the matrix Q of the transform.
 * Q is an orthogonal matrix, i.e. its transpose is also its inverse.
 */
func (tdf *TriDiagonalTransformer) QT() RealMatrix {
	if tdf.cachedQt == nil {
		m := len(tdf.householderVectors)
		qta := make([][]float64, m)
		for i := 0; i < m; i++ {
			qta[i] = make([]float64, m)
		}

		// build up first part of the matrix by applying Householder transforms
		for k := m - 1; k >= 1; k-- {
			hK := tdf.householderVectors[k-1]
			qta[k][k] = 1
			if hK[k] != 0.0 {
				inv := 1.0 / (tdf.secondary[k-1] * hK[k])
				beta := 1.0 / tdf.secondary[k-1]
				qta[k][k] = 1 + beta*hK[k]
				for i := k + 1; i < m; i++ {
					qta[k][i] = beta * hK[i]
				}
				for j := k + 1; j < m; j++ {
					beta = 0
					for i := k + 1; i < m; i++ {
						beta += qta[j][i] * hK[i]
					}
					beta *= inv
					qta[j][k] = beta * hK[k]
					for i := k + 1; i < m; i++ {
						qta[j][i] += beta * hK[i]
					}
				}
			}
		}
		qta[0][0] = 1
		var err error
		tdf.cachedQt, err = NewRealMatrixFromSlices(qta)
		if err != nil {
			panic(err)
		}
	}

	// return the cached matrix
	return tdf.cachedQt
}

/**
 * Returns the tridiagonal matrix T of the transform.
 */
func (tdf *TriDiagonalTransformer) T() RealMatrix {
	if tdf.cachedT == nil {
		m := len(tdf.main)
		ta := make([][]float64, m)
		for i := 0; i < m; i++ {
			if ta[i] == nil {
				ta[i] = make([]float64, m)
			}
			ta[i][i] = tdf.main[i]
			if i > 0 {
				ta[i][i-1] = tdf.secondary[i-1]
			}
			if i < len(tdf.main)-1 {
				ta[i][i+1] = tdf.secondary[i]
			}
		}
		var err error
		tdf.cachedT, err = NewRealMatrixFromSlices(ta)
		if err != nil {
			panic(err)
		}
	}

	// return the cached matrix
	return tdf.cachedT
}

/**
 * Get the Householder vectors of the transform.
 * Note that since this class is only intended for internal use,
 * it returns directly a reference to its internal arrays, not a copy.
 */
func (tdf *TriDiagonalTransformer) HouseholderVectorsRef() [][]float64 {
	return tdf.householderVectors
}

/**
 * Get the main diagonal elements of the matrix T of the transform.
 * Note that since this class is only intended for internal use,
 * it returns directly a reference to its internal arrays, not a copy.
 */
func (tdf *TriDiagonalTransformer) MainDiagonalRef() []float64 {
	return tdf.main
}

/**
 * Get the secondary diagonal elements of the matrix T of the transform.
 * Note that since this class is only intended for internal use,
 * it returns directly a reference to its internal arrays, not a copy.
 */
func (tdf *TriDiagonalTransformer) SecondaryDiagonalRef() []float64 {
	return tdf.secondary
}

/**
 * Transform original matrix to tridiagonal form.
 * Transformation is done using Householder transforms.
 */
func (tdf *TriDiagonalTransformer) transform() {
	m := len(tdf.householderVectors)
	z := make([]float64, m)
	for k := 0; k < m-1; k++ {

		//zero-out a row and a column simultaneously
		hK := tdf.householderVectors[k]
		tdf.main[k] = hK[k]
		var xNormSqr float64
		for j := k + 1; j < m; j++ {
			c := hK[j]
			xNormSqr += c * c
		}

		a := math.Sqrt(xNormSqr)
		if hK[k+1] > 0 {
			a *= -1
		}
		tdf.secondary[k] = a
		if a != 0.0 {
			// apply Householder transform from left and right simultaneously

			hK[k+1] -= a
			beta := -1 / (a * hK[k+1])

			// compute a = beta A v, where v is the Householder vector
			// this loop is written in such a way
			//   1) only the upper triangular part of the matrix is accessed
			//   2) access is cache-friendly for a matrix stored in rows

			for i := k + 1; i < m; i++ {
				z[i] = 0
			}
			for i := k + 1; i < m; i++ {
				hI := tdf.householderVectors[i]
				hKI := hK[i]
				zI := hI[i] * hKI
				for j := i + 1; j < m; j++ {
					hIJ := hI[j]
					zI += hIJ * hK[j]
					z[j] += hIJ * hKI
				}
				z[i] = beta * (z[i] + zI)
			}

			// compute gamma = beta vT z / 2
			var gamma float64
			for i := k + 1; i < m; i++ {
				gamma += z[i] * hK[i]
			}
			gamma *= beta / 2

			// compute z = z - gamma v
			for i := k + 1; i < m; i++ {
				z[i] -= gamma * hK[i]
			}

			// update matrix: A = A - v zT - z vT
			// only the upper triangular part of the matrix is updated
			for i := k + 1; i < m; i++ {
				hI := tdf.householderVectors[i]
				for j := i; j < m; j++ {
					hI[j] -= hK[i]*z[j] + z[i]*hK[j]
				}
			}
		}
	}
	tdf.main[m-1] = tdf.householderVectors[m-1][m-1]
}
