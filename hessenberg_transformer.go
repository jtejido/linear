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
 * Class transforming a general real matrix to Hessenberg form.
 * A m &times; m matrix A can be written as the product of three matrices: A = P
 * &times; H &times; P<sup>T</sup> with P an orthogonal matrix and H a Hessenberg
 * matrix. Both P and H are m &times; m matrices.
 * Transformation to Hessenberg form is often not a goal by itself, but it is an
 * intermediate step in more general decomposition algorithms like
 * EigenDecomposition. This class is therefore
 * intended for internal use by the library and is not public. As a consequence
 * of this explicitly limited scope, many methods directly returns references to
 * internal arrays, not copies.
 * This class is based on the method orthes in class EigenvalueDecomposition
 * from the JAMA library.
 */
type HessenbergTransformer struct {
	householderVectors         [][]float64
	ort                        []float64
	cachedP, cachedPt, cachedH RealMatrix
}

/**
 * Build the transformation to Hessenberg form of a general matrix.
 */
func NewHessenbergTransformer(matrix RealMatrix) (*HessenbergTransformer, error) {

	if !IsSquare(matrix) {
		return nil, nonSquareMatrixSimpleErrorf(matrix.RowDimension(), matrix.ColumnDimension())
	}
	ans := new(HessenbergTransformer)

	m := matrix.RowDimension()
	ans.householderVectors = matrix.Data()
	ans.ort = make([]float64, m)

	// transform matrix
	ans.transform()

	return ans, nil
}

/**
 * Returns the matrix P of the transform.
 * P is an orthogonal matrix, i.e. its inverse is also its transpose.
 */
func (hs *HessenbergTransformer) P() RealMatrix {
	if hs.cachedP == nil {
		n := len(hs.householderVectors)
		high := n - 1
		pa := make([][]float64, n)

		for i := 0; i < n; i++ {
			if pa[i] == nil {
				pa[i] = make([]float64, n)
			}
			for j := 0; j < n; j++ {
				if i == j {
					pa[i][j] = 1
				}
			}
		}

		for m := high - 1; m >= 1; m-- {
			if hs.householderVectors[m][m-1] != 0.0 {
				for i := m + 1; i <= high; i++ {
					hs.ort[i] = hs.householderVectors[i][m-1]
				}

				for j := m; j <= high; j++ {
					g := 0.0

					for i := m; i <= high; i++ {
						g += hs.ort[i] * pa[i][j]
					}

					// Double division avoids possible underflow
					g = (g / hs.ort[m]) / hs.householderVectors[m][m-1]

					for i := m; i <= high; i++ {
						pa[i][j] += g * hs.ort[i]
					}
				}
			}
		}
		var err error
		hs.cachedP, err = NewRealMatrixFromSlices(pa)
		if err != nil {
			panic(err)
		}
	}
	return hs.cachedP
}

/**
 * Returns the transpose of the matrix P of the transform.
 * P is an orthogonal matrix, i.e. its inverse is also its transpose.
 */
func (hs *HessenbergTransformer) PT() RealMatrix {
	if hs.cachedPt == nil {
		hs.cachedPt = hs.P().Transpose()
	}

	// return the cached matrix
	return hs.cachedPt
}

/**
 * Returns the Hessenberg matrix H of the transform.
 */
func (hs *HessenbergTransformer) H() RealMatrix {
	if hs.cachedH == nil {
		m := len(hs.householderVectors)
		h := make([][]float64, m)
		for i := 0; i < m; i++ {
			if h[i] == nil {
				h[i] = make([]float64, m)
			}
			if i > 0 {
				// copy the entry of the lower sub-diagonal
				h[i][i-1] = hs.householderVectors[i][i-1]
			}

			// copy upper triangular part of the matrix
			for j := i; j < m; j++ {
				h[i][j] = hs.householderVectors[i][j]
			}
		}
		var err error
		hs.cachedH, err = NewRealMatrixFromSlices(h)
		if err != nil {
			panic(err)
		}
	}

	// return the cached matrix
	return hs.cachedH
}

/**
 * Get the Householder vectors of the transform.
 * Note that since this class is only intended for internal use, it returns
 * directly a reference to its internal arrays, not a copy.
 */
func (hs *HessenbergTransformer) HouseholderVectorsRef() [][]float64 {
	return hs.householderVectors
}

/**
 * Transform original matrix to Hessenberg form.
 * Transformation is done using Householder transforms.
 */
func (hs *HessenbergTransformer) transform() {
	n := len(hs.householderVectors)
	high := n - 1

	for m := 1; m <= high-1; m++ {
		// Scale column.
		var scale float64
		for i := m; i <= high; i++ {
			scale += math.Abs(hs.householderVectors[i][m-1])
		}

		if !Equals(scale, 0) {
			// Compute Householder transformation.
			h := 0.
			for i := high; i >= m; i-- {
				hs.ort[i] = hs.householderVectors[i][m-1] / scale
				h += hs.ort[i] * hs.ort[i]
			}
			g := math.Sqrt(h)
			if hs.ort[m] > 0 {
				g *= -1
			}

			h -= hs.ort[m] * g
			hs.ort[m] -= g

			// Apply Householder similarity transformation
			// H = (I - u*u' / h) * H * (I - u*u' / h)

			for j := m; j < n; j++ {
				var f float64
				for i := high; i >= m; i-- {
					f += hs.ort[i] * hs.householderVectors[i][j]
				}
				f /= h
				for i := m; i <= high; i++ {
					hs.householderVectors[i][j] -= f * hs.ort[i]
				}
			}

			for i := 0; i <= high; i++ {
				var f float64
				for j := high; j >= m; j-- {
					f += hs.ort[j] * hs.householderVectors[i][j]
				}
				f /= h
				for j := m; j <= high; j++ {
					hs.householderVectors[i][j] -= f * hs.ort[j]
				}
			}

			hs.ort[m] = scale * hs.ort[m]
			hs.householderVectors[m][m-1] = scale * g
		}
	}
}
