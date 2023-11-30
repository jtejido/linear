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

var (
	max_iter_st = 100
)

/**
 * Class transforming a general real matrix to Schur form.
 * <p>A m &times; m matrix A can be written as the product of three matrices: A = P
 * &times; T &times; P<sup>T</sup> with P an orthogonal matrix and T an quasi-triangular
 * matrix. Both P and T are m &times; m matrices.</p>
 * <p>Transformation to Schur form is often not a goal by itself, but it is an
 * intermediate step in more general decomposition algorithms like
 * {@link EigenDecomposition eigen decomposition}. This class is therefore
 * intended for internal use by the library and is not public. As a consequence
 * of this explicitly limited scope, many methods directly returns references to
 * internal arrays, not copies.</p>
 */
type SchurTransformer struct {
	matrixP, matrixT           [][]float64
	cachedP, cachedT, cachedPt RealMatrix
}

func NewSchurTransformer(matrix RealMatrix) (*SchurTransformer, error) {
	if !IsSquare(matrix) {
		return nil, nonSquareMatrixSimpleErrorf(matrix.RowDimension(), matrix.ColumnDimension())
	}
	ans := new(SchurTransformer)
	transformer, err := NewHessenbergTransformer(matrix)
	if err != nil {
		return nil, err
	}

	ans.matrixT = transformer.H().Data()
	ans.matrixP = transformer.P().Data()

	// transform matrix
	ans.transform()
	return ans, nil
}

/**
 * Returns the matrix P of the transform.
 * P is an orthogonal matrix, i.e. its inverse is also its transpose.
 */
func (st *SchurTransformer) P() RealMatrix {
	if st.cachedP == nil {
		mat, err := NewRealMatrixFromSlices(st.matrixP)
		if err != nil {
			panic(err)
		}

		st.cachedP = mat
	}
	return st.cachedP
}

/**
 * Returns the transpose of the matrix P of the transform.
 * P is an orthogonal matrix, i.e. its inverse is also its transpose.
 */
func (st *SchurTransformer) PT() RealMatrix {
	if st.cachedPt == nil {
		st.cachedPt = st.P().Transpose()
	}

	// return the cached matrix
	return st.cachedPt
}

/**
 * Returns the quasi-triangular Schur matrix T of the transform.
 *
 * @return the T matrix
 */
func (st *SchurTransformer) T() RealMatrix {
	if st.cachedT == nil {
		mat, err := NewRealMatrixFromSlices(st.matrixT)
		if err != nil {
			panic(err)
		}

		st.cachedT = mat
	}

	// return the cached matrix
	return st.cachedT
}

/**
 * Transform original matrix to Schur form.
 */
func (st *SchurTransformer) transform() {
	n := len(st.matrixT)

	// compute matrix norm
	norm := st.norm()

	// shift information
	shift := new(ShiftInfo)

	// Outer loop over eigenvalue index
	iteration := 0
	iu := n - 1
	for iu >= 0 {

		// Look for single small sub-diagonal element
		il := st.findSmallSubDiagonalElement(iu, norm)

		// Check for convergence
		if il == iu {
			// One root found
			st.matrixT[iu][iu] += shift.exShift
			iu--
			iteration = 0
		} else if il == iu-1 {
			// Two roots found
			p := (st.matrixT[iu-1][iu-1] - st.matrixT[iu][iu]) / 2.0
			q := p*p + st.matrixT[iu][iu-1]*st.matrixT[iu-1][iu]
			st.matrixT[iu][iu] += shift.exShift
			st.matrixT[iu-1][iu-1] += shift.exShift

			if q >= 0 {
				z := math.Sqrt(math.Abs(q))
				if p >= 0 {
					z = p + z
				} else {
					z = p - z
				}
				x := st.matrixT[iu][iu-1]
				s := math.Abs(x) + math.Abs(z)
				p = x / s
				q = z / s
				r := math.Sqrt(p*p + q*q)
				p /= r
				q /= r

				// Row modification
				for j := iu - 1; j < n; j++ {
					z = st.matrixT[iu-1][j]
					st.matrixT[iu-1][j] = q*z + p*st.matrixT[iu][j]
					st.matrixT[iu][j] = q*st.matrixT[iu][j] - p*z
				}

				// Column modification
				for i := 0; i <= iu; i++ {
					z = st.matrixT[i][iu-1]
					st.matrixT[i][iu-1] = q*z + p*st.matrixT[i][iu]
					st.matrixT[i][iu] = q*st.matrixT[i][iu] - p*z
				}

				// Accumulate transformations
				for i := 0; i <= n-1; i++ {
					z = st.matrixP[i][iu-1]
					st.matrixP[i][iu-1] = q*z + p*st.matrixP[i][iu]
					st.matrixP[i][iu] = q*st.matrixP[i][iu] - p*z
				}
			}
			iu -= 2
			iteration = 0
		} else {
			// No convergence yet
			st.computeShift(il, iu, iteration, shift)

			// stop transformation after too many iterations
			iteration++
			if iteration > max_iter_st {
				panic(maxIterationErrorf(convergence_failed, max_iter_st))
			}

			// the initial houseHolder vector for the QR step
			hVec := make([]float64, 3)

			im := st.initQRStep(il, iu, shift, hVec)
			st.performDoubleQRStep(il, im, iu, shift, hVec)
		}
	}
}

/**
 * Computes the L1 norm of the (quasi-)triangular matrix T.
 */
func (st *SchurTransformer) norm() float64 {
	var norm float64
	for i := 0; i < len(st.matrixT); i++ {
		// as matrix T is (quasi-)triangular, also take the sub-diagonal element into account
		for j := int(math.Max(float64(i-1), 0.)); j < len(st.matrixT); j++ {
			norm += math.Abs(st.matrixT[i][j])
		}
	}
	return norm
}

/**
 * Find the first small sub-diagonal element and returns its index.
 */
func (st *SchurTransformer) findSmallSubDiagonalElement(startIdx int, norm float64) int {
	l := startIdx
	for l > 0 {
		s := math.Abs(st.matrixT[l-1][l-1]) + math.Abs(st.matrixT[l][l])
		if s == 0.0 {
			s = norm
		}
		if math.Abs(st.matrixT[l][l-1]) < doubleeps*s {
			break
		}
		l--
	}
	return l
}

/**
 * Compute the shift for the current iteration.
 */
func (st *SchurTransformer) computeShift(l, idx, iteration int, shift *ShiftInfo) {
	// Form shift
	shift.x = st.matrixT[idx][idx]
	shift.y = 0.
	shift.w = 0.
	if l < idx {
		shift.y = st.matrixT[idx-1][idx-1]
		shift.w = st.matrixT[idx][idx-1] * st.matrixT[idx-1][idx]
	}

	// Wilkinson's original ad hoc shift
	if iteration == 10 {
		shift.exShift += shift.x
		for i := 0; i <= idx; i++ {
			st.matrixT[i][i] -= shift.x
		}
		s := math.Abs(st.matrixT[idx][idx-1]) + math.Abs(st.matrixT[idx-1][idx-2])
		shift.x = 0.75 * s
		shift.y = 0.75 * s
		shift.w = -0.4375 * s * s
	}

	// MATLAB's new ad hoc shift
	if iteration == 30 {
		s := (shift.y - shift.x) / 2.0
		s = s*s + shift.w
		if s > 0.0 {
			s = math.Sqrt(s)
			if shift.y < shift.x {
				s = -s
			}
			s = shift.x - shift.w/((shift.y-shift.x)/2.0+s)
			for i := 0; i <= idx; i++ {
				st.matrixT[i][i] -= s
			}
			shift.exShift += s
			shift.x = 0.964
			shift.y = 0.964
			shift.w = 0.964
		}
	}
}

/**
 * Initialize the householder vectors for the QR step.
 */
func (st *SchurTransformer) initQRStep(il, iu int, shift *ShiftInfo, hVec []float64) int {
	// Look for two consecutive small sub-diagonal elements
	im := iu - 2
	for im >= il {
		z := st.matrixT[im][im]
		r := shift.x - z
		s := shift.y - z
		hVec[0] = (r*s-shift.w)/st.matrixT[im+1][im] + st.matrixT[im][im+1]
		hVec[1] = st.matrixT[im+1][im+1] - z - r - s
		hVec[2] = st.matrixT[im+2][im+1]

		if im == il {
			break
		}

		lhs := math.Abs(st.matrixT[im][im-1]) * (math.Abs(hVec[1]) + math.Abs(hVec[2]))
		rhs := math.Abs(hVec[0]) * (math.Abs(st.matrixT[im-1][im-1]) +
			math.Abs(z) +
			math.Abs(st.matrixT[im+1][im+1]))

		if lhs < doubleeps*rhs {
			break
		}
		im--
	}

	return im
}

/**
 * Perform a double QR step involving rows l:idx and columns m:n
 */
func (st *SchurTransformer) performDoubleQRStep(il, im, iu int, shift *ShiftInfo, hVec []float64) {

	n := len(st.matrixT)
	p := hVec[0]
	q := hVec[1]
	r := hVec[2]

	for k := im; k <= iu-1; k++ {
		notlast := k != (iu - 1)
		if k != im {
			p = st.matrixT[k][k-1]
			q = st.matrixT[k+1][k-1]
			if notlast {
				r = st.matrixT[k+2][k-1]
			} else {
				r = 0.
			}

			shift.x = math.Abs(p) + math.Abs(q) + math.Abs(r)
			if equalsWithError(shift.x, 0.0, doubleeps) {
				continue
			}
			p /= shift.x
			q /= shift.x
			r /= shift.x
		}
		s := math.Sqrt(p*p + q*q + r*r)
		if p < 0.0 {
			s = -s
		}
		if s != 0.0 {
			if k != im {
				st.matrixT[k][k-1] = -s * shift.x
			} else if il != im {
				st.matrixT[k][k-1] = -st.matrixT[k][k-1]
			}
			p += s
			shift.x = p / s
			shift.y = q / s
			z := r / s
			q /= p
			r /= p

			// Row modification
			for j := k; j < n; j++ {
				p = st.matrixT[k][j] + q*st.matrixT[k+1][j]
				if notlast {
					p += r * st.matrixT[k+2][j]
					st.matrixT[k+2][j] -= p * z
				}
				st.matrixT[k][j] -= p * shift.x
				st.matrixT[k+1][j] -= p * shift.y
			}

			// Column modification
			for i := 0; i <= int(math.Min(float64(iu), float64(k)+3)); i++ {
				p = shift.x*st.matrixT[i][k] + shift.y*st.matrixT[i][k+1]
				if notlast {
					p += z * st.matrixT[i][k+2]
					st.matrixT[i][k+2] -= p * r
				}
				st.matrixT[i][k] -= p
				st.matrixT[i][k+1] -= p * q
			}

			// Accumulate transformations
			high := len(st.matrixT) - 1
			for i := 0; i <= high; i++ {
				p = shift.x*st.matrixP[i][k] + shift.y*st.matrixP[i][k+1]
				if notlast {
					p += z * st.matrixP[i][k+2]
					st.matrixP[i][k+2] -= p * r
				}
				st.matrixP[i][k] -= p
				st.matrixP[i][k+1] -= p * q
			}
		} // (s != 0)
	} // k loop

	// clean up pollution due to round-off errors
	for i := im + 2; i <= iu; i++ {
		st.matrixT[i][i-2] = 0.0
		if i > im+2 {
			st.matrixT[i][i-3] = 0.0
		}
	}
}

type ShiftInfo struct {
	x, y, w, exShift float64
}
