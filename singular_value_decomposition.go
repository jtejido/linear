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
	eps_svd  = 2.220446049250313e-16
	tiny_svd = 1.6033346880071782e-291
)

/**
 * Calculates the compact Singular Value Decomposition of a matrix.
 *
 * The Singular Value Decomposition of matrix A is a set of three matrices: U,
 * &Sigma; and V such that A = U &times; &Sigma; &times; V<sup>T</sup>. Let A be
 * a m &times; n matrix, then U is a m &times; p orthogonal matrix, &Sigma; is a
 * p &times; p diagonal matrix with positive or null elements, V is a p &times;
 * n orthogonal matrix (hence V<sup>T</sup> is also orthogonal) where
 * p=min(m,n).
 */
type SingularValueDecomposition struct {
	singularValues                                []float64
	m, n                                          int
	transposed                                    bool
	cachedU, cachedUt, cachedS, cachedV, cachedVt RealMatrix
	tol                                           float64
}

func NewSingularValueDecomposition(matrix RealMatrix) (ans *SingularValueDecomposition, err error) {
	var A [][]float64
	ans = new(SingularValueDecomposition)
	// "m" is always the largest dimension.
	if matrix.RowDimension() < matrix.ColumnDimension() {
		ans.transposed = true
		A = matrix.Transpose().Data()
		ans.m = matrix.ColumnDimension()
		ans.n = matrix.RowDimension()
	} else {
		A = matrix.Data()
		ans.m = matrix.RowDimension()
		ans.n = matrix.ColumnDimension()
	}
	m := ans.m
	n := ans.n
	ans.singularValues = make([]float64, n)
	U := make([][]float64, m)
	for i := 0; i < m; i++ {
		U[i] = make([]float64, n)
	}
	V := make([][]float64, n)
	for i := 0; i < n; i++ {
		V[i] = make([]float64, n)
	}

	e := make([]float64, n)
	work := make([]float64, m)
	// Reduce A to bidiagonal form, storing the diagonal elements
	// in s and the super-diagonal elements in e.
	nct := int(math.Min(float64(m-1), float64(n)))
	nrt := int(math.Max(0., float64(n-2)))
	for k := 0; k < int(math.Max(float64(nct), float64(nrt))); k++ {
		if k < nct {
			// Compute the transformation for the k-th column and
			// place the k-th diagonal in s[k].
			// Compute 2-norm of k-th column without under/overflow.
			ans.singularValues[k] = 0
			for i := k; i < m; i++ {
				ans.singularValues[k] = math.Hypot(ans.singularValues[k], A[i][k])
			}
			if ans.singularValues[k] != 0 {
				if A[k][k] < 0 {
					ans.singularValues[k] = -ans.singularValues[k]
				}
				for i := k; i < m; i++ {
					A[i][k] /= ans.singularValues[k]
				}
				A[k][k] += 1
			}
			ans.singularValues[k] = -ans.singularValues[k]
		}
		for j := k + 1; j < n; j++ {
			if k < nct && ans.singularValues[k] != 0 {
				// Apply the transformation.
				t := 0.
				for i := k; i < m; i++ {
					t += A[i][k] * A[i][j]
				}
				t = -t / A[k][k]
				for i := k; i < m; i++ {
					A[i][j] += t * A[i][k]
				}
			}
			// Place the k-th row of A into e for the
			// subsequent calculation of the row transformation.
			e[j] = A[k][j]
		}
		if k < nct {
			// Place the transformation in U for subsequent back
			// multiplication.
			for i := k; i < m; i++ {
				U[i][k] = A[i][k]
			}
		}
		if k < nrt {
			// Compute the k-th row transformation and place the
			// k-th super-diagonal in e[k].
			// Compute 2-norm without under/overflow.
			e[k] = 0
			for i := k + 1; i < n; i++ {
				e[k] = math.Hypot(e[k], e[i])
			}
			if e[k] != 0 {
				if e[k+1] < 0 {
					e[k] = -e[k]
				}
				for i := k + 1; i < n; i++ {
					e[i] /= e[k]
				}
				e[k+1] += 1
			}
			e[k] = -e[k]
			if k+1 < m &&
				e[k] != 0 {
				// Apply the transformation.
				for i := k + 1; i < m; i++ {
					work[i] = 0
				}
				for j := k + 1; j < n; j++ {
					for i := k + 1; i < m; i++ {
						work[i] += e[j] * A[i][j]
					}
				}
				for j := k + 1; j < n; j++ {
					t := -e[j] / e[k+1]
					for i := k + 1; i < m; i++ {
						A[i][j] += t * work[i]
					}
				}
			}

			// Place the transformation in V for subsequent
			// back multiplication.
			for i := k + 1; i < n; i++ {
				V[i][k] = e[i]
			}
		}
	}
	// Set up the final bidiagonal matrix or order p.
	p := n
	if nct < n {
		ans.singularValues[nct] = A[nct][nct]
	}
	if m < p {
		ans.singularValues[p-1] = 0
	}
	if nrt+1 < p {
		e[nrt] = A[nrt][p-1]
	}
	e[p-1] = 0

	// Generate U.
	for j := nct; j < n; j++ {
		for i := 0; i < m; i++ {
			U[i][j] = 0
		}
		U[j][j] = 1
	}
	for k := nct - 1; k >= 0; k-- {
		if ans.singularValues[k] != 0 {
			for j := k + 1; j < n; j++ {
				t := 0.
				for i := k; i < m; i++ {
					t += U[i][k] * U[i][j]
				}
				t = -t / U[k][k]
				for i := k; i < m; i++ {
					U[i][j] += t * U[i][k]
				}
			}
			for i := k; i < m; i++ {
				U[i][k] = -U[i][k]
			}
			U[k][k] = 1 + U[k][k]
			for i := 0; i < k-1; i++ {
				U[i][k] = 0
			}
		} else {
			for i := 0; i < m; i++ {
				U[i][k] = 0
			}
			U[k][k] = 1
		}
	}

	// Generate V.
	for k := n - 1; k >= 0; k-- {
		if k < nrt && e[k] != 0 {
			for j := k + 1; j < n; j++ {
				t := 0.
				for i := k + 1; i < n; i++ {
					t += V[i][k] * V[i][j]
				}
				t = -t / V[k+1][k]
				for i := k + 1; i < n; i++ {
					V[i][j] += t * V[i][k]
				}
			}
		}
		for i := 0; i < n; i++ {
			V[i][k] = 0
		}
		V[k][k] = 1
	}

	// Main iteration loop for the singular values.
	pp := p - 1
	for p > 0 {
		var k, kase int
		// Here is where a test for too many iterations would go.
		// This section of the program inspects for
		// negligible elements in the s and e arrays.  On
		// completion the variables kase and k are set as follows.
		// kase = 1     if s(p) and e[k-1] are negligible and k<p
		// kase = 2     if s(k) is negligible and k<p
		// kase = 3     if e[k-1] is negligible, k<p, and
		//              s(k), ..., s(p) are not negligible (qr step).
		// kase = 4     if e(p-1) is negligible (convergence).
		for k = p - 2; k >= 0; k-- {
			threshold := tiny_svd + eps_svd*(math.Abs(ans.singularValues[k])+math.Abs(ans.singularValues[k+1]))

			// the following condition is written this way in order
			// to break out of the loop when NaN occurs, writing it
			// as "if (FastMath.abs(e[k]) <= threshold)" would loop
			// indefinitely in case of NaNs because comparison on NaNs
			// always return false, regardless of what is checked
			// see issue MATH-947
			if !(math.Abs(e[k]) > threshold) {
				e[k] = 0
				break
			}

		}

		if k == p-2 {
			kase = 4
		} else {
			var ks int
			for ks = p - 1; ks >= k; ks-- {
				if ks == k {
					break
				}

				t := 0.

				if ks != p {
					t = math.Abs(e[ks])
				}

				if ks != k+1 {
					t += math.Abs(e[ks-1])
				}

				if math.Abs(ans.singularValues[ks]) <= tiny_svd+eps_svd*t {
					ans.singularValues[ks] = 0
					break
				}
			}
			if ks == k {
				kase = 3
			} else if ks == p-1 {
				kase = 1
			} else {
				kase = 2
				k = ks
			}
		}
		k++
		// Perform the task indicated by kase.
		switch kase {
		// Deflate negligible s(p).
		case 1:
			{
				f := e[p-2]
				e[p-2] = 0
				for j := p - 2; j >= k; j-- {
					t := math.Hypot(ans.singularValues[j], f)
					cs := ans.singularValues[j] / t
					sn := f / t
					ans.singularValues[j] = t
					if j != k {
						f = -sn * e[j-1]
						e[j-1] = cs * e[j-1]
					}

					for i := 0; i < n; i++ {
						t = cs*V[i][j] + sn*V[i][p-1]
						V[i][p-1] = -sn*V[i][j] + cs*V[i][p-1]
						V[i][j] = t
					}
				}
			}
			break
		// Split at negligible s(k).
		case 2:
			{
				f := e[k-1]
				e[k-1] = 0
				for j := k; j < p; j++ {
					t := math.Hypot(ans.singularValues[j], f)
					cs := ans.singularValues[j] / t
					sn := f / t
					ans.singularValues[j] = t
					f = -sn * e[j]
					e[j] = cs * e[j]

					for i := 0; i < m; i++ {
						t = cs*U[i][j] + sn*U[i][k-1]
						U[i][k-1] = -sn*U[i][j] + cs*U[i][k-1]
						U[i][j] = t
					}
				}
			}
			break
		// Perform one qr step.
		case 3:
			{
				// Calculate the shift.
				maxPm1Pm2 := math.Max(math.Abs(ans.singularValues[p-1]), math.Abs(ans.singularValues[p-2]))
				scale := math.Max(math.Max(math.Max(maxPm1Pm2, math.Abs(e[p-2])), math.Abs(ans.singularValues[k])), math.Abs(e[k]))
				sp := ans.singularValues[p-1] / scale
				spm1 := ans.singularValues[p-2] / scale
				epm1 := e[p-2] / scale
				sk := ans.singularValues[k] / scale
				ek := e[k] / scale
				b := ((spm1+sp)*(spm1-sp) + epm1*epm1) / 2.0
				c := (sp * epm1) * (sp * epm1)
				shift := 0.
				if b != 0 || c != 0 {
					shift = math.Sqrt(b*b + c)
					if b < 0 {
						shift = -shift
					}
					shift = c / (b + shift)
				}
				f := (sk+sp)*(sk-sp) + shift
				g := sk * ek
				// Chase zeros.
				for j := k; j < p-1; j++ {
					t := math.Hypot(f, g)
					cs := f / t
					sn := g / t
					if j != k {
						e[j-1] = t
					}
					f = cs*ans.singularValues[j] + sn*e[j]
					e[j] = cs*e[j] - sn*ans.singularValues[j]
					g = sn * ans.singularValues[j+1]
					ans.singularValues[j+1] = cs * ans.singularValues[j+1]

					for i := 0; i < n; i++ {
						t = cs*V[i][j] + sn*V[i][j+1]
						V[i][j+1] = -sn*V[i][j] + cs*V[i][j+1]
						V[i][j] = t
					}
					t = math.Hypot(f, g)
					cs = f / t
					sn = g / t
					ans.singularValues[j] = t
					f = cs*e[j] + sn*ans.singularValues[j+1]
					ans.singularValues[j+1] = -sn*e[j] + cs*ans.singularValues[j+1]
					g = sn * e[j+1]
					e[j+1] = cs * e[j+1]
					if j < m-1 {
						for i := 0; i < m; i++ {
							t = cs*U[i][j] + sn*U[i][j+1]
							U[i][j+1] = -sn*U[i][j] + cs*U[i][j+1]
							U[i][j] = t
						}
					}
				}
				e[p-2] = f
			}
			break
		// Convergence.
		default:
			{
				// Make the singular values positive.
				if ans.singularValues[k] <= 0 {
					if ans.singularValues[k] < 0 {
						ans.singularValues[k] = -ans.singularValues[k]
					} else {
						ans.singularValues[k] = 0
					}

					for i := 0; i <= pp; i++ {
						V[i][k] = -V[i][k]
					}
				}
				// Order the singular values.
				for k < pp {
					if ans.singularValues[k] >= ans.singularValues[k+1] {
						break
					}
					t := ans.singularValues[k]
					ans.singularValues[k] = ans.singularValues[k+1]
					ans.singularValues[k+1] = t
					if k < n-1 {
						for i := 0; i < n; i++ {
							t = V[i][k+1]
							V[i][k+1] = V[i][k]
							V[i][k] = t
						}
					}
					if k < m-1 {
						for i := 0; i < m; i++ {
							t = U[i][k+1]
							U[i][k+1] = U[i][k]
							U[i][k] = t
						}
					}
					k++
				}
				p--
			}
			break
		}
	}

	// Set the small value tolerance used to calculate rank and pseudo-inverse
	ans.tol = math.Max(float64(m)*ans.singularValues[0]*eps_svd, math.Sqrt(doubleeps))

	if !ans.transposed {
		ans.cachedU, err = NewRealMatrixFromSlices(U)
		if err != nil {
			return nil, err
		}
		ans.cachedV, err = NewRealMatrixFromSlices(V)
		if err != nil {
			return nil, err
		}
	} else {
		ans.cachedU, err = NewRealMatrixFromSlices(V)
		if err != nil {
			return nil, err
		}
		ans.cachedV, err = NewRealMatrixFromSlices(U)
		if err != nil {
			return nil, err
		}
	}

	return ans, nil
}

/**
 * Returns the matrix U of the decomposition.
 * U is an orthogonal matrix, i.e. its transpose is also its inverse.
 */
func (svd *SingularValueDecomposition) U() RealMatrix {
	// return the cached matrix
	return svd.cachedU

}

/**
 * Returns the transpose of the matrix U of the decomposition.
 * U is an orthogonal matrix, i.e. its transpose is also its inverse.
 */
func (svd *SingularValueDecomposition) UT() RealMatrix {
	if svd.cachedUt == nil {
		svd.cachedUt = svd.U().Transpose()
	}
	// return the cached matrix
	return svd.cachedUt
}

/**
 * Returns the diagonal matrix &Sigma; of the decomposition.
 * &Sigma; is a diagonal matrix. The singular values are provided in
 * non-increasing order, for compatibility with Jama.
 */
func (svd *SingularValueDecomposition) S() RealMatrix {
	if svd.cachedS == nil {
		// cache the matrix for subsequent calls
		var err error
		svd.cachedS, err = NewRealDiagonalMatrix(svd.singularValues)
		if err != nil {
			panic(err)
		}
	}
	return svd.cachedS
}

/**
 * Returns the diagonal elements of the matrix &Sigma; of the decomposition.
 * The singular values are provided in non-increasing order, for
 * compatibility with Jama.
 */
func (svd *SingularValueDecomposition) SingularValues() []float64 {
	return append([]float64{}, svd.singularValues...)
}

/**
 * Returns the matrix V of the decomposition.
 * V is an orthogonal matrix, i.e. its transpose is also its inverse.
 */
func (svd *SingularValueDecomposition) V() RealMatrix {
	// return the cached matrix
	return svd.cachedV
}

/**
 * Returns the transpose of the matrix V of the decomposition.
 * V is an orthogonal matrix, i.e. its transpose is also its inverse.
 */
func (svd *SingularValueDecomposition) VT() RealMatrix {
	if svd.cachedVt == nil {
		svd.cachedVt = svd.V().Transpose()
	}
	// return the cached matrix
	return svd.cachedVt
}

type svdPreservingVisitor struct {
	visit func(row, column int, value float64)
}

func (t svdPreservingVisitor) Start(rows, columns, startRow, endRow, startColumn, endColumn int) {}
func (t svdPreservingVisitor) Visit(row, column int, value float64)                              { t.visit(row, column, value) }
func (t svdPreservingVisitor) End() float64                                                      { return 0 }

/**
 * Returns the n &times; n covariance matrix.
 * The covariance matrix is V &times; J &times; V<sup>T</sup>
 * where J is the diagonal matrix of the inverse of the squares of
 * the singular values.
 */
func (svd *SingularValueDecomposition) Covariance(minSingularValue float64) RealMatrix {
	// get the number of singular values to consider
	p := len(svd.singularValues)
	var dimension int
	for dimension < p && svd.singularValues[dimension] >= minSingularValue {
		dimension++
	}

	if dimension == 0 {
		panic(numberIsTooLargeErrorf(too_large_cutoff_singular_value, minSingularValue, svd.singularValues[0], true))
	}

	data := make([][]float64, dimension)
	for i := 0; i < dimension; i++ {
		data[i] = make([]float64, p)
	}

	t := struct{ svdPreservingVisitor }{}

	t.visit = func(row, column int, value float64) {
		data[row][column] = value / svd.singularValues[row]
	}

	svd.VT().WalkInOptimizedOrderBounded(t, 0, dimension-1, 0, p-1)

	jv, err := NewArray2DRowRealMatrixFromSlices(data, true)
	if err != nil {
		panic(err)
	}

	return jv.Transpose().Multiply(jv)
}

/**
 * Returns the L<sub>2</sub> norm of the matrix.
 * The L<sub>2</sub> norm is max(|A &times; u|<sub>2</sub> /
 * |u|<sub>2</sub>), where |.|<sub>2</sub> denotes the vectorial 2-norm
 * (i.e. the traditional euclidian norm).
 */
func (svd *SingularValueDecomposition) Norm() float64 {
	return svd.singularValues[0]
}

/**
 * Return the condition number of the matrix.
 */
func (svd *SingularValueDecomposition) ConditionNumber() float64 {
	return svd.singularValues[0] / svd.singularValues[svd.n-1]
}

/**
 * Computes the inverse of the condition number.
 * In cases of rank deficiency, the ConditionNumber() will become undefined.
 */
func (svd *SingularValueDecomposition) InverseConditionNumber() float64 {
	return svd.singularValues[svd.n-1] / svd.singularValues[0]
}

/**
 * Return the effective numerical matrix rank.
 * The effective numerical rank is the number of non-negligible
 * singular values. The threshold used to identify non-negligible
 * terms is max(m,n) &times; ulp(s<sub>1</sub>) where ulp(s<sub>1</sub>)
 * is the least significant bit of the largest singular value.
 */
func (svd *SingularValueDecomposition) Rank() int {
	r := 0
	for i := 0; i < len(svd.singularValues); i++ {
		if svd.singularValues[i] > svd.tol {
			r++
		}
	}
	return r
}

/**
 * Get a solver for finding the A &times; X = B solution in least square sense.
 */
func (svd *SingularValueDecomposition) Solver() DecompositionSolver {
	return newSVDecompositionSolver(svd)
}

type svDecompositionSolver struct {
	svd           *SingularValueDecomposition
	pseudoInverse RealMatrix
}

func newSVDecompositionSolver(svd *SingularValueDecomposition) *svDecompositionSolver {
	ans := new(svDecompositionSolver)
	ans.svd = svd
	suT := svd.UT().Data()
	for i := 0; i < len(svd.singularValues); i++ {
		var a float64
		if svd.singularValues[i] > svd.tol {
			a = 1 / svd.singularValues[i]
		}
		suTi := suT[i]
		for j := 0; j < len(suTi); j++ {
			suTi[j] *= a
		}
	}

	mat, err := NewArray2DRowRealMatrixFromSlices(suT, true)
	if err != nil {
		panic(err)
	}

	ans.pseudoInverse = svd.V().Multiply(mat)
	return ans
}

func (svds *svDecompositionSolver) SolveVector(b RealVector) RealVector {
	return svds.pseudoInverse.OperateVector(b)
}

func (svds *svDecompositionSolver) SolveMatrix(b RealMatrix) RealMatrix {
	return svds.pseudoInverse.Multiply(b)
}

func (svds *svDecompositionSolver) IsNonSingular() bool {
	return svds.svd.Rank() == svds.svd.m
}

func (svds *svDecompositionSolver) Inverse() RealMatrix {
	return svds.pseudoInverse
}
