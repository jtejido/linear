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
	eps_ed    = 1e-12
	doubleeps = 1.0 / (1 << 53)
)

var (
	max_iter_ed = 30
)

/**
 * Calculates the eigen decomposition of a real matrix.
 *
 * The eigen decomposition of matrix A is a set of two matrices:
 * V and D such that A = V &times; D &times; V<sup>T</sup>.
 * A, V and D are all m &times; m matrices.
 *
 * This supports general real matrices (both symmetric and non-symmetric):
 *
 * If A is symmetric, then A = V*D*V' where the eigenvalue matrix D is diagonal
 * and the eigenvector matrix V is orthogonal, i.e.
 * A = V.multiply(D.multiply(V.transpose())) and V.multiply(V.transpose()) equals the identity matrix.
 *
 * If A is not symmetric, then the eigenvalue matrix D is block diagonal with the real
 * eigenvalues in 1-by-1 blocks and any complex eigenvalues, lambda + i*mu, in 2-by-2
 * blocks:
 *
 *    [lambda, mu    ]
 *    [   -mu, lambda]
 *
 * The columns of V represent the eigenvectors in the sense that {@code A*V = V*D},
 * i.e. A.multiply(V) equals V.multiply(D).
 * The matrix V may be badly conditioned, or even singular, so the validity of the
 * equation {@code A = V*D*inverse(V)} depends upon the condition of V.
 *
 * This implementation is based on the paper by A. Drubrulle, R.S. Martin and
 * J.H. Wilkinson "The Implicit QL Algorithm" in Wilksinson and Reinsch (1971)
 * Handbook for automatic computation, vol. 2, Linear algebra, Springer-Verlag,
 * New-York.
 */
type EigenDecomposition struct {
	main, secondary                  []float64
	transformer                      *TriDiagonalTransformer
	realEigenvalues, imagEigenvalues []float64
	eigenvectors                     []*ArrayRealVector
	cachedV, cachedD, cachedVt       RealMatrix
	isSymmetric                      bool
}

/**
 * Calculates the eigen decomposition of the given real matrix.
 *
 * Supports decomposition of a general matrix.
 */
func NewEigenDecomposition(matrix RealMatrix) (*EigenDecomposition, error) {
	ans := new(EigenDecomposition)
	symTol := 10. * float64(matrix.RowDimension()) * float64(matrix.ColumnDimension()) * doubleeps
	ans.isSymmetric = isSymmetric(matrix, symTol)
	if ans.isSymmetric {
		if err := ans.transformToTridiagonal(matrix); err != nil {
			return nil, err
		}
		if err := ans.findEigenVectors(ans.transformer.Q().Data()); err != nil {
			return nil, err
		}
	} else {
		if t, err := ans.transformToSchur(matrix); err != nil {
			return nil, err
		} else {
			if err := ans.findEigenVectorsFromSchur(t); err != nil {
				return nil, err
			}
		}
	}

	return ans, nil
}

/**
 * Calculates the eigen decomposition of the symmetric tridiagonal
 * matrix.  The Householder matrix is assumed to be the identity matrix.
 */
func NewEigenDecompositionFromTridiagonal(main, secondary []float64) (*EigenDecomposition, error) {
	ans := new(EigenDecomposition)
	ans.isSymmetric = true
	ans.main = append([]float64{}, main...)
	ans.secondary = append([]float64{}, secondary...)
	ans.transformer = nil
	size := len(main)
	z := make([][]float64, size)

	for i := 0; i < size; i++ {
		if z[i] == nil {
			z[i] = make([]float64, size)
		}
		z[i][i] = 1.0
	}

	if err := ans.findEigenVectors(z); err != nil {
		return nil, err
	}

	return ans, nil
}

/**
 * Gets the matrix V of the decomposition.
 * V is an orthogonal matrix, i.e. its transpose is also its inverse.
 * The columns of V are the eigenvectors of the original matrix.
 * No assumption is made about the orientation of the system axes formed
 * by the columns of V (e.g. in a 3-dimension space, V can form a left-
 * or right-handed system).
 */
func (ed *EigenDecomposition) V() RealMatrix {

	if ed.cachedV == nil {
		m := len(ed.eigenvectors)
		var err error
		ed.cachedV, err = NewRealMatrixWithDimension(m, m)
		if err != nil {
			panic(err)
		}

		for k := 0; k < m; k++ {
			ed.cachedV.SetColumnVector(k, ed.eigenvectors[k])
		}
	}

	return ed.cachedV
}

/**
 * Gets the block diagonal matrix D of the decomposition.
 * D is a block diagonal matrix.
 * Real eigenvalues are on the diagonal while complex values are on
 * 2x2 blocks { {real +imaginary}, {-imaginary, real} }.
 */
func (ed *EigenDecomposition) D() RealMatrix {

	if ed.cachedD == nil {
		// cache the matrix for subsequent calls
		var err error
		ed.cachedD, err = NewRealMatrixWithDiagonal(ed.realEigenvalues)
		if err != nil {
			panic(err)
		}

		for i := 0; i < len(ed.imagEigenvalues); i++ {
			if compareTo(ed.imagEigenvalues[i], 0.0, eps_ed) > 0 {
				ed.cachedD.SetEntry(i, i+1, ed.imagEigenvalues[i])
			} else if compareTo(ed.imagEigenvalues[i], 0.0, eps_ed) < 0 {
				ed.cachedD.SetEntry(i, i-1, ed.imagEigenvalues[i])
			}
		}
	}
	return ed.cachedD
}

/**
 * Gets the transpose of the matrix V of the decomposition.
 * V is an orthogonal matrix, i.e. its transpose is also its inverse.
 * The columns of V are the eigenvectors of the original matrix.
 * No assumption is made about the orientation of the system axes formed
 * by the columns of V (e.g. in a 3-dimension space, V can form a left-
 * or right-handed system).
 */
func (ed *EigenDecomposition) VT() RealMatrix {

	if ed.cachedVt == nil {
		m := len(ed.eigenvectors)
		var err error
		ed.cachedVt, err = NewRealMatrixWithDimension(m, m)
		if err != nil {
			panic(err)
		}
		for k := 0; k < m; k++ {
			ed.cachedVt.SetRowVector(k, ed.eigenvectors[k])
		}
	}

	// return the cached matrix
	return ed.cachedVt
}

/**
 * Returns whether the calculated eigen values are complex or real.
 * The method performs a zero check for each element of the
 * ImagEigenvalues() array and returns true if any
 * element is not equal to zero.
 */
func (ed *EigenDecomposition) HasComplexEigenvalues() bool {
	for i := 0; i < len(ed.imagEigenvalues); i++ {
		if !equalsWithError(ed.imagEigenvalues[i], 0.0, eps_ed) {
			return true
		}
	}
	return false
}

/**
 * Gets a copy of the real parts of the eigenvalues of the original matrix.
 */
func (ed *EigenDecomposition) RealEigenvalues() []float64 {
	return append([]float64{}, ed.realEigenvalues...)
}

/**
 * Returns the real part of the i<sup>th</sup> eigenvalue of the original
 * matrix.
 */
func (ed *EigenDecomposition) RealEigenvalueAt(i int) float64 {
	return ed.realEigenvalues[i]
}

/**
 * Gets a copy of the imaginary parts of the eigenvalues of the original
 * matrix.
 */
func (ed *EigenDecomposition) ImagEigenvalues() []float64 {
	return append([]float64{}, ed.imagEigenvalues...)
}

/**
 * Gets the imaginary part of the i<sup>th</sup> eigenvalue of the original
 * matrix.
 */
func (ed *EigenDecomposition) ImagEigenvalue(i int) float64 {
	return ed.imagEigenvalues[i]
}

/**
 * Gets a copy of the i<sup>th</sup> eigenvector of the original matrix.
 */
func (ed *EigenDecomposition) EigenvectorAt(i int) RealVector {
	return ed.eigenvectors[i].Copy()
}

/**
 * Computes the determinant of the matrix.
 */
func (ed *EigenDecomposition) Determinant() float64 {
	determinant := 1.
	for _, lambda := range ed.realEigenvalues {
		determinant *= lambda
	}
	return determinant
}

/**
 * Computes the square-root of the matrix.
 * This implementation assumes that the matrix is symmetric and positive
 * definite.
 */
func (ed *EigenDecomposition) SquareRoot() RealMatrix {
	if !ed.isSymmetric {
		panic(mathUnsupportedOperationErrorf())
	}

	sqrtEigenValues := make([]float64, len(ed.realEigenvalues))
	for i := 0; i < len(ed.realEigenvalues); i++ {
		eigen := ed.realEigenvalues[i]
		if eigen <= 0 {
			panic(mathUnsupportedOperationErrorf())
		}
		sqrtEigenValues[i] = math.Sqrt(eigen)
	}
	sqrtEigen, err := NewRealDiagonalMatrix(sqrtEigenValues)
	if err != nil {
		panic(err)
	}
	v := ed.V()
	vT := ed.VT()

	return v.Multiply(sqrtEigen).Multiply(vT)
}

/**
 * Gets a solver for finding the A &times; X = B solution in exact
 * linear sense.
 *
 * eigen decomposition of a general matrix is supported,
 * but the DecompositionSolver only supports real eigenvalues.
 */
func (ed *EigenDecomposition) Solver() DecompositionSolver {
	if ed.HasComplexEigenvalues() {
		panic(mathUnsupportedOperationErrorf())
	}
	return newEigenDecompositionSolver(ed)
}

/**
 * Transforms the matrix to tridiagonal form.
 */
func (ed *EigenDecomposition) transformToTridiagonal(matrix RealMatrix) error {
	// transform the matrix to tridiagonal
	var err error
	ed.transformer, err = NewTriDiagonalTransformer(matrix)
	if err != nil {
		return err
	}

	ed.main = ed.transformer.MainDiagonalRef()
	ed.secondary = ed.transformer.SecondaryDiagonalRef()
	return nil
}

/**
 * Find eigenvalues and eigenvectors (Dubrulle et al., 1971)
 */
func (ed *EigenDecomposition) findEigenVectors(householderMatrix [][]float64) error {
	z := make([][]float64, len(householderMatrix))
	for i := 0; i < len(householderMatrix); i++ {
		if z[i] == nil {
			z[i] = make([]float64, len(householderMatrix[i]))
		}
		z[i] = append([]float64{}, householderMatrix[i]...)
	}

	n := len(ed.main)
	ed.realEigenvalues = make([]float64, n)
	ed.imagEigenvalues = make([]float64, n)
	e := make([]float64, n)
	for i := 0; i < n-1; i++ {
		ed.realEigenvalues[i] = ed.main[i]
		e[i] = ed.secondary[i]
	}
	ed.realEigenvalues[n-1] = ed.main[n-1]
	e[n-1] = 0

	// Determine the largest main and secondary value in absolute term.
	var maxAbsoluteValue float64
	for i := 0; i < n; i++ {
		if math.Abs(ed.realEigenvalues[i]) > maxAbsoluteValue {
			maxAbsoluteValue = math.Abs(ed.realEigenvalues[i])
		}
		if math.Abs(e[i]) > maxAbsoluteValue {
			maxAbsoluteValue = math.Abs(e[i])
		}
	}
	// Make null any main and secondary value too small to be significant
	if maxAbsoluteValue != 0 {
		for i := 0; i < n; i++ {
			if math.Abs(ed.realEigenvalues[i]) <= doubleeps*maxAbsoluteValue {
				ed.realEigenvalues[i] = 0
			}
			if math.Abs(e[i]) <= doubleeps*maxAbsoluteValue {
				e[i] = 0
			}
		}
	}

	for j := 0; j < n; j++ {
		its := 0
		var m int
		for ok := true; ok; ok = m != j {
			for m = j; m < n-1; m++ {
				delta := math.Abs(ed.realEigenvalues[m]) +
					math.Abs(ed.realEigenvalues[m+1])
				if math.Abs(e[m])+delta == delta {
					break
				}
			}
			if m != j {
				if its == max_iter_ed {
					return maxIterationErrorf(convergence_failed, max_iter_ed)
				}
				its++
				q := (ed.realEigenvalues[j+1] - ed.realEigenvalues[j]) / (2 * e[j])
				t := math.Sqrt(1 + q*q)
				if q < 0.0 {
					q = ed.realEigenvalues[m] - ed.realEigenvalues[j] + e[j]/(q-t)
				} else {
					q = ed.realEigenvalues[m] - ed.realEigenvalues[j] + e[j]/(q+t)
				}
				u := 0.0
				s := 1.0
				c := 1.0
				var i int
				for i = m - 1; i >= j; i-- {
					p := s * e[i]
					h := c * e[i]
					if math.Abs(p) >= math.Abs(q) {
						c = q / p
						t = math.Sqrt(c*c + 1.0)
						e[i+1] = p * t
						s = 1.0 / t
						c *= s
					} else {
						s = p / q
						t = math.Sqrt(s*s + 1.0)
						e[i+1] = q * t
						c = 1.0 / t
						s *= c
					}
					if e[i+1] == 0.0 {
						ed.realEigenvalues[i+1] -= u
						e[m] = 0.0
						break
					}
					q = ed.realEigenvalues[i+1] - u
					t = (ed.realEigenvalues[i]-q)*s + 2.0*c*h
					u = s * t
					ed.realEigenvalues[i+1] = q + u
					q = c*t - h
					for ia := 0; ia < n; ia++ {
						p = z[ia][i+1]
						z[ia][i+1] = s*z[ia][i] + c*p
						z[ia][i] = c*z[ia][i] - s*p
					}
				}
				if t == 0.0 && i >= j {
					continue
				}
				ed.realEigenvalues[j] -= u
				e[j] = q
				e[m] = 0.0
			}

		}
	}

	//Sort the eigen values (and vectors) in increase order
	for i := 0; i < n; i++ {
		k := i
		p := ed.realEigenvalues[i]
		for j := i + 1; j < n; j++ {
			if ed.realEigenvalues[j] > p {
				k = j
				p = ed.realEigenvalues[j]
			}
		}
		if k != i {
			ed.realEigenvalues[k] = ed.realEigenvalues[i]
			ed.realEigenvalues[i] = p
			for j := 0; j < n; j++ {
				p = z[j][i]
				z[j][i] = z[j][k]
				z[j][k] = p
			}
		}
	}

	// Determine the largest eigen value in absolute term.
	maxAbsoluteValue = 0
	for i := 0; i < n; i++ {
		if math.Abs(ed.realEigenvalues[i]) > maxAbsoluteValue {
			maxAbsoluteValue = math.Abs(ed.realEigenvalues[i])
		}
	}
	// Make null any eigen value too small to be significant
	if maxAbsoluteValue != 0.0 {
		for i := 0; i < n; i++ {
			if math.Abs(ed.realEigenvalues[i]) < doubleeps*maxAbsoluteValue {
				ed.realEigenvalues[i] = 0
			}
		}
	}
	ed.eigenvectors = make([]*ArrayRealVector, n)
	tmp := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			tmp[j] = z[j][i]
		}
		ed.eigenvectors[i] = new(ArrayRealVector)
		ed.eigenvectors[i].data = append([]float64{}, tmp...)
	}

	return nil
}

/**
 * Transforms the matrix to Schur form and calculates the eigenvalues.
 */
func (ed *EigenDecomposition) transformToSchur(matrix RealMatrix) (*SchurTransformer, error) {
	schurTransform, err := NewSchurTransformer(matrix)
	if err != nil {
		return nil, err
	}

	matT := schurTransform.T().Data()

	ed.realEigenvalues = make([]float64, len(matT))
	ed.imagEigenvalues = make([]float64, len(matT))

	for i := 0; i < len(ed.realEigenvalues); i++ {
		if i == (len(ed.realEigenvalues)-1) || equalsWithError(matT[i+1][i], 0.0, eps_ed) {
			ed.realEigenvalues[i] = matT[i][i]
		} else {
			x := matT[i+1][i+1]
			p := 0.5 * (matT[i][i] - x)
			z := math.Sqrt(math.Abs(p*p + matT[i+1][i]*matT[i][i+1]))
			ed.realEigenvalues[i] = x + p
			ed.imagEigenvalues[i] = z
			ed.realEigenvalues[i+1] = x + p
			ed.imagEigenvalues[i+1] = -z
			i++
		}
	}
	return schurTransform, nil
}

/**
 * Find eigenvectors from a matrix transformed to Schur form.
 */
func (ed *EigenDecomposition) findEigenVectorsFromSchur(schur *SchurTransformer) error {
	matrixT := schur.T().Data()
	matrixP := schur.P().Data()

	n := len(matrixT)

	// compute matrix norm
	var norm float64
	for i := 0; i < n; i++ {
		for j := int(math.Max(float64(i-1), 0.)); j < n; j++ {
			norm += math.Abs(matrixT[i][j])
		}
	}

	// we can not handle a matrix with zero norm
	if equalsWithError(norm, 0.0, eps_ed) {
		return mathArithmeticErrorf(zero_norm)
	}

	// Backsubstitute to find vectors of upper triangular form

	var r, s, z float64

	for idx := n - 1; idx >= 0; idx-- {
		p := ed.realEigenvalues[idx]
		q := ed.imagEigenvalues[idx]

		if equals(q, 0.0) {
			// Real vector
			l := idx
			matrixT[idx][idx] = 1.0
			for i := idx - 1; i >= 0; i-- {
				w := matrixT[i][i] - p
				r = 0.0
				for j := l; j <= idx; j++ {
					r += matrixT[i][j] * matrixT[j][idx]
				}
				if compareTo(ed.imagEigenvalues[i], 0.0, eps_ed) < 0 {
					z = w
					s = r
				} else {
					l = i
					if equals(ed.imagEigenvalues[i], 0.0) {
						if w != 0.0 {
							matrixT[i][idx] = -r / w
						} else {
							matrixT[i][idx] = -r / (doubleeps * norm)
						}
					} else {
						// Solve real equations
						x := matrixT[i][i+1]
						y := matrixT[i+1][i]
						q = (ed.realEigenvalues[i]-p)*(ed.realEigenvalues[i]-p) + ed.imagEigenvalues[i]*ed.imagEigenvalues[i]
						t := (x*s - z*r) / q
						matrixT[i][idx] = t
						if math.Abs(x) > math.Abs(z) {
							matrixT[i+1][idx] = (-r - w*t) / x
						} else {
							matrixT[i+1][idx] = (-s - y*t) / z
						}
					}

					// Overflow control
					t := math.Abs(matrixT[i][idx])
					if (doubleeps*t)*t > 1 {
						for j := i; j <= idx; j++ {
							matrixT[j][idx] /= t
						}
					}
				}
			}
		} else if q < 0.0 {
			// Complex vector
			l := idx - 1

			// Last vector component imaginary so matrix is triangular
			if math.Abs(matrixT[idx][idx-1]) > math.Abs(matrixT[idx-1][idx]) {
				matrixT[idx-1][idx-1] = q / matrixT[idx][idx-1]
				matrixT[idx-1][idx] = -(matrixT[idx][idx] - p) / matrixT[idx][idx-1]
			} else {
				result := complex(0.0, -matrixT[idx-1][idx]) / complex(matrixT[idx-1][idx-1]-p, q)

				matrixT[idx-1][idx-1] = real(result)
				matrixT[idx-1][idx] = imag(result)
			}

			matrixT[idx][idx-1] = 0.0
			matrixT[idx][idx] = 1.0

			for i := idx - 2; i >= 0; i-- {
				var ra, sa float64
				for j := l; j <= idx; j++ {
					ra += matrixT[i][j] * matrixT[j][idx-1]
					sa += matrixT[i][j] * matrixT[j][idx]
				}
				w := matrixT[i][i] - p

				if compareTo(ed.imagEigenvalues[i], 0.0, eps_ed) < 0 {
					z = w
					r = ra
					s = sa
				} else {
					l = i
					if equals(ed.imagEigenvalues[i], 0.0) {
						c := complex(-ra, -sa) / complex(w, q)
						matrixT[i][idx-1] = real(c)
						matrixT[i][idx] = imag(c)
					} else {
						// Solve complex equations
						x := matrixT[i][i+1]
						y := matrixT[i+1][i]
						vr := (ed.realEigenvalues[i]-p)*(ed.realEigenvalues[i]-p) + ed.imagEigenvalues[i]*ed.imagEigenvalues[i] - q*q
						vi := (ed.realEigenvalues[i] - p) * 2.0 * q
						if equals(vr, 0.0) && equals(vi, 0.0) {
							vr = doubleeps * norm * (math.Abs(w) + math.Abs(q) + math.Abs(x) + math.Abs(y) + math.Abs(z))
						}
						c := complex(x*r-z*ra+q*sa, x*s-z*sa-q*ra) / complex(vr, vi)
						matrixT[i][idx-1] = real(c)
						matrixT[i][idx] = imag(c)

						if math.Abs(x) > (math.Abs(z) + math.Abs(q)) {
							matrixT[i+1][idx-1] = (-ra - w*matrixT[i][idx-1] +
								q*matrixT[i][idx]) / x
							matrixT[i+1][idx] = (-sa - w*matrixT[i][idx] -
								q*matrixT[i][idx-1]) / x
						} else {
							c2 := complex(-r-y*matrixT[i][idx-1], -s-y*matrixT[i][idx]) / complex(z, q)
							matrixT[i+1][idx-1] = real(c2)
							matrixT[i+1][idx] = imag(c2)
						}
					}

					// Overflow control
					t := math.Max(math.Abs(matrixT[i][idx-1]),
						math.Abs(matrixT[i][idx]))
					if (doubleeps*t)*t > 1 {
						for j := i; j <= idx; j++ {
							matrixT[j][idx-1] /= t
							matrixT[j][idx] /= t
						}
					}
				}
			}
		}
	}

	// Back transformation to get eigenvectors of original matrix
	for j := n - 1; j >= 0; j-- {
		for i := 0; i <= n-1; i++ {
			z = 0.0
			for k := 0; k <= int(math.Min(float64(j), float64(n-1))); k++ {
				z += matrixP[i][k] * matrixT[k][j]
			}
			matrixP[i][j] = z
		}
	}

	ed.eigenvectors = make([]*ArrayRealVector, n)
	tmp := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			tmp[j] = matrixP[j][i]
		}
		ed.eigenvectors[i] = new(ArrayRealVector)
		ed.eigenvectors[i].data = append([]float64{}, tmp...)
	}

	return nil
}

type eigenDecompositionSolver struct {
	ed *EigenDecomposition
}

/**
 * Gets a solver for finding the A &times; X = B solution in exact
 * linear sense.
 *
 * eigen decomposition of a general matrix is supported,
 * but the DecompositionSolver only supports real eigenvalues.
 */
func newEigenDecompositionSolver(ed *EigenDecomposition) *eigenDecompositionSolver {
	return &eigenDecompositionSolver{ed: ed}
}

func (s *eigenDecompositionSolver) IsNonSingular() bool {
	largestEigenvalueNorm := 0.0
	// Looping over all values (in case they are not sorted in decreasing
	// order of their norm).
	for i := 0; i < len(s.ed.realEigenvalues); i++ {
		largestEigenvalueNorm = math.Max(largestEigenvalueNorm, s.eigenvalueNormAt(i))
	}
	// Corner case: zero matrix, all exactly 0 eigenvalues
	if largestEigenvalueNorm == 0.0 {
		return false
	}
	for i := 0; i < len(s.ed.realEigenvalues); i++ {
		// Looking for eigenvalues that are 0, where we consider anything much much smaller
		// than the largest eigenvalue to be effectively 0.
		if equalsWithError(s.eigenvalueNormAt(i)/largestEigenvalueNorm, 0, eps_ed) {
			return false
		}
	}

	return true
}

/**
 * Solves the linear equation A &times; X = B for symmetric matrices A.
 *
 * This method only finds exact linear solutions, i.e. solutions for
 * which ||A &times; X - B|| is exactly 0.
 */
func (s *eigenDecompositionSolver) SolveVector(b RealVector) RealVector {
	m := len(s.ed.realEigenvalues)
	if b.Dimension() != m {
		panic(dimensionsMismatchSimpleErrorf(b.Dimension(), m))
	}
	if !s.IsNonSingular() {
		panic(singularMatrixSimpleErrorf())
	}

	bp := make([]float64, m)
	for i := 0; i < m; i++ {
		v := s.ed.eigenvectors[i]
		vData := v.DataRef()
		s := VecDotProduct(v, b) / s.ed.realEigenvalues[i]
		for j := 0; j < m; j++ {
			bp[j] += s * vData[j]
		}
	}

	mat := new(ArrayRealVector)
	mat.data = bp
	return mat
}

func (s *eigenDecompositionSolver) SolveMatrix(b RealMatrix) RealMatrix {
	m := len(s.ed.realEigenvalues)
	if b.RowDimension() != m {
		panic(dimensionsMismatchSimpleErrorf(b.RowDimension(), m))
	}
	if !s.IsNonSingular() {
		panic(singularMatrixSimpleErrorf())
	}

	nColB := b.ColumnDimension()
	bp := make([][]float64, m)
	tmpCol := make([]float64, m)
	for k := 0; k < nColB; k++ {
		for i := 0; i < m; i++ {
			tmpCol[i] = b.At(i, k)
			if bp[i] == nil {
				bp[i] = make([]float64, nColB)
			}
			bp[i][k] = 0
		}
		for i := 0; i < m; i++ {
			v := s.ed.eigenvectors[i]
			vData := v.DataRef()
			var ss float64
			for j := 0; j < m; j++ {
				ss += v.At(j) * tmpCol[j]
			}
			ss /= s.ed.realEigenvalues[i]
			for j := 0; j < m; j++ {
				bp[j][k] += ss * vData[j]
			}
		}
	}

	mat := new(Array2DRowRealMatrix)
	mat.copyIn(bp)
	return mat
}

func (s *eigenDecompositionSolver) Inverse() RealMatrix {
	if !s.IsNonSingular() {
		panic(singularMatrixSimpleErrorf())
	}

	m := len(s.ed.realEigenvalues)
	invData := make([][]float64, m)

	for i := 0; i < m; i++ {
		if invData[i] == nil {
			invData[i] = make([]float64, m)
		}
		invI := invData[i]
		for j := 0; j < m; j++ {
			invIJ := 0.
			for k := 0; k < m; k++ {
				vK := s.ed.eigenvectors[k].DataRef()
				invIJ += vK[i] * vK[j] / s.ed.realEigenvalues[k]
			}
			invI[j] = invIJ
		}
	}

	mat, err := NewRealMatrixFromSlices(invData)
	if err != nil {
		panic(err)
	}

	return mat
}

func (s *eigenDecompositionSolver) eigenvalueNormAt(i int) float64 {
	re := s.ed.realEigenvalues[i]
	im := s.ed.imagEigenvalues[i]
	return math.Sqrt(re*re + im*im)
}
