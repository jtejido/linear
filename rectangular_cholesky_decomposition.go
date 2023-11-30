package linear

import (
	"math"
)

/**
 * Calculates the rectangular Cholesky decomposition of a matrix.
 * <p>The rectangular Cholesky decomposition of a real symmetric positive
 * semidefinite matrix A consists of a rectangular matrix B with the same
 * number of rows such that: A is almost equal to BB<sup>T</sup>, depending
 * on a user-defined tolerance. In a sense, this is the square root of A.</p>
 * <p>The difference with respect to the regular {@link CholeskyDecomposition}
 * is that rows/columns may be permuted (hence the rectangular shape instead
 * of the traditional triangular shape) and there is a threshold to ignore
 * small diagonal elements. This is used for example to generate {@link
 * org.apache.commons.math4.random.CorrelatedRandomVectorGenerator correlated
 * random n-dimensions vectors} in a p-dimension subspace (p &lt; n).
 * In other words, it allows generating random vectors from a covariance
 * matrix that is only positive semidefinite, and not positive definite.</p>
 * <p>Rectangular Cholesky decomposition is <em>not</em> suited for solving
 * linear systems, so it does not provide any {@link DecompositionSolver
 * decomposition solver}.</p>
 */
type RectangularCholeskyDecomposition struct {
	root RealMatrix
	rank int
}

func NewRectangularCholeskyDecomposition(matrix RealMatrix) (*RectangularCholeskyDecomposition, error) {
	return NewRectangularCholeskyDecompositionWithThreshold(matrix, 0)
}

func NewRectangularCholeskyDecompositionWithThreshold(matrix RealMatrix, small float64) (*RectangularCholeskyDecomposition, error) {
	ans := new(RectangularCholeskyDecomposition)
	order := matrix.RowDimension()
	c := matrix.Data()
	b := make([][]float64, order)

	index := make([]int, order)
	for i := 0; i < order; i++ {
		b[i] = make([]float64, order)
		index[i] = i
	}

	var r int
	for {
		// find maximal diagonal element
		swapR := r
		for i := r + 1; i < order; i++ {
			ii := index[i]
			isr := index[swapR]
			if c[ii][ii] > c[isr][isr] {
				swapR = i
			}
		}

		// swap elements
		if swapR != r {
			tmpIndex := index[r]
			index[r] = index[swapR]
			index[swapR] = tmpIndex
			tmpRow := b[r]
			b[r] = b[swapR]
			b[swapR] = tmpRow
		}

		// check diagonal element
		ir := index[r]
		if c[ir][ir] <= small {
			if r == 0 {
				return nil, nonPositiveDefiniteMatrixErrorf(c[ir][ir], ir, small)
			}

			// check remaining diagonal elements
			for i := r; i < order; i++ {
				if c[index[i]][index[i]] < -small {
					// there is at least one sufficiently negative diagonal element,
					// the symmetric positive semidefinite matrix is wrong
					return nil, nonPositiveDefiniteMatrixErrorf(c[index[i]][index[i]], i, small)
				}
			}

			// all remaining diagonal elements are close to zero, we consider we have
			// found the rank of the symmetric positive semidefinite matrix
			break

		} else {
			// transform the matrix
			sqrt := math.Sqrt(c[ir][ir])
			b[r][r] = sqrt
			inverse := 1 / sqrt
			inverse2 := 1 / c[ir][ir]
			for i := r + 1; i < order; i++ {
				ii := index[i]
				e := inverse * c[ii][ir]
				b[i][r] = e
				c[ii][ii] -= c[ii][ir] * c[ii][ir] * inverse2
				for j := r + 1; j < i; j++ {
					ij := index[j]
					f := c[ii][ij] - e*b[j][r]
					c[ii][ij] = f
					c[ij][ii] = f
				}
			}

			// prepare next iteration
			r++
			if r >= order {
				break
			}
		}
	}

	// build the root matrix
	ans.rank = r
	var err error
	ans.root, err = NewRealMatrixWithDimension(order, r)
	if err != nil {
		return nil, err
	}
	for i := 0; i < order; i++ {
		for j := 0; j < r; j++ {
			ans.root.SetEntry(index[i], j, b[i][j])
		}
	}

	return ans, nil
}

func (rcd *RectangularCholeskyDecomposition) RootMatrix() RealMatrix {
	return rcd.root
}

func (rcd *RectangularCholeskyDecomposition) Rank() int {
	return rcd.rank
}
