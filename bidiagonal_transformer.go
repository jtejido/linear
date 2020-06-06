package linear

import (
	"math"
)

type BiDiagonalTransformer struct {
	householderVectors        [][]float64
	main, secondary           []float64
	cachedU, cachedB, cachedV RealMatrix
}

func NewBiDiagonalTransformer(matrix RealMatrix) (*BiDiagonalTransformer, error) {
	ans := new(BiDiagonalTransformer)
	m := matrix.RowDimension()
	n := matrix.ColumnDimension()
	p := int(math.Min(float64(m), float64(n)))
	ans.householderVectors = matrix.Data()
	ans.main = make([]float64, p)
	ans.secondary = make([]float64, p-1)
	// transform matrix
	if m >= n {
		ans.transformToUpperBiDiagonal()
	} else {
		ans.transformToLowerBiDiagonal()
	}

	return ans, nil
}

func (bdt *BiDiagonalTransformer) U() RealMatrix {
	if bdt.cachedU == nil {
		m := len(bdt.householderVectors)
		n := len(bdt.householderVectors[0])
		p := len(bdt.main)

		diagonal := bdt.secondary
		diagOffset := 1
		if m >= n {
			diagOffset = 0
			diagonal = bdt.main
		}

		ua := make([][]float64, m)
		for i := 0; i < m; i++ {
			ua[i] = make([]float64, m)
		}
		// fill up the part of the matrix not affected by Householder transforms
		for k := m - 1; k >= p; k-- {
			ua[k][k] = 1
		}

		// build up first part of the matrix by applying Householder transforms
		for k := p - 1; k >= diagOffset; k-- {
			hK := bdt.householderVectors[k]
			ua[k][k] = 1
			if hK[k-diagOffset] != 0.0 {
				for j := k; j < m; j++ {
					var alpha float64
					for i := k; i < m; i++ {
						alpha -= ua[i][j] * bdt.householderVectors[i][k-diagOffset]
					}
					alpha /= diagonal[k-diagOffset] * hK[k-diagOffset]

					for i := k; i < m; i++ {
						ua[i][j] += -alpha * bdt.householderVectors[i][k-diagOffset]
					}
				}
			}
		}
		if diagOffset > 0 {
			ua[0][0] = 1
		}
		var err error
		bdt.cachedU, err = NewRealMatrixFromSlices(ua)
		if err != nil {
			panic(err)
		}
	}

	// return the cached matrix
	return bdt.cachedU

}

func (bdt *BiDiagonalTransformer) B() RealMatrix {
	if bdt.cachedB == nil {
		m := len(bdt.householderVectors)
		n := len(bdt.householderVectors[0])
		ba := make([][]float64, m)
		for i := 0; i < m; i++ {
			ba[i] = make([]float64, n)
		}
		for i := 0; i < len(bdt.main); i++ {
			ba[i][i] = bdt.main[i]
			if m < n {
				if i > 0 {
					ba[i][i-1] = bdt.secondary[i-1]
				}
			} else {

				if i < len(bdt.main)-1 {
					ba[i][i+1] = bdt.secondary[i]
				}
			}
		}
		var err error
		bdt.cachedB, err = NewRealMatrixFromSlices(ba)
		if err != nil {
			panic(err)
		}
	}

	// return the cached matrix
	return bdt.cachedB

}

func (bdt *BiDiagonalTransformer) V() RealMatrix {
	if bdt.cachedV == nil {
		m := len(bdt.householderVectors)
		n := len(bdt.householderVectors[0])
		p := len(bdt.main)
		var diagOffset int
		diagonal := bdt.main

		if m >= n {
			diagOffset = 1
			diagonal = bdt.secondary
		}
		va := make([][]float64, n)
		for i := 0; i < n; i++ {
			va[i] = make([]float64, n)
		}

		// fill up the part of the matrix not affected by Householder transforms
		for k := n - 1; k >= p; k-- {
			va[k][k] = 1
		}

		// build up first part of the matrix by applying Householder transforms
		for k := p - 1; k >= diagOffset; k-- {
			hK := bdt.householderVectors[k-diagOffset]
			va[k][k] = 1
			if hK[k] != 0.0 {
				for j := k; j < n; j++ {
					var beta float64
					for i := k; i < n; i++ {
						beta -= va[i][j] * hK[i]
					}
					beta /= diagonal[k-diagOffset] * hK[k]

					for i := k; i < n; i++ {
						va[i][j] += -beta * hK[i]
					}
				}
			}
		}
		if diagOffset > 0 {
			va[0][0] = 1
		}
		var err error
		bdt.cachedV, err = NewRealMatrixFromSlices(va)
		if err != nil {
			panic(err)
		}
	}

	// return the cached matrix
	return bdt.cachedV
}

func (bdt *BiDiagonalTransformer) HouseholderVectorsRef() [][]float64 {
	return bdt.householderVectors
}

func (bdt *BiDiagonalTransformer) MainDiagonalRef() []float64 {
	return bdt.main
}

func (bdt *BiDiagonalTransformer) SecondaryDiagonalRef() []float64 {
	return bdt.secondary
}

func (bdt *BiDiagonalTransformer) IsUpperBiDiagonal() bool {
	return len(bdt.householderVectors) >= len(bdt.householderVectors[0])
}

func (bdt *BiDiagonalTransformer) transformToUpperBiDiagonal() {

	m := len(bdt.householderVectors)
	n := len(bdt.householderVectors[0])
	for k := 0; k < n; k++ {
		//zero-out a column
		var xNormSqr float64
		for i := k; i < m; i++ {
			c := bdt.householderVectors[i][k]
			xNormSqr += c * c
		}
		hK := bdt.householderVectors[k]
		var a float64
		if hK[k] > 0 {
			a = -math.Sqrt(xNormSqr)
		} else {
			a = math.Sqrt(xNormSqr)
		}

		bdt.main[k] = a
		if a != 0.0 {
			hK[k] -= a
			for j := k + 1; j < n; j++ {
				var alpha float64
				for i := k; i < m; i++ {
					hI := bdt.householderVectors[i]
					alpha -= hI[j] * hI[k]
				}
				alpha /= a * bdt.householderVectors[k][k]
				for i := k; i < m; i++ {
					hI := bdt.householderVectors[i]
					hI[j] -= alpha * hI[k]
				}
			}
		}

		if k < n-1 {
			//zero-out a row
			xNormSqr = 0
			for j := k + 1; j < n; j++ {
				c := hK[j]
				xNormSqr += c * c
			}
			var b float64
			if hK[k+1] > 0 {
				b = -math.Sqrt(xNormSqr)
			} else {
				b = math.Sqrt(xNormSqr)
			}
			bdt.secondary[k] = b
			if b != 0.0 {
				hK[k+1] -= b
				for i := k + 1; i < m; i++ {
					hI := bdt.householderVectors[i]
					var beta float64
					for j := k + 1; j < n; j++ {
						beta -= hI[j] * hK[j]
					}
					beta /= b * hK[k+1]
					for j := k + 1; j < n; j++ {
						hI[j] -= beta * hK[j]
					}
				}
			}
		}
	}
}

func (bdt *BiDiagonalTransformer) transformToLowerBiDiagonal() {
	m := len(bdt.householderVectors)
	n := len(bdt.householderVectors[0])
	for k := 0; k < m; k++ {
		//zero-out a row
		hK := bdt.householderVectors[k]
		var xNormSqr float64
		for j := k; j < n; j++ {
			c := hK[j]
			xNormSqr += c * c
		}
		var a float64
		if hK[k] > 0 {
			a = -math.Sqrt(xNormSqr)
		} else {
			a = math.Sqrt(xNormSqr)
		}
		bdt.main[k] = a
		if a != 0.0 {
			hK[k] -= a
			for i := k + 1; i < m; i++ {
				hI := bdt.householderVectors[i]
				var alpha float64
				for j := k; j < n; j++ {
					alpha -= hI[j] * hK[j]
				}
				alpha /= a * bdt.householderVectors[k][k]
				for j := k; j < n; j++ {
					hI[j] -= alpha * hK[j]
				}
			}
		}

		if k < m-1 {
			//zero-out a column
			hKp1 := bdt.householderVectors[k+1]
			xNormSqr = 0
			for i := k + 1; i < m; i++ {
				c := bdt.householderVectors[i][k]
				xNormSqr += c * c
			}
			var b float64
			if hKp1[k] > 0 {
				b = -math.Sqrt(xNormSqr)
			} else {
				b = math.Sqrt(xNormSqr)
			}
			bdt.secondary[k] = b
			if b != 0.0 {
				hKp1[k] -= b
				for j := k + 1; j < n; j++ {
					var beta float64
					for i := k + 1; i < m; i++ {
						hI := bdt.householderVectors[i]
						beta -= hI[j] * hI[k]
					}
					beta /= b * hKp1[k]
					for i := k + 1; i < m; i++ {
						hI := bdt.householderVectors[i]
						hI[j] -= beta * hI[k]
					}
				}
			}
		}

	}
}
