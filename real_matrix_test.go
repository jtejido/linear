package linear

import (
	"math"
	"math/rand"
	"testing"
)

func createArrayRealVectorFromSlice(t *testing.T, d []float64) *ArrayRealVector {
	v, err := NewArrayRealVectorFromSlice(d)

	if err != nil {
		t.Errorf("Error while creating Vector %s", err)
	}

	return v
}

func createSizedArrayRealVector(t *testing.T, size int) *ArrayRealVector {
	v, err := NewSizedArrayRealVector(size)

	if err != nil {
		t.Errorf("Error while creating Vector %s", err)
	}

	return v
}

func createSizedArrayRealVectorWithPreset(t *testing.T, size int, preset float64) *ArrayRealVector {
	v, err := NewSizedArrayRealVectorWithPreset(size, preset)

	if err != nil {
		t.Errorf("Error while creating Vector %s", err)
	}

	return v
}

func createRealMatrixWithDimension(t *testing.T, rows, columns int) RealMatrix {
	m, err := NewRealMatrixWithDimension(rows, columns)

	if err != nil {
		t.Errorf("Error while creating Matrix %s", err)
	}

	return m
}

func createRealMatrixFromSlices(t *testing.T, data [][]float64) RealMatrix {
	m, err := NewRealMatrixFromSlices(data)

	if err != nil {
		t.Errorf("Error while creating Matrix %s", err)
	}

	return m
}

func createRealIdentityMatrix(t *testing.T, dim int) RealMatrix {
	m, err := NewRealIdentityMatrix(dim)

	if err != nil {
		t.Errorf("Error while creating Matrix %s", err)
	}

	return m
}

func createRealMatrixWithDiagonal(t *testing.T, diagonal []float64) RealMatrix {
	m, err := NewRealMatrixWithDiagonal(diagonal)

	if err != nil {
		t.Errorf("Error while creating Matrix %s", err)
	}

	return m
}

func createRealDiagonalMatrix(t *testing.T, diagonal []float64) RealMatrix {
	m, err := NewRealDiagonalMatrix(diagonal)

	if err != nil {
		t.Errorf("Error while creating Matrix %s", err)
	}

	return m
}

func createRealVector(t *testing.T, data []float64) RealVector {
	v, err := NewRealVector(data)

	if err != nil {
		t.Errorf("Error while creating Vector %s", err)
	}

	return v
}

type SetVisitor struct{}

func (s *SetVisitor) Start(rows, columns, startRow, endRow, startColumn, endColumn int) {}
func (s *SetVisitor) Visit(row, column int, value float64) float64 {
	return float64(row) + float64(column)/1024.0
}
func (s *SetVisitor) End() float64 { return 0 }

type GetVisitor struct {
	count int
	t     *testing.T
}

func (g *GetVisitor) Start(rows, columns, startRow, endRow, startColumn, endColumn int) {}
func (g *GetVisitor) Visit(row, column int, value float64) {
	g.count++
	if (float64(row) + float64(column)/1024.0) != value {
		g.t.Errorf("Mismatch. Value doesn't match.")
	}
}
func (g *GetVisitor) End() float64 { return 0 }

func columnToVector(column [][]float64) RealVector {
	data := make([]float64, len(column))
	for i := 0; i < len(data); i++ {
		data[i] = column[i][0]
	}

	v, _ := NewArrayRealVectorFromSlice(data)

	return v
}

func checkSubMatrix(t *testing.T, m RealMatrix, reference [][]float64, startRow, endRow, startColumn, endColumn int) {

	if reference != nil {
		sub := m.SubMatrix(startRow, endRow, startColumn, endColumn)

		if !createBlockRealMatrixFromSlices(t, reference).Equals(sub) {
			t.Errorf("Mismatch. submatrix not equal reference mat. want: true, got: false")
		}
	} else {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("panic expected.")
			}
		}()
		m.SubMatrix(startRow, endRow, startColumn, endColumn)
	}

}

func checkSubMatrixFromIndices(t *testing.T, m RealMatrix, reference [][]float64, selectedRows, selectedColumns []int) {
	if reference != nil {
		sub := m.SubMatrixFromIndices(selectedRows, selectedColumns)

		if !createBlockRealMatrixFromSlices(t, reference).Equals(sub) {
			t.Errorf("Mismatch. submatrix not equal reference mat. want: true, got: false")
		}
	} else {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("panic expected.")
			}
		}()
		m.SubMatrixFromIndices(selectedRows, selectedColumns)
	}

}

func createRandomMatrix(r *rand.Rand, rows, columns int) *BlockRealMatrix {
	m, _ := NewBlockRealMatrix(rows, columns)
	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			m.SetEntry(i, j, 200*r.Float64()-100)
		}
	}
	return m
}

func columnToArray(column [][]float64) []float64 {
	data := make([]float64, len(column))
	for i := 0; i < len(data); i++ {
		data[i] = column[i][0]
	}
	return data
}

func checkArrays(t *testing.T, expected, actual []float64) {

	if len(expected) != len(actual) {
		t.Errorf("Mismatch. expected and actual slice length doesn't match.")
	}
	for i := 0; i < len(expected); i++ {
		if expected[i] != actual[i] {
			t.Errorf("Mismatch. expected and actual value doesn't match.")
		}

	}
}

func assertClose(t *testing.T, m, n []float64, tolerance float64) {
	if len(m) != len(n) {
		t.Errorf("vectors not same length")
	}
	for i := 0; i < len(m); i++ {
		if math.Abs(m[i]-n[i]) > tolerance {
			t.Errorf("Mismatch. identity trace. want: %v, got: %v", n[i], m[i])
		}
	}
}

func assertCloseMatrix(t *testing.T, m, n RealMatrix, tolerance float64) {
	if MatLInfNorm(m.Subtract(n)) >= tolerance {
		t.Errorf("Mismatch. values not within tolerance level.")
	}
}

func checkOrthogonal(t *testing.T, m RealMatrix) {
	mTm := m.Transpose().Multiply(m)
	id := createRealIdentityMatrix(t, mTm.RowDimension())
	if math.Abs(0-MatLInfNorm(mTm.Subtract(id))) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, MatLInfNorm(mTm.Subtract(id)))
	}

}

func checkAEqualQTQt(t *testing.T, m RealMatrix) {
	transformer := createTriDiagonalTransformer(t, m)
	q := transformer.Q()
	qT := transformer.QT()
	tt := transformer.T()
	norm := MatLInfNorm(q.Multiply(tt).Multiply(qT).Subtract(m))

	if math.Abs(0-norm) > 4.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, norm)
	}
}

func checkNoAccessBelowDiagonal(t *testing.T, data [][]float64) {
	modifiedData := make([][]float64, len(data))
	for i := 0; i < len(data); i++ {
		modifiedData[i] = append([]float64{}, data[i]...)
		for j := 0; j < i; j++ {
			modifiedData[i][j] = math.NaN()
		}
	}
	m := createRealMatrixFromSlices(t, modifiedData)
	transformer := createTriDiagonalTransformer(t, m)
	q := transformer.Q()
	qT := transformer.QT()
	tt := transformer.T()
	norm := MatLInfNorm(q.Multiply(tt).Multiply(qT).Subtract(createRealMatrixFromSlices(t, data)))
	if math.Abs(0-norm) > 4.0e-15 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, norm)
	}
}

func checkTriDiagonalMatricesValues(t *testing.T, m, qRef [][]float64, mainDiagnonal, secondaryDiagonal []float64) {
	transformer := createTriDiagonalTransformer(t, createRealMatrixFromSlices(t, m))

	// check values against known references
	q := transformer.Q()
	norm := MatLInfNorm(q.Subtract(createRealMatrixFromSlices(t, qRef)))
	if math.Abs(0-norm) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, norm)
	}

	tt := transformer.T()
	tData := make([][]float64, len(mainDiagnonal))
	for i := 0; i < len(mainDiagnonal); i++ {
		if tData[i] == nil {
			tData[i] = make([]float64, len(mainDiagnonal))
		}
		tData[i][i] = mainDiagnonal[i]
		if i > 0 {
			tData[i][i-1] = secondaryDiagonal[i-1]
		}
		if i < len(secondaryDiagonal) {
			tData[i][i+1] = secondaryDiagonal[i]
		}
	}
	normDiff := MatLInfNorm(tt.Subtract(createRealMatrixFromSlices(t, tData)))
	if math.Abs(0-normDiff) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, normDiff)
	}

	// check the same cached instance is returned the second time
	if !q.Equals(transformer.Q()) {
		t.Errorf("Mismatch. Should equal Q()")
	}
	if !tt.Equals(transformer.T()) {
		t.Errorf("Mismatch. Should equal T()")
	}

}

func checkAEqualPTPt(t *testing.T, m RealMatrix) RealMatrix {
	transformer := createSchurTransformer(t, m)
	p := transformer.P()
	tt := transformer.T()
	pT := transformer.PT()

	result := p.Multiply(tt).Multiply(pT)

	norm := MatLInfNorm(result.Subtract(m))

	if math.Abs(0-norm) > 1.0e-9 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, norm)
	}
	return tt
}

func checkTransformedMatrix(t *testing.T, m RealMatrix) {
	rows := m.RowDimension()
	cols := m.ColumnDimension()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if i > j+1 {
				if math.Abs(0-m.At(i, j)) > 1.0e-16 {
					t.Errorf("Mismatch. want: %v, got: %v", 0, m.At(i, j))
				}
			}
		}
	}
}

func checkAEqualPHPt(t *testing.T, m RealMatrix) RealMatrix {
	transformer := createHessenbergTransformer(t, m)
	p := transformer.P()
	pT := transformer.PT()
	h := transformer.H()

	result := p.Multiply(h).Multiply(pT)
	norm := MatLInfNorm(result.Subtract(m))

	if math.Abs(0-norm) > 1.0e-10 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, norm)
	}

	rows := m.RowDimension()
	cols := m.ColumnDimension()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if i > j+1 {
				if math.Abs(m.At(i, j)-result.At(i, j)) > 1.0e-12 {
					t.Errorf("Mismatch. want: %v, got: %v", m.At(i, j), result.At(i, j))
				}

			}
		}
	}

	return transformer.H()
}

func checkHessenbergMatricesValues(t *testing.T, m, pRef, hRef [][]float64) {
	transformer := createHessenbergTransformer(t, createRealMatrixFromSlices(t, m))

	// check values against known references
	p := transformer.P()
	normDiff := MatLInfNorm(p.Subtract(createRealMatrixFromSlices(t, pRef)))
	if math.Abs(0-normDiff) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, normDiff)
	}

	h := transformer.H()
	normDiff = MatLInfNorm(h.Subtract(createRealMatrixFromSlices(t, hRef)))
	if math.Abs(0-normDiff) > 1.0e-14 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, normDiff)
	}

	// check the same cached instance is returned the second time
	if !p.Equals(transformer.P()) {
		t.Errorf("Mismatch. Should equal P()")
	}
	if !h.Equals(transformer.H()) {
		t.Errorf("Mismatch. Should equal H()")
	}

}
