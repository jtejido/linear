package linear

import (
	"math"
	"testing"
)

func createDiagonalMatrixWithDimension(t *testing.T, dim int) *DiagonalMatrix {
	m, err := NewDiagonalMatrixWithDimension(dim)

	if err != nil {
		t.Errorf("Error while creating Matrix %s", err)
	}

	return m
}

func createDiagonalMatrixFromSlice(t *testing.T, d []float64) *DiagonalMatrix {
	m, err := NewDiagonalMatrixFromSlice(d)

	if err != nil {
		t.Errorf("Error while creating Matrix %s", err)
	}

	return m
}

func createDiagonalMatrix(t *testing.T, data []float64, copyArray bool) *DiagonalMatrix {
	m, err := NewDiagonalMatrix(data, copyArray)

	if err != nil {
		t.Errorf("Error while creating Matrix %s", err)
	}

	return m
}

func TestDiagonalMatrixConstructor1(t *testing.T) {
	dim := 3
	m := createDiagonalMatrixWithDimension(t, dim)
	if m.RowDimension() != dim {
		t.Errorf("Mismatch. want: %v, got: %v", dim, m.RowDimension())
	}
	if m.ColumnDimension() != dim {
		t.Errorf("Mismatch. want: %v, got: %v", dim, m.ColumnDimension())
	}

}

func TestDiagonalMatrixConstructor2(t *testing.T) {
	d := []float64{-1.2, 3.4, 5}
	m := createDiagonalMatrixFromSlice(t, d)
	for i := 0; i < m.RowDimension(); i++ {
		for j := 0; j < m.RowDimension(); j++ {
			if i == j {
				if d[i] != m.At(i, j) {
					t.Errorf("Mismatch. want: %v, got: %v", d[i], m.At(i, j))
				}

			} else {
				if 0 != m.At(i, j) {
					t.Errorf("Mismatch. want: %v, got: %v", 0, m.At(i, j))
				}

			}
		}
	}

	// Check that the underlying was copied.
	d[0] = 0
	if 0 == m.At(0, 0) {
		t.Errorf("Mismatch. want: %v, got: %v", 0, m.At(0, 0))
	}

}

func TestDiagonalMatrixConstructor3(t *testing.T) {
	d := []float64{-1.2, 3.4, 5}
	m := createDiagonalMatrix(t, d, false)
	for i := 0; i < m.RowDimension(); i++ {
		for j := 0; j < m.RowDimension(); j++ {
			if i == j {
				if d[i] != m.At(i, j) {
					t.Errorf("Mismatch. want: %v, got: %v", d[i], m.At(i, j))
				}

			} else {
				if 0 != m.At(i, j) {
					t.Errorf("Mismatch. want: %v, got: %v", 0, m.At(i, j))
				}

			}
		}
	}

	// Check that the underlying is referenced.
	d[0] = 0
	if 0 != m.At(0, 0) {
		t.Errorf("Mismatch. want: %v, got: %v", 0, m.At(0, 0))
	}

}

func TestDiagonalMatrixCopy(t *testing.T) {
	d := []float64{-1.2, 3.4, 5}
	m := createDiagonalMatrix(t, d, false)
	p := m.Copy().(*DiagonalMatrix)
	for i := 0; i < m.RowDimension(); i++ {

		if math.Abs(m.At(i, i)-p.At(i, i)) > 1.0e-20 {
			t.Errorf("Mismatch. want: %v, got: %v", m.At(i, i), p.At(i, i))
		}

	}
}

func TestDiagonalMatrixGetData(t *testing.T) {
	data := []float64{-1.2, 3.4, 5}
	dim := 3
	m := createDiagonalMatrixWithDimension(t, dim)
	for i := 0; i < dim; i++ {
		m.SetEntry(i, i, data[i])
	}

	out := m.Data()
	if dim != len(out) {
		t.Errorf("Mismatch. want: %v, got: %v", dim, len(out))
	}

	for i := 0; i < m.RowDimension(); i++ {
		if dim != len(out[i]) {
			t.Errorf("Mismatch. want: %v, got: %v", dim, len(out[i]))
		}
		for j := 0; j < m.RowDimension(); j++ {
			if i == j {
				if data[i] != out[i][j] {
					t.Errorf("Mismatch. want: %v, got: %v", data[i], out[i][j])
				}

			} else {
				if 0 != out[i][j] {
					t.Errorf("Mismatch. want: %v, got: %v", 0, out[i][j])
				}

			}
		}

	}
}

func TestDiagonalMatrixAdd(t *testing.T) {
	data1 := []float64{-1.2, 3.4, 5}
	m1 := createDiagonalMatrixFromSlice(t, data1)

	data2 := []float64{10.1, 2.3, 45}
	m2 := createDiagonalMatrixFromSlice(t, data2)

	result := m1.Add(m2)
	if m1.RowDimension() != result.RowDimension() {
		t.Errorf("Mismatch. want: %v, got: %v", m1.RowDimension(), result.RowDimension())
	}

	for i := 0; i < result.RowDimension(); i++ {
		for j := 0; j < result.RowDimension(); j++ {
			if i == j {
				if data1[i]+data2[i] != result.At(i, j) {
					t.Errorf("Mismatch. want: %v, got: %v", data1[i]+data2[i], result.At(i, j))
				}

			} else {
				if 0 != result.At(i, j) {
					t.Errorf("Mismatch. want: %v, got: %v", 0, result.At(i, j))
				}
			}
		}
	}
}

func TestDiagonalMatrixSubtract(t *testing.T) {
	data1 := []float64{-1.2, 3.4, 5}
	m1 := createDiagonalMatrixFromSlice(t, data1)

	data2 := []float64{10.1, 2.3, 45}
	m2 := createDiagonalMatrixFromSlice(t, data2)

	result := m1.Subtract(m2)
	if m1.RowDimension() != result.RowDimension() {
		t.Errorf("Mismatch. want: %v, got: %v", m1.RowDimension(), result.RowDimension())
	}
	for i := 0; i < result.RowDimension(); i++ {
		for j := 0; j < result.RowDimension(); j++ {
			if i == j {
				if data1[i]-data2[i] != result.At(i, j) {
					t.Errorf("Mismatch. want: %v, got: %v", data1[i]-data2[i], result.At(i, j))
				}
			} else {
				if 0 != result.At(i, j) {
					t.Errorf("Mismatch. want: %v, got: %v", 0, result.At(i, j))
				}
			}
		}
	}
}

func TestDiagonalMatrixAddToEntry(t *testing.T) {
	data := []float64{-1.2, 3.4, 5}
	m := createDiagonalMatrixFromSlice(t, data)

	for i := 0; i < m.RowDimension(); i++ {
		m.AddToEntry(i, i, float64(i))
		if data[i]+float64(i) != m.At(i, i) {
			t.Errorf("Mismatch. want: %v, got: %v", data[i]+float64(i), m.At(i, i))
		}

	}
}

func TestDiagonalMatrixMultiplyEntry(t *testing.T) {
	data := []float64{-1.2, 3.4, 5}
	m := createDiagonalMatrixFromSlice(t, data)

	for i := 0; i < m.RowDimension(); i++ {
		m.MultiplyEntry(i, i, float64(i))
		if data[i]*float64(i) != m.At(i, i) {
			t.Errorf("Mismatch. want: %v, got: %v", data[i]*float64(i), m.At(i, i))
		}
	}
}

func TestDiagonalMatrixMultiply1(t *testing.T) {
	data1 := []float64{-1.2, 3.4, 5}
	m1 := createDiagonalMatrixFromSlice(t, data1)
	data2 := []float64{10.1, 2.3, 45}
	m2 := createDiagonalMatrixFromSlice(t, data2)

	result := m1.Multiply(m2).(*DiagonalMatrix)
	if m1.RowDimension() != result.RowDimension() {
		t.Errorf("Mismatch. want: %v, got: %v", m1.RowDimension(), result.RowDimension())
	}
	for i := 0; i < result.RowDimension(); i++ {
		for j := 0; j < result.RowDimension(); j++ {
			if i == j {
				if data1[i]*data2[i] != result.At(i, j) {
					t.Errorf("Mismatch. want: %v, got: %v", data1[i]*data2[i], result.At(i, j))
				}
			} else {
				if 0 != result.At(i, j) {
					t.Errorf("Mismatch. want: %v, got: %v", 0, result.At(i, j))
				}
			}
		}
	}
}

func TestDiagonalMatrixMultiply2(t *testing.T) {
	data1 := []float64{-1.2, 3.4, 5}
	diag1 := createDiagonalMatrixFromSlice(t, data1)

	data2 := [][]float64{
		{-1.2, 3.4},
		{-5.6, 7.8},
		{9.1, 2.3},
	}

	dense2 := createArray2DRowRealMatrixFromSlices(t, data2)
	dense1 := createArray2DRowRealMatrixFromSlices(t, diag1.Data())

	diagResult := diag1.Multiply(dense2)
	denseResult := dense1.Multiply(dense2)

	for i := 0; i < dense1.RowDimension(); i++ {
		for j := 0; j < dense2.ColumnDimension(); j++ {
			if denseResult.At(i, j) != diagResult.At(i, j) {
				t.Errorf("Mismatch. want: %v, got: %v", denseResult.At(i, j), diagResult.At(i, j))
			}

		}
	}
}

func TestDiagonalMatrixOperate(t *testing.T) {
	data := []float64{-1.2, 3.4, 5}
	diag := createDiagonalMatrixFromSlice(t, data)
	dense := createArray2DRowRealMatrixFromSlices(t, diag.Data())

	v := []float64{6.7, 890.1, 23.4}
	diagResult := diag.Operate(v)
	denseResult := dense.Operate(v)
	if len(diagResult) != len(denseResult) {
		t.Errorf("Mismatch. want: %v, got: %v", len(diagResult), len(denseResult))
	}
	for i := 0; i < len(diagResult); i++ {
		if diagResult[i] != denseResult[i] {
			t.Errorf("Mismatch. want: %v, got: %v", diagResult[i], denseResult[i])
		}
	}

}

func TestDiagonalMatrixPreMultiply(t *testing.T) {
	data := []float64{-1.2, 3.4, 5}
	diag := createDiagonalMatrixFromSlice(t, data)
	dense := createArray2DRowRealMatrixFromSlices(t, diag.Data())

	v := []float64{6.7, 890.1, 23.4}
	diagResult := diag.PreMultiply(v)
	denseResult := dense.PreMultiply(v)

	if len(diagResult) != len(denseResult) {
		t.Errorf("Mismatch. want: %v, got: %v", len(diagResult), len(denseResult))
	}
	for i := 0; i < len(diagResult); i++ {
		if diagResult[i] != denseResult[i] {
			t.Errorf("Mismatch. want: %v, got: %v", diagResult[i], denseResult[i])
		}
	}
}

func TestDiagonalMatrixPreMultiplyVector(t *testing.T) {
	data := []float64{-1.2, 3.4, 5}
	diag, _ := NewDiagonalMatrixFromSlice(data)
	dense, _ := NewArray2DRowRealMatrixFromSlices(diag.Data(), true)

	v := []float64{6.7, 890.1, 23.4}
	vector := createRealVector(t, v)
	diagResult := diag.PreMultiplyVector(vector)

	denseResult := dense.PreMultiplyVector(vector)

	if !diagResult.Equals(denseResult) {
		t.Errorf("Mismatch. preMultiply(Vector) returns wrong result.")
	}

}

func TestDiagonalMatrixSetNonDiagonalEntry(t *testing.T) {
	diag := createDiagonalMatrixWithDimension(t, 3)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	diag.SetEntry(1, 2, 3.4)
}

func TestDiagonalMatrixSetNonDiagonalZero(t *testing.T) {
	diag := createDiagonalMatrixWithDimension(t, 3)
	diag.SetEntry(1, 2, 0.0)
	if math.Abs(0.0-diag.At(1, 2)) > math.SmallestNonzeroFloat64 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, diag.At(1, 2))
	}

}

func TestDiagonalMatrixAddNonDiagonalEntry(t *testing.T) {
	diag := createDiagonalMatrixWithDimension(t, 3)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	diag.AddToEntry(1, 2, 3.4)
}

func TestDiagonalMatrixAddNonDiagonalZero(t *testing.T) {
	diag := createDiagonalMatrixWithDimension(t, 3)
	diag.AddToEntry(1, 2, 0.0)
	if math.Abs(0.0-diag.At(1, 2)) > math.SmallestNonzeroFloat64 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, diag.At(1, 2))
	}
}

func TestDiagonalMatrixMultiplyNonDiagonalEntry(t *testing.T) {
	diag := createDiagonalMatrixWithDimension(t, 3)
	diag.MultiplyEntry(1, 2, 3.4)
	if math.Abs(0.0-diag.At(1, 2)) > math.SmallestNonzeroFloat64 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, diag.At(1, 2))
	}
}

func TestDiagonalMatrixMultiplyNonDiagonalZero(t *testing.T) {
	diag := createDiagonalMatrixWithDimension(t, 3)
	diag.MultiplyEntry(1, 2, 0.0)
	if math.Abs(0.0-diag.At(1, 2)) > math.SmallestNonzeroFloat64 {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, diag.At(1, 2))
	}
}

func TestDiagonalMatrixSetEntryOutOfRange(t *testing.T) {
	diag := createDiagonalMatrixWithDimension(t, 3)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	diag.SetEntry(3, 3, 3.4)
}

func TestDiagonalMatrixNull(t *testing.T) {
	_, err := NewDiagonalMatrix(nil, false)
	if err == nil {
		t.Errorf("panic expected.")
	}
}

func TestDiagonalMatrixSetSubMatrixError(t *testing.T) {
	data := []float64{-1.2, 3.4, 5}
	diag := createDiagonalMatrixFromSlice(t, data)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	diag.SetSubMatrix([][]float64{{1.0, 1.0}, {1.0, 1.0}}, 1, 1)
}

func TestDiagonalMatrixSetSubMatrix(t *testing.T) {
	data := []float64{-1.2, 3.4, 5}
	diag := createDiagonalMatrixFromSlice(t, data)
	diag.SetSubMatrix([][]float64{{0.0, 5.0, 0.0}, {0.0, 0.0, 6.0}}, 1, 0)
	if math.Abs(-1.2-diag.At(0, 0)) > 1.0e-20 {
		t.Errorf("Mismatch. want: %v, got: %v", -1.2, diag.At(0, 0))
	}
	if math.Abs(5.0-diag.At(1, 1)) > 1.0e-20 {
		t.Errorf("Mismatch. want: %v, got: %v", 5.0, diag.At(1, 1))
	}
	if math.Abs(6.0-diag.At(2, 2)) > 1.0e-20 {
		t.Errorf("Mismatch. want: %v, got: %v", 6.0, diag.At(2, 2))
	}

}

func TestDiagonalMatrixInverseError(t *testing.T) {
	data := []float64{1, 2, 0}
	diag := createDiagonalMatrixFromSlice(t, data)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	diag.Inverse()
}

func TestDiagonalMatrixInverseWithThresholdError(t *testing.T) {
	data := []float64{1, 2, 1e-6}
	diag := createDiagonalMatrixFromSlice(t, data)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	diag.InverseWithThreshold(1e-5)
}

func TestDiagonalMatrixInverse(t *testing.T) {
	data := []float64{1, 2, 3}
	m := createDiagonalMatrixFromSlice(t, data)
	inverse := m.Inverse()

	result := m.Multiply(inverse)
	if !result.Equals(createRealIdentityMatrix(t, len(data))) {
		t.Errorf("Diagonalinverse() returns wrong result")
	}

}
