package linear

import (
	"math"
	"testing"
)

var (
	testValues []float64
)

func init() {
	x := preferredEntryValue()
	y := x + 1
	z := y + 1
	testValues = []float64{
		math.NaN(), math.Inf(1), math.Inf(-1), 0, -0, x, y, z, 2 * x, -x, 1 / x, x * x, x + y, x - y, y - x,
	}
}

func create(t *testing.T, data []float64) RealVector {
	v, err := NewArrayRealVectorFromSlice(data)
	if err != nil {
		t.Errorf("Error while creating Matrix %s", err)
	}

	return v
}

func TestArrayRealVectorConstructors(t *testing.T) {
	vec1 := []float64{1, 2, 3}

	dvec1 := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
	v1, _ := NewSizedArrayRealVector(7)
	if 7 != v1.Dimension() {
		t.Errorf("Mismatch. want: %v, got: %v", 7, v1.Dimension())
	}
	if 0.0 != v1.At(6) {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, v1.At(6))
	}

	v2, _ := NewSizedArrayRealVectorWithPreset(5, 1.23)
	if 5 != v2.Dimension() {
		t.Errorf("Mismatch. want: %v, got: %v", 5, v2.Dimension())
	}
	if 1.23 != v2.At(4) {
		t.Errorf("Mismatch. want: %v, got: %v", 1.23, v2.At(4))
	}

	v3, _ := NewArrayRealVectorFromSlice(vec1)
	if 3 != v3.Dimension() {
		t.Errorf("Mismatch. want: %v, got: %v", 3, v3.Dimension())
	}
	if 2.0 != v3.At(1) {
		t.Errorf("Mismatch. want: %v, got: %v", 2.0, v3.At(1))
	}

	v3_bis, _ := NewArrayRealVector(vec1, true)
	if 3 != v3_bis.Dimension() {
		t.Errorf("Mismatch. want: %v, got: %v", 3, v3_bis.Dimension())
	}
	if 2.0 != v3_bis.At(1) {
		t.Errorf("Mismatch. want: %v, got: %v", 2.0, v3_bis.At(1))
	}

	v3_ter, _ := NewArrayRealVector(vec1, false)
	if 3 != v3_ter.Dimension() {
		t.Errorf("Mismatch. want: %v, got: %v", 3, v3_ter.Dimension())
	}
	if 2.0 != v3_ter.At(1) {
		t.Errorf("Mismatch. want: %v, got: %v", 2.0, v3_ter.At(1))
	}

	v5_i, _ := NewArrayRealVectorFromSlice(dvec1)
	if 9 != v5_i.Dimension() {
		t.Errorf("Mismatch. want: %v, got: %v", 9, v5_i.Dimension())
	}
	if 9.0 != v5_i.At(8) {
		t.Errorf("Mismatch. want: %v, got: %v", 9.0, v5_i.At(8))
	}

	v7, _ := NewArrayRealVectorCopy(v1)
	if 7 != v7.Dimension() {
		t.Errorf("Mismatch. want: %v, got: %v", 7, v7.Dimension())
	}
	if 0.0 != v7.At(6) {
		t.Errorf("Mismatch. want: %v, got: %v", 0.0, v7.At(6))
	}

}

func TestArrayRealVectorData(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	v, _ := NewArrayRealVectorFromSlice(data)
	v.DataRef()[0] = 0

	if v.At(0) != 0 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, v.At(0))
	}

}

func TestArrayRealVectorZeroVectors(t *testing.T) {
	data := []float64{}
	v, _ := NewArrayRealVectorFromSlice(data)

	if v.Dimension() != 0 {
		t.Errorf("Mismatch. want: %v, got: %v", 0, v.Dimension())
	}

}

func preferredEntryValue() float64 {
	return 0.0
}

func TestArrayRealVectorDimension(t *testing.T) {
	x := preferredEntryValue()
	data1 := []float64{x, x, x, x}
	if len(data1) != create(t, data1).Dimension() {
		t.Errorf("Mismatch. want: %v, got: %v", len(data1), create(t, data1).Dimension())
	}

	y := x + 1
	data2 := []float64{y, y, y, y}
	if len(data2) != create(t, data2).Dimension() {
		t.Errorf("Mismatch. want: %v, got: %v", len(data1), create(t, data1).Dimension())
	}

}

func TestArrayRealVectorEntryAt(t *testing.T) {
	x := preferredEntryValue()
	data := []float64{x, 1, 2, x, x}
	v := create(t, data)
	for i := 0; i < len(data); i++ {
		if data[i] != v.At(i) {
			t.Errorf("Mismatch. want: %v, got: %v", data[i], v.At(i))
		}

	}
}

func TestArrayRealVectorEntryInvalidIndex1(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	create(t, make([]float64, 4)).At(-1)
}

func TestArrayRealVectorEntryEntryInvalidIndex2(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	create(t, make([]float64, 4)).At(4)
}

func TestArrayRealVectorSetEntry(t *testing.T) {
	x := preferredEntryValue()
	data := []float64{x, 1, 2, x, x}
	expected := append([]float64{}, data...)
	actual := create(t, data)

	/*
	 * Try setting to any value.
	 */
	for i := 0; i < len(data); i++ {
		oldValue := data[i]
		newValue := oldValue + 1
		expected[i] = newValue
		actual.SetEntry(i, newValue)
		for j := 0; j < len(expected); j++ {
			if expected[j] != actual.At(j) {
				t.Errorf("Mismatch. want: %v, got: %v", expected[j], actual.At(j))
			}
		}

		expected[i] = oldValue
		actual.SetEntry(i, oldValue)
	}

	/*
	 * Try setting to the preferred value.
	 */
	for i := 0; i < len(data); i++ {
		oldValue := data[i]
		newValue := x
		expected[i] = newValue
		actual.SetEntry(i, newValue)
		for j := 0; j < len(expected); j++ {
			if expected[j] != actual.At(j) {
				t.Errorf("Mismatch. want: %v, got: %v", expected[j], actual.At(j))
			}
		}
		expected[i] = oldValue
		actual.SetEntry(i, oldValue)
	}
}

func TestArrayRealVectorSetEntryInvalidIndex1(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	create(t, make([]float64, 4)).SetEntry(-1, preferredEntryValue())
}

func TestArrayRealVectorSetEntryInvalidIndex2(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	create(t, make([]float64, 4)).SetEntry(4, preferredEntryValue())
}

func TestArrayRealVectorAddToEntry(t *testing.T) {
	x := preferredEntryValue()
	data1 := []float64{x, 1, 2, x, x}

	expected := append([]float64{}, data1...)
	actual := create(t, data1)

	/*
	 * Try adding any value.
	 */
	increment := 1.
	for i := 0; i < len(data1); i++ {
		oldValue := data1[i]
		expected[i] += increment
		actual.AddToEntry(i, increment)
		for j := 0; j < len(expected); j++ {
			if expected[j] != actual.At(j) {
				t.Errorf("Mismatch. want: %v, got: %v", expected[j], actual.At(j))
			}
		}

		expected[i] = oldValue
		actual.SetEntry(i, oldValue)
	}

	/*
	 * Try incrementing so that result is equal to preferred value.
	 */
	for i := 0; i < len(data1); i++ {
		oldValue := data1[i]
		increment = x - oldValue
		expected[i] = x
		actual.AddToEntry(i, increment)
		for j := 0; j < len(expected); j++ {
			if expected[j] != actual.At(j) {
				t.Errorf("Mismatch. want: %v, got: %v", expected[j], actual.At(j))
			}
		}

		expected[i] = oldValue
		actual.SetEntry(i, oldValue)
	}
}

func TestArrayRealVectorAddToEntryInvalidIndex1(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	create(t, make([]float64, 3)).AddToEntry(-1, preferredEntryValue())
}

func TestArrayRealVectorAddToEntryInvalidIndex2(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	create(t, make([]float64, 3)).AddToEntry(4, preferredEntryValue())
}

func TestArrayRealVectorAppendVector(t *testing.T) {
	x := preferredEntryValue()
	data1 := []float64{x, 1, 2, x, x}
	data2 := []float64{x, x, 3, x, 4, x}

	doTestAppendVector(t, create(t, data1), create(t, data2), 0)
}

func TestArrayRealVectorAppendScalar(t *testing.T) {
	x := preferredEntryValue()
	data := []float64{x, 1, 2, x, x}

	doTestAppendScalar(t, create(t, data), 1, 0)
	doTestAppendScalar(t, create(t, data), x, 0)
}

func TestArrayRealVectorSubVectorAt(t *testing.T) {
	x := preferredEntryValue()
	data := []float64{x, x, x, 1, x, 2, x, x, 3, x, x, x, 4, x, x, x}
	index := 1
	n := len(data) - 5
	actual := create(t, data).SubVector(index, n)
	expected := make([]float64, n)
	copy(expected[0:n], data[index:index+n])
	for i := 0; i < len(expected); i++ {
		if math.Abs(expected[i]-actual.At(i)) > 0 {
			t.Errorf("Mismatch. want: %v, got: %v", expected[i], actual.At(i))
		}
	}
}

func TestArrayRealVectorSubVectorInvalidIndex1(t *testing.T) {
	n := 10
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	create(t, make([]float64, n)).SubVector(-1, 2)
}

func TestArrayRealVectorSubVectorInvalidIndex2(t *testing.T) {
	n := 10
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	create(t, make([]float64, n)).SubVector(n, 2)
}

func TestArrayRealVectorSubVectorInvalidIndex3(t *testing.T) {
	n := 10
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	create(t, make([]float64, n)).SubVector(0, n+1)
}

func TestArrayRealVectorSubVectorInvalidIndex4(t *testing.T) {
	n := 10
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	create(t, make([]float64, n)).SubVector(3, -2)
}

func TestArrayRealVectorSetSubVectorSameType(t *testing.T) {
	x := preferredEntryValue()
	expected := []float64{x, x, x, 1, x, 2, x, x, 3, x, x, x, 4, x, x, x}
	sub := []float64{5, x, 6, 7, 8}
	actual := create(t, expected)
	index := 2
	actual.SetSubVector(index, create(t, sub))

	for i := 0; i < len(sub); i++ {
		expected[index+i] = sub[i]
	}

	for i := 0; i < len(expected); i++ {
		if math.Abs(expected[i]-actual.At(i)) > 0 {
			t.Errorf("Mismatch. want: %v, got: %v", expected[i], actual.At(i))
		}
	}
}

func TestArrayRealVectorSetSubVectorInvalidIndex1(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	create(t, make([]float64, 10)).SetSubVector(-1, create(t, make([]float64, 2)))
}

func TestArrayRealVectorSetSubVectorInvalidIndex2(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	create(t, make([]float64, 10)).SetSubVector(10, create(t, make([]float64, 2)))
}

func TestArrayRealVectorSetSubVectorInvalidIndex3(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	create(t, make([]float64, 10)).SetSubVector(9, create(t, make([]float64, 2)))
}

func TestArrayRealVectorIsNaN(t *testing.T) {
	v := create(t, []float64{0, 1, 2})
	if v.IsNaN() {
		t.Errorf("Mismatch. expected not NaN.")
	}

	v.SetEntry(1, math.NaN())
	if !v.IsNaN() {
		t.Errorf("Mismatch. expected NaN.")
	}
}

func TestArrayRealVectorIsInfinite(t *testing.T) {
	v := create(t, []float64{0, 1, 2})
	if v.IsInf() {
		t.Errorf("Mismatch. expected not Inf.")
	}

	v.SetEntry(0, math.Inf(1))
	if !v.IsInf() {
		t.Errorf("Mismatch. expected Inf.")
	}

	v.SetEntry(1, math.NaN())
	if v.IsInf() {
		t.Errorf("Mismatch. expected not Inf.")
	}

}

func TestArrayRealVectorAdd(t *testing.T) {
	data1 := make([]float64, len(testValues)*len(testValues))
	data2 := make([]float64, len(testValues)*len(testValues))
	var k int
	for i := 0; i < len(testValues); i++ {
		for j := 0; j < len(testValues); j++ {
			data1[k] = testValues[i]
			data2[k] = testValues[j]
			k++
		}
	}
	v1 := create(t, data1)
	v2 := create(t, data2)
	actual := v1.Add(v2)
	expected := make([]float64, len(data1))
	for i := 0; i < len(expected); i++ {
		expected[i] = data1[i] + data2[i]
	}
	for i := 0; i < len(expected); i++ {
		isSpecial := math.IsNaN(expected[i]) || math.IsInf(expected[i], 1) || math.IsInf(expected[i], -1)
		if !isSpecial {
			if math.Abs(expected[i]-actual.At(i)) > 0 {
				t.Errorf("Mismatch. want: %v, got: %v", expected[i], actual.At(i))
			}
		}
	}
}

func TestArrayRealVectorAddDimensionMismatch(t *testing.T) {
	n := 10
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	create(t, make([]float64, n)).Add(create(t, make([]float64, n+1)))
}

func TestArrayRealVectorSubtract(t *testing.T) {
	data1 := make([]float64, len(testValues)*len(testValues))
	data2 := make([]float64, len(testValues)*len(testValues))
	var k int
	for i := 0; i < len(testValues); i++ {
		for j := 0; j < len(testValues); j++ {
			data1[k] = testValues[i]
			data2[k] = testValues[j]
			k++
		}
	}
	v1 := create(t, data1)
	v2 := create(t, data2)
	actual := v1.Subtract(v2)
	expected := make([]float64, len(data1))
	for i := 0; i < len(expected); i++ {
		expected[i] = data1[i] - data2[i]
	}
	for i := 0; i < len(expected); i++ {
		isSpecial := math.IsNaN(expected[i]) || math.IsInf(expected[i], 1) || math.IsInf(expected[i], -1)
		if !isSpecial {
			if math.Abs(expected[i]-actual.At(i)) > 0 {
				t.Errorf("Mismatch. want: %v, got: %v", expected[i], actual.At(i))
			}
		}
	}
}

func TestArrayRealVectorSubtractDimensionMismatch(t *testing.T) {
	n := 10
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	create(t, make([]float64, n)).Subtract(create(t, make([]float64, n+1)))
}

func TestArrayRealVectorEBEMultiply(t *testing.T) {
	data1 := make([]float64, len(testValues)*len(testValues))
	data2 := make([]float64, len(testValues)*len(testValues))
	var k int
	for i := 0; i < len(testValues); i++ {
		for j := 0; j < len(testValues); j++ {
			data1[k] = testValues[i]
			data2[k] = testValues[j]
			k++
		}
	}
	v1 := create(t, data1)
	v2 := create(t, data2)
	actual := v1.EBEMultiply(v2)
	expected := make([]float64, len(data1))
	for i := 0; i < len(expected); i++ {
		expected[i] = data1[i] * data2[i]
	}
	for i := 0; i < len(expected); i++ {
		isSpecial := math.IsNaN(expected[i]) || math.IsInf(expected[i], 1) || math.IsInf(expected[i], -1)
		if !isSpecial {
			if math.Abs(expected[i]-actual.At(i)) > 0 {
				t.Errorf("Mismatch. want: %v, got: %v", expected[i], actual.At(i))
			}
		}
	}
}

func TestArrayRealVectorEBEMultiplyDimensionMismatch(t *testing.T) {
	n := 10
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	create(t, make([]float64, n)).EBEMultiply(create(t, make([]float64, n+1)))
}

func TestArrayRealVectorEBEDivide(t *testing.T) {
	data1 := make([]float64, len(testValues)*len(testValues))
	data2 := make([]float64, len(testValues)*len(testValues))
	var k int
	for i := 0; i < len(testValues); i++ {
		for j := 0; j < len(testValues); j++ {
			data1[k] = testValues[i]
			data2[k] = testValues[j]
			k++
		}
	}
	v1 := create(t, data1)
	v2 := create(t, data2)
	actual := v1.EBEDivide(v2)
	expected := make([]float64, len(data1))
	for i := 0; i < len(expected); i++ {
		expected[i] = data1[i] / data2[i]
	}
	for i := 0; i < len(expected); i++ {
		isSpecial := math.IsNaN(expected[i]) || math.IsInf(expected[i], 1) || math.IsInf(expected[i], -1)
		if !isSpecial {
			if math.Abs(expected[i]-actual.At(i)) > 0 {
				t.Errorf("Mismatch. want: %v, got: %v", expected[i], actual.At(i))
			}
		}
	}
}

func TestArrayRealVectorEBEDivideDimensionMismatch(t *testing.T) {
	n := 10
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	create(t, make([]float64, n)).EBEDivide(create(t, make([]float64, n+1)))
}

func TestArrayRealVectorDistance(t *testing.T) {
	x := preferredEntryValue()
	data1 := []float64{x, x, 1, x, 2, x, x, 3, x}
	data2 := []float64{4, x, x, 5, 6, 7, x, x, 8}
	v1 := create(t, data1)
	v2 := create(t, data2)
	actual := VecDistance(v1, v2)
	var expected float64
	for i := 0; i < len(data1); i++ {
		delta := data2[i] - data1[i]
		expected += delta * delta
	}
	expected = math.Sqrt(expected)
	if expected != actual {
		t.Errorf("Mismatch. want: %v, got: %v", expected, actual)
	}

}

func TestArrayRealVectorDistanceDimensionMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	VecDistance(create(t, make([]float64, 4)), create(t, make([]float64, 5)))
}

func TestArrayRealVectorNorm(t *testing.T) {
	x := preferredEntryValue()
	data := []float64{x, x, 1, x, 2, x, x, 3, x}
	v := create(t, data)
	actual := VecNorm(v)
	var expected float64
	for i := 0; i < len(data); i++ {
		expected += data[i] * data[i]
	}
	expected = math.Sqrt(expected)
	if expected != actual {
		t.Errorf("Mismatch. want: %v, got: %v", expected, actual)
	}
}

func TestArrayRealVectorL1Distance(t *testing.T) {
	x := preferredEntryValue()
	data1 := []float64{x, x, 1, x, 2, x, x, 3, x}
	data2 := []float64{4, x, x, 5, 6, 7, x, x, 8}
	v1 := create(t, data1)
	v2 := create(t, data2)
	actual := VecL1Distance(v1, v2)
	var expected float64
	for i := 0; i < len(data1); i++ {
		delta := data2[i] - data1[i]
		expected += math.Abs(delta)
	}
	if expected != actual {
		t.Errorf("Mismatch. want: %v, got: %v", expected, actual)
	}
}

func TestArrayRealVectorL1DistanceDimensionMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	VecL1Distance(create(t, make([]float64, 4)), create(t, make([]float64, 5)))
}

func TestArrayRealVectorL1Norm(t *testing.T) {
	x := preferredEntryValue()
	data := []float64{x, x, 1, x, 2, x, x, 3, x}
	v := create(t, data)
	actual := VecL1Norm(v)
	var expected float64
	for i := 0; i < len(data); i++ {
		expected += math.Abs(data[i])
	}
	if expected != actual {
		t.Errorf("Mismatch. want: %v, got: %v", expected, actual)
	}
}

func TestArrayRealVectorLInfDistance(t *testing.T) {
	x := preferredEntryValue()
	data1 := []float64{x, x, 1, x, 2, x, x, 3, x}
	data2 := []float64{4, x, x, 5, 6, 7, x, x, 8}
	v1 := create(t, data1)
	v2 := create(t, data2)
	actual := VecLInfDistance(v1, v2)
	var expected float64
	for i := 0; i < len(data1); i++ {
		delta := data2[i] - data1[i]
		expected = math.Max(expected, math.Abs(delta))
	}
	if expected != actual {
		t.Errorf("Mismatch. want: %v, got: %v", expected, actual)
	}
}

func TestArrayRealVectorLInfDistanceDimensionMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	VecLInfDistance(create(t, make([]float64, 4)), create(t, make([]float64, 5)))
}

func TestArrayRealVectorMapAdd(t *testing.T) {
	expected := make([]float64, len(testValues))
	for i := 0; i < len(testValues); i++ {
		d := testValues[i]
		for j := 0; j < len(expected); j++ {
			expected[j] = testValues[j] + d
		}
		actual := create(t, testValues)
		actual.MapAdd(d)
		for j := 0; j < len(expected); j++ {
			if math.Abs(expected[j]-actual.At(j)) > 0 {
				t.Errorf("Mismatch. want: %v, got: %v", expected[j], actual.At(j))
			}
		}
	}
}

func TestArrayRealVectorMapSubtract(t *testing.T) {
	expected := make([]float64, len(testValues))
	for i := 0; i < len(testValues); i++ {
		d := testValues[i]
		for j := 0; j < len(expected); j++ {
			expected[j] = testValues[j] - d
		}
		actual := create(t, testValues)
		actual.MapSubtract(d)
		for j := 0; j < len(expected); j++ {
			if math.Abs(expected[j]-actual.At(j)) > 0 {
				t.Errorf("Mismatch. want: %v, got: %v", expected[j], actual.At(j))
			}
		}
	}
}

func TestArrayRealVectorMapMultiply(t *testing.T) {
	expected := make([]float64, len(testValues))
	for i := 0; i < len(testValues); i++ {
		d := testValues[i]
		for j := 0; j < len(expected); j++ {
			expected[j] = testValues[j] * d
		}
		actual := create(t, testValues)
		actual.MapMultiply(d)
		for j := 0; j < len(expected); j++ {
			if math.Abs(expected[j]-actual.At(j)) > 0 {
				t.Errorf("Mismatch. want: %v, got: %v", expected[j], actual.At(j))
			}
		}
	}
}

func TestArrayRealVectorMapDivide(t *testing.T) {
	expected := make([]float64, len(testValues))
	for i := 0; i < len(testValues); i++ {
		d := testValues[i]
		for j := 0; j < len(expected); j++ {
			expected[j] = testValues[j] / d
		}
		actual := create(t, testValues)
		actual.MapDivide(d)
		for j := 0; j < len(expected); j++ {
			if math.Abs(expected[j]-actual.At(j)) > 0 {
				t.Errorf("Mismatch. want: %v, got: %v", expected[j], actual.At(j))
			}
		}
	}
}

func TestArrayRealVectorMap(t *testing.T) {
	functions := createFunctions()
	for _, f := range functions {
		doTestMapFunction(t, f)
	}
}

// Move to matrix space
func TestArrayRealVectorOuterProduct(t *testing.T) {
	dataU := testValues
	u := create(t, dataU)
	dataV := make([]float64, len(testValues)+3)
	copy(dataV[:len(testValues)], testValues[:len(testValues)])

	dataV[len(testValues)] = 1
	dataV[len(testValues)] = -2
	dataV[len(testValues)] = 3
	v := create(t, dataV)
	uv := OuterProduct(u, v)
	if len(dataU) != uv.RowDimension() {
		t.Errorf("Mismatch. want: %v, got: %v", len(dataU), uv.RowDimension())
	}

	if len(dataV) != uv.ColumnDimension() {
		t.Errorf("Mismatch. want: %v, got: %v", len(dataU), uv.ColumnDimension())
	}

	for i := 0; i < len(dataU); i++ {
		for j := 0; j < len(dataV); j++ {
			expected := dataU[i] * dataV[j]
			actual := uv.At(i, j)
			if math.Abs(expected-actual) > 0 {
				t.Errorf("Mismatch. want: %v, got: %v", expected, actual)
			}

		}
	}
}

func TestArrayRealVectorProjection(t *testing.T) {
	x := preferredEntryValue()
	data1 := []float64{
		x, 1, x, x, 2, x, x, x, 3, x, x, x, x,
	}
	data2 := []float64{
		5, -6, 7, x, x, -8, -9, 10, 11, x, 12, 13, -15,
	}
	var dotProduct, norm2 float64

	for i := 0; i < len(data1); i++ {
		dotProduct += data1[i] * data2[i]
		norm2 += data2[i] * data2[i]
	}
	s := dotProduct / norm2
	expected := make([]float64, len(data1))
	for i := 0; i < len(data2); i++ {
		expected[i] = s * data2[i]
	}
	v1 := create(t, data1)
	v2 := create(t, data2)
	actual := Projection(v1, v2)
	for j := 0; j < len(expected); j++ {
		if math.Abs(expected[j]-actual.At(j)) > 0 {
			t.Errorf("Mismatch. want: %v, got: %v", expected[j], actual.At(j))
		}
	}

}

func TestArrayRealVectorProjectionNullVector(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	Projection(create(t, make([]float64, 4)), create(t, make([]float64, 4)))
}

func TestArrayRealVectorProjectionDimensionMismatch(t *testing.T) {
	v1 := create(t, make([]float64, 4))
	v2 := create(t, make([]float64, 5))
	v2.Set(1.0)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	Projection(v1, v2)
}

func TestArrayRealVectorSet(t *testing.T) {
	for i := 0; i < len(testValues); i++ {
		expected := testValues[i]
		v := create(t, testValues)
		v.Set(expected)
		for j := 0; j < len(testValues); j++ {
			if math.Abs(expected-v.At(j)) > 0 {
				t.Errorf("Mismatch. want: %v, got: %v", expected, v.At(j))
			}
		}
	}
}

func TestArrayRealVectorToArray(t *testing.T) {
	data := create(t, testValues).ToArray()
	for i := 0; i < len(testValues); i++ {
		if math.Abs(testValues[i]-data[i]) > 0 {
			t.Errorf("Mismatch. want: %v, got: %v", testValues[i], data[i])
		}
	}
}

func TestArrayRealVectorTestUnitize(t *testing.T) {
	x := preferredEntryValue()
	data := []float64{
		x, 1, x, x, 2, x, x, x, 3, x, x, x, x,
	}
	var norm float64
	for i := 0; i < len(data); i++ {
		norm += data[i] * data[i]
	}
	norm = math.Sqrt(norm)
	expected := make([]float64, len(data))
	for i := 0; i < len(expected); i++ {
		expected[i] = data[i] / norm
	}
	actual := create(t, data)
	actual.Unitize()
	for j := 0; j < len(expected); j++ {
		if math.Abs(expected[j]-actual.At(j)) > 0 {
			t.Errorf("Mismatch. want: %v, got: %v", expected[j], actual.At(j))
		}
	}
}

func TestArrayRealVectorTestUnitVector(t *testing.T) {
	x := preferredEntryValue()
	data := []float64{
		x, 1, x, x, 2, x, x, x, 3, x, x, x, x,
	}
	var norm float64
	for i := 0; i < len(data); i++ {
		norm += data[i] * data[i]
	}
	norm = math.Sqrt(norm)
	expected := make([]float64, len(data))
	for i := 0; i < len(expected); i++ {
		expected[i] = data[i] / norm
	}
	v := create(t, data)
	actual := UnitVector(v)
	for j := 0; j < len(expected); j++ {
		if math.Abs(expected[j]-actual.At(j)) > 0 {
			t.Errorf("Mismatch. want: %v, got: %v", expected[j], actual.At(j))
		}
	}
}

func TestArrayRealVectorUnitizeNullVector(t *testing.T) {
	data := []float64{
		0, 0, 0, 0, 0,
	}
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	create(t, data).Unitize()
}

func TestArrayRealVectorUnitVectorNullVector(t *testing.T) {
	data := []float64{
		0, 0, 0, 0, 0,
	}
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	UnitVector(create(t, data))
}

func TestArrayRealVectorIterator(t *testing.T) {
	v := create(t, testValues)
	it := v.Iterator()
	for i := 0; i < len(testValues); i++ {
		if !it.HasNext() {
			t.Errorf("Mismatch. Should have next entry.")
		}

		e := it.Next()
		if i != e.Index() {
			t.Errorf("Mismatch. want: %v, got: %v", i, e.Index())
		}

		if math.Abs(testValues[i]-e.Value()) > 0 {
			t.Errorf("Mismatch. want: %v, got: %v", testValues[i], e.Value())
		}
	}

	if it.HasNext() {
		t.Errorf("Mismatch. Should have no entry left.")
	}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	it.Next()

}

func TestArrayRealVectorCombine(t *testing.T) {
	n := len(testValues) * len(testValues)
	data1 := make([]float64, n)
	data2 := make([]float64, n)
	for i := 0; i < len(testValues); i++ {
		for j := 0; j < len(testValues); j++ {
			index := len(testValues)*i + j
			data1[index] = testValues[i]
			data2[index] = testValues[j]
		}
	}
	v1 := create(t, data1)
	v2 := create(t, data2)
	expected := make([]float64, n)
	for i := 0; i < len(testValues); i++ {
		a1 := testValues[i]
		for j := 0; j < len(testValues); j++ {
			a2 := testValues[j]
			for k := 0; k < n; k++ {
				expected[k] = a1*data1[k] + a2*data2[k]
			}
			actual := v1.Copy()
			actual.Combine(a1, a2, v2)
			for ii := 0; ii < len(expected); ii++ {
				if math.Abs(expected[ii]-actual.At(ii)) > 0 {
					t.Errorf("Mismatch. want: %v, got: %v", expected[ii], actual.At(ii))
				}
			}

		}
	}
}

func TestArrayRealVectorCombineDimensionMismatch(t *testing.T) {
	v1 := create(t, make([]float64, 10))
	v2 := create(t, make([]float64, 15))
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	v1.Combine(1.0, 1.0, v2)
}

func TestArrayRealVectorCopy(t *testing.T) {
	v := create(t, testValues)
	w := v.Copy()
	if v == w {
		t.Errorf("Mismatch. structs should not be the same pointer.")
	}

	for ii := 0; ii < len(testValues); ii++ {
		if math.Abs(testValues[ii]-w.At(ii)) > 0 {
			t.Errorf("Mismatch. want: %v, got: %v", testValues[ii], w.At(ii))
		}
	}
}

func TestArrayRealVectorDotProductRegulartestValues(t *testing.T) {
	x := preferredEntryValue()
	data1 := []float64{
		x, 1, x, x, 2, x, x, x, 3, x, x, x, x,
	}
	data2 := []float64{
		5, -6, 7, x, x, -8, -9, 10, 11, x, 12, 13, -15,
	}
	var expected float64
	for i := 0; i < len(data1); i++ {
		expected += data1[i] * data2[i]
	}
	v1 := create(t, data1)
	v2 := create(t, data2)
	actual := VecDotProduct(v1, v2)

	if math.Abs(expected-actual) > 0 {
		t.Errorf("Mismatch. want: %v, got: %v", expected, actual)
	}
}

func TestArrayRealVectorDotProductSpecialtestValues(t *testing.T) {
	for i := 0; i < len(testValues); i++ {
		data1 := []float64{
			testValues[i],
		}
		v1 := create(t, data1)
		for j := 0; j < len(testValues); j++ {
			data2 := []float64{
				testValues[j],
			}
			v2 := create(t, data2)
			expected := data1[0] * data2[0]
			actual := VecDotProduct(v1, v2)
			if math.Abs(expected-actual) > 0 {
				t.Errorf("Mismatch. want: %v, got: %v", expected, actual)
			}
		}
	}
}

func TestArrayRealVectorDotProductDimensionMismatch(t *testing.T) {
	data1 := make([]float64, 10)
	data2 := make([]float64, len(data1)+1)
	v1 := create(t, data1)
	v2 := create(t, data2)
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	VecDotProduct(v1, v2)
}

func TestArrayRealVectorCosine(t *testing.T) {
	x := preferredEntryValue()
	data1 := []float64{
		x, 1, x, x, 2, x, x, x, 3, x, x, x, x,
	}
	data2 := []float64{
		5, -6, 7, x, x, -8, -9, 10, 11, x, 12, 13, -15,
	}
	var norm1, norm2, dotProduct float64
	for i := 0; i < len(data1); i++ {
		norm1 += data1[i] * data1[i]
		norm2 += data2[i] * data2[i]
		dotProduct += data1[i] * data2[i]
	}
	norm1 = math.Sqrt(norm1)
	norm2 = math.Sqrt(norm2)
	expected := dotProduct / (norm1 * norm2)
	v1 := create(t, data1)
	v2 := create(t, data2)
	actual := VecCosine(v1, v2)
	if math.Abs(expected-actual) > 0 {
		t.Errorf("Mismatch. want: %v, got: %v", expected, actual)
	}
}

func TestArrayRealVectorCosineLeftNullVector(t *testing.T) {
	v := create(t, []float64{0, 0, 0})
	w := create(t, []float64{1, 0, 0})
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	VecCosine(v, w)
}

func TestArrayRealVectorCosineRightNullVector(t *testing.T) {
	v := create(t, []float64{0, 0, 0})
	w := create(t, []float64{1, 0, 0})
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	VecCosine(w, v)
}

func TestArrayRealVectorCosineDimensionMismatch(t *testing.T) {
	v := create(t, []float64{1, 2, 3})
	w := create(t, []float64{1, 2, 3, 4})
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	VecCosine(v, w)
}

func TestArrayRealVectorEquals(t *testing.T) {
	v := create(t, []float64{0, 1, 2})
	if !v.Equals(v) {
		t.Errorf("Mismatch. v vector should be equal to itself")
	}

	if !v.Equals(v.Copy()) {
		t.Errorf("Mismatch. v vector should be equal to a copy of itself")
	}

	if v.Equals(nil) {
		t.Errorf("Mismatch. v vector should not be equal to nil")
	}

	if v.Equals(v.SubVector(0, v.Dimension()-1)) {
		t.Errorf("Mismatch. v vector should not be equal its sub vector (w/ dim -1).")
	}

	if !v.Equals(v.SubVector(0, v.Dimension())) {
		t.Errorf("Mismatch. v vector should be equal to its subvector (same dim)")
	}
}

func TestArrayRealVectorMinMax(t *testing.T) {
	v1 := create(t, []float64{0, -6, 4, 12, 7})
	if 1 != MinIndex(v1) {
		t.Errorf("Mismatch. want: %v, got: %v", 1, MinIndex(v1))
	}

	if math.Abs(-6-MinValue(v1)) > 1.0e-12 {
		t.Errorf("Mismatch. want: %v, got: %v", -6, MinValue(v1))
	}

	if 3 != MaxIndex(v1) {
		t.Errorf("Mismatch. want: %v, got: %v", 3, MaxIndex(v1))
	}

	if math.Abs(12-MaxValue(v1)) > 1.0e-12 {
		t.Errorf("Mismatch. want: %v, got: %v", 12, MaxValue(v1))
	}

	v2 := create(t, []float64{math.NaN(), 3, math.NaN(), -2})

	if 3 != MinIndex(v2) {
		t.Errorf("Mismatch. want: %v, got: %v", 3, MinIndex(v2))
	}

	if math.Abs(-2-MinValue(v2)) > 1.0e-12 {
		t.Errorf("Mismatch. want: %v, got: %v", -2, MinValue(v2))
	}

	if 1 != MaxIndex(v2) {
		t.Errorf("Mismatch. want: %v, got: %v", 1, MaxIndex(v2))
	}

	if math.Abs(3-MaxValue(v2)) > 1.0e-12 {
		t.Errorf("Mismatch. want: %v, got: %v", 3, MaxValue(v2))
	}

	v3 := create(t, []float64{math.NaN(), math.NaN()})

	if -1 != MinIndex(v3) {
		t.Errorf("Mismatch. want: %v, got: %v", -1, MinIndex(v3))
	}

	if !math.IsNaN(MinValue(v3)) {
		t.Errorf("Mismatch. want: %v, got: %v", math.NaN(), MinValue(v3))
	}

	if -1 != MaxIndex(v3) {
		t.Errorf("Mismatch. want: %v, got: %v", -1, MaxIndex(v3))
	}

	if !math.IsNaN(MaxValue(v3)) {
		t.Errorf("Mismatch. want: %v, got: %v", math.NaN(), MaxValue(v3))
	}

	v4 := create(t, make([]float64, 0))
	if -1 != MinIndex(v4) {
		t.Errorf("Mismatch. want: %v, got: %v", -1, MinIndex(v4))
	}

	if !math.IsNaN(MinValue(v4)) {
		t.Errorf("Mismatch. want: %v, got: %v", math.NaN(), MinValue(v4))
	}

	if -1 != MaxIndex(v4) {
		t.Errorf("Mismatch. want: %v, got: %v", -1, MaxIndex(v4))
	}

	if !math.IsNaN(MaxValue(v4)) {
		t.Errorf("Mismatch. want: %v, got: %v", math.NaN(), MaxValue(v4))
	}
}

type testRealVectorPreservingVisitor struct {
	start func(actualSize, actualStart, actualEnd int)
	visit func(actualIndex int, actualValue float64)
	end   func() float64
	t     *testing.T
}

func (trv *testRealVectorPreservingVisitor) Start(actualSize, actualStart, actualEnd int) {
	trv.start(actualSize, actualStart, actualEnd)
}
func (trv *testRealVectorPreservingVisitor) Visit(actualIndex int, actualValue float64) {
	trv.visit(actualIndex, actualValue)
}
func (trv *testRealVectorPreservingVisitor) End() float64 { return trv.end() }

func TestArrayRealVectorWalkInDefaultOrderPreservingVisitor1(t *testing.T) {
	data := []float64{
		0, 1, 0, 0, 2, 0, 0, 0, 3,
	}
	var expectedIndex int
	v := create(t, data)
	visitor := new(testRealVectorPreservingVisitor)
	visitor.start = func(actualSize, actualStart, actualEnd int) {
		if len(data) != actualSize {
			t.Errorf("Mismatch. want: %v, got: %v", len(data), actualSize)
		}
		if 0 != actualStart {
			t.Errorf("Mismatch. want: %v, got: %v", 0, actualStart)
		}
		if len(data)-1 != actualEnd {
			t.Errorf("Mismatch. want: %v, got: %v", len(data)-1, actualEnd)
		}
		expectedIndex = 0
	}

	visitor.visit = func(actualIndex int, actualValue float64) {
		expectedIndex++
	}

	visitor.end = func() float64 {
		return 0.0
	}

	v.WalkInDefaultOrder(visitor)
}

func TestArrayRealVectorWalkInDefaultOrderPreservingVisitor2(t *testing.T) {
	v := create(t, make([]float64, 5))

	visitor := new(testRealVectorPreservingVisitor)

	visitor.start = func(actualSize, actualStart, actualEnd int) {}

	visitor.visit = func(actualIndex int, actualValue float64) {}

	visitor.end = func() float64 {
		return 0.0
	}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	v.WalkInDefaultOrderBounded(visitor, -1, 4)
	v.WalkInDefaultOrderBounded(visitor, 5, 4)
	v.WalkInDefaultOrderBounded(visitor, 0, -1)
	v.WalkInDefaultOrderBounded(visitor, 0, 5)
	v.WalkInDefaultOrderBounded(visitor, 4, 0)

}

func TestArrayRealVectorWalkInDefaultOrderPreservingVisitor3(t *testing.T) {
	data := []float64{
		0, 1, 0, 0, 2, 0, 0, 0, 3,
	}
	expectedStart := 2
	expectedEnd := 7
	v := create(t, data)
	var expectedIndex int
	visitor := new(testRealVectorPreservingVisitor)

	visitor.start = func(actualSize, actualStart, actualEnd int) {
		if len(data) != actualSize {
			t.Errorf("Mismatch. want: %v, got: %v", len(data), actualSize)
		}
		if expectedStart != actualStart {
			t.Errorf("Mismatch. want: %v, got: %v", expectedStart, actualStart)
		}
		if expectedEnd != actualEnd {
			t.Errorf("Mismatch. want: %v, got: %v", expectedEnd, actualEnd)
		}
		expectedIndex = expectedStart
	}

	visitor.visit = func(actualIndex int, actualValue float64) {
		if expectedIndex != actualIndex {
			t.Errorf("Mismatch. want: %v, got: %v", expectedIndex, actualIndex)
		}
		if data[actualIndex] != actualValue {
			t.Errorf("Mismatch. want: %v, got: %v", data[actualIndex], actualValue)
		}
		expectedIndex++
	}

	visitor.end = func() float64 {
		return 0.0
	}

	v.WalkInDefaultOrderBounded(visitor, expectedStart, expectedEnd)
}

func TestArrayRealVectorWalkInOptimizedOrderPreservingVisitor1(t *testing.T) {
	data := []float64{
		0, 1, 0, 0, 2, 0, 0, 0, 3,
	}
	v := create(t, data)
	visited := make([]bool, len(data))
	visitor := new(testRealVectorPreservingVisitor)
	visitor.start = func(actualSize, actualStart, actualEnd int) {
		if len(data) != actualSize {
			t.Errorf("Mismatch. want: %v, got: %v", len(data), actualSize)
		}
		if 0 != actualStart {
			t.Errorf("Mismatch. want: %v, got: %v", 0, actualStart)
		}
		if len(data)-1 != actualEnd {
			t.Errorf("Mismatch. want: %v, got: %v", len(data)-1, actualEnd)
		}
		visited = make([]bool, len(data))
	}

	visitor.visit = func(actualIndex int, actualValue float64) {
		if data[actualIndex] != actualValue {
			t.Errorf("Mismatch. want: %v, got: %v", data[actualIndex], actualValue)
		}
		visited[actualIndex] = true
	}

	visitor.end = func() float64 {
		for i := 0; i < len(data); i++ {
			if !visited[i] {
				t.Errorf("Mismatch. index %d is already visited.", i)
			}
		}

		return 0.0
	}

	v.WalkInOptimizedOrder(visitor)
}

func TestArrayRealVectorWalkInOptimizedOrderPreservingVisitor2(t *testing.T) {
	v := create(t, make([]float64, 5))
	visitor := new(testRealVectorPreservingVisitor)
	visitor.start = func(actualSize, actualStart, actualEnd int) {}

	visitor.visit = func(actualIndex int, actualValue float64) {}

	visitor.end = func() float64 {
		return 0.0
	}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()
	v.WalkInOptimizedOrderBounded(visitor, -1, 4)
	v.WalkInOptimizedOrderBounded(visitor, 5, 4)
	v.WalkInOptimizedOrderBounded(visitor, 0, -1)
	v.WalkInOptimizedOrderBounded(visitor, 0, 5)
	v.WalkInOptimizedOrderBounded(visitor, 4, 0)
}

func TestArrayRealVectorWalkInOptimizedOrderPreservingVisitor3(t *testing.T) {
	data := []float64{
		0, 1, 0, 0, 2, 0, 0, 0, 3,
	}
	v := create(t, data)

	expectedStart := 2
	expectedEnd := 7
	visited := make([]bool, len(data))
	visitor := new(testRealVectorPreservingVisitor)

	visitor.start = func(actualSize, actualStart, actualEnd int) {
		if len(data) != actualSize {
			t.Errorf("Mismatch. want: %v, got: %v", len(data), actualSize)
		}
		if expectedStart != actualStart {
			t.Errorf("Mismatch. want: %v, got: %v", expectedStart, actualStart)
		}
		if expectedEnd != actualEnd {
			t.Errorf("Mismatch. want: %v, got: %v", expectedEnd, actualEnd)
		}
		for i := 0; i < len(visited); i++ {
			visited[i] = true
		}
	}

	visitor.visit = func(actualIndex int, actualValue float64) {
		if data[actualIndex] != actualValue {
			t.Errorf("Mismatch. want: %v, got: %v", data[actualIndex], actualValue)
		}
		visited[actualIndex] = true
	}

	visitor.end = func() float64 {
		for i := expectedStart; i <= expectedEnd; i++ {
			if !visited[i] {
				t.Errorf("Mismatch. index %d is already visited.", i)
			}
		}
		return 0.0

	}

	v.WalkInOptimizedOrderBounded(visitor, expectedStart, expectedEnd)
}

type testRealVectorChangingVisitor struct {
	start func(actualSize, actualStart, actualEnd int)
	visit func(actualIndex int, actualValue float64) float64
	end   func() float64
	t     *testing.T
}

func (trv *testRealVectorChangingVisitor) Start(actualSize, actualStart, actualEnd int) {
	trv.start(actualSize, actualStart, actualEnd)
}
func (trv *testRealVectorChangingVisitor) Visit(actualIndex int, actualValue float64) float64 {
	return trv.visit(actualIndex, actualValue)
}
func (trv *testRealVectorChangingVisitor) End() float64 { return trv.end() }

func TestArrayRealVectorWalkInDefaultOrderChangingVisitor1(t *testing.T) {
	data := []float64{
		0, 1, 0, 0, 2, 0, 0, 0, 3,
	}
	v := create(t, data)
	visitor := new(testRealVectorChangingVisitor)
	var expectedIndex int
	visitor.start = func(actualSize, actualStart, actualEnd int) {
		if len(data) != actualSize {
			t.Errorf("Mismatch. want: %v, got: %v", len(data), actualSize)
		}

		if 0 != actualStart {
			t.Errorf("Mismatch. want: %v, got: %v", 0, actualStart)
		}

		if len(data)-1 != actualEnd {
			t.Errorf("Mismatch. want: %v, got: %v", len(data)-1, actualEnd)
		}

		expectedIndex = 0
	}

	visitor.visit = func(actualIndex int, actualValue float64) float64 {
		if expectedIndex != actualIndex {
			t.Errorf("Mismatch. want: %v, got: %v", expectedIndex, actualIndex)
		}
		if data[actualIndex] != actualValue {
			t.Errorf("Mismatch. want: %v, got: %v", data[actualIndex], actualValue)
		}

		expectedIndex++
		return float64(actualIndex) + actualValue
	}

	visitor.end = func() float64 {
		return 0.0
	}

	v.WalkInUpdateDefaultOrder(visitor)
	for i := 0; i < len(data); i++ {
		if float64(i)+data[i] != v.At(i) {
			t.Errorf("Mismatch. want: %v, got: %v", float64(i)+data[i], v.At(i))
		}
	}
}

func TestArrayRealVectorWalkInDefaultOrderChangingVisitor2(t *testing.T) {
	v := create(t, make([]float64, 5))
	visitor := new(testRealVectorChangingVisitor)
	visitor.start = func(actualSize, actualStart, actualEnd int) {}
	visitor.visit = func(actualIndex int, actualValue float64) float64 { return 0.0 }

	visitor.end = func() float64 { return 0.0 }

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	v.WalkInUpdateDefaultOrderBounded(visitor, -1, 4)
	v.WalkInUpdateDefaultOrderBounded(visitor, 5, 4)
	v.WalkInUpdateDefaultOrderBounded(visitor, 0, -1)
	v.WalkInUpdateDefaultOrderBounded(visitor, 0, 5)
	v.WalkInUpdateDefaultOrderBounded(visitor, 4, 0)

}

func TestArrayRealVectorWalkInDefaultOrderChangingVisitor3(t *testing.T) {
	data := []float64{
		0, 1, 0, 0, 2, 0, 0, 0, 3,
	}
	v := create(t, data)
	expectedStart := 2
	expectedEnd := 7
	var expectedIndex int
	visitor := new(testRealVectorChangingVisitor)
	visitor.visit = func(actualIndex int, actualValue float64) float64 {
		if expectedIndex != actualIndex {
			t.Errorf("Mismatch. want: %v, got: %v", expectedIndex, actualIndex)
		}
		if data[actualIndex] != actualValue {
			t.Errorf("Mismatch. want: %v, got: %v", data[actualIndex], actualValue)
		}

		expectedIndex++
		return float64(actualIndex) + actualValue
	}

	visitor.start = func(actualSize, actualStart, actualEnd int) {
		if len(data) != actualSize {
			t.Errorf("Mismatch. want: %v, got: %v", len(data), actualSize)
		}
		if expectedStart != actualStart {
			t.Errorf("Mismatch. want: %v, got: %v", expectedStart, actualStart)
		}
		if expectedEnd != actualEnd {
			t.Errorf("Mismatch. want: %v, got: %v", expectedEnd, actualEnd)
		}
		expectedIndex = expectedStart
	}

	visitor.end = func() float64 {
		return 0.0
	}

	v.WalkInUpdateDefaultOrderBounded(visitor, expectedStart, expectedEnd)
	for i := expectedStart; i <= expectedEnd; i++ {
		if float64(i)+data[i] != v.At(i) {
			t.Errorf("Mismatch. want: %v, got: %v", float64(i)+data[i], v.At(i))
		}
	}

}

func TestArrayRealVectorWalkInOptimizedOrderChangingVisitor1(t *testing.T) {
	data := []float64{
		0, 1, 0, 0, 2, 0, 0, 0, 3,
	}
	v := create(t, data)
	visited := make([]bool, len(data))
	visitor := new(testRealVectorChangingVisitor)
	visitor.start = func(actualSize, actualStart, actualEnd int) {
		if len(data) != actualSize {
			t.Errorf("Mismatch. want: %v, got: %v", len(data), actualSize)
		}
		if 0 != actualStart {
			t.Errorf("Mismatch. want: %v, got: %v", 0, actualStart)
		}
		if len(data)-1 != actualEnd {
			t.Errorf("Mismatch. want: %v, got: %v", len(data)-1, actualEnd)
		}
		visited = make([]bool, len(data))
	}

	visitor.visit = func(actualIndex int, actualValue float64) float64 {
		visited[actualIndex] = true
		if data[actualIndex] != actualValue {
			t.Errorf("Mismatch. want: %v, got: %v", data[actualIndex], actualValue)
		}

		return float64(actualIndex) + actualValue
	}

	visitor.end = func() float64 {
		for i := 0; i < len(data); i++ {
			if !visited[i] {
				t.Errorf("Mismatch. index %d is already visited.", i)
			}
		}
		return 0.0
	}

	v.WalkInUpdateOptimizedOrder(visitor)
	for i := 0; i < len(data); i++ {
		if float64(i)+data[i] != v.At(i) {
			t.Errorf("Mismatch. want: %v, got: %v", float64(i)+data[i], v.At(i))
		}

	}
}

func TestArrayRealVectorWalkInOptimizedOrderChangingVisitor2(t *testing.T) {
	v := create(t, make([]float64, 5))
	visitor := new(testRealVectorChangingVisitor)

	visitor.start = func(dimension, start, end int) {}
	visitor.visit = func(index int, value float64) float64 {
		return 0.0
	}
	visitor.end = func() float64 { return 0.0 }

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("panic expected.")
		}
	}()

	v.WalkInUpdateOptimizedOrderBounded(visitor, -1, 4)
	v.WalkInUpdateOptimizedOrderBounded(visitor, 5, 4)
	v.WalkInUpdateOptimizedOrderBounded(visitor, 0, -1)
	v.WalkInUpdateOptimizedOrderBounded(visitor, 0, 5)
	v.WalkInUpdateOptimizedOrderBounded(visitor, 4, 0)
}

func TestArrayRealVectorWalkInOptimizedOrderChangingVisitor3(t *testing.T) {
	data := []float64{
		0, 1, 0, 0, 2, 0, 0, 0, 3,
	}
	v := create(t, data)
	expectedStart := 2
	expectedEnd := 7

	visited := make([]bool, len(data))
	visitor := new(testRealVectorChangingVisitor)
	visitor.start = func(actualSize, actualStart, actualEnd int) {
		if len(data) != actualSize {
			t.Errorf("Mismatch. want: %v, got: %v", len(data), actualSize)
		}
		if expectedStart != actualStart {
			t.Errorf("Mismatch. want: %v, got: %v", expectedStart, actualStart)
		}
		if expectedEnd != actualEnd {
			t.Errorf("Mismatch. want: %v, got: %v", expectedEnd, actualEnd)
		}

		for i := 0; i < len(visited); i++ {
			visited[i] = true
		}
	}

	visitor.visit = func(actualIndex int, actualValue float64) float64 {
		if data[actualIndex] != actualValue {
			t.Errorf("Mismatch. want: %v, got: %v", data[actualIndex], actualValue)
		}

		visited[actualIndex] = true
		return float64(actualIndex) + actualValue
	}

	visitor.end = func() float64 {
		for i := expectedStart; i <= expectedEnd; i++ {
			if !visited[i] {
				t.Errorf("Mismatch. index %d is already visited.", i)
			}
		}
		return 0.0
	}

	v.WalkInUpdateOptimizedOrderBounded(visitor, expectedStart, expectedEnd)
	for i := expectedStart; i <= expectedEnd; i++ {
		if float64(i)+data[i] != v.At(i) {
			t.Errorf("Mismatch. want: %v, got: %v", float64(i)+data[i], v.At(i))
		}
	}
}

func doTestMapFunction(t *testing.T, f func(x float64) float64) {
	data := make([]float64, len(testValues)+6)
	copy(data[:len(testValues)], testValues[:len(testValues)])

	data[len(testValues)+0] = 0.5 * math.Pi
	data[len(testValues)+1] = -0.5 * math.Pi
	data[len(testValues)+2] = math.E
	data[len(testValues)+3] = -math.E
	data[len(testValues)+4] = 1.0
	data[len(testValues)+5] = -1.0
	expected := make([]float64, len(data))
	for i := 0; i < len(data); i++ {
		expected[i] = f(data[i])
	}
	actual := create(t, data)
	actual.Map(f)

	for j := 0; j < len(expected); j++ {
		if math.Abs(expected[j]-actual.At(j)) > 1e-16 {
			t.Errorf("Mismatch. want: %v, got: %v", expected[j], actual.At(j))
		}
	}
}

func createFunctions() []func(float64) float64 {
	return []func(float64) float64{
		math.Exp, math.Expm1, math.Log, math.Log10,
		math.Log1p, math.Cosh, math.Sinh, math.Tanh, math.Cos,
		math.Sin, math.Tan, math.Acos, math.Asin, math.Atan,
		math.Abs, math.Sqrt, math.Cbrt, math.Ceil,
		math.Floor, math.Round, math.Gamma, math.Erf, math.Erfc,
	}
}

func doTestAppendScalar(t *testing.T, v RealVector, d, delta float64) {
	n := v.Dimension()
	w := v.Append(d)
	if n+1 != w.Dimension() {
		t.Errorf("Mismatch. want: %v, got: %v", n+1, w.Dimension())
	}

	for i := 0; i < n; i++ {
		if math.Abs(v.At(i)-w.At(i)) > delta {
			t.Errorf("Mismatch. want: %v, got: %v", v.At(i), w.At(i))
		}
	}
	if math.Abs(d-w.At(n)) > delta {
		t.Errorf("Mismatch. want: %v, got: %v", d, w.At(n))
	}
}

func doTestAppendVector(t *testing.T, v1, v2 RealVector, delta float64) {
	n1 := v1.Dimension()
	n2 := v2.Dimension()
	v := v1.AppendVector(v2)

	if n1+n2 != v.Dimension() {
		t.Errorf("Mismatch. want: %v, got: %v", n1+n2, v.Dimension())
	}
	for i := 0; i < n1; i++ {
		if math.Abs(v1.At(i)-v.At(i)) > delta {
			t.Errorf("Mismatch. want: %v, got: %v", v1.At(i), v.At(i))
		}
	}
	for i := 0; i < n2; i++ {
		if math.Abs(v2.At(i)-v.At(n1+i)) > delta {
			t.Errorf("Mismatch. want: %v, got: %v", v2.At(i), v.At(n1+i))
		}
	}
}
