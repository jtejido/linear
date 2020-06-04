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

type ArrayRealVector struct {
    data []float64
}

/**
 * Construct a vector of zeroes.
 *
 * @param size Size of the vector.
 */
func NewSizedArrayRealVector(size int) (*ArrayRealVector, error) {
    return &ArrayRealVector{data: make([]float64, size)}, nil
}

/**
 * Construct a vector from another vector, using a deep copy.
 *
 * @param v Vector to copy.
 */
func NewArrayRealVectorCopy(v RealVector) (*ArrayRealVector, error) {
    if v == nil {
        return nil, invalidArgumentSimpleErrorf()
    }

    data := make([]float64, v.Dimension())
    for i := 0; i < len(data); i++ {
        data[i] = v.At(i)
    }

    return &ArrayRealVector{data: data}, nil
}

/**
 * Construct a vector from an array, copying the input array.
 *
 * @param d Array.
 */
func NewArrayRealVectorFromSlice(d []float64) (*ArrayRealVector, error) {
    if d == nil {
        return nil, invalidArgumentSimpleErrorf()
    }

    ans := new(ArrayRealVector)
    ans.data = append([]float64{}, d...)
    return ans, nil
}

/**
 * Create a new ArrayRealVector using the input array as the underlying
 * data array.
 * If an array is built specially in order to be embedded in a
 * ArrayRealVector and not used directly, the {@code copyArray} may be
 * set to {@code false}. This will prevent the copying and improve
 * performance as no new array will be built and no data will be copied.
 *
 * @param d Data for the new vector.
 * @param copyArray if {@code true}, the input array will be copied,
 * otherwise it will be referenced.
 * @see #ArrayRealVector(double[])
 */
func NewArrayRealVector(d []float64, copyArray bool) (*ArrayRealVector, error) {
    if copyArray {
        return NewArrayRealVectorFromSlice(d)
    }

    if d == nil {
        return nil, invalidArgumentSimpleErrorf()
    }

    ans := new(ArrayRealVector)
    ans.data = d
    return ans, nil
}

/**
 * Construct a vector with preset values.
 *
 * @param size Size of the vector
 * @param preset All entries will be set with this value.
 */
func NewSizedArrayRealVectorWithPreset(size int, preset float64) (*ArrayRealVector, error) {
    ans := &ArrayRealVector{data: make([]float64, size)}
    for i := 0; i < size; i++ {
        ans.data[i] = preset
    }
    return ans, nil
}

/**
 * Construct a vector by appending one vector to another vector.
 * @param v1 First vector (will be put in front of the new vector).
 * @param v2 Second vector (will be put at back of the new vector).
 */
func NewArrayRealVectorFromTwoArrayRealVector(v1, v2 *ArrayRealVector) (*ArrayRealVector, error) {
    if v1 == nil || v2 == nil {
        return nil, invalidArgumentSimpleErrorf()
    }

    data := make([]float64, len(v1.data)+len(v2.data))
    copy(data[:len(v1.data)], v1.data)
    copy(data[len(v1.data):len(v1.data)+len(v2.data)], v2.data)
    return &ArrayRealVector{data: data}, nil
}

/**
 * Construct a vector by appending one vector to another vector.
 * @param v1 First vector (will be put in front of the new vector).
 * @param v2 Second vector (will be put at back of the new vector).
 */
func NewArrayRealVectorFromTwoRealVector(v1, v2 RealVector) (*ArrayRealVector, error) {
    if v1 == nil || v2 == nil {
        return nil, invalidArgumentSimpleErrorf()
    }

    l1 := v1.Dimension()
    l2 := v2.Dimension()
    data := make([]float64, l1+l2)
    for i := 0; i < l1; i++ {
        data[i] = v1.At(i)
    }

    for i := l1; i < l2; i++ {
        data[i] = v2.At(i)
    }

    return &ArrayRealVector{data: data}, nil
}

func (arv *ArrayRealVector) CopyFrom(vec RealVector) {
    err := checkVectorDimensions(arv, vec)
    if err != nil {
        panic(err)
    }

    dim := vec.Dimension()
    for i := 0; i < dim; i++ {
        arv.data[i] = vec.At(i)
    }

}

func (arv *ArrayRealVector) Copy() RealVector {
    r, err := NewArrayRealVectorCopy(arv)
    if err != nil {
        panic(err)
    }
    return r
}

func (arv *ArrayRealVector) Map(f func(float64) float64) {
    for i := 0; i < len(arv.data); i++ {
        arv.data[i] = f(arv.data[i])
    }
}

func (arv *ArrayRealVector) MapAdd(d float64) {
    for i := 0; i < len(arv.data); i++ {
        arv.data[i] += d
    }

}

func (arv *ArrayRealVector) MapSubtract(d float64) {
    for i := 0; i < len(arv.data); i++ {
        arv.data[i] -= d
    }

}

func (arv *ArrayRealVector) MapMultiply(d float64) {
    for i := 0; i < len(arv.data); i++ {
        arv.data[i] *= d
    }

}

func (arv *ArrayRealVector) MapDivide(d float64) {
    for i := 0; i < len(arv.data); i++ {
        arv.data[i] /= d
    }
}

func (arv *ArrayRealVector) Add(vec RealVector) RealVector {
    if v, ok := vec.(*ArrayRealVector); ok {
        vData := v.data
        dim := len(vData)
        err := checkDimensions(arv, dim)
        if err != nil {
            panic(err)
        }

        ret, _ := NewSizedArrayRealVector(dim)
        resultData := ret.data

        for i := 0; i < dim; i++ {
            resultData[i] = arv.data[i] + vData[i]
        }

        return ret

    }

    err := checkVectorDimensions(arv, vec)
    if err != nil {
        panic(err)
    }

    ret, _ := NewSizedArrayRealVector(vec.Dimension())
    resultData := ret.data

    for i := 0; i < len(arv.data); i++ {
        resultData[i] = arv.data[i] + vec.At(i)
    }

    return ret

}

func (arv *ArrayRealVector) Subtract(vec RealVector) RealVector {
    if v, ok := vec.(*ArrayRealVector); ok {
        vData := v.data
        dim := len(vData)
        err := checkDimensions(arv, dim)
        if err != nil {
            panic(err)
        }

        ret, _ := NewSizedArrayRealVector(dim)
        resultData := ret.data

        for i := 0; i < dim; i++ {
            resultData[i] = arv.data[i] - vData[i]
        }

        return ret

    }

    err := checkVectorDimensions(arv, vec)
    if err != nil {
        panic(err)
    }

    ret, _ := NewSizedArrayRealVector(vec.Dimension())
    resultData := ret.data
    for i := 0; i < len(arv.data); i++ {
        resultData[i] = arv.data[i] - vec.At(i)
    }

    return ret

}

func (arv *ArrayRealVector) EBEMultiply(vec RealVector) RealVector {
    if v, ok := vec.(*ArrayRealVector); ok {
        vData := v.data
        dim := len(vData)
        err := checkDimensions(arv, dim)
        if err != nil {
            panic(err)
        }
        ret, _ := NewSizedArrayRealVector(dim)
        resultData := ret.data

        for i := 0; i < dim; i++ {
            resultData[i] = arv.data[i] * vData[i]
        }

        return ret
    }

    err := checkVectorDimensions(arv, vec)
    if err != nil {
        panic(err)
    }

    ret, _ := NewSizedArrayRealVector(vec.Dimension())
    resultData := ret.data

    for i := 0; i < len(arv.data); i++ {
        resultData[i] = arv.data[i] * vec.At(i)
    }

    return ret
}

func (arv *ArrayRealVector) EBEDivide(vec RealVector) RealVector {
    if v, ok := vec.(*ArrayRealVector); ok {
        vData := v.data
        dim := len(vData)
        err := checkDimensions(arv, dim)
        if err != nil {
            panic(err)
        }

        ret, _ := NewSizedArrayRealVector(dim)
        resultData := ret.data

        for i := 0; i < dim; i++ {
            resultData[i] = arv.data[i] / vData[i]
        }

        return ret
    }

    err := checkVectorDimensions(arv, vec)
    if err != nil {
        panic(err)
    }

    ret, _ := NewSizedArrayRealVector(vec.Dimension())
    resultData := ret.data

    for i := 0; i < len(arv.data); i++ {
        resultData[i] = arv.data[i] / vec.At(i)
    }

    return ret
}

func (arv *ArrayRealVector) DataRef() []float64 {
    return arv.data
}

func (arv *ArrayRealVector) Unitize() {
    norm := VecNorm(arv)
    if norm == 0 {
        panic(mathArithmeticErrorf(zero_norm))
    }

    arv.MapDivide(norm)
}

func (arv *ArrayRealVector) Iterator() EntryIterator {
    return newEntryIterator(arv)
}

func (arv *ArrayRealVector) At(index int) float64 {
    err := checkIndex(arv, index)
    if err != nil {
        panic(err)
    }

    return arv.data[index]
}

func (arv *ArrayRealVector) Dimension() int {
    return len(arv.data)
}

func (arv *ArrayRealVector) AppendVector(vec RealVector) RealVector {
    if v, ok := vec.(*ArrayRealVector); ok {
        r, err := NewArrayRealVectorFromTwoArrayRealVector(arv, v)
        if err != nil {
            panic(err)
        }

        return r
    } else {
        r, err := NewArrayRealVectorFromTwoRealVector(arv, vec)
        if err != nil {
            panic(err)
        }

        return r
    }
}

func (arv *ArrayRealVector) Append(in float64) RealVector {
    out := make([]float64, len(arv.data)+1)
    copy(out[:len(arv.data)], arv.data)
    out[len(arv.data)] = in
    r, err := NewArrayRealVector(out, false)
    if err != nil {
        panic(err)
    }

    return r
}

func (arv *ArrayRealVector) SubVector(index, n int) RealVector {
    if n < 0 {
        panic(notPositiveErrorf(number_of_elements_should_be_positive, float64(n)))
    }

    out, err := NewSizedArrayRealVector(n)
    if err != nil {
        panic(err)
    }

    err = checkIndex(arv, index)
    if err != nil {
        panic(err)
    }

    err = checkIndex(arv, index+n-1)
    if err != nil {
        panic(err)
    }

    copy(out.data[0:n], arv.data[index:])

    return out
}

func (arv *ArrayRealVector) SetEntry(index int, value float64) {
    err := checkIndex(arv, index)
    if err != nil {
        panic(err)
    }

    arv.data[index] = value
}

func (arv *ArrayRealVector) AddToEntry(index int, increment float64) {
    err := checkIndex(arv, index)
    if err != nil {
        panic(err)
    }

    arv.data[index] += increment
}

func (arv *ArrayRealVector) SetSubVector(index int, vec RealVector) {
    err := checkIndex(arv, index)
    if err != nil {
        panic(err)
    }

    err = checkIndex(arv, index+vec.Dimension()-1)
    if err != nil {
        panic(err)
    }

    if v, ok := vec.(*ArrayRealVector); ok {
        arv.SetSubVectorWithSlice(index, v.data)
    }

    for i := index; i < index+vec.Dimension(); i++ {
        arv.data[i] = vec.At(i - index)
    }

}

func (arv *ArrayRealVector) SetSubVectorWithSlice(index int, v []float64) {
    err := checkIndex(arv, index)
    if err != nil {
        panic(err)
    }

    err = checkIndex(arv, index+len(v)-1)
    if err != nil {
        panic(err)
    }

    copy(arv.data[index:index+len(v)], v)
}

func (arv *ArrayRealVector) Set(v float64) {
    for i := 0; i < len(arv.data); i++ {
        arv.data[i] = v
    }
}

func (arv *ArrayRealVector) ToArray() []float64 {
    return append([]float64{}, arv.data...)
}

func (arv *ArrayRealVector) IsNaN() bool {
    for i := 0; i < len(arv.data); i++ {
        if math.IsNaN(arv.data[i]) {
            return true
        }
    }
    return false
}

func (arv *ArrayRealVector) IsInf() bool {
    if arv.IsNaN() {
        return false
    }

    for i := 0; i < len(arv.data); i++ {
        if math.IsInf(arv.data[i], 1) || math.IsInf(arv.data[i], -1) {
            return true
        }
    }

    return false
}

func (arv *ArrayRealVector) Equals(other interface{}) bool {
    if arv == other {
        return true
    }

    if _, ok := other.(RealVector); !ok {
        return false
    }

    rhs := other.(RealVector)
    if len(arv.data) != rhs.Dimension() {
        return false
    }

    if rhs.IsNaN() {
        return arv.IsNaN()
    }

    for i := 0; i < len(arv.data); i++ {
        if arv.data[i] != rhs.At(i) {
            return false
        }
    }
    return true
}

func (arv *ArrayRealVector) Combine(a, b float64, y RealVector) {
    err := checkVectorDimensions(arv, y)
    if err != nil {
        panic(err)
    }

    for i := 0; i < len(arv.data); i++ {
        arv.data[i] = a*arv.data[i] + b*y.At(i)
    }
}

func (arv *ArrayRealVector) WalkInDefaultOrder(visitor RealVectorPreservingVisitor) float64 {
    visitor.Start(len(arv.data), 0, len(arv.data)-1)

    for i := 0; i < len(arv.data); i++ {
        visitor.Visit(i, arv.data[i])
    }

    return visitor.End()
}

func (arv *ArrayRealVector) WalkInDefaultOrderBounded(visitor RealVectorPreservingVisitor, start, end int) float64 {
    err := checkIndices(arv, start, end)
    if err != nil {
        panic(err)
    }

    visitor.Start(len(arv.data), start, end)
    for i := start; i <= end; i++ {
        visitor.Visit(i, arv.data[i])
    }

    return visitor.End()
}

func (arv *ArrayRealVector) WalkInUpdateDefaultOrder(visitor RealVectorChangingVisitor) float64 {
    visitor.Start(len(arv.data), 0, len(arv.data)-1)

    for i := 0; i < len(arv.data); i++ {
        arv.data[i] = visitor.Visit(i, arv.data[i])
    }

    return visitor.End()
}

func (arv *ArrayRealVector) WalkInUpdateDefaultOrderBounded(visitor RealVectorChangingVisitor, start, end int) float64 {
    err := checkIndices(arv, start, end)
    if err != nil {
        panic(err)
    }

    visitor.Start(len(arv.data), start, end)
    for i := start; i <= end; i++ {
        arv.data[i] = visitor.Visit(i, arv.data[i])
    }
    return visitor.End()
}

func (arv *ArrayRealVector) WalkInOptimizedOrder(visitor RealVectorPreservingVisitor) float64 {
    return arv.WalkInDefaultOrder(visitor)
}

func (arv *ArrayRealVector) WalkInOptimizedOrderBounded(visitor RealVectorPreservingVisitor, start, end int) float64 {
    return arv.WalkInDefaultOrderBounded(visitor, start, end)
}

func (arv *ArrayRealVector) WalkInUpdateOptimizedOrder(visitor RealVectorChangingVisitor) float64 {
    return arv.WalkInUpdateDefaultOrder(visitor)
}

func (arv *ArrayRealVector) WalkInUpdateOptimizedOrderBounded(visitor RealVectorChangingVisitor, start, end int) float64 {
    return arv.WalkInUpdateDefaultOrderBounded(visitor, start, end)
}

func checkIndex(v RealVector, idx int) error {
    dim := v.Dimension()
    if idx < 0 || idx >= dim {
        return outOfRangeErrorf(index, float64(idx), 0, float64(dim)-1)
    }

    return nil
}

func checkIndices(v RealVector, start, end int) error {
    dim := v.Dimension()
    if (start < 0) || (start >= dim) {
        return outOfRangeErrorf(index, float64(start), 0, float64(dim-1))
    }
    if (end < 0) || (end >= dim) {
        return outOfRangeErrorf(index, float64(end), 0, float64(dim-1))
    }
    if end < start {
        return numberIsTooSmallErrorf(initial_row_after_final_row, float64(end), float64(start), false)
    }

    return nil
}

func checkVectorDimensions(v1, v2 RealVector) error {
    return checkDimensions(v1, v2.Dimension())
}

func checkDimensions(v RealVector, n int) error {
    if v.Dimension() != n {
        return dimensionsMismatchSimpleErrorf(v.Dimension(), n)
    }

    return nil
}
