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
 * Class defining a real-valued vector with basic algebraic operations.
 *
 * vector element indexing is 0-based -- e.g., At(0) returns the first element of the vector.
 *
 * The map method operate on vectors element-wise, i.e. they perform the same operation (adding a scalar,
 * applying a function ...) on each element in turn. It uses the instance itself to store the
 * results, so the instance is changed by this method. In all cases, the result
 * vector is returned by the methods.
 *
 */
type RealVector interface {
	/**
	 * Returns the size of the vector.
	 *
	 * @return the size of this vector.
	 */
	Dimension() int
	/**
	 * Return the entry at the specified index.
	 */
	At(index int) float64
	/**
	 * Set a single element.
	 */
	SetEntry(index int, value float64)
	/**
	 * Change an entry at the specified index.
	 */
	AddToEntry(index int, increment float64)
	/**
	 * Construct a new vector by appending a vector to this vector.
	 */
	AppendVector(v RealVector) RealVector
	/**
	 * Construct a new vector by appending a double to this vector.
	 */
	Append(d float64) RealVector
	/**
	 * Get a subvector from consecutive elements.
	 */
	SubVector(index, n int) RealVector
	/**
	 * Set a sequence of consecutive elements.
	 */
	SetSubVector(index int, v RealVector)
	/**
	 * Check whether any coordinate of this vector is NaN.
	 */
	IsNaN() bool
	/**
	 * Check whether any coordinate of this vector is infinite and none are NaN.
	 */
	IsInf() bool
	/**
	 * Compute the sum of this vector and v.
	 * Returns a new vector. Does not change instance data.
	 */
	Add(v RealVector) RealVector
	/**
	 * Subtract v from this vector.
	 * Returns a new vector. Does not change instance data.
	 */
	Subtract(v RealVector) RealVector
	/**
	 * Add a value to each entry.
	 * Returns a new vector. Does not change instance data.
	 */
	MapAdd(d float64)
	/**
	 * Element-by-element division.
	 */
	EBEDivide(v RealVector) RealVector
	/**
	 * Element-by-element multiplication.
	 */
	EBEMultiply(v RealVector) RealVector
	/**
	 * Subtract a value from each entry. Returns a new vector.
	 * Does not change instance data.
	 */
	MapSubtract(d float64)
	/**
	 * Multiply each entry by the argument. Returns a new vector.
	 * Does not change instance data.
	 */
	MapMultiply(d float64)
	/**
	 * Divide each entry by the argument. Returns a new vector.
	 * Does not change instance data.
	 */
	MapDivide(d float64)
	/**
	 * Copies entries from a vector with same size as this instance.
	 */
	CopyFrom(vec RealVector)
	/**
	 * Returns a (deep) copy of this vector.
	 */
	Copy() RealVector

	/**
	 * Set all elements to a single value.
	 */
	Set(value float64)
	/**
	 * Convert the vector to an array of {@code double}s.
	 * The array is independent from this vector data: the elements
	 * are copied.
	 */
	ToArray() []float64
	/**
	 * Converts this vector into a unit vector.
	 * The instance itself is changed by this method.
	 */
	Unitize()

	/**
	 * Generic dense iterator. Iteration is in increasing order
	 * of the vector index.
	 *
	 * Note: derived classes are required to return an Iterator that
	 * returns non-nil Entry objects as long as hasNext() returns true.
	 */
	Iterator() EntryIterator
	/**
	 * Entries of this vector are modified in-place.
	 */
	Map(f func(float64) float64)

	/**
	 * Updates this vector with the linear combination of this and y.
	 */
	Combine(a, b float64, y RealVector)
	/**
	 * Visits (but does not alter) all entries of this vector in default order
	 * (increasing index).
	 */
	WalkInDefaultOrder(visitor RealVectorPreservingVisitor) float64
	/**
	 * Visits (but does not alter) some entries of this vector in default order
	 * (increasing index).
	 */
	WalkInDefaultOrderBounded(visitor RealVectorPreservingVisitor, start, end int) float64
	/**
	 * Visits (and possibly alters) all entries of this vector in default order
	 * (increasing index).
	 */
	WalkInUpdateDefaultOrder(visitor RealVectorChangingVisitor) float64
	/**
	 * Visits (and possibly alters) some entries of this vector in default order
	 * (increasing index).
	 */
	WalkInUpdateDefaultOrderBounded(visitor RealVectorChangingVisitor, start, end int) float64
	/**
	 * Visits (but does not alter) all entries of this vector in optimized
	 * order. The order in which the entries are visited is selected so as to
	 * lead to the most efficient implementation; it might depend on the
	 * concrete implementation of this abstract class.
	 */
	WalkInOptimizedOrder(visitor RealVectorPreservingVisitor) float64
	/**
	 * Visits (but does not alter) some entries of this vector in optimized
	 * order. The order in which the entries are visited is selected so as to
	 * lead to the most efficient implementation; it might depend on the
	 * concrete implementation of this abstract class.
	 */
	WalkInOptimizedOrderBounded(visitor RealVectorPreservingVisitor, start, end int) float64
	/**
	 * Visits (and possibly alters) all entries of this vector in optimized
	 * order. The order in which the entries are visited is selected so as to
	 * lead to the most efficient implementation; it might depend on the
	 * concrete implementation of this abstract class.
	 */
	WalkInUpdateOptimizedOrder(visitor RealVectorChangingVisitor) float64
	/**
	 * Visits (and possibly change) some entries of this vector in optimized
	 * order. The order in which the entries are visited is selected so as to
	 * lead to the most efficient implementation; it might depend on the
	 * concrete implementation of this abstract class.
	 */
	WalkInUpdateOptimizedOrderBounded(visitor RealVectorChangingVisitor, start, end int) float64
	/**
	 * Test for the equality of two real vectors. If all coordinates of two real
	 * vectors are exactly the same, and none are NaN, the two real
	 * vectors are considered to be equal. NaN coordinates are
	 * considered to affect globally the vector and be equals to each other -
	 * i.e, if either (or all) coordinates of the real vector are equal to
	 * NaN, the real vector is equal to a vector with all NaN
	 * coordinates.
	 */
	Equals(other interface{}) bool
}

type EntryIterator interface {
	HasNext() bool
	Next() Entry
}

type Entry interface {
	Index() int
	Value() float64
}

type entryImpl struct {
	idx   int
	value float64
}

func (e *entryImpl) Index() int {
	return e.idx
}

func (e *entryImpl) Value() float64 {
	return e.value
}

type entryIteratorImpl struct {
	vec RealVector
	idx int
}

func newEntryIterator(vec RealVector) *entryIteratorImpl {
	return &entryIteratorImpl{vec: vec}
}

func (ei *entryIteratorImpl) HasNext() bool {
	return ei.idx < ei.vec.Dimension()
}

func (ei *entryIteratorImpl) Next() Entry {
	if ei.idx >= ei.vec.Dimension() {
		panic("no entry left")
	}

	e := &entryImpl{idx: ei.idx, value: ei.vec.At(ei.idx)}
	ei.idx++
	return e

}

func NewRealVector(data []float64) (RealVector, error) {
	if data == nil {
		return nil, invalidArgumentSimpleErrorf()
	}

	return NewArrayRealVector(data, true)
}

/**
 * Returns the L2 norm of the vector. The root of the sum of
 * the squared elements.
 */
func VecNorm(v RealVector) float64 {
	var sum float64
	it := v.Iterator()
	for it.HasNext() {
		e := it.Next()
		value := e.Value()
		sum += value * value
	}
	return math.Sqrt(sum)
}

/**
 * Computes the cosine of the angle between this vector and the
 * argument.
 */
func VecCosine(v1, v2 RealVector) float64 {
	norm := VecNorm(v1)
	vNorm := VecNorm(v2)

	if norm == 0 || vNorm == 0 {
		panic(mathArithmeticErrorf(zero_norm))
	}

	return VecDotProduct(v1, v2) / (norm * vNorm)
}

/**
 * Compute the dot product of v1 with v2.
 */
func VecDotProduct(vec1, vec2 RealVector) float64 {
	err := checkVectorDimensions(vec1, vec2)
	if err != nil {
		panic(err)
	}

	var d float64
	n := vec1.Dimension()
	for i := 0; i < n; i++ {
		d += vec1.At(i) * vec2.At(i)
	}
	return d
}

/**
 * Returns the L1 norm of the vector. The sum of the absolute
 * values of the elements.
 */
func VecL1Norm(v RealVector) float64 {
	var norm float64
	it := v.Iterator()
	for it.HasNext() {
		e := it.Next()
		norm += math.Abs(e.Value())
	}
	return norm
}

/**
 * Returns the L-inf norm of the vector.The max of the absolute
 * values of the elements.
 */
func VecLInfNorm(v RealVector) float64 {
	var norm float64
	it := v.Iterator()
	for it.HasNext() {
		e := it.Next()
		norm = math.Max(norm, math.Abs(e.Value()))
	}
	return norm
}

/**
 * Distance between two vectors.
 * This method computes the distance consistent with the
 * L2 norm, i.e. the square root of the sum of
 * element differences, or Euclidean distance.
 */
func VecDistance(v1, v2 RealVector) float64 {
	err := checkVectorDimensions(v1, v2)
	if err != nil {
		panic(err)
	}

	var d float64
	it := v1.Iterator()
	for it.HasNext() {
		e := it.Next()
		diff := e.Value() - v2.At(e.Index())
		d += diff * diff
	}
	return math.Sqrt(d)

}

/**
 * Distance between two vectors.
 * This method computes the distance consistent with
 * L1 norm, i.e. the sum of the absolute values of
 * the elements differences.
 */
func VecL1Distance(v1, v2 RealVector) float64 {
	err := checkVectorDimensions(v1, v2)
	if err != nil {
		panic(err)
	}

	var d float64
	it := v1.Iterator()
	for it.HasNext() {
		e := it.Next()
		d += math.Abs(e.Value() - v2.At(e.Index()))
	}
	return d
}

/**
 * Distance between two vectors.
 * This method computes the distance consistent with
 * L-inf norm, i.e. the max of the absolute values of
 * element differences.
 */
func VecLInfDistance(v1, v2 RealVector) float64 {
	err := checkVectorDimensions(v1, v2)
	if err != nil {
		panic(err)
	}

	var d float64
	it := v1.Iterator()
	for it.HasNext() {
		e := it.Next()
		d = math.Max(math.Abs(e.Value()-v2.At(e.Index())), d)
	}
	return d
}

/**
 * Get the index of the minimum entry.
 */
func MinIndex(v RealVector) int {
	minIndex := -1
	minValue := math.Inf(1)
	iterator := v.Iterator()
	for iterator.HasNext() {
		entry := iterator.Next()
		if entry.Value() <= minValue {
			minIndex = entry.Index()
			minValue = entry.Value()
		}
	}
	return minIndex
}

/**
 * Get the value of the minimum entry.
 */
func MinValue(v RealVector) float64 {
	minIndex := MinIndex(v)
	if minIndex < 0 {
		return math.NaN()
	}

	return v.At(minIndex)

}

/**
 * Get the index of the maximum entry.
 */
func MaxIndex(v RealVector) int {
	maxIndex := -1
	maxValue := math.Inf(-1)
	iterator := v.Iterator()
	for iterator.HasNext() {
		entry := iterator.Next()
		if entry.Value() >= maxValue {
			maxIndex = entry.Index()
			maxValue = entry.Value()
		}
	}
	return maxIndex
}

/**
 * Get the value of the maximum entry.
 */
func MaxValue(v RealVector) float64 {
	maxIndex := MaxIndex(v)
	if maxIndex < 0 {
		return math.NaN()
	}
	return v.At(maxIndex)
}

/**
 * Find the orthogonal projection of this vector onto another vector.
 */
func Projection(src, dest RealVector) RealVector {
	norm2 := VecDotProduct(dest, dest)
	if norm2 == 0.0 {
		panic(mathArithmeticErrorf(zero_norm))
	}

	dest.MapMultiply(VecDotProduct(src, dest) / VecDotProduct(dest, dest))
	return dest
}

/**
 * Creates a unit vector pointing in the direction of this vector.
 * The instance is not changed by this method.
 */
func UnitVector(v RealVector) RealVector {
	a, err := NewArrayRealVectorCopy(v)
	if err != nil {
		panic(err)
	}

	a.Unitize()

	return a
}
