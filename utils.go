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
	"fmt"
	"math"
)

const (
	sgn_mask_float = 0x80000000
)

var (
	positive_zero_float64      float64 = 0.0
	negative_zero_float64      float64 = -positive_zero_float64
	positive_zero_float64_bits         = math.Float64bits(positive_zero_float64)
	negative_zero_float64_bits         = math.Float64bits(negative_zero_float64)
)

func _isSymmetric(matrix RealMatrix, relativeTolerance float64, raiseException bool) bool {
	rows := matrix.RowDimension()
	if rows != matrix.ColumnDimension() {
		if raiseException {
			panic(fmt.Sprintf("non square (%dx%d) matrix", rows, matrix.ColumnDimension()))
		} else {
			return false
		}
	}
	for i := 0; i < rows; i++ {
		for j := i + 1; j < rows; j++ {
			mij := matrix.At(i, j)
			mji := matrix.At(j, i)
			if math.Abs(mij-mji) >
				math.Max(math.Abs(mij), math.Abs(mji))*relativeTolerance {
				if raiseException {
					panic(fmt.Sprintf("non symmetric matrix: the difference between entries at (%v,%v) and (%v,%v) is larger than %v", i, j, j, i, relativeTolerance))
				} else {
					return false
				}
			}
		}
	}
	return true
}

func isSymmetric(matrix RealMatrix, eps float64) bool {
	return _isSymmetric(matrix, eps, false)
}

func compareTo(x, y, eps float64) int {
	if equalsWithError(x, y, eps) {
		return 0
	} else if x < y {
		return -1
	}
	return 1
}

func equals(x, y float64) bool {
	return equalsWithULP(x, y, 1)
}

func equalsWithError(x, y, eps float64) bool {
	return equalsWithULP(x, y, 1) || math.Abs(y-x) <= eps
}

func equalsWithULP(x, y float64, maxUlps int) bool {

	xInt := math.Float64bits(x)
	yInt := math.Float64bits(y)

	var isEqual bool
	if ((xInt ^ yInt) & sgn_mask_float) == 0 {
		// number have same sign, there is no risk of overflow
		isEqual = uint64(math.Abs(float64(xInt-yInt))) <= uint64(maxUlps)
	} else {
		// number have opposite signs, take care of overflow
		var deltaPlus, deltaMinus uint64
		if xInt < yInt {
			deltaPlus = yInt - positive_zero_float64_bits
			deltaMinus = xInt - negative_zero_float64_bits
		} else {
			deltaPlus = xInt - positive_zero_float64_bits
			deltaMinus = yInt - negative_zero_float64_bits
		}

		if deltaPlus > uint64(maxUlps) {
			isEqual = false
		} else {
			isEqual = deltaMinus <= (uint64(maxUlps) - deltaPlus)
		}

	}

	return isEqual && !math.IsNaN(x) && !math.IsNaN(y)

}
