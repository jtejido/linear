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

type DecompositionSolver interface {
	/**
	 * Solve the linear equation A &times; X = B for matrices A.
	 *
	 * The A matrix is implicit, it is provided by the underlying
	 * decomposition algorithm.
	 */
	SolveVector(b RealVector) RealVector

	/**
	 * Solve the linear equation A &times; X = B for matrices A.
	 *
	 * The A matrix is implicit, it is provided by the underlying
	 * decomposition algorithm.
	 */
	SolveMatrix(b RealMatrix) RealMatrix

	/**
	 * Check if the decomposed matrix is non-singular.
	 */
	IsNonSingular() bool

	/**
	 * Get the pseudo-inverse of the decomposed matrix.
	 *
	 * This is equal to the inverse  of the decomposed matrix, if such an inverse exists.
	 *
	 * If no such inverse exists, then the result has properties that resemble that of an inverse.
	 *
	 * In particular, in this case, if the decomposed matrix is A, then the system of equations
	 * \( A x = b \) may have no solutions, or many. If it has no solutions, then the pseudo-inverse
	 * \( A^+ \) gives the "closest" solution \( z = A^+ b \), meaning \( \left \| A z - b \right \|_2 \)
	 * is minimized. If there are many solutions, then \( z = A^+ b \) is the smallest solution,
	 * meaning \( \left \| z \right \|_2 \) is minimized.
	 *
	 * Note however that some decompositions cannot compute a pseudo-inverse for all matrices.
	 * For example, the LUDecomposition is not defined for non-square matrices to begin
	 * with. The QRDecomposition can operate on non-square matrices, but will throw
	 * panic if the decomposed matrix is singular. Refer to the javadoc
	 * of specific decomposition implementations for more details.
	 */
	Inverse() RealMatrix
}
