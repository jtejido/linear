package linear

import (
	"fmt"
)

type LinearMathError interface {
	Error() string
	Status() int
}

const (
	FAILURE = iota - 1
	SUCCESS // sanity purpose
	ERANGE
	EMAXITER
	ESING
	EUNSUP
	EARITH
	EINV
	ENODAT
	EMATDIM
	EMULTIDIM
	ESYMM
	ESQMAT
	EDIM
	ENOTSPOS
	ENOTPOS
	ENSMALL
	ENBIG
	EILLN
	EILLARG
	EILLSTATE
)

const (
	invalid_input                         = "invalid input"
	arithmetic_exception                  = "arithmetic exception"
	zero_norm                             = "zero norm"
	dimensions_mismatch_simple            = "%v != %v"
	dimensions_mismatch                   = "dimensions mismatch"
	no_data                               = "no data"
	out_of_range_simple                   = "%v out of [%v, %v] range"
	index                                 = "index (%v)"
	number_of_elements_should_be_positive = "number of elements should be positive (%v)"
	initial_row_after_final_row           = "initial row %v after final row %v"
	initial_column_after_final_column     = "initial column %v after final column %v"
	number_too_small                      = "%v is smaller than the minimum (%v)"
	number_too_small_bound_excluded       = "%v is smaller than, or equal to, the minimum (%v)"
	number_too_large                      = "%v is larger than the maximum (%v)"
	number_too_large_bound_excluded       = "%v is larger than, or equal to, the maximum (%v)"
	empty_selected_column_index_array     = "empty selected column index array"
	empty_selected_row_index_array        = "empty selected row index array"
	row_index                             = "row index (%v)"
	column_index                          = "column index ((%v)"
	illegal_state                         = "illegal state"
	first_rows_not_initialized_yet        = "first %v rows are not initialized yet"
	first_columns_not_initialized_yet     = "first %v columns are not initialized yet"
	at_least_one_column                   = "matrix must have at least one column"
	at_least_one_row                      = "matrix must have at least one row"
	dimensions_mismatch_2x2               = "got %vx%v but expected %vx%v"
	non_square_matrix                     = "non square (%vx%v) matrix"
	non_symmetric_matrix                  = "non symmetric matrix: the difference between entries at (%v,%v) and (%v,%v) is larger than %v"
	not_positive_definite_matrix          = "not positive definite matrix"
	array_element                         = "value %v at index %v"
	not_positive_exponent                 = "invalid exponent %v (must be positive)"
	singular_matrix                       = "matrix is singular"
	unsupported_operation                 = "unsupported operation"
	convergence_failed                    = "convergence failed"
	max_count_exceeded                    = "maximal count (%d) exceeded"
	too_large_cutoff_singular_value       = "cutoff singular value is {0}, should be at most {1}"
)

type maxIterationError struct {
	mathIllegalStateError
	max int
}

func (e maxIterationError) Max() int {
	return e.max
}

func (e maxIterationError) Status() int {
	return EMAXITER
}

func maxIterationSimpleErrorf(max int) maxIterationError {
	return maxIterationErrorf(max_count_exceeded, max)
}

func maxIterationErrorf(pattern string, max int, arguments ...interface{}) maxIterationError {
	ret := maxIterationError{max: max}
	ret.mathIllegalStateError = mathIllegalStateErrorf(pattern, arguments...)
	return ret
}

type mathUnsupportedOperationError struct {
	mathError
}

func (e mathUnsupportedOperationError) Status() int {
	return EUNSUP
}

func mathUnsupportedOperationErrorf() mathUnsupportedOperationError {
	ret := mathUnsupportedOperationError{}
	ret.mathError = mathErrorf(unsupported_operation)
	return ret
}

type mathArithmeticError struct {
	mathError
}

func (e mathArithmeticError) Status() int {
	return EARITH
}

func mathArithmeticSimpleErrorf() mathArithmeticError {
	return mathArithmeticErrorf(arithmetic_exception)
}

func mathArithmeticErrorf(pattern string, arguments ...interface{}) mathArithmeticError {
	ret := mathArithmeticError{}
	ret.mathError = mathErrorf(pattern, arguments...)
	return ret
}

type invalidArgumentError struct {
	mathError
}

func (e invalidArgumentError) Status() int {
	return EINV
}

func invalidArgumentSimpleErrorf() invalidArgumentError {
	return invalidArgumentErrorf(invalid_input)
}

func invalidArgumentErrorf(pattern string, arguments ...interface{}) invalidArgumentError {
	ret := invalidArgumentError{}
	ret.mathError = mathErrorf(pattern, arguments...)
	return ret
}

type noDataError struct {
	mathIllegalArgumentError
}

func (e noDataError) Status() int {
	return ENODAT
}

func noDataErrorSimpleErrorf() noDataError {
	return noDataErrorf(no_data)
}

func noDataErrorf(pattern string, arguments ...interface{}) noDataError {
	ret := noDataError{}
	ret.mathIllegalArgumentError = mathIllegalArgumentErrorf(pattern, arguments...)
	return ret
}

type matrixDimensionMismatchError struct {
	multiDimensionMismatchError
}

func (e matrixDimensionMismatchError) Status() int {
	return EMATDIM
}

func (e matrixDimensionMismatchError) WrongRowDimension() int {
	return e.WrongDimension(0)
}

func (e matrixDimensionMismatchError) ExpectedRowDimension() int {
	return e.ExpectedDimension(0)
}

func (e matrixDimensionMismatchError) WrongColumnDimension() int {
	return e.WrongDimension(1)
}

func (e matrixDimensionMismatchError) ExpectedColumnDimension() int {
	return e.ExpectedDimension(1)
}

func matrixDimensionMismatchErrorf(wrongRowDim, wrongColDim, expectedRowDim, expectedColDim int) matrixDimensionMismatchError {
	ret := matrixDimensionMismatchError{}
	ret.multiDimensionMismatchError = multiDimensionMismatchErrorf(dimensions_mismatch_2x2, []int{wrongRowDim, wrongColDim}, []int{expectedRowDim, expectedColDim})
	return ret
}

type multiDimensionMismatchError struct {
	mathIllegalArgumentError
	wrong, expected []int
}

func (e multiDimensionMismatchError) Status() int {
	return EMULTIDIM
}

func (e multiDimensionMismatchError) WrongDimensions() []int {
	return e.wrong
}

func (e multiDimensionMismatchError) ExpectedDimensions() []int {
	return e.expected
}

func (e multiDimensionMismatchError) WrongDimension(index int) int {
	return e.wrong[index]
}

func (e multiDimensionMismatchError) ExpectedDimension(index int) int {
	return e.expected[index]
}

func multiDimensionMismatchSimpleErrorf(wrong, expected []int) multiDimensionMismatchError {
	return multiDimensionMismatchErrorf(dimensions_mismatch, wrong, expected)
}

func multiDimensionMismatchErrorf(pattern string, wrong, expected []int) multiDimensionMismatchError {
	ret := multiDimensionMismatchError{wrong: wrong, expected: expected}
	ret.mathIllegalArgumentError = mathIllegalArgumentErrorf(pattern, wrong, expected)
	return ret
}

type nonSymmetricMatrixError struct {
	mathIllegalArgumentError
	row, column int
	threshold   float64
}

func (e nonSymmetricMatrixError) Status() int {
	return ESYMM
}

func (e nonSymmetricMatrixError) Row() int {
	return e.row
}

func (e nonSymmetricMatrixError) Column() int {
	return e.column
}

func (e nonSymmetricMatrixError) Threshold() float64 {
	return e.threshold
}

func nonSymmetricMatrixSimpleErrorf(row, column int, threshold float64) nonSymmetricMatrixError {
	return nonSymmetricMatrixErrorf(non_symmetric_matrix, row, column, threshold)
}

func nonSymmetricMatrixErrorf(pattern string, row, column int, threshold float64) nonSymmetricMatrixError {
	ret := nonSymmetricMatrixError{row: row, column: column, threshold: threshold}
	ret.mathIllegalArgumentError = mathIllegalArgumentErrorf(pattern, row, column, column, row, threshold)
	return ret
}

type nonSquareMatrixError struct {
	dimensionsMismatchError
}

func (e nonSquareMatrixError) Status() int {
	return ESQMAT
}

func nonSquareMatrixSimpleErrorf(wrong, expected int) nonSquareMatrixError {
	return nonSquareMatrixErrorf(non_square_matrix, wrong, expected)
}

func nonSquareMatrixErrorf(pattern string, wrong, expected int) nonSquareMatrixError {
	ret := nonSquareMatrixError{}
	ret.dimensionsMismatchError = dimensionsMismatchErrorf(pattern, wrong, expected)
	return ret
}

type dimensionsMismatchError struct {
	mathIllegalNumberError
	dimension int
}

func (e dimensionsMismatchError) Status() int {
	return EDIM
}

func dimensionsMismatchSimpleErrorf(wrong, expected int) dimensionsMismatchError {
	return dimensionsMismatchErrorf(dimensions_mismatch_simple, wrong, expected)
}

func dimensionsMismatchErrorf(pattern string, wrong, expected int) dimensionsMismatchError {
	ret := dimensionsMismatchError{dimension: expected}
	ret.mathIllegalNumberError = mathIllegalNumberErrorf(pattern, float64(wrong), expected)
	return ret
}

type notStrictlyPositiveError struct {
	numberIsTooSmallError
}

func (e notStrictlyPositiveError) Status() int {
	return ENOTSPOS
}

func notStrictlyPositiveErrorf(number float64) notStrictlyPositiveError {
	ret := notStrictlyPositiveError{}
	ret.numberIsTooSmallError = numberIsTooSmallBoundedErrorf(number, 0, true)
	return ret
}

type notPositiveError struct {
	numberIsTooSmallError
}

func (e notPositiveError) Status() int {
	return ENOTPOS
}

func notPositiveErrorf(pattern string, number float64) notPositiveError {
	ret := notPositiveError{}
	ret.numberIsTooSmallError = numberIsTooSmallErrorf(pattern, number, 0, true)
	return ret
}

type outOfRangeError struct {
	mathIllegalNumberError
	lo, high float64
}

func (e outOfRangeError) Status() int {
	return ERANGE
}

func (e outOfRangeError) Low() float64 {
	return e.lo
}

func (e outOfRangeError) High() float64 {
	return e.high
}

func outOfRangeSimpleErrorf(wrong, lo, high float64) outOfRangeError {
	return outOfRangeErrorf(out_of_range_simple, wrong, lo, high)
}

func outOfRangeErrorf(pattern string, wrong, lo, high float64) outOfRangeError {
	ret := outOfRangeError{lo: lo, high: high}
	ret.mathIllegalNumberError = mathIllegalNumberErrorf(pattern, wrong, lo, high)
	return ret
}

type nonPositiveDefiniteMatrixError struct {
	numberIsTooSmallError
	index     int
	threshold float64
}

func (e nonPositiveDefiniteMatrixError) Status() int {
	return ENSMALL
}

func (e nonPositiveDefiniteMatrixError) Row() int {
	return e.index
}

func (e nonPositiveDefiniteMatrixError) Column() int {
	return e.index
}

func (e nonPositiveDefiniteMatrixError) Threshold() float64 {
	return e.threshold
}

func nonPositiveDefiniteMatrixErrorf(wrong float64, index int, threshold float64) nonPositiveDefiniteMatrixError {
	ret := nonPositiveDefiniteMatrixError{index: index, threshold: threshold}
	ret.numberIsTooSmallError = numberIsTooSmallBoundedErrorf(wrong, threshold, false)
	ret.addMsg(not_positive_definite_matrix)
	ret.addMsg(fmt.Sprintf(array_element, wrong, index))
	return ret
}

type singularMatrixError struct {
	mathIllegalArgumentError
}

func (e singularMatrixError) Status() int {
	return ESING
}

func singularMatrixSimpleErrorf() singularMatrixError {
	return singularMatrixErrorf(singular_matrix)
}

func singularMatrixErrorf(pattern string, arguments ...interface{}) singularMatrixError {
	ret := singularMatrixError{}
	ret.mathIllegalArgumentError = mathIllegalArgumentErrorf(pattern, arguments...)
	return ret
}

type numberIsTooLargeError struct {
	mathIllegalNumberError
	max            float64
	boundIsAllowed bool
}

func (e numberIsTooLargeError) Status() int {
	return ENBIG
}

func (e numberIsTooLargeError) Max() float64 {
	return e.max
}

func (e numberIsTooLargeError) IsBoundAllowed() bool {
	return e.boundIsAllowed
}

func numberIsTooLargeBoundedErrorf(wrong, max float64, boundIsAllowed bool) numberIsTooLargeError {
	if boundIsAllowed {
		return numberIsTooLargeErrorf(number_too_large, wrong, max, boundIsAllowed)
	}

	return numberIsTooLargeErrorf(number_too_large_bound_excluded, wrong, max, boundIsAllowed)
}

func numberIsTooLargeErrorf(pattern string, wrong, max float64, boundIsAllowed bool) numberIsTooLargeError {
	ret := numberIsTooLargeError{max: max, boundIsAllowed: boundIsAllowed}
	ret.mathIllegalNumberError = mathIllegalNumberErrorf(pattern, wrong, max)
	return ret
}

type numberIsTooSmallError struct {
	mathIllegalNumberError
	min            float64
	boundIsAllowed bool
}

func (e numberIsTooSmallError) Status() int {
	return ENSMALL
}

func (e numberIsTooSmallError) Min() float64 {
	return e.min
}

func (e numberIsTooSmallError) IsBoundAllowed() bool {
	return e.boundIsAllowed
}

func numberIsTooSmallBoundedErrorf(wrong, min float64, boundIsAllowed bool) numberIsTooSmallError {
	if boundIsAllowed {
		return numberIsTooSmallErrorf(number_too_small, wrong, min, boundIsAllowed)
	}

	return numberIsTooSmallErrorf(number_too_small_bound_excluded, wrong, min, boundIsAllowed)
}

func numberIsTooSmallErrorf(pattern string, wrong, min float64, boundIsAllowed bool) numberIsTooSmallError {
	ret := numberIsTooSmallError{min: min, boundIsAllowed: boundIsAllowed}
	ret.mathIllegalNumberError = mathIllegalNumberErrorf(pattern, wrong, min)
	return ret
}

type mathIllegalNumberError struct {
	mathIllegalArgumentError
	argument float64
}

func (e mathIllegalNumberError) Status() int {
	return EILLN
}

func (e mathIllegalNumberError) Argument() float64 {
	return e.argument
}

func mathIllegalNumberErrorf(pattern string, wrong float64, arguments ...interface{}) mathIllegalNumberError {
	ret := mathIllegalNumberError{argument: wrong}
	ret.mathIllegalArgumentError = mathIllegalArgumentErrorf(pattern, wrong, arguments)
	return ret
}

type mathIllegalArgumentError struct {
	mathError
}

func (e mathIllegalArgumentError) Status() int {
	return EILLARG
}

func mathIllegalArgumentErrorf(pattern string, arguments ...interface{}) mathIllegalArgumentError {
	ret := mathIllegalArgumentError{}
	ret.mathError = mathErrorf(pattern, arguments...)
	return ret
}

type mathIllegalStateError struct {
	mathError
}

func (e mathIllegalStateError) Status() int {
	return EILLSTATE
}

func mathIllegalStateSimpleErrorf(pattern string, arguments ...interface{}) mathIllegalStateError {
	return mathIllegalStateErrorf(illegal_state)
}

func mathIllegalStateErrorf(pattern string, arguments ...interface{}) mathIllegalStateError {
	ret := mathIllegalStateError{}
	ret.mathError = mathErrorf(pattern, arguments...)
	return ret
}

type mathError struct {
	msg string
}

func (e mathError) Status() int {
	return FAILURE
}

func (e mathError) Error() string {
	return e.msg
}

func (e mathError) addMsg(msg string) {
	e.msg += " : " + msg
}

func mathErrorf(pattern string, arguments ...interface{}) mathError {
	return mathError{msg: fmt.Sprintf("Math Error: "+pattern, arguments)}
}
