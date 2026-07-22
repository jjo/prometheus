// Copyright The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package promql

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

// designFor builds the [1, x₁, x₂, …] design matrix from column-major predictors.
func designFor(cols ...[]float64) [][]float64 {
	n := len(cols[0])
	a := make([][]float64, n)
	for i := range n {
		row := make([]float64, len(cols)+1)
		row[0] = 1
		for j, c := range cols {
			row[j+1] = c[i]
		}
		a[i] = row
	}
	return a
}

func TestHouderholderLeastSquares_ExactFit(t *testing.T) {
	// y = 1 + 2·x1 + 3·x2, exactly recoverable.
	x1 := []float64{1, 2, 3, 4, 5}
	x2 := []float64{1, 0, 2, 1, 3}
	y := make([]float64, len(x1))
	for i := range y {
		y[i] = 1 + 2*x1[i] + 3*x2[i]
	}
	sol, ok := householderLeastSquares(designFor(x1, x2), y)
	require.True(t, ok)
	require.InDelta(t, 1.0, sol[0], 1e-9, "intercept")
	require.InDelta(t, 2.0, sol[1], 1e-9, "x1 coefficient")
	require.InDelta(t, 3.0, sol[2], 1e-9, "x2 coefficient")
}

func TestHouseholderLeastSquares_Overdetermined(t *testing.T) {
	// Simple bivariate OLS: y = 2x, slope 2, intercept 0.
	x := []float64{0, 1, 2, 3, 4}
	y := []float64{0, 2, 4, 6, 8}
	sol, ok := householderLeastSquares(designFor(x), y)
	require.True(t, ok)
	require.InDelta(t, 0.0, sol[0], 1e-9)
	require.InDelta(t, 2.0, sol[1], 1e-9)
}

func TestHouseholderLeastSquares_RankDeficient(t *testing.T) {
	// x2 = 2·x1: collinear predictors make the design rank deficient.
	x1 := []float64{1, 2, 3, 4, 5}
	x2 := []float64{2, 4, 6, 8, 10}
	y := []float64{1, 2, 3, 4, 5}
	_, ok := householderLeastSquares(designFor(x1, x2), y)
	require.False(t, ok)
}

func TestHouseholderLeastSquares_Underdetermined(t *testing.T) {
	// Fewer rows than columns (2 rows, 3 columns).
	x1 := []float64{1, 2}
	x2 := []float64{3, 4}
	y := []float64{1, 1}
	_, ok := householderLeastSquares(designFor(x1, x2), y)
	require.False(t, ok)
}

func TestRidgeAugment_ShrinksSlope(t *testing.T) {
	// With a large penalty the slope shrinks toward 0 and the (unpenalized)
	// intercept tends to the response mean.
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{6, 5, 13, 12, 20} // mean 11.2
	design := designFor(x)

	ols, ok := householderLeastSquares(design, y)
	require.True(t, ok)

	a, b := ridgeAugment(design, y, 1e6)
	ridge, ok := householderLeastSquares(a, b)
	require.True(t, ok)

	require.Less(t, math.Abs(ridge[1]), math.Abs(ols[1]), "ridge slope should be smaller in magnitude")
	require.InDelta(t, 0.0, ridge[1], 1e-3, "slope shrinks toward 0")
	require.InDelta(t, 11.2, ridge[0], 1e-3, "intercept tends to the mean")
}
