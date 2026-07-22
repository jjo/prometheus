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
	"context"
	"fmt"
	"math"
	"slices"

	"github.com/prometheus/common/model"

	"github.com/prometheus/prometheus/model/labels"
	"github.com/prometheus/prometheus/promql/parser"
	"github.com/prometheus/prometheus/schema"
	"github.com/prometheus/prometheus/storage"
	"github.com/prometheus/prometheus/tsdb/chunkenc"
	"github.com/prometheus/prometheus/util/annotations"
)

// lm_over_time method names.
const (
	lmMethodOLS   = "lm"
	lmMethodRidge = "ridge"
)

// lmInterceptLabelValue is the reserved value placed on the pivot label of the
// emitted intercept coefficient series. A predictor whose pivot value equals
// this string would be indistinguishable from the intercept, so such a group
// is skipped with a warning.
const lmInterceptLabelValue = "(intercept)"

// lmR2LabelValue is the reserved pivot-label value for the emitted
// coefficient-of-determination (r²) series, which reports the fit quality
// (fraction of the response variance explained) alongside the coefficients.
// It lets callers tell a meaningful fit from a spurious one, which the
// coefficients alone cannot.
const lmR2LabelValue = "(r2)"

// maxLMPredictors caps the number of distinct pivot-label values (design-matrix
// columns) per group. The per-step solve is O(n·p² + p³), so this is set far
// below timeseries_gen's series cap to keep worst-case latency bounded.
const maxLMPredictors = 256

// qrRankTolerance is the absolute threshold below which a reflected column norm
// is treated as zero (rank-deficient design).
const qrRankTolerance = 1e-9

// householderLeastSquares solves min ||A·x - b|| for the least-squares
// coefficient vector x using Householder QR. A is n×p row-major (n ≥ p); b has
// length n. It returns the length-p solution and ok=true, or ok=false when the
// design is rank deficient (a reflected column norm falls below
// qrRankTolerance, i.e. collinear columns or too few rows). A and b are copied,
// not mutated.
func householderLeastSquares(a [][]float64, b []float64) ([]float64, bool) {
	n := len(a)
	if n == 0 {
		return nil, false
	}
	p := len(a[0])
	if n < p || p == 0 {
		return nil, false
	}

	// Work on copies so callers can reuse their buffers.
	r := make([][]float64, n)
	for i := range a {
		r[i] = slices.Clone(a[i])
	}
	y := slices.Clone(b)

	for k := range p {
		// Norm of the sub-column r[k:n][k].
		var sigma float64
		for i := k; i < n; i++ {
			sigma += r[i][k] * r[i][k]
		}
		sigma = math.Sqrt(sigma)
		if sigma < qrRankTolerance {
			return nil, false
		}
		// alpha = -sign(x_k)·||x|| chosen to avoid cancellation in v_k.
		if r[k][k] > 0 {
			sigma = -sigma
		}
		// Householder vector v lives in r[k:n][k]; v_k = x_k - alpha.
		r[k][k] -= sigma
		var vtv float64
		for i := k; i < n; i++ {
			vtv += r[i][k] * r[i][k]
		}
		if vtv < qrRankTolerance {
			return nil, false
		}
		// Reflect the trailing columns: w -= (2·vᵀw/vᵀv)·v.
		for j := k + 1; j < p; j++ {
			var s float64
			for i := k; i < n; i++ {
				s += r[i][k] * r[i][j]
			}
			s = 2 * s / vtv
			for i := k; i < n; i++ {
				r[i][j] -= s * r[i][k]
			}
		}
		// Reflect b.
		var s float64
		for i := k; i < n; i++ {
			s += r[i][k] * y[i]
		}
		s = 2 * s / vtv
		for i := k; i < n; i++ {
			y[i] -= s * r[i][k]
		}
		// The R diagonal is alpha; the reflector below it is no longer needed.
		r[k][k] = sigma
	}

	// Back-substitution on the upper-triangular R (p×p) against y[0:p].
	x := make([]float64, p)
	for i := p - 1; i >= 0; i-- {
		sum := y[i]
		for j := i + 1; j < p; j++ {
			sum -= r[i][j] * x[j]
		}
		if math.Abs(r[i][i]) < qrRankTolerance {
			return nil, false
		}
		x[i] = sum / r[i][i]
	}
	return x, true
}

// ridgeAugment appends p-1 penalty rows to the design matrix and zero entries to
// the response so that householderLeastSquares yields the ridge estimator
// β = (XᵀX + λI)⁻¹Xᵀy, with the intercept (column 0) left unpenalized. lambda
// must be > 0; the design matrix columns are [intercept, predictor₁, …].
func ridgeAugment(design [][]float64, resp []float64, lambda float64) ([][]float64, []float64) {
	p := len(design[0])
	sqrtLambda := math.Sqrt(lambda)
	aug := make([][]float64, len(design), len(design)+p-1)
	copy(aug, design)
	augResp := make([]float64, len(resp), len(resp)+p-1)
	copy(augResp, resp)
	for j := 1; j < p; j++ { // Skip column 0 (intercept).
		row := make([]float64, p)
		row[j] = sqrtLambda
		aug = append(aug, row)
		augResp = append(augResp, 0)
	}
	return aug, augResp
}

// lmR2 returns the coefficient of determination (r² = 1 − SS_res/SS_tot) of the
// fit with the given coefficients over the (unaugmented) design rows. It is NaN
// when there are no rows or the response has zero variance.
func lmR2(design [][]float64, resp, coeffs []float64) float64 {
	n := len(resp)
	if n == 0 {
		return math.NaN()
	}
	var mean float64
	for _, y := range resp {
		mean += y
	}
	mean /= float64(n)
	var ssRes, ssTot float64
	for i := range resp {
		var pred float64
		for j := range coeffs {
			pred += coeffs[j] * design[i][j]
		}
		d := resp[i] - pred
		ssRes += d * d
		dt := resp[i] - mean
		ssTot += dt * dt
	}
	if ssTot == 0 {
		return math.NaN()
	}
	return 1 - ssRes/ssTot
}

// lmPredictor is one design-matrix column: the pivot-label value and the index
// of the contributing series in the second range vector.
type lmPredictor struct {
	value     string
	seriesIdx int
}

// evalLMOverTime implements the lm_over_time PromQL function. It fits a multiple
// linear regression of the response range vector y on the predictor range
// vector X at each evaluation step, pivoting X into a design matrix whose
// columns are the distinct values of labelName.
//
// Signature: lm_over_time(method string, y range-vector, X range-vector,
// labelName string, lambda=0 scalar). method is "lm" (ordinary least squares)
// or "ridge" (L2-penalized, requires lambda > 0). Predictor series are grouped
// by all labels except __name__ and labelName; each group is one regression,
// and its distinct labelName values become the design-matrix columns. The
// response series is matched to a group by those same grouping labels. Each
// emitted series carries the group's labels plus labelName set to the
// predictor's value (or the reserved "(intercept)" for the intercept), and its
// value is the fitted coefficient.
//
// When labelName is empty the function degenerates to the bivariate case and
// returns the regression slope per matched pair, matching regression_over_time
// with the default (slope) output.
//
// A step yields NaN coefficients when the design is rank deficient (collinear
// predictors or fewer samples than columns); an info annotation is emitted.
// Groups whose predictor cardinality exceeds maxLMPredictors, or that include a
// predictor whose pivot value collides with "(intercept)", are skipped with a
// warning. Histogram samples are ignored.
func (ev *evaluator) evalLMOverTime(ctx context.Context, e *parser.Call) (parser.Value, annotations.Annotations) {
	var warnings annotations.Annotations

	method := stringFromArg(e.Args[0])
	if method != lmMethodOLS && method != lmMethodRidge {
		warnings.Add(annotations.NewInvalidLMMethodWarning(method, e.Args[0].PositionRange()))
		return Matrix{}, warnings
	}
	labelName := stringFromArg(e.Args[3])

	// Optional lambda scalar, evaluated per step like correlation's method arg.
	var lambdaMat Matrix
	lambdaHasArg := len(e.Args) > 4
	if lambdaHasArg {
		val, ws := ev.eval(ctx, e.Args[4])
		warnings.Merge(ws)
		var ok bool
		lambdaMat, ok = val.(Matrix)
		if !ok {
			ev.error(errWithWarnings{
				fmt.Errorf("lm_over_time: expected scalar lambda argument, got %T", val),
				warnings,
			})
		}
	}

	resolveMatrixArg := func(arg parser.Expr) *parser.MatrixSelector {
		switch a := arg.(type) {
		case *parser.MatrixSelector:
			return a
		case *parser.SubqueryExpr:
			ms, _, ws := ev.evalSubquery(ctx, a, durationMilliseconds(a.OriginalOffset), durationMilliseconds(a.Range))
			warnings.Merge(ws)
			return ms
		default:
			ev.error(errWithWarnings{
				fmt.Errorf("lm_over_time: expected a range-vector argument, got %T", arg),
				warnings,
			})
			return nil
		}
	}
	selY := resolveMatrixArg(e.Args[1])
	selX := resolveMatrixArg(e.Args[2])

	ws, err := checkAndExpandSeriesSet(ctx, selY)
	warnings.Merge(ws)
	if err != nil {
		ev.error(errWithWarnings{fmt.Errorf("expanding series: %w", err), warnings})
	}
	ws, err = checkAndExpandSeriesSet(ctx, selX)
	warnings.Merge(ws)
	if err != nil {
		ev.error(errWithWarnings{fmt.Errorf("expanding series: %w", err), warnings})
	}

	vsY := selY.VectorSelector.(*parser.VectorSelector)
	vsX := selX.VectorSelector.(*parser.VectorSelector)

	rangeY := durationMilliseconds(selY.Range)
	rangeX := durationMilliseconds(selX.Range)
	offsetY := durationMilliseconds(vsY.Offset)
	offsetX := durationMilliseconds(vsX.Offset)
	numSteps := int(1 + (ev.endTimestamp-ev.startTimestamp)/ev.interval)

	lambdaAt := func(step int) float64 {
		if lambdaHasArg && len(lambdaMat) > 0 && len(lambdaMat[0].Floats) > step {
			return lambdaMat[0].Floats[step].F
		}
		return 0
	}

	// Build the response lookup keyed by the grouping signature (all labels
	// except __name__ and, defensively, labelName).
	dropForGroup := func(lb labels.Labels, buf []byte) (uint64, []byte) {
		if labelName == "" {
			return lb.HashWithoutLabels(buf, model.MetricNameLabel)
		}
		return lb.HashWithoutLabels(buf, model.MetricNameLabel, labelName)
	}

	var hashBuf []byte
	respBySig := make(map[uint64]int, len(vsY.Series))
	ambiguousResp := make(map[uint64]struct{})
	for i, s := range vsY.Series {
		var sig uint64
		sig, hashBuf = dropForGroup(s.Labels(), hashBuf)
		if _, dup := respBySig[sig]; dup {
			if _, seen := ambiguousResp[sig]; !seen {
				ambiguousResp[sig] = struct{}{}
				warnings.Add(annotations.NewAmbiguousLMResponseWarning(
					s.Labels().DropReserved(schema.IsMetadataLabel).String(),
					e.Args[1].PositionRange(),
				))
			}
			continue
		}
		respBySig[sig] = i
	}

	if labelName == "" {
		return ev.lmBivariate(ctx, e, selX, selY, vsX, vsY, respBySig,
			rangeX, rangeY, offsetX, offsetY, numSteps, &warnings)
	}

	// Group predictor series by the grouping signature.
	type lmGroup struct {
		sig        uint64
		metric     labels.Labels // Group labels (without __name__/labelName).
		predictors []lmPredictor
	}
	groupBySig := make(map[uint64]*lmGroup)
	var groups []*lmGroup
	for i, s := range vsX.Series {
		var sig uint64
		sig, hashBuf = dropForGroup(s.Labels(), hashBuf)
		g := groupBySig[sig]
		if g == nil {
			g = &lmGroup{sig: sig, metric: s.Labels().DropReserved(schema.IsMetadataLabel)}
			groupBySig[sig] = g
			groups = append(groups, g)
		}
		g.predictors = append(g.predictors, lmPredictor{value: s.Labels().Get(labelName), seriesIdx: i})
	}

	var mat Matrix
	var chkIter chunkenc.Iterator

	for _, g := range groups {
		yIdx, ok := respBySig[g.sig]
		if !ok {
			continue // No response series for this predictor group.
		}
		// Deterministic column order by pivot value.
		slices.SortFunc(g.predictors, func(a, b lmPredictor) int {
			switch {
			case a.value < b.value:
				return -1
			case a.value > b.value:
				return +1
			default:
				return 0
			}
		})
		if len(g.predictors) > maxLMPredictors {
			warnings.Add(annotations.NewTooManyLMPredictorsWarning(len(g.predictors), maxLMPredictors, e.Args[2].PositionRange()))
			continue
		}
		reserved := false
		for _, pr := range g.predictors {
			if pr.value == lmInterceptLabelValue || pr.value == lmR2LabelValue {
				reserved = true
				break
			}
		}
		if reserved {
			warnings.Add(annotations.NewReservedLMInterceptLabelWarning(g.metric.String(), e.Args[2].PositionRange()))
			continue
		}

		k := len(g.predictors)

		if err := contextDone(ctx, "expression evaluation"); err != nil {
			ev.error(err)
		}

		// One buffered iterator for the response and each predictor.
		respIt := storage.NewBuffer(rangeY)
		chkIter = vsY.Series[yIdx].Iterator(chkIter)
		respIt.Reset(chkIter)

		predIts := make([]*storage.BufferedSeriesIterator, k)
		for j, pr := range g.predictors {
			it := storage.NewBuffer(rangeX)
			pit := vsX.Series[pr.seriesIdx].Iterator(nil)
			it.Reset(pit)
			predIts[j] = it
		}

		// Output series: one per coefficient (predictors + intercept) plus a
		// trailing r² fit-quality series.
		coeffSeries := make([]*Series, k+2)
		lb := labels.NewBuilder(g.metric)
		for j, pr := range g.predictors {
			lb.Reset(g.metric)
			lb.Set(labelName, pr.value)
			coeffSeries[j] = &Series{Metric: lb.Labels(), DropName: true}
		}
		lb.Reset(g.metric)
		lb.Set(labelName, lmInterceptLabelValue)
		coeffSeries[k] = &Series{Metric: lb.Labels(), DropName: true}
		lb.Reset(g.metric)
		lb.Set(labelName, lmR2LabelValue)
		coeffSeries[k+1] = &Series{Metric: lb.Labels(), DropName: true}

		var respFloats []FPoint
		var respHists []HPoint
		predFloats := make([][]FPoint, k)
		predHists := make([][]HPoint, k)
		rankDeficientSeen := false

		step := -1
		for ts := ev.startTimestamp; ts <= ev.endTimestamp; ts += ev.interval {
			step++
			maxtY := ts - offsetY
			mintY := maxtY - rangeY
			respFloats, respHists, _ = ev.matrixIterSlice(respIt, mintY, maxtY, respFloats, respHists, nil)

			// rows keyed by timestamp; only timestamps present in y and all
			// predictors form a design-matrix row.
			type rowAcc struct {
				y      float64
				x      []float64
				filled int
			}
			rows := make(map[int64]*rowAcc, len(respFloats))
			for _, fp := range respFloats {
				rows[fp.T] = &rowAcc{y: fp.F, x: make([]float64, k)}
			}
			for j := range g.predictors {
				maxtX := ts - offsetX
				mintX := maxtX - rangeX
				predFloats[j], predHists[j], _ = ev.matrixIterSlice(predIts[j], mintX, maxtX, predFloats[j], predHists[j], nil)
				for _, fp := range predFloats[j] {
					if ra, okRow := rows[fp.T]; okRow {
						ra.x[j] = fp.F
						ra.filled++
					}
				}
			}

			lambda := lambdaAt(step)
			useRidge := method == lmMethodRidge
			if useRidge && !(lambda > 0) {
				warnings.Add(annotations.NewInvalidRidgeLambdaWarning(lambda, e.Args[0].PositionRange()))
			}

			// Assemble the design matrix [1, x₁, …, x_k] over complete rows.
			design := make([][]float64, 0, len(rows))
			resp := make([]float64, 0, len(rows))
			for _, ra := range rows {
				if ra.filled != k {
					continue
				}
				row := make([]float64, k+1)
				row[0] = 1
				copy(row[1:], ra.x)
				design = append(design, row)
				resp = append(resp, ra.y)
			}

			coeffs := make([]float64, k+1)
			solvable := len(design) >= k+1
			if solvable {
				a, b := design, resp
				if useRidge && lambda > 0 {
					a, b = ridgeAugment(design, resp, lambda)
				}
				sol, ok := householderLeastSquares(a, b)
				if ok {
					copy(coeffs, sol)
				} else {
					solvable = false
				}
			}
			if !solvable {
				if !rankDeficientSeen {
					rankDeficientSeen = true
					warnings.Add(annotations.NewRankDeficientLMDesignInfo(e.Args[2].PositionRange()))
				}
				for i := range coeffs {
					coeffs[i] = math.NaN()
				}
			}

			// r² over the actual (unaugmented) rows; NaN when the fit failed.
			r2 := math.NaN()
			if solvable {
				r2 = lmR2(design, resp, coeffs)
			}

			// Emit: coeffs[0] is the intercept, coeffs[1..k] the predictors in
			// sorted column order, and a trailing r² fit-quality series.
			ev.currentSamples += k + 2
			ev.samplesStats.IncrementSamplesAtStep(step, int64(k+2))
			if ev.currentSamples > ev.maxSamples {
				ev.error(ErrTooManySamples(env))
			}
			for j := range k {
				if coeffSeries[j].Floats == nil {
					coeffSeries[j].Floats = getFPointSlice(numSteps)
				}
				coeffSeries[j].Floats = append(coeffSeries[j].Floats, FPoint{F: coeffs[j+1], T: ts})
			}
			if coeffSeries[k].Floats == nil {
				coeffSeries[k].Floats = getFPointSlice(numSteps)
			}
			coeffSeries[k].Floats = append(coeffSeries[k].Floats, FPoint{F: coeffs[0], T: ts})
			if coeffSeries[k+1].Floats == nil {
				coeffSeries[k+1].Floats = getFPointSlice(numSteps)
			}
			coeffSeries[k+1].Floats = append(coeffSeries[k+1].Floats, FPoint{F: r2, T: ts})

			stepRangeY := min(rangeY, ev.interval)
			respIt.ReduceDelta(stepRangeY)
			for j := range predIts {
				predIts[j].ReduceDelta(min(rangeX, ev.interval))
			}
		}

		ev.samplesStats.UpdatePeak(ev.currentSamples)
		ev.currentSamples -= len(respFloats) + totalHPointSize(respHists)
		putFPointSlice(respFloats)
		putMatrixSelectorHPointSlice(respHists)
		for j := range predFloats {
			ev.currentSamples -= len(predFloats[j]) + totalHPointSize(predHists[j])
			putFPointSlice(predFloats[j])
			putMatrixSelectorHPointSlice(predHists[j])
		}

		for _, s := range coeffSeries {
			if len(s.Floats) > 0 {
				mat = append(mat, *s)
			}
		}
	}

	if !ev.enableDelayedNameRemoval && mat.ContainsSameLabelset() {
		ev.errorf("vector cannot contain metrics with the same labelset")
	}
	return mat, warnings
}

// lmBivariate handles the no-pivot degenerate case: pair the response and
// predictor range vectors by label set (ignoring __name__) and emit the OLS
// slope per matched pair, matching regression_over_time's default output.
func (ev *evaluator) lmBivariate(
	ctx context.Context, _ *parser.Call,
	_, _ *parser.MatrixSelector,
	vsX, vsY *parser.VectorSelector,
	respBySig map[uint64]int,
	rangeX, rangeY, offsetX, offsetY int64,
	numSteps int, warnings *annotations.Annotations,
) (parser.Value, annotations.Annotations) {
	var mat Matrix
	var chkIterX, chkIterY chunkenc.Iterator
	var hashBuf []byte

	itX := storage.NewBuffer(rangeX)
	itY := storage.NewBuffer(rangeY)

	for _, sx := range vsX.Series {
		var sig uint64
		sig, hashBuf = sx.Labels().HashWithoutLabels(hashBuf, model.MetricNameLabel)
		yIdx, ok := respBySig[sig]
		if !ok {
			continue
		}
		if err := contextDone(ctx, "expression evaluation"); err != nil {
			ev.error(err)
		}
		chkIterX = sx.Iterator(chkIterX)
		itX.Reset(chkIterX)
		chkIterY = vsY.Series[yIdx].Iterator(chkIterY)
		itY.Reset(chkIterY)

		metric := sx.Labels().DropReserved(schema.IsMetadataLabel)
		ss := Series{Metric: metric, DropName: true}

		var floatsX, floatsY []FPoint
		var histsX, histsY []HPoint

		step := -1
		for ts := ev.startTimestamp; ts <= ev.endTimestamp; ts += ev.interval {
			step++
			maxtX := ts - offsetX
			maxtY := ts - offsetY
			floatsX, histsX, _ = ev.matrixIterSlice(itX, maxtX-rangeX, maxtX, floatsX, histsX, nil)
			floatsY, histsY, _ = ev.matrixIterSlice(itY, maxtY-rangeY, maxtY, floatsY, histsY, nil)
			if len(floatsX) == 0 || len(floatsY) == 0 {
				continue
			}
			// y is the response (first arg), x the predictor (second arg).
			y, x := alignByTimestamp(floatsY, floatsX)
			r := math.NaN()
			if fit := olsOnSlices(x, y); fit.ok {
				r = fit.slope
			}
			if ss.Floats == nil {
				ss.Floats = getFPointSlice(numSteps)
			}
			ev.currentSamples++
			ev.samplesStats.IncrementSamplesAtStep(step, 1)
			if ev.currentSamples > ev.maxSamples {
				ev.error(ErrTooManySamples(env))
			}
			ss.Floats = append(ss.Floats, FPoint{F: r, T: ts})
		}

		if len(ss.Floats) > 0 {
			mat = append(mat, ss)
		}
		ev.samplesStats.UpdatePeak(ev.currentSamples)
		ev.currentSamples -= len(floatsX) + len(floatsY) + totalHPointSize(histsX) + totalHPointSize(histsY)
		putFPointSlice(floatsX)
		putFPointSlice(floatsY)
		putMatrixSelectorHPointSlice(histsX)
		putMatrixSelectorHPointSlice(histsY)
		itX.ReduceDelta(min(rangeX, ev.interval))
		itY.ReduceDelta(min(rangeY, ev.interval))
	}

	if !ev.enableDelayedNameRemoval && mat.ContainsSameLabelset() {
		ev.errorf("vector cannot contain metrics with the same labelset")
	}
	return mat, *warnings
}
