// Copyright 2016 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package promql_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/goleak"

	"github.com/prometheus/prometheus/model/histogram"
	"github.com/prometheus/prometheus/model/labels"
	"github.com/prometheus/prometheus/model/timestamp"
	"github.com/prometheus/prometheus/promql"
	"github.com/prometheus/prometheus/promql/parser"
	"github.com/prometheus/prometheus/promql/parser/posrange"
	"github.com/prometheus/prometheus/promql/promqltest"
	"github.com/prometheus/prometheus/storage"
	"github.com/prometheus/prometheus/util/annotations"
	"github.com/prometheus/prometheus/util/stats"
	"github.com/prometheus/prometheus/util/teststorage"
	"github.com/prometheus/prometheus/util/testutil"
)

const (
	env                  = "query execution"
	defaultLookbackDelta = 5 * time.Minute
	defaultEpsilon       = 0.000001 // Relative error allowed for sample values.
	// The largest SampleValue that can be converted to an int64 without overflow.
	maxInt64 = 9223372036854774784
)

func TestMain(m *testing.M) {
	// Enable experimental functions testing
	parser.EnableExperimentalFunctions = true
	goleak.VerifyTestMain(m)
}

func TestQueryConcurrency(t *testing.T) {
	maxConcurrency := 10

	dir, err := os.MkdirTemp("", "test_concurrency")
	require.NoError(t, err)
	defer os.RemoveAll(dir)
	queryTracker := promql.NewActiveQueryTracker(dir, maxConcurrency, nil)
	t.Cleanup(func() {
		require.NoError(t, queryTracker.Close())
	})

	opts := promql.EngineOpts{
		Logger:             nil,
		Reg:                nil,
		MaxSamples:         10,
		Timeout:            100 * time.Second,
		ActiveQueryTracker: queryTracker,
	}

	engine := promql.NewEngine(opts)
	ctx, cancelCtx := context.WithCancel(context.Background())
	defer cancelCtx()

	block := make(chan struct{})
	processing := make(chan struct{})
	done := make(chan int)
	defer close(done)

	f := func(context.Context) error {
		select {
		case processing <- struct{}{}:
		case <-done:
		}

		select {
		case <-block:
		case <-done:
		}
		return nil
	}

	var wg sync.WaitGroup
	for i := 0; i < maxConcurrency; i++ {
		q := engine.NewTestQuery(f)
		wg.Add(1)
		go func() {
			q.Exec(ctx)
			wg.Done()
		}()
		select {
		case <-processing:
			// Expected.
		case <-time.After(20 * time.Millisecond):
			require.Fail(t, "Query within concurrency threshold not being executed")
		}
	}

	q := engine.NewTestQuery(f)
	wg.Add(1)
	go func() {
		q.Exec(ctx)
		wg.Done()
	}()

	select {
	case <-processing:
		require.Fail(t, "Query above concurrency threshold being executed")
	case <-time.After(20 * time.Millisecond):
		// Expected.
	}

	// Terminate a running query.
	block <- struct{}{}

	select {
	case <-processing:
		// Expected.
	case <-time.After(20 * time.Millisecond):
		require.Fail(t, "Query within concurrency threshold not being executed")
	}

	// Terminate remaining queries.
	for i := 0; i < maxConcurrency; i++ {
		block <- struct{}{}
	}

	wg.Wait()
}

// contextDone returns an error if the context was canceled or timed out.
func contextDone(ctx context.Context, env string) error {
	if err := ctx.Err(); err != nil {
		switch {
		case errors.Is(err, context.Canceled):
			return promql.ErrQueryCanceled(env)
		case errors.Is(err, context.DeadlineExceeded):
			return promql.ErrQueryTimeout(env)
		default:
			return err
		}
	}
	return nil
}

func TestQueryTimeout(t *testing.T) {
	opts := promql.EngineOpts{
		Logger:     nil,
		Reg:        nil,
		MaxSamples: 10,
		Timeout:    5 * time.Millisecond,
	}
	engine := promql.NewEngine(opts)
	ctx, cancelCtx := context.WithCancel(context.Background())
	defer cancelCtx()

	query := engine.NewTestQuery(func(ctx context.Context) error {
		time.Sleep(100 * time.Millisecond)
		return contextDone(ctx, "test statement execution")
	})

	res := query.Exec(ctx)
	require.Error(t, res.Err, "expected timeout error but got none")

	var e promql.ErrQueryTimeout
	require.ErrorAs(t, res.Err, &e, "expected timeout error but got: %s", res.Err)
}

const errQueryCanceled = promql.ErrQueryCanceled("test statement execution")

func TestQueryCancel(t *testing.T) {
	opts := promql.EngineOpts{
		Logger:     nil,
		Reg:        nil,
		MaxSamples: 10,
		Timeout:    10 * time.Second,
	}
	engine := promql.NewEngine(opts)
	ctx, cancelCtx := context.WithCancel(context.Background())
	defer cancelCtx()

	// Cancel a running query before it completes.
	block := make(chan struct{})
	processing := make(chan struct{})

	query1 := engine.NewTestQuery(func(ctx context.Context) error {
		processing <- struct{}{}
		<-block
		return contextDone(ctx, "test statement execution")
	})

	var res *promql.Result

	go func() {
		res = query1.Exec(ctx)
		processing <- struct{}{}
	}()

	<-processing
	query1.Cancel()
	block <- struct{}{}
	<-processing

	require.Error(t, res.Err, "expected cancellation error for query1 but got none")
	require.Equal(t, errQueryCanceled, res.Err)

	// Canceling a query before starting it must have no effect.
	query2 := engine.NewTestQuery(func(ctx context.Context) error {
		return contextDone(ctx, "test statement execution")
	})

	query2.Cancel()
	res = query2.Exec(ctx)
	require.NoError(t, res.Err)
}

// errQuerier implements storage.Querier which always returns error.
type errQuerier struct {
	err error
}

func (q *errQuerier) Select(context.Context, bool, *storage.SelectHints, ...*labels.Matcher) storage.SeriesSet {
	return errSeriesSet{err: q.err}
}

func (*errQuerier) LabelValues(context.Context, string, ...*labels.Matcher) ([]string, annotations.Annotations, error) {
	return nil, nil, nil
}

func (*errQuerier) LabelNames(context.Context, ...*labels.Matcher) ([]string, annotations.Annotations, error) {
	return nil, nil, nil
}
func (*errQuerier) Close() error { return nil }

// errSeriesSet implements storage.SeriesSet which always returns error.
type errSeriesSet struct {
	err error
}

func (errSeriesSet) Next() bool                          { return false }
func (errSeriesSet) At() storage.Series                  { return nil }
func (e errSeriesSet) Err() error                        { return e.err }
func (e errSeriesSet) Warnings() annotations.Annotations { return nil }

func TestQueryError(t *testing.T) {
	opts := promql.EngineOpts{
		Logger:     nil,
		Reg:        nil,
		MaxSamples: 10,
		Timeout:    10 * time.Second,
	}
	engine := promql.NewEngine(opts)
	errStorage := promql.ErrStorage{errors.New("storage error")}
	queryable := storage.QueryableFunc(func(mint, maxt int64) (storage.Querier, error) {
		return &errQuerier{err: errStorage}, nil
	})
	ctx, cancelCtx := context.WithCancel(context.Background())
	defer cancelCtx()

	vectorQuery, err := engine.NewInstantQuery(ctx, queryable, nil, "foo", time.Unix(1, 0))
	require.NoError(t, err)

	res := vectorQuery.Exec(ctx)
	require.Error(t, res.Err, "expected error on failed select but got none")
	require.ErrorIs(t, res.Err, errStorage, "expected error doesn't match")

	matrixQuery, err := engine.NewInstantQuery(ctx, queryable, nil, "foo[1m]", time.Unix(1, 0))
	require.NoError(t, err)

	res = matrixQuery.Exec(ctx)
	require.Error(t, res.Err, "expected error on failed select but got none")
	require.ErrorIs(t, res.Err, errStorage, "expected error doesn't match")
}

type noopHintRecordingQueryable struct {
	hints []*storage.SelectHints
}

func (h *noopHintRecordingQueryable) Querier(int64, int64) (storage.Querier, error) {
	return &hintRecordingQuerier{Querier: &errQuerier{}, h: h}, nil
}

type hintRecordingQuerier struct {
	storage.Querier

	h *noopHintRecordingQueryable
}

func (h *hintRecordingQuerier) Select(ctx context.Context, sortSeries bool, hints *storage.SelectHints, matchers ...*labels.Matcher) storage.SeriesSet {
	h.h.hints = append(h.h.hints, hints)
	return h.Querier.Select(ctx, sortSeries, hints, matchers...)
}

func TestSelectHintsSetCorrectly(t *testing.T) {
	opts := promql.EngineOpts{
		Logger:           nil,
		Reg:              nil,
		MaxSamples:       10,
		Timeout:          10 * time.Second,
		LookbackDelta:    5 * time.Second,
		EnableAtModifier: true,
	}

	for _, tc := range []struct {
		query string

		// All times are in milliseconds.
		start int64
		end   int64

		// TODO(bwplotka): Add support for better hints when subquerying.
		expected []*storage.SelectHints
	}{
		{
			query: "foo", start: 10000,
			expected: []*storage.SelectHints{
				{Start: 5000, End: 10000},
			},
		}, {
			query: "foo @ 15", start: 10000,
			expected: []*storage.SelectHints{
				{Start: 10000, End: 15000},
			},
		}, {
			query: "foo @ 1", start: 10000,
			expected: []*storage.SelectHints{
				{Start: -4000, End: 1000},
			},
		}, {
			query: "foo[2m]", start: 200000,
			expected: []*storage.SelectHints{
				{Start: 80000, End: 200000, Range: 120000},
			},
		}, {
			query: "foo[2m] @ 180", start: 200000,
			expected: []*storage.SelectHints{
				{Start: 60000, End: 180000, Range: 120000},
			},
		}, {
			query: "foo[2m] @ 300", start: 200000,
			expected: []*storage.SelectHints{
				{Start: 180000, End: 300000, Range: 120000},
			},
		}, {
			query: "foo[2m] @ 60", start: 200000,
			expected: []*storage.SelectHints{
				{Start: -60000, End: 60000, Range: 120000},
			},
		}, {
			query: "foo[2m] offset 2m", start: 300000,
			expected: []*storage.SelectHints{
				{Start: 60000, End: 180000, Range: 120000},
			},
		}, {
			query: "foo[2m] @ 200 offset 2m", start: 300000,
			expected: []*storage.SelectHints{
				{Start: -40000, End: 80000, Range: 120000},
			},
		}, {
			query: "foo[2m:1s]", start: 300000,
			expected: []*storage.SelectHints{
				{Start: 175000, End: 300000, Step: 1000},
			},
		}, {
			query: "count_over_time(foo[2m:1s])", start: 300000,
			expected: []*storage.SelectHints{
				{Start: 175000, End: 300000, Func: "count_over_time", Step: 1000},
			},
		}, {
			query: "count_over_time(foo[2m:1s] @ 300)", start: 200000,
			expected: []*storage.SelectHints{
				{Start: 175000, End: 300000, Func: "count_over_time", Step: 1000},
			},
		}, {
			query: "count_over_time(foo[2m:1s] @ 200)", start: 200000,
			expected: []*storage.SelectHints{
				{Start: 75000, End: 200000, Func: "count_over_time", Step: 1000},
			},
		}, {
			query: "count_over_time(foo[2m:1s] @ 100)", start: 200000,
			expected: []*storage.SelectHints{
				{Start: -25000, End: 100000, Func: "count_over_time", Step: 1000},
			},
		}, {
			query: "count_over_time(foo[2m:1s] offset 10s)", start: 300000,
			expected: []*storage.SelectHints{
				{Start: 165000, End: 290000, Func: "count_over_time", Step: 1000},
			},
		}, {
			query: "count_over_time((foo offset 10s)[2m:1s] offset 10s)", start: 300000,
			expected: []*storage.SelectHints{
				{Start: 155000, End: 280000, Func: "count_over_time", Step: 1000},
			},
		}, {
			// When the @ is on the vector selector, the enclosing subquery parameters
			// don't affect the hint ranges.
			query: "count_over_time((foo @ 200 offset 10s)[2m:1s] offset 10s)", start: 300000,
			expected: []*storage.SelectHints{
				{Start: 185000, End: 190000, Func: "count_over_time", Step: 1000},
			},
		}, {
			// When the @ is on the vector selector, the enclosing subquery parameters
			// don't affect the hint ranges.
			query: "count_over_time((foo @ 200 offset 10s)[2m:1s] @ 100 offset 10s)", start: 300000,
			expected: []*storage.SelectHints{
				{Start: 185000, End: 190000, Func: "count_over_time", Step: 1000},
			},
		}, {
			query: "count_over_time((foo offset 10s)[2m:1s] @ 100 offset 10s)", start: 300000,
			expected: []*storage.SelectHints{
				{Start: -45000, End: 80000, Func: "count_over_time", Step: 1000},
			},
		}, {
			query: "foo", start: 10000, end: 20000,
			expected: []*storage.SelectHints{
				{Start: 5000, End: 20000, Step: 1000},
			},
		}, {
			query: "foo @ 15", start: 10000, end: 20000,
			expected: []*storage.SelectHints{
				{Start: 10000, End: 15000, Step: 1000},
			},
		}, {
			query: "foo @ 1", start: 10000, end: 20000,
			expected: []*storage.SelectHints{
				{Start: -4000, End: 1000, Step: 1000},
			},
		}, {
			query: "rate(foo[2m] @ 180)", start: 200000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: 60000, End: 180000, Range: 120000, Func: "rate", Step: 1000},
			},
		}, {
			query: "rate(foo[2m] @ 300)", start: 200000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: 180000, End: 300000, Range: 120000, Func: "rate", Step: 1000},
			},
		}, {
			query: "rate(foo[2m] @ 60)", start: 200000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: -60000, End: 60000, Range: 120000, Func: "rate", Step: 1000},
			},
		}, {
			query: "rate(foo[2m])", start: 200000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: 80000, End: 500000, Range: 120000, Func: "rate", Step: 1000},
			},
		}, {
			query: "rate(foo[2m] offset 2m)", start: 300000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: 60000, End: 380000, Range: 120000, Func: "rate", Step: 1000},
			},
		}, {
			query: "rate(foo[2m:1s])", start: 300000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: 175000, End: 500000, Func: "rate", Step: 1000},
			},
		}, {
			query: "count_over_time(foo[2m:1s])", start: 300000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: 175000, End: 500000, Func: "count_over_time", Step: 1000},
			},
		}, {
			query: "count_over_time(foo[2m:1s] offset 10s)", start: 300000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: 165000, End: 490000, Func: "count_over_time", Step: 1000},
			},
		}, {
			query: "count_over_time(foo[2m:1s] @ 300)", start: 200000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: 175000, End: 300000, Func: "count_over_time", Step: 1000},
			},
		}, {
			query: "count_over_time(foo[2m:1s] @ 200)", start: 200000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: 75000, End: 200000, Func: "count_over_time", Step: 1000},
			},
		}, {
			query: "count_over_time(foo[2m:1s] @ 100)", start: 200000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: -25000, End: 100000, Func: "count_over_time", Step: 1000},
			},
		}, {
			query: "count_over_time((foo offset 10s)[2m:1s] offset 10s)", start: 300000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: 155000, End: 480000, Func: "count_over_time", Step: 1000},
			},
		}, {
			// When the @ is on the vector selector, the enclosing subquery parameters
			// don't affect the hint ranges.
			query: "count_over_time((foo @ 200 offset 10s)[2m:1s] offset 10s)", start: 300000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: 185000, End: 190000, Func: "count_over_time", Step: 1000},
			},
		}, {
			// When the @ is on the vector selector, the enclosing subquery parameters
			// don't affect the hint ranges.
			query: "count_over_time((foo @ 200 offset 10s)[2m:1s] @ 100 offset 10s)", start: 300000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: 185000, End: 190000, Func: "count_over_time", Step: 1000},
			},
		}, {
			query: "count_over_time((foo offset 10s)[2m:1s] @ 100 offset 10s)", start: 300000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: -45000, End: 80000, Func: "count_over_time", Step: 1000},
			},
		}, {
			query: "sum by (dim1) (foo)", start: 10000,
			expected: []*storage.SelectHints{
				{Start: 5000, End: 10000, Func: "sum", By: true, Grouping: []string{"dim1"}},
			},
		}, {
			query: "sum without (dim1) (foo)", start: 10000,
			expected: []*storage.SelectHints{
				{Start: 5000, End: 10000, Func: "sum", Grouping: []string{"dim1"}},
			},
		}, {
			query: "sum by (dim1) (avg_over_time(foo[1s]))", start: 10000,
			expected: []*storage.SelectHints{
				{Start: 9000, End: 10000, Func: "avg_over_time", Range: 1000},
			},
		}, {
			query: "sum by (dim1) (max by (dim2) (foo))", start: 10000,
			expected: []*storage.SelectHints{
				{Start: 5000, End: 10000, Func: "max", By: true, Grouping: []string{"dim2"}},
			},
		}, {
			query: "(max by (dim1) (foo))[5s:1s]", start: 10000,
			expected: []*storage.SelectHints{
				{Start: 0, End: 10000, Func: "max", By: true, Grouping: []string{"dim1"}, Step: 1000},
			},
		}, {
			query: "(sum(http_requests{group=~\"p.*\"})+max(http_requests{group=~\"c.*\"}))[20s:5s]", start: 120000,
			expected: []*storage.SelectHints{
				{Start: 95000, End: 120000, Func: "sum", By: true, Step: 5000},
				{Start: 95000, End: 120000, Func: "max", By: true, Step: 5000},
			},
		}, {
			query: "foo @ 50 + bar @ 250 + baz @ 900", start: 100000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: 45000, End: 50000, Step: 1000},
				{Start: 245000, End: 250000, Step: 1000},
				{Start: 895000, End: 900000, Step: 1000},
			},
		}, {
			query: "foo @ 50 + bar + baz @ 900", start: 100000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: 45000, End: 50000, Step: 1000},
				{Start: 95000, End: 500000, Step: 1000},
				{Start: 895000, End: 900000, Step: 1000},
			},
		}, {
			query: "rate(foo[2s] @ 50) + bar @ 250 + baz @ 900", start: 100000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: 48000, End: 50000, Step: 1000, Func: "rate", Range: 2000},
				{Start: 245000, End: 250000, Step: 1000},
				{Start: 895000, End: 900000, Step: 1000},
			},
		}, {
			query: "rate(foo[2s:1s] @ 50) + bar + baz", start: 100000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: 43000, End: 50000, Step: 1000, Func: "rate"},
				{Start: 95000, End: 500000, Step: 1000},
				{Start: 95000, End: 500000, Step: 1000},
			},
		}, {
			query: "rate(foo[2s:1s] @ 50) + bar + rate(baz[2m:1s] @ 900 offset 2m) ", start: 100000, end: 500000,
			expected: []*storage.SelectHints{
				{Start: 43000, End: 50000, Step: 1000, Func: "rate"},
				{Start: 95000, End: 500000, Step: 1000},
				{Start: 655000, End: 780000, Step: 1000, Func: "rate"},
			},
		}, { // Hints are based on the inner most subquery timestamp.
			query: `sum_over_time(sum_over_time(metric{job="1"}[100s])[100s:25s] @ 50)[3s:1s] @ 3000`, start: 100000,
			expected: []*storage.SelectHints{
				{Start: -150000, End: 50000, Range: 100000, Func: "sum_over_time", Step: 25000},
			},
		}, { // Hints are based on the inner most subquery timestamp.
			query: `sum_over_time(sum_over_time(metric{job="1"}[100s])[100s:25s] @ 3000)[3s:1s] @ 50`,
			expected: []*storage.SelectHints{
				{Start: 2800000, End: 3000000, Range: 100000, Func: "sum_over_time", Step: 25000},
			},
		},
	} {
		t.Run(tc.query, func(t *testing.T) {
			engine := promql.NewEngine(opts)
			hintsRecorder := &noopHintRecordingQueryable{}

			var (
				query promql.Query
				err   error
			)
			ctx := context.Background()

			if tc.end == 0 {
				query, err = engine.NewInstantQuery(ctx, hintsRecorder, nil, tc.query, timestamp.Time(tc.start))
			} else {
				query, err = engine.NewRangeQuery(ctx, hintsRecorder, nil, tc.query, timestamp.Time(tc.start), timestamp.Time(tc.end), time.Second)
			}
			require.NoError(t, err)

			res := query.Exec(context.Background())
			require.NoError(t, res.Err)

			require.Equal(t, tc.expected, hintsRecorder.hints)
		})
	}
}

func TestEngineShutdown(t *testing.T) {
	opts := promql.EngineOpts{
		Logger:     nil,
		Reg:        nil,
		MaxSamples: 10,
		Timeout:    10 * time.Second,
	}
	engine := promql.NewEngine(opts)
	ctx, cancelCtx := context.WithCancel(context.Background())

	block := make(chan struct{})
	processing := make(chan struct{})

	// Shutdown engine on first handler execution. Should handler execution ever become
	// concurrent this test has to be adjusted accordingly.
	f := func(ctx context.Context) error {
		processing <- struct{}{}
		<-block
		return contextDone(ctx, "test statement execution")
	}
	query1 := engine.NewTestQuery(f)

	// Stopping the engine must cancel the base context. While executing queries is
	// still possible, their context is canceled from the beginning and execution should
	// terminate immediately.

	var res *promql.Result
	go func() {
		res = query1.Exec(ctx)
		processing <- struct{}{}
	}()

	<-processing
	cancelCtx()
	block <- struct{}{}
	<-processing

	require.Error(t, res.Err, "expected error on shutdown during query but got none")
	require.Equal(t, errQueryCanceled, res.Err)

	query2 := engine.NewTestQuery(func(context.Context) error {
		require.FailNow(t, "reached query execution unexpectedly")
		return nil
	})

	// The second query is started after the engine shut down. It must
	// be canceled immediately.
	res2 := query2.Exec(ctx)
	require.Error(t, res2.Err, "expected error on querying with canceled context but got none")

	var e promql.ErrQueryCanceled
	require.ErrorAs(t, res2.Err, &e, "expected cancellation error but got: %s", res2.Err)
}

func TestEngineEvalStmtTimestamps(t *testing.T) {
	storage := promqltest.LoadedStorage(t, `
load 10s
  metric 1 2
`)
	t.Cleanup(func() { storage.Close() })

	cases := []struct {
		Query       string
		Result      parser.Value
		Start       time.Time
		End         time.Time
		Interval    time.Duration
		ShouldError bool
	}{
		// Instant queries.
		{
			Query:  "1",
			Result: promql.Scalar{V: 1, T: 1000},
			Start:  time.Unix(1, 0),
		},
		{
			Query: "metric",
			Result: promql.Vector{
				promql.Sample{
					F:      1,
					T:      1000,
					Metric: labels.FromStrings("__name__", "metric"),
				},
			},
			Start: time.Unix(1, 0),
		},
		{
			Query: "metric[20s]",
			Result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{{F: 1, T: 0}, {F: 2, T: 10000}},
					Metric: labels.FromStrings("__name__", "metric"),
				},
			},
			Start: time.Unix(10, 0),
		},
		// Range queries.
		{
			Query: "1",
			Result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{{F: 1, T: 0}, {F: 1, T: 1000}, {F: 1, T: 2000}},
					Metric: labels.EmptyLabels(),
				},
			},
			Start:    time.Unix(0, 0),
			End:      time.Unix(2, 0),
			Interval: time.Second,
		},
		{
			Query: "metric",
			Result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{{F: 1, T: 0}, {F: 1, T: 1000}, {F: 1, T: 2000}},
					Metric: labels.FromStrings("__name__", "metric"),
				},
			},
			Start:    time.Unix(0, 0),
			End:      time.Unix(2, 0),
			Interval: time.Second,
		},
		{
			Query: "metric",
			Result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{{F: 1, T: 0}, {F: 1, T: 5000}, {F: 2, T: 10000}},
					Metric: labels.FromStrings("__name__", "metric"),
				},
			},
			Start:    time.Unix(0, 0),
			End:      time.Unix(10, 0),
			Interval: 5 * time.Second,
		},
		{
			Query:       `count_values("wrong label!", metric)`,
			ShouldError: true,
		},
	}

	for i, c := range cases {
		t.Run(fmt.Sprintf("%d query=%s", i, c.Query), func(t *testing.T) {
			var err error
			var qry promql.Query
			engine := newTestEngine()
			if c.Interval == 0 {
				qry, err = engine.NewInstantQuery(context.Background(), storage, nil, c.Query, c.Start)
			} else {
				qry, err = engine.NewRangeQuery(context.Background(), storage, nil, c.Query, c.Start, c.End, c.Interval)
			}
			require.NoError(t, err)

			res := qry.Exec(context.Background())
			if c.ShouldError {
				require.Error(t, res.Err, "expected error for the query %q", c.Query)
				return
			}

			require.NoError(t, res.Err)
			require.Equal(t, c.Result, res.Value, "query %q failed", c.Query)
		})
	}
}

func TestQueryStatistics(t *testing.T) {
	storage := promqltest.LoadedStorage(t, `
load 10s
  metricWith1SampleEvery10Seconds 1+1x100
  metricWith3SampleEvery10Seconds{a="1",b="1"} 1+1x100
  metricWith3SampleEvery10Seconds{a="2",b="2"} 1+1x100
  metricWith3SampleEvery10Seconds{a="3",b="2"} 1+1x100
  metricWith1HistogramEvery10Seconds {{schema:1 count:5 sum:20 buckets:[1 2 1 1]}}+{{schema:1 count:10 sum:5 buckets:[1 2 3 4]}}x100
`)
	t.Cleanup(func() { storage.Close() })

	cases := []struct {
		Query               string
		SkipMaxCheck        bool
		TotalSamples        int64
		TotalSamplesPerStep stats.TotalSamplesPerStep
		PeakSamples         int
		Start               time.Time
		End                 time.Time
		Interval            time.Duration
	}{
		{
			Query:        `"literal string"`,
			SkipMaxCheck: true, // This can't fail from a max samples limit.
			Start:        time.Unix(21, 0),
			TotalSamples: 0,
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				21000: 0,
			},
		},
		{
			Query:        "1",
			Start:        time.Unix(21, 0),
			TotalSamples: 0,
			PeakSamples:  1,
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				21000: 0,
			},
		},
		{
			Query:        "metricWith1SampleEvery10Seconds",
			Start:        time.Unix(21, 0),
			PeakSamples:  1,
			TotalSamples: 1, // 1 sample / 10 seconds
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				21000: 1,
			},
		},
		{
			Query:        "metricWith1HistogramEvery10Seconds",
			Start:        time.Unix(21, 0),
			PeakSamples:  13,
			TotalSamples: 13, // 1 histogram HPoint of size 13 / 10 seconds
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				21000: 13,
			},
		},
		{
			// timestamp function has a special handling.
			Query:        "timestamp(metricWith1SampleEvery10Seconds)",
			Start:        time.Unix(21, 0),
			PeakSamples:  2,
			TotalSamples: 1, // 1 sample / 10 seconds
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				21000: 1,
			},
		},
		{
			Query:        "timestamp(metricWith1HistogramEvery10Seconds)",
			Start:        time.Unix(21, 0),
			PeakSamples:  2,
			TotalSamples: 1, // 1 float sample (because of timestamp) / 10 seconds
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				21000: 1,
			},
		},
		{
			Query:        "metricWith1SampleEvery10Seconds",
			Start:        time.Unix(22, 0),
			PeakSamples:  1,
			TotalSamples: 1, // 1 sample / 10 seconds
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				22000: 1, // Aligned to the step time, not the sample time.
			},
		},
		{
			Query:        "metricWith1SampleEvery10Seconds offset 10s",
			Start:        time.Unix(21, 0),
			PeakSamples:  1,
			TotalSamples: 1, // 1 sample / 10 seconds
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				21000: 1,
			},
		},
		{
			Query:        "metricWith1SampleEvery10Seconds @ 15",
			Start:        time.Unix(21, 0),
			PeakSamples:  1,
			TotalSamples: 1, // 1 sample / 10 seconds
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				21000: 1,
			},
		},
		{
			Query:        `metricWith3SampleEvery10Seconds{a="1"}`,
			Start:        time.Unix(21, 0),
			PeakSamples:  1,
			TotalSamples: 1, // 1 sample / 10 seconds
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				21000: 1,
			},
		},
		{
			Query:        `metricWith3SampleEvery10Seconds{a="1"} @ 19`,
			Start:        time.Unix(21, 0),
			PeakSamples:  1,
			TotalSamples: 1, // 1 sample / 10 seconds
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				21000: 1,
			},
		},
		{
			Query:        `metricWith3SampleEvery10Seconds{a="1"}[20s] @ 19`,
			Start:        time.Unix(21, 0),
			PeakSamples:  2,
			TotalSamples: 2, // (1 sample / 10 seconds) * 20s
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				21000: 2,
			},
		},
		{
			Query:        "metricWith3SampleEvery10Seconds",
			Start:        time.Unix(21, 0),
			PeakSamples:  3,
			TotalSamples: 3, // 3 samples / 10 seconds
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				21000: 3,
			},
		},
		{
			Query:        "metricWith1SampleEvery10Seconds[60s]",
			Start:        time.Unix(201, 0),
			PeakSamples:  6,
			TotalSamples: 6, // 1 sample / 10 seconds * 60 seconds
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 6,
			},
		},
		{
			Query:        "metricWith1HistogramEvery10Seconds[60s]",
			Start:        time.Unix(201, 0),
			PeakSamples:  78,
			TotalSamples: 78, // 1 histogram (size 13 HPoint) / 10 seconds * 60 seconds
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 78,
			},
		},
		{
			Query:        "max_over_time(metricWith1SampleEvery10Seconds[59s])[20s:5s]",
			Start:        time.Unix(201, 0),
			PeakSamples:  10,
			TotalSamples: 24, // (1 sample / 10 seconds * 60 seconds) * 20/5 (using 59s so we always return 6 samples
			// as if we run a query on 00 looking back 60 seconds we will return 7 samples;
			// see next test).
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 24,
			},
		},
		{
			Query:        "max_over_time(metricWith1SampleEvery10Seconds[60s])[20s:5s]",
			Start:        time.Unix(201, 0),
			PeakSamples:  11,
			TotalSamples: 26, // (1 sample / 10 seconds * 60 seconds) * 4 + 2 as
			// max_over_time(metricWith1SampleEvery10Seconds[60s]) @ 190 and 200 will return 7 samples.
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 26,
			},
		},
		{
			Query:        "max_over_time(metricWith1HistogramEvery10Seconds[60s])[20s:5s]",
			Start:        time.Unix(201, 0),
			PeakSamples:  78,
			TotalSamples: 338, // (1 histogram (size 13 HPoint) / 10 seconds * 60 seconds) * 4 + 2 * 13 as
			// max_over_time(metricWith1SampleEvery10Seconds[60s]) @ 190 and 200 will return 7 samples.
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 338,
			},
		},
		{
			Query:        "metricWith1SampleEvery10Seconds[60s] @ 30",
			Start:        time.Unix(201, 0),
			PeakSamples:  4,
			TotalSamples: 4, // @ modifier force the evaluation to at 30 seconds - So it brings 4 datapoints (0, 10, 20, 30 seconds) * 1 series
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 4,
			},
		},
		{
			Query:        "metricWith1HistogramEvery10Seconds[60s] @ 30",
			Start:        time.Unix(201, 0),
			PeakSamples:  52,
			TotalSamples: 52, // @ modifier force the evaluation to at 30 seconds - So it brings 4 datapoints (0, 10, 20, 30 seconds) * 1 series
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 52,
			},
		},
		{
			Query:        "sum(max_over_time(metricWith3SampleEvery10Seconds[60s] @ 30))",
			Start:        time.Unix(201, 0),
			PeakSamples:  7,
			TotalSamples: 12, // @ modifier force the evaluation to at 30 seconds - So it brings 4 datapoints (0, 10, 20, 30 seconds) * 3 series
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 12,
			},
		},
		{
			Query:        "sum by (b) (max_over_time(metricWith3SampleEvery10Seconds[60s] @ 30))",
			Start:        time.Unix(201, 0),
			PeakSamples:  7,
			TotalSamples: 12, // @ modifier force the evaluation to at 30 seconds - So it brings 4 datapoints (0, 10, 20, 30 seconds) * 3 series
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 12,
			},
		},
		{
			Query:        "metricWith1SampleEvery10Seconds[60s] offset 10s",
			Start:        time.Unix(201, 0),
			PeakSamples:  6,
			TotalSamples: 6, // 1 sample / 10 seconds * 60 seconds
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 6,
			},
		},
		{
			Query:        "metricWith3SampleEvery10Seconds[60s]",
			Start:        time.Unix(201, 0),
			PeakSamples:  18,
			TotalSamples: 18, // 3 sample / 10 seconds * 60 seconds
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 18,
			},
		},
		{
			Query:        "max_over_time(metricWith1SampleEvery10Seconds[60s])",
			Start:        time.Unix(201, 0),
			PeakSamples:  7,
			TotalSamples: 6, // 1 sample / 10 seconds * 60 seconds
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 6,
			},
		},
		{
			Query:        "absent_over_time(metricWith1SampleEvery10Seconds[60s])",
			Start:        time.Unix(201, 0),
			PeakSamples:  7,
			TotalSamples: 6, // 1 sample / 10 seconds * 60 seconds
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 6,
			},
		},
		{
			Query:        "max_over_time(metricWith3SampleEvery10Seconds[60s])",
			Start:        time.Unix(201, 0),
			PeakSamples:  9,
			TotalSamples: 18, // 3 sample / 10 seconds * 60 seconds
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 18,
			},
		},
		{
			Query:        "metricWith1SampleEvery10Seconds[60s:5s]",
			Start:        time.Unix(201, 0),
			PeakSamples:  12,
			TotalSamples: 12, // 1 sample per query * 12 queries (60/5)
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 12,
			},
		},
		{
			Query:        "metricWith1SampleEvery10Seconds[60s:5s] offset 10s",
			Start:        time.Unix(201, 0),
			PeakSamples:  12,
			TotalSamples: 12, // 1 sample per query * 12 queries (60/5)
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 12,
			},
		},
		{
			Query:        "max_over_time(metricWith3SampleEvery10Seconds[60s:5s])",
			Start:        time.Unix(201, 0),
			PeakSamples:  51,
			TotalSamples: 36, // 3 sample per query * 12 queries (60/5)
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 36,
			},
		},
		{
			Query:        "sum(max_over_time(metricWith3SampleEvery10Seconds[60s:5s])) + sum(max_over_time(metricWith3SampleEvery10Seconds[60s:5s]))",
			Start:        time.Unix(201, 0),
			PeakSamples:  52,
			TotalSamples: 72, // 2 * (3 sample per query * 12 queries (60/5))
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 72,
			},
		},
		{
			Query:        `metricWith3SampleEvery10Seconds{a="1"}`,
			Start:        time.Unix(201, 0),
			End:          time.Unix(220, 0),
			Interval:     5 * time.Second,
			PeakSamples:  4,
			TotalSamples: 4, // 1 sample per query * 4 steps
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 1,
				206000: 1,
				211000: 1,
				216000: 1,
			},
		},
		{
			Query:        `metricWith3SampleEvery10Seconds{a="1"}`,
			Start:        time.Unix(204, 0),
			End:          time.Unix(223, 0),
			Interval:     5 * time.Second,
			PeakSamples:  4,
			TotalSamples: 4, // 1 sample per query * 4 steps
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				204000: 1, // aligned to the step time, not the sample time
				209000: 1,
				214000: 1,
				219000: 1,
			},
		},
		{
			Query:        `metricWith1HistogramEvery10Seconds`,
			Start:        time.Unix(204, 0),
			End:          time.Unix(223, 0),
			Interval:     5 * time.Second,
			PeakSamples:  52,
			TotalSamples: 52, // 1 histogram (size 13 HPoint) per query * 4 steps
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				204000: 13, // aligned to the step time, not the sample time
				209000: 13,
				214000: 13,
				219000: 13,
			},
		},
		{
			// timestamp function has a special handling
			Query:        "timestamp(metricWith1SampleEvery10Seconds)",
			Start:        time.Unix(201, 0),
			End:          time.Unix(220, 0),
			Interval:     5 * time.Second,
			PeakSamples:  5,
			TotalSamples: 4, // 1 sample per query * 4 steps
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 1,
				206000: 1,
				211000: 1,
				216000: 1,
			},
		},
		{
			// timestamp function has a special handling
			Query:        "timestamp(metricWith1HistogramEvery10Seconds)",
			Start:        time.Unix(201, 0),
			End:          time.Unix(220, 0),
			Interval:     5 * time.Second,
			PeakSamples:  5,
			TotalSamples: 4, // 1 sample per query * 4 steps
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 1,
				206000: 1,
				211000: 1,
				216000: 1,
			},
		},
		{
			Query:        `max_over_time(metricWith3SampleEvery10Seconds{a="1"}[10s])`,
			Start:        time.Unix(991, 0),
			End:          time.Unix(1021, 0),
			Interval:     10 * time.Second,
			PeakSamples:  2,
			TotalSamples: 2, // 1 sample per query * 2 steps with data
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				991000:  1,
				1001000: 1,
				1011000: 0,
				1021000: 0,
			},
		},
		{
			Query:        `metricWith3SampleEvery10Seconds{a="1"} offset 10s`,
			Start:        time.Unix(201, 0),
			End:          time.Unix(220, 0),
			Interval:     5 * time.Second,
			PeakSamples:  4,
			TotalSamples: 4, // 1 sample per query * 4 steps
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 1,
				206000: 1,
				211000: 1,
				216000: 1,
			},
		},
		{
			Query:        "max_over_time(metricWith3SampleEvery10Seconds[60s] @ 30)",
			Start:        time.Unix(201, 0),
			End:          time.Unix(220, 0),
			Interval:     5 * time.Second,
			PeakSamples:  12,
			TotalSamples: 48, // @ modifier force the evaluation timestamp at 30 seconds - So it brings 4 datapoints (0, 10, 20, 30 seconds) * 3 series * 4 steps
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 12,
				206000: 12,
				211000: 12,
				216000: 12,
			},
		},
		{
			Query:        `metricWith3SampleEvery10Seconds`,
			Start:        time.Unix(201, 0),
			End:          time.Unix(220, 0),
			PeakSamples:  12,
			Interval:     5 * time.Second,
			TotalSamples: 12, // 3 sample per query * 4 steps
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 3,
				206000: 3,
				211000: 3,
				216000: 3,
			},
		},
		{
			Query:        `max_over_time(metricWith3SampleEvery10Seconds[60s])`,
			Start:        time.Unix(201, 0),
			End:          time.Unix(220, 0),
			Interval:     5 * time.Second,
			PeakSamples:  18,
			TotalSamples: 72, // (3 sample / 10 seconds * 60 seconds) * 4 steps = 72
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 18,
				206000: 18,
				211000: 18,
				216000: 18,
			},
		},
		{
			Query:        "max_over_time(metricWith3SampleEvery10Seconds[60s:5s])",
			Start:        time.Unix(201, 0),
			End:          time.Unix(220, 0),
			Interval:     5 * time.Second,
			PeakSamples:  72,
			TotalSamples: 144, // 3 sample per query * 12 queries (60/5) * 4 steps
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 36,
				206000: 36,
				211000: 36,
				216000: 36,
			},
		},
		{
			Query:        "max_over_time(metricWith1SampleEvery10Seconds[60s:5s])",
			Start:        time.Unix(201, 0),
			End:          time.Unix(220, 0),
			Interval:     5 * time.Second,
			PeakSamples:  32,
			TotalSamples: 48, // 1 sample per query * 12 queries (60/5) * 4 steps
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 12,
				206000: 12,
				211000: 12,
				216000: 12,
			},
		},
		{
			Query:        "sum by (b) (max_over_time(metricWith1SampleEvery10Seconds[60s:5s]))",
			Start:        time.Unix(201, 0),
			End:          time.Unix(220, 0),
			Interval:     5 * time.Second,
			PeakSamples:  32,
			TotalSamples: 48, // 1 sample per query * 12 queries (60/5) * 4 steps
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 12,
				206000: 12,
				211000: 12,
				216000: 12,
			},
		},
		{
			Query:        "sum(max_over_time(metricWith3SampleEvery10Seconds[60s:5s])) + sum(max_over_time(metricWith3SampleEvery10Seconds[60s:5s]))",
			Start:        time.Unix(201, 0),
			End:          time.Unix(220, 0),
			Interval:     5 * time.Second,
			PeakSamples:  76,
			TotalSamples: 288, // 2 * (3 sample per query * 12 queries (60/5) * 4 steps)
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 72,
				206000: 72,
				211000: 72,
				216000: 72,
			},
		},
		{
			Query:        "sum(max_over_time(metricWith3SampleEvery10Seconds[60s:5s])) + sum(max_over_time(metricWith1SampleEvery10Seconds[60s:5s]))",
			Start:        time.Unix(201, 0),
			End:          time.Unix(220, 0),
			Interval:     5 * time.Second,
			PeakSamples:  72,
			TotalSamples: 192, // (1 sample per query * 12 queries (60/5) + 3 sample per query * 12 queries (60/5)) * 4 steps
			TotalSamplesPerStep: stats.TotalSamplesPerStep{
				201000: 48,
				206000: 48,
				211000: 48,
				216000: 48,
			},
		},
	}

	for _, c := range cases {
		t.Run(c.Query, func(t *testing.T) {
			opts := promql.NewPrometheusQueryOpts(true, 0)
			engine := promqltest.NewTestEngine(true, 0, promqltest.DefaultMaxSamplesPerQuery)

			runQuery := func(expErr error) *stats.Statistics {
				var err error
				var qry promql.Query
				if c.Interval == 0 {
					qry, err = engine.NewInstantQuery(context.Background(), storage, opts, c.Query, c.Start)
				} else {
					qry, err = engine.NewRangeQuery(context.Background(), storage, opts, c.Query, c.Start, c.End, c.Interval)
				}
				require.NoError(t, err)

				res := qry.Exec(context.Background())
				require.Equal(t, expErr, res.Err)

				return qry.Stats()
			}

			stats := runQuery(nil)
			require.Equal(t, c.TotalSamples, stats.Samples.TotalSamples, "Total samples mismatch")
			require.Equal(t, &c.TotalSamplesPerStep, stats.Samples.TotalSamplesPerStepMap(), "Total samples per time mismatch")
			require.Equal(t, c.PeakSamples, stats.Samples.PeakSamples, "Peak samples mismatch")

			// Check that the peak is correct by setting the max to one less.
			if c.SkipMaxCheck {
				return
			}
			engine = promqltest.NewTestEngine(true, 0, stats.Samples.PeakSamples-1)
			runQuery(promql.ErrTooManySamples(env))
		})
	}
}

func TestMaxQuerySamples(t *testing.T) {
	storage := promqltest.LoadedStorage(t, `
load 10s
  metric 1+1x100
  bigmetric{a="1"} 1+1x100
  bigmetric{a="2"} 1+1x100
`)
	t.Cleanup(func() { storage.Close() })

	// These test cases should be touching the limit exactly (hence no exceeding).
	// Exceeding the limit will be tested by doing -1 to the MaxSamples.
	cases := []struct {
		Query      string
		MaxSamples int
		Start      time.Time
		End        time.Time
		Interval   time.Duration
	}{
		// Instant queries.
		{
			Query:      "1",
			MaxSamples: 1,
			Start:      time.Unix(1, 0),
		},
		{
			Query:      "metric",
			MaxSamples: 1,
			Start:      time.Unix(1, 0),
		},
		{
			Query:      "metric[20s]",
			MaxSamples: 2,
			Start:      time.Unix(10, 0),
		},
		{
			Query:      "rate(metric[20s])",
			MaxSamples: 3,
			Start:      time.Unix(10, 0),
		},
		{
			Query:      "metric[20s:5s]",
			MaxSamples: 3,
			Start:      time.Unix(10, 0),
		},
		{
			Query:      "metric[20s] @ 10",
			MaxSamples: 2,
			Start:      time.Unix(0, 0),
		},
		// Range queries.
		{
			Query:      "1",
			MaxSamples: 3,
			Start:      time.Unix(0, 0),
			End:        time.Unix(2, 0),
			Interval:   time.Second,
		},
		{
			Query:      "1",
			MaxSamples: 3,
			Start:      time.Unix(0, 0),
			End:        time.Unix(2, 0),
			Interval:   time.Second,
		},
		{
			Query:      "metric",
			MaxSamples: 3,
			Start:      time.Unix(0, 0),
			End:        time.Unix(2, 0),
			Interval:   time.Second,
		},
		{
			Query:      "metric",
			MaxSamples: 3,
			Start:      time.Unix(0, 0),
			End:        time.Unix(10, 0),
			Interval:   5 * time.Second,
		},
		{
			Query:      "rate(bigmetric[1s])",
			MaxSamples: 1,
			Start:      time.Unix(0, 0),
			End:        time.Unix(10, 0),
			Interval:   5 * time.Second,
		},
		{
			// Result is duplicated, so @ also produces 3 samples.
			Query:      "metric @ 10",
			MaxSamples: 3,
			Start:      time.Unix(0, 0),
			End:        time.Unix(10, 0),
			Interval:   5 * time.Second,
		},
		{
			// The peak samples in memory is during the first evaluation:
			//   - Subquery takes 22 samples, 11 for each bigmetric,
			//   - Result is calculated per series where the series samples is buffered, hence 11 more here.
			//   - The result of two series is added before the last series buffer is discarded, so 2 more here.
			//   Hence at peak it is 22 (subquery) + 11 (buffer of a series) + 2 (result from 2 series).
			// The subquery samples and the buffer is discarded before duplicating.
			Query:      `rate(bigmetric[10s:1s] @ 10)`,
			MaxSamples: 35,
			Start:      time.Unix(0, 0),
			End:        time.Unix(10, 0),
			Interval:   5 * time.Second,
		},
		{
			// Here the reasoning is same as above. But LHS and RHS are done one after another.
			// So while one of them takes 35 samples at peak, we need to hold the 2 sample
			// result of the other till then.
			Query:      `rate(bigmetric[10s:1s] @ 10) + rate(bigmetric[10s:1s] @ 30)`,
			MaxSamples: 37,
			Start:      time.Unix(0, 0),
			End:        time.Unix(10, 0),
			Interval:   5 * time.Second,
		},
		{
			// promql.Sample as above but with only 1 part as step invariant.
			// Here the peak is caused by the non-step invariant part as it touches more time range.
			// Hence at peak it is 2*21 (subquery from 0s to 20s)
			//                     + 11 (buffer of a series per evaluation)
			//                     + 6 (result from 2 series at 3 eval times).
			Query:      `rate(bigmetric[10s:1s]) + rate(bigmetric[10s:1s] @ 30)`,
			MaxSamples: 59,
			Start:      time.Unix(10, 0),
			End:        time.Unix(20, 0),
			Interval:   5 * time.Second,
		},
		{
			// Nested subquery.
			// We saw that innermost rate takes 35 samples which is still the peak
			// since the other two subqueries just duplicate the result.
			Query:      `rate(rate(bigmetric[10s:1s] @ 10)[100s:25s] @ 1000)[100s:20s] @ 2000`,
			MaxSamples: 35,
			Start:      time.Unix(10, 0),
		},
		{
			// Nested subquery.
			// Now the outmost subquery produces more samples than inner most rate.
			Query:      `rate(rate(bigmetric[10s:1s] @ 10)[100s:25s] @ 1000)[17s:1s] @ 2000`,
			MaxSamples: 36,
			Start:      time.Unix(10, 0),
		},
	}

	for _, c := range cases {
		t.Run(c.Query, func(t *testing.T) {
			engine := newTestEngine()
			testFunc := func(expError error) {
				var err error
				var qry promql.Query
				if c.Interval == 0 {
					qry, err = engine.NewInstantQuery(context.Background(), storage, nil, c.Query, c.Start)
				} else {
					qry, err = engine.NewRangeQuery(context.Background(), storage, nil, c.Query, c.Start, c.End, c.Interval)
				}
				require.NoError(t, err)

				res := qry.Exec(context.Background())
				stats := qry.Stats()
				require.Equal(t, expError, res.Err)
				require.NotNil(t, stats)
				if expError == nil {
					require.Equal(t, c.MaxSamples, stats.Samples.PeakSamples, "peak samples mismatch for query %q", c.Query)
				}
			}

			// Within limit.
			engine = promqltest.NewTestEngine(false, 0, c.MaxSamples)
			testFunc(nil)

			// Exceeding limit.
			engine = promqltest.NewTestEngine(false, 0, c.MaxSamples-1)
			testFunc(promql.ErrTooManySamples(env))
		})
	}
}

func TestAtModifier(t *testing.T) {
	engine := newTestEngine()
	storage := promqltest.LoadedStorage(t, `
load 10s
  metric{job="1"} 0+1x1000
  metric{job="2"} 0+2x1000
  metric_topk{instance="1"} 0+1x1000
  metric_topk{instance="2"} 0+2x1000
  metric_topk{instance="3"} 1000-1x1000

load 1ms
  metric_ms 0+1x10000
`)
	t.Cleanup(func() { storage.Close() })

	lbls1 := labels.FromStrings("__name__", "metric", "job", "1")
	lbls2 := labels.FromStrings("__name__", "metric", "job", "2")
	lblstopk2 := labels.FromStrings("__name__", "metric_topk", "instance", "2")
	lblstopk3 := labels.FromStrings("__name__", "metric_topk", "instance", "3")
	lblsms := labels.FromStrings("__name__", "metric_ms")
	lblsneg := labels.FromStrings("__name__", "metric_neg")

	// Add some samples with negative timestamp.
	db := storage.DB
	app := db.Appender(context.Background())
	ref, err := app.Append(0, lblsneg, -1000000, 1000)
	require.NoError(t, err)
	for ts := int64(-1000000 + 1000); ts <= 0; ts += 1000 {
		_, err := app.Append(ref, labels.EmptyLabels(), ts, -float64(ts/1000)+1)
		require.NoError(t, err)
	}

	// To test the fix for https://github.com/prometheus/prometheus/issues/8433.
	_, err = app.Append(0, labels.FromStrings("__name__", "metric_timestamp"), 3600*1000, 1000)
	require.NoError(t, err)

	require.NoError(t, app.Commit())

	cases := []struct {
		query                string
		start, end, interval int64 // Time in seconds.
		result               parser.Value
	}{
		{ // Time of the result is the evaluation time.
			query: `metric_neg @ 0`,
			start: 100,
			result: promql.Vector{
				promql.Sample{F: 1, T: 100000, Metric: lblsneg},
			},
		}, {
			query: `metric_neg @ -200`,
			start: 100,
			result: promql.Vector{
				promql.Sample{F: 201, T: 100000, Metric: lblsneg},
			},
		}, {
			query: `metric{job="2"} @ 50`,
			start: -2, end: 2, interval: 1,
			result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{{F: 10, T: -2000}, {F: 10, T: -1000}, {F: 10, T: 0}, {F: 10, T: 1000}, {F: 10, T: 2000}},
					Metric: lbls2,
				},
			},
		}, { // Timestamps for matrix selector does not depend on the evaluation time.
			query: "metric[20s] @ 300",
			start: 10,
			result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{{F: 28, T: 280000}, {F: 29, T: 290000}, {F: 30, T: 300000}},
					Metric: lbls1,
				},
				promql.Series{
					Floats: []promql.FPoint{{F: 56, T: 280000}, {F: 58, T: 290000}, {F: 60, T: 300000}},
					Metric: lbls2,
				},
			},
		}, {
			query: `metric_neg[2s] @ 0`,
			start: 100,
			result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{{F: 3, T: -2000}, {F: 2, T: -1000}, {F: 1, T: 0}},
					Metric: lblsneg,
				},
			},
		}, {
			query: `metric_neg[3s] @ -500`,
			start: 100,
			result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{{F: 504, T: -503000}, {F: 503, T: -502000}, {F: 502, T: -501000}, {F: 501, T: -500000}},
					Metric: lblsneg,
				},
			},
		}, {
			query: `metric_ms[3ms] @ 2.345`,
			start: 100,
			result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{{F: 2342, T: 2342}, {F: 2343, T: 2343}, {F: 2344, T: 2344}, {F: 2345, T: 2345}},
					Metric: lblsms,
				},
			},
		}, {
			query: "metric[100s:25s] @ 300",
			start: 100,
			result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{{F: 20, T: 200000}, {F: 22, T: 225000}, {F: 25, T: 250000}, {F: 27, T: 275000}, {F: 30, T: 300000}},
					Metric: lbls1,
				},
				promql.Series{
					Floats: []promql.FPoint{{F: 40, T: 200000}, {F: 44, T: 225000}, {F: 50, T: 250000}, {F: 54, T: 275000}, {F: 60, T: 300000}},
					Metric: lbls2,
				},
			},
		}, {
			query: "metric_neg[50s:25s] @ 0",
			start: 100,
			result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{{F: 51, T: -50000}, {F: 26, T: -25000}, {F: 1, T: 0}},
					Metric: lblsneg,
				},
			},
		}, {
			query: "metric_neg[50s:25s] @ -100",
			start: 100,
			result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{{F: 151, T: -150000}, {F: 126, T: -125000}, {F: 101, T: -100000}},
					Metric: lblsneg,
				},
			},
		}, {
			query: `metric_ms[100ms:25ms] @ 2.345`,
			start: 100,
			result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{{F: 2250, T: 2250}, {F: 2275, T: 2275}, {F: 2300, T: 2300}, {F: 2325, T: 2325}},
					Metric: lblsms,
				},
			},
		}, {
			query: `metric_topk and topk(1, sum_over_time(metric_topk[50s] @ 100))`,
			start: 50, end: 80, interval: 10,
			result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{{F: 995, T: 50000}, {F: 994, T: 60000}, {F: 993, T: 70000}, {F: 992, T: 80000}},
					Metric: lblstopk3,
				},
			},
		}, {
			query: `metric_topk and topk(1, sum_over_time(metric_topk[50s] @ 5000))`,
			start: 50, end: 80, interval: 10,
			result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{{F: 10, T: 50000}, {F: 12, T: 60000}, {F: 14, T: 70000}, {F: 16, T: 80000}},
					Metric: lblstopk2,
				},
			},
		}, {
			query: `metric_topk and topk(1, sum_over_time(metric_topk[50s] @ end()))`,
			start: 70, end: 100, interval: 10,
			result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{{F: 993, T: 70000}, {F: 992, T: 80000}, {F: 991, T: 90000}, {F: 990, T: 100000}},
					Metric: lblstopk3,
				},
			},
		}, {
			query: `metric_topk and topk(1, sum_over_time(metric_topk[50s] @ start()))`,
			start: 100, end: 130, interval: 10,
			result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{{F: 990, T: 100000}, {F: 989, T: 110000}, {F: 988, T: 120000}, {F: 987, T: 130000}},
					Metric: lblstopk3,
				},
			},
		}, {
			// Tests for https://github.com/prometheus/prometheus/issues/8433.
			// The trick here is that the query range should be > lookback delta.
			query: `timestamp(metric_timestamp @ 3600)`,
			start: 0, end: 7 * 60, interval: 60,
			result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{
						{F: 3600, T: 0},
						{F: 3600, T: 60 * 1000},
						{F: 3600, T: 2 * 60 * 1000},
						{F: 3600, T: 3 * 60 * 1000},
						{F: 3600, T: 4 * 60 * 1000},
						{F: 3600, T: 5 * 60 * 1000},
						{F: 3600, T: 6 * 60 * 1000},
						{F: 3600, T: 7 * 60 * 1000},
					},
					Metric: labels.EmptyLabels(),
				},
			},
		},
	}

	for _, c := range cases {
		t.Run(c.query, func(t *testing.T) {
			if c.interval == 0 {
				c.interval = 1
			}
			start, end, interval := time.Unix(c.start, 0), time.Unix(c.end, 0), time.Duration(c.interval)*time.Second
			var err error
			var qry promql.Query
			if c.end == 0 {
				qry, err = engine.NewInstantQuery(context.Background(), storage, nil, c.query, start)
			} else {
				qry, err = engine.NewRangeQuery(context.Background(), storage, nil, c.query, start, end, interval)
			}
			require.NoError(t, err)

			res := qry.Exec(context.Background())
			require.NoError(t, res.Err)
			if expMat, ok := c.result.(promql.Matrix); ok {
				sort.Sort(expMat)
				sort.Sort(res.Value.(promql.Matrix))
			}
			testutil.RequireEqual(t, c.result, res.Value, "query %q failed", c.query)
		})
	}
}

// Convenience function that returns a "full matrix" for the specified number
// of jobs and time range.
func fullMatrix(jobs, start, end, interval, increase int) (promql.Matrix, []labels.Labels) {
	var mat promql.Matrix
	var lbls []labels.Labels
	for job := 1; job <= jobs; job++ {
		lbls = append(lbls, labels.FromStrings("__name__", "metric", "job", strconv.Itoa(job)))
		series := func(start, end, interval int) promql.Series {
			var points []promql.FPoint
			for i := start; i <= end; i += interval {
				t := int64(increase) * int64(i)
				points = append(points, promql.FPoint{F: float64(job * i / interval), T: t})
			}
			return promql.Series{
				Floats: points,
				Metric: labels.FromStrings("__name__", "metric", "job", strconv.Itoa(job)),
			}
		}(start, end, interval)
		mat = append(mat, series)
	}
	return mat, lbls
}

func TestLimitK(t *testing.T) {
	engine := newTestEngine()
	storage := promqltest.LoadedStorage(t, `
load 10s
  metric{job="1"} 0+1x1000
  metric{job="2"} 0+2x1000
  metric{job="3"} 0+3x1000
  metric{job="4"} 0+4x1000
  metric{job="5"} 0+5x1000
`)
	t.Cleanup(func() { storage.Close() })

	fullMatrix20, _ := fullMatrix(5, 0, 20, 10, 1000)
	cases := []struct {
		query                string
		start, end, interval int64 // Time in seconds.
		expectError          bool
		result               parser.Value
		resultIn             parser.Value
		resultLen            int // Required if resultIn is set
	}{
		{
			// Limit==0 -> empty matrix
			query: `limitk(0, metric)`,
			start: 0, end: 20, interval: 10,
			result: promql.Matrix(nil),
		},
		{
			// Limit==2 -> return 2 timeseries (resultLen: 2),
			// also asserting that they are member of full TSs
			query: `limitk(2, metric)`,
			start: 0, end: 20, interval: 10,
			resultLen: 2,
			resultIn:  fullMatrix20,
		},
		{
			// Limit==5 -> return full matrix (5 timeseries)
			query: `limitk(5, metric)`,
			start: 0, end: 20, interval: 10,
			result: fullMatrix20,
		},
		{
			// Limit==10 (>5 timeseries) -> still return full matrix (5 timeseries)
			query: `limitk(10, metric)`,
			start: 0, end: 20, interval: 10,
			result: fullMatrix20,
		},
		{
			// Limit <0 -> return empty matrix
			query: `limitk(10, metric)`,
			start: 0, end: 20, interval: 10,
		},
		{
			// Limit==<over maxInt64> -> error
			query: fmt.Sprintf(`limitk(%d, metric)`, maxInt64+1000),
			start: 0, end: 20, interval: 10,
			expectError: true,
		},
	}
	for _, c := range cases {
		t.Run(c.query, func(t *testing.T) {
			if c.interval == 0 {
				c.interval = 1
			}
			start, end, interval := time.Unix(c.start, 0), time.Unix(c.end, 0), time.Duration(c.interval)*time.Second
			var err error
			var qry promql.Query
			qry, err = engine.NewRangeQuery(context.Background(), storage, nil, c.query, start, end, interval)
			require.NoError(t, err)

			res := qry.Exec(context.Background())
			if c.expectError {
				require.Error(t, res.Err)
				return
			}
			require.NoError(t, res.Err)
			switch {
			case c.result != nil:
				if expMat, ok := c.result.(promql.Matrix); ok {
					sort.Sort(expMat)
					sort.Sort(res.Value.(promql.Matrix))
				}
				require.Equal(t, c.result, res.Value, "query %q failed for require.Equal()", c.query)
			case c.resultIn != nil:
				require.Len(t, res.Value.(promql.Matrix), c.resultLen, "query %q failed", c.query)
				if expMat, ok := c.resultIn.(promql.Matrix); ok {
					sort.Sort(expMat)
					sort.Sort(res.Value.(promql.Matrix))
					for _, e := range res.Value.(promql.Matrix) {
						require.Contains(t, c.resultIn, e, "query %q failed for require.Contains()", c.query)
					}
				}
			}
		})
	}
}

func TestLimitRatio(t *testing.T) {
	engine := newTestEngine()
	storage := promqltest.LoadedStorage(t, `
load 10s
  metric{job="1"} 0+1x1000
  metric{job="2"} 0+2x1000
  metric{job="3"} 0+3x1000
  metric{job="4"} 0+4x1000
  metric{job="5"} 0+5x1000
`)
	t.Cleanup(func() { storage.Close() })

	fullMatrix20, lbls := fullMatrix(5, 0, 20, 10, 1000)
	cases := []struct {
		query                string
		start, end, interval int64 // Time in seconds.
		result               parser.Value
		expectError          bool
		expectWarn           bool
	}{
		{
			// ratioLimit=0 -> empty matrix
			query: `limit_ratio(0, metric)`,
			start: 0, end: 20, interval: 10,
			result: promql.Matrix(nil),
		},
		{
			// ratioLimit=0.2 -> return some timeseries
			// NB: given that involves Metric.Hash() this was _manually_ cherry-picked to match
			query: `limit_ratio(0.2, metric)`,
			start: 0, end: 20, interval: 10,
			result: promql.Matrix{
				promql.Series{
					Metric: lbls[0],
					Floats: []promql.FPoint{{T: 0, F: 0}, {T: 10000, F: 1}, {T: 20000, F: 2}},
				},
			},
		},
		{
			// ratioLimit=-0.8 -> the _completement_ of limit_ratio(0.2, metric)
			query: `limit_ratio(-0.8, metric)`,
			start: 0, end: 20, interval: 10,
			result: promql.Matrix{
				promql.Series{
					Floats: []promql.FPoint{{T: 0, F: 0}, {T: 10000, F: 2}, {T: 20000, F: 4}},
					Metric: lbls[1],
				},
				promql.Series{
					Floats: []promql.FPoint{{T: 0, F: 0}, {T: 10000, F: 3}, {T: 20000, F: 6}},
					Metric: lbls[2],
				},
				promql.Series{
					Floats: []promql.FPoint{{T: 0, F: 0}, {T: 10000, F: 4}, {T: 20000, F: 8}},
					Metric: lbls[3],
				},
				promql.Series{
					Floats: []promql.FPoint{{F: 0, T: 0}, {T: 10000, F: 5}, {T: 20000, F: 10}},
					Metric: lbls[4],
				},
			},
		},
		{
			// ratioLimit=1.0 -> full matrix
			query: `limit_ratio(1.0, metric)`,
			start: 0, end: 20, interval: 10,
			result: fullMatrix20,
		},
		{
			// ratioLimit=-1.0 -> full matrix also
			query: `limit_ratio(-1.0, metric)`,
			start: 0, end: 20, interval: 10,
			result: fullMatrix20,
		},
		{
			// Treat ratioLimit > 1.0 as 1.0
			query: `limit_ratio(1.01, metric)`,
			start: 0, end: 20, interval: 10,
			result:     fullMatrix20,
			expectWarn: true,
		},
		{
			// Treat ratioLimit < 1.0 as -1.0
			query: `limit_ratio(-1.01, metric)`,
			start: 0, end: 20, interval: 10,
			result:     fullMatrix20,
			expectWarn: true,
		},
		{
			// Handle NaN
			query: `limit_ratio(1.0.1, metric)`,
			start: 0, end: 20, interval: 10,
			expectError: true,
		},
	}

	for _, c := range cases {
		t.Run(c.query, func(t *testing.T) {
			if c.interval == 0 {
				c.interval = 1
			}
			start, end, interval := time.Unix(c.start, 0), time.Unix(c.end, 0), time.Duration(c.interval)*time.Second
			var err error
			var qry promql.Query
			qry, err = engine.NewRangeQuery(context.Background(), storage, nil, c.query, start, end, interval)
			if c.expectError {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)

			res := qry.Exec(context.Background())
			require.NoError(t, res.Err)
			if c.expectWarn {
				require.NotEmpty(t, res.Warnings)
			} else {
				require.Empty(t, res.Warnings)
			}
			if expMat, ok := c.result.(promql.Matrix); ok {
				sort.Sort(expMat)
				sort.Sort(res.Value.(promql.Matrix))
			}
			require.Equal(t, c.result, res.Value, "query %q failed for require.Equal()", c.query)
		})
	}
}

func TestLimitRatioDynamic(t *testing.T) {
	engine := newTestEngine()
	storage := promqltest.LoadedStorage(t, `
load 10s
  metric{job="1"} 0+1x1000
  metric{job="2"} 0+2x1000
  metric{job="3"} 0+3x1000
  metric{job="4"} 0+4x1000
  metric{job="5"} 0+5x1000
`)
	t.Cleanup(func() { storage.Close() })

	fullMatrix20, _ := fullMatrix(5, 0, 20, 10, 1000)

	// Dynamically build limit_ratio() cases, to verify that
	//   limit_ratio(v, metric)
	// is the *exact complement* of
	//   limit_ratio(-(1.0-v), metric)
	//
	// e.g. v=0.2 is the complement (set of timeseries) of v=-0.8
	type ratioCase struct {
		query                string
		start, end, interval int64 // Time in seconds.
		result               parser.Value
	}
	orQuery := `limit_ratio(%f, metric) or limit_ratio(%f, metric)`
	andQuery := `limit_ratio(%f, metric) and limit_ratio(%f, metric)`
	var ratioCases []ratioCase
	for v := 0.0; v < 1.0; v += 0.1 {
		// OR-ing both must return the full matrix
		ratioCases = append(ratioCases, ratioCase{
			query: fmt.Sprintf(orQuery, v, -(1.0 - v)),
			start: 0, end: 20, interval: 10,
			result: fullMatrix20,
		})
		// AND-ing both must return an empty matrix
		ratioCases = append(ratioCases, ratioCase{
			query: fmt.Sprintf(andQuery, v, -(1.0 - v)),
			start: 0, end: 20, interval: 10,
			result: promql.Matrix{},
		})
	}

	for _, c := range ratioCases {
		t.Run(c.query, func(t *testing.T) {
			if c.interval == 0 {
				c.interval = 1
			}
			start, end, interval := time.Unix(c.start, 0), time.Unix(c.end, 0), time.Duration(c.interval)*time.Second
			var err error
			var qry promql.Query
			qry, err = engine.NewRangeQuery(context.Background(), storage, nil, c.query, start, end, interval)
			require.NoError(t, err)

			res := qry.Exec(context.Background())
			require.NoError(t, res.Err)
			if expMat, ok := c.result.(promql.Matrix); ok {
				sort.Sort(expMat)
				sort.Sort(res.Value.(promql.Matrix))
			}
			require.Equal(t, c.result, res.Value, "query %q failed", c.query)
		})
	}
}

func TestSubquerySelector(t *testing.T) {
	type caseType struct {
		Query  string
		Result promql.Result
		Start  time.Time
	}

	for _, tst := range []struct {
		loadString string
		cases      []caseType
	}{
		{
			loadString: `load 10s
							metric 1 2`,
			cases: []caseType{
				{
					Query: "metric[20s:10s]",
					Result: promql.Result{
						nil,
						promql.Matrix{
							promql.Series{
								Floats: []promql.FPoint{{F: 1, T: 0}, {F: 2, T: 10000}},
								Metric: labels.FromStrings("__name__", "metric"),
							},
						},
						nil,
					},
					Start: time.Unix(10, 0),
				},
				{
					Query: "metric[20s:5s]",
					Result: promql.Result{
						nil,
						promql.Matrix{
							promql.Series{
								Floats: []promql.FPoint{{F: 1, T: 0}, {F: 1, T: 5000}, {F: 2, T: 10000}},
								Metric: labels.FromStrings("__name__", "metric"),
							},
						},
						nil,
					},
					Start: time.Unix(10, 0),
				},
				{
					Query: "metric[20s:5s] offset 2s",
					Result: promql.Result{
						nil,
						promql.Matrix{
							promql.Series{
								Floats: []promql.FPoint{{F: 1, T: 0}, {F: 1, T: 5000}, {F: 2, T: 10000}},
								Metric: labels.FromStrings("__name__", "metric"),
							},
						},
						nil,
					},
					Start: time.Unix(12, 0),
				},
				{
					Query: "metric[20s:5s] offset 6s",
					Result: promql.Result{
						nil,
						promql.Matrix{
							promql.Series{
								Floats: []promql.FPoint{{F: 1, T: 0}, {F: 1, T: 5000}, {F: 2, T: 10000}},
								Metric: labels.FromStrings("__name__", "metric"),
							},
						},
						nil,
					},
					Start: time.Unix(20, 0),
				},
				{
					Query: "metric[20s:5s] offset 4s",
					Result: promql.Result{
						nil,
						promql.Matrix{
							promql.Series{
								Floats: []promql.FPoint{{F: 2, T: 15000}, {F: 2, T: 20000}, {F: 2, T: 25000}, {F: 2, T: 30000}},
								Metric: labels.FromStrings("__name__", "metric"),
							},
						},
						nil,
					},
					Start: time.Unix(35, 0),
				},
				{
					Query: "metric[20s:5s] offset 5s",
					Result: promql.Result{
						nil,
						promql.Matrix{
							promql.Series{
								Floats: []promql.FPoint{{F: 2, T: 10000}, {F: 2, T: 15000}, {F: 2, T: 20000}, {F: 2, T: 25000}, {F: 2, T: 30000}},
								Metric: labels.FromStrings("__name__", "metric"),
							},
						},
						nil,
					},
					Start: time.Unix(35, 0),
				},
				{
					Query: "metric[20s:5s] offset 6s",
					Result: promql.Result{
						nil,
						promql.Matrix{
							promql.Series{
								Floats: []promql.FPoint{{F: 2, T: 10000}, {F: 2, T: 15000}, {F: 2, T: 20000}, {F: 2, T: 25000}},
								Metric: labels.FromStrings("__name__", "metric"),
							},
						},
						nil,
					},
					Start: time.Unix(35, 0),
				},
				{
					Query: "metric[20s:5s] offset 7s",
					Result: promql.Result{
						nil,
						promql.Matrix{
							promql.Series{
								Floats: []promql.FPoint{{F: 2, T: 10000}, {F: 2, T: 15000}, {F: 2, T: 20000}, {F: 2, T: 25000}},
								Metric: labels.FromStrings("__name__", "metric"),
							},
						},
						nil,
					},
					Start: time.Unix(35, 0),
				},
			},
		},
		{
			loadString: `load 10s
							http_requests{job="api-server", instance="0", group="production"}	0+10x1000 100+30x1000
							http_requests{job="api-server", instance="1", group="production"}	0+20x1000 200+30x1000
							http_requests{job="api-server", instance="0", group="canary"}		0+30x1000 300+80x1000
							http_requests{job="api-server", instance="1", group="canary"}		0+40x2000`,
			cases: []caseType{
				{ // Normal selector.
					Query: `http_requests{group=~"pro.*",instance="0"}[30s:10s]`,
					Result: promql.Result{
						nil,
						promql.Matrix{
							promql.Series{
								Floats: []promql.FPoint{{F: 9990, T: 9990000}, {F: 10000, T: 10000000}, {F: 100, T: 10010000}, {F: 130, T: 10020000}},
								Metric: labels.FromStrings("__name__", "http_requests", "job", "api-server", "instance", "0", "group", "production"),
							},
						},
						nil,
					},
					Start: time.Unix(10020, 0),
				},
				{ // Default step.
					Query: `http_requests{group=~"pro.*",instance="0"}[5m:]`,
					Result: promql.Result{
						nil,
						promql.Matrix{
							promql.Series{
								Floats: []promql.FPoint{{F: 9840, T: 9840000}, {F: 9900, T: 9900000}, {F: 9960, T: 9960000}, {F: 130, T: 10020000}, {F: 310, T: 10080000}},
								Metric: labels.FromStrings("__name__", "http_requests", "job", "api-server", "instance", "0", "group", "production"),
							},
						},
						nil,
					},
					Start: time.Unix(10100, 0),
				},
				{ // Checking if high offset (>LookbackDelta) is being taken care of.
					Query: `http_requests{group=~"pro.*",instance="0"}[5m:] offset 20m`,
					Result: promql.Result{
						nil,
						promql.Matrix{
							promql.Series{
								Floats: []promql.FPoint{{F: 8640, T: 8640000}, {F: 8700, T: 8700000}, {F: 8760, T: 8760000}, {F: 8820, T: 8820000}, {F: 8880, T: 8880000}},
								Metric: labels.FromStrings("__name__", "http_requests", "job", "api-server", "instance", "0", "group", "production"),
							},
						},
						nil,
					},
					Start: time.Unix(10100, 0),
				},
				{
					Query: `rate(http_requests[1m])[15s:5s]`,
					Result: promql.Result{
						nil,
						promql.Matrix{
							promql.Series{
								Floats: []promql.FPoint{{F: 3, T: 7985000}, {F: 3, T: 7990000}, {F: 3, T: 7995000}, {F: 3, T: 8000000}},
								Metric: labels.FromStrings("job", "api-server", "instance", "0", "group", "canary"),
							},
							promql.Series{
								Floats: []promql.FPoint{{F: 4, T: 7985000}, {F: 4, T: 7990000}, {F: 4, T: 7995000}, {F: 4, T: 8000000}},
								Metric: labels.FromStrings("job", "api-server", "instance", "1", "group", "canary"),
							},
							promql.Series{
								Floats: []promql.FPoint{{F: 1, T: 7985000}, {F: 1, T: 7990000}, {F: 1, T: 7995000}, {F: 1, T: 8000000}},
								Metric: labels.FromStrings("job", "api-server", "instance", "0", "group", "production"),
							},
							promql.Series{
								Floats: []promql.FPoint{{F: 2, T: 7985000}, {F: 2, T: 7990000}, {F: 2, T: 7995000}, {F: 2, T: 8000000}},
								Metric: labels.FromStrings("job", "api-server", "instance", "1", "group", "production"),
							},
						},
						nil,
					},
					Start: time.Unix(8000, 0),
				},
				{
					Query: `sum(http_requests{group=~"pro.*"})[30s:10s]`,
					Result: promql.Result{
						nil,
						promql.Matrix{
							promql.Series{
								Floats: []promql.FPoint{{F: 270, T: 90000}, {F: 300, T: 100000}, {F: 330, T: 110000}, {F: 360, T: 120000}},
								Metric: labels.EmptyLabels(),
							},
						},
						nil,
					},
					Start: time.Unix(120, 0),
				},
				{
					Query: `sum(http_requests)[40s:10s]`,
					Result: promql.Result{
						nil,
						promql.Matrix{
							promql.Series{
								Floats: []promql.FPoint{{F: 800, T: 80000}, {F: 900, T: 90000}, {F: 1000, T: 100000}, {F: 1100, T: 110000}, {F: 1200, T: 120000}},
								Metric: labels.EmptyLabels(),
							},
						},
						nil,
					},
					Start: time.Unix(120, 0),
				},
				{
					Query: `(sum(http_requests{group=~"p.*"})+sum(http_requests{group=~"c.*"}))[20s:5s]`,
					Result: promql.Result{
						nil,
						promql.Matrix{
							promql.Series{
								Floats: []promql.FPoint{{F: 1000, T: 100000}, {F: 1000, T: 105000}, {F: 1100, T: 110000}, {F: 1100, T: 115000}, {F: 1200, T: 120000}},
								Metric: labels.EmptyLabels(),
							},
						},
						nil,
					},
					Start: time.Unix(120, 0),
				},
			},
		},
	} {
		t.Run("", func(t *testing.T) {
			engine := newTestEngine()
			storage := promqltest.LoadedStorage(t, tst.loadString)
			t.Cleanup(func() { storage.Close() })

			for _, c := range tst.cases {
				t.Run(c.Query, func(t *testing.T) {
					qry, err := engine.NewInstantQuery(context.Background(), storage, nil, c.Query, c.Start)
					require.NoError(t, err)

					res := qry.Exec(context.Background())
					require.Equal(t, c.Result.Err, res.Err)
					mat := res.Value.(promql.Matrix)
					sort.Sort(mat)
					testutil.RequireEqual(t, c.Result.Value, mat)
				})
			}
		})
	}
}

type FakeQueryLogger struct {
	closed bool
	logs   []interface{}
}

func NewFakeQueryLogger() *FakeQueryLogger {
	return &FakeQueryLogger{
		closed: false,
		logs:   make([]interface{}, 0),
	}
}

func (f *FakeQueryLogger) Close() error {
	f.closed = true
	return nil
}

func (f *FakeQueryLogger) Log(l ...interface{}) error {
	f.logs = append(f.logs, l...)
	return nil
}

func TestQueryLogger_basic(t *testing.T) {
	opts := promql.EngineOpts{
		Logger:     nil,
		Reg:        nil,
		MaxSamples: 10,
		Timeout:    10 * time.Second,
	}
	engine := promql.NewEngine(opts)

	queryExec := func() {
		ctx, cancelCtx := context.WithCancel(context.Background())
		defer cancelCtx()
		query := engine.NewTestQuery(func(ctx context.Context) error {
			return contextDone(ctx, "test statement execution")
		})
		res := query.Exec(ctx)
		require.NoError(t, res.Err)
	}

	// promql.Query works without query log initialized.
	queryExec()

	f1 := NewFakeQueryLogger()
	engine.SetQueryLogger(f1)
	queryExec()
	for i, field := range []interface{}{"params", map[string]interface{}{"query": "test statement"}} {
		require.Equal(t, field, f1.logs[i])
	}

	l := len(f1.logs)
	queryExec()
	require.Len(t, f1.logs, 2*l)

	// Test that we close the query logger when unsetting it.
	require.False(t, f1.closed, "expected f1 to be open, got closed")
	engine.SetQueryLogger(nil)
	require.True(t, f1.closed, "expected f1 to be closed, got open")
	queryExec()

	// Test that we close the query logger when swapping.
	f2 := NewFakeQueryLogger()
	f3 := NewFakeQueryLogger()
	engine.SetQueryLogger(f2)
	require.False(t, f2.closed, "expected f2 to be open, got closed")
	queryExec()
	engine.SetQueryLogger(f3)
	require.True(t, f2.closed, "expected f2 to be closed, got open")
	require.False(t, f3.closed, "expected f3 to be open, got closed")
	queryExec()
}

func TestQueryLogger_fields(t *testing.T) {
	opts := promql.EngineOpts{
		Logger:     nil,
		Reg:        nil,
		MaxSamples: 10,
		Timeout:    10 * time.Second,
	}
	engine := promql.NewEngine(opts)

	f1 := NewFakeQueryLogger()
	engine.SetQueryLogger(f1)

	ctx, cancelCtx := context.WithCancel(context.Background())
	ctx = promql.NewOriginContext(ctx, map[string]interface{}{"foo": "bar"})
	defer cancelCtx()
	query := engine.NewTestQuery(func(ctx context.Context) error {
		return contextDone(ctx, "test statement execution")
	})

	res := query.Exec(ctx)
	require.NoError(t, res.Err)

	expected := []string{"foo", "bar"}
	for i, field := range expected {
		v := f1.logs[len(f1.logs)-len(expected)+i].(string)
		require.Equal(t, field, v)
	}
}

func TestQueryLogger_error(t *testing.T) {
	opts := promql.EngineOpts{
		Logger:     nil,
		Reg:        nil,
		MaxSamples: 10,
		Timeout:    10 * time.Second,
	}
	engine := promql.NewEngine(opts)

	f1 := NewFakeQueryLogger()
	engine.SetQueryLogger(f1)

	ctx, cancelCtx := context.WithCancel(context.Background())
	ctx = promql.NewOriginContext(ctx, map[string]interface{}{"foo": "bar"})
	defer cancelCtx()
	testErr := errors.New("failure")
	query := engine.NewTestQuery(func(ctx context.Context) error {
		return testErr
	})

	res := query.Exec(ctx)
	require.Error(t, res.Err, "query should have failed")

	for i, field := range []interface{}{"params", map[string]interface{}{"query": "test statement"}, "error", testErr} {
		require.Equal(t, f1.logs[i], field)
	}
}

func TestPreprocessAndWrapWithStepInvariantExpr(t *testing.T) {
	startTime := time.Unix(1000, 0)
	endTime := time.Unix(9999, 0)
	testCases := []struct {
		input      string      // The input to be parsed.
		expected   parser.Expr // The expected expression AST.
		outputTest bool
	}{
		{
			input: "123.4567",
			expected: &parser.StepInvariantExpr{
				Expr: &parser.NumberLiteral{
					Val:      123.4567,
					PosRange: posrange.PositionRange{Start: 0, End: 8},
				},
			},
		},
		{
			input: `"foo"`,
			expected: &parser.StepInvariantExpr{
				Expr: &parser.StringLiteral{
					Val:      "foo",
					PosRange: posrange.PositionRange{Start: 0, End: 5},
				},
			},
		},
		{
			input: "foo * bar",
			expected: &parser.BinaryExpr{
				Op: parser.MUL,
				LHS: &parser.VectorSelector{
					Name: "foo",
					LabelMatchers: []*labels.Matcher{
						parser.MustLabelMatcher(labels.MatchEqual, "__name__", "foo"),
					},
					PosRange: posrange.PositionRange{
						Start: 0,
						End:   3,
					},
				},
				RHS: &parser.VectorSelector{
					Name: "bar",
					LabelMatchers: []*labels.Matcher{
						parser.MustLabelMatcher(labels.MatchEqual, "__name__", "bar"),
					},
					PosRange: posrange.PositionRange{
						Start: 6,
						End:   9,
					},
				},
				VectorMatching: &parser.VectorMatching{Card: parser.CardOneToOne},
			},
		},
		{
			input: "foo * bar @ 10",
			expected: &parser.BinaryExpr{
				Op: parser.MUL,
				LHS: &parser.VectorSelector{
					Name: "foo",
					LabelMatchers: []*labels.Matcher{
						parser.MustLabelMatcher(labels.MatchEqual, "__name__", "foo"),
					},
					PosRange: posrange.PositionRange{
						Start: 0,
						End:   3,
					},
				},
				RHS: &parser.StepInvariantExpr{
					Expr: &parser.VectorSelector{
						Name: "bar",
						LabelMatchers: []*labels.Matcher{
							parser.MustLabelMatcher(labels.MatchEqual, "__name__", "bar"),
						},
						PosRange: posrange.PositionRange{
							Start: 6,
							End:   14,
						},
						Timestamp: makeInt64Pointer(10000),
					},
				},
				VectorMatching: &parser.VectorMatching{Card: parser.CardOneToOne},
			},
		},
		{
			input: "foo @ 20 * bar @ 10",
			expected: &parser.StepInvariantExpr{
				Expr: &parser.BinaryExpr{
					Op: parser.MUL,
					LHS: &parser.VectorSelector{
						Name: "foo",
						LabelMatchers: []*labels.Matcher{
							parser.MustLabelMatcher(labels.MatchEqual, "__name__", "foo"),
						},
						PosRange: posrange.PositionRange{
							Start: 0,
							End:   8,
						},
						Timestamp: makeInt64Pointer(20000),
					},
					RHS: &parser.VectorSelector{
						Name: "bar",
						LabelMatchers: []*labels.Matcher{
							parser.MustLabelMatcher(labels.MatchEqual, "__name__", "bar"),
						},
						PosRange: posrange.PositionRange{
							Start: 11,
							End:   19,
						},
						Timestamp: makeInt64Pointer(10000),
					},
					VectorMatching: &parser.VectorMatching{Card: parser.CardOneToOne},
				},
			},
		},
		{
			input: "test[5s]",
			expected: &parser.MatrixSelector{
				VectorSelector: &parser.VectorSelector{
					Name: "test",
					LabelMatchers: []*labels.Matcher{
						parser.MustLabelMatcher(labels.MatchEqual, "__name__", "test"),
					},
					PosRange: posrange.PositionRange{
						Start: 0,
						End:   4,
					},
				},
				Range:  5 * time.Second,
				EndPos: 8,
			},
		},
		{
			input: `test{a="b"}[5y] @ 1603774699`,
			expected: &parser.StepInvariantExpr{
				Expr: &parser.MatrixSelector{
					VectorSelector: &parser.VectorSelector{
						Name:      "test",
						Timestamp: makeInt64Pointer(1603774699000),
						LabelMatchers: []*labels.Matcher{
							parser.MustLabelMatcher(labels.MatchEqual, "a", "b"),
							parser.MustLabelMatcher(labels.MatchEqual, "__name__", "test"),
						},
						PosRange: posrange.PositionRange{
							Start: 0,
							End:   11,
						},
					},
					Range:  5 * 365 * 24 * time.Hour,
					EndPos: 28,
				},
			},
		},
		{
			input: "sum by (foo)(some_metric)",
			expected: &parser.AggregateExpr{
				Op: parser.SUM,
				Expr: &parser.VectorSelector{
					Name: "some_metric",
					LabelMatchers: []*labels.Matcher{
						parser.MustLabelMatcher(labels.MatchEqual, "__name__", "some_metric"),
					},
					PosRange: posrange.PositionRange{
						Start: 13,
						End:   24,
					},
				},
				Grouping: []string{"foo"},
				PosRange: posrange.PositionRange{
					Start: 0,
					End:   25,
				},
			},
		},
		{
			input: "sum by (foo)(some_metric @ 10)",
			expected: &parser.StepInvariantExpr{
				Expr: &parser.AggregateExpr{
					Op: parser.SUM,
					Expr: &parser.VectorSelector{
						Name: "some_metric",
						LabelMatchers: []*labels.Matcher{
							parser.MustLabelMatcher(labels.MatchEqual, "__name__", "some_metric"),
						},
						PosRange: posrange.PositionRange{
							Start: 13,
							End:   29,
						},
						Timestamp: makeInt64Pointer(10000),
					},
					Grouping: []string{"foo"},
					PosRange: posrange.PositionRange{
						Start: 0,
						End:   30,
					},
				},
			},
		},
		{
			input: "sum(some_metric1 @ 10) + sum(some_metric2 @ 20)",
			expected: &parser.StepInvariantExpr{
				Expr: &parser.BinaryExpr{
					Op:             parser.ADD,
					VectorMatching: &parser.VectorMatching{},
					LHS: &parser.AggregateExpr{
						Op: parser.SUM,
						Expr: &parser.VectorSelector{
							Name: "some_metric1",
							LabelMatchers: []*labels.Matcher{
								parser.MustLabelMatcher(labels.MatchEqual, "__name__", "some_metric1"),
							},
							PosRange: posrange.PositionRange{
								Start: 4,
								End:   21,
							},
							Timestamp: makeInt64Pointer(10000),
						},
						PosRange: posrange.PositionRange{
							Start: 0,
							End:   22,
						},
					},
					RHS: &parser.AggregateExpr{
						Op: parser.SUM,
						Expr: &parser.VectorSelector{
							Name: "some_metric2",
							LabelMatchers: []*labels.Matcher{
								parser.MustLabelMatcher(labels.MatchEqual, "__name__", "some_metric2"),
							},
							PosRange: posrange.PositionRange{
								Start: 29,
								End:   46,
							},
							Timestamp: makeInt64Pointer(20000),
						},
						PosRange: posrange.PositionRange{
							Start: 25,
							End:   47,
						},
					},
				},
			},
		},
		{
			input: "some_metric and topk(5, rate(some_metric[1m] @ 20))",
			expected: &parser.BinaryExpr{
				Op: parser.LAND,
				VectorMatching: &parser.VectorMatching{
					Card: parser.CardManyToMany,
				},
				LHS: &parser.VectorSelector{
					Name: "some_metric",
					LabelMatchers: []*labels.Matcher{
						parser.MustLabelMatcher(labels.MatchEqual, "__name__", "some_metric"),
					},
					PosRange: posrange.PositionRange{
						Start: 0,
						End:   11,
					},
				},
				RHS: &parser.StepInvariantExpr{
					Expr: &parser.AggregateExpr{
						Op: parser.TOPK,
						Expr: &parser.Call{
							Func: parser.MustGetFunction("rate"),
							Args: parser.Expressions{
								&parser.MatrixSelector{
									VectorSelector: &parser.VectorSelector{
										Name: "some_metric",
										LabelMatchers: []*labels.Matcher{
											parser.MustLabelMatcher(labels.MatchEqual, "__name__", "some_metric"),
										},
										PosRange: posrange.PositionRange{
											Start: 29,
											End:   40,
										},
										Timestamp: makeInt64Pointer(20000),
									},
									Range:  1 * time.Minute,
									EndPos: 49,
								},
							},
							PosRange: posrange.PositionRange{
								Start: 24,
								End:   50,
							},
						},
						Param: &parser.NumberLiteral{
							Val: 5,
							PosRange: posrange.PositionRange{
								Start: 21,
								End:   22,
							},
						},
						PosRange: posrange.PositionRange{
							Start: 16,
							End:   51,
						},
					},
				},
			},
		},
		{
			input: "time()",
			expected: &parser.Call{
				Func: parser.MustGetFunction("time"),
				Args: parser.Expressions{},
				PosRange: posrange.PositionRange{
					Start: 0,
					End:   6,
				},
			},
		},
		{
			input: `foo{bar="baz"}[10m:6s]`,
			expected: &parser.SubqueryExpr{
				Expr: &parser.VectorSelector{
					Name: "foo",
					LabelMatchers: []*labels.Matcher{
						parser.MustLabelMatcher(labels.MatchEqual, "bar", "baz"),
						parser.MustLabelMatcher(labels.MatchEqual, "__name__", "foo"),
					},
					PosRange: posrange.PositionRange{
						Start: 0,
						End:   14,
					},
				},
				Range:  10 * time.Minute,
				Step:   6 * time.Second,
				EndPos: 22,
			},
		},
		{
			input: `foo{bar="baz"}[10m:6s] @ 10`,
			expected: &parser.StepInvariantExpr{
				Expr: &parser.SubqueryExpr{
					Expr: &parser.VectorSelector{
						Name: "foo",
						LabelMatchers: []*labels.Matcher{
							parser.MustLabelMatcher(labels.MatchEqual, "bar", "baz"),
							parser.MustLabelMatcher(labels.MatchEqual, "__name__", "foo"),
						},
						PosRange: posrange.PositionRange{
							Start: 0,
							End:   14,
						},
					},
					Range:     10 * time.Minute,
					Step:      6 * time.Second,
					Timestamp: makeInt64Pointer(10000),
					EndPos:    27,
				},
			},
		},
		{ // Even though the subquery is step invariant, the inside is also wrapped separately.
			input: `sum(foo{bar="baz"} @ 20)[10m:6s] @ 10`,
			expected: &parser.StepInvariantExpr{
				Expr: &parser.SubqueryExpr{
					Expr: &parser.StepInvariantExpr{
						Expr: &parser.AggregateExpr{
							Op: parser.SUM,
							Expr: &parser.VectorSelector{
								Name: "foo",
								LabelMatchers: []*labels.Matcher{
									parser.MustLabelMatcher(labels.MatchEqual, "bar", "baz"),
									parser.MustLabelMatcher(labels.MatchEqual, "__name__", "foo"),
								},
								PosRange: posrange.PositionRange{
									Start: 4,
									End:   23,
								},
								Timestamp: makeInt64Pointer(20000),
							},
							PosRange: posrange.PositionRange{
								Start: 0,
								End:   24,
							},
						},
					},
					Range:     10 * time.Minute,
					Step:      6 * time.Second,
					Timestamp: makeInt64Pointer(10000),
					EndPos:    37,
				},
			},
		},
		{
			input: `min_over_time(rate(foo{bar="baz"}[2s])[5m:] @ 1603775091)[4m:3s]`,
			expected: &parser.SubqueryExpr{
				Expr: &parser.StepInvariantExpr{
					Expr: &parser.Call{
						Func: parser.MustGetFunction("min_over_time"),
						Args: parser.Expressions{
							&parser.SubqueryExpr{
								Expr: &parser.Call{
									Func: parser.MustGetFunction("rate"),
									Args: parser.Expressions{
										&parser.MatrixSelector{
											VectorSelector: &parser.VectorSelector{
												Name: "foo",
												LabelMatchers: []*labels.Matcher{
													parser.MustLabelMatcher(labels.MatchEqual, "bar", "baz"),
													parser.MustLabelMatcher(labels.MatchEqual, "__name__", "foo"),
												},
												PosRange: posrange.PositionRange{
													Start: 19,
													End:   33,
												},
											},
											Range:  2 * time.Second,
											EndPos: 37,
										},
									},
									PosRange: posrange.PositionRange{
										Start: 14,
										End:   38,
									},
								},
								Range:     5 * time.Minute,
								Timestamp: makeInt64Pointer(1603775091000),
								EndPos:    56,
							},
						},
						PosRange: posrange.PositionRange{
							Start: 0,
							End:   57,
						},
					},
				},
				Range:  4 * time.Minute,
				Step:   3 * time.Second,
				EndPos: 64,
			},
		},
		{
			input: `some_metric @ 123 offset 1m [10m:5s]`,
			expected: &parser.SubqueryExpr{
				Expr: &parser.StepInvariantExpr{
					Expr: &parser.VectorSelector{
						Name: "some_metric",
						LabelMatchers: []*labels.Matcher{
							parser.MustLabelMatcher(labels.MatchEqual, "__name__", "some_metric"),
						},
						PosRange: posrange.PositionRange{
							Start: 0,
							End:   27,
						},
						Timestamp:      makeInt64Pointer(123000),
						OriginalOffset: 1 * time.Minute,
					},
				},
				Range:  10 * time.Minute,
				Step:   5 * time.Second,
				EndPos: 36,
			},
		},
		{
			input: `some_metric[10m:5s] offset 1m @ 123`,
			expected: &parser.StepInvariantExpr{
				Expr: &parser.SubqueryExpr{
					Expr: &parser.VectorSelector{
						Name: "some_metric",
						LabelMatchers: []*labels.Matcher{
							parser.MustLabelMatcher(labels.MatchEqual, "__name__", "some_metric"),
						},
						PosRange: posrange.PositionRange{
							Start: 0,
							End:   11,
						},
					},
					Timestamp:      makeInt64Pointer(123000),
					OriginalOffset: 1 * time.Minute,
					Range:          10 * time.Minute,
					Step:           5 * time.Second,
					EndPos:         35,
				},
			},
		},
		{
			input: `(foo + bar{nm="val"} @ 1234)[5m:] @ 1603775019`,
			expected: &parser.StepInvariantExpr{
				Expr: &parser.SubqueryExpr{
					Expr: &parser.ParenExpr{
						Expr: &parser.BinaryExpr{
							Op: parser.ADD,
							VectorMatching: &parser.VectorMatching{
								Card: parser.CardOneToOne,
							},
							LHS: &parser.VectorSelector{
								Name: "foo",
								LabelMatchers: []*labels.Matcher{
									parser.MustLabelMatcher(labels.MatchEqual, "__name__", "foo"),
								},
								PosRange: posrange.PositionRange{
									Start: 1,
									End:   4,
								},
							},
							RHS: &parser.StepInvariantExpr{
								Expr: &parser.VectorSelector{
									Name: "bar",
									LabelMatchers: []*labels.Matcher{
										parser.MustLabelMatcher(labels.MatchEqual, "nm", "val"),
										parser.MustLabelMatcher(labels.MatchEqual, "__name__", "bar"),
									},
									Timestamp: makeInt64Pointer(1234000),
									PosRange: posrange.PositionRange{
										Start: 7,
										End:   27,
									},
								},
							},
						},
						PosRange: posrange.PositionRange{
							Start: 0,
							End:   28,
						},
					},
					Range:     5 * time.Minute,
					Timestamp: makeInt64Pointer(1603775019000),
					EndPos:    46,
				},
			},
		},
		{
			input: "abs(abs(metric @ 10))",
			expected: &parser.StepInvariantExpr{
				Expr: &parser.Call{
					Func: &parser.Function{
						Name:       "abs",
						ArgTypes:   []parser.ValueType{parser.ValueTypeVector},
						ReturnType: parser.ValueTypeVector,
					},
					Args: parser.Expressions{&parser.Call{
						Func: &parser.Function{
							Name:       "abs",
							ArgTypes:   []parser.ValueType{parser.ValueTypeVector},
							ReturnType: parser.ValueTypeVector,
						},
						Args: parser.Expressions{&parser.VectorSelector{
							Name: "metric",
							LabelMatchers: []*labels.Matcher{
								parser.MustLabelMatcher(labels.MatchEqual, "__name__", "metric"),
							},
							PosRange: posrange.PositionRange{
								Start: 8,
								End:   19,
							},
							Timestamp: makeInt64Pointer(10000),
						}},
						PosRange: posrange.PositionRange{
							Start: 4,
							End:   20,
						},
					}},
					PosRange: posrange.PositionRange{
						Start: 0,
						End:   21,
					},
				},
			},
		},
		{
			input: "sum(sum(some_metric1 @ 10) + sum(some_metric2 @ 20))",
			expected: &parser.StepInvariantExpr{
				Expr: &parser.AggregateExpr{
					Op: parser.SUM,
					Expr: &parser.BinaryExpr{
						Op:             parser.ADD,
						VectorMatching: &parser.VectorMatching{},
						LHS: &parser.AggregateExpr{
							Op: parser.SUM,
							Expr: &parser.VectorSelector{
								Name: "some_metric1",
								LabelMatchers: []*labels.Matcher{
									parser.MustLabelMatcher(labels.MatchEqual, "__name__", "some_metric1"),
								},
								PosRange: posrange.PositionRange{
									Start: 8,
									End:   25,
								},
								Timestamp: makeInt64Pointer(10000),
							},
							PosRange: posrange.PositionRange{
								Start: 4,
								End:   26,
							},
						},
						RHS: &parser.AggregateExpr{
							Op: parser.SUM,
							Expr: &parser.VectorSelector{
								Name: "some_metric2",
								LabelMatchers: []*labels.Matcher{
									parser.MustLabelMatcher(labels.MatchEqual, "__name__", "some_metric2"),
								},
								PosRange: posrange.PositionRange{
									Start: 33,
									End:   50,
								},
								Timestamp: makeInt64Pointer(20000),
							},
							PosRange: posrange.PositionRange{
								Start: 29,
								End:   52,
							},
						},
					},
					PosRange: posrange.PositionRange{
						Start: 0,
						End:   52,
					},
				},
			},
		},
		{
			input: `foo @ start()`,
			expected: &parser.StepInvariantExpr{
				Expr: &parser.VectorSelector{
					Name: "foo",
					LabelMatchers: []*labels.Matcher{
						parser.MustLabelMatcher(labels.MatchEqual, "__name__", "foo"),
					},
					PosRange: posrange.PositionRange{
						Start: 0,
						End:   13,
					},
					Timestamp:  makeInt64Pointer(timestamp.FromTime(startTime)),
					StartOrEnd: parser.START,
				},
			},
		},
		{
			input: `foo @ end()`,
			expected: &parser.StepInvariantExpr{
				Expr: &parser.VectorSelector{
					Name: "foo",
					LabelMatchers: []*labels.Matcher{
						parser.MustLabelMatcher(labels.MatchEqual, "__name__", "foo"),
					},
					PosRange: posrange.PositionRange{
						Start: 0,
						End:   11,
					},
					Timestamp:  makeInt64Pointer(timestamp.FromTime(endTime)),
					StartOrEnd: parser.END,
				},
			},
		},
		{
			input: `test[5y] @ start()`,
			expected: &parser.StepInvariantExpr{
				Expr: &parser.MatrixSelector{
					VectorSelector: &parser.VectorSelector{
						Name:       "test",
						Timestamp:  makeInt64Pointer(timestamp.FromTime(startTime)),
						StartOrEnd: parser.START,
						LabelMatchers: []*labels.Matcher{
							parser.MustLabelMatcher(labels.MatchEqual, "__name__", "test"),
						},
						PosRange: posrange.PositionRange{
							Start: 0,
							End:   4,
						},
					},
					Range:  5 * 365 * 24 * time.Hour,
					EndPos: 18,
				},
			},
		},
		{
			input: `test[5y] @ end()`,
			expected: &parser.StepInvariantExpr{
				Expr: &parser.MatrixSelector{
					VectorSelector: &parser.VectorSelector{
						Name:       "test",
						Timestamp:  makeInt64Pointer(timestamp.FromTime(endTime)),
						StartOrEnd: parser.END,
						LabelMatchers: []*labels.Matcher{
							parser.MustLabelMatcher(labels.MatchEqual, "__name__", "test"),
						},
						PosRange: posrange.PositionRange{
							Start: 0,
							End:   4,
						},
					},
					Range:  5 * 365 * 24 * time.Hour,
					EndPos: 16,
				},
			},
		},
		{
			input: `some_metric[10m:5s] @ start()`,
			expected: &parser.StepInvariantExpr{
				Expr: &parser.SubqueryExpr{
					Expr: &parser.VectorSelector{
						Name: "some_metric",
						LabelMatchers: []*labels.Matcher{
							parser.MustLabelMatcher(labels.MatchEqual, "__name__", "some_metric"),
						},
						PosRange: posrange.PositionRange{
							Start: 0,
							End:   11,
						},
					},
					Timestamp:  makeInt64Pointer(timestamp.FromTime(startTime)),
					StartOrEnd: parser.START,
					Range:      10 * time.Minute,
					Step:       5 * time.Second,
					EndPos:     29,
				},
			},
		},
		{
			input: `some_metric[10m:5s] @ end()`,
			expected: &parser.StepInvariantExpr{
				Expr: &parser.SubqueryExpr{
					Expr: &parser.VectorSelector{
						Name: "some_metric",
						LabelMatchers: []*labels.Matcher{
							parser.MustLabelMatcher(labels.MatchEqual, "__name__", "some_metric"),
						},
						PosRange: posrange.PositionRange{
							Start: 0,
							End:   11,
						},
					},
					Timestamp:  makeInt64Pointer(timestamp.FromTime(endTime)),
					StartOrEnd: parser.END,
					Range:      10 * time.Minute,
					Step:       5 * time.Second,
					EndPos:     27,
				},
			},
		},
		{
			input:      `floor(some_metric / (3 * 1024))`,
			outputTest: true,
			expected: &parser.Call{
				Func: &parser.Function{
					Name:       "floor",
					ArgTypes:   []parser.ValueType{parser.ValueTypeVector},
					ReturnType: parser.ValueTypeVector,
				},
				Args: parser.Expressions{
					&parser.BinaryExpr{
						Op: parser.DIV,
						LHS: &parser.VectorSelector{
							Name: "some_metric",
							LabelMatchers: []*labels.Matcher{
								parser.MustLabelMatcher(labels.MatchEqual, "__name__", "some_metric"),
							},
							PosRange: posrange.PositionRange{
								Start: 6,
								End:   17,
							},
						},
						RHS: &parser.StepInvariantExpr{
							Expr: &parser.ParenExpr{
								Expr: &parser.BinaryExpr{
									Op: parser.MUL,
									LHS: &parser.NumberLiteral{
										Val: 3,
										PosRange: posrange.PositionRange{
											Start: 21,
											End:   22,
										},
									},
									RHS: &parser.NumberLiteral{
										Val: 1024,
										PosRange: posrange.PositionRange{
											Start: 25,
											End:   29,
										},
									},
								},
								PosRange: posrange.PositionRange{
									Start: 20,
									End:   30,
								},
							},
						},
					},
				},
				PosRange: posrange.PositionRange{
					Start: 0,
					End:   31,
				},
			},
		},
	}

	for _, test := range testCases {
		t.Run(test.input, func(t *testing.T) {
			expr, err := parser.ParseExpr(test.input)
			require.NoError(t, err)
			expr = promql.PreprocessExpr(expr, startTime, endTime)
			if test.outputTest {
				require.Equal(t, test.input, expr.String(), "error on input '%s'", test.input)
			}
			require.Equal(t, test.expected, expr, "error on input '%s'", test.input)
		})
	}
}

func TestEngineOptsValidation(t *testing.T) {
	cases := []struct {
		opts     promql.EngineOpts
		query    string
		fail     bool
		expError error
	}{
		{
			opts:  promql.EngineOpts{EnableAtModifier: false},
			query: "metric @ 100", fail: true, expError: promql.ErrValidationAtModifierDisabled,
		}, {
			opts:  promql.EngineOpts{EnableAtModifier: false},
			query: "rate(metric[1m] @ 100)", fail: true, expError: promql.ErrValidationAtModifierDisabled,
		}, {
			opts:  promql.EngineOpts{EnableAtModifier: false},
			query: "rate(metric[1h:1m] @ 100)", fail: true, expError: promql.ErrValidationAtModifierDisabled,
		}, {
			opts:  promql.EngineOpts{EnableAtModifier: false},
			query: "metric @ start()", fail: true, expError: promql.ErrValidationAtModifierDisabled,
		}, {
			opts:  promql.EngineOpts{EnableAtModifier: false},
			query: "rate(metric[1m] @ start())", fail: true, expError: promql.ErrValidationAtModifierDisabled,
		}, {
			opts:  promql.EngineOpts{EnableAtModifier: false},
			query: "rate(metric[1h:1m] @ start())", fail: true, expError: promql.ErrValidationAtModifierDisabled,
		}, {
			opts:  promql.EngineOpts{EnableAtModifier: false},
			query: "metric @ end()", fail: true, expError: promql.ErrValidationAtModifierDisabled,
		}, {
			opts:  promql.EngineOpts{EnableAtModifier: false},
			query: "rate(metric[1m] @ end())", fail: true, expError: promql.ErrValidationAtModifierDisabled,
		}, {
			opts:  promql.EngineOpts{EnableAtModifier: false},
			query: "rate(metric[1h:1m] @ end())", fail: true, expError: promql.ErrValidationAtModifierDisabled,
		}, {
			opts:  promql.EngineOpts{EnableAtModifier: true},
			query: "metric @ 100",
		}, {
			opts:  promql.EngineOpts{EnableAtModifier: true},
			query: "rate(metric[1m] @ start())",
		}, {
			opts:  promql.EngineOpts{EnableAtModifier: true},
			query: "rate(metric[1h:1m] @ end())",
		}, {
			opts:  promql.EngineOpts{EnableNegativeOffset: false},
			query: "metric offset -1s", fail: true, expError: promql.ErrValidationNegativeOffsetDisabled,
		}, {
			opts:  promql.EngineOpts{EnableNegativeOffset: true},
			query: "metric offset -1s",
		}, {
			opts:  promql.EngineOpts{EnableAtModifier: true, EnableNegativeOffset: true},
			query: "metric @ 100 offset -2m",
		}, {
			opts:  promql.EngineOpts{EnableAtModifier: true, EnableNegativeOffset: true},
			query: "metric offset -2m @ 100",
		},
	}

	for _, c := range cases {
		eng := promql.NewEngine(c.opts)
		_, err1 := eng.NewInstantQuery(context.Background(), nil, nil, c.query, time.Unix(10, 0))
		_, err2 := eng.NewRangeQuery(context.Background(), nil, nil, c.query, time.Unix(0, 0), time.Unix(10, 0), time.Second)
		if c.fail {
			require.Equal(t, c.expError, err1)
			require.Equal(t, c.expError, err2)
		} else {
			require.NoError(t, err1)
			require.NoError(t, err2)
		}
	}
}

func TestInstantQueryWithRangeVectorSelector(t *testing.T) {
	engine := newTestEngine()

	baseT := timestamp.Time(0)
	storage := promqltest.LoadedStorage(t, `
		load 1m
			some_metric{env="1"} 0+1x4
			some_metric{env="2"} 0+2x4
			some_metric_with_stale_marker 0 1 stale 3
	`)
	t.Cleanup(func() { require.NoError(t, storage.Close()) })

	testCases := map[string]struct {
		expr     string
		expected promql.Matrix
		ts       time.Time
	}{
		"matches series with points in range": {
			expr: "some_metric[1m]",
			ts:   baseT.Add(2 * time.Minute),
			expected: promql.Matrix{
				{
					Metric: labels.FromStrings("__name__", "some_metric", "env", "1"),
					Floats: []promql.FPoint{
						{T: timestamp.FromTime(baseT.Add(time.Minute)), F: 1},
						{T: timestamp.FromTime(baseT.Add(2 * time.Minute)), F: 2},
					},
				},
				{
					Metric: labels.FromStrings("__name__", "some_metric", "env", "2"),
					Floats: []promql.FPoint{
						{T: timestamp.FromTime(baseT.Add(time.Minute)), F: 2},
						{T: timestamp.FromTime(baseT.Add(2 * time.Minute)), F: 4},
					},
				},
			},
		},
		"matches no series": {
			expr:     "some_nonexistent_metric[1m]",
			ts:       baseT,
			expected: promql.Matrix{},
		},
		"no samples in range": {
			expr:     "some_metric[1m]",
			ts:       baseT.Add(20 * time.Minute),
			expected: promql.Matrix{},
		},
		"metric with stale marker": {
			expr: "some_metric_with_stale_marker[3m]",
			ts:   baseT.Add(3 * time.Minute),
			expected: promql.Matrix{
				{
					Metric: labels.FromStrings("__name__", "some_metric_with_stale_marker"),
					Floats: []promql.FPoint{
						{T: timestamp.FromTime(baseT), F: 0},
						{T: timestamp.FromTime(baseT.Add(time.Minute)), F: 1},
						{T: timestamp.FromTime(baseT.Add(3 * time.Minute)), F: 3},
					},
				},
			},
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			q, err := engine.NewInstantQuery(context.Background(), storage, nil, testCase.expr, testCase.ts)
			require.NoError(t, err)
			defer q.Close()

			res := q.Exec(context.Background())
			require.NoError(t, res.Err)
			testutil.RequireEqual(t, testCase.expected, res.Value)
		})
	}
}

func TestNativeHistogram_Sum_Count_Add_AvgOperator(t *testing.T) {
	// TODO(codesome): Integrate histograms into the PromQL testing framework
	// and write more tests there.
	cases := []struct {
		histograms  []histogram.Histogram
		expected    histogram.FloatHistogram
		expectedAvg histogram.FloatHistogram
	}{
		{
			histograms: []histogram.Histogram{
				{
					CounterResetHint: histogram.GaugeType,
					Schema:           0,
					Count:            25,
					Sum:              1234.5,
					ZeroThreshold:    0.001,
					ZeroCount:        4,
					PositiveSpans: []histogram.Span{
						{Offset: 0, Length: 2},
						{Offset: 1, Length: 2},
					},
					PositiveBuckets: []int64{1, 1, -1, 0},
					NegativeSpans: []histogram.Span{
						{Offset: 0, Length: 2},
						{Offset: 2, Length: 2},
					},
					NegativeBuckets: []int64{2, 2, -3, 8},
				},
				{
					CounterResetHint: histogram.GaugeType,
					Schema:           0,
					Count:            41,
					Sum:              2345.6,
					ZeroThreshold:    0.001,
					ZeroCount:        5,
					PositiveSpans: []histogram.Span{
						{Offset: 0, Length: 4},
						{Offset: 0, Length: 0},
						{Offset: 0, Length: 3},
					},
					PositiveBuckets: []int64{1, 2, -2, 1, -1, 0, 0},
					NegativeSpans: []histogram.Span{
						{Offset: 1, Length: 4},
						{Offset: 2, Length: 0},
						{Offset: 2, Length: 3},
					},
					NegativeBuckets: []int64{1, 3, -2, 5, -2, 0, -3},
				},
				{
					CounterResetHint: histogram.GaugeType,
					Schema:           0,
					Count:            41,
					Sum:              1111.1,
					ZeroThreshold:    0.001,
					ZeroCount:        5,
					PositiveSpans: []histogram.Span{
						{Offset: 0, Length: 4},
						{Offset: 0, Length: 0},
						{Offset: 0, Length: 3},
					},
					PositiveBuckets: []int64{1, 2, -2, 1, -1, 0, 0},
					NegativeSpans: []histogram.Span{
						{Offset: 1, Length: 4},
						{Offset: 2, Length: 0},
						{Offset: 2, Length: 3},
					},
					NegativeBuckets: []int64{1, 3, -2, 5, -2, 0, -3},
				},
				{
					CounterResetHint: histogram.GaugeType,
					Schema:           1, // Everything is 0 just to make the count 4 so avg has nicer numbers.
				},
			},
			expected: histogram.FloatHistogram{
				CounterResetHint: histogram.GaugeType,
				Schema:           0,
				ZeroThreshold:    0.001,
				ZeroCount:        14,
				Count:            107,
				Sum:              4691.2,
				PositiveSpans: []histogram.Span{
					{Offset: 0, Length: 7},
				},
				PositiveBuckets: []float64{3, 8, 2, 5, 3, 2, 2},
				NegativeSpans: []histogram.Span{
					{Offset: 0, Length: 6},
					{Offset: 3, Length: 3},
				},
				NegativeBuckets: []float64{2, 6, 8, 4, 15, 9, 10, 10, 4},
			},
			expectedAvg: histogram.FloatHistogram{
				CounterResetHint: histogram.GaugeType,
				Schema:           0,
				ZeroThreshold:    0.001,
				ZeroCount:        3.5,
				Count:            26.75,
				Sum:              1172.8,
				PositiveSpans: []histogram.Span{
					{Offset: 0, Length: 7},
				},
				PositiveBuckets: []float64{0.75, 2, 0.5, 1.25, 0.75, 0.5, 0.5},
				NegativeSpans: []histogram.Span{
					{Offset: 0, Length: 6},
					{Offset: 3, Length: 3},
				},
				NegativeBuckets: []float64{0.5, 1.5, 2, 1, 3.75, 2.25, 2.5, 2.5, 1},
			},
		},
	}

	idx0 := int64(0)
	for _, c := range cases {
		for _, floatHisto := range []bool{true, false} {
			t.Run(fmt.Sprintf("floatHistogram=%t %d", floatHisto, idx0), func(t *testing.T) {
				storage := teststorage.New(t)
				t.Cleanup(func() { storage.Close() })

				seriesName := "sparse_histogram_series"
				seriesNameOverTime := "sparse_histogram_series_over_time"

				engine := newTestEngine()

				ts := idx0 * int64(10*time.Minute/time.Millisecond)
				app := storage.Appender(context.Background())
				_, err := app.Append(0, labels.FromStrings("__name__", "float_series", "idx", "0"), ts, 42)
				require.NoError(t, err)
				for idx1, h := range c.histograms {
					lbls := labels.FromStrings("__name__", seriesName, "idx", strconv.Itoa(idx1))
					// Since we mutate h later, we need to create a copy here.
					var err error
					if floatHisto {
						_, err = app.AppendHistogram(0, lbls, ts, nil, h.Copy().ToFloat(nil))
					} else {
						_, err = app.AppendHistogram(0, lbls, ts, h.Copy(), nil)
					}
					require.NoError(t, err)

					lbls = labels.FromStrings("__name__", seriesNameOverTime)
					newTs := ts + int64(idx1)*int64(time.Minute/time.Millisecond)
					// Since we mutate h later, we need to create a copy here.
					if floatHisto {
						_, err = app.AppendHistogram(0, lbls, newTs, nil, h.Copy().ToFloat(nil))
					} else {
						_, err = app.AppendHistogram(0, lbls, newTs, h.Copy(), nil)
					}
					require.NoError(t, err)
				}
				require.NoError(t, app.Commit())

				queryAndCheck := func(queryString string, ts int64, exp promql.Vector) {
					qry, err := engine.NewInstantQuery(context.Background(), storage, nil, queryString, timestamp.Time(ts))
					require.NoError(t, err)

					res := qry.Exec(context.Background())
					require.NoError(t, res.Err)
					require.Empty(t, res.Warnings)

					vector, err := res.Vector()
					require.NoError(t, err)

					testutil.RequireEqual(t, exp, vector)
				}
				queryAndCheckAnnotations := func(queryString string, ts int64, expWarnings annotations.Annotations) {
					qry, err := engine.NewInstantQuery(context.Background(), storage, nil, queryString, timestamp.Time(ts))
					require.NoError(t, err)

					res := qry.Exec(context.Background())
					require.NoError(t, res.Err)
					require.Equal(t, expWarnings, res.Warnings)
				}

				// sum().
				queryString := fmt.Sprintf("sum(%s)", seriesName)
				queryAndCheck(queryString, ts, []promql.Sample{{T: ts, H: &c.expected, Metric: labels.EmptyLabels()}})

				queryString = `sum({idx="0"})`
				var annos annotations.Annotations
				annos.Add(annotations.NewMixedFloatsHistogramsAggWarning(posrange.PositionRange{Start: 4, End: 13}))
				queryAndCheckAnnotations(queryString, ts, annos)

				// + operator.
				queryString = fmt.Sprintf(`%s{idx="0"}`, seriesName)
				for idx := 1; idx < len(c.histograms); idx++ {
					queryString += fmt.Sprintf(` + ignoring(idx) %s{idx="%d"}`, seriesName, idx)
				}
				queryAndCheck(queryString, ts, []promql.Sample{{T: ts, H: &c.expected, Metric: labels.EmptyLabels()}})

				// count().
				queryString = fmt.Sprintf("count(%s)", seriesName)
				queryAndCheck(queryString, ts, []promql.Sample{{T: ts, F: 4, Metric: labels.EmptyLabels()}})

				// avg().
				queryString = fmt.Sprintf("avg(%s)", seriesName)
				queryAndCheck(queryString, ts, []promql.Sample{{T: ts, H: &c.expectedAvg, Metric: labels.EmptyLabels()}})

				offset := int64(len(c.histograms) - 1)
				newTs := ts + offset*int64(time.Minute/time.Millisecond)

				// sum_over_time().
				queryString = fmt.Sprintf("sum_over_time(%s[%dm:1m])", seriesNameOverTime, offset)
				queryAndCheck(queryString, newTs, []promql.Sample{{T: newTs, H: &c.expected, Metric: labels.EmptyLabels()}})

				// avg_over_time().
				queryString = fmt.Sprintf("avg_over_time(%s[%dm:1m])", seriesNameOverTime, offset)
				queryAndCheck(queryString, newTs, []promql.Sample{{T: newTs, H: &c.expectedAvg, Metric: labels.EmptyLabels()}})
			})
			idx0++
		}
	}
}

func TestNativeHistogram_SubOperator(t *testing.T) {
	// TODO(codesome): Integrate histograms into the PromQL testing framework
	// and write more tests there.
	cases := []struct {
		histograms []histogram.Histogram
		expected   histogram.FloatHistogram
	}{
		{
			histograms: []histogram.Histogram{
				{
					Schema:        0,
					Count:         41,
					Sum:           2345.6,
					ZeroThreshold: 0.001,
					ZeroCount:     5,
					PositiveSpans: []histogram.Span{
						{Offset: 0, Length: 4},
						{Offset: 0, Length: 0},
						{Offset: 0, Length: 3},
					},
					PositiveBuckets: []int64{1, 2, -2, 1, -1, 0, 0},
					NegativeSpans: []histogram.Span{
						{Offset: 1, Length: 4},
						{Offset: 2, Length: 0},
						{Offset: 2, Length: 3},
					},
					NegativeBuckets: []int64{1, 3, -2, 5, -2, 0, -3},
				},
				{
					Schema:        0,
					Count:         11,
					Sum:           1234.5,
					ZeroThreshold: 0.001,
					ZeroCount:     3,
					PositiveSpans: []histogram.Span{
						{Offset: 1, Length: 2},
					},
					PositiveBuckets: []int64{2, -1},
					NegativeSpans: []histogram.Span{
						{Offset: 2, Length: 2},
					},
					NegativeBuckets: []int64{3, -1},
				},
			},
			expected: histogram.FloatHistogram{
				Schema:        0,
				Count:         30,
				Sum:           1111.1,
				ZeroThreshold: 0.001,
				ZeroCount:     2,
				PositiveSpans: []histogram.Span{
					{Offset: 0, Length: 2},
					{Offset: 1, Length: 4},
				},
				PositiveBuckets: []float64{1, 1, 2, 1, 1, 1},
				NegativeSpans: []histogram.Span{
					{Offset: 1, Length: 2},
					{Offset: 1, Length: 1},
					{Offset: 4, Length: 3},
				},
				NegativeBuckets: []float64{1, 1, 7, 5, 5, 2},
			},
		},
		{
			histograms: []histogram.Histogram{
				{
					Schema:        0,
					Count:         41,
					Sum:           2345.6,
					ZeroThreshold: 0.001,
					ZeroCount:     5,
					PositiveSpans: []histogram.Span{
						{Offset: 0, Length: 4},
						{Offset: 0, Length: 0},
						{Offset: 0, Length: 3},
					},
					PositiveBuckets: []int64{1, 2, -2, 1, -1, 0, 0},
					NegativeSpans: []histogram.Span{
						{Offset: 1, Length: 4},
						{Offset: 2, Length: 0},
						{Offset: 2, Length: 3},
					},
					NegativeBuckets: []int64{1, 3, -2, 5, -2, 0, -3},
				},
				{
					Schema:        1,
					Count:         11,
					Sum:           1234.5,
					ZeroThreshold: 0.001,
					ZeroCount:     3,
					PositiveSpans: []histogram.Span{
						{Offset: 1, Length: 2},
					},
					PositiveBuckets: []int64{2, -1},
					NegativeSpans: []histogram.Span{
						{Offset: 2, Length: 2},
					},
					NegativeBuckets: []int64{3, -1},
				},
			},
			expected: histogram.FloatHistogram{
				Schema:        0,
				Count:         30,
				Sum:           1111.1,
				ZeroThreshold: 0.001,
				ZeroCount:     2,
				PositiveSpans: []histogram.Span{
					{Offset: 0, Length: 1},
					{Offset: 1, Length: 5},
				},
				PositiveBuckets: []float64{1, 1, 2, 1, 1, 1},
				NegativeSpans: []histogram.Span{
					{Offset: 1, Length: 4},
					{Offset: 4, Length: 3},
				},
				NegativeBuckets: []float64{-2, 2, 2, 7, 5, 5, 2},
			},
		},
		{
			histograms: []histogram.Histogram{
				{
					Schema:        1,
					Count:         11,
					Sum:           1234.5,
					ZeroThreshold: 0.001,
					ZeroCount:     3,
					PositiveSpans: []histogram.Span{
						{Offset: 1, Length: 2},
					},
					PositiveBuckets: []int64{2, -1},
					NegativeSpans: []histogram.Span{
						{Offset: 2, Length: 2},
					},
					NegativeBuckets: []int64{3, -1},
				},
				{
					Schema:        0,
					Count:         41,
					Sum:           2345.6,
					ZeroThreshold: 0.001,
					ZeroCount:     5,
					PositiveSpans: []histogram.Span{
						{Offset: 0, Length: 4},
						{Offset: 0, Length: 0},
						{Offset: 0, Length: 3},
					},
					PositiveBuckets: []int64{1, 2, -2, 1, -1, 0, 0},
					NegativeSpans: []histogram.Span{
						{Offset: 1, Length: 4},
						{Offset: 2, Length: 0},
						{Offset: 2, Length: 3},
					},
					NegativeBuckets: []int64{1, 3, -2, 5, -2, 0, -3},
				},
			},
			expected: histogram.FloatHistogram{
				Schema:        0,
				Count:         -30,
				Sum:           -1111.1,
				ZeroThreshold: 0.001,
				ZeroCount:     -2,
				PositiveSpans: []histogram.Span{
					{Offset: 0, Length: 1},
					{Offset: 1, Length: 5},
				},
				PositiveBuckets: []float64{-1, -1, -2, -1, -1, -1},
				NegativeSpans: []histogram.Span{
					{Offset: 1, Length: 4},
					{Offset: 4, Length: 3},
				},
				NegativeBuckets: []float64{2, -2, -2, -7, -5, -5, -2},
			},
		},
	}

	idx0 := int64(0)
	for _, c := range cases {
		for _, floatHisto := range []bool{true, false} {
			t.Run(fmt.Sprintf("floatHistogram=%t %d", floatHisto, idx0), func(t *testing.T) {
				engine := newTestEngine()
				storage := teststorage.New(t)
				t.Cleanup(func() { storage.Close() })

				seriesName := "sparse_histogram_series"

				ts := idx0 * int64(10*time.Minute/time.Millisecond)
				app := storage.Appender(context.Background())
				for idx1, h := range c.histograms {
					lbls := labels.FromStrings("__name__", seriesName, "idx", strconv.Itoa(idx1))
					// Since we mutate h later, we need to create a copy here.
					var err error
					if floatHisto {
						_, err = app.AppendHistogram(0, lbls, ts, nil, h.Copy().ToFloat(nil))
					} else {
						_, err = app.AppendHistogram(0, lbls, ts, h.Copy(), nil)
					}
					require.NoError(t, err)
				}
				require.NoError(t, app.Commit())

				queryAndCheck := func(queryString string, exp promql.Vector) {
					qry, err := engine.NewInstantQuery(context.Background(), storage, nil, queryString, timestamp.Time(ts))
					require.NoError(t, err)

					res := qry.Exec(context.Background())
					require.NoError(t, res.Err)

					vector, err := res.Vector()
					require.NoError(t, err)

					if len(vector) == len(exp) {
						for i, e := range exp {
							got := vector[i].H
							if got != e.H {
								// Error messages are better if we compare structs, not pointers.
								require.Equal(t, *e.H, *got)
							}
						}
					}

					testutil.RequireEqual(t, exp, vector)
				}

				// - operator.
				queryString := fmt.Sprintf(`%s{idx="0"}`, seriesName)
				for idx := 1; idx < len(c.histograms); idx++ {
					queryString += fmt.Sprintf(` - ignoring(idx) %s{idx="%d"}`, seriesName, idx)
				}
				queryAndCheck(queryString, []promql.Sample{{T: ts, H: &c.expected, Metric: labels.EmptyLabels()}})
			})
		}
		idx0++
	}
}

func TestNativeHistogram_MulDivOperator(t *testing.T) {
	// TODO(codesome): Integrate histograms into the PromQL testing framework
	// and write more tests there.
	originalHistogram := histogram.Histogram{
		Schema:        0,
		Count:         21,
		Sum:           33,
		ZeroThreshold: 0.001,
		ZeroCount:     3,
		PositiveSpans: []histogram.Span{
			{Offset: 0, Length: 3},
		},
		PositiveBuckets: []int64{3, 0, 0},
		NegativeSpans: []histogram.Span{
			{Offset: 0, Length: 3},
		},
		NegativeBuckets: []int64{3, 0, 0},
	}

	cases := []struct {
		scalar      float64
		histogram   histogram.Histogram
		expectedMul histogram.FloatHistogram
		expectedDiv histogram.FloatHistogram
	}{
		{
			scalar:    3,
			histogram: originalHistogram,
			expectedMul: histogram.FloatHistogram{
				Schema:        0,
				Count:         63,
				Sum:           99,
				ZeroThreshold: 0.001,
				ZeroCount:     9,
				PositiveSpans: []histogram.Span{
					{Offset: 0, Length: 3},
				},
				PositiveBuckets: []float64{9, 9, 9},
				NegativeSpans: []histogram.Span{
					{Offset: 0, Length: 3},
				},
				NegativeBuckets: []float64{9, 9, 9},
			},
			expectedDiv: histogram.FloatHistogram{
				Schema:        0,
				Count:         7,
				Sum:           11,
				ZeroThreshold: 0.001,
				ZeroCount:     1,
				PositiveSpans: []histogram.Span{
					{Offset: 0, Length: 3},
				},
				PositiveBuckets: []float64{1, 1, 1},
				NegativeSpans: []histogram.Span{
					{Offset: 0, Length: 3},
				},
				NegativeBuckets: []float64{1, 1, 1},
			},
		},
		{
			scalar:    0,
			histogram: originalHistogram,
			expectedMul: histogram.FloatHistogram{
				Schema:        0,
				Count:         0,
				Sum:           0,
				ZeroThreshold: 0.001,
				ZeroCount:     0,
				PositiveSpans: []histogram.Span{
					{Offset: 0, Length: 3},
				},
				PositiveBuckets: []float64{0, 0, 0},
				NegativeSpans: []histogram.Span{
					{Offset: 0, Length: 3},
				},
				NegativeBuckets: []float64{0, 0, 0},
			},
			expectedDiv: histogram.FloatHistogram{
				Schema:        0,
				Count:         math.Inf(1),
				Sum:           math.Inf(1),
				ZeroThreshold: 0.001,
				ZeroCount:     math.Inf(1),
				PositiveSpans: []histogram.Span{
					{Offset: 0, Length: 3},
				},
				PositiveBuckets: []float64{math.Inf(1), math.Inf(1), math.Inf(1)},
				NegativeSpans: []histogram.Span{
					{Offset: 0, Length: 3},
				},
				NegativeBuckets: []float64{math.Inf(1), math.Inf(1), math.Inf(1)},
			},
		},
	}

	idx0 := int64(0)
	for _, c := range cases {
		for _, floatHisto := range []bool{true, false} {
			t.Run(fmt.Sprintf("floatHistogram=%t %d", floatHisto, idx0), func(t *testing.T) {
				storage := teststorage.New(t)
				t.Cleanup(func() { storage.Close() })

				seriesName := "sparse_histogram_series"
				floatSeriesName := "float_series"

				engine := newTestEngine()

				ts := idx0 * int64(10*time.Minute/time.Millisecond)
				app := storage.Appender(context.Background())
				h := c.histogram
				lbls := labels.FromStrings("__name__", seriesName)
				// Since we mutate h later, we need to create a copy here.
				var err error
				if floatHisto {
					_, err = app.AppendHistogram(0, lbls, ts, nil, h.Copy().ToFloat(nil))
				} else {
					_, err = app.AppendHistogram(0, lbls, ts, h.Copy(), nil)
				}
				require.NoError(t, err)
				_, err = app.Append(0, labels.FromStrings("__name__", floatSeriesName), ts, c.scalar)
				require.NoError(t, err)
				require.NoError(t, app.Commit())

				queryAndCheck := func(queryString string, exp promql.Vector) {
					qry, err := engine.NewInstantQuery(context.Background(), storage, nil, queryString, timestamp.Time(ts))
					require.NoError(t, err)

					res := qry.Exec(context.Background())
					require.NoError(t, res.Err)

					vector, err := res.Vector()
					require.NoError(t, err)

					testutil.RequireEqual(t, exp, vector)
				}

				// histogram * scalar.
				queryString := fmt.Sprintf(`%s * %f`, seriesName, c.scalar)
				queryAndCheck(queryString, []promql.Sample{{T: ts, H: &c.expectedMul, Metric: labels.EmptyLabels()}})

				// scalar * histogram.
				queryString = fmt.Sprintf(`%f * %s`, c.scalar, seriesName)
				queryAndCheck(queryString, []promql.Sample{{T: ts, H: &c.expectedMul, Metric: labels.EmptyLabels()}})

				// histogram * float.
				queryString = fmt.Sprintf(`%s * %s`, seriesName, floatSeriesName)
				queryAndCheck(queryString, []promql.Sample{{T: ts, H: &c.expectedMul, Metric: labels.EmptyLabels()}})

				// float * histogram.
				queryString = fmt.Sprintf(`%s * %s`, floatSeriesName, seriesName)
				queryAndCheck(queryString, []promql.Sample{{T: ts, H: &c.expectedMul, Metric: labels.EmptyLabels()}})

				// histogram / scalar.
				queryString = fmt.Sprintf(`%s / %f`, seriesName, c.scalar)
				queryAndCheck(queryString, []promql.Sample{{T: ts, H: &c.expectedDiv, Metric: labels.EmptyLabels()}})

				// histogram / float.
				queryString = fmt.Sprintf(`%s / %s`, seriesName, floatSeriesName)
				queryAndCheck(queryString, []promql.Sample{{T: ts, H: &c.expectedDiv, Metric: labels.EmptyLabels()}})
			})
			idx0++
		}
	}
}

func TestQueryLookbackDelta(t *testing.T) {
	var (
		load = `load 5m
metric 0 1 2
`
		query           = "metric"
		lastDatapointTs = time.Unix(600, 0)
	)

	cases := []struct {
		name                          string
		ts                            time.Time
		engineLookback, queryLookback time.Duration
		expectSamples                 bool
	}{
		{
			name:          "default lookback delta",
			ts:            lastDatapointTs.Add(defaultLookbackDelta),
			expectSamples: true,
		},
		{
			name:          "outside default lookback delta",
			ts:            lastDatapointTs.Add(defaultLookbackDelta + time.Millisecond),
			expectSamples: false,
		},
		{
			name:           "custom engine lookback delta",
			ts:             lastDatapointTs.Add(10 * time.Minute),
			engineLookback: 10 * time.Minute,
			expectSamples:  true,
		},
		{
			name:           "outside custom engine lookback delta",
			ts:             lastDatapointTs.Add(10*time.Minute + time.Millisecond),
			engineLookback: 10 * time.Minute,
			expectSamples:  false,
		},
		{
			name:           "custom query lookback delta",
			ts:             lastDatapointTs.Add(20 * time.Minute),
			engineLookback: 10 * time.Minute,
			queryLookback:  20 * time.Minute,
			expectSamples:  true,
		},
		{
			name:           "outside custom query lookback delta",
			ts:             lastDatapointTs.Add(20*time.Minute + time.Millisecond),
			engineLookback: 10 * time.Minute,
			queryLookback:  20 * time.Minute,
			expectSamples:  false,
		},
		{
			name:           "negative custom query lookback delta",
			ts:             lastDatapointTs.Add(20 * time.Minute),
			engineLookback: -10 * time.Minute,
			queryLookback:  20 * time.Minute,
			expectSamples:  true,
		},
	}

	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			engine := promqltest.NewTestEngine(false, c.engineLookback, promqltest.DefaultMaxSamplesPerQuery)
			storage := promqltest.LoadedStorage(t, load)
			t.Cleanup(func() { storage.Close() })

			opts := promql.NewPrometheusQueryOpts(false, c.queryLookback)
			qry, err := engine.NewInstantQuery(context.Background(), storage, opts, query, c.ts)
			require.NoError(t, err)

			res := qry.Exec(context.Background())
			require.NoError(t, res.Err)
			vec, ok := res.Value.(promql.Vector)
			require.True(t, ok)
			if c.expectSamples {
				require.NotEmpty(t, vec)
			} else {
				require.Empty(t, vec)
			}
		})
	}
}

func makeInt64Pointer(val int64) *int64 {
	valp := new(int64)
	*valp = val
	return valp
}
