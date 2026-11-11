// Command cluster creates validated equal-size AnchorANN preprocessing artifacts.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	"privatesearch/anchorann"
)

type iterationReport struct {
	anchorann.Step
	Containment       *anchorann.ContainmentRates `json:"containment,omitempty"`
	EvaluationSeconds float64                     `json:"evaluation_seconds"`
}

type report struct {
	CompletedIterations int                         `json:"completed_iterations"`
	StopReason          string                      `json:"stop_reason"`
	Error               string                      `json:"error,omitempty"`
	History             []iterationReport           `json:"history"`
	Rates               *anchorann.ContainmentRates `json:"containment,omitempty"`
	GroundTruthSeconds  float64                     `json:"ground_truth_seconds"`
	Method              anchorann.Method            `json:"method"`
	Seed                int64                       `json:"seed"`
	Iterations          int                         `json:"iterations"`
	Workers             int                         `json:"workers"`
	N                   int                         `json:"n"`
	Size                int                         `json:"s"`
	Count               int                         `json:"c"`
	Dimension           int                         `json:"dimension"`
	LoadSeconds         float64                     `json:"input_load_seconds"`
	ClusterSeconds      float64                     `json:"cluster_seconds"`
	SaveSeconds         float64                     `json:"save_seconds"`
	PreprocessSeconds   float64                     `json:"preprocess_seconds"`
	PeakRSSBytes        int64                       `json:"peak_rss_bytes"`
	SSE                 float64                     `json:"sse,omitempty"`
	Queries             int                         `json:"queries"`
	TopK                int                         `json:"top_k"`
	Containment         *float64                    `json:"top_k_containment,omitempty"`
	EvaluationSeconds   float64                     `json:"evaluation_seconds"`
	Output              string                      `json:"output"`
	ArtifactBytes       int64                       `json:"artifact_bytes"`
	ReloadVerified      bool                        `json:"reload_verified"`
	GoVersion           string                      `json:"go_version"`
	Platform            string                      `json:"platform"`
}

func run(args []string, out io.Writer) error {
	defaults := anchorann.DefaultOptions()
	f := flag.NewFlagSet("cluster", flag.ContinueOnError)
	input := f.String("input", "", "input vector file (required)")
	format := f.String("format", "auto", "auto, fvecs, or bvecs")
	limit := f.Int("limit", 0, "explicit input prefix size; 0 reads all points")
	size := f.Int("s", defaults.Size, "real points per cluster (must divide N)")
	seed := f.Int64("seed", defaults.Seed, "deterministic seed recorded in artifact")
	method := f.String("method", string(defaults.Method), "recursive, global (greedy), or common (certified optimal)")
	iterations := f.Int("iterations", defaults.Iterations, "Lloyd iterations per split or global pass")
	workers := f.Int("workers", defaults.Workers, "parallel branches or full-centroid distance workers")
	output := f.String("output", "", "output artifact file (required)")
	queries := f.String("queries", "", "optional query vector file for exact top-k containment")
	queryFormat := f.String("query-format", "auto", "auto, fvecs, or bvecs")
	queryLimit := f.Int("query-limit", 100, "explicit query prefix size; 0 reads all")
	k := f.Int("top-k", 10, "ground-truth neighbor count for containment")
	historyPath := f.String("history", "", "durable completed-iteration JSONL file")
	gtPath := f.String("ground-truth", "", "GT JSON cache, generated if absent; contents are validated")
	budget := f.Duration("time-budget", 0, "wall budget for clustering and observers, excludes GT preparation; 0 is unlimited")
	if err := f.Parse(args); err != nil {
		return err
	}
	if *input == "" || *output == "" || f.NArg() != 0 {
		return fmt.Errorf("-input and -output are required; positional arguments are unsupported")
	}
	outputPath, err := filepath.Abs(*output)
	if err != nil {
		return fmt.Errorf("resolve output: %w", err)
	}
	outputInfo, _ := os.Stat(outputPath)
	for _, source := range []string{*input, *queries, *historyPath, *gtPath} {
		if source == "" {
			continue
		}
		sourcePath, err := filepath.Abs(source)
		if err != nil {
			return fmt.Errorf("resolve vector input: %w", err)
		}
		sourceInfo, _ := os.Stat(sourcePath)
		if sourcePath == outputPath || (sourceInfo != nil && outputInfo != nil && os.SameFile(sourceInfo, outputInfo)) {
			return fmt.Errorf("output must differ from vector inputs")
		}
	}
	start := time.Now()
	d, err := anchorann.ReadVectors(*input, *format, *limit)
	if err != nil {
		return fmt.Errorf("load input (s=%d): %w", *size, err)
	}
	loaded := time.Now()
	if *size <= 0 || d.N%*size != 0 {
		remainder := 0
		if *size > 0 {
			remainder = d.N % *size
		}
		return fmt.Errorf("invalid clustering size N=%d s=%d remainder=%d", d.N, *size, remainder)
	}
	// Cache outputs may not alias source vectors, even when currently absent.
	if *gtPath != "" {
		gp, _ := filepath.Abs(*gtPath)
		gi, _ := os.Stat(gp)
		for _, source := range []string{*input, *queries} {
			if source == "" {
				continue
			}
			sp, _ := filepath.Abs(source)
			si, _ := os.Stat(sp)
			if gp == sp || (gi != nil && si != nil && os.SameFile(gi, si)) {
				return fmt.Errorf("GT must differ from vector inputs")
			}
		}
	}

	var q anchorann.Dataset
	var gt *anchorann.GroundTruth
	var gtSeconds float64
	if *queries != "" {
		q, err = anchorann.ReadVectors(*queries, *queryFormat, *queryLimit)
		if err != nil {
			return fmt.Errorf("load queries: %w", err)
		}
		gtStart := time.Now()
		if *gtPath != "" {
			data, readErr := os.ReadFile(*gtPath)
			if readErr == nil {
				gt = &anchorann.GroundTruth{}
				if err := json.Unmarshal(data, gt); err != nil {
					return fmt.Errorf("decode GT: %w", err)
				}
			} else if !os.IsNotExist(readErr) {
				return fmt.Errorf("read GT: %w", readErr)
			}
		}
		if gt == nil {
			gt, err = anchorann.ExactGroundTruth(d, q, *k)
			if err != nil {
				return err
			}
			if *gtPath != "" {
				data, err := json.Marshal(gt)
				if err != nil {
					return err
				}
				if err := os.WriteFile(*gtPath, data, 0600); err != nil {
					return fmt.Errorf("save GT: %w", err)
				}
			}
		}
		if gt.K != *k {
			return fmt.Errorf("GT k=%d requested=%d", gt.K, *k)
		}
		if err := anchorann.ValidateGroundTruth(d, q, gt); err != nil {
			return err
		}
		gtSeconds = time.Since(gtStart).Seconds()
	}
	var historyFile *os.File
	if *historyPath != "" {
		// Refuse source/cache aliases before truncating a history file.
		path, _ := filepath.Abs(*historyPath)
		info, _ := os.Stat(path)
		for _, source := range []string{*input, *queries, *gtPath} {
			if source == "" {
				continue
			}
			sp, _ := filepath.Abs(source)
			si, _ := os.Stat(sp)
			if path == sp || (info != nil && si != nil && os.SameFile(info, si)) {
				return fmt.Errorf("history must differ from vector inputs and GT")
			}
		}
		historyFile, err = os.Create(*historyPath)
		if err != nil {
			return fmt.Errorf("create history: %w", err)
		}
		defer historyFile.Close()
	}
	ctx := context.Background()
	cancel := func() {}
	if *budget < 0 {
		return fmt.Errorf("negative time budget")
	}
	if *budget > 0 {
		ctx, cancel = context.WithTimeout(ctx, *budget)
	}
	defer cancel()
	history := make([]iterationReport, 0)
	var evaluationSeconds float64
	observer := func(step anchorann.Step) error {
		record := iterationReport{Step: step}
		if gt != nil {
			t := time.Now()
			rates, err := anchorann.Containment(step.Artifact, q, gt)
			if err != nil {
				return err
			}
			record.Containment = &rates
			record.EvaluationSeconds = time.Since(t).Seconds()
			evaluationSeconds += record.EvaluationSeconds
		}
		record.Artifact = nil // History keeps metrics, not every iteration's point arrays.
		history = append(history, record)
		if historyFile != nil {
			if err := json.NewEncoder(historyFile).Encode(record); err != nil {
				return err
			}
			if err := historyFile.Sync(); err != nil {
				return err
			}
		}
		return nil
	}
	clusterStart := time.Now()
	a, summary, err := anchorann.ClusterWithProgress(ctx, d, anchorann.Options{Size: *size, Seed: *seed, Method: anchorann.Method(*method), Iterations: *iterations, Workers: *workers}, observer)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			reason := "cancelled"
			if errors.Is(err, context.DeadlineExceeded) {
				reason = "time_budget"
			}
			r := report{Queries: q.N, TopK: *k, Method: anchorann.Method(*method), Seed: *seed, Iterations: *iterations, Workers: *workers, N: d.N, Size: *size, Count: d.N / *size, Dimension: d.Dim, LoadSeconds: loaded.Sub(start).Seconds(), ClusterSeconds: summary.ClusterSeconds, PreprocessSeconds: loaded.Sub(start).Seconds() + summary.ClusterSeconds, CompletedIterations: summary.CompletedIterations, StopReason: reason, Error: err.Error(), History: history, GroundTruthSeconds: gtSeconds, EvaluationSeconds: evaluationSeconds, Output: *output, GoVersion: runtime.Version(), Platform: runtime.GOOS + "/" + runtime.GOARCH}
			var usage syscall.Rusage
			syscall.Getrusage(syscall.RUSAGE_SELF, &usage)
			r.PeakRSSBytes = usage.Maxrss
			if runtime.GOOS != "darwin" {
				r.PeakRSSBytes *= 1024
			}
			if len(history) > 0 {
				last := history[len(history)-1]
				r.SSE = last.SSE
				r.Rates = last.Containment
			}
			enc := json.NewEncoder(out)
			enc.SetIndent("", "  ")
			if writeErr := enc.Encode(r); writeErr != nil {
				return fmt.Errorf("write interrupted report: %w", writeErr)
			}
		}
		return err
	}
	clustered := time.Now()
	if err := anchorann.Save(*output, a); err != nil {
		return err
	}
	saved := time.Now()
	// Maxrss covers the process through Save, including GT preparation and
	// per-step observer evaluation; timing separation does not isolate memory.
	// The final reload and evaluation below happen after this measurement.
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return fmt.Errorf("measure peak RSS: %w", err)
	}
	peak := usage.Maxrss
	if runtime.GOOS != "darwin" {
		peak *= 1024
	}
	r := report{Method: a.Method, Seed: a.Seed, Iterations: a.Iterations, Workers: *workers, N: a.N, Size: a.Size, Count: a.Count, Dimension: a.Dim, LoadSeconds: loaded.Sub(start).Seconds(), ClusterSeconds: summary.ClusterSeconds, SaveSeconds: saved.Sub(clustered).Seconds(), PreprocessSeconds: loaded.Sub(start).Seconds() + summary.ClusterSeconds + saved.Sub(clustered).Seconds(), PeakRSSBytes: peak, Output: *output, GoVersion: runtime.Version(), Platform: runtime.GOOS + "/" + runtime.GOARCH}
	r.CompletedIterations, r.StopReason, r.History, r.GroundTruthSeconds = summary.CompletedIterations, summary.StopReason, history, gtSeconds
	if a.Method == anchorann.Recursive {
		r.StopReason = "recursive_complete"
		r.ClusterSeconds = clustered.Sub(clusterStart).Seconds()
		r.PreprocessSeconds = r.LoadSeconds + r.ClusterSeconds + r.SaveSeconds
	}
	// Independently load and validate the saved file before reporting success.
	reloaded, err := anchorann.Load(*output)
	if err != nil {
		return err
	}
	r.ReloadVerified = true
	r.SSE, err = anchorann.SSE(d, reloaded)
	if err != nil {
		return err
	}
	if gt != nil {
		rates, err := anchorann.Containment(reloaded, q, gt)
		if err != nil {
			return err
		}
		r.Queries, r.TopK, r.Containment, r.Rates = q.N, *k, &rates.At1, &rates
	}
	r.EvaluationSeconds = evaluationSeconds + time.Since(saved).Seconds()
	info, err := os.Stat(*output)
	if err != nil {
		return fmt.Errorf("stat saved artifact: %w", err)
	}
	r.ArtifactBytes = info.Size()
	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	if err := enc.Encode(r); err != nil {
		return fmt.Errorf("write report: %w", err)
	}
	return nil
}

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "anchorann cluster:", err)
		os.Exit(1)
	}
}
