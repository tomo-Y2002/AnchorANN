package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"privatesearch/anchorann"
)

func fixture(t *testing.T, rows int) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "vectors.fvecs")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	for i := 0; i < rows; i++ {
		if err := binary.Write(f, binary.LittleEndian, uint32(2)); err != nil {
			t.Fatal(err)
		}
		if err := binary.Write(f, binary.LittleEndian, []float32{float32(i), float32(i * 2)}); err != nil {
			t.Fatal(err)
		}
	}
	return path
}

func TestRun(t *testing.T) {
	input, queries := fixture(t, 64), fixture(t, 2)
	for _, method := range []string{"recursive", "global", "common"} {
		output := filepath.Join(t.TempDir(), "clusters.anchor")
		args := []string{"-input", input, "-output", output, "-seed", "-7", "-method", method, "-workers", "2", "-queries", queries, "-query-limit", "2"}
		var buf bytes.Buffer
		if err := run(args, &buf); err != nil {
			t.Fatal(err)
		}
		var r report
		if err := json.Unmarshal(buf.Bytes(), &r); err != nil {
			t.Fatal(err)
		}
		if r.N != 64 || r.Size != 32 || r.Count != 2 || r.Dimension != 2 || r.Seed != -7 || r.Method != anchorann.Method(method) || r.Iterations != 6 || !r.ReloadVerified || r.Queries != 2 || r.TopK != 10 || r.Containment == nil {
			t.Fatalf("invalid report: %+v", r)
		}
		if r.PreprocessSeconds <= 0 || r.PeakRSSBytes <= 0 || r.SSE <= 0 || r.ArtifactBytes <= 0 || r.EvaluationSeconds <= 0 {
			t.Fatalf("missing measurements: %+v", r)
		}
		a, err := anchorann.Load(output)
		if err != nil || a.Seed != -7 || a.Size != 32 || a.Method != anchorann.Method(method) {
			t.Fatalf("invalid CLI artifact: %+v %v", a, err)
		}
	}
	var buf bytes.Buffer
	if err := run([]string{"-input", input, "-output", filepath.Join(t.TempDir(), "prefix.anchor"), "-limit", "32"}, &buf); err != nil {
		t.Fatal(err)
	}
	var r report
	if err := json.Unmarshal(buf.Bytes(), &r); err != nil || r.N != 32 || r.Count != 1 || r.Containment != nil {
		t.Fatalf("prefix report: %+v %v", r, err)
	}
}

func TestRunRejectsInvalidInput(t *testing.T) {
	input := fixture(t, 7)
	for _, tc := range []struct {
		name, want string
		extra      []string
	}{
		{"remainder", "remainder=1", []string{"-s", "3"}},
		{"zero size", "s=0", []string{"-s", "0"}},
		{"method", "options", []string{"-s", "1", "-method", "bad"}},
		{"negative limit", "negative", []string{"-limit", "-1"}},
		{"missing input", "open vectors", []string{"-input", filepath.Join(t.TempDir(), "absent.fvecs")}},
		{"missing parent", "create artifact", []string{"-s", "1", "-output", filepath.Join(t.TempDir(), "absent", "data.anchor")}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			output := filepath.Join(t.TempDir(), "clusters.anchor")
			args := append([]string{"-input", input, "-output", output}, tc.extra...)
			var buf bytes.Buffer
			if err := run(args, &buf); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("run()=%v; want %q", err, tc.want)
			}
			if buf.Len() != 0 {
				t.Fatal("failure emitted success report")
			}
			if _, err := os.Stat(output); !os.IsNotExist(err) {
				t.Fatalf("invalid input created artifact: %v", err)
			}
		})
	}
	for _, args := range [][]string{nil, {"-unknown"}, {"-input", input}, {"-input", input, "-output", input}, {"-input", input, "-output", filepath.Join(filepath.Dir(input), ".", "vectors.fvecs")}, {"-input", input, "-output", "unused.anchor", "positional"}} {
		if err := run(args, &bytes.Buffer{}); err == nil {
			t.Fatalf("accepted arguments %v", args)
		}
	}
	alias := filepath.Join(t.TempDir(), "alias.fvecs")
	if err := os.Link(input, alias); err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"-input", input, "-output", alias}, &bytes.Buffer{}); err == nil {
		t.Fatal("accepted input/output hard link")
	}
}

type failingWriter struct{}

func (failingWriter) Write([]byte) (int, error) { return 0, errors.New("report unavailable") }

func TestRunEvaluationAndReportErrors(t *testing.T) {
	input := fixture(t, 4)
	for _, extra := range [][]string{{"-queries", filepath.Join(t.TempDir(), "absent.fvecs")}, {"-queries", fixture(t, 1), "-query-limit", "1", "-top-k", "10"}} {
		args := append([]string{"-input", input, "-s", "2", "-output", filepath.Join(t.TempDir(), "clusters.anchor")}, extra...)
		if err := run(args, &bytes.Buffer{}); err == nil {
			t.Fatal("accepted invalid evaluation")
		}
	}
	args := []string{"-input", input, "-s", "2", "-output", filepath.Join(t.TempDir(), "clusters.anchor")}
	if err := run(args, failingWriter{}); err == nil || !strings.Contains(err.Error(), "write report") {
		t.Fatalf("report writer error lost: %v", err)
	}
}

func TestCommonHistoryAndGTRoundtrip(t *testing.T) {
	input, queries := fixture(t, 64), fixture(t, 2)
	dir := t.TempDir()
	gt := filepath.Join(dir, "gt.json")
	for _, method := range []string{"global", "common"} {
		history := filepath.Join(dir, method+".jsonl")
		var buf bytes.Buffer
		args := []string{"-input", input, "-s", "8", "-output", filepath.Join(dir, method+".anchor"), "-queries", queries, "-query-limit", "2", "-method", method, "-history", history, "-ground-truth", gt}
		if err := run(args, &buf); err != nil {
			t.Fatal(err)
		}
		var r report
		if err := json.Unmarshal(buf.Bytes(), &r); err != nil {
			t.Fatal(err)
		}
		if r.CompletedIterations != len(r.History) || len(r.History) == 0 || r.Rates == nil || r.Rates.At1 > r.Rates.At4 || r.Rates.At4 > r.Rates.At8 {
			t.Fatalf("invalid history=%+v", r)
		}
		data, err := os.ReadFile(history)
		if err != nil {
			t.Fatal(err)
		}
		lines := bytes.Split(bytes.TrimSpace(data), []byte{'\n'})
		if len(lines) != r.CompletedIterations {
			t.Fatalf("durable history=%d completed=%d", len(lines), r.CompletedIterations)
		}
		for i, line := range lines {
			var step iterationReport
			if err := json.Unmarshal(line, &step); err != nil {
				t.Fatal(err)
			}
			if step.Iteration != i+1 || step.Containment == nil || step.Seconds <= 0 || step.EvaluationSeconds <= 0 {
				t.Fatalf("incomplete step=%+v", step)
			}
		}
	}
	var buf bytes.Buffer
	if err := run([]string{"-input", input, "-s", "8", "-limit", "32", "-output", filepath.Join(dir, "subset.anchor"), "-queries", queries, "-ground-truth", gt}, &buf); err == nil {
		t.Fatal("accepted full GT for subset")
	}
}

func TestTimeoutDoesNotPublishPartialArtifact(t *testing.T) {
	for _, method := range []string{"global", "common"} {
		dir := t.TempDir()
		output := filepath.Join(dir, "timeout.anchor")
		history := filepath.Join(dir, "timeout.jsonl")
		var buf bytes.Buffer
		args := []string{"-input", fixture(t, 64), "-s", "8", "-output", output, "-method", method, "-time-budget", "1ns", "-history", history}
		if err := run(args, &buf); err == nil {
			t.Fatal("timeout succeeded")
		}
		var r report
		if err := json.Unmarshal(buf.Bytes(), &r); err != nil {
			t.Fatal(err)
		}
		if r.StopReason != "time_budget" || r.CompletedIterations != 0 || len(r.History) != 0 || r.ReloadVerified || r.Rates != nil {
			t.Fatalf("invented completion=%+v", r)
		}
		if _, err := os.Stat(output); !os.IsNotExist(err) {
			t.Fatalf("partial artifact: %v", err)
		}
		data, err := os.ReadFile(history)
		if err != nil || len(data) != 0 {
			t.Fatalf("partial history=%q %v", data, err)
		}
	}
}

func TestGeneratedFilesCannotOverwriteInputs(t *testing.T) {
	input, queries := fixture(t, 64), fixture(t, 2)
	for _, flag := range []string{"-history", "-ground-truth"} {
		for _, source := range []string{input, queries} {
			before, err := os.ReadFile(source)
			if err != nil {
				t.Fatal(err)
			}
			args := []string{"-input", input, "-output", filepath.Join(t.TempDir(), "output.anchor"), "-queries", queries, flag, source}
			if err := run(args, &bytes.Buffer{}); err == nil {
				t.Fatalf("accepted %s input alias", flag)
			}
			after, err := os.ReadFile(source)
			if err != nil || !bytes.Equal(before, after) {
				t.Fatalf("source overwritten: %s", flag)
			}
		}
	}
}
