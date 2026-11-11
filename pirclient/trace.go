package pirclient

import (
	"context"

	"example.com/private-search/rpc"
)

const traceUnknown = -1

// StepEvent is emitted by the PIR client path.
type StepEvent struct {
	Step           string
	Phase          string
	QueryIndex     int
	StepIndex      int
	PartitionIndex int
	RequestID      string
}

// StepLogger receives step events for tracing.
type StepLogger func(StepEvent)

var stepLogger StepLogger

// SetStepLogger configures a logger for step events.
func SetStepLogger(logger StepLogger) {
	stepLogger = logger
}

func logStep(step string, phase string, queryIndex int, stepIndex int, partitionIndex int, requestID string) {
	if stepLogger == nil {
		return
	}
	stepLogger(StepEvent{
		Step:           step,
		Phase:          phase,
		QueryIndex:     queryIndex,
		StepIndex:      stepIndex,
		PartitionIndex: partitionIndex,
		RequestID:      requestID,
	})
}

func traceInfoFromContext(ctx context.Context) rpc.TraceInfo {
	if info, ok := rpc.TraceFromContext(ctx); ok {
		return info
	}
	return rpc.TraceInfo{
		QueryIndex:     traceUnknown,
		StepIndex:      traceUnknown,
		PartitionIndex: traceUnknown,
	}
}
