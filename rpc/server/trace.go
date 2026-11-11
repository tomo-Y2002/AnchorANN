package server

// StepEvent is emitted by the PIR server for tracing.
type StepEvent struct {
	Step           string
	Phase          string
	RequestID      string
	QueryIndex     int
	StepIndex      int
	PartitionIndex int
}

// StepLogger receives step events for tracing.
type StepLogger func(StepEvent)

var stepLogger StepLogger

// SetStepLogger configures a logger for step events.
func SetStepLogger(logger StepLogger) {
	stepLogger = logger
}

func logStep(step string, phase string, requestID string, queryIndex int, stepIndex int, partitionIndex int) {
	if stepLogger == nil {
		return
	}
	stepLogger(StepEvent{
		Step:           step,
		Phase:          phase,
		RequestID:      requestID,
		QueryIndex:     queryIndex,
		StepIndex:      stepIndex,
		PartitionIndex: partitionIndex,
	})
}
