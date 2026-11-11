package rpc

import (
	"context"
	"fmt"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/metadata"
)

// TraceInfo carries client-side tracing context across RPC calls.
// Use -1 for unknown indices.
type TraceInfo struct {
	QueryIndex     int
	StepIndex      int
	PartitionIndex int
}

type traceKey struct{}
const requestIDHeader = "x-request-id"

var requestCounter uint64

// WithTrace attaches tracing info to a context.
func WithTrace(ctx context.Context, info TraceInfo) context.Context {
	return context.WithValue(ctx, traceKey{}, info)
}

// TraceFromContext reads tracing info from a context.
func TraceFromContext(ctx context.Context) (TraceInfo, bool) {
	info, ok := ctx.Value(traceKey{}).(TraceInfo)
	return info, ok
}

// NewRequestID generates a unique request id for correlating client/server traces.
func NewRequestID() string {
	seq := atomic.AddUint64(&requestCounter, 1)
	return fmt.Sprintf("%d-%d", time.Now().UnixNano(), seq)
}

// WithRequestID attaches a request id to an outgoing gRPC context.
func WithRequestID(ctx context.Context, requestID string) context.Context {
	if requestID == "" {
		return ctx
	}
	md, ok := metadata.FromOutgoingContext(ctx)
	if !ok {
		md = metadata.New(nil)
	} else {
		md = md.Copy()
	}
	md.Set(requestIDHeader, requestID)
	return metadata.NewOutgoingContext(ctx, md)
}

// RequestIDFromContext extracts a request id from gRPC metadata.
func RequestIDFromContext(ctx context.Context) string {
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		if vals := md.Get(requestIDHeader); len(vals) > 0 {
			return vals[0]
		}
	}
	if md, ok := metadata.FromOutgoingContext(ctx); ok {
		if vals := md.Get(requestIDHeader); len(vals) > 0 {
			return vals[0]
		}
	}
	return ""
}
