package rpc

import (
	"bytes"
	"testing"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

func TestMetadataReplyRoundTrip(t *testing.T) {
	msg := &MetadataReply{
		N:                   8,
		Dim:                 4,
		M:                   2,
		DbEntryByteNum:      24,
		PartitionNum:        2,
		PartitionSize:       4,
		ChunkSize:           4,
		SetSize:             4,
		FailureProbLog2:     40,
		PqNeighborCodeCount: 3,
	}

	data, err := proto.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var got MetadataReply
	if err := proto.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if !proto.Equal(msg, &got) {
		t.Fatalf("round-trip mismatch: want %+v got %+v", msg, &got)
	}
}

func TestPrivateQueryRequestRoundTrip(t *testing.T) {
	msg := &PrivateQueryRequest{
		PartitionId: 1,
		Offsets:     []uint32{1, 2, 3, 4},
	}

	data, err := proto.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var got PrivateQueryRequest
	if err := proto.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if !proto.Equal(msg, &got) {
		t.Fatalf("round-trip mismatch: want %+v got %+v", msg, &got)
	}
	if len(got.Offsets) != 4 {
		t.Fatalf("offset length mismatch: want %d got %d", 4, len(got.Offsets))
	}
}

func TestPreprocessBlockRoundTrip(t *testing.T) {
	entries := bytes.Repeat([]byte{0x04}, 3*16)
	msg := &PreprocessChunk{
		PartitionId:     2,
		ProtocolVersion: PreprocessProtocolClientHints,
		BlockIndex:      5,
		FirstEntry:      15,
		Entries:         entries,
	}
	data, err := proto.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got PreprocessChunk
	if err := proto.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !proto.Equal(msg, &got) {
		t.Fatalf("round-trip mismatch: want %+v got %+v", msg, &got)
	}
}

func TestPreprocessMessagesHaveNoSecretFields(t *testing.T) {
	for _, m := range []proto.Message{&PreprocessRequest{}, &PreprocessChunk{}} {
		fields := m.ProtoReflect().Descriptor().Fields()
		for _, name := range []string{"PrfKey", "PrimaryTags", "BackupTags", "PrimaryParity", "BackupParity", "ReplacementIdx", "ReplacementVal"} {
			if fields.ByName(protoreflect.Name(name)) != nil {
				t.Fatalf("%T still has field %s", m, name)
			}
		}
	}
}
