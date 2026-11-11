package rpc

// PreprocessProtocolClientHints is the StartPreprocess protocol in which the
// server streams the partition DB in public, fixed-size blocks and the client
// builds every hint itself. Version 0 (unset) is the retired protocol in which
// the client sent its PRF key and hint tags and the server built the hints.
const PreprocessProtocolClientHints uint32 = 1
