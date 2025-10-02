package fsst

/*
#cgo LDFLAGS: -Lmojo -lfsst_decoder -Lmojo/.magic/envs/default/lib -lKGENCompilerRTShared -Wl,-rpath,${SRCDIR}/mojo/.magic/envs/default/lib
#include <stdint.h>
#include <stdlib.h>

// Forward declarations of Mojo-compiled functions
extern void* fsst_decoder_create(uint8_t* table_data, int64_t table_len);
extern int64_t fsst_decoder_decode(void* decoder, uint8_t* src, int64_t src_len, uint8_t* dst, int64_t dst_capacity);
extern void fsst_decoder_destroy(void* decoder);
*/
import "C"
import (
	"bytes"
	"errors"
	"unsafe"
)

// SIMDDecoder is a Mojo-backed FSST decoder with SIMD optimizations.
// It owns its state and reads the serialized table format.
type SIMDDecoder struct {
	handle unsafe.Pointer // opaque pointer to Mojo SIMDDecoder
}

// NewSIMDDecoder creates a new SIMD decoder from serialized table bytes.
// The table bytes should be in the same format as Table.WriteTo() produces.
//
// Returns error if the table format is invalid or Mojo decoder creation fails.
func NewSIMDDecoder(tableBytes []byte) (*SIMDDecoder, error) {
	if len(tableBytes) < 16 {
		return nil, errors.New("fsst: table too short")
	}

	var tablePtr *C.uint8_t
	if len(tableBytes) > 0 {
		tablePtr = (*C.uint8_t)(unsafe.Pointer(&tableBytes[0]))
	}

	handle := C.fsst_decoder_create(tablePtr, C.int64_t(len(tableBytes)))
	if handle == nil {
		return nil, errors.New("fsst: failed to create SIMD decoder")
	}

	return &SIMDDecoder{handle: handle}, nil
}

// NewSIMDDecoderFromTable creates a SIMD decoder from a trained Table.
// This is a convenience wrapper that serializes the table and creates the decoder.
func NewSIMDDecoderFromTable(t *Table) (*SIMDDecoder, error) {
	var buf bytes.Buffer
	if _, err := t.WriteTo(&buf); err != nil {
		return nil, err
	}
	return NewSIMDDecoder(buf.Bytes())
}

// Decode decompresses src using the SIMD decoder, optionally reusing buf for output.
// buf can be nil or undersized; it will be grown as needed.
// Returns the decompressed data (may have different backing array than buf).
func (d *SIMDDecoder) Decode(buf, src []byte) []byte {
	if d.handle == nil {
		return nil
	}

	// Allocate output buffer with reasonable capacity
	if buf == nil {
		buf = make([]byte, len(src)*4+8)
	} else if cap(buf) < len(src)*4+8 {
		buf = make([]byte, len(src)*4+8)
	} else {
		buf = buf[:cap(buf)]
	}

	var srcPtr *C.uint8_t
	var dstPtr *C.uint8_t

	if len(src) > 0 {
		srcPtr = (*C.uint8_t)(unsafe.Pointer(&src[0]))
	}
	if len(buf) > 0 {
		dstPtr = (*C.uint8_t)(unsafe.Pointer(&buf[0]))
	}

	// Call Mojo decoder
	result := C.fsst_decoder_decode(
		d.handle,
		srcPtr,
		C.int64_t(len(src)),
		dstPtr,
		C.int64_t(cap(buf)),
	)

	// Check for error (buffer too small)
	if result < 0 {
		// Grow buffer and retry
		newCap := len(src) * 8
		buf = make([]byte, newCap)
		dstPtr = (*C.uint8_t)(unsafe.Pointer(&buf[0]))

		result = C.fsst_decoder_decode(
			d.handle,
			srcPtr,
			C.int64_t(len(src)),
			dstPtr,
			C.int64_t(newCap),
		)
	}

	if result < 0 {
		return nil
	}

	return buf[:result]
}

// DecodeAll decompresses src using the SIMD decoder and returns a newly allocated byte slice.
func (d *SIMDDecoder) DecodeAll(src []byte) []byte {
	return d.Decode(nil, src)
}

// Close frees the SIMD decoder and its resources.
// The decoder must not be used after calling Close.
func (d *SIMDDecoder) Close() {
	if d.handle != nil {
		C.fsst_decoder_destroy(d.handle)
		d.handle = nil
	}
}
