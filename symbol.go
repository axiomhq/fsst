package fsst

import (
	"encoding/binary"
)

const (
	lenBits  = 12
	codeBits = 9
	codeBase = 256              // First code for learned symbols (0-255 are escapes)
	codeMax  = 1 << codeBits    // 512
	codeMask = codeMax - 1      // 0x1FF

	hashLog2Size = 11
	hashTabSize  = 1 << hashLog2Size    // 2048 entries
	hashPrime    = uint64(2971215073)   // Prime for multiplicative hashing
	hashShift    = 15

	// iclFree marks unused hash table slots.
	// Layout: length=15 (impossible) at bits 28-31, code=0x1FF at bits 16-27
	iclFree = (uint64(15) << 28) | (uint64(codeMask) << 16)

	escapeCode = 255 // Code 255 indicates next byte is literal
	maxSymbols = 255 // Maximum number of learned symbols (codes 0-254)
	chunkSize  = 511 // Process input in 511-byte chunks for cache efficiency

	// Training subsampling mask (0-127 range for deterministic sampling)
	sampleMask = 127

	// Buffer padding for safe unaligned 8-byte loads at chunk boundaries
	chunkPadding = 9 // 511+9=520: allows 8-byte load at position 511 (511+8-1=518 < 520)

	// Output buffer growth factor for worst-case expansion
	// Worst case: every byte escapes (2 bytes per input byte) + small safety margin
	outputPadding = 7 // Safety margin for edge cases

	mask8  = 0xFF     // 8-bit mask (1 byte)
	mask16 = 0xFFFF   // 16-bit mask (2 bytes)
	mask24 = 0xFFFFFF // 24-bit mask (3 bytes)
)

func unalignedLoad(b []byte) uint64 { return binary.LittleEndian.Uint64(b) }
func hashWord(w uint64) uint64       { x := w * hashPrime; return x ^ (x >> hashShift) }

// packCodeLength combines a code and length into a packed uint16 used by byteCodes/shortCodes.
// Format: (length << 12) | code, where length is 1-8 and code is 0-511.
func packCodeLength(code uint16, length int) uint16 {
	return uint16(code) | uint16(length<<lenBits)
}

// symbol is the internal representation of a compression symbol (1-8 bytes).
// It packs the symbol value and metadata into two uint64 fields:
//
//	val: the actual symbol bytes in little-endian (up to 8 bytes)
//	icl: packed metadata with bit layout:
//	     bits 28-31: length (1-8)
//	     bits 16-27: code (0-511)
//	     bits 0-15:  ignoredBits = (8-length)*8, used for hash table masking
type symbol struct {
	val uint64 // Symbol bytes in little-endian
	icl uint64 // Packed: [length:4][code:12][ignoredBits:16]
}

func newSymbolFromByte(b byte, code uint16) symbol {
	return symbol{
		val: uint64(b),
		// length=1 at bits 28-31, code at bits 16-27,
		// ignoredBits=56 at bits 0-15 (masks out top 7 bytes in 8-byte loads).
		icl: (uint64(1) << 28) | (uint64(code) << 16) | 56,
	}
}

func newSymbolFromBytes(in []byte) symbol {
	length := min(len(in), 8)
	var value uint64
	for i := 0; i < length; i++ {
		value |= uint64(in[i]) << (8 * i)
	}
	sym := symbol{val: value}
	sym.setCodeLen(codeMax, uint32(length))
	return sym
}

// setCodeLen packs code and length into the icl field.
//
//	bits 28-31: length (1-8)
//	bits 16-27: code (0-511)
//	bits  0-15: ignoredBits = (8-length)*8
//
// ignoredBits is used to build a mask that zeroes out bytes beyond the
// symbol's length when comparing against an 8-byte load from the input.
// For example, a 3-byte symbol has ignoredBits=40, so the mask is
// ^uint64(0) >> 40 = 0x0000_00FF_FFFF (keeps only the low 3 bytes).
func (s *symbol) setCodeLen(code uint32, length uint32) {
	s.icl = (uint64(length) << 28) | (uint64(code) << 16) | uint64((8-length)*8)
}

func (s symbol) length() uint32      { return uint32(s.icl >> 28) }       // bits 28-31
func (s symbol) code() uint16        { return uint16((s.icl >> 16) & codeMask) } // bits 16-27
func (s symbol) ignoredBits() uint32 { return uint32(s.icl & mask16) }           // bits 0-15
func (s symbol) first() byte         { return byte(s.val & mask8) }
func (s symbol) first2() uint16      { return uint16(s.val & mask16) }
func (s symbol) hash() uint64        { return hashWord(s.val & mask24) }

func concatSymbols(a, b symbol) symbol {
	lengthA := a.length()
	lengthB := b.length()
	combinedLength := min(lengthA+lengthB, 8)
	combinedValue := (b.val << (8 * lengthA)) | a.val
	result := symbol{val: combinedValue}
	result.setCodeLen(codeMask, uint32(combinedLength))
	return result
}
