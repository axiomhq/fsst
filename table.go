package fsst

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"slices"
)

// Table holds a trained symbol table for compression and decompression.
// Create with Train or TrainStrings; do not use the zero value.
type Table struct {
	// Symbol lookup structures
	shortCodes [65536]uint16       // 2-byte prefix -> packed [length|code]
	byteCodes  [256]uint16         // 1-byte -> packed [length|code]
	symbols    [codeMax]symbol     // code -> symbol (for decoding and training)
	hashTab    [hashTabSize]symbol // direct-mapped 3-8 byte symbols

	// Symbol metadata
	nSymbols  uint16    // number of learned symbols (0-255)
	suffixLim uint16    // 2-byte symbols with unique prefixes: [0..suffixLim)
	lenHisto  [8]uint16 // histogram of lengths 1-8

	// Decoder tables (built at finalization)
	decLen    [255]byte   // code -> symbol length
	decSymbol [255]uint64 // code -> symbol value

	// Encoder scratch buffer
	encBuf []byte
}

// tableVersion identifies the on-disk format for Table serialization.
//
// Compatibility policy: the version is checked with an exact match.
// A reader rejects any version it does not recognize. To evolve the format,
// define a new version constant and update ReadFrom to accept both the old
// and new versions, preserving backward compatibility. Writers always emit
// the latest version.
const tableVersion uint64 = 20190218

var (
	ErrBadVersion = errors.New("fsst: unsupported table version")
	ErrCorrupted  = errors.New("fsst: corrupted table data")
)

// newTable creates an initialized empty table.
func newTable() *Table {
	t := &Table{}
	// Codes 0-255 are escape codes for literal bytes
	for i := range 256 {
		t.symbols[i] = newSymbolFromByte(byte(i), packCodeLength(uint16(i), 1))
	}
	// Mark remaining symbol slots and hash table as unused
	empty := symbol{val: 0, icl: iclFree}
	for i := 256; i < codeMax; i++ {
		t.symbols[i] = empty
	}
	for i := range hashTabSize {
		t.hashTab[i] = empty
	}
	// byteCodes: each byte escapes to itself
	for i := range 256 {
		t.byteCodes[i] = packCodeLength(uint16(i), 1)
	}
	// shortCodes: fall back to first byte's code
	for i := range 65536 {
		t.shortCodes[i] = packCodeLength(uint16(i&mask8), 1)
	}
	return t
}

// clearSymbols removes all learned symbols and restores lookup tables to defaults.
func (t *Table) clearSymbols() {
	for i := range t.lenHisto {
		t.lenHisto[i] = 0
	}
	for i := codeBase; i < int(codeBase)+int(t.nSymbols); i++ {
		sym := t.symbols[i]
		switch sym.length() {
		case 1:
			t.byteCodes[sym.first()] = packCodeLength(uint16(sym.first()), 1)
		case 2:
			t.shortCodes[sym.first2()] = packCodeLength(uint16(sym.first2()&mask8), 1)
		default:
			idx := sym.hash() & (hashTabSize - 1)
			t.hashTab[idx] = symbol{val: 0, icl: iclFree}
		}
	}
	t.nSymbols = 0
}

// hashInsert adds a 3+ byte symbol to the hash table.
// Returns false if the slot is already occupied.
func (t *Table) hashInsert(sym symbol) bool {
	idx := sym.hash() & (hashTabSize - 1)
	if t.hashTab[idx].icl < iclFree {
		return false
	}
	// Mask off unused high bytes before storing
	mask := ^uint64(0) >> sym.ignoredBits()
	t.hashTab[idx] = symbol{val: sym.val & mask, icl: sym.icl}
	return true
}

// addSymbol assigns a new code to sym and installs it into the lookup tables.
// Returns false if capacity exceeded or hash slot taken.
func (t *Table) addSymbol(sym symbol) bool {
	if int(codeBase)+int(t.nSymbols) >= codeMax {
		return false
	}
	length := sym.length()
	sym.setCodeLen(uint32(codeBase)+uint32(t.nSymbols), length)

	switch length {
	case 1:
		t.byteCodes[sym.first()] = packCodeLength(uint16(codeBase+t.nSymbols), 1)
	case 2:
		t.shortCodes[sym.first2()] = packCodeLength(uint16(codeBase+t.nSymbols), 2)
	default:
		if !t.hashInsert(sym) {
			return false
		}
	}
	t.symbols[int(codeBase)+int(t.nSymbols)] = sym
	t.nSymbols++
	t.lenHisto[length-1]++
	return true
}

// findLongestSymbol returns the code for the longest matching symbol.
func (t *Table) findLongestSymbol(sym symbol) uint16 {
	idx := sym.hash() & (hashTabSize - 1)
	entry := t.hashTab[idx]
	if entry.icl <= sym.icl {
		mask := ^uint64(0) >> entry.ignoredBits()
		if entry.val == (sym.val & mask) {
			return entry.code() & codeMask
		}
	}
	if sym.length() >= 2 {
		code := t.shortCodes[sym.first2()] & codeMask
		if code >= codeBase {
			return code
		}
	}
	return t.byteCodes[sym.first()] & codeMask
}

// finalize reorders codes by length for encoding efficiency and builds decoder tables.
// Called automatically by Train; users should not call this directly.
func (t *Table) finalize() {
	newCode := make([]uint8, 256)
	var codeStart [8]uint8
	byteLim := uint8(t.nSymbols) - uint8(t.lenHisto[0])

	// Layout: [0..byteLim) = 2-byte unique, [byteLim..nSymbols) = 1-byte
	codeStart[0] = byteLim
	codeStart[1] = 0
	for i := 1; i < 7; i++ {
		codeStart[i+1] = codeStart[i] + uint8(t.lenHisto[i])
	}

	t.suffixLim = uint16(codeStart[1])
	t.symbols[newCode[0]] = t.symbols[256]

	// Partition 2-byte symbols by prefix uniqueness
	conflictCode := int(codeStart[2])
	for i := range int(t.nSymbols) {
		sym := t.symbols[int(codeBase)+i]
		length := sym.length()

		if length == 2 {
			hasConflict := false
			first2 := sym.first2()
			for k := 0; k < int(t.nSymbols); k++ {
				if k != i {
					other := t.symbols[int(codeBase)+k]
					if other.length() > 1 && other.first2() == first2 {
						hasConflict = true
						break
					}
				}
			}
			if hasConflict {
				conflictCode--
				newCode[i] = uint8(conflictCode)
			} else {
				newCode[i] = uint8(t.suffixLim)
				t.suffixLim++
			}
		} else {
			lengthIdx := int(length - 1)
			newCode[i] = codeStart[lengthIdx]
			codeStart[lengthIdx]++
		}

		sym.setCodeLen(uint32(newCode[i]), length)
		t.symbols[int(newCode[i])] = sym
	}

	// Build encoder and decoder lookup tables
	t.rebuildIndices()
	t.buildDecoderTables()
	t.encBuf = make([]byte, chunkSize+chunkPadding)
}

// rebuildIndices reconstructs byteCodes, shortCodes, and hashTab from symbols.
func (t *Table) rebuildIndices() {
	// Reset byteCodes to escape
	for i := range 256 {
		t.byteCodes[i] = packCodeLength(codeMask, 1)
	}

	// Clear hash table
	empty := symbol{val: 0, icl: iclFree}
	for i := range hashTabSize {
		t.hashTab[i] = empty
	}

	// Apply 1-byte symbols
	for i := range int(t.nSymbols) {
		sym := t.symbols[i]
		if sym.length() == 1 {
			t.byteCodes[sym.first()] = packCodeLength(uint16(i), 1)
		}
	}

	// shortCodes mirrors byteCodes for first byte
	for i := range 65536 {
		t.shortCodes[i] = t.byteCodes[i&mask8]
	}

	// Apply 2-byte symbols
	for i := range int(t.nSymbols) {
		sym := t.symbols[i]
		if sym.length() == 2 {
			t.shortCodes[sym.first2()] = packCodeLength(uint16(i), 2)
		}
	}

	// Insert 3+ byte symbols into hash table
	for i := range int(t.nSymbols) {
		sym := t.symbols[i]
		if sym.length() >= 3 {
			t.hashInsert(sym)
		}
	}
}

// buildDecoderTables populates the flat decLen/decSymbol arrays.
func (t *Table) buildDecoderTables() {
	// A zero length marks codes outside the current table as invalid for the
	// strict decoder, including when a Table is rebuilt with fewer symbols.
	clear(t.decLen[:])
	clear(t.decSymbol[:])
	for code := uint16(0); code < t.nSymbols; code++ {
		sym := t.symbols[code]
		t.decLen[code] = byte(sym.length())
		t.decSymbol[code] = sym.val
	}
}

// WriteTo serializes the Table to w.
func (t *Table) WriteTo(w io.Writer) (int64, error) {
	ver := (tableVersion << 32) |
		(uint64(t.suffixLim) << 16) |
		(uint64(t.nSymbols) << 8) |
		1
	var (
		n    int64
		buf8 [8]byte
	)
	binary.LittleEndian.PutUint64(buf8[:], ver)
	if nn, err := w.Write(buf8[:]); err != nil {
		return n, err
	} else {
		n += int64(nn)
	}

	// Write lenHisto
	var lh [8]byte
	var hist [8]uint16
	for i := range int(t.nSymbols) {
		length := t.symbols[i].length()
		if length >= 1 && length <= 8 {
			hist[length-1]++
		}
	}
	for i := range 8 {
		lh[i] = byte(hist[i])
	}
	if nn, err := w.Write(lh[:]); err != nil {
		return n, err
	} else {
		n += int64(nn)
	}

	// Write symbol bytes
	for i := range int(t.nSymbols) {
		sym := t.symbols[i]
		symbolLength := int(sym.length())
		for byteIdx := range symbolLength {
			buf8[byteIdx] = byte(sym.val >> (8 * byteIdx))
		}
		if nn, err := w.Write(buf8[:symbolLength]); err != nil {
			return n, err
		} else {
			n += int64(nn)
		}
	}
	return n, nil
}

// ReadFrom deserializes a Table from r.
func (t *Table) ReadFrom(r io.Reader) (int64, error) {
	*t = *newTable()
	var (
		n   int64
		hdr [8]byte
	)
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return n, err
	}
	n += 8
	ver := binary.LittleEndian.Uint64(hdr[:])
	if ver>>32 != tableVersion {
		return n, ErrBadVersion
	}
	t.suffixLim = uint16((ver >> 16) & mask8)
	t.nSymbols = uint16((ver >> 8) & mask8)

	var lh [8]byte
	if _, err := io.ReadFull(r, lh[:]); err != nil {
		return n, err
	}
	n += 8
	for i := range 8 {
		t.lenHisto[i] = uint16(lh[i])
	}

	// Validate lenHisto: sum must equal nSymbols, each length must be 1-8.
	var histoSum uint16
	for i := range 8 {
		histoSum += t.lenHisto[i]
	}
	if histoSum != t.nSymbols {
		return n, ErrCorrupted
	}

	// Build length schedule
	lens := make([]uint8, t.nSymbols)
	pos := 0
	for l := 2; l <= 8; l++ {
		for range int(t.lenHisto[l-1]) {
			lens[pos] = uint8(l)
			pos++
		}
	}
	for range int(t.lenHisto[0]) {
		lens[pos] = 1
		pos++
	}

	// Read symbols
	for i := range int(t.nSymbols) {
		symbolLength := int(lens[i])
		if symbolLength < 1 || symbolLength > 8 {
			return n, ErrCorrupted
		}
		var b8 [8]byte
		if _, err := io.ReadFull(r, b8[:symbolLength]); err != nil {
			return n, err
		}
		n += int64(symbolLength)
		var symbolValue uint64
		for byteIdx := range symbolLength {
			symbolValue |= uint64(b8[byteIdx]) << (8 * byteIdx)
		}
		sym := symbol{val: symbolValue}
		sym.setCodeLen(uint32(i), uint32(symbolLength))
		t.symbols[i] = sym
	}

	t.rebuildIndices()
	t.buildDecoderTables()
	t.encBuf = make([]byte, chunkSize+chunkPadding)
	return n, nil
}

// MarshalBinary implements encoding.BinaryMarshaler.
func (t *Table) MarshalBinary() ([]byte, error) {
	var buf bytes.Buffer
	if _, err := t.WriteTo(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// UnmarshalBinary implements encoding.BinaryUnmarshaler.
func (t *Table) UnmarshalBinary(data []byte) error {
	_, err := t.ReadFrom(bytes.NewReader(data))
	return err
}

// Encode compresses input, reusing buf if provided.
// Returns compressed data (may be a different slice than buf).
func (t *Table) Encode(buf, input []byte) []byte {
	if buf == nil || cap(buf) < 2*len(input)+outputPadding {
		buf = make([]byte, 2*len(input)+outputPadding)
	} else {
		buf = buf[:cap(buf)]
	}
	return buf[:t.encode(buf, 0, input)]
}

func (t *Table) encode(buf []byte, outPos int, input []byte) int {
	inputLen := len(input)
	position := 0

	// Process with safe unaligned loads while >=8 bytes remain
	for position+8 <= inputLen {
		chunkEnd := min(position+chunkSize, inputLen-7)
		outPos = t.encodeChunk(buf, outPos, input[position:], chunkEnd-position)
		position = chunkEnd
	}

	// Handle tail with padded buffer
	if position < inputLen {
		tailLen := inputLen - position
		copy(t.encBuf[:tailLen], input[position:])
		outPos = t.encodeChunk(buf, outPos, t.encBuf, tailLen)
	}
	return outPos
}

// EncodeInto compresses input while reusing buf. It is the named-buffer form
// for integrations that also support the legacy allocating Encode API.
func (t *Table) EncodeInto(buf, input []byte) []byte {
	return t.Encode(buf, input)
}

// EncodeAll compresses input and returns a newly allocated slice.
func (t *Table) EncodeAll(input []byte) []byte {
	return t.Encode(nil, input)
}

// EncodeBatch compresses inputs independently into one code buffer. The
// returned offsets delimit each input's encoded bytes and both result buffers
// reuse dst and dstOffsets when their capacities are sufficient.
func (t *Table) EncodeBatch(dst []byte, dstOffsets []int, inputs [][]byte) ([]byte, []int) {
	dst = dst[:0]
	if cap(dstOffsets) < len(inputs)+1 {
		dstOffsets = make([]int, len(inputs)+1)
	} else {
		dstOffsets = dstOffsets[:len(inputs)+1]
	}

	outPos := 0
	for i, input := range inputs {
		dstOffsets[i] = outPos
		dst = slices.Grow(dst[:outPos], 2*len(input)+outputPadding)
		dst = dst[:cap(dst)]
		outPos = t.encode(dst, outPos, input)
	}
	dstOffsets[len(inputs)] = outPos
	return dst[:outPos], dstOffsets
}

// encodeChunk compresses buf[0:end] into dst starting at dstPos using a
// three-tier lookup strategy, ordered by expected match length:
//
//  1. 2-byte fast path: symbols whose 2-byte prefix is unique (code < suffixLim)
//     can be emitted without further checks. This is the most common hit for
//     tables with many unique 2-byte patterns.
//  2. 3-8 byte hash lookup: a direct-mapped hash table keyed on the first 3
//     bytes. If the full symbol matches (after masking unused high bytes), emit it.
//  3. Fallback: use the 2-byte shortCodes table (which may yield a 2-byte or
//     1-byte match), or emit a 1-byte/escape code.
//
// buf must have >=8 bytes of padding after end for safe unaligned 8-byte loads.
func (t *Table) encodeChunk(dst []byte, dstPos int, buf []byte, end int) int {
	suffixLim := uint8(t.suffixLim)
	position := 0

	for position < end {
		// Load 8 bytes starting at position (safe due to padding guarantee).
		word := unalignedLoad(buf[position:])

		// Tier 1: 2-byte fast path for symbols with unique prefixes.
		// suffixLim partitions shortCodes: codes below it have unique 2-byte
		// prefixes and need no further disambiguation.
		code := t.shortCodes[uint16(word&mask16)]
		if uint8(code) < suffixLim && position+2 <= end {
			dst[dstPos] = uint8(code)
			dstPos++
			position += 2
			continue
		}

		// Tier 2: 3-8 byte hash table lookup (direct-mapped, keyed on first 3 bytes).
		idx := hashWord(word&mask24) & (hashTabSize - 1)
		entry := t.hashTab[idx]
		if entry.icl < iclFree {
			// Mask off bytes beyond the symbol's length before comparing.
			mask := ^uint64(0) >> uint32(entry.icl&mask16)
			symLen := int(entry.icl >> 28)
			if entry.val == (word&mask) && position+symLen <= end {
				dst[dstPos] = uint8(entry.icl >> 16)
				dstPos++
				position += symLen
				continue
			}
		}

		// Tier 3: fall back to 2-byte shortCode (if valid) or 1-byte/escape.
		// The top bits of code encode the advance length (1 or 2).
		advance := int(code >> lenBits)
		if position+advance > end {
			code = t.byteCodes[uint8(word)]
			advance = 1
		}

		dst[dstPos] = uint8(code)
		dstPos++
		// If code >= codeBase, this is an escape: emit the literal byte too.
		if code&codeBase != 0 {
			dst[dstPos] = uint8(word)
			dstPos++
		}
		position += advance
	}
	return dstPos
}

// Decode decompresses src, reusing buf if provided.
func (t *Table) Decode(buf, src []byte) []byte {
	srcLen := len(src)
	// Worst case: every code is an escape (2 src bytes → 1 output byte),
	// best case: every code is an 8-byte symbol (1 src byte → 8 output bytes).
	// Use 2x+8 as a reasonable estimate that limits over-allocation while
	// avoiding regrowth for typical inputs.
	needed := srcLen*2 + 8
	if buf == nil {
		buf = make([]byte, needed)
	} else {
		if cap(buf) < needed {
			buf = make([]byte, needed)
		} else {
			buf = buf[:cap(buf)]
		}
	}

	bufPos := 0
	srcPos := 0
	bufCap := len(buf)

	for srcPos < srcLen {
		// Fast inner loop: process codes while output buffer has room for
		// 8-byte writes. This avoids per-iteration capacity checks.
		for srcPos < srcLen && bufPos+8 <= bufCap {
			code := src[srcPos]
			srcPos++

			if code < escapeCode {
				// Always write 8 bytes; only advance bufPos by the
				// actual symbol length. Trailing bytes are harmless
				// and will be overwritten by subsequent symbols.
				binary.LittleEndian.PutUint64(buf[bufPos:bufPos+8], t.decSymbol[code])
				bufPos += int(t.decLen[code])
			} else {
				// Escape: next byte is literal
				if srcPos >= srcLen {
					return buf[:bufPos]
				}
				buf[bufPos] = src[srcPos]
				bufPos++
				srcPos++
			}
		}

		// Grow buffer if needed
		if srcPos < srcLen && bufPos+8 > bufCap {
			newCap := max(bufCap*2, bufPos+srcLen*8+8)
			newBuf := make([]byte, newCap)
			copy(newBuf, buf[:bufPos])
			buf = newBuf
			bufCap = newCap
		}
	}
	return buf[:bufPos]
}

// DecodeInto decompresses src while reusing buf. It is the named-buffer form
// for integrations that also support the legacy Decode API.
func (t *Table) DecodeInto(buf, src []byte) []byte {
	return t.Decode(buf, src)
}

// DecodeIntoExact decompresses src into caller-provided storage and requires
// the decoded output to contain exactly decodedLen bytes. The existing length
// of buf is ignored; its capacity must be at least decodedLen. DecodeIntoExact
// does not allocate and src must not overlap buf.
//
// It returns io.ErrShortBuffer when buf is too small and ErrCorrupted when src
// is malformed or its decoded length does not equal decodedLen.
func (t *Table) DecodeIntoExact(buf, src []byte, decodedLen int) ([]byte, error) {
	if decodedLen < 0 {
		return buf[:0], ErrCorrupted
	}
	if cap(buf) < decodedLen {
		return buf[:0], io.ErrShortBuffer
	}

	dst := buf[:decodedLen]
	n, err := t.decodeIntoExactKernel(dst, src)
	if err != nil || n != decodedLen {
		return buf[:0], ErrCorrupted
	}
	return dst, nil
}

// DecodeIntoExactTrusted is DecodeIntoExact for a code stream that was
// already validated against t. It skips the code-validity pass, but still
// checks the decoded size and never reads or writes outside src or buf.
//
// This is intended for storage readers that validate immutable encoded pages
// once when loading them and decode the same page repeatedly. Callers must not
// mutate src after validation.
func (t *Table) DecodeIntoExactTrusted(buf, src []byte, decodedLen int) ([]byte, error) {
	if decodedLen < 0 {
		return buf[:0], ErrCorrupted
	}
	if cap(buf) < decodedLen {
		return buf[:0], io.ErrShortBuffer
	}

	dst := buf[:decodedLen]
	n, err := t.decodeIntoExactTrustedKernel(dst, src)
	if err != nil || n != decodedLen {
		return buf[:0], ErrCorrupted
	}
	return dst, nil
}

// decodeIntoExactSafe is the portable exact-size decoder and reference
// implementation. It never writes beyond len(dst).
func (t *Table) decodeIntoExactSafe(dst, src []byte) (int, error) {
	dstPos := 0
	for srcPos := 0; srcPos < len(src); {
		code := src[srcPos]
		srcPos++
		if code == escapeCode {
			if srcPos >= len(src) || dstPos >= len(dst) {
				return 0, ErrCorrupted
			}
			dst[dstPos] = src[srcPos]
			dstPos++
			srcPos++
			continue
		}
		if uint16(code) >= t.nSymbols {
			return 0, ErrCorrupted
		}

		symbolLen := int(t.decLen[code])
		if symbolLen == 0 || symbolLen > len(dst)-dstPos {
			return 0, ErrCorrupted
		}
		if len(dst)-dstPos >= 8 {
			binary.LittleEndian.PutUint64(dst[dstPos:dstPos+8], t.decSymbol[code])
		} else {
			var symbol [8]byte
			binary.LittleEndian.PutUint64(symbol[:], t.decSymbol[code])
			copy(dst[dstPos:], symbol[:symbolLen])
		}
		dstPos += symbolLen
	}
	return dstPos, nil
}

// DecodeAll decompresses src and returns a newly allocated slice.
func (t *Table) DecodeAll(src []byte) []byte {
	return t.Decode(nil, src)
}

// DecodeBatch decompresses multiple compressed strings packed contiguously in
// src. srcOffsets must have n+1 entries where srcOffsets[i] to srcOffsets[i+1]
// delimits the i-th compressed string. The first offset must be 0 and the last
// must equal len(src).
//
// dst and dstOffsets are reused when they have enough capacity. dstOffsets may
// be the same slice as srcOffsets; dst must not overlap src. The returned
// offsets have n+1 entries marking each decompressed string's boundaries.
//
// This avoids per-string allocation and function-call overhead, keeping the
// decoder tables hot in cache across all strings.
func (t *Table) DecodeBatch(dst []byte, dstOffsets []int, src []byte, srcOffsets []int) ([]byte, []int, error) {
	dst = dst[:0]
	dstOffsets = dstOffsets[:0]
	if len(srcOffsets) == 0 || srcOffsets[0] != 0 {
		return dst, dstOffsets, ErrCorrupted
	}
	previous := 0
	for _, offset := range srcOffsets[1:] {
		if offset < previous || offset > len(src) {
			return dst, dstOffsets, ErrCorrupted
		}
		previous = offset
	}
	if previous != len(src) {
		return dst, dstOffsets, ErrCorrupted
	}

	n := len(srcOffsets) - 1
	if cap(dstOffsets) < n+1 {
		dstOffsets = make([]int, n+1)
	} else {
		dstOffsets = dstOffsets[:n+1]
	}

	for i := range n {
		srcStart, srcEnd := srcOffsets[i], srcOffsets[i+1]
		dstOffsets[i] = len(dst)
		for srcPos := srcStart; srcPos < srcEnd; {
			code := src[srcPos]
			srcPos++
			if code < escapeCode {
				symbolLength := int(t.decLen[code])
				start := len(dst)
				if cap(dst)-start >= 8 {
					dst = dst[:start+8]
					binary.LittleEndian.PutUint64(dst[start:start+8], t.decSymbol[code])
					dst = dst[:start+symbolLength]
					continue
				}
				var symbol [8]byte
				binary.LittleEndian.PutUint64(symbol[:], t.decSymbol[code])
				dst = append(dst, symbol[:symbolLength]...)
				continue
			}
			if srcPos >= srcEnd {
				return dst, dstOffsets[:i+1], ErrCorrupted
			}
			dst = append(dst, src[srcPos])
			srcPos++
		}
	}

	dstOffsets[n] = len(dst)
	return dst, dstOffsets, nil
}

// DecodeString decompresses a string.
func (t *Table) DecodeString(s string) []byte {
	return t.Decode(nil, []byte(s))
}
