package fsst

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"unsafe"
)

// Table holds a trained symbol table for compression and decompression.
// Create with Train or TrainStrings; do not use the zero value.
type Table struct {
	// Symbol lookup structures
	shortCodes [65536]uint16           // 2-byte prefix -> packed [length|code]
	byteCodes  [256]uint16             // 1-byte -> packed [length|code]
	symbols    [fsstCodeMax]symbol     // code -> symbol (for decoding and training)
	hashTab    [fsstHashTabSize]symbol // direct-mapped 3-8 byte symbols

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

const fsstVersion uint64 = 20190218

var ErrBadVersion = errors.New("fsst: unsupported table version")

// newTable creates an initialized empty table.
func newTable() *Table {
	t := &Table{}
	// Codes 0-255 are escape codes for literal bytes
	for i := range 256 {
		t.symbols[i] = newSymbolFromByte(byte(i), packCodeLength(uint16(i), 1))
	}
	// Mark remaining slots unused
	unused := symbol{val: 0, icl: fsstICLFree}
	for i := 256; i < fsstCodeMax; i++ {
		t.symbols[i] = unused
	}
	// Empty hash table slots
	empty := symbol{val: 0, icl: fsstICLFree}
	for i := range fsstHashTabSize {
		t.hashTab[i] = empty
	}
	// byteCodes: each byte escapes to itself
	for i := range 256 {
		t.byteCodes[i] = packCodeLength(uint16(i), 1)
	}
	// shortCodes: fall back to first byte's code
	for i := range 65536 {
		t.shortCodes[i] = packCodeLength(uint16(i&fsstMask8), 1)
	}
	return t
}

// clearSymbols removes all learned symbols and restores lookup tables to defaults.
func (t *Table) clearSymbols() {
	for i := range t.lenHisto {
		t.lenHisto[i] = 0
	}
	for i := fsstCodeBase; i < int(fsstCodeBase)+int(t.nSymbols); i++ {
		sym := t.symbols[i]
		switch sym.length() {
		case 1:
			t.byteCodes[sym.first()] = packCodeLength(uint16(sym.first()), 1)
		case 2:
			t.shortCodes[sym.first2()] = packCodeLength(uint16(sym.first2()&fsstMask8), 1)
		default:
			idx := sym.hash() & (fsstHashTabSize - 1)
			t.hashTab[idx] = symbol{val: 0, icl: fsstICLFree}
		}
	}
	t.nSymbols = 0
}

// hashInsert adds a 3+ byte symbol to the hash table.
// Returns false if the slot is already occupied.
func (t *Table) hashInsert(sym symbol) bool {
	idx := sym.hash() & (fsstHashTabSize - 1)
	if t.hashTab[idx].icl < fsstICLFree {
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
	if int(fsstCodeBase)+int(t.nSymbols) >= fsstCodeMax {
		return false
	}
	length := sym.length()
	sym.setCodeLen(uint32(fsstCodeBase)+uint32(t.nSymbols), length)

	switch length {
	case 1:
		t.byteCodes[sym.first()] = packCodeLength(uint16(fsstCodeBase+t.nSymbols), 1)
	case 2:
		t.shortCodes[sym.first2()] = packCodeLength(uint16(fsstCodeBase+t.nSymbols), 2)
	default:
		if !t.hashInsert(sym) {
			return false
		}
	}
	t.symbols[int(fsstCodeBase)+int(t.nSymbols)] = sym
	t.nSymbols++
	t.lenHisto[length-1]++
	return true
}

// findLongestSymbol returns the code for the longest matching symbol.
func (t *Table) findLongestSymbol(sym symbol) uint16 {
	idx := sym.hash() & (fsstHashTabSize - 1)
	entry := t.hashTab[idx]
	if entry.icl <= sym.icl {
		mask := ^uint64(0) >> entry.ignoredBits()
		if entry.val == (sym.val & mask) {
			return entry.code() & fsstCodeMask
		}
	}
	if sym.length() >= 2 {
		code := t.shortCodes[sym.first2()] & fsstCodeMask
		if code >= fsstCodeBase {
			return code
		}
	}
	return t.byteCodes[sym.first()] & fsstCodeMask
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
		sym := t.symbols[int(fsstCodeBase)+i]
		length := sym.length()

		if length == 2 {
			hasConflict := false
			first2 := sym.first2()
			for k := 0; k < int(t.nSymbols); k++ {
				if k != i {
					other := t.symbols[int(fsstCodeBase)+k]
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
	t.encBuf = make([]byte, fsstChunkSize+fsstChunkPadding)
}

// rebuildIndices reconstructs byteCodes, shortCodes, and hashTab from symbols.
func (t *Table) rebuildIndices() {
	// Reset byteCodes to escape
	for i := range 256 {
		t.byteCodes[i] = packCodeLength(fsstCodeMask, 1)
	}

	// Clear hash table
	empty := symbol{val: 0, icl: fsstICLFree}
	for i := range fsstHashTabSize {
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
		t.shortCodes[i] = t.byteCodes[i&fsstMask8]
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
	for code := uint16(0); code < t.nSymbols; code++ {
		sym := t.symbols[code]
		t.decLen[code] = byte(sym.length())
		t.decSymbol[code] = sym.val
	}
}

// WriteTo serializes the Table to w.
func (t *Table) WriteTo(w io.Writer) (int64, error) {
	ver := (fsstVersion << 32) |
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
	if ver>>32 != fsstVersion {
		return n, ErrBadVersion
	}
	t.suffixLim = uint16((ver >> 16) & fsstMask8)
	t.nSymbols = uint16((ver >> 8) & fsstMask8)

	var lh [8]byte
	if _, err := io.ReadFull(r, lh[:]); err != nil {
		return n, err
	}
	n += 8
	for i := range 8 {
		t.lenHisto[i] = uint16(lh[i])
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
	t.encBuf = make([]byte, fsstChunkSize+fsstChunkPadding)
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
	if buf == nil || cap(buf) < 2*len(input)+fsstOutputPadding {
		buf = make([]byte, 2*len(input)+fsstOutputPadding)
	} else {
		buf = buf[:cap(buf)]
	}

	outPos := 0
	inputLen := len(input)
	position := 0

	// Process with safe unaligned loads while >=8 bytes remain
	for position+8 <= inputLen {
		chunkEnd := min(position+fsstChunkSize, inputLen-7)
		outPos = t.encodeChunk(buf, outPos, input[position:], chunkEnd-position)
		position = chunkEnd
	}

	// Handle tail with padded buffer
	if position < inputLen {
		tailLen := inputLen - position
		copy(t.encBuf[:tailLen], input[position:])
		outPos = t.encodeChunk(buf, outPos, t.encBuf, tailLen)
	}
	return buf[:outPos]
}

// EncodeAll compresses input and returns a newly allocated slice.
func (t *Table) EncodeAll(input []byte) []byte {
	return t.Encode(nil, input)
}

// encodeChunk compresses buf[0:end] to dst starting at dstPos.
// buf must have >=8 bytes padding after end for safe unaligned loads.
func (t *Table) encodeChunk(dst []byte, dstPos int, buf []byte, end int) int {
	suffixLim := uint8(t.suffixLim)
	position := 0

	for position < end {
		word := fsstUnalignedLoad(buf[position:])

		// Try 2-byte fast path (unique prefix)
		code := t.shortCodes[uint16(word&fsstMask16)]
		if uint8(code) < suffixLim && position+2 <= end {
			dst[dstPos] = uint8(code)
			dstPos++
			position += 2
			continue
		}

		// Try 3-8 byte hash table match
		idx := fsstHash(word&fsstMask24) & (fsstHashTabSize - 1)
		entry := t.hashTab[idx]
		if entry.icl < fsstICLFree {
			mask := ^uint64(0) >> entry.ignoredBits()
			symLen := int(entry.length())
			if entry.val == (word&mask) && position+symLen <= end {
				dst[dstPos] = uint8(entry.code())
				dstPos++
				position += symLen
				continue
			}
		}

		// Fall back to 2-byte (if valid) or 1-byte/escape
		advance := int(code >> fsstLenBits)
		if position+advance > end {
			code = t.byteCodes[uint8(word)]
			advance = 1
		}

		dst[dstPos] = uint8(code)
		dstPos++
		if code&fsstCodeBase != 0 {
			dst[dstPos] = uint8(word)
			dstPos++
		}
		position += advance
	}
	return dstPos
}

// Decode decompresses src, reusing buf if provided.
func (t *Table) Decode(buf, src []byte) []byte {
	if buf == nil {
		buf = make([]byte, 0, len(src)*4+8)
	} else {
		buf = buf[:0]
	}

	bufPos := 0
	srcPos := 0
	bufCap := cap(buf)
	if bufCap > 0 {
		buf = buf[:bufCap]
	}

	for srcPos < len(src) {
		code := src[srcPos]
		srcPos++

		if code < fsstEscapeCode {
			symLen := int(t.decLen[code])
			symVal := t.decSymbol[code]

			if bufPos+symLen > bufCap {
				newCap := max(bufCap*2, bufPos+symLen)
				newBuf := make([]byte, newCap)
				copy(newBuf, buf[:bufPos])
				buf = newBuf
				bufCap = newCap
			}

			// Write symbol bytes (unrolled for common lengths)
			switch symLen {
			case 1:
				buf[bufPos] = byte(symVal)
			case 2:
				binary.LittleEndian.PutUint16(buf[bufPos:], uint16(symVal))
			case 3:
				binary.LittleEndian.PutUint16(buf[bufPos:], uint16(symVal))
				buf[bufPos+2] = byte(symVal >> 16)
			case 4:
				binary.LittleEndian.PutUint32(buf[bufPos:], uint32(symVal))
			case 5:
				binary.LittleEndian.PutUint32(buf[bufPos:], uint32(symVal))
				buf[bufPos+4] = byte(symVal >> 32)
			case 6:
				binary.LittleEndian.PutUint32(buf[bufPos:], uint32(symVal))
				binary.LittleEndian.PutUint16(buf[bufPos+4:], uint16(symVal>>32))
			case 7:
				binary.LittleEndian.PutUint32(buf[bufPos:], uint32(symVal))
				binary.LittleEndian.PutUint16(buf[bufPos+4:], uint16(symVal>>32))
				buf[bufPos+6] = byte(symVal >> 48)
			case 8:
				binary.LittleEndian.PutUint64(buf[bufPos:], symVal)
			}
			bufPos += symLen
		} else {
			// Escape: next byte is literal
			if srcPos >= len(src) {
				break
			}
			if bufPos >= bufCap {
				newCap := max(bufCap*2, bufPos+1)
				newBuf := make([]byte, newCap)
				copy(newBuf, buf[:bufPos])
				buf = newBuf
				bufCap = newCap
			}
			buf[bufPos] = src[srcPos]
			bufPos++
			srcPos++
		}
	}
	return buf[:bufPos]
}

// DecodeAll decompresses src and returns a newly allocated slice.
func (t *Table) DecodeAll(src []byte) []byte {
	return t.Decode(nil, src)
}

// DecodeString decompresses a string.
func (t *Table) DecodeString(s string) []byte {
	return t.Decode(nil, unsafe.Slice(unsafe.StringData(s), len(s)))
}
