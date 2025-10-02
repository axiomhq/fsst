package fsst

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"unsafe"
)

// Table holds a trained symbol table for compression and decompression.
// A Table is created via Train and can encode or decode byte slices.
// After training, a Table can be serialized with WriteTo and restored with ReadFrom.
type Table struct {
	// Symbol lookup structures (for encoding)
	shortCodes [65536]uint16           // 2-byte prefix -> [length|code], fast unique 2B path
	byteCodes  [256]uint16             // 1-byte -> [length|code], single-byte and escape fallback
	symbols    [fsstCodeMax]symbol     // canonical code -> symbol (value+length)
	hashTab    [fsstHashTabSize]symbol // direct-mapped 3–8B symbols keyed by first 3 bytes

	// Symbol metadata
	nSymbols  uint16    // number of learned symbols (0..255)
	suffixLim uint16    // end of unique 2B region [0..suffixLim)
	lenHisto  [8]uint16 // histogram of lengths 1..8 at indices 0..7

	// Encoder state (lazy-initialized on first Encode)
	// accelReady: true when shortCodes/byteCodes/hashTab are populated for encoding.
	//             Rebuilt lazily after deserialization to avoid cost if only decoding.
	// noSuffixOpt/avoidBranch: encoding strategy flags chosen based on symbol statistics.
	// encBuf: reusable chunk buffer (fsstChunkSize+fsstChunkPadding bytes) to avoid allocation per call.
	accelReady  bool   // encoder lookup structures are ready
	noSuffixOpt bool   // enable 2-byte fast path without suffix check
	avoidBranch bool   // prefer branchless emission in encodeChunk
	encBuf      []byte // scratch chunk buffer used by Encode

	// Decoder state (lazy-initialized on first Decode)
	// decLen/decSymbol: flattened arrays for fast decoding (indexed by code).
	//                   Built lazily to avoid cost if only encoding.
	// decReady: true when decoder arrays are populated.
	decLen    [255]byte   // code → symbol length
	decSymbol [255]uint64 // code → symbol value (little-endian)
	decReady  bool        // decoder lookup tables are ready
}

// Version is the FSST format version (publication date: February 18, 2019).
const fsstVersion uint64 = 20190218

// ErrBadVersion indicates the serialized table version is not supported.
var ErrBadVersion = errors.New("fsst: unsupported table version")

// newTable initializes a new empty table with defaults.
func newTable() *Table {
	t := &Table{}
	// pseudo symbols 0..255 (escaped bytes)
	for i := range 256 {
		t.symbols[i] = newSymbolFromByte(byte(i), packCodeLength(uint16(i), 1))
	}
	// mark unused
	unused := newSymbolFromByte(0, fsstCodeMask)
	for i := 256; i < fsstCodeMax; i++ {
		t.symbols[i] = unused
	}
	// empty hash table markers
	emptySymbol := symbol{}
	emptySymbol.val = 0
	emptySymbol.icl = fsstICLFree
	for i := range fsstHashTabSize {
		t.hashTab[i] = emptySymbol
	}
	// fill byteCodes with pseudo code (escaped bytes)
	for i := range 256 {
		t.byteCodes[i] = packCodeLength(uint16(i), 1)
	}
	// fill shortCodes with pseudo code for first byte
	for i := range 65536 {
		t.shortCodes[i] = packCodeLength(uint16(i&fsstMask8), 1)
	}
	return t
}

// clearSymbols removes all learned symbols from the Table and restores the
// lookup structures (byteCodes/shortCodes/hashTab) to their default state.
// It also resets the length histogram and learned symbol count.
func (t *Table) clearSymbols() {
	for i := range t.lenHisto {
		t.lenHisto[i] = 0
	}
	for i := fsstCodeBase; i < int(fsstCodeBase)+int(t.nSymbols); i++ {
		switch t.symbols[i].length() {
		case 1:
			firstByte := t.symbols[i].first()
			t.byteCodes[firstByte] = packCodeLength(uint16(firstByte), 1)
		case 2:
			first2Bytes := t.symbols[i].first2()
			t.shortCodes[first2Bytes] = packCodeLength(uint16(first2Bytes&fsstMask8), 1)
		default:
			hashIndex := t.symbols[i].hash() & (fsstHashTabSize - 1)
			t.hashTab[hashIndex].val = 0
			t.hashTab[hashIndex].icl = fsstICLFree
		}
	}
	t.nSymbols = 0
}

// hashInsert inserts a 3+ byte symbol into the direct-mapped hash table.
// It stores the symbol with masked value (ignore high bytes) and returns
// false if the target slot is already occupied.
func (t *Table) hashInsert(sym symbol) bool {
	hashIndex := sym.hash() & (fsstHashTabSize - 1)
	taken := t.hashTab[hashIndex].icl < fsstICLFree
	if taken {
		return false
	}
	t.hashTab[hashIndex].icl = sym.icl
	// mask high ignored bits before storing
	mask := ^uint64(0) >> sym.ignoredBits()
	t.hashTab[hashIndex].val = sym.val & mask
	return true
}

// addSymbol assigns a new code to sym and installs it into the appropriate
// lookup structure based on its length:
//
//	1 byte -> byteCodes, 2 bytes -> shortCodes, 3–8 bytes -> hashTab.
//
// Returns false if capacity is exceeded or hash slot is taken.
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

// findLongestSymbol decides the longest match at cur represented as a temporary symbol.
func (t *Table) findLongestSymbol(sym symbol) uint16 {
	hashIndex := sym.hash() & (fsstHashTabSize - 1)
	hashEntry := t.hashTab[hashIndex]
	if hashEntry.icl <= sym.icl {
		mask := ^uint64(0) >> uint(hashEntry.ignoredBits())
		if hashEntry.val == (sym.val & mask) {
			return (hashEntry.code() & fsstCodeMask)
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

// finalize reorders symbol codes by length for encoding efficiency.
//
// Code layout after finalization:
//
//	[0..byteLim):          1-byte symbols (direct byteCodes lookup)
//	[byteLim..suffixLim):  2-byte symbols with unique prefixes (fast shortCodes lookup)
//	[suffixLim..nSymbols): 2-byte symbols with conflicts + 3-8 byte symbols (hash table)
//
// This ordering enables the encoder to use faster lookup paths for common cases:
// - 1-byte symbols decode directly from byteCodes array
// - 2-byte symbols without prefix conflicts decode from shortCodes without hash check
// - Remaining symbols (conflicting 2-byte and long symbols) use hash table lookup
//
// Effects: updates code assignments in symbols[], sets suffixLim accordingly,
// preserves lengths, and leaves rebuilding of fast lookup tables to rebuildIndices.
func (t *Table) finalize() {
	// Precondition: nSymbols <= 255
	newCode := make([]uint8, 256)
	var codeStart [8]uint8 // Starting code for each length group (1-8 bytes)
	byteLim := uint8(t.nSymbols) - uint8(t.lenHisto[0])

	// Initialize code ranges: 1-byte symbols get [byteLim, nSymbols)
	codeStart[0] = byteLim
	codeStart[1] = 0 // 2-byte symbols start at 0 (will be partitioned)
	for i := 1; i < 7; i++ {
		codeStart[i+1] = codeStart[i] + uint8(t.lenHisto[i])
	}

	t.suffixLim = uint16(codeStart[1])
	t.symbols[newCode[0]] = t.symbols[256]

	// Assign new codes, partitioning 2-byte symbols by prefix uniqueness
	conflictingTwoByteCode := int(codeStart[2]) // Codes for conflicting 2-byte symbols (count down)
	for i := range int(t.nSymbols) {
		sym := t.symbols[int(fsstCodeBase)+i]
		length := sym.length()

		if length == 2 {
			// Check if this 2-byte symbol has a unique prefix (no other symbols share first2)
			hasConflict := false
			first2 := sym.first2()
			for k := 0; k < int(t.nSymbols); k++ {
				if k == i {
					continue
				}
				other := t.symbols[int(fsstCodeBase)+k]
				if other.length() > 1 && other.first2() == first2 {
					hasConflict = true
					break
				}
			}

			if !hasConflict {
				// Unique prefix: assign to fast-path range [0..suffixLim)
				newCode[i] = uint8(t.suffixLim)
				t.suffixLim++
			} else {
				// Conflicting prefix: assign to slow-path range [suffixLim..codeStart[2])
				conflictingTwoByteCode--
				newCode[i] = uint8(conflictingTwoByteCode)
			}
		} else {
			// Non-2-byte symbols: assign sequentially within length group
			lengthIdx := int(length - 1)
			newCode[i] = codeStart[lengthIdx]
			codeStart[lengthIdx]++
		}

		sym.setCodeLen(uint32(newCode[i]), length)
		t.symbols[int(newCode[i])] = sym
	}
}

// WriteTo serializes the finalized Table to w using the compact FSST header format.
// Layout:
// - 8 bytes version word: (version<<32)|(suffixLim<<16)|(nSymbols<<8)|1
// - 8 bytes lenHisto (u8)
// - concatenated symbol bytes for codes [0..nSymbols) in length-group order
func (t *Table) WriteTo(w io.Writer) (int64, error) {
	// pack version
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
	// Write lenHisto derived from symbols to avoid relying on stored state
	var (
		lh   [8]byte
		hist [8]uint16
	)
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
	// symbol bytes
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

// ReadFrom deserializes a Table from r using the compact FSST header format.
func (t *Table) ReadFrom(r io.Reader) (int64, error) {
	*t = *newTable() // reset
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
	// endian marker ignored (lowest byte)
	var lh [8]byte
	if _, err := io.ReadFull(r, lh[:]); err != nil {
		return n, err
	}
	n += 8
	for i := range 8 {
		t.lenHisto[i] = uint16(lh[i])
	}
	// read symbol bytes into symbols[0..nSymbols)
	// Build code->length schedule from lenHisto
	lens := make([]uint8, t.nSymbols)
	pos := 0
	// lengths 2..8
	for l := 2; l <= 8; l++ {
		cnt := int(t.lenHisto[l-1])
		for range cnt {
			lens[pos] = uint8(l)
			pos++
		}
	}
	// then 1-byte
	cnt1 := int(t.lenHisto[0])
	for j := 0; j < cnt1; j++ {
		lens[pos] = 1
		pos++
	}
	// now read symbols accordingly
	for i := range int(t.nSymbols) {
		symbolLength := int(lens[i])
		var b8 [8]byte
		if _, err := io.ReadFull(r, b8[:symbolLength]); err != nil {
			return n, err
		}
		n += int64(symbolLength)
		// pack into symbol (little-endian)
		var symbolValue uint64
		for byteIdx := range symbolLength {
			symbolValue |= uint64(b8[byteIdx]) << (8 * byteIdx)
		}
		sym := symbol{val: symbolValue}
		sym.setCodeLen(uint32(i), uint32(symbolLength))
		t.symbols[i] = sym
	}
	t.accelReady = false
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

// rebuildIndices reconstructs byteCodes, shortCodes, and hashTab from the
// finalized symbols. It preserves existing code assignments (already set in
// symbols[i]) and only rebuilds the derived lookup structures. Safe to call
// multiple times; it is a no-op if accelReady is already true.
func (t *Table) rebuildIndices() {
	if t.accelReady {
		return
	}
	// 1) Reset to defaults
	// byteCodes default to ESCAPE (fsstCodeMask) with len=1 marker
	for i := range 256 {
		t.byteCodes[i] = packCodeLength(fsstCodeMask, 1)
	}
	// Clear hash table
	empty := symbol{}
	empty.val = 0
	empty.icl = fsstICLFree
	for i := range fsstHashTabSize {
		t.hashTab[i] = empty
	}

	// 2) Apply single-byte symbols to byteCodes
	for i := range int(t.nSymbols) {
		sym := t.symbols[i]
		length := sym.length()
		if length == 1 {
			firstByte := sym.first()
			t.byteCodes[firstByte] = packCodeLength(uint16(i), 1)
		}
	}

	// 3) Initialize shortCodes to mirror byteCodes of the first byte
	for i := range 65536 {
		first := i & fsstMask8
		t.shortCodes[i] = t.byteCodes[first]
	}

	// 4) Apply two-byte symbols to shortCodes
	for i := range int(t.nSymbols) {
		sym := t.symbols[i]
		if sym.length() == 2 {
			t.shortCodes[sym.first2()] = packCodeLength(uint16(i), 2)
		}
	}

	// 5) Insert 3+ byte symbols into hash table
	for i := range int(t.nSymbols) {
		sym := t.symbols[i]
		if sym.length() >= 3 {
			_ = t.hashInsert(sym)
		}
	}

	t.accelReady = true
}

// Encode compresses input, optionally reusing buf for output.
// buf can be nil or undersized; it will be grown as needed.
// Returns the compressed data (may have different backing array than buf).
func (t *Table) Encode(buf, input []byte) []byte {
	// Lazy-initialize encoder structures
	if t.encBuf == nil {
		if !t.accelReady {
			t.rebuildIndices()
		}
		t.noSuffixOpt, t.avoidBranch = chooseVariant(t)
		t.encBuf = make([]byte, fsstChunkSize+fsstChunkPadding)
	}

	if buf == nil {
		buf = make([]byte, 2*len(input)+fsstOutputPadding)
	} else if cap(buf) < 2*len(input)+fsstOutputPadding {
		buf = make([]byte, 2*len(input)+fsstOutputPadding)
	} else {
		buf = buf[:cap(buf)]
	}

	outPos := 0
	chunkBuf := t.encBuf
	byteLim := uint8(t.nSymbols) - uint8(t.lenHisto[0])

	// Process input in chunks for cache efficiency
	for chunkStart := 0; chunkStart < len(input); {
		chunk := min(len(input)-chunkStart, fsstChunkSize)
		copy(chunkBuf[:chunk], input[chunkStart:chunkStart+chunk])
		chunkBuf[chunk] = 0 // Zero terminator + padding for unaligned loads
		outPos = t.encodeChunk(buf, outPos, chunkBuf, chunk, byteLim)
		chunkStart += chunk
	}
	return buf[:outPos]
}

// EncodeAll compresses input and returns a newly allocated byte slice.
func (t *Table) EncodeAll(input []byte) []byte {
	return t.Encode(nil, input)
}

// encodeChunk compresses a single chunk using index-based writes.
// dst is the output buffer, dstPos is the starting write position.
// buf must have at least 8 bytes of padding after end for safe unaligned loads.
// Returns the new output position.
//
// Match order (fastest first):
// 1) Optional fast 2-byte unique-prefix path (when noSuffixOpt is true)
// 2) 3–8 byte hash-table match (longest available)
// 3) 2-byte short-code path (when within suffixLim)
// 4) 1-byte or escape fallback
//
// Strategy flags:
// - noSuffixOpt: skip suffix checking for most 2-byte symbols (higher hit rate)
// - avoidBranch: use branchless emission when distribution makes branches costly
func (t *Table) encodeChunk(dst []byte, dstPos int, buf []byte, end int, byteLim uint8) int {
	position := 0

	for position < end {
		word := fsstUnalignedLoad(buf[position:])
		code := t.shortCodes[uint16(word&fsstMask16)]

		// Fast path: 2-byte code without suffix check
		if t.noSuffixOpt && uint8(code) < uint8(t.suffixLim) {
			dst[dstPos] = uint8(code)
			dstPos++
			position += 2
			continue
		}

		// Check hash table for 3+ byte matches
		prefix24 := word & fsstMask24 // First 3 bytes for hash lookup
		hashIndex := fsstHash(prefix24) & (fsstHashTabSize - 1)
		hashSymbol := t.hashTab[hashIndex]
		escapeByte := uint8(word) // First byte to emit if no match found

		// Build mask to compare only relevant bytes (mask out high bytes beyond symbol length)
		// Example: for 3-byte symbol, mask = 0x0000000000FFFFFF (ignore top 5 bytes)
		symbolMask := ^uint64(0) >> hashSymbol.ignoredBits()
		maskedWord := word & symbolMask

		if hashSymbol.icl < fsstICLFree && hashSymbol.val == maskedWord {
			// Hash table hit: 3+ byte symbol match
			dst[dstPos] = uint8(hashSymbol.code())
			dstPos++
			position += int(hashSymbol.length())
		} else if t.avoidBranch {
			// Branchless path: emit code and conditional escape
			// code format: [length:4 bits][code:12 bits]
			// Extract length to advance position
			outputCode := uint8(code)
			dst[dstPos] = outputCode
			dstPos++
			// If code >= 256, it's an escape marker (emit literal byte)
			if (code & fsstCodeBase) != 0 {
				dst[dstPos] = escapeByte
				dstPos++
			}
			position += int(code >> fsstLenBits) // Extract length field
		} else if uint8(code) < byteLim {
			// 2-byte code (after checking for longer match)
			dst[dstPos] = uint8(code)
			dstPos++
			position += 2
		} else {
			// 1-byte code or escape
			outputCode := uint8(code)
			dst[dstPos] = outputCode
			dstPos++
			// If code >= 256, it's an escape marker (emit literal byte)
			if (code & fsstCodeBase) != 0 {
				dst[dstPos] = escapeByte
				dstPos++
			}
			position++
		}
	}
	return dstPos
}

// Decode decompresses src, optionally reusing buf for output.
// buf can be nil or undersized; it will be grown as needed.
// Returns the decompressed data (may have different backing array than buf).
func (t *Table) Decode(buf, src []byte) []byte {
	// Lazy-initialize decoder structures
	if !t.decReady {
		for code := uint16(0); code < t.nSymbols; code++ {
			sym := t.symbols[code]
			t.decLen[code] = byte(sym.length())
			t.decSymbol[code] = sym.val
		}
		t.decReady = true
	}

	if buf == nil {
		buf = make([]byte, 0, len(src)*4+8)
	} else {
		buf = buf[:0] // Reset length but keep capacity
	}

	bufPos := 0
	srcPos := 0
	bufCap := cap(buf)

	// Extend to max capacity upfront to reduce slice bound checks
	if bufCap > 0 {
		buf = buf[:bufCap]
	}

	for srcPos < len(src) {
		code := src[srcPos]
		srcPos++

		if code < fsstEscapeCode {
			// Decode learned symbol
			symbolLength := int(t.decLen[code])
			symbolValue := t.decSymbol[code]

			// Grow buffer if needed
			if bufPos+symbolLength > bufCap {
				newCap := max(bufCap*2, bufPos+symbolLength)
				newBuf := make([]byte, newCap)
				copy(newBuf, buf[:bufPos])
				buf = newBuf
				bufCap = newCap
			}

			// Direct write: unrolled for common lengths
			switch symbolLength {
			case 1:
				buf[bufPos] = byte(symbolValue)
			case 2:
				binary.LittleEndian.PutUint16(buf[bufPos:], uint16(symbolValue))
			case 3:
				binary.LittleEndian.PutUint16(buf[bufPos:], uint16(symbolValue))
				buf[bufPos+2] = byte(symbolValue >> 16)
			case 4:
				binary.LittleEndian.PutUint32(buf[bufPos:], uint32(symbolValue))
			case 5:
				binary.LittleEndian.PutUint32(buf[bufPos:], uint32(symbolValue))
				buf[bufPos+4] = byte(symbolValue >> 32)
			case 6:
				binary.LittleEndian.PutUint32(buf[bufPos:], uint32(symbolValue))
				binary.LittleEndian.PutUint16(buf[bufPos+4:], uint16(symbolValue>>32))
			case 7:
				binary.LittleEndian.PutUint32(buf[bufPos:], uint32(symbolValue))
				binary.LittleEndian.PutUint16(buf[bufPos+4:], uint16(symbolValue>>32))
				buf[bufPos+6] = byte(symbolValue >> 48)
			case 8:
				binary.LittleEndian.PutUint64(buf[bufPos:], symbolValue)
			}
			bufPos += symbolLength
		} else {
			// Escape code: next byte is literal
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

// DecodeAll decompresses src and returns a newly allocated byte slice with the result.
func (t *Table) DecodeAll(src []byte) []byte {
	return t.Decode(nil, src)
}

// DecodeString decompresses a string and returns a newly allocated byte slice.
func (t *Table) DecodeString(s string) []byte {
	return t.Decode(nil, unsafe.Slice(unsafe.StringData(s), len(s)))
}

// chooseVariant selects the best encoding strategy based on symbol statistics.
// Returns flags for two encoding optimizations:
//   - noSuffixOpt: skip suffix checking for 2-byte symbols (when >65% are 2-byte and >95% have no suffix conflicts)
//   - avoidBranch: use branchless encoding (helps when symbol distribution is balanced)
//
// Thresholds are empirically derived from benchmarking on text corpora.
func chooseVariant(t *Table) (noSuffixOpt, avoidBranch bool) {
	// noSuffixOpt: most symbols are 2-byte with few conflicts
	// Check: 2-byte symbols > 65% of total AND non-conflicting 2-byte > 95% of 2-byte symbols
	if 100*int(t.lenHisto[1]) > 65*int(t.nSymbols) && 100*int(t.suffixLim) > 95*int(t.lenHisto[1]) {
		noSuffixOpt = true
		return
	}

	// avoidBranch: symbol distribution is balanced, causing poor branch prediction
	// Use branchless (predicated) execution to avoid CPU pipeline stalls
	//
	// Heuristic checks (empirically derived from benchmarking):
	//   1. Moderate number of 1-byte symbols (24-92): not too rare, not too common
	//   2. Either few 1-byte (<43) OR few long symbols (7-8 bytes < 29 combined)
	//   3. Either moderate 1-byte (<72) OR few 3-byte symbols (<72)
	//
	// These thresholds detect "balanced" distributions where branch predictor
	// struggles to learn patterns, making branchless code faster despite extra work.
	if (t.lenHisto[0] > 24 && t.lenHisto[0] < 92) &&
		(t.lenHisto[0] < 43 || t.lenHisto[6]+t.lenHisto[7] < 29) &&
		(t.lenHisto[0] < 72 || t.lenHisto[2] < 72) {
		avoidBranch = true
	}
	return
}
