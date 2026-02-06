package fsst

import (
	"bytes"
	"container/heap"
	"encoding/binary"
	"io"
	"unsafe"
)

// FSST12 constants
const (
	fsst12CodeBits   = 12
	fsst12CodeBase   = 256                 // Codes 0-255 are literals
	fsst12CodeMax    = 1 << fsst12CodeBits // 4096
	fsst12MaxSymbols = fsst12CodeMax - 256 // 3840 learnable symbols
	fsst12HashSize   = 1 << 13             // 8192 entries

	fsst12Version uint64 = 20250116 // FSST12 format version
)

// Table12 holds a trained FSST12 symbol table.
// FSST12 uses 12-bit codes: 0-255 are byte literals, 256-4095 are learned symbols.
// No escape mechanism is needed since every byte has a valid code.
type Table12 struct {
	// Encoder lookup
	shortCodes [65536]uint16          // 2-byte prefix -> packed [length:4|code:12]
	byteCodes  [256]uint16            // 1-byte -> packed [length:4|code:12]
	hashTab    [fsst12HashSize]symbol // 3-8 byte symbols

	// Symbol storage: indices 0-255 are literals, 256+ are learned
	symbols [fsst12CodeMax]symbol

	// Metadata
	nSymbols uint16    // count of learned symbols (max 3840)
	lenHisto [8]uint16 // histogram of lengths 1-8

	// Decoder tables
	decLen    [fsst12CodeMax]byte   // code -> length (1 for literals)
	decSymbol [fsst12CodeMax]uint64 // code -> symbol value

	// Scratch buffer
	encBuf []byte
}

// Train12 builds a Table12 from training data.
func Train12(inputs [][]byte) *Table12 {
	// Use standard FSST training to find good symbols, then convert
	// We train with more iterations to find more symbols
	sample := makeSample(inputs)
	t := newTable12()
	c := &counters12{}

	// Reuse allocations
	candidates := make(map[[2]uint64]qsym12, 1024)
	heap := make(qsymHeap12, 0, fsst12MaxSymbols+1)
	list := make([]qsym12, 0, fsst12MaxSymbols)

	for frac := 8; ; frac += 30 {
		*c = counters12{}
		compressCount12(t, c, sample, frac)
		buildCandidates12(t, c, frac, candidates, &heap, &list)
		if frac >= 128 {
			break
		}
	}

	t.finalize()
	return t
}

// Train12Strings is a convenience wrapper for string slices.
func Train12Strings(inputs []string) *Table12 {
	bytes := make([][]byte, len(inputs))
	for i := range inputs {
		bytes[i] = unsafe.Slice(unsafe.StringData(inputs[i]), len(inputs[i]))
	}
	return Train12(bytes)
}

func newTable12() *Table12 {
	t := &Table12{}
	// Codes 0-255 are byte literals
	for i := range 256 {
		t.symbols[i] = newSymbolFromByte(byte(i), uint16(i))
		t.byteCodes[i] = packCodeLength12(uint16(i), 1)
	}
	// Mark remaining symbol slots and hash table as unused
	empty := symbol{val: 0, icl: iclFree}
	for i := 256; i < fsst12CodeMax; i++ {
		t.symbols[i] = empty
	}
	for i := range fsst12HashSize {
		t.hashTab[i] = empty
	}
	// shortCodes: default to first byte
	for i := range 65536 {
		t.shortCodes[i] = packCodeLength12(uint16(i&0xFF), 1)
	}
	return t
}

func packCodeLength12(code uint16, length int) uint16 {
	return code | uint16(length<<12)
}

// hashInsert12 adds a 3+ byte symbol to the hash table.
func (t *Table12) hashInsert(sym symbol) bool {
	idx := sym.hash() & (fsst12HashSize - 1)
	if t.hashTab[idx].icl < iclFree {
		return false
	}
	mask := ^uint64(0) >> sym.ignoredBits()
	t.hashTab[idx] = symbol{val: sym.val & mask, icl: sym.icl}
	return true
}

// addSymbol adds a new learned symbol.
func (t *Table12) addSymbol(sym symbol) bool {
	if int(t.nSymbols) >= fsst12MaxSymbols {
		return false
	}
	code := uint32(fsst12CodeBase) + uint32(t.nSymbols)
	length := sym.length()
	sym.setCodeLen(code, length)

	switch length {
	case 1:
		t.byteCodes[sym.first()] = packCodeLength12(uint16(code), 1)
	case 2:
		t.shortCodes[sym.first2()] = packCodeLength12(uint16(code), 2)
	default:
		if !t.hashInsert(sym) {
			return false
		}
	}

	t.symbols[code] = sym
	t.nSymbols++
	t.lenHisto[length-1]++
	return true
}

// clearSymbols removes all learned symbols.
func (t *Table12) clearSymbols() {
	for i := range t.lenHisto {
		t.lenHisto[i] = 0
	}
	// Reset encoder tables to literal-only state
	for i := range 256 {
		t.byteCodes[i] = packCodeLength12(uint16(i), 1)
	}
	for i := range 65536 {
		t.shortCodes[i] = packCodeLength12(uint16(i&0xFF), 1)
	}
	empty := symbol{val: 0, icl: iclFree}
	for i := range fsst12HashSize {
		t.hashTab[i] = empty
	}
	t.nSymbols = 0
}

// findLongestSymbol returns the best matching code.
func (t *Table12) findLongestSymbol(sym symbol) uint16 {
	// Try hash table (3-8 bytes)
	idx := sym.hash() & (fsst12HashSize - 1)
	entry := t.hashTab[idx]
	if entry.icl <= sym.icl {
		mask := ^uint64(0) >> entry.ignoredBits()
		if entry.val == (sym.val & mask) {
			return entry.code()
		}
	}
	// Try 2-byte
	if sym.length() >= 2 {
		code := t.shortCodes[sym.first2()] & 0xFFF
		if code >= fsst12CodeBase {
			return code
		}
	}
	// Fall back to literal
	return uint16(sym.first())
}

// finalize builds encoder and decoder tables.
func (t *Table12) finalize() {
	t.rebuildIndices()
	t.buildDecoderTables()
	t.encBuf = make([]byte, chunkSize+chunkPadding)
}

func (t *Table12) rebuildIndices() {
	// Reset to literals
	for i := range 256 {
		t.byteCodes[i] = packCodeLength12(uint16(i), 1)
	}
	for i := range 65536 {
		t.shortCodes[i] = packCodeLength12(uint16(i&0xFF), 1)
	}
	empty := symbol{val: 0, icl: iclFree}
	for i := range fsst12HashSize {
		t.hashTab[i] = empty
	}

	// Apply learned symbols
	for i := uint16(0); i < t.nSymbols; i++ {
		code := fsst12CodeBase + i
		sym := t.symbols[code]
		length := sym.length()

		switch length {
		case 1:
			t.byteCodes[sym.first()] = packCodeLength12(code, 1)
		case 2:
			t.shortCodes[sym.first2()] = packCodeLength12(code, 2)
		default:
			t.hashInsert(sym)
		}
	}
}

func (t *Table12) buildDecoderTables() {
	// Literals 0-255
	for i := range 256 {
		t.decLen[i] = 1
		t.decSymbol[i] = uint64(i)
	}
	// Learned symbols
	for i := uint16(0); i < t.nSymbols; i++ {
		code := fsst12CodeBase + i
		sym := t.symbols[code]
		t.decLen[code] = byte(sym.length())
		t.decSymbol[code] = sym.val
	}
}

// Encode compresses input using 12-bit packed codes.
// Output format: pairs of 12-bit codes packed into 3 bytes each.
// If odd number of codes, last code is padded with 0.
func (t *Table12) Encode(buf, input []byte) []byte {
	if len(input) == 0 {
		return buf[:0]
	}

	// Worst case: all literals = same size as input (1 code per byte)
	// Each code is 12 bits, so n codes = n * 1.5 bytes
	maxCodes := len(input)
	maxOut := (maxCodes*3 + 1) / 2
	if buf == nil || cap(buf) < maxOut {
		buf = make([]byte, maxOut)
	} else {
		buf = buf[:cap(buf)]
	}

	// First pass: generate codes
	codes := make([]uint16, 0, len(input))
	position := 0
	inputLen := len(input)

	for position < inputLen {
		remaining := inputLen - position
		if remaining >= 8 {
			word := unalignedLoad(input[position:])

			// Try hash table (3-8 bytes)
			idx := hashWord(word&mask24) & (fsst12HashSize - 1)
			entry := t.hashTab[idx]
			if entry.icl < iclFree {
				mask := ^uint64(0) >> entry.ignoredBits()
				symLen := int(entry.length())
				if entry.val == (word&mask) && symLen <= remaining {
					codes = append(codes, entry.code())
					position += symLen
					continue
				}
			}

			// Try 2-byte
			code := t.shortCodes[uint16(word&mask16)]
			codeVal := code & 0xFFF
			codeLen := int(code >> 12)
			if codeLen == 2 && codeVal >= fsst12CodeBase {
				codes = append(codes, codeVal)
				position += 2
				continue
			}

			// 1-byte (literal or learned)
			codes = append(codes, t.byteCodes[uint8(word)]&0xFFF)
			position++
		} else {
			// Tail: byte-by-byte to avoid out-of-bounds
			codes = append(codes, t.byteCodes[input[position]]&0xFFF)
			position++
		}
	}

	// Pack codes into output: 2 codes -> 3 bytes
	outPos := 0
	for i := 0; i+1 < len(codes); i += 2 {
		c0, c1 := codes[i], codes[i+1]
		buf[outPos] = byte(c0)
		buf[outPos+1] = byte(c0>>8) | byte(c1<<4)
		buf[outPos+2] = byte(c1 >> 4)
		outPos += 3
	}
	// Handle odd code
	if len(codes)%2 == 1 {
		c := codes[len(codes)-1]
		buf[outPos] = byte(c)
		buf[outPos+1] = byte(c >> 8)
		outPos += 2
	}

	return buf[:outPos]
}

// EncodeAll compresses input and returns a new slice.
func (t *Table12) EncodeAll(input []byte) []byte {
	return t.Encode(nil, input)
}

// Decode decompresses 12-bit packed codes.
func (t *Table12) Decode(buf, src []byte) []byte {
	if len(src) == 0 {
		return buf[:0]
	}

	if buf == nil {
		buf = make([]byte, 0, len(src)*2)
	} else {
		buf = buf[:0]
	}

	bufPos := 0
	bufCap := cap(buf)
	if bufCap > 0 {
		buf = buf[:bufCap]
	}

	srcPos := 0
	srcLen := len(src)

	for srcPos+2 < srcLen {
		// Unpack 2 codes from 3 bytes
		b0, b1, b2 := src[srcPos], src[srcPos+1], src[srcPos+2]
		c0 := uint16(b0) | (uint16(b1&0x0F) << 8)
		c1 := uint16(b1>>4) | (uint16(b2) << 4)
		srcPos += 3

		// Decode first code
		if c0 < fsst12CodeMax {
			symLen := int(t.decLen[c0])
			symVal := t.decSymbol[c0]
			if bufPos+symLen > bufCap {
				newCap := max(bufCap*2, bufPos+symLen)
				newBuf := make([]byte, newCap)
				copy(newBuf, buf[:bufPos])
				buf = newBuf
				bufCap = newCap
			}
			writeSymbol(buf, bufPos, symVal, symLen)
			bufPos += symLen
		}

		// Decode second code
		if c1 < fsst12CodeMax {
			symLen := int(t.decLen[c1])
			symVal := t.decSymbol[c1]
			if bufPos+symLen > bufCap {
				newCap := max(bufCap*2, bufPos+symLen)
				newBuf := make([]byte, newCap)
				copy(newBuf, buf[:bufPos])
				buf = newBuf
				bufCap = newCap
			}
			writeSymbol(buf, bufPos, symVal, symLen)
			bufPos += symLen
		}
	}

	// Handle remaining 2 bytes (1 code, odd length)
	if srcPos+1 < srcLen {
		b0, b1 := src[srcPos], src[srcPos+1]
		c0 := uint16(b0) | (uint16(b1&0x0F) << 8)
		if c0 < fsst12CodeMax {
			symLen := int(t.decLen[c0])
			symVal := t.decSymbol[c0]
			if bufPos+symLen > bufCap {
				newCap := max(bufCap*2, bufPos+symLen)
				newBuf := make([]byte, newCap)
				copy(newBuf, buf[:bufPos])
				buf = newBuf
				bufCap = newCap
			}
			writeSymbol(buf, bufPos, symVal, symLen)
			bufPos += symLen
		}
	}

	return buf[:bufPos]
}

// DecodeAll decompresses and returns a new slice.
func (t *Table12) DecodeAll(src []byte) []byte {
	return t.Decode(nil, src)
}

// writeSymbol writes a symbol value to buf at pos.
func writeSymbol(buf []byte, pos int, val uint64, length int) {
	switch length {
	case 1:
		buf[pos] = byte(val)
	case 2:
		binary.LittleEndian.PutUint16(buf[pos:], uint16(val))
	case 3:
		binary.LittleEndian.PutUint16(buf[pos:], uint16(val))
		buf[pos+2] = byte(val >> 16)
	case 4:
		binary.LittleEndian.PutUint32(buf[pos:], uint32(val))
	case 5:
		binary.LittleEndian.PutUint32(buf[pos:], uint32(val))
		buf[pos+4] = byte(val >> 32)
	case 6:
		binary.LittleEndian.PutUint32(buf[pos:], uint32(val))
		binary.LittleEndian.PutUint16(buf[pos+4:], uint16(val>>32))
	case 7:
		binary.LittleEndian.PutUint32(buf[pos:], uint32(val))
		binary.LittleEndian.PutUint16(buf[pos+4:], uint16(val>>32))
		buf[pos+6] = byte(val >> 48)
	case 8:
		binary.LittleEndian.PutUint64(buf[pos:], val)
	}
}

// WriteTo serializes the Table12.
func (t *Table12) WriteTo(w io.Writer) (int64, error) {
	var (
		n    int64
		buf8 [8]byte
	)

	// Version header
	binary.LittleEndian.PutUint64(buf8[:], fsst12Version)
	if nn, err := w.Write(buf8[:]); err != nil {
		return n, err
	} else {
		n += int64(nn)
	}

	// nSymbols (2 bytes) + lenHisto (8 bytes)
	binary.LittleEndian.PutUint16(buf8[:2], t.nSymbols)
	if nn, err := w.Write(buf8[:2]); err != nil {
		return n, err
	} else {
		n += int64(nn)
	}

	for i := range 8 {
		buf8[i] = byte(t.lenHisto[i])
	}
	if nn, err := w.Write(buf8[:]); err != nil {
		return n, err
	} else {
		n += int64(nn)
	}

	// Symbol bytes
	for i := uint16(0); i < t.nSymbols; i++ {
		sym := t.symbols[fsst12CodeBase+i]
		length := int(sym.length())
		for j := 0; j < length; j++ {
			buf8[j] = byte(sym.val >> (8 * j))
		}
		if nn, err := w.Write(buf8[:length]); err != nil {
			return n, err
		} else {
			n += int64(nn)
		}
	}

	return n, nil
}

// ReadFrom deserializes a Table12.
func (t *Table12) ReadFrom(r io.Reader) (int64, error) {
	*t = *newTable12()
	var (
		n    int64
		buf8 [8]byte
	)

	// Read version
	if _, err := io.ReadFull(r, buf8[:]); err != nil {
		return n, err
	}
	n += 8
	if binary.LittleEndian.Uint64(buf8[:]) != fsst12Version {
		return n, ErrBadVersion
	}

	// Read nSymbols
	if _, err := io.ReadFull(r, buf8[:2]); err != nil {
		return n, err
	}
	n += 2
	t.nSymbols = binary.LittleEndian.Uint16(buf8[:2])

	// Read lenHisto
	if _, err := io.ReadFull(r, buf8[:]); err != nil {
		return n, err
	}
	n += 8
	for i := range 8 {
		t.lenHisto[i] = uint16(buf8[i])
	}

	// Build length schedule
	lens := make([]uint8, t.nSymbols)
	pos := 0
	for l := 1; l <= 8; l++ {
		for range int(t.lenHisto[l-1]) {
			if pos < len(lens) {
				lens[pos] = uint8(l)
				pos++
			}
		}
	}

	// Read symbols
	for i := uint16(0); i < t.nSymbols; i++ {
		length := int(lens[i])
		if _, err := io.ReadFull(r, buf8[:length]); err != nil {
			return n, err
		}
		n += int64(length)

		var val uint64
		for j := 0; j < length; j++ {
			val |= uint64(buf8[j]) << (8 * j)
		}

		code := fsst12CodeBase + i
		sym := symbol{val: val}
		sym.setCodeLen(uint32(code), uint32(length))
		t.symbols[code] = sym
	}

	t.rebuildIndices()
	t.buildDecoderTables()
	t.encBuf = make([]byte, chunkSize+chunkPadding)
	return n, nil
}

// MarshalBinary implements encoding.BinaryMarshaler.
func (t *Table12) MarshalBinary() ([]byte, error) {
	var buf bytes.Buffer
	if _, err := t.WriteTo(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// UnmarshalBinary implements encoding.BinaryUnmarshaler.
func (t *Table12) UnmarshalBinary(data []byte) error {
	_, err := t.ReadFrom(bytes.NewReader(data))
	return err
}

// Training support for Table12

const fsst12CodeMax2 = 512 // Use same code space as FSST for training

type counters12 struct {
	single   [fsst12CodeMax2]uint16
	pair     [fsst12CodeMax2][fsst12CodeMax2]uint16
	pairList [][2]uint32
}

func (c *counters12) incSingle(code uint32) {
	if code < fsst12CodeMax2 && c.single[code] < 0xFFFF {
		c.single[code]++
	}
}

func (c *counters12) incPair(code1, code2 uint32) {
	if code1 >= fsst12CodeMax2 || code2 >= fsst12CodeMax2 {
		return
	}
	if c.pair[code1][code2] == 0 {
		c.pairList = append(c.pairList, [2]uint32{code1, code2})
	}
	if c.pair[code1][code2] < 0xFFFF {
		c.pair[code1][code2]++
	}
}

func compressCount12(t *Table12, c *counters12, sample [][]byte, frac int) {
	for i := range sample {
		if frac < 128 && int(hashWord(uint64(i))&sampleMask) > frac {
			continue
		}
		data := sample[i]
		end := len(data)
		if end == 0 {
			continue
		}

		pos := 0
		cur := t.findLongestSymbol(newSymbolFromBytes(data[pos:min(pos+8, end)]))
		curLen := int(t.symbols[cur].length())
		if cur >= fsst12CodeBase {
			curLen = int(t.symbols[cur].length())
		} else {
			curLen = 1
		}
		pos += curLen
		start := 0

		for {
			if cur < fsst12CodeMax2 {
				c.incSingle(uint32(cur))
			}
			if pos-start != 1 && start < len(data) {
				c.incSingle(uint32(data[start]))
			}
			if pos >= end {
				break
			}

			start = pos
			next := t.findLongestSymbol(newSymbolFromBytes(data[pos:min(pos+8, end)]))
			nextLen := 1
			if next >= fsst12CodeBase {
				nextLen = int(t.symbols[next].length())
			}
			pos += nextLen

			if frac < 128 && cur < fsst12CodeMax2 && next < fsst12CodeMax2 {
				c.incPair(uint32(cur), uint32(next))
				if pos-start > 1 && start < len(data) {
					c.incPair(uint32(cur), uint32(data[start]))
				}
			}
			cur = next
		}
	}
}

type qsym12 struct {
	symbol symbol
	gain   uint32
}

// qsymHeap12 is a min-heap of qsym12 ordered by ascending gain, breaking ties
// by larger symbol value for determinism. Implements heap.Interface.
type qsymHeap12 []qsym12

func (h qsymHeap12) Len() int { return len(h) }
func (h qsymHeap12) Less(i, j int) bool {
	if h[i].gain != h[j].gain {
		return h[i].gain < h[j].gain
	}
	return h[i].symbol.val > h[j].symbol.val
}
func (h qsymHeap12) Swap(i, j int)   { h[i], h[j] = h[j], h[i] }
func (h *qsymHeap12) Push(x any)     { *h = append(*h, x.(qsym12)) }
func (h *qsymHeap12) Pop() any       { old := *h; x := old[len(old)-1]; *h = old[:len(old)-1]; return x }

func buildCandidates12(t *Table12, c *counters12, frac int, candidates map[[2]uint64]qsym12, h *qsymHeap12, list *[]qsym12) {
	clear(candidates)
	minCount := max((minCountNumerator*frac)/minCountDenominator, 1)

	// Process single symbols
	for code := uint32(0); code < fsst12CodeMax2; code++ {
		count := c.single[code]
		if count == 0 {
			continue
		}

		var sym symbol
		if code < 256 {
			sym = newSymbolFromByte(byte(code), uint16(code))
		} else if code < uint32(fsst12CodeBase)+uint32(t.nSymbols) {
			sym = t.symbols[code]
		} else {
			continue
		}

		weight := uint64(count)
		if sym.length() == 1 {
			weight *= singleByteBoost
		}
		if int(weight) >= minCount {
			key := [2]uint64{sym.val, uint64(sym.length())}
			gain := uint32(weight) * uint32(sym.length())
			if existing, ok := candidates[key]; ok {
				gain += existing.gain
			}
			candidates[key] = qsym12{symbol: sym, gain: gain}
		}
	}

	// Process pairs
	if frac < 128 {
		for _, pair := range c.pairList {
			code1, code2 := pair[0], pair[1]
			count := c.pair[code1][code2]
			if count == 0 || int(count) < minCount {
				continue
			}

			var sym1, sym2 symbol
			if code1 < 256 {
				sym1 = newSymbolFromByte(byte(code1), uint16(code1))
			} else if code1 < uint32(fsst12CodeBase)+uint32(t.nSymbols) {
				sym1 = t.symbols[code1]
			} else {
				continue
			}

			if sym1.length() == 8 {
				continue
			}

			if code2 < 256 {
				sym2 = newSymbolFromByte(byte(code2), uint16(code2))
			} else if code2 < uint32(fsst12CodeBase)+uint32(t.nSymbols) {
				sym2 = t.symbols[code2]
			} else {
				continue
			}

			merged := concatSymbols(sym1, sym2)
			key := [2]uint64{merged.val, uint64(merged.length())}
			gain := uint32(count) * uint32(merged.length())
			if existing, ok := candidates[key]; ok {
				gain += existing.gain
			}
			candidates[key] = qsym12{symbol: merged, gain: gain}
		}
	}

	// Select top candidates using min-heap
	*h = (*h)[:0]
	heap.Init(h)

	for _, candidate := range candidates {
		if len(*h) < fsst12MaxSymbols {
			heap.Push(h, candidate)
		} else if candidate.gain > (*h)[0].gain ||
			(candidate.gain == (*h)[0].gain && candidate.symbol.val < (*h)[0].symbol.val) {
			heap.Pop(h)
			heap.Push(h, candidate)
		}
	}

	// Extract sorted list
	*list = (*list)[:0]
	if cap(*list) < len(*h) {
		*list = make([]qsym12, len(*h))
	} else {
		*list = (*list)[:len(*h)]
	}
	for i := len(*h) - 1; i >= 0; i-- {
		(*list)[i] = heap.Pop(h).(qsym12)
	}
	for i, j := 0, len(*list)-1; i < j; i, j = i+1, j-1 {
		(*list)[i], (*list)[j] = (*list)[j], (*list)[i]
	}

	// Update table
	t.clearSymbols()
	for i := 0; i < len(*list) && int(t.nSymbols) < fsst12MaxSymbols; i++ {
		t.addSymbol((*list)[i].symbol)
	}
}

