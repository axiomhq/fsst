package fsst

import (
	"container/heap"
	"unsafe"
)

const (
	// fsstSampleTarget is the target sample size for training (~16KB).
	// Large enough to capture recurring patterns but small enough to keep
	// the O(n × 5 iterations) training fast.
	fsstSampleTarget = 1 << 14 // 16KB
	fsstSampleMaxSz  = 2 * fsstSampleTarget

	// fsstSampleLine is the slice size when sampling from inputs.
	// 512 bytes is large enough to capture multi-byte patterns in context
	// but small enough to collect many diverse slices within the 16KB budget.
	fsstSampleLine = 512

	// singleByteBoost inflates single-byte symbol weights so they survive
	// candidate selection despite their low per-occurrence gain (length=1).
	// Without this boost, multi-byte symbols would crowd out useful 1-byte
	// mappings and hurt compression of bytes that appear frequently.
	singleByteBoost = 8

	// minCountNumerator/minCountDenominator define the minimum frequency
	// threshold as a fraction of the current subsampling fraction (frac).
	// Candidates appearing fewer than (5*frac)/128 times are discarded as
	// noise. This adaptive threshold filters more aggressively in early
	// (low-frac) rounds and relaxes as frac increases toward 128.
	minCountNumerator   = 5
	minCountDenominator = 128

	// rngSeed is a fixed seed for the pseudo-random sampling in makeSample.
	// A fixed seed ensures deterministic, reproducible training: the same
	// input always produces the same symbol table.
	rngSeed = 4637947
)

// Train builds and finalizes a compression Table from the provided corpora.
// It samples inputs, iteratively parses and counts symbol usage, proposes
// merged symbols, retains top-gain candidates, and finalizes code layout.
func Train(inputs [][]byte) *Table {
	var (
		sample  = makeSample(inputs)
		table   = newTable()
		counter = &counters{}
		// Reuse allocations across iterations
		candidates = make(map[[2]uint64]qsym, 512)
		heap       = make(qsymHeap, 0, fsstMaxSymbols+1)
		list       = make([]qsym, 0, fsstMaxSymbols)
	)

	for frac := 8; ; frac += 30 {
		*counter = counters{}
		compressCount(table, counter, sample, frac)
		buildCandidates(table, counter, frac, candidates, &heap, &list)
		if frac >= 128 {
			break
		}
	}
	table.finalize()
	return table
}

// findNextSymbolFast returns the best match at data[position:] using the
// current Table: prefer 3–8 byte hash hits, then unique 2-byte short codes,
// otherwise fall back to single-byte. Returns code and matched length.
func findNextSymbolFast(t *Table, data []byte, position int) (code uint16, advance int) {
	var (
		word       = fsstUnalignedLoad(data[position:])
		prefix24   = word & fsstMask24
		hashIndex  = fsstHash(prefix24) & (fsstHashTabSize - 1)
		hashSymbol = t.hashTab[hashIndex]
		shortCode  = t.shortCodes[uint16(word&fsstMask16)] & fsstCodeMask
		symbolMask = ^uint64(0) >> hashSymbol.ignoredBits()
		maskedWord = word & symbolMask
	)

	if hashSymbol.icl < fsstICLFree && hashSymbol.val == maskedWord {
		return hashSymbol.code(), int(hashSymbol.length())
	}
	if shortCode >= fsstCodeBase {
		return shortCode, 2
	}
	return t.byteCodes[byte(word&fsstMask8)] & fsstCodeMask, 1
}

// compressCount simulates encoding the sample with the current symbol table
// and records how often each symbol (and symbol pair) is used.
//
// For each sample slice, it greedily matches the longest symbol at each
// position—mirroring the real encoder's strategy. It counts:
//   - Single symbol usage: how often each code appears.
//   - First-byte fallback: when a multi-byte symbol matches, also count
//     its first byte so single-byte codes aren't starved.
//   - Pair co-occurrence (early rounds only, frac < 128): consecutive
//     symbol pairs, used by buildCandidates to propose merged symbols.
//
// The frac parameter controls subsampling: in early rounds only a fraction
// of sample slices are processed (determined by hashing the slice index),
// making initial iterations cheaper. The final round (frac=128) uses all slices.
func compressCount(t *Table, c *counters, sample [][]byte, frac int) {
	for i := range sample {
		// Subsample: skip slices whose hash exceeds the current fraction.
		if frac < 128 && int(fsstHash(uint64(i))&fsstSampleMask) > frac {
			continue
		}
		end := len(sample[i])
		if end == 0 {
			continue
		}
		pos := 0
		cur := t.findLongestSymbol(newSymbolFromBytes(sample[i][pos:min(pos+8, end)]))
		pos += int(t.symbols[cur].length())
		start := 0
		for {
			c.incSingle(uint32(cur))
			// If the matched symbol spans >1 byte, also count the first byte
			// so single-byte codes maintain representative frequencies.
			if pos-start != 1 {
				c.incSingle(uint32(sample[i][start]))
			}
			if pos == end {
				break
			}
			start = pos
			var (
				next uint16
				adv  int
			)
			// Use the fast path (unaligned 8-byte load) when >=8 bytes remain;
			// fall back to the safe path near the end of the slice.
			if pos < end-7 {
				next, adv = findNextSymbolFast(t, sample[i], pos)
				pos += adv
			} else {
				next = t.findLongestSymbol(newSymbolFromBytes(sample[i][pos:min(pos+8, end)]))
				pos += int(t.symbols[next].length())
			}
			// In early rounds, record consecutive pairs so buildCandidates
			// can propose merged (concatenated) symbols.
			if frac < 128 {
				n := pos - start
				c.incPair(uint32(cur), uint32(next))
				if n > 1 {
					c.incPair(uint32(cur), uint32(sample[i][start]))
				}
			}
			cur = next
		}
	}
}

type qsym struct {
	symbol symbol
	gain   uint32
}

// qsymHeap is a min-heap of qsym based on gain (with tiebreak on symbol.val).
// We use a min-heap to maintain top-K elements efficiently.
type qsymHeap []qsym

// Len implements heap.Interface and returns the number of elements.
func (h qsymHeap) Len() int { return len(h) }

// Less implements heap.Interface ordering by ascending gain, breaking ties
// by larger symbol value to keep selection deterministic.
func (h qsymHeap) Less(i, j int) bool {
	// Min-heap: smaller gain at root (or larger val for tiebreak)
	if h[i].gain != h[j].gain {
		return h[i].gain < h[j].gain
	}
	return h[i].symbol.val > h[j].symbol.val
}

// Swap implements heap.Interface swap.
func (h qsymHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

// Push implements heap.Interface push.
func (h *qsymHeap) Push(x any) { *h = append(*h, x.(qsym)) }

// Pop implements heap.Interface pop.
func (h *qsymHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// buildCandidates selects the best symbol candidates from the frequency
// counters and installs them into the table for the next iteration.
//
// The algorithm has three phases:
//  1. Score existing symbols: gain = frequency × length. Single-byte symbols
//     get their frequency boosted by singleByteBoost to survive selection.
//     Symbols below the adaptive minimum count threshold are discarded.
//  2. Score merged pairs (early rounds only): concatenate each observed
//     (sym1, sym2) pair into a candidate up to 8 bytes, scored the same way.
//     This is how multi-byte symbols grow across iterations.
//  3. Top-K selection via min-heap: keep the fsstMaxSymbols (255) highest-gain
//     candidates. The min-heap root always holds the weakest candidate, so
//     replacements are O(log k). The final list is extracted in descending order.
//
// All map/heap/list arguments are reused across iterations to reduce GC pressure.
func buildCandidates(t *Table, c *counters, frac int, candidates map[[2]uint64]qsym, h *qsymHeap, list *[]qsym) {
	// Clear candidates map for reuse (clear() is more efficient than delete loop)
	clear(candidates)
	minCount := max((minCountNumerator*frac)/minCountDenominator, 1)

	for code := uint32(0); code < fsstCodeBase+uint32(t.nSymbols); code++ {
		count := c.nextSingle(&code)
		if count == 0 {
			continue
		}
		sym := t.symbols[code]
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
			candidates[key] = qsym{symbol: sym, gain: gain}
		}

	}

	// Process pairs using sparse list (much faster than nested iteration)
	if frac < 128 {
		for _, pair := range c.pairList {
			code := pair[0]
			code2 := pair[1]
			count2 := c.pairCount(code, code2)

			if count2 == 0 || int(count2) < minCount {
				continue
			}

			sym := t.symbols[code]
			if sym.length() == 8 {
				continue
			}

			sym2 := t.symbols[code2]
			merged := fsstConcat(sym, sym2)
			key := [2]uint64{merged.val, uint64(merged.length())}
			gain := count2 * uint32(merged.length())
			if existing, ok := candidates[key]; ok {
				gain += existing.gain
			}
			candidates[key] = qsym{symbol: merged, gain: gain}
		}
	}

	// Use min-heap to efficiently select top fsstMaxSymbols candidates
	// This is O(n log k) instead of O(n log n) where k=255, n=candidates
	*h = (*h)[:0] // Reuse heap, clear contents
	heap.Init(h)

	for _, candidate := range candidates {
		if len(*h) < fsstMaxSymbols {
			heap.Push(h, candidate)
		} else if candidate.gain > (*h)[0].gain ||
			(candidate.gain == (*h)[0].gain && candidate.symbol.val < (*h)[0].symbol.val) {
			// Replace minimum with this better candidate
			heap.Pop(h)
			heap.Push(h, candidate)
		}
	}

	// Extract and sort the top-K (small enough to sort efficiently)
	*list = (*list)[:0] // Reuse list, clear contents
	if cap(*list) < len(*h) {
		*list = make([]qsym, len(*h))
	} else {
		*list = (*list)[:len(*h)]
	}
	for i := len(*h) - 1; i >= 0; i-- {
		(*list)[i] = heap.Pop(h).(qsym)
	}

	// Reverse to get descending order (heap gave us ascending)
	for i, j := 0, len(*list)-1; i < j; i, j = i+1, j-1 {
		(*list)[i], (*list)[j] = (*list)[j], (*list)[i]
	}

	t.clearSymbols()
	for i := 0; i < len(*list) && int(t.nSymbols) < fsstMaxSymbols; i++ {
		t.addSymbol((*list)[i].symbol)
	}
}

// TrainStrings converts []string to [][]byte and calls Train.
func TrainStrings(inputs []string) *Table {
	bytes := make([][]byte, len(inputs))
	for i := range inputs {
		bytes[i] = unsafe.Slice(unsafe.StringData(inputs[i]), len(inputs[i]))
	}
	return Train(bytes)
}

// makeSample assembles a ~16KB deterministic pseudo-random sample composed of
// 512-byte slices from the inputs to keep training fast yet representative.
func makeSample(inputs [][]byte) [][]byte {
	var total int
	for i := range inputs {
		total += len(inputs[i])
	}

	if total < fsstSampleTarget {
		return inputs
	}

	var (
		buf    = make([]byte, fsstSampleMaxSz)
		sample = make([][]byte, 0, len(inputs))
		pos    = 0
	)

	rng := fsstHash(rngSeed)

	for pos < fsstSampleMaxSz {
		rng = fsstHash(rng)
		idx := int(rng % uint64(len(inputs)))

		for len(inputs[idx]) == 0 {
			idx = (idx + 1) % len(inputs)
		}

		numChunks := (len(inputs[idx]) + fsstSampleLine - 1) / fsstSampleLine
		rng = fsstHash(rng)
		off := fsstSampleLine * int(rng%uint64(numChunks))

		n := min(len(inputs[idx])-off, fsstSampleLine)
		if pos+n > fsstSampleMaxSz {
			break
		}
		copy(buf[pos:pos+n], inputs[idx][off:off+n])
		sample = append(sample, buf[pos:pos+n:pos+n])
		pos += n

		if pos >= fsstSampleTarget {
			break
		}
	}
	return sample
}
