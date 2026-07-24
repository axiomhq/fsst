package fsst

import (
	"slices"
	"sync"
)

const (
	// sampleTarget is the target sample size for training (~16KB).
	// Large enough to capture recurring patterns but small enough to keep
	// the O(n × 5 iterations) training fast.
	sampleTarget  = 1 << 14 // 16KB
	sampleMaxSize = 2 * sampleTarget

	// sampleLine is the slice size when sampling from inputs.
	// 512 bytes is large enough to capture multi-byte patterns in context
	// but small enough to collect many diverse slices within the 16KB budget.
	sampleLine = 512

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

	// Keep enough ranked candidates to replace symbols rejected by hash-table
	// collisions. Retaining only maxSymbols candidates can leave the table
	// partially filled and materially degrade compression.
	maxCandidateSymbols = maxSymbols * 2
)

// Train builds and finalizes a compression Table from the provided corpora.
// It samples inputs, iteratively parses and counts symbol usage, proposes
// merged symbols, retains top-gain candidates, and finalizes code layout.
func Train(inputs [][]byte) *Table {
	sample := makeSample(inputs)
	table := newTable()
	workspace := trainingWorkspacePool.Get().(*trainingWorkspace)
	defer func() {
		workspace.reset()
		trainingWorkspacePool.Put(workspace)
	}()

	for frac := 8; ; frac += 30 {
		workspace.counter.reset()
		compressCount(table, &workspace.counter, sample, frac)
		buildCandidates(table, &workspace.counter, frac, workspace.candidates, &workspace.scratch)
		if frac >= 128 {
			break
		}
	}
	table.finalize()
	return table
}

type trainingWorkspace struct {
	counter    counters
	candidates map[[2]uint64]qsym
	scratch    [maxCandidateSymbols * 2]qsym
}

func newTrainingWorkspace() *trainingWorkspace {
	return &trainingWorkspace{
		candidates: make(map[[2]uint64]qsym, 512),
	}
}

func (w *trainingWorkspace) reset() {
	w.counter.reset()
	clear(w.candidates)
}

var trainingWorkspacePool = sync.Pool{
	New: func() any { return newTrainingWorkspace() },
}

// findNextSymbolFast returns the best match at data[position:] using the
// current Table: prefer 3–8 byte hash hits, then unique 2-byte short codes,
// otherwise fall back to single-byte. Returns code and matched length.
func findNextSymbolFast(t *Table, data []byte, position int) (code uint16, advance int) {
	var (
		word       = unalignedLoad(data[position:])
		prefix24   = word & mask24
		hashIndex  = hashWord(prefix24) & (hashTabSize - 1)
		hashSymbol = t.hashTab[hashIndex]
		shortCode  = t.shortCodes[uint16(word&mask16)] & codeMask
		symbolMask = ^uint64(0) >> hashSymbol.ignoredBits()
		maskedWord = word & symbolMask
	)

	if hashSymbol.icl < iclFree && hashSymbol.val == maskedWord {
		return hashSymbol.code(), int(hashSymbol.length())
	}
	if shortCode >= codeBase {
		return shortCode, 2
	}
	return t.byteCodes[byte(word&mask8)] & codeMask, 1
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
		if frac < 128 && int(hashWord(uint64(i))&sampleMask) > frac {
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

func (q qsym) betterThan(other qsym) bool {
	if q.gain != other.gain {
		return q.gain > other.gain
	}
	if q.symbol.val != other.symbol.val {
		return q.symbol.val < other.symbol.val
	}
	return q.symbol.length() < other.symbol.length()
}

func compareQsym(a, b qsym) int {
	if a.betterThan(b) {
		return -1
	}
	if b.betterThan(a) {
		return 1
	}
	return 0
}

// partitionTopCandidates moves the strongest maxCandidateSymbols candidates
// into the prefix without fully ordering the remaining candidates.
func partitionTopCandidates(candidates []qsym) {
	target := maxCandidateSymbols - 1
	left, right := 0, len(candidates)-1
	for left < right {
		pivot := candidates[left+(right-left)/2]
		i, j := left, right
		for i <= j {
			for candidates[i].betterThan(pivot) {
				i++
			}
			for pivot.betterThan(candidates[j]) {
				j--
			}
			if i <= j {
				candidates[i], candidates[j] = candidates[j], candidates[i]
				i++
				j--
			}
		}
		switch {
		case target <= j:
			right = j
		case target >= i:
			left = i
		default:
			return
		}
	}
}

// selectCandidates keeps candidates in bounded reusable storage, retaining the
// strongest maxCandidateSymbols values before sorting only that prefix.
func selectCandidates(candidates map[[2]uint64]qsym, scratch *[maxCandidateSymbols * 2]qsym) []qsym {
	n := 0
	for _, candidate := range candidates {
		if n == len(scratch) {
			partitionTopCandidates(scratch[:n])
			n = maxCandidateSymbols
		}
		scratch[n] = candidate
		n++
	}
	if n > maxCandidateSymbols {
		partitionTopCandidates(scratch[:n])
		n = maxCandidateSymbols
	}
	selected := scratch[:n]
	slices.SortFunc(selected, compareQsym)
	return selected
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
//  3. Keep the strongest candidates in bounded reusable scratch storage,
//     periodically partitioning to retain the best prefix before sorting only
//     that prefix in descending rank order.
//
// The map and scratch storage are reused across iterations to reduce GC pressure.
func buildCandidates(t *Table, c *counters, frac int, candidates map[[2]uint64]qsym, scratch *[maxCandidateSymbols * 2]qsym) {
	// Clear candidates map for reuse (clear() is more efficient than delete loop)
	clear(candidates)
	minCount := max((minCountNumerator*frac)/minCountDenominator, 1)

	for code := uint32(0); code < codeBase+uint32(t.nSymbols); code++ {
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
			merged := concatSymbols(sym, sym2)
			key := [2]uint64{merged.val, uint64(merged.length())}
			gain := count2 * uint32(merged.length())
			if existing, ok := candidates[key]; ok {
				gain += existing.gain
			}
			candidates[key] = qsym{symbol: merged, gain: gain}
		}
	}

	selected := selectCandidates(candidates, scratch)

	t.clearSymbols()
	for i := 0; i < len(selected) && int(t.nSymbols) < maxSymbols; i++ {
		t.addSymbol(selected[i].symbol)
	}
}

// TrainStrings converts []string to [][]byte and calls Train.
func TrainStrings(inputs []string) *Table {
	bytes := make([][]byte, len(inputs))
	for i := range inputs {
		bytes[i] = []byte(inputs[i])
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

	if total < sampleTarget {
		return inputs
	}

	var (
		buf    = make([]byte, sampleMaxSize)
		sample = make([][]byte, 0, len(inputs))
		pos    = 0
	)

	rng := hashWord(rngSeed)

	for pos < sampleMaxSize {
		rng = hashWord(rng)
		idx := int(rng % uint64(len(inputs)))

		for len(inputs[idx]) == 0 {
			idx = (idx + 1) % len(inputs)
		}

		numChunks := (len(inputs[idx]) + sampleLine - 1) / sampleLine
		rng = hashWord(rng)
		off := sampleLine * int(rng%uint64(numChunks))

		n := min(len(inputs[idx])-off, sampleLine)
		if pos+n > sampleMaxSize {
			break
		}
		copy(buf[pos:pos+n], inputs[idx][off:off+n])
		sample = append(sample, buf[pos:pos+n:pos+n])
		pos += n

		if pos >= sampleTarget {
			break
		}
	}
	return sample
}
