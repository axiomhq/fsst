package fsst

import (
	"sort"
	"unsafe"
)

const (
	fsstSampleTarget = 1 << 14 // 16KB
	fsstSampleMaxSz  = 2 * fsstSampleTarget
	fsstSampleLine   = 512

	singleByteBoost     = 8
	minCountNumerator   = 5
	minCountDenominator = 128
	rngSeed             = 4637947
)

func Train(inputs [][]byte) *Table {
	sample := makeSample(inputs)
	table := newTable()
	counter := &counters{}

	for frac := 8; ; frac += 30 {
		*counter = counters{}
		compressCount(table, counter, sample, frac)
		buildCandidates(table, counter, frac)
		if frac >= 128 {
			break
		}
	}
	table.finalize()
	return table
}

func findNextSymbolFast(t *Table, data []byte, position int) (code uint16, advance int) {
	word := fsstUnalignedLoad(data[position:])
	prefix24 := word & fsstMask24
	hashIndex := fsstHash(prefix24) & (fsstHashTabSize - 1)
	hashSymbol := t.hashTab[hashIndex]
	shortCode := t.shortCodes[uint16(word&fsstMask16)] & fsstCodeMask
	symbolMask := ^uint64(0) >> hashSymbol.ignoredBits()
	maskedWord := word & symbolMask
	if hashSymbol.icl < fsstICLFree && hashSymbol.val == maskedWord {
		return hashSymbol.code(), int(hashSymbol.length())
	}
	if shortCode >= fsstCodeBase {
		return shortCode, 2
	}
	return t.byteCodes[byte(word&fsstMask8)] & fsstCodeMask, 1
}

func compressCount(t *Table, c *counters, sample [][]byte, frac int) {
	for i := range sample {
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
			if pos-start != 1 {
				c.incSingle(uint32(sample[i][start]))
			}
			if pos == end {
				break
			}
			start = pos
			var next uint16
			var adv int
			if pos < end-7 {
				next, adv = findNextSymbolFast(t, sample[i], pos)
				pos += adv
			} else {
				next = t.findLongestSymbol(newSymbolFromBytes(sample[i][pos:min(pos+8, end)]))
				pos += int(t.symbols[next].length())
			}
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

func buildCandidates(t *Table, c *counters, frac int) {
	candidates := make(map[[2]uint64]qsym)
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

		if sym.length() == 8 || frac >= 128 {
			continue
		}
		for code2 := uint32(0); code2 < fsstCodeBase+uint32(t.nSymbols); code2++ {
			count2 := c.nextPair(code, &code2)
			if count2 == 0 || int(count2) < minCount {
				continue
			}
			sym2 := t.symbols[code2]
			merged := fsstConcat(sym, sym2)
			key := [2]uint64{merged.val, uint64(merged.length())}
			gain := uint32(count2) * uint32(merged.length())
			if existing, ok := candidates[key]; ok {
				gain += existing.gain
			}
			candidates[key] = qsym{symbol: merged, gain: gain}
		}
	}

	list := make([]qsym, 0, len(candidates))
	for _, candidate := range candidates {
		list = append(list, candidate)
	}
	sort.Slice(list, func(i, j int) bool {
		if list[i].gain != list[j].gain {
			return list[i].gain > list[j].gain
		}
		return list[i].symbol.val < list[j].symbol.val
	})

	t.clearSymbols()
	for i := 0; i < len(list) && int(t.nSymbols) < fsstMaxSymbols; i++ {
		t.addSymbol(list[i].symbol)
	}
}

func TrainStrings(inputs []string) *Table {
	bytes := make([][]byte, len(inputs))
	for i := range inputs {
		bytes[i] = unsafe.Slice(unsafe.StringData(inputs[i]), len(inputs[i]))
	}
	return Train(bytes)
}

func makeSample(inputs [][]byte) [][]byte {
	var total int
	for i := range inputs {
		total += len(inputs[i])
	}

	if total < fsstSampleTarget {
		return inputs
	}

	buf := make([]byte, fsstSampleMaxSz)
	sample := make([][]byte, 0, len(inputs))
	pos := 0

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
