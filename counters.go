package fsst

// counters tracks symbol and symbol-pair frequencies during training.
//
// Memory layout optimizes for space (512 codes × 5 training rounds = lots of counting):
//   - Single-symbol counts: split into high/low bytes (16-bit range: 0-65535)
//   - Pair counts: nibble-packed high byte (4-bit) + full low byte (12-bit range: 0-4095)
//   - Sparse pair tracking: list of [code1, code2] pairs with non-zero counts for fast iteration
//
// The "early increment" trick: we increment the high byte when low byte wraps from 255→0,
// but this happens one cycle early (at 0→1 transition). The getNext methods compensate
// by decrementing high when both high and low are non-zero.
//
// Total size: ~512KB (acceptable for training phase, discarded after).
type counters struct {
	singleHigh [fsstCodeMax]uint8                  // High byte of single-symbol counts
	singleLow  [fsstCodeMax]uint8                  // Low byte of single-symbol counts
	pairHigh   [fsstCodeMax][fsstCodeMax / 2]uint8 // High nibble of pair counts (packed)
	pairLow    [fsstCodeMax][fsstCodeMax]uint8     // Low byte of pair counts
	pairList   [][2]uint32                         // Sparse list of non-zero pairs [code1, code2]
}

// incSingle increments the frequency count for a single symbol.
func (c *counters) incSingle(symbolCode uint32) {
	if c.singleLow[symbolCode] == 0 {
		// Early increment trick: increment high byte at 0→1 transition
		// instead of at 255→0 wraparound. Saves a comparison in hot path.
		c.singleLow[symbolCode] = 1
		c.singleHigh[symbolCode]++
	} else {
		c.singleLow[symbolCode]++
	}
}

// incPair increments the frequency count for a symbol pair.
func (c *counters) incPair(code1, code2 uint32) {
	if c.pairLow[code1][code2] == 0 {
		// Early increment trick for pair counts (nibble-packed)
		byteIndex := code2 >> 1         // Which byte in the packed array
		nibbleShift := (code2 & 1) << 2 // 0 or 4 (low/high nibble)
		c.pairHigh[code1][byteIndex] += 1 << nibbleShift
		// Track this pair for fast iteration
		c.pairList = append(c.pairList, [2]uint32{code1, code2})
	}
	c.pairLow[code1][code2]++
}

// nextSingle advances symbolCode to the next non-zero count and returns it.
// Returns 0 if no more non-zero counts exist.
func (c *counters) nextSingle(symbolCode *uint32) uint32 {
	code := *symbolCode
	for code < fsstCodeMax {
		high := c.singleHigh[code]
		low := c.singleLow[code]
		if high != 0 || low != 0 {
			*symbolCode = code
			highVal := uint32(high)
			lowVal := uint32(low)
			// Compensate for early increment: if both bytes non-zero, decrement high
			if lowVal != 0 && highVal > 0 {
				highVal--
			}
			return (highVal << 8) + lowVal
		}
		code++
	}
	*symbolCode = code
	return 0
}

// nextPair advances code2 to the next non-zero pair count for (code1, code2).
// Returns 0 if no more non-zero counts exist.
func (c *counters) nextPair(code1 uint32, code2 *uint32) uint32 {
	current := *code2
	for current < fsstCodeMax {
		byteIndex := current >> 1
		nibbleShift := (current & 1) << 2
		highNibble := (c.pairHigh[code1][byteIndex] >> nibbleShift) & 0xF
		low := c.pairLow[code1][current]
		if highNibble != 0 || low != 0 {
			*code2 = current
			highVal := uint32(highNibble)
			lowVal := uint32(low)
			// Compensate for early increment
			if lowVal != 0 && highVal > 0 {
				highVal--
			}
			return (highVal << 8) + lowVal
		}
		current++
	}
	*code2 = current
	return 0
}
