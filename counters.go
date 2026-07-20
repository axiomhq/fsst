package fsst

// counters tracks symbol and symbol-pair frequencies during training.
//
// Memory layout:
//   - Single-symbol counts: 16-bit counters (0-65535)
//   - Pair counts: 16-bit counters for each (code1, code2) pair
//   - Sparse pair tracking: list of non-zero pairs for fast iteration
//
// Total size: ~520KB, retained in a pooled training workspace between calls.
type counters struct {
	single   [codeMax]uint16          // Single-symbol counts
	pair     [codeMax][codeMax]uint16 // Pair counts
	pairList [][2]uint32              // Sparse list of non-zero pairs
}

func (c *counters) reset() {
	c.single = [codeMax]uint16{}
	c.pair = [codeMax][codeMax]uint16{}
	c.pairList = c.pairList[:0]
}

// incSingle increments the frequency count for a single symbol.
func (c *counters) incSingle(code uint32) {
	if c.single[code] < 0xFFFF {
		c.single[code]++
	}
}

// incPair increments the frequency count for a symbol pair.
func (c *counters) incPair(code1, code2 uint32) {
	if c.pair[code1][code2] == 0 {
		c.pairList = append(c.pairList, [2]uint32{code1, code2})
	}
	if c.pair[code1][code2] < 0xFFFF {
		c.pair[code1][code2]++
	}
}

// nextSingle advances code to the next non-zero count and returns it.
// Returns 0 if no more non-zero counts exist.
func (c *counters) nextSingle(code *uint32) uint32 {
	for *code < codeMax {
		if count := c.single[*code]; count != 0 {
			return uint32(count)
		}
		*code++
	}
	return 0
}

// pairCount returns the count for a specific pair.
func (c *counters) pairCount(code1, code2 uint32) uint32 {
	return uint32(c.pair[code1][code2])
}
