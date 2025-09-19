package fsst

import "testing"

func TestCountersBasic(t *testing.T) {
	var c counters

	// Test single-symbol counting
	c.incSingle(5)
	if c.singleLow[5] != 1 || c.singleHigh[5] != 1 {
		t.Fatalf("incSingle first increment failed")
	}
	c.incSingle(5)
	if c.singleLow[5] != 2 || c.singleHigh[5] != 1 {
		t.Fatalf("incSingle second increment failed")
	}

	// Test pair counting
	c.incPair(3, 4)
	if c.pairLow[3][4] == 0 {
		t.Fatalf("incPair low byte not set")
	}
	// Check nibble-packed high byte (4 is even, so low nibble of byte 2)
	if c.pairHigh[3][2] == 0 {
		t.Fatalf("incPair high nibble not set")
	}

	// Test nextSingle
	symbolCode := uint32(6)
	c.incSingle(10)
	count := c.nextSingle(&symbolCode)
	if count == 0 || symbolCode != 10 {
		t.Fatalf("nextSingle failed: symbolCode=%d count=%d", symbolCode, count)
	}

	// Test nextPair
	pairCode := uint32(0)
	c.incPair(10, 2)
	pairCount := c.nextPair(10, &pairCode)
	if pairCount == 0 || pairCode != 2 {
		t.Fatalf("nextPair failed: pairCode=%d count=%d", pairCode, pairCount)
	}

	// Test early increment compensation
	var c2 counters
	// Increment 256 times: high should be 1 after compensation
	for i := 0; i < 256; i++ {
		c2.incSingle(0)
	}
	code := uint32(0)
	got := c2.nextSingle(&code)
	if got != 256 {
		t.Fatalf("early increment compensation failed: expected 256, got %d", got)
	}
}
