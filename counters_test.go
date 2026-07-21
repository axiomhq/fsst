package fsst

import "testing"

func TestCountersBasic(t *testing.T) {
	var c counters

	// Test single-symbol counting
	c.incSingle(5)
	if c.single[5] != 1 {
		t.Fatalf("incSingle first increment: got %d, want 1", c.single[5])
	}
	c.incSingle(5)
	if c.single[5] != 2 {
		t.Fatalf("incSingle second increment: got %d, want 2", c.single[5])
	}

	// Test pair counting
	c.incPair(3, 4)
	if c.pair[3][4] != 1 {
		t.Fatalf("incPair first increment: got %d, want 1", c.pair[3][4])
	}

	// Test nextSingle
	code := uint32(6)
	c.incSingle(10)
	count := c.nextSingle(&code)
	if count == 0 || code != 10 {
		t.Fatalf("nextSingle failed: code=%d count=%d", code, count)
	}

	// Test pairCount
	c.incPair(10, 2)
	pairCnt := c.pairCount(10, 2)
	if pairCnt != 1 {
		t.Fatalf("pairCount failed: got %d, want 1", pairCnt)
	}

	// Test counting to 256 and beyond
	var c2 counters
	for i := 0; i < 300; i++ {
		c2.incSingle(0)
	}
	code = 0
	got := c2.nextSingle(&code)
	if got != 300 {
		t.Fatalf("high count failed: expected 300, got %d", got)
	}
}

func TestCountersReset(t *testing.T) {
	var c counters
	c.incSingle(7)
	c.incPair(3, 4)
	c.incPair(3, 4)
	c.incPair(8, 9)

	c.reset()

	if c.single[7] != 0 {
		t.Fatalf("single count survived reset: %d", c.single[7])
	}
	if c.pair[3][4] != 0 || c.pair[8][9] != 0 {
		t.Fatalf("pair count survived reset: %d, %d", c.pair[3][4], c.pair[8][9])
	}
	if len(c.pairList) != 0 {
		t.Fatalf("pair list length after reset: %d", len(c.pairList))
	}

	c.incPair(3, 4)
	if c.pair[3][4] != 1 || len(c.pairList) != 1 {
		t.Fatalf("counter not reusable after reset: count=%d list=%d", c.pair[3][4], len(c.pairList))
	}
}
