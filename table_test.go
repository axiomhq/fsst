package fsst

import (
	"bytes"
	"strings"
	"testing"
)

func TestTableAddFind(t *testing.T) {
	tbl := newTable()
	s1 := newSymbolFromBytes([]byte{'x'})
	if !tbl.addSymbol(s1) {
		t.Fatalf("add single-byte")
	}
	s2 := newSymbolFromBytes([]byte{'a', 'b'})
	if !tbl.addSymbol(s2) {
		t.Fatalf("add two-byte")
	}
	s3 := newSymbolFromBytes([]byte{'a', 'b', 'c'})
	if !tbl.addSymbol(s3) {
		t.Fatalf("add long")
	}

	// find longest for prefix "abc..."
	code := tbl.findLongestSymbol(newSymbolFromBytes([]byte{'a', 'b', 'c', 'd'}))
	got := tbl.symbols[code]
	if got.length() < 2 {
		t.Fatalf("expected len>=2 got %d", got.length())
	}
}

func TestFinalize(t *testing.T) {
	tbl := newTable()
	tbl.addSymbol(newSymbolFromBytes([]byte{'a'}))
	tbl.addSymbol(newSymbolFromBytes([]byte{'b', 'c'}))
	tbl.addSymbol(newSymbolFromBytes([]byte{'d', 'e', 'f'}))
	tbl.finalize()
	if tbl.nSymbols == 0 {
		t.Fatalf("no symbols after finalize")
	}
	// shortCodes for unknown 2-byte pattern must map to byteCodes of first byte
	sc := tbl.shortCodes[int('Z')<<8|int('Q')]
	if (sc&fsstCodeMask) >= fsstCodeBase && sc>>fsstLenBits != 1 {
		t.Fatalf("shortCodes not patched for single byte fallback")
	}
}

func TestRebuildTableRoundtrip(t *testing.T) {
	input := []byte("When in the Course of human events, it becomes necessary for one people to dissolve")
	tbl := Train([][]byte{input})
	var buf bytes.Buffer
	if _, err := tbl.WriteTo(&buf); err != nil {
		t.Fatalf("write: %v", err)
	}
	var tbl2 Table
	if _, err := tbl2.ReadFrom(&buf); err != nil {
		t.Fatalf("read: %v", err)
	}
	comp := tbl2.Encode(input)
	got := tbl2.Decode(comp)
	if !bytes.Equal(got, input) {
		t.Fatalf("rebuild roundtrip mismatch")
	}
}

// TestTableLimits tests table behavior at limits
func TestTableLimits(t *testing.T) {
	// Test with many unique patterns to approach symbol limit
	var inputs [][]byte
	for i := 0; i < 300; i++ {
		inputs = append(inputs, []byte(strings.Repeat(string(rune('a'+i%26)), i%8+1)))
	}

	tbl := Train(inputs)
	// Verify it still works
	comp := tbl.Encode(inputs[0])
	got := tbl.Decode(comp)
	if !bytes.Equal(got, inputs[0]) {
		t.Fatalf("roundtrip failed with many symbols")
	}
}
