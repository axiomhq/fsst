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
	got := tbl2.DecodeAll(comp)
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
	got := tbl.DecodeAll(comp)
	if !bytes.Equal(got, inputs[0]) {
		t.Fatalf("roundtrip failed with many symbols")
	}
}

// TestDecodeAPIs tests all decode variants
func TestDecodeAPIs(t *testing.T) {
	input := []byte("Hello, World! This is a test message for FSST compression.")
	tbl := Train([][]byte{input})
	comp := tbl.Encode(input)

	// Test DecodeAll
	t.Run("DecodeAll", func(t *testing.T) {
		got := tbl.DecodeAll(comp)
		if !bytes.Equal(got, input) {
			t.Fatalf("DecodeAll mismatch: got %q, want %q", got, input)
		}
	})

	// Test Decode with sufficient buffer
	t.Run("Decode_sufficient", func(t *testing.T) {
		buf := make([]byte, len(input)*2) // Generous buffer
		got := tbl.Decode(buf, comp)
		if !bytes.Equal(got, input) {
			t.Fatalf("Decode mismatch: got %q, want %q", got, input)
		}
	})

	// Test Decode with small buffer (should grow)
	t.Run("Decode_small", func(t *testing.T) {
		buf := make([]byte, 5) // Too small
		got := tbl.Decode(buf, comp)
		if !bytes.Equal(got, input) {
			t.Fatalf("Decode mismatch: got %q, want %q", got, input)
		}
	})

	// Test Decode with nil buffer (should allocate)
	t.Run("Decode_nil", func(t *testing.T) {
		got := tbl.Decode(nil, comp)
		if !bytes.Equal(got, input) {
			t.Fatalf("Decode mismatch: got %q, want %q", got, input)
		}
	})

	// Test DecodeString
	t.Run("DecodeString", func(t *testing.T) {
		compStr := string(comp)
		got := tbl.DecodeString(compStr)
		if !bytes.Equal(got, input) {
			t.Fatalf("DecodeString mismatch: got %q, want %q", got, input)
		}
	})

}

// BenchmarkDecode benchmarks different decode scenarios
func BenchmarkDecode(b *testing.B) {
	inputs := []struct {
		name string
		data []byte
	}{
		{"small_100B", bytes.Repeat([]byte("hello world "), 8)},
		{"medium_1KB", bytes.Repeat([]byte("The quick brown fox jumps over the lazy dog. "), 22)},
		{"large_10KB", bytes.Repeat([]byte("FSST compression algorithm for structured text data. "), 192)},
		{"json_like", bytes.Repeat([]byte(`{"name":"John","age":30,"city":"New York","active":true}`), 10)},
		{"repetitive", bytes.Repeat([]byte("aaaaaaaaaa"), 100)},
	}

	for _, input := range inputs {
		tbl := Train([][]byte{input.data})
		comp := tbl.Encode(input.data)

		b.Run(input.name+"/DecodeAll", func(b *testing.B) {
			b.SetBytes(int64(len(input.data)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = tbl.DecodeAll(comp)
			}
		})

		b.Run(input.name+"/Decode_with_buf", func(b *testing.B) {
			buf := make([]byte, len(input.data)*2)
			b.SetBytes(int64(len(input.data)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = tbl.Decode(buf, comp)
			}
		})

		b.Run(input.name+"/Decode_nil", func(b *testing.B) {
			b.SetBytes(int64(len(input.data)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = tbl.Decode(nil, comp)
			}
		})

		b.Run(input.name+"/DecodeString", func(b *testing.B) {
			compStr := string(comp)
			b.SetBytes(int64(len(input.data)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = tbl.DecodeString(compStr)
			}
		})
	}
}
