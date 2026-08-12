package fsst

import (
	"bytes"
	"slices"
	"strconv"
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
	if (sc&codeMask) >= codeBase && sc>>lenBits != 1 {
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
	comp := tbl2.EncodeAll(input)
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
	comp := tbl.EncodeAll(inputs[0])
	got := tbl.DecodeAll(comp)
	if !bytes.Equal(got, inputs[0]) {
		t.Fatalf("roundtrip failed with many symbols")
	}
}

// TestDecodeAPIs tests all decode variants
func TestDecodeAPIs(t *testing.T) {
	input := []byte("Hello, World! This is a test message for FSST compression.")
	tbl := Train([][]byte{input})
	comp := tbl.EncodeAll(input)

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

// TestReadFromMalformed verifies that ReadFrom rejects crafted inputs
// without panicking.
func TestReadFromMalformed(t *testing.T) {
	// Helper: build a valid serialized table to use as a base.
	validTable := func() []byte {
		tbl := Train([][]byte{[]byte("hello world hello")})
		var buf bytes.Buffer
		tbl.WriteTo(&buf)
		return buf.Bytes()
	}

	t.Run("bad_version", func(t *testing.T) {
		data := validTable()
		// Corrupt version (first 4 bytes of the 8-byte header)
		data[4] = 0xFF
		var tbl Table
		_, err := tbl.ReadFrom(bytes.NewReader(data))
		if err != ErrBadVersion {
			t.Fatalf("expected ErrBadVersion, got %v", err)
		}
	})

	t.Run("lenHisto_sum_exceeds_nSymbols", func(t *testing.T) {
		data := validTable()
		// lenHisto starts at byte 8, 8 bytes long.
		// Set all histo entries to 255 so sum far exceeds nSymbols.
		for i := 8; i < 16; i++ {
			data[i] = 255
		}
		var tbl Table
		_, err := tbl.ReadFrom(bytes.NewReader(data))
		if err != ErrCorrupted {
			t.Fatalf("expected ErrCorrupted, got %v", err)
		}
	})

	t.Run("lenHisto_sum_less_than_nSymbols", func(t *testing.T) {
		data := validTable()
		// Zero out lenHisto so sum=0 but nSymbols>0.
		for i := 8; i < 16; i++ {
			data[i] = 0
		}
		var tbl Table
		_, err := tbl.ReadFrom(bytes.NewReader(data))
		if err != ErrCorrupted {
			t.Fatalf("expected ErrCorrupted, got %v", err)
		}
	})

	t.Run("truncated_input", func(t *testing.T) {
		data := validTable()
		// Truncate to just the header
		var tbl Table
		_, err := tbl.ReadFrom(bytes.NewReader(data[:8]))
		if err == nil {
			t.Fatalf("expected error on truncated input")
		}
	})
}

func TestEncodeBatch(t *testing.T) {
	inputs := [][]byte{
		[]byte("Hello, World!"),
		[]byte("FSST compression is fast"),
		nil,
		[]byte("x"),
		[]byte("The quick brown fox jumps over the lazy dog"),
	}
	tbl := Train(inputs)

	var want []byte
	wantOffsets := []int{0}
	for _, input := range inputs {
		want = append(want, tbl.EncodeAll(input)...)
		wantOffsets = append(wantOffsets, len(want))
	}

	got, gotOffsets := tbl.EncodeBatch(nil, nil, inputs)
	if !bytes.Equal(got, want) {
		t.Fatalf("encoded bytes differ: got %v, want %v", got, want)
	}
	if !slices.Equal(gotOffsets, wantOffsets) {
		t.Fatalf("offsets differ: got %v, want %v", gotOffsets, wantOffsets)
	}

	t.Run("empty", func(t *testing.T) {
		got, offsets := tbl.EncodeBatch(nil, nil, nil)
		if len(got) != 0 || !slices.Equal(offsets, []int{0}) {
			t.Fatalf("got %v, %v", got, offsets)
		}
	})

	t.Run("reuses_buffers", func(t *testing.T) {
		dst := make([]byte, 0, 2*len(bytes.Join(inputs, nil))+outputPadding)
		offsets := make([]int, 0, len(inputs)+1)
		allocs := testing.AllocsPerRun(100, func() {
			dst, offsets = tbl.EncodeBatch(dst, offsets, inputs)
		})
		if allocs != 0 {
			t.Fatalf("EncodeBatch allocated %v times, want 0", allocs)
		}
	})
}

// TestDecodeBatch tests the batch decode API
func TestDecodeBatch(t *testing.T) {
	strings := [][]byte{
		[]byte("Hello, World!"),
		[]byte("FSST compression is fast"),
		[]byte(`{"name":"Alice","age":30}`),
		[]byte(""),
		[]byte("x"),
		[]byte("The quick brown fox jumps over the lazy dog"),
	}

	tbl := Train(strings)

	// Encode all strings and build concatenated src + offsets
	var src []byte
	offsets := []int{0}
	for _, s := range strings {
		comp := tbl.EncodeAll(s)
		src = append(src, comp...)
		offsets = append(offsets, len(src))
	}

	t.Run("basic_roundtrip", func(t *testing.T) {
		dst, dstOffsets, err := tbl.DecodeBatch(nil, nil, src, offsets)
		if err != nil {
			t.Fatal(err)
		}
		if len(dstOffsets) != len(offsets) {
			t.Fatalf("expected %d offsets, got %d", len(offsets), len(dstOffsets))
		}
		for i, s := range strings {
			got := dst[dstOffsets[i]:dstOffsets[i+1]]
			if !bytes.Equal(got, s) {
				t.Fatalf("string %d: got %q, want %q", i, got, s)
			}
		}
	})

	t.Run("with_buffer", func(t *testing.T) {
		buf := make([]byte, 1024)
		offsetsBuf := make([]int, len(offsets))
		dst, dstOffsets, err := tbl.DecodeBatch(buf, offsetsBuf, src, offsets)
		if err != nil {
			t.Fatal(err)
		}
		for i, s := range strings {
			got := dst[dstOffsets[i]:dstOffsets[i+1]]
			if !bytes.Equal(got, s) {
				t.Fatalf("string %d: got %q, want %q", i, got, s)
			}
		}
	})

	t.Run("small_buffer_grows", func(t *testing.T) {
		buf := make([]byte, 4)
		dst, dstOffsets, err := tbl.DecodeBatch(buf, nil, src, offsets)
		if err != nil {
			t.Fatal(err)
		}
		for i, s := range strings {
			got := dst[dstOffsets[i]:dstOffsets[i+1]]
			if !bytes.Equal(got, s) {
				t.Fatalf("string %d: got %q, want %q", i, got, s)
			}
		}
	})

	t.Run("empty_batch", func(t *testing.T) {
		dst, dstOffsets, err := tbl.DecodeBatch(nil, nil, nil, []int{0})
		if err != nil {
			t.Fatal(err)
		}
		if len(dst) != 0 {
			t.Fatalf("expected empty dst, got %d bytes", len(dst))
		}
		if len(dstOffsets) != 1 {
			t.Fatalf("expected 1 offset, got %d", len(dstOffsets))
		}
	})

	t.Run("single_string", func(t *testing.T) {
		comp := tbl.EncodeAll(strings[0])
		dst, dstOffsets, err := tbl.DecodeBatch(nil, nil, comp, []int{0, len(comp)})
		if err != nil {
			t.Fatal(err)
		}
		got := dst[dstOffsets[0]:dstOffsets[1]]
		if !bytes.Equal(got, strings[0]) {
			t.Fatalf("got %q, want %q", got, strings[0])
		}
	})

	t.Run("invalid_offsets", func(t *testing.T) {
		for _, invalid := range [][]int{nil, {1}, {0, len(src) + 1}, {0, 2, 1}, {0, len(src) - 1}} {
			if _, _, err := tbl.DecodeBatch(nil, nil, src, invalid); err != ErrCorrupted {
				t.Fatalf("offsets %v: got %v, want ErrCorrupted", invalid, err)
			}
		}
	})

	t.Run("escape_does_not_cross_string_boundary", func(t *testing.T) {
		if _, _, err := tbl.DecodeBatch(nil, nil, []byte{escapeCode, 'x'}, []int{0, 1, 2}); err != ErrCorrupted {
			t.Fatalf("got %v, want ErrCorrupted", err)
		}
	})

	t.Run("reuses_buffers", func(t *testing.T) {
		dst := make([]byte, 0, 1024)
		dstOffsets := make([]int, 0, len(offsets))
		allocs := testing.AllocsPerRun(100, func() {
			var err error
			dst, dstOffsets, err = tbl.DecodeBatch(dst, dstOffsets, src, offsets)
			if err != nil {
				panic(err)
			}
		})
		if allocs != 0 {
			t.Fatalf("DecodeBatch allocated %v times, want 0", allocs)
		}
	})

	t.Run("reuses_source_offsets", func(t *testing.T) {
		reusedOffsets := append([]int(nil), offsets...)
		dst, dstOffsets, err := tbl.DecodeBatch(nil, reusedOffsets, src, reusedOffsets)
		if err != nil {
			t.Fatal(err)
		}
		for i, s := range strings {
			got := dst[dstOffsets[i]:dstOffsets[i+1]]
			if !bytes.Equal(got, s) {
				t.Fatalf("string %d: got %q, want %q", i, got, s)
			}
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
		comp := tbl.EncodeAll(input.data)

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

// BenchmarkDecodeBatch compares per-string DecodeAll vs batch decode.
func BenchmarkDecodeBatch(b *testing.B) {
	// Generate 1000 strings of varying sizes
	baseStrings := []string{
		`{"name":"Alice","age":30,"city":"New York"}`,
		`{"name":"Bob","age":25,"city":"San Francisco"}`,
		`{"id":123,"type":"event","ts":"2024-01-15T10:30:00Z"}`,
		`SELECT * FROM users WHERE active = true`,
		`https://example.com/api/v2/users?page=1&limit=100`,
	}

	const numStrings = 1000
	inputs := make([][]byte, numStrings)
	for i := range numStrings {
		inputs[i] = []byte(baseStrings[i%len(baseStrings)])
	}

	tbl := Train(inputs)

	// Encode all and build batch structures
	compressed := make([][]byte, numStrings)
	var batchSrc []byte
	batchOffsets := make([]int, numStrings+1)
	var totalDecompressed int64
	for i, input := range inputs {
		compressed[i] = tbl.EncodeAll(input)
		batchOffsets[i] = len(batchSrc)
		batchSrc = append(batchSrc, compressed[i]...)
		totalDecompressed += int64(len(input))
	}
	batchOffsets[numStrings] = len(batchSrc)

	b.Run("per_string_DecodeAll", func(b *testing.B) {
		b.SetBytes(totalDecompressed)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for _, c := range compressed {
				_ = tbl.DecodeAll(c)
			}
		}
	})

	b.Run("per_string_Decode_reuse", func(b *testing.B) {
		buf := make([]byte, 4096)
		b.SetBytes(totalDecompressed)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for _, c := range compressed {
				_ = tbl.Decode(buf, c)
			}
		}
	})

	b.Run("DecodeBatch", func(b *testing.B) {
		dst := make([]byte, 0, totalDecompressed)
		dstOffsets := make([]int, 0, len(batchOffsets))
		var err error
		b.SetBytes(totalDecompressed)
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			dst, dstOffsets, err = tbl.DecodeBatch(dst, dstOffsets, batchSrc, batchOffsets)
		}
		b.StopTimer()
		if err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkEncodeBatch(b *testing.B) {
	const n = 100_000
	inputs := make([][]byte, n)
	totalInput := 0
	paths := []string{"/api/v1/users/", "/api/v1/orders/", "/api/v1/products/", "/api/v2/users/", "/api/v2/orders/"}
	for i := range inputs {
		inputs[i] = []byte("https://example.com" + paths[i%len(paths)] + strconv.Itoa(i))
		totalInput += len(inputs[i])
	}
	tbl := Train(inputs)

	b.Run("per_string", func(b *testing.B) {
		dst := make([]byte, 0, totalInput/2+1)
		offsets := make([]int, n+1)
		var scratch []byte
		b.SetBytes(int64(totalInput))
		b.ReportAllocs()
		for b.Loop() {
			dst = dst[:0]
			for i, input := range inputs {
				offsets[i] = len(dst)
				encoded := tbl.EncodeInto(scratch[:0], input)
				dst = append(dst, encoded...)
				scratch = encoded[:0]
			}
			offsets[n] = len(dst)
		}
	})

	b.Run("batch", func(b *testing.B) {
		dst := make([]byte, 0, totalInput/2+1)
		offsets := make([]int, 0, n+1)
		b.SetBytes(int64(totalInput))
		b.ReportAllocs()
		for b.Loop() {
			dst, offsets = tbl.EncodeBatch(dst, offsets, inputs)
		}
	})
}
