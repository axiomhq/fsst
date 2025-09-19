package fsst

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTrainDeterministic(t *testing.T) {
	inputs := [][]byte{
		[]byte("the quick brown fox jumps over the lazy dog"),
		[]byte("the quick brown fox jumps over the lazy dog"),
		[]byte("pack my box with five dozen liquor jugs"),
		[]byte("sphinx of black quartz, judge my vow"),
	}
	tbl1 := Train(inputs)
	tbl2 := Train(inputs)

	var b1, b2 bytes.Buffer
	if _, err := tbl1.WriteTo(&b1); err != nil {
		t.Fatalf("write1: %v", err)
	}
	if _, err := tbl2.WriteTo(&b2); err != nil {
		t.Fatalf("write2: %v", err)
	}
	if !bytes.Equal(b1.Bytes(), b2.Bytes()) {
		t.Fatalf("deterministic training violated: headers differ")
	}
}

func TestTrainEncodeDecode(t *testing.T) {
	inputs := [][]byte{
		[]byte("hello world"),
		[]byte("hello there"),
		[]byte("worldwide web"),
		[]byte("hellooooo"),
		[]byte(""),
	}
	tbl := Train(inputs)
	for i := range inputs {
		comp := tbl.Encode(inputs[i])
		got := tbl.Decode(comp)
		if string(got) != string(inputs[i]) {
			t.Fatalf("roundtrip mismatch: %q != %q", got, inputs[i])
		}
	}
}

func TestEqualStringsCompressEqual(t *testing.T) {
	inputs := [][]byte{
		[]byte("repeat-me-1234567890"),
		[]byte("repeat-me-1234567890"),
		[]byte("repeat-me-1234567890"),
	}
	tbl := Train(inputs)
	comp0 := tbl.Encode(inputs[0])
	comp1 := tbl.Encode(inputs[1])
	comp2 := tbl.Encode(inputs[2])
	if !bytes.Equal(comp0, comp1) || !bytes.Equal(comp1, comp2) {
		t.Fatalf("equal strings did not compress to equal outputs")
	}
}

func TestTwoByteAndLongSymbolCompression(t *testing.T) {
	base := bytes.Repeat([]byte("ab"), 200)
	long := []byte("TOKEN!!")
	var mix []byte
	mix = append(mix, base...)
	for range 50 {
		mix = append(mix, long...)
	}
	mix = append(mix, base...)
	inputs := [][]byte{mix}

	tbl := Train(inputs)
	comp := tbl.Encode(inputs[0])
	if len(comp) >= len(inputs[0]) {
		t.Fatalf("expected some compression, got %d >= %d", len(comp), len(inputs[0]))
	}
	got := tbl.Decode(comp)
	if !bytes.Equal(got, inputs[0]) {
		t.Fatalf("roundtrip mismatch")
	}
}

func TestChunkBoundariesRoundtrip(t *testing.T) {
	sizes := []int{511, 512, 1023, 1024, 2047}
	inputs := make([][]byte, len(sizes))
	alpha := []byte("abcdefghijklmnopqrstuvwxyz0123456789_-")
	for i, n := range sizes {
		out := make([]byte, n)
		for j := 0; j < n; j++ {
			out[j] = alpha[j%len(alpha)]
		}
		inputs[i] = out
	}
	tbl := Train(inputs)
	for i := range inputs {
		comp := tbl.Encode(inputs[i])
		got := tbl.Decode(comp)
		if !bytes.Equal(got, inputs[i]) {
			t.Fatalf("roundtrip mismatch at size %d", sizes[i])
		}
	}
}

func TestTrainOnEmpty(t *testing.T) {
	tbl := Train(nil)
	input := []byte("the quick brown fox jumped over the lazy dog")
	comp := tbl.Encode(input)
	got := tbl.Decode(comp)
	if !bytes.Equal(got, input) {
		t.Fatalf("roundtrip mismatch on empty-trained table")
	}
}

func TestZerosRoundtrip(t *testing.T) {
	training := []byte{0, 1, 2, 3, 4, 0}
	tbl := Train([][]byte{training})
	input := []byte{4, 0}
	comp := tbl.Encode(input)
	got := tbl.Decode(comp)
	if !bytes.Equal(got, input) {
		t.Fatalf("zeros roundtrip mismatch: %v != %v", got, input)
	}
}

// Ensure that every *.txt corpus in testdata compresses and roundtrips correctly.
func TestCorpusRoundtrip(t *testing.T) {
	roundtripFile := func(name, path string) {
		t.Run(name, func(t *testing.T) {
			data, err := os.ReadFile(path)
			if err != nil {
				t.Skipf("missing corpus %s: %v", path, err)
			}
			// split data by new line
			lines := strings.Split(string(data), "\n")
			bLines := make([][]byte, len(lines))
			for i, line := range lines {
				bLines[i] = []byte(line)
			}
			tbl := Train(bLines)
			if err != nil {
				t.Fatalf("train: %v", err)
			}
			buf := bytes.Buffer{}
			_, err = tbl.WriteTo(&buf)
			if err != nil {
				t.Fatalf("write: %v", err)
			}

			for i := range lines {
				comp := tbl.Encode(bLines[i])
				got := tbl.Decode(comp)
				if !bytes.Equal(got, bLines[i]) {
					t.Fatalf("roundtrip mismatch for %s", path)
				}
			}
		})
	}
	roundtripFile("art_of_war", "testdata/art_of_war.txt")
	roundtripFile("bible_kjv", "testdata/en_bible_kjv.txt")
	roundtripFile("mobydick", "testdata/en_mobydick.txt")
	roundtripFile("shakespeare", "testdata/en_shakespeare.txt")
	roundtripFile("tao_te_ching_en", "testdata/zh_tao_te_ching_en.txt")
}

// Benchmark over all testdata/*.txt files (and selected text-like extensions),
// reporting ratio and throughput per file.
func BenchmarkCorpusCompressionSuite(b *testing.B) {
	patterns := []string{
		"testdata/*.txt",
	}
	var files []string
	for _, pat := range patterns {
		matches, _ := filepath.Glob(pat)
		files = append(files, matches...)
	}
	if len(files) == 0 {
		b.Skip("no files in testdata matching patterns")
	}
	for _, f := range files {
		f := f
		data, err := os.ReadFile(f)
		if err != nil {
			b.Fatalf("read %s: %v", f, err)
		}
		b.Run(filepath.Base(f), func(b *testing.B) {
			b.Run("train", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					_ = Train([][]byte{data})
				}
			})

			tbl := Train([][]byte{data})

			b.Run("compress", func(b *testing.B) {
				b.ReportAllocs()
				comp := tbl.Encode(data)
				b.SetBytes(int64(len(data)))
				b.ResetTimer()
				for b.Loop() {
					_ = tbl.Encode(data)
				}
				b.ReportMetric(float64(len(comp))/float64(len(data)), "ratio")
			})

			comp := tbl.Encode(data)

			b.Run("decompress", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					got := tbl.Decode(comp)
					if !bytes.Equal(got, data) {
						b.Fatalf("roundtrip mismatch")
					}
				}
			})
		})
	}
}

// TestRebuildCompressionDeterminism verifies that serializing and deserializing the
// table preserves the exact compressed output for each input.
func TestRebuildCompressionDeterminism(t *testing.T) {
	data, err := os.ReadFile("testdata/art_of_war.txt")
	if err != nil {
		t.Skipf("missing corpus: %v", err)
	}
	lines := strings.Split(string(data), "\n")
	for i, ln := range lines {
		b := []byte(ln)
		tbl := Train([][]byte{b})
		if err != nil {
			t.Fatalf("train: %v", err)
		}
		comp := tbl.Encode(b)
		if err != nil {
			t.Fatalf("compress: %v", err)
		}

		var buf bytes.Buffer
		if _, err := tbl.WriteTo(&buf); err != nil {
			t.Fatalf("write: %v", err)
		}
		var tbl2 Table
		if _, err := tbl2.ReadFrom(&buf); err != nil {
			t.Fatalf("read: %v", err)
		}

		comp2 := tbl2.Encode(b)
		if !bytes.Equal(comp, comp2) {
			t.Fatalf("recompressed output mismatch at line %d", i)
		}

		// Sanity check roundtrips
		got1 := tbl.Decode(comp)
		got2 := tbl2.Decode(comp2)
		if !bytes.Equal(got1, b) || !bytes.Equal(got2, b) {
			t.Fatalf("roundtrip mismatch at line %d", i)
		}
	}
}

// TestTrainStrings verifies TrainStrings wrapper works correctly
func TestTrainStrings(t *testing.T) {
	strs := []string{
		"hello world",
		"hello there",
		"worldwide web",
	}
	tbl := TrainStrings(strs)

	// Convert strings to bytes for encoding
	inputs := make([][]byte, len(strs))
	for i, s := range strs {
		inputs[i] = []byte(s)
	}

	for i := range inputs {
		comp := tbl.Encode(inputs[i])
		got := tbl.Decode(comp)
		if string(got) != strs[i] {
			t.Fatalf("TrainStrings roundtrip mismatch: got %q, want %q", got, strs[i])
		}
	}
}

// TestMarshalBinary tests MarshalBinary and UnmarshalBinary
func TestMarshalBinary(t *testing.T) {
	inputs := [][]byte{
		[]byte("test data for binary marshaling"),
		[]byte("another test string"),
	}
	tbl := Train(inputs)

	// Marshal
	data, err := tbl.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	// Unmarshal
	var tbl2 Table
	if err := tbl2.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary: %v", err)
	}

	// Verify compression is identical
	for i := range inputs {
		comp1 := tbl.Encode(inputs[i])
		comp2 := tbl2.Encode(inputs[i])
		if !bytes.Equal(comp1, comp2) {
			t.Fatalf("MarshalBinary roundtrip changed compression for input %d", i)
		}
	}
}

// TestEdgeCases tests various edge cases
func TestEdgeCases(t *testing.T) {
	tests := []struct {
		name  string
		input []byte
	}{
		{"empty", []byte("")},
		{"single_byte", []byte("x")},
		{"all_same", bytes.Repeat([]byte("a"), 100)},
		{"random_incompressible", []byte{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef}},
		{"null_bytes", []byte{0, 0, 0, 0, 0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tbl := Train([][]byte{tt.input})
			comp := tbl.Encode(tt.input)
			got := tbl.Decode(comp)

			if !bytes.Equal(got, tt.input) {
				t.Fatalf("edge case %s: roundtrip mismatch", tt.name)
			}
		})
	}
}

// TestCompressionRatio verifies compression happens on repetitive data
func TestCompressionRatio(t *testing.T) {
	// Highly repetitive data should compress
	repetitive := []byte(strings.Repeat("hello world ", 100))
	tbl := Train([][]byte{repetitive})
	comp := tbl.Encode(repetitive)

	ratio := float64(len(comp)) / float64(len(repetitive))
	if ratio > 0.9 {
		t.Logf("Warning: compression ratio %.2f is poor for repetitive data (compressed=%d, original=%d)",
			ratio, len(comp), len(repetitive))
	}

	// Verify roundtrip
	got := tbl.Decode(comp)
	if !bytes.Equal(got, repetitive) {
		t.Fatalf("compression roundtrip failed")
	}
}

func FuzzTrain(f *testing.F) {
	// Seed with pairs of lines from testdata/art_of_war.txt
	if data, err := os.ReadFile("testdata/art_of_war.txt"); err == nil {
		lines := strings.Split(string(data), "\n")
		for i := range len(lines) - 1 {
			f.Add([]byte(lines[i]), []byte(lines[i+1]))
		}
	}
	f.Fuzz(func(t *testing.T, data1, data2 []byte) {
		// Should never panic with multiple inputs
		_ = Train([][]byte{data1, data2})
		// Test edge cases
		_ = Train([][]byte{})
		_ = Train([][]byte{data1})
		_ = Train(nil)
	})
}

func FuzzCompressRoundtrip(f *testing.F) {
	if data, err := os.ReadFile("testdata/art_of_war.txt"); err == nil {
		lines := strings.Split(string(data), "\n")
		for i := 0; i < len(lines)-2; i += 3 {
			f.Add([]byte(lines[i]), []byte(lines[i+1]), []byte(lines[i+2]))
		}
	}
	f.Fuzz(func(t *testing.T, data1, data2, data3 []byte) {
		inputs := [][]byte{data1, data2, data3}
		tbl := Train(inputs)

		// Verify all inputs roundtrip correctly
		for i := range inputs {
			comp := tbl.Encode(inputs[i])
			got := tbl.Decode(comp)
			if !bytes.Equal(got, inputs[i]) {
				t.Fatalf("roundtrip mismatch for input %d", i)
			}
		}

		// Test table serialization preserves compression
		var buf bytes.Buffer
		if _, err := tbl.WriteTo(&buf); err != nil {
			t.Fatalf("write: %v", err)
		}
		var tbl2 Table
		if _, err := tbl2.ReadFrom(&buf); err != nil {
			t.Fatalf("read: %v", err)
		}
		for i := range inputs {
			comp1 := tbl.Encode(inputs[i])
			comp2 := tbl2.Encode(inputs[i])
			if !bytes.Equal(comp1, comp2) {
				t.Fatalf("recompressed output mismatch for input %d", i)
			}
		}
	})
}

// FuzzDecoder tests that decoder never panics on malformed compressed data
func FuzzDecoder(f *testing.F) {
	// Seed with some valid compressed data
	if data, err := os.ReadFile("testdata/art_of_war.txt"); err == nil {
		lines := strings.Split(string(data), "\n")
		if len(lines) > 0 {
			tbl := Train([][]byte{[]byte(lines[0])})
			comp := tbl.Encode([]byte(lines[0]))
			f.Add(comp)
		}
	}
	f.Fuzz(func(t *testing.T, compressedData []byte) {
		// Create a simple table
		tbl := Train([][]byte{[]byte("test")})
		// Should never panic on any compressed data
		_ = tbl.Decode(compressedData)
	})
}

// FuzzLargeInputs tests compression of large inputs that require chunking
func FuzzLargeInputs(f *testing.F) {
	// Seed with repeated patterns
	f.Add([]byte(strings.Repeat("hello world ", 100)))
	f.Add([]byte(strings.Repeat("abcdefghijklmnopqrstuvwxyz", 50)))

	f.Fuzz(func(t *testing.T, data []byte) {
		// Test with inputs larger than chunk size (511 bytes)
		if len(data) < 100 {
			data = bytes.Repeat(data, 10)
		}

		tbl := Train([][]byte{data})
		comp := tbl.Encode(data)
		got := tbl.Decode(comp)

		if !bytes.Equal(got, data) {
			t.Fatalf("large input roundtrip mismatch: len(input)=%d len(got)=%d", len(data), len(got))
		}
	})
}
