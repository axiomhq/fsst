package fsst

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

func TestSIMDDecoderBasic(t *testing.T) {
	input := []byte("hello world hello world hello world")

	// Train table
	table := Train([][]byte{input})

	// Create SIMD decoder
	decoder, err := NewSIMDDecoderFromTable(table)
	if err != nil {
		t.Fatalf("NewSIMDDecoderFromTable failed: %v", err)
	}
	defer decoder.Close()

	// Encode
	compressed := table.Encode(nil, input)

	// Decode with SIMD
	decompressed := decoder.Decode(nil, compressed)

	// Verify
	if !bytes.Equal(input, decompressed) {
		t.Errorf("Roundtrip failed:\nwant: %q\ngot:  %q", input, decompressed)
	}
}

func TestSIMDDecoderFromBytes(t *testing.T) {
	input := []byte("compression test compression test")

	// Train and serialize
	table := Train([][]byte{input})
	var buf bytes.Buffer
	if _, err := table.WriteTo(&buf); err != nil {
		t.Fatalf("WriteTo failed: %v", err)
	}

	// Create decoder from serialized bytes
	decoder, err := NewSIMDDecoder(buf.Bytes())
	if err != nil {
		t.Fatalf("NewSIMDDecoder failed: %v", err)
	}
	defer decoder.Close()

	// Encode and decode
	compressed := table.Encode(nil, input)
	decompressed := decoder.DecodeAll(compressed)

	if !bytes.Equal(input, decompressed) {
		t.Errorf("Roundtrip failed:\nwant: %q\ngot:  %q", input, decompressed)
	}
}

func TestSIMDDecoderEmptyInput(t *testing.T) {
	input := []byte("test data")
	table := Train([][]byte{input})

	decoder, err := NewSIMDDecoderFromTable(table)
	if err != nil {
		t.Fatalf("NewSIMDDecoderFromTable failed: %v", err)
	}
	defer decoder.Close()

	// Decode empty input
	result := decoder.Decode(nil, []byte{})
	if len(result) != 0 {
		t.Errorf("Expected empty result, got: %v", result)
	}
}

func TestSIMDDecoderBufferReuse(t *testing.T) {
	input := []byte("reuse buffer test reuse buffer test")
	table := Train([][]byte{input})

	decoder, err := NewSIMDDecoderFromTable(table)
	if err != nil {
		t.Fatalf("NewSIMDDecoderFromTable failed: %v", err)
	}
	defer decoder.Close()

	compressed := table.Encode(nil, input)

	// First decode with nil buffer
	result1 := decoder.Decode(nil, compressed)

	// Second decode reusing buffer
	buf := make([]byte, 0, len(result1))
	result2 := decoder.Decode(buf, compressed)

	if !bytes.Equal(result1, result2) {
		t.Errorf("Buffer reuse failed:\nfirst:  %q\nsecond: %q", result1, result2)
	}
}

func TestSIMDDecoderVsGoDecoder(t *testing.T) {
	inputs := [][]byte{
		[]byte("hello world"),
		[]byte(""),
		[]byte("a"),
		[]byte("aaaaaaaaaa"),
		[]byte("The quick brown fox jumps over the lazy dog"),
		[]byte("compress compress compress"),
		bytes.Repeat([]byte("pattern "), 100),
	}

	for _, input := range inputs {
		table := Train([][]byte{input})

		// Create SIMD decoder
		decoder, err := NewSIMDDecoderFromTable(table)
		if err != nil {
			t.Fatalf("NewSIMDDecoderFromTable failed for input %q: %v", input, err)
		}

		compressed := table.Encode(nil, input)

		// Decode with both decoders
		goResult := table.Decode(nil, compressed)
		simdResult := decoder.Decode(nil, compressed)

		decoder.Close()

		// Results must match
		if !bytes.Equal(goResult, simdResult) {
			t.Errorf("Decoder mismatch for input %q:\nGo:   %q\nSIMD: %q",
				input, goResult, simdResult)
		}

		// Both must match original
		if !bytes.Equal(input, simdResult) {
			t.Errorf("SIMD decoder failed for input %q:\nwant: %q\ngot:  %q",
				input, input, simdResult)
		}
	}
}

func TestSIMDDecoderInvalidTable(t *testing.T) {
	// Too short
	_, err := NewSIMDDecoder([]byte{1, 2, 3})
	if err == nil {
		t.Error("Expected error for too-short table")
	}

	// Invalid version
	badTable := make([]byte, 32)
	badTable[0] = 0xFF // wrong version
	_, err = NewSIMDDecoder(badTable)
	if err == nil {
		t.Error("Expected error for invalid version")
	}
}

func TestSIMDDecoderMultipleDecoders(t *testing.T) {
	input1 := []byte("first dataset first dataset")
	input2 := []byte("second dataset second dataset")

	table1 := Train([][]byte{input1})
	table2 := Train([][]byte{input2})

	decoder1, err := NewSIMDDecoderFromTable(table1)
	if err != nil {
		t.Fatalf("Failed to create decoder1: %v", err)
	}
	defer decoder1.Close()

	decoder2, err := NewSIMDDecoderFromTable(table2)
	if err != nil {
		t.Fatalf("Failed to create decoder2: %v", err)
	}
	defer decoder2.Close()

	compressed1 := table1.Encode(nil, input1)
	compressed2 := table2.Encode(nil, input2)

	// Decode with correct decoders
	result1 := decoder1.Decode(nil, compressed1)
	result2 := decoder2.Decode(nil, compressed2)

	if !bytes.Equal(input1, result1) {
		t.Errorf("Decoder1 failed:\nwant: %q\ngot:  %q", input1, result1)
	}

	if !bytes.Equal(input2, result2) {
		t.Errorf("Decoder2 failed:\nwant: %q\ngot:  %q", input2, result2)
	}
}

func BenchmarkSIMDDecoder(b *testing.B) {
	input := bytes.Repeat([]byte("benchmark data "), 1000)
	table := Train([][]byte{input})

	decoder, err := NewSIMDDecoderFromTable(table)
	if err != nil {
		b.Fatalf("NewSIMDDecoderFromTable failed: %v", err)
	}
	defer decoder.Close()

	compressed := table.Encode(nil, input)
	buf := make([]byte, 0, len(input))

	b.SetBytes(int64(len(compressed)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = decoder.Decode(buf, compressed)
	}
}

func BenchmarkSIMDDecoderVsGo(b *testing.B) {
	input := bytes.Repeat([]byte("comparison benchmark "), 1000)
	table := Train([][]byte{input})
	compressed := table.Encode(nil, input)

	b.Run("Go", func(b *testing.B) {
		buf := make([]byte, 0, len(input))
		b.SetBytes(int64(len(compressed)))
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			_ = table.Decode(buf, compressed)
		}
	})

	b.Run("SIMD", func(b *testing.B) {
		decoder, err := NewSIMDDecoderFromTable(table)
		if err != nil {
			b.Fatalf("NewSIMDDecoderFromTable failed: %v", err)
		}
		defer decoder.Close()

		buf := make([]byte, 0, len(input))
		b.SetBytes(int64(len(compressed)))
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			_ = decoder.Decode(buf, compressed)
		}
	})
}

func BenchmarkTestdataGoVsMojo(b *testing.B) {
	testFiles := []string{
		"testdata/art_of_war.txt",
		"testdata/en_bible_kjv.txt",
		"testdata/logs_apache_2k.log",
	}

	for _, file := range testFiles {
		data, err := os.ReadFile(file)
		if err != nil {
			b.Skipf("Skipping %s: %v", file, err)
			continue
		}

		// Split into lines
		lines := bytes.Split(data, []byte("\n"))
		if len(lines) == 0 {
			continue
		}

		// Train table on all lines
		table := Train(lines)

		// Encode each line separately
		compressedLines := make([][]byte, len(lines))
		totalCompressed := 0
		totalOriginal := 0
		for i, line := range lines {
			compressedLines[i] = table.Encode(nil, line)
			totalCompressed += len(compressedLines[i])
			totalOriginal += len(line)
		}

		// Get just the filename for benchmark name
		name := file[strings.LastIndex(file, "/")+1:]

		b.Run(name+"/Go", func(b *testing.B) {
			buf := make([]byte, 0, 1024)
			b.SetBytes(int64(totalCompressed))
			b.ReportMetric(float64(totalOriginal)/float64(totalCompressed), "ratio")
			b.ReportMetric(float64(len(lines)), "lines")
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				for _, compressed := range compressedLines {
					_ = table.Decode(buf, compressed)
				}
			}
		})

		b.Run(name+"/Mojo", func(b *testing.B) {
			decoder, err := NewSIMDDecoderFromTable(table)
			if err != nil {
				b.Fatalf("NewSIMDDecoderFromTable failed: %v", err)
			}
			defer decoder.Close()

			buf := make([]byte, 0, 1024)
			b.SetBytes(int64(totalCompressed))
			b.ReportMetric(float64(totalOriginal)/float64(totalCompressed), "ratio")
			b.ReportMetric(float64(len(lines)), "lines")
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				for _, compressed := range compressedLines {
					_ = decoder.Decode(buf, compressed)
				}
			}
		})
	}
}
