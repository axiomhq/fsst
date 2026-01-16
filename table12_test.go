package fsst

import (
	"bytes"
	"testing"
)

func TestTable12Basic(t *testing.T) {
	data := [][]byte{
		[]byte("hello world hello world hello"),
		[]byte("the quick brown fox jumps over the lazy dog"),
		[]byte("lorem ipsum dolor sit amet consectetur"),
	}

	tbl := Train12(data)
	if tbl.nSymbols == 0 {
		t.Fatal("expected some learned symbols")
	}

	for _, input := range data {
		encoded := tbl.EncodeAll(input)
		decoded := tbl.DecodeAll(encoded)
		if !bytes.Equal(decoded, input) {
			t.Errorf("roundtrip failed: got %q, want %q", decoded, input)
		}
	}
}

func TestTable12Roundtrip(t *testing.T) {
	inputs := [][]byte{
		[]byte(""),
		[]byte("a"),
		[]byte("ab"),
		[]byte("abc"),
		[]byte("hello"),
		[]byte("the quick brown fox jumps over the lazy dog multiple times to generate patterns"),
		bytes.Repeat([]byte("pattern "), 100),
	}

	tbl := Train12(inputs)

	for _, input := range inputs {
		encoded := tbl.EncodeAll(input)
		decoded := tbl.DecodeAll(encoded)
		if !bytes.Equal(decoded, input) {
			t.Errorf("roundtrip failed for len=%d: got %q, want %q", len(input), decoded, input)
		}
	}
}

func TestTable12Serialize(t *testing.T) {
	data := [][]byte{
		[]byte("serialize test data with patterns patterns patterns"),
		[]byte("more patterns to learn symbols from the data"),
	}

	tbl1 := Train12(data)
	input := []byte("test input for serialization roundtrip")
	encoded1 := tbl1.EncodeAll(input)

	// Serialize
	serialized, err := tbl1.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	// Deserialize
	var tbl2 Table12
	if err := tbl2.UnmarshalBinary(serialized); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	// Verify encoding produces same result
	encoded2 := tbl2.EncodeAll(input)
	if !bytes.Equal(encoded1, encoded2) {
		t.Error("encoded outputs differ after serialization roundtrip")
	}

	// Verify decoding works
	decoded := tbl2.DecodeAll(encoded1)
	if !bytes.Equal(decoded, input) {
		t.Errorf("decode failed: got %q, want %q", decoded, input)
	}
}

func TestTable12Compression(t *testing.T) {
	// Highly repetitive data should compress well
	data := bytes.Repeat([]byte("the quick brown fox jumps "), 1000)
	inputs := [][]byte{data}

	tbl := Train12(inputs)
	encoded := tbl.EncodeAll(data)

	ratio := float64(len(encoded)) / float64(len(data))
	if ratio > 0.8 {
		t.Errorf("compression ratio too high: %.2f (expected < 0.8)", ratio)
	}

	decoded := tbl.DecodeAll(encoded)
	if !bytes.Equal(decoded, data) {
		t.Error("roundtrip failed")
	}
}

func TestTable12AllBytes(t *testing.T) {
	// Test all byte values
	var allBytes [256]byte
	for i := range 256 {
		allBytes[i] = byte(i)
	}
	input := allBytes[:]

	tbl := Train12([][]byte{input})
	encoded := tbl.EncodeAll(input)
	decoded := tbl.DecodeAll(encoded)

	if !bytes.Equal(decoded, input) {
		t.Error("all-bytes roundtrip failed")
	}
}

func TestTable12Strings(t *testing.T) {
	inputs := []string{
		"hello world",
		"the quick brown fox",
		"testing string interface",
	}

	tbl := Train12Strings(inputs)
	if tbl.nSymbols == 0 {
		t.Fatal("expected some learned symbols")
	}

	for _, s := range inputs {
		encoded := tbl.EncodeAll([]byte(s))
		decoded := tbl.DecodeAll(encoded)
		if string(decoded) != s {
			t.Errorf("string roundtrip failed: got %q, want %q", decoded, s)
		}
	}
}

func BenchmarkTable12Encode(b *testing.B) {
	data := bytes.Repeat([]byte("the quick brown fox jumps over the lazy dog "), 1000)
	tbl := Train12([][]byte{data})

	b.ResetTimer()
	b.SetBytes(int64(len(data)))
	for i := 0; i < b.N; i++ {
		_ = tbl.EncodeAll(data)
	}
}

func BenchmarkTable12Decode(b *testing.B) {
	data := bytes.Repeat([]byte("the quick brown fox jumps over the lazy dog "), 1000)
	tbl := Train12([][]byte{data})
	encoded := tbl.EncodeAll(data)

	b.ResetTimer()
	b.SetBytes(int64(len(data)))
	for i := 0; i < b.N; i++ {
		_ = tbl.DecodeAll(encoded)
	}
}
