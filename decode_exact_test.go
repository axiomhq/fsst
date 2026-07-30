package fsst

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"testing"
)

func fullDecodeTable() *Table {
	t := &Table{nSymbols: 255}
	for code := range 255 {
		length := code%8 + 1
		var symbol [8]byte
		for i := range length {
			symbol[i] = byte(code + i*29)
		}
		t.decLen[code] = byte(length)
		t.decSymbol[code] = binary.LittleEndian.Uint64(symbol[:])
	}
	return t
}

func referenceDecode(t *Table, src []byte) ([]byte, error) {
	dst := make([]byte, 0, len(src)*4)
	for pos := 0; pos < len(src); {
		code := src[pos]
		pos++
		if code == escapeCode {
			if pos == len(src) {
				return nil, ErrCorrupted
			}
			dst = append(dst, src[pos])
			pos++
			continue
		}
		if uint16(code) >= t.nSymbols || t.decLen[code] == 0 {
			return nil, ErrCorrupted
		}
		var symbol [8]byte
		binary.LittleEndian.PutUint64(symbol[:], t.decSymbol[code])
		dst = append(dst, symbol[:t.decLen[code]]...)
	}
	return dst, nil
}

func TestDecodeIntoExactRoundTrip(t *testing.T) {
	tests := []struct {
		name  string
		input []byte
	}{
		{name: "empty"},
		{name: "one_byte", input: []byte("x")},
		{name: "short_tail", input: []byte("the quick brown fox")},
		{name: "long", input: bytes.Repeat([]byte("https://logs.example.com/v1/tenant/common/path/event/00001234\n"), 256)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			table := Train([][]byte{tt.input})
			compressed := table.EncodeAll(tt.input)
			backing := bytes.Repeat([]byte{0xa5}, len(tt.input)+16)
			buf := backing[:0:len(tt.input)]
			decoded, err := table.DecodeIntoExact(buf, compressed, len(tt.input))
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(decoded, tt.input) {
				t.Fatalf("decoded mismatch: got %q, want %q", decoded, tt.input)
			}
			if !bytes.Equal(backing[len(tt.input):], bytes.Repeat([]byte{0xa5}, 16)) {
				t.Fatal("decoder wrote past the exact-capacity output")
			}
			if legacy := table.Decode(nil, compressed); !bytes.Equal(decoded, legacy) {
				t.Fatal("exact and legacy decoders disagree")
			}
			trusted, err := table.DecodeIntoExactTrusted(make([]byte, 0, len(tt.input)), compressed, len(tt.input))
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(trusted, decoded) {
				t.Fatal("trusted and checked exact decoders disagree")
			}
		})
	}
}

func TestDecodeIntoExactEscapesAndTails(t *testing.T) {
	table := fullDecodeTable()
	for escapePos := range 8 {
		t.Run(fmt.Sprintf("escape_at_%d", escapePos), func(t *testing.T) {
			src := bytes.Repeat([]byte{17}, 96)
			src[escapePos] = escapeCode
			src = append(src, 0)
			copy(src[escapePos+2:], src[escapePos+1:])
			src[escapePos+1] = byte(200 + escapePos)

			want, err := referenceDecode(table, src)
			if err != nil {
				t.Fatal(err)
			}
			got, err := table.DecodeIntoExact(make([]byte, 0, len(want)), src, len(want))
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("got %x, want %x", got, want)
			}
		})
	}

	t.Run("mixed_escape_blocks", func(t *testing.T) {
		rng := rand.New(rand.NewSource(1))
		for iteration := 0; iteration < 500; iteration++ {
			src := make([]byte, 0, 256)
			for range 1 + rng.Intn(128) {
				if rng.Intn(5) == 0 {
					src = append(src, escapeCode, byte(rng.Intn(256)))
				} else {
					src = append(src, byte(rng.Intn(255)))
				}
			}
			want, err := referenceDecode(table, src)
			if err != nil {
				t.Fatal(err)
			}
			got, err := table.DecodeIntoExact(make([]byte, 0, len(want)), src, len(want))
			if err != nil {
				t.Fatalf("iteration %d: %v", iteration, err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("iteration %d mismatch", iteration)
			}
			safe := make([]byte, len(want))
			n, err := table.decodeIntoExactSafe(safe, src)
			if err != nil || n != len(want) || !bytes.Equal(safe, want) {
				t.Fatalf("iteration %d portable mismatch: n=%d err=%v", iteration, n, err)
			}
		}
	})
}

func TestDecodeIntoExactRejectsCorruption(t *testing.T) {
	table := fullDecodeTable()
	validSrc := bytes.Repeat([]byte{7}, 32)
	want, err := referenceDecode(table, validSrc)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name       string
		bufCap     int
		src        []byte
		decodedLen int
		wantErr    error
	}{
		{name: "short_buffer", bufCap: len(want) - 1, src: validSrc, decodedLen: len(want), wantErr: io.ErrShortBuffer},
		{name: "decoded_size_too_small", bufCap: len(want) - 1, src: validSrc, decodedLen: len(want) - 1, wantErr: ErrCorrupted},
		{name: "decoded_size_too_large", bufCap: len(want) + 1, src: validSrc, decodedLen: len(want) + 1, wantErr: ErrCorrupted},
		{name: "trailing_escape", bufCap: 1, src: []byte{escapeCode}, decodedLen: 1, wantErr: ErrCorrupted},
		{name: "negative_size", decodedLen: -1, wantErr: ErrCorrupted},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := make([]byte, 0, max(tt.bufCap, 0))
			got, err := table.DecodeIntoExact(buf, tt.src, tt.decodedLen)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("got error %v, want %v", err, tt.wantErr)
			}
			if len(got) != 0 {
				t.Fatalf("got %d output bytes after error", len(got))
			}
		})
	}

	t.Run("code_outside_incomplete_table", func(t *testing.T) {
		incomplete := &Table{nSymbols: 1}
		incomplete.decLen[0] = 1
		incomplete.decSymbol[0] = 'x'
		if _, err := incomplete.DecodeIntoExact(nil, []byte{1}, 0); !errors.Is(err, ErrCorrupted) {
			t.Fatalf("got %v, want ErrCorrupted", err)
		}

		// Keep enough valid output for the invalid code to pass through the
		// eight-code kernel, and choose the expected size as if it emitted zero
		// bytes. The kernel must still reject it.
		src := make([]byte, 72)
		src[3] = 1
		if _, err := incomplete.DecodeIntoExact(make([]byte, 0, 71), src, 71); !errors.Is(err, ErrCorrupted) {
			t.Fatalf("block kernel got %v, want ErrCorrupted", err)
		}
	})
}

var benchmarkDecoded []byte

func BenchmarkDecodeIntoExactCommonPrefix(b *testing.B) {
	const rows = 65_536
	inputs := make([][]byte, rows)
	var plain []byte
	for i := range inputs {
		inputs[i] = fmt.Appendf(nil, "https://logs.example.com/v1/tenant/common/path/event/%08d", i)
		plain = append(plain, inputs[i]...)
	}
	table := Train(inputs)
	var compressed []byte
	scratch := make([]byte, 0, 256)
	for _, input := range inputs {
		encoded := table.Encode(scratch[:0], input)
		compressed = append(compressed, encoded...)
	}
	legacyCapacity := max(len(plain)+7, len(compressed)*2+8)
	legacyCheck := table.Decode(make([]byte, 0, legacyCapacity), compressed)
	exactCheck, err := table.DecodeIntoExact(make([]byte, 0, len(plain)), compressed, len(plain))
	if err != nil || !bytes.Equal(legacyCheck, plain) || !bytes.Equal(exactCheck, plain) {
		b.Fatalf("benchmark roundtrip mismatch: exact error %v", err)
	}

	b.Run("Legacy", func(b *testing.B) {
		buf := make([]byte, 0, legacyCapacity)
		b.SetBytes(int64(len(plain)))
		b.ReportAllocs()
		for b.Loop() {
			benchmarkDecoded = table.Decode(buf, compressed)
		}
	})
	b.Run("Exact", func(b *testing.B) {
		buf := make([]byte, 0, len(plain))
		b.SetBytes(int64(len(plain)))
		b.ReportAllocs()
		for b.Loop() {
			var err error
			benchmarkDecoded, err = table.DecodeIntoExact(buf, compressed, len(plain))
			if err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("ExactTrusted", func(b *testing.B) {
		buf := make([]byte, 0, len(plain))
		b.SetBytes(int64(len(plain)))
		b.ReportAllocs()
		for b.Loop() {
			var err error
			benchmarkDecoded, err = table.DecodeIntoExactTrusted(buf, compressed, len(plain))
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkDecodeIntoExactEscapes(b *testing.B) {
	const decodedBytes = 1 << 20
	table := fullDecodeTable()
	compressed := make([]byte, decodedBytes*2)
	for i := range decodedBytes {
		compressed[2*i] = escapeCode
		compressed[2*i+1] = byte(i)
	}
	legacyCapacity := len(compressed)*2 + 8
	want := make([]byte, decodedBytes)
	for i := range want {
		want[i] = byte(i)
	}
	legacyCheck := table.Decode(make([]byte, 0, legacyCapacity), compressed)
	exactCheck, err := table.DecodeIntoExact(make([]byte, 0, decodedBytes), compressed, decodedBytes)
	if err != nil || !bytes.Equal(legacyCheck, want) || !bytes.Equal(exactCheck, want) {
		b.Fatalf("benchmark roundtrip mismatch: exact error %v", err)
	}

	b.Run("Legacy", func(b *testing.B) {
		buf := make([]byte, 0, legacyCapacity)
		b.SetBytes(decodedBytes)
		b.ReportAllocs()
		for b.Loop() {
			benchmarkDecoded = table.Decode(buf, compressed)
		}
	})
	b.Run("Exact", func(b *testing.B) {
		buf := make([]byte, 0, decodedBytes)
		b.SetBytes(decodedBytes)
		b.ReportAllocs()
		for b.Loop() {
			var err error
			benchmarkDecoded, err = table.DecodeIntoExact(buf, compressed, decodedBytes)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}
