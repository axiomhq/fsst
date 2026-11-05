package fsst

import (
	"bytes"
	"math/rand"
	"reflect"
	"testing"
)

// Compare arbitrary code streams against the scalar decoder. Capacity sweeps
// exercise fast blocks, scalar tails, retries, and growth at row boundaries.
func TestDecodeBatchKernelCapacity(t *testing.T) {
	table := fullDecodeTable()
	rng := rand.New(rand.NewSource(42))
	var src, want []byte
	srcOffsets, wantOffsets := []int{0}, []int{0}
	for row := 0; row < 80; row++ {
		encoded := make([]byte, row*3)
		rng.Read(encoded)
		// Complete a possible terminal escape within this row.
		if len(encoded) > 0 && encoded[len(encoded)-1] == escapeCode {
			encoded = append(encoded, 0)
		}
		// A terminal 255 can also be a literal. The reference decides validity.
		decoded, err := referenceDecode(table, encoded)
		if err != nil {
			encoded = append(encoded, 0)
			decoded, err = referenceDecode(table, encoded)
		}
		if err != nil {
			t.Fatal(err)
		}
		src = append(src, encoded...)
		want = append(want, decoded...)
		srcOffsets = append(srcOffsets, len(src))
		wantOffsets = append(wantOffsets, len(want))
	}
	capacities := []int{0, 1, 7, 8, 63, 64, 65, len(want) / 2, len(want) - 1, len(want), len(want) + 64}
	for _, capacity := range capacities {
		backing := bytes.Repeat([]byte{0xa5}, capacity+16)
		// Test the documented alias between input and output offsets as well.
		offsets := append([]int(nil), srcOffsets...)
		got, outOffsets, err := table.DecodeBatch(backing[:0:capacity], offsets, src, offsets)
		if err != nil || !bytes.Equal(got, want) || !reflect.DeepEqual(outOffsets, wantOffsets) {
			t.Fatalf("capacity %d: mismatch, err=%v", capacity, err)
		}
		if !bytes.Equal(backing[capacity:], bytes.Repeat([]byte{0xa5}, 16)) {
			t.Fatalf("capacity %d: out-of-bounds write", capacity)
		}
	}
	dst := make([]byte, 0, len(want))
	offsets := make([]int, 0, len(srcOffsets))
	if allocs := testing.AllocsPerRun(100, func() {
		var err error
		dst, offsets, err = table.DecodeBatch(dst[:0], offsets[:0], src, srcOffsets)
		if err != nil {
			panic(err)
		}
	}); allocs != 0 {
		t.Fatalf("allocations: %v", allocs)
	}
}

func TestDecodeBatchKernelLegacyErrors(t *testing.T) {
	table := fullDecodeTable()
	for _, capacity := range []int{0, 64, 1024} {
		// An escape cannot consume the first byte of the next row.
		src := append(bytes.Repeat([]byte{7}, 16), escapeCode, 1)
		got, offsets, err := table.DecodeBatch(make([]byte, 0, capacity), nil, src, []int{0, 17, 18})
		want, _ := referenceDecode(table, src[:16])
		if err != ErrCorrupted || !bytes.Equal(got, want) || !reflect.DeepEqual(offsets, []int{0}) {
			t.Fatalf("capacity %d: malformed row result differs", capacity)
		}
		// Legacy batch decode skips undefined codes. Preserve that behavior.
		table.decLen[17] = 0
		got, _, err = table.DecodeBatch(make([]byte, 0, capacity), nil, bytes.Repeat([]byte{17}, 80), []int{0, 80})
		table.decLen[17] = 2
		if err != nil || len(got) != 0 {
			t.Fatalf("undefined code: %v", err)
		}
	}
}
