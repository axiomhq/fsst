//go:build amd64 || arm64

package fsst

import (
	"encoding/binary"
	"math/bits"
	"runtime"
	"unsafe"
)

const (
	decodeLowBits  = uint64(0x7f7f7f7f7f7f7f7f)
	decodeHighBits = uint64(0x8080808080808080)
)

// decodeIntoExactKernel uses little-endian, unaligned 64-bit loads and stores.
// Block and scalar-tail guards keep every pointer access within dst and src
// even when the stream or expected output size is corrupt.
func (t *Table) decodeIntoExactKernel(dst, src []byte) (int, error) {
	if len(src) == 0 {
		return 0, nil
	}

	srcBase := unsafe.Pointer(unsafe.SliceData(src))
	dstBase := unsafe.Pointer(unsafe.SliceData(dst))
	symbolBase := unsafe.Pointer(&t.decSymbol[0])
	lengthBase := unsafe.Pointer(&t.decLen[0])
	srcPos, dstPos := 0, 0
	invalid := uint64(0)

	for len(src)-srcPos >= 8 && len(dst)-dstPos >= 64 {
		codes := *(*uint64)(unsafe.Add(srcBase, srcPos))
		escapeMask := (codes & decodeHighBits) & ((((^codes) & decodeLowBits) + decodeLowBits) ^ decodeHighBits)
		if escapeMask == 0 {
			code0 := byte(codes)
			*(*uint64)(unsafe.Add(dstBase, dstPos)) = *(*uint64)(unsafe.Add(symbolBase, uintptr(code0)*8))
			length0 := uint64(*(*byte)(unsafe.Add(lengthBase, uintptr(code0))))
			invalid |= (length0 - 1) >> 63
			dstPos += int(length0)
			code1 := byte(codes >> 8)
			*(*uint64)(unsafe.Add(dstBase, dstPos)) = *(*uint64)(unsafe.Add(symbolBase, uintptr(code1)*8))
			length1 := uint64(*(*byte)(unsafe.Add(lengthBase, uintptr(code1))))
			invalid |= (length1 - 1) >> 63
			dstPos += int(length1)
			code2 := byte(codes >> 16)
			*(*uint64)(unsafe.Add(dstBase, dstPos)) = *(*uint64)(unsafe.Add(symbolBase, uintptr(code2)*8))
			length2 := uint64(*(*byte)(unsafe.Add(lengthBase, uintptr(code2))))
			invalid |= (length2 - 1) >> 63
			dstPos += int(length2)
			code3 := byte(codes >> 24)
			*(*uint64)(unsafe.Add(dstBase, dstPos)) = *(*uint64)(unsafe.Add(symbolBase, uintptr(code3)*8))
			length3 := uint64(*(*byte)(unsafe.Add(lengthBase, uintptr(code3))))
			invalid |= (length3 - 1) >> 63
			dstPos += int(length3)
			code4 := byte(codes >> 32)
			*(*uint64)(unsafe.Add(dstBase, dstPos)) = *(*uint64)(unsafe.Add(symbolBase, uintptr(code4)*8))
			length4 := uint64(*(*byte)(unsafe.Add(lengthBase, uintptr(code4))))
			invalid |= (length4 - 1) >> 63
			dstPos += int(length4)
			code5 := byte(codes >> 40)
			*(*uint64)(unsafe.Add(dstBase, dstPos)) = *(*uint64)(unsafe.Add(symbolBase, uintptr(code5)*8))
			length5 := uint64(*(*byte)(unsafe.Add(lengthBase, uintptr(code5))))
			invalid |= (length5 - 1) >> 63
			dstPos += int(length5)
			code6 := byte(codes >> 48)
			*(*uint64)(unsafe.Add(dstBase, dstPos)) = *(*uint64)(unsafe.Add(symbolBase, uintptr(code6)*8))
			length6 := uint64(*(*byte)(unsafe.Add(lengthBase, uintptr(code6))))
			invalid |= (length6 - 1) >> 63
			dstPos += int(length6)
			code7 := byte(codes >> 56)
			*(*uint64)(unsafe.Add(dstBase, dstPos)) = *(*uint64)(unsafe.Add(symbolBase, uintptr(code7)*8))
			length7 := uint64(*(*byte)(unsafe.Add(lengthBase, uintptr(code7))))
			invalid |= (length7 - 1) >> 63
			dstPos += int(length7)
			srcPos += 8
			continue
		}
		if codes&0x00ff00ff00ff00ff == 0x00ff00ff00ff00ff {
			// Four consecutive escapes: pack their literal bytes without
			// reclassifying the same input block four times.
			*(*byte)(unsafe.Add(dstBase, dstPos)) = byte(codes >> 8)
			*(*byte)(unsafe.Add(dstBase, dstPos+1)) = byte(codes >> 24)
			*(*byte)(unsafe.Add(dstBase, dstPos+2)) = byte(codes >> 40)
			*(*byte)(unsafe.Add(dstBase, dstPos+3)) = byte(codes >> 56)
			dstPos += 4
			srcPos += 8
			continue
		}

		firstEscape := bits.TrailingZeros64(escapeMask) >> 3
		for i := 0; i < firstEscape; i++ {
			code := byte(codes >> (8 * i))
			*(*uint64)(unsafe.Add(dstBase, dstPos)) = *(*uint64)(unsafe.Add(symbolBase, uintptr(code)*8))
			length := uint64(*(*byte)(unsafe.Add(lengthBase, uintptr(code))))
			invalid |= (length - 1) >> 63
			dstPos += int(length)
		}
		srcPos += firstEscape
		if firstEscape == 7 {
			// The escaped literal is the first byte beyond this block.
			if srcPos+1 >= len(src) {
				return 0, ErrCorrupted
			}
			*(*byte)(unsafe.Add(dstBase, dstPos)) = src[srcPos+1]
			dstPos++
			srcPos += 2
			continue
		}
		literal := byte(codes >> (8 * (firstEscape + 1)))
		*(*byte)(unsafe.Add(dstBase, dstPos)) = literal
		dstPos++
		srcPos += 2
	}

	for srcPos < len(src) {
		code := src[srcPos]
		srcPos++
		if code == escapeCode {
			if srcPos >= len(src) || dstPos >= len(dst) {
				return 0, ErrCorrupted
			}
			dst[dstPos] = src[srcPos]
			dstPos++
			srcPos++
			continue
		}

		symbolLen := int(t.decLen[code])
		if symbolLen == 0 || symbolLen > len(dst)-dstPos {
			return 0, ErrCorrupted
		}
		if len(dst)-dstPos >= 8 {
			*(*uint64)(unsafe.Add(dstBase, dstPos)) = t.decSymbol[code]
		} else {
			var symbol [8]byte
			binary.LittleEndian.PutUint64(symbol[:], t.decSymbol[code])
			copy(dst[dstPos:], symbol[:symbolLen])
		}
		dstPos += symbolLen
	}

	runtime.KeepAlive(dst)
	runtime.KeepAlive(src)
	runtime.KeepAlive(t)
	if invalid != 0 {
		return 0, ErrCorrupted
	}
	return dstPos, nil
}
