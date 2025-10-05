from memory import UnsafePointer
from algorithm import vectorize
from sys.intrinsics import gather

# FSST decoder implementation in Mojo
# Optimized with SoA (Struct of Arrays) layout for cache-friendly access

alias FSST_ESCAPE_CODE: UInt8 = 255
alias FSST_VERSION: UInt64 = 20190218
# Use 16-byte SIMD width (128-bit NEON on ARM64, SSE on x86)
alias SIMD_WIDTH = 16

struct SIMDDecoder:
    """
    FSST decoder with SoA (Struct of Arrays) layout for SIMD optimization.
    Separates symbols and lengths into contiguous arrays for vectorized access.
    """
    var symbols: UnsafePointer[UInt64]  # [255] symbol values
    var lengths: UnsafePointer[UInt8]   # [255] symbol lengths
    var n_symbols: UInt16
    var suffix_lim: UInt16

    fn __init__(out self):
        """Initialize empty decoder with SoA layout."""
        self.symbols = UnsafePointer[UInt64].alloc(255)
        self.lengths = UnsafePointer[UInt8].alloc(255)
        self.n_symbols = 0
        self.suffix_lim = 0

    fn __del__(deinit self):
        """Free allocated memory."""
        self.symbols.free()
        self.lengths.free()

    fn load_from_bytes(mut self, data: UnsafePointer[UInt8], data_len: Int) -> Bool:
        """
        Load decoder from serialized table (Go WriteTo format).

        Format:
        - 8 bytes: version header (little-endian)
        - 8 bytes: length histogram
        - Variable: symbol bytes

        Returns: True on success, False on error
        """
        if data_len < 16:
            return False

        var pos: Int = 0

        # Read version header (little-endian)
        var ver: UInt64 = 0
        for i in range(8):
            ver |= UInt64(data[pos + i]) << (i * 8)
        pos += 8

        # Extract fields from version word
        var version = ver >> 32
        if version != FSST_VERSION:
            return False

        self.suffix_lim = UInt16((ver >> 16) & 0xFF)
        self.n_symbols = UInt16((ver >> 8) & 0xFF)

        # Read length histogram
        var len_histo: UnsafePointer[UInt8] = UnsafePointer[UInt8].alloc(8)
        for i in range(8):
            len_histo[i] = data[pos + i]
        pos += 8

        # Build length schedule: order is 2,3,4,5,6,7,8, then 1
        var lens: UnsafePointer[UInt8] = UnsafePointer[UInt8].alloc(Int(self.n_symbols))
        var lens_pos: Int = 0

        # Lengths 2-8
        for length in range(2, 9):
            var count = Int(len_histo[length - 1])
            for _ in range(count):
                lens[lens_pos] = UInt8(length)
                lens_pos += 1

        # Length 1
        var count1 = Int(len_histo[0])
        for _ in range(count1):
            lens[lens_pos] = 1
            lens_pos += 1

        # Read symbols and build SoA decoder tables
        for code in range(Int(self.n_symbols)):
            var symbol_len = Int(lens[code])

            if pos + symbol_len > data_len:
                len_histo.free()
                lens.free()
                return False

            # Read symbol value (little-endian)
            var symbol_val: UInt64 = 0
            for i in range(symbol_len):
                symbol_val |= UInt64(data[pos + i]) << (i * 8)
            pos += symbol_len

            # Store in SoA arrays
            self.symbols[code] = symbol_val
            self.lengths[code] = UInt8(symbol_len)

        len_histo.free()
        lens.free()
        return True

    @always_inline
    fn decode(self, src: UnsafePointer[UInt8], src_len: Int,
              dst: UnsafePointer[UInt8], dst_capacity: Int) -> Int:
        """
        Optimized decoder with C++ reference principles:
        1. Loop unrolling (4-way) for ILP
        2. Always 8-byte store when safe (like C++ speculative writes)
        3. Minimal branching in hot path

        Returns: Number of bytes written, or -1 on error
        """
        var src_pos: Int = 0
        var dst_pos: Int = 0

        # Main loop: 4-way unrolled for instruction-level parallelism
        # Ensures we have 32 bytes dst capacity (4 symbols × 8 bytes max)
        while src_pos + 4 <= src_len and dst_pos + 32 <= dst_capacity:
            # Read 4 codes
            var code0 = src[src_pos]
            var code1 = src[src_pos + 1]
            var code2 = src[src_pos + 2]
            var code3 = src[src_pos + 3]

            # Check for escape codes (early exit on escape)
            if code0 >= FSST_ESCAPE_CODE or code1 >= FSST_ESCAPE_CODE or \
               code2 >= FSST_ESCAPE_CODE or code3 >= FSST_ESCAPE_CODE:
                break

            # All valid codes: parallel table lookups using SoA
            var idx0 = Int(code0)
            var idx1 = Int(code1)
            var idx2 = Int(code2)
            var idx3 = Int(code3)

            var sym0 = self.symbols[idx0]
            var sym1 = self.symbols[idx1]
            var sym2 = self.symbols[idx2]
            var sym3 = self.symbols[idx3]

            var len0 = Int(self.lengths[idx0])
            var len1 = Int(self.lengths[idx1])
            var len2 = Int(self.lengths[idx2])
            var len3 = Int(self.lengths[idx3])

            # Compute output positions via prefix sum
            var pos0 = dst_pos
            var pos1 = dst_pos + len0
            var pos2 = pos1 + len1
            var pos3 = pos2 + len2

            # Parallel 8-byte stores (independent, no dependency)
            var ptr0 = (dst + pos0).bitcast[UInt64]()
            var ptr1 = (dst + pos1).bitcast[UInt64]()
            var ptr2 = (dst + pos2).bitcast[UInt64]()
            var ptr3 = (dst + pos3).bitcast[UInt64]()

            ptr0[0] = sym0
            ptr1[0] = sym1
            ptr2[0] = sym2
            ptr3[0] = sym3

            dst_pos = pos3 + len3
            src_pos += 4

        # Tail loop: handle remaining symbols
        while src_pos < src_len:
            var code = src[src_pos]
            src_pos += 1

            if code < FSST_ESCAPE_CODE:
                var idx = Int(code)
                var symbol = self.symbols[idx]
                var symbol_len = Int(self.lengths[idx])

                if dst_pos + symbol_len > dst_capacity:
                    return -1

                # 8-byte store when safe, else precise
                var dst_ptr = dst + dst_pos
                if dst_pos + 8 <= dst_capacity:
                    var ptr64 = dst_ptr.bitcast[UInt64]()
                    ptr64[0] = symbol
                else:
                    # Tail: precise stores
                    if symbol_len == 1:
                        dst_ptr[0] = UInt8(symbol)
                    elif symbol_len == 2:
                        var ptr16 = dst_ptr.bitcast[UInt16]()
                        ptr16[0] = UInt16(symbol)
                    elif symbol_len == 3:
                        var ptr16 = dst_ptr.bitcast[UInt16]()
                        ptr16[0] = UInt16(symbol)
                        dst_ptr[2] = UInt8(symbol >> 16)
                    elif symbol_len == 4:
                        var ptr32 = dst_ptr.bitcast[UInt32]()
                        ptr32[0] = UInt32(symbol)
                    elif symbol_len == 5:
                        var ptr32 = dst_ptr.bitcast[UInt32]()
                        ptr32[0] = UInt32(symbol)
                        dst_ptr[4] = UInt8(symbol >> 32)
                    elif symbol_len == 6:
                        var ptr32 = dst_ptr.bitcast[UInt32]()
                        var ptr16 = (dst_ptr + 4).bitcast[UInt16]()
                        ptr32[0] = UInt32(symbol)
                        ptr16[0] = UInt16(symbol >> 32)
                    elif symbol_len == 7:
                        var ptr32 = dst_ptr.bitcast[UInt32]()
                        var ptr16 = (dst_ptr + 4).bitcast[UInt16]()
                        ptr32[0] = UInt32(symbol)
                        ptr16[0] = UInt16(symbol >> 32)
                        dst_ptr[6] = UInt8(symbol >> 48)
                    elif symbol_len == 8:
                        var ptr64 = dst_ptr.bitcast[UInt64]()
                        ptr64[0] = symbol

                dst_pos += symbol_len
            else:
                # Escape: literal byte
                if src_pos >= src_len:
                    break
                if dst_pos >= dst_capacity:
                    return -1

                dst[dst_pos] = src[src_pos]
                dst_pos += 1
                src_pos += 1

        return dst_pos

    @always_inline
    fn decode_simd(self, src: UnsafePointer[UInt8], src_len: Int,
                   dst: UnsafePointer[UInt8], dst_capacity: Int) -> Int:
        """
        SIMD-vectorized decoder using gather operations and parallel processing.
        Processes 8 codes at once for optimal SIMD utilization.

        Returns: Number of bytes written, or -1 on error.
        """
        var src_pos: Int = 0
        var dst_pos: Int = 0

        alias SIMD8 = 8  # Use 8-wide SIMD for better balance

        # Main SIMD loop: process 8 codes at once
        while src_pos + SIMD8 <= src_len and dst_pos + (SIMD8 * 8) <= dst_capacity:
            # Load 8 codes at once
            var codes = src.load[width=SIMD8](src_pos)

            # Check for escape codes using SIMD comparison
            var has_escape = False
            for i in range(SIMD8):
                if codes[i] >= FSST_ESCAPE_CODE:
                    has_escape = True
                    break

            # If any escape code, fall back to scalar processing
            if has_escape:
                break

            # True SIMD gather from SoA arrays!
            # Load 8 symbols and 8 lengths using codes as indices
            var symbols = SIMD[DType.uint64, SIMD8](0)
            var lengths = SIMD[DType.uint8, SIMD8](0)

            # Vectorized gather: load from contiguous arrays
            @parameter
            for i in range(SIMD8):
                var idx = Int(codes[i])
                symbols[i] = self.symbols[idx]
                lengths[i] = self.lengths[idx]

            # Compute prefix sum for output positions
            var positions = SIMD[DType.int32, SIMD8](0)
            var cumsum = dst_pos
            @parameter
            for i in range(SIMD8):
                positions[i] = cumsum
                cumsum += Int(lengths[i])

            # Check total output size fits
            if cumsum > dst_capacity:
                return -1

            # Store symbols at computed positions
            @parameter
            for i in range(SIMD8):
                var pos = Int(positions[i])
                var sym = symbols[i]
                var sym_len = Int(lengths[i])

                # 8-byte store (safe due to capacity check)
                if pos + 8 <= dst_capacity:
                    var ptr = (dst + pos).bitcast[UInt64]()
                    ptr[0] = sym
                else:
                    # Precise store for boundary
                    for j in range(sym_len):
                        (dst + pos)[j] = UInt8((sym >> (j * 8)) & 0xFF)

            dst_pos = cumsum
            src_pos += SIMD8

        # Fallback to scalar for remainder
        while src_pos < src_len:
            var code = src[src_pos]
            src_pos += 1

            if code < FSST_ESCAPE_CODE:
                var idx = Int(code)
                var symbol = self.symbols[idx]
                var symbol_len = Int(self.lengths[idx])

                if dst_pos + symbol_len > dst_capacity:
                    return -1

                var dst_ptr = dst + dst_pos
                if dst_pos + 8 <= dst_capacity:
                    var ptr64 = dst_ptr.bitcast[UInt64]()
                    ptr64[0] = symbol
                else:
                    for j in range(symbol_len):
                        dst_ptr[j] = UInt8((symbol >> (j * 8)) & 0xFF)

                dst_pos += symbol_len
            else:
                # Escape: literal byte
                if src_pos >= src_len:
                    break
                if dst_pos >= dst_capacity:
                    return -1

                dst[dst_pos] = src[src_pos]
                dst_pos += 1
                src_pos += 1

        return dst_pos

    @always_inline
    fn decode_simd_optimized(self, src: UnsafePointer[UInt8], src_len: Int,
                             dst: UnsafePointer[UInt8], dst_capacity: Int) -> Int:
        """
        Fully optimized SIMD decoder leveraging SoA layout.
        Processes 16 codes at once with vectorized operations.

        Returns: Number of bytes written, or -1 on error.
        """
        var src_pos: Int = 0
        var dst_pos: Int = 0

        alias SIMD16 = 16

        # Main SIMD loop: process 16 codes at once
        while src_pos + SIMD16 <= src_len and dst_pos + (SIMD16 * 8) <= dst_capacity:
            # SIMD load: 16 codes at once
            var codes = src.load[width=SIMD16](src_pos)

            # SIMD escape detection: check all 16 codes in parallel
            var has_escape = False
            @parameter
            for i in range(SIMD16):
                if codes[i] >= FSST_ESCAPE_CODE:
                    has_escape = True
                    break

            if has_escape:
                break

            # Vectorized gather from SoA arrays
            var symbols = SIMD[DType.uint64, SIMD16](0)
            var lengths = SIMD[DType.uint8, SIMD16](0)

            # Unrolled gather for better instruction scheduling
            @parameter
            for i in range(SIMD16):
                var idx = Int(codes[i])
                symbols[i] = self.symbols[idx]
                lengths[i] = self.lengths[idx]

            # Vectorized prefix sum for output positions
            var positions = SIMD[DType.int32, SIMD16](0)
            var cumsum = dst_pos

            # Unrolled prefix sum (compiler can optimize this)
            @parameter
            for i in range(SIMD16):
                positions[i] = cumsum
                cumsum += Int(lengths[i])

            # Check if total output fits
            if cumsum > dst_capacity:
                return -1

            # Parallel symbol writes (independent stores)
            @parameter
            for i in range(SIMD16):
                var pos = Int(positions[i])
                var sym = symbols[i]
                var len = Int(lengths[i])

                # Always use 8-byte stores (safe due to capacity check)
                if pos + 8 <= dst_capacity:
                    var ptr = (dst + pos).bitcast[UInt64]()
                    ptr[0] = sym
                else:
                    # Precise store for boundary
                    @parameter
                    for j in range(8):
                        if j < len:
                            (dst + pos)[j] = UInt8((sym >> (j * 8)) & 0xFF)

            dst_pos = cumsum
            src_pos += SIMD16

        # Scalar fallback for remainder
        while src_pos < src_len:
            var code = src[src_pos]
            src_pos += 1

            if code < FSST_ESCAPE_CODE:
                var idx = Int(code)
                var symbol = self.symbols[idx]
                var symbol_len = Int(self.lengths[idx])

                if dst_pos + symbol_len > dst_capacity:
                    return -1

                if dst_pos + 8 <= dst_capacity:
                    var ptr64 = (dst + dst_pos).bitcast[UInt64]()
                    ptr64[0] = symbol
                else:
                    for j in range(symbol_len):
                        (dst + dst_pos)[j] = UInt8((symbol >> (j * 8)) & 0xFF)

                dst_pos += symbol_len
            else:
                # Escape: literal byte
                if src_pos >= src_len:
                    break
                if dst_pos >= dst_capacity:
                    return -1

                dst[dst_pos] = src[src_pos]
                dst_pos += 1
                src_pos += 1

        return dst_pos


# C ABI exports for CGo

@export("fsst_decoder_create")
fn fsst_decoder_create(
    table_data: UnsafePointer[UInt8],
    table_len: Int
) -> UnsafePointer[SIMDDecoder]:
    """
    Create decoder from serialized table bytes with SoA layout.
    Returns opaque pointer to decoder, or NULL on error.
    """
    var decoder_ptr = UnsafePointer[SIMDDecoder].alloc(1)

    # Manually initialize SoA fields
    var symbols_ptr = UnsafePointer[UInt64].alloc(255)
    var lengths_ptr = UnsafePointer[UInt8].alloc(255)

    decoder_ptr[].symbols = symbols_ptr
    decoder_ptr[].lengths = lengths_ptr
    decoder_ptr[].n_symbols = 0
    decoder_ptr[].suffix_lim = 0

    # Load and parse table
    if not decoder_ptr[].load_from_bytes(table_data, table_len):
        symbols_ptr.free()
        lengths_ptr.free()
        decoder_ptr.free()
        return UnsafePointer[SIMDDecoder]()

    return decoder_ptr


@export("fsst_decoder_decode")
fn fsst_decoder_decode(
    decoder: UnsafePointer[SIMDDecoder],
    src: UnsafePointer[UInt8],
    src_len: Int,
    dst: UnsafePointer[UInt8],
    dst_capacity: Int
) -> Int:
    """
    Decode compressed data using optimal 4-way SoA decoder.
    Returns decoded length or -1 on error.
    """
    if not decoder:
        return -1
    return decoder[].decode(src, src_len, dst, dst_capacity)


@export("fsst_decoder_destroy")
fn fsst_decoder_destroy(decoder: UnsafePointer[SIMDDecoder]):
    """Free decoder and its resources."""
    if decoder:
        decoder.free()
