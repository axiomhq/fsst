from memory import UnsafePointer

# FSST decoder implementation in Mojo
# Matches C++ reference implementation with scalar branchless decoding

alias FSST_ESCAPE_CODE: UInt8 = 255
alias FSST_VERSION: UInt64 = 20190218

@register_passable("trivial")
struct DecoderEntry:
    """
    Decoder table entry matching C++ Symbol layout.
    Packed: symbol value (8 bytes) + length (1 byte) in same cache line.
    """
    var symbol: UInt64  # symbol value (union of char buf[8] / uint64)
    var len: UInt8      # symbol length (1-8 bytes)

struct SIMDDecoder:
    """
    FSST decoder matching C++ SymbolTable design.
    Owns decoder state: symbol table loaded from serialized format.
    """
    var entries: UnsafePointer[DecoderEntry]  # [255] decoder entries
    var n_symbols: UInt16
    var suffix_lim: UInt16

    fn __init__(out self):
        """Initialize empty decoder."""
        self.entries = UnsafePointer[DecoderEntry].alloc(255)
        self.n_symbols = 0
        self.suffix_lim = 0

    fn __del__(deinit self):
        """Free allocated memory."""
        self.entries.free()

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

        # Read symbols and build decoder tables
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

            # Store in decoder table
            self.entries[code].symbol = symbol_val
            self.entries[code].len = UInt8(symbol_len)

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

            # All valid codes: parallel table lookups (independent memory ops)
            var entry0 = self.entries[Int(code0)]
            var entry1 = self.entries[Int(code1)]
            var entry2 = self.entries[Int(code2)]
            var entry3 = self.entries[Int(code3)]

            # Compute output positions via prefix sum (breaks dependency chain)
            var len0 = Int(entry0.len)
            var len1 = Int(entry1.len)
            var len2 = Int(entry2.len)
            var len3 = Int(entry3.len)

            var pos0 = dst_pos
            var pos1 = dst_pos + len0
            var pos2 = pos1 + len1
            var pos3 = pos2 + len2

            # Parallel 8-byte stores (independent, no dependency)
            # C++ style: always write 8 bytes, overwrite is harmless
            var ptr0 = (dst + pos0).bitcast[UInt64]()
            var ptr1 = (dst + pos1).bitcast[UInt64]()
            var ptr2 = (dst + pos2).bitcast[UInt64]()
            var ptr3 = (dst + pos3).bitcast[UInt64]()

            ptr0[0] = entry0.symbol
            ptr1[0] = entry1.symbol
            ptr2[0] = entry2.symbol
            ptr3[0] = entry3.symbol

            dst_pos = pos3 + len3
            src_pos += 4

        # Tail loop: handle remaining symbols
        while src_pos < src_len:
            var code = src[src_pos]
            src_pos += 1

            if code < FSST_ESCAPE_CODE:
                var entry = self.entries[Int(code)]
                var symbol_len = Int(entry.len)

                if dst_pos + symbol_len > dst_capacity:
                    return -1

                # 8-byte store when safe, else precise
                var dst_ptr = dst + dst_pos
                if dst_pos + 8 <= dst_capacity:
                    var ptr64 = dst_ptr.bitcast[UInt64]()
                    ptr64[0] = entry.symbol
                else:
                    # Tail: precise stores
                    var symbol_val = entry.symbol
                    if symbol_len == 1:
                        dst_ptr[0] = UInt8(symbol_val)
                    elif symbol_len == 2:
                        var ptr16 = dst_ptr.bitcast[UInt16]()
                        ptr16[0] = UInt16(symbol_val)
                    elif symbol_len == 3:
                        var ptr16 = dst_ptr.bitcast[UInt16]()
                        ptr16[0] = UInt16(symbol_val)
                        dst_ptr[2] = UInt8(symbol_val >> 16)
                    elif symbol_len == 4:
                        var ptr32 = dst_ptr.bitcast[UInt32]()
                        ptr32[0] = UInt32(symbol_val)
                    elif symbol_len == 5:
                        var ptr32 = dst_ptr.bitcast[UInt32]()
                        ptr32[0] = UInt32(symbol_val)
                        dst_ptr[4] = UInt8(symbol_val >> 32)
                    elif symbol_len == 6:
                        var ptr32 = dst_ptr.bitcast[UInt32]()
                        var ptr16 = (dst_ptr + 4).bitcast[UInt16]()
                        ptr32[0] = UInt32(symbol_val)
                        ptr16[0] = UInt16(symbol_val >> 32)
                    elif symbol_len == 7:
                        var ptr32 = dst_ptr.bitcast[UInt32]()
                        var ptr16 = (dst_ptr + 4).bitcast[UInt16]()
                        ptr32[0] = UInt32(symbol_val)
                        ptr16[0] = UInt16(symbol_val >> 32)
                        dst_ptr[6] = UInt8(symbol_val >> 48)
                    elif symbol_len == 8:
                        var ptr64 = dst_ptr.bitcast[UInt64]()
                        ptr64[0] = symbol_val

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
    Create decoder from serialized table bytes.
    Returns opaque pointer to decoder, or NULL on error.
    """
    var decoder_ptr = UnsafePointer[SIMDDecoder].alloc(1)

    # Manually initialize fields
    var entries_ptr = UnsafePointer[DecoderEntry].alloc(255)

    decoder_ptr[].entries = entries_ptr
    decoder_ptr[].n_symbols = 0
    decoder_ptr[].suffix_lim = 0

    # Load and parse table
    if not decoder_ptr[].load_from_bytes(table_data, table_len):
        entries_ptr.free()
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
    Decode compressed data using decoder.
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
