from memory import UnsafePointer, memcpy
from sys import sizeof

# FSST decoder implementation in Mojo
# Reads serialized table and owns decoder state

alias FSST_ESCAPE_CODE: UInt8 = 255
alias FSST_VERSION: UInt64 = 20190218
alias FSST_CODE_MAX: Int = 512

struct SIMDDecoder:
    """
    FSST decoder that owns its state.
    Reads serialized table format from Go's WriteTo() method.
    """
    var dec_len: UnsafePointer[UInt8]      # [255]byte - code → symbol length
    var dec_symbol: UnsafePointer[UInt64]  # [255]uint64 - code → symbol value
    var n_symbols: UInt16                   # number of learned symbols
    var suffix_lim: UInt16                  # end of unique 2B region

    fn __init__(out self):
        """Initialize empty decoder."""
        self.dec_len = UnsafePointer[UInt8].alloc(255)
        self.dec_symbol = UnsafePointer[UInt64].alloc(255)
        self.n_symbols = 0
        self.suffix_lim = 0

    fn __del__(deinit self):
        """Free allocated memory."""
        self.dec_len.free()
        self.dec_symbol.free()

    fn load_from_bytes(mut self, data: UnsafePointer[UInt8], data_len: Int) -> Bool:
        """
        Load decoder tables from serialized format.

        Format (matches Go's WriteTo):
        - 8 bytes: version header
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

            # Store in decoder tables
            self.dec_len[code] = UInt8(symbol_len)
            self.dec_symbol[code] = symbol_val

        len_histo.free()
        lens.free()
        return True

    fn decode(self, src: UnsafePointer[UInt8], src_len: Int,
              dst: UnsafePointer[UInt8], dst_capacity: Int) -> Int:
        """
        Decode FSST-compressed data.

        Returns: Number of bytes written to dst, or -1 if dst is too small
        """
        var src_pos: Int = 0
        var dst_pos: Int = 0

        while src_pos < src_len:
            var code = src[src_pos]
            src_pos += 1

            if code < FSST_ESCAPE_CODE:
                # Decode learned symbol
                var symbol_len = Int(self.dec_len[Int(code)])
                var symbol_val = self.dec_symbol[Int(code)]

                # Check buffer capacity
                if dst_pos + symbol_len > dst_capacity:
                    return -1

                # SIMD-friendly wide store fast path: store 8 bytes unaligned when safe.
                # This writes extra bytes beyond symbol_len, which will be overwritten by
                # subsequent symbols; guarded to never exceed dst_capacity.
                if dst_pos + 8 <= dst_capacity:
                    var ptr64_wide = (dst + dst_pos).bitcast[UInt64]()
                    ptr64_wide[0] = symbol_val
                else:
                    # Tail-safe precise stores by exact length
                    var dst_ptr = dst + dst_pos
                    if symbol_len == 1:
                        dst_ptr[0] = UInt8(symbol_val)
                    elif symbol_len == 2:
                        # 2-byte store
                        var ptr16 = dst_ptr.bitcast[UInt16]()
                        ptr16[0] = UInt16(symbol_val)
                    elif symbol_len == 3:
                        # 2-byte + 1-byte
                        var ptr16 = dst_ptr.bitcast[UInt16]()
                        ptr16[0] = UInt16(symbol_val)
                        dst_ptr[2] = UInt8(symbol_val >> 16)
                    elif symbol_len == 4:
                        # 4-byte store
                        var ptr32 = dst_ptr.bitcast[UInt32]()
                        ptr32[0] = UInt32(symbol_val)
                    elif symbol_len == 5:
                        # 4-byte + 1-byte
                        var ptr32 = dst_ptr.bitcast[UInt32]()
                        ptr32[0] = UInt32(symbol_val)
                        dst_ptr[4] = UInt8(symbol_val >> 32)
                    elif symbol_len == 6:
                        # 4-byte + 2-byte
                        var ptr32 = dst_ptr.bitcast[UInt32]()
                        var ptr16 = (dst_ptr + 4).bitcast[UInt16]()
                        ptr32[0] = UInt32(symbol_val)
                        ptr16[0] = UInt16(symbol_val >> 32)
                    elif symbol_len == 7:
                        # 4-byte + 2-byte + 1-byte
                        var ptr32 = dst_ptr.bitcast[UInt32]()
                        var ptr16 = (dst_ptr + 4).bitcast[UInt16]()
                        ptr32[0] = UInt32(symbol_val)
                        ptr16[0] = UInt16(symbol_val >> 32)
                        dst_ptr[6] = UInt8(symbol_val >> 48)
                    elif symbol_len == 8:
                        # 8-byte store (single write)
                        var ptr64 = dst_ptr.bitcast[UInt64]()
                        ptr64[0] = symbol_val

                dst_pos += symbol_len
            else:
                # Escape code: next byte is literal
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
    # Allocate decoder struct
    var decoder_ptr = UnsafePointer[SIMDDecoder].alloc(1)

    # Manually initialize fields (avoid constructor issues with C ABI)
    var dec_len_ptr = UnsafePointer[UInt8].alloc(255)
    var dec_symbol_ptr = UnsafePointer[UInt64].alloc(255)

    decoder_ptr[].dec_len = dec_len_ptr
    decoder_ptr[].dec_symbol = dec_symbol_ptr
    decoder_ptr[].n_symbols = 0
    decoder_ptr[].suffix_lim = 0

    # Load and parse table
    if not decoder_ptr[].load_from_bytes(table_data, table_len):
        dec_len_ptr.free()
        dec_symbol_ptr.free()
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
