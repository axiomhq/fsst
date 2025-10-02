# FSST SIMD Decoder (Mojo)

High-performance FSST decoder implementation in Mojo with full SIMD optimizations.

## Architecture

```
Go Training/Encoding (table.go)
    ↓ serialize table
SIMDDecoder constructor (simd_decoder.go)
    ↓ CGo
Mojo decoder (fsst_decoder.mojo)
    ↓ owns state, SIMD-optimized decode
Static library (libfsst_decoder.a)
```

## Files

- **fsst_decoder.mojo** - Mojo decoder with state ownership and table parsing
- **mojoproject.toml** - Mojo/Magic project configuration
- **../simd_decoder.go** - CGo wrapper with `NewSIMDDecoder()` constructor
- **../build_mojo.sh** - Build script (run from repo root)

## Building

```bash
# From repo root
./build_mojo.sh

# Produces: mojo/libfsst_decoder.a (static library for CGo)
```

### SIMD Features Enabled

| Architecture | SIMD Extensions |
|-------------|----------------|
| ARM64 (Apple Silicon) | NEON, FP-ARMV8, Crypto |
| ARM64 (Linux) | NEON, FP-ARMV8, Crypto, SVE |
| x86_64 | AVX2, FMA, BMI2, AVX-512 (F/CD/BW/DQ/VL) |

The build script auto-detects your architecture and applies appropriate SIMD optimizations.

## Usage

### Basic Usage

```go
import "github.com/axiomhq/fsst"

// Train table
table := fsst.Train(data)

// Serialize and create SIMD decoder
decoder, err := fsst.NewSIMDDecoderFromTable(table)
if err != nil {
    panic(err)
}
defer decoder.Close()

// Encode (still uses Go encoder)
compressed := table.Encode(nil, input)

// Decode with SIMD (Mojo)
decompressed := decoder.Decode(nil, compressed)
```

### From Serialized Bytes

```go
// If you have serialized table bytes
tableBytes := ... // from table.WriteTo() or network/disk

decoder, err := fsst.NewSIMDDecoder(tableBytes)
if err != nil {
    panic(err)
}
defer decoder.Close()

decompressed := decoder.DecodeAll(compressed)
```

## API

### Constructor

```go
func NewSIMDDecoder(tableBytes []byte) (*SIMDDecoder, error)
```

Creates decoder from serialized table bytes (same format as `Table.WriteTo()`).

**Separation of concerns:**
- Mojo decoder owns its state
- Reads and parses table format internally
- No per-call overhead of passing arrays

### Methods

```go
func (d *SIMDDecoder) Decode(buf, src []byte) []byte
func (d *SIMDDecoder) DecodeAll(src []byte) []byte
func (d *SIMDDecoder) Close()
```

- `Decode()` - Decompress with optional buffer reuse
- `DecodeAll()` - Decompress with fresh allocation
- `Close()` - Free Mojo decoder (required, prevents memory leak)

## Data Flow

1. **Go serializes table** → bytes (version + lenHisto + symbols)
2. **NewSIMDDecoder()** → passes bytes to Mojo via CGo
3. **Mojo parses table** → builds `dec_len[255]`, `dec_symbol[255]` arrays
4. **Decode calls** → pure Mojo decode loop, SIMD-optimized
5. **Close()** → frees Mojo memory

## Performance

- **State ownership**: No array passing on every decode call
- **SIMD vectorization**: Mojo compiler leverages NEON/AVX-512
- **Static linking**: Embedded in Go binary via CGo
- **Optimized memory stores**: 16/32/64-bit wide writes for symbols

## Requirements

- Mojo installed (`magic run mojo`)
- CGo enabled (`CGO_ENABLED=1`)
- C toolchain (clang/gcc)

## Platform Support

- **macOS**: arm64 (Apple Silicon) or x86_64 (Intel)
- **Linux**: arm64 or x86_64
- **Windows**: Not supported (Mojo limitation)

## Comparison: Go vs SIMD Decoder

| Feature | Go Decoder | SIMD Decoder (Mojo) |
|---------|-----------|---------------------|
| Performance | Fast (~1-2 GB/s) | Faster (SIMD-optimized) |
| Dependencies | None | Requires Mojo + CGo |
| State | Lazy-initialized arrays | Owns decoder state |
| Distribution | Pure Go | Static lib via CGo |
| Platforms | All Go platforms | macOS/Linux (arm64/x86_64) |

Use SIMD decoder when:
- Maximum decode performance is critical
- Running on x86_64 or ARM64
- Can tolerate Mojo build dependency
- Want separation of encoder/decoder concerns
