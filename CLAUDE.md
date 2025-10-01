# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FSST (Fast Static Symbol Table) is a string compression library that learns up to 255 symbols (1-8 bytes each) from training data to achieve fast decompression speeds (~1-2 GB/s) on structured text. This is a Go implementation based on the VLDB 2020 paper by Boncz, Neumann, and Leis.

## Development Commands

### Testing
```bash
# Run all tests
go test

# Run specific test
go test -run TestName

# Run tests with verbose output
go test -v

# Run benchmarks
go test -bench=.

# Run benchmarks with memory stats
go test -bench=. -benchmem
```

### Building
```bash
# Build the package
go build

# Install as a library
go get github.com/axiomhq/fsst
```

### Code Quality
```bash
# Format code
go fmt ./...

# Run static analysis
go vet ./...

# View documentation
go doc -all
```

## Architecture Overview

### Core Components

**Three-phase pipeline:**
1. **Training** (`train.go`) - Learns optimal symbol table from sample data
2. **Encoding** (`table.go`) - Compresses data using learned symbols
3. **Decoding** (`table.go`) - Decompresses using table lookups

### Key Data Structures

**Table** (`table.go:14-43`)
The central compression/decompression structure containing:
- `byteCodes[256]` - Fast path for 1-byte symbols and escape codes
- `shortCodes[65536]` - Fast path for 2-byte symbols with unique prefixes
- `hashTab[fsstHashTabSize]` - Direct-mapped hash for 3-8 byte symbols
- `symbols[fsstCodeMax]` - Canonical code→symbol mapping for decoding
- Lazy initialization: encoder tables built on first `Encode()`, decoder tables on first `Decode()`

**symbol** (`symbol.go`)
Packed representation of multi-byte patterns:
- `val` - up to 8 bytes (little-endian)
- `icl` - packed metadata: [length:4][code:12][ignoredBits:16]

**counters** (`counters.go`)
Space-efficient frequency tracking during training with early-increment optimization

### Training Pipeline (5 iterations)

1. **Sampling** (`makeSample`) - Collects ~16KB of representative 512-byte slices
2. **Iterative learning** (`Train`, iterations with frac=8,38,68,98,128):
   - `compressCount`: Parse sample with current table, count symbol usage and pairs
   - `buildCandidates`: Generate candidates from counts, score by gain (frequency × length), keep top 255
   - Early rounds merge symbol pairs to form longer symbols
3. **Finalization** (`finalize`): Reorder codes by length, partition 2-byte symbols into fast/slow paths based on prefix uniqueness

### Encoding Strategy (three-tier lookup)

Located in `table.go:encodeChunk()`:
1. **Optional 2-byte fast path** - When `suffixLim` indicates many unique 2-byte prefixes
2. **3-8 byte hash lookup** - Direct-mapped hash table keyed by first 3 bytes
3. **2-byte shortCodes fallback** - Full 2-byte prefix table
4. **1-byte/escape** - Single byte or escape sequence for unmatched bytes

Strategy flags (`chooseVariant`):
- `noSuffixOpt` - Skip suffix checks when 2-byte path dominates
- `avoidBranch` - Use branchless code when branches are costly

### Decoding (flat table lookup)

Located in `table.go:Decode()`, `DecodeAll()`:
- Pre-computed `decLen[255]` and `decSymbol[255]` arrays
- Each code maps to length + symbol bytes (up to 8 bytes)
- Escape codes trigger reading next literal byte
- Zero allocations when using `Decode(dst, src)`

## File Organization

- `doc.go` - Package documentation with usage examples
- `table.go` - Core Table struct, encoding/decoding, serialization
- `train.go` - Training algorithm and sample generation
- `symbol.go` - Symbol representation and bit-packing utilities
- `counters.go` - Space-efficient frequency counters
- `*_test.go` - Unit and integration tests
- `testdata/` - Test corpora for validation
- `docs/FSST.md` - Detailed algorithm walkthrough with examples

## Important Implementation Details

### Encoding Hot Path
- Uses unaligned 8-byte loads (`fsstUnalignedLoad`) for performance
- Chunk-based processing (4KB chunks with padding)
- Strategy selection based on symbol statistics to minimize branches

### Training Determinism
- Uses fixed RNG seed (`rngSeed = 4637947`) for reproducible samples
- Order-independent counting ensures same input produces same table

### Serialization Format
Compact binary format in `WriteTo()`/`ReadFrom()`:
- Header: version (8 bytes), suffixLim (2), nSymbols (2), lenHisto (16)
- Symbol data: packed symbols with lengths

### Zero-Allocation Decoding
- `Decode(dst, src)` writes into provided buffer
- Decoder tables built once and reused
- Critical for high-throughput scenarios

## Performance Characteristics

- **Training**: O(n × 5) where n is sample size (~16KB), 5 rounds
- **Encoding**: ~200-500 MB/s, O(input size)
- **Decoding**: ~1-2 GB/s, O(compressed size)
- **Table size**: 2-8 KB
- **Compression ratio**: 1.5-3x on structured text (JSON, logs, CSV)

## Testing Patterns

Test files follow standard Go conventions:
- `*_test.go` files contain table-driven tests
- `TestCorpusRoundtrip` validates against real-world data in `testdata/`
- Determinism tests ensure reproducible training
- Roundtrip tests verify encode→decode correctness
