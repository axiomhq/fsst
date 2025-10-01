## FSST: Fast Succinct Symbol Table – Project Guide

This repository implements FSST, a byte-level compression scheme that learns a compact codebook (up to 255 multi-byte “symbols”) from sample data, then encodes inputs as single-byte codes plus escapes. It is fast, branch-efficient, and well-suited to natural language and log-like data.

### What FSST delivers
- Learns 1–8 byte symbols that replace frequent substrings with 1-byte codes
- Encoder uses a three-tier lookup: 1-byte table, 2-byte prefix table, and a small hash table for 3–8 bytes
- Decoder is a flat code→bytes table, enabling fast linear-time decompression

---

## High-level algorithm

1) Sampling
- We don’t train on all data. `makeSample` collects about 16 KB of representative 512-byte slices from the inputs.

2) Iterative training
- `Train` runs several passes with increasing inclusion (frac = 8, 38, 68, 98, 128):
  - Parse the sample using the current table (`compressCount`) and count single symbols and (early rounds) symbol pairs.
  - Build symbol candidates from counts (`buildCandidates`), scoring by gain ≈ frequency × length. Consider existing symbols and merged pairs (concat current→next) when allowed.
  - Keep only the top-K candidates via a min-heap and replace the learned symbol set.

3) Finalization
- `finalize` reassigns codes by length and prefix uniqueness to unlock encoder fast paths:
  - 1-byte symbols placed for direct `byteCodes` lookup
  - 2-byte symbols split into unique prefix (fast `shortCodes`) and conflicting (slow path)
  - 3–8 byte symbols handled through a tiny direct-mapped hash

4) Encoding/Decoding
- Encoding (`Encode`/`encodeChunk`) uses the tiered lookup and optional strategy flags for branch reduction.
- Decoding (`Decode`, `DecodeAll`, `decodeAppend`) uses pre-flattened arrays mapping code→bytes.

---

## Key data structures (Table)

Defined in `table.go` (see inline one-liners in the `Table` struct):
- `byteCodes[256]` – 1-byte → packed [length|code]; single-byte symbols and escape fallback
- `shortCodes[65536]` – 2-byte prefix → packed [length|code]; fast path for unique 2-byte symbols
- `hashTab[fsstHashTabSize]` – direct-mapped 3–8B symbols keyed by the first 3 bytes (no chaining)
- `symbols[fsstCodeMax]` – canonical code → symbol (value+length); authoritative for decode
- `nSymbols`, `suffixLim`, `lenHisto` – metadata: symbol count, boundary for unique 2B region, length histogram
- Encoder state: `accelReady`, `noSuffixOpt`, `avoidBranch`, `encBuf`
- Decoder state: `decLen`, `decSymbol`, `decReady`

Symbol representation (in `symbol.go`):
- `symbol.val` – up to 8 bytes, little-endian
- `symbol.icl` – packed metadata: [length:4][code:12][ignoredBits:16]
- `packCodeLength` – packs [length|code] used in code tables
- `fsstHash`, `fsstUnalignedLoad`, masks – hot-path utilities

---

## Training pipeline – function map

File: `train.go`
- `makeSample(inputs)`
  - Builds ~16KB deterministic pseudo-random sample from inputs in 512-byte slices.

- `Train(inputs)`
  - Orchestrates training passes: clears counters, runs `compressCount`, builds candidates via `buildCandidates`, and finalizes.

- `compressCount(t, c, sample, frac)`
  - Parses the sample using the current table. Increments single-symbol counts on each step, and in early rounds increments pair counts for current→next symbol and current→first-byte-of-next-run.

- `buildCandidates(t, c, frac)`
  - Generates candidates from observed counts and merged pairs (when allowed). Scores by gain≈frequency×length, keeps top 255 using a min-heap, resets and installs winners into the table.

- `findNextSymbolFast(t, data, pos)`
  - Returns best match at position using 3–8B hash, then 2B short code, else 1B fallback.

- `TrainStrings(inputs)`
  - Convenience: wraps strings to bytes and calls `Train`.

File: `counters.go`
- `incSingle`, `incPair` – space-efficient counters with an early-increment trick
- `nextSingle`, `nextPair` – iterate over non-zero counts, compensating for early increment

File: `table.go`
- `newTable()`
  - Initializes pseudo 1-byte symbols and default code tables; clears hash.

- `clearSymbols()`
  - Removes learned symbols and restores lookup structures to defaults; clears histogram and count.

- `addSymbol(sym)` / `hashInsert(sym)`
  - Assigns code and installs symbol in the appropriate structure: 1B→`byteCodes`, 2B→`shortCodes`, 3–8B→`hashTab`.

- `findLongestSymbol(tempSym)`
  - Chooses the longest match given a temporary 1–8 byte window: 3–8B via hash, else 2B via shortCodes (unique prefixes), else 1B via byteCodes.

- `finalize()`
  - Reorders codes by length and partitions 2B symbols by unique prefix vs conflicting; sets `suffixLim` and writes back code assignments.

- `rebuildIndices()`
  - Rebuilds `byteCodes`, `shortCodes`, and `hashTab` from `symbols[]`. Idempotent; used when encoding after deserialize.

- `Encode(input)` / `encodeChunk(...)`
  - Chunked encode with unaligned 8-byte loads. Match order: optional 2B fast path (unique prefixes), then 3–8B hash, then 2B, else 1B/escape. Strategy flags:
    - `noSuffixOpt`: skip suffix checks when 2B dominates and is mostly unique
    - `avoidBranch`: branchless path when distribution makes branches costly

- `Decode(dst, src)` / `DecodeAll(src)` / `decodeAppend(dst, src)`
  - Build decoder arrays on first use. For each code: copy 1–8 bytes for learned symbols or read the next literal byte for escapes.

Serialization
- `WriteTo(w)` / `ReadFrom(r)` – compact header (version, `suffixLim`, `nSymbols`), histogram, then symbol bytes; rehydrate into `symbols[]`.
- `MarshalBinary()` / `UnmarshalBinary(data)` – helpers over `WriteTo/ReadFrom`.

Heuristics
- `chooseVariant(t)` – sets `noSuffixOpt` and/or `avoidBranch` based on `lenHisto` and `suffixLim` characteristics.

---

## Practical usage

Training
```go
table := fsst.TrainStrings([]string{"some text", "more text"})
// or: table := fsst.Train([][]byte{b1, b2, ...})
```

Encoding / Decoding
```go
compressed := table.Encode(input)
output := table.DecodeAll(compressed)
```

Persistence
```go
data, _ := table.MarshalBinary()
var t fsst.Table
_ = t.UnmarshalBinary(data)
```

---

## Tips for contributors

- To improve compression ratio: start in `buildCandidates` and `compressCount`. Tweak thresholds (`minCountNumerator/Denominator`, `singleByteBoost`), pair-merge logic, or gain scoring.
- To improve speed: focus on `encodeChunk` ordering and `chooseVariant` thresholds; consider hot-path masking and allocation behavior.
- Always verify: encode→decode roundtrips on `testdata/`, monitor `lenHisto` and `suffixLim`, and benchmark speed/ratio trade-offs.


