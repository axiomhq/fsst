# FSST - Fast Static Symbol Table compression

[![GoDoc](https://godoc.org/github.com/axiomhq/fsst?status.svg)](https://godoc.org/github.com/axiomhq/fsst) [![Go Report Card](https://goreportcard.com/badge/github.com/axiomhq/fsst)](https://goreportcard.com/report/github.com/axiomhq/fsst)

A fast string compression algorithm optimized for random access and decompression speed. FSST learns a compact symbol table from training data to achieve high compression ratios on structured and repetitive text.

## Implementation

This implementation is based on the FSST algorithm described in:

["FSST: Fast Random Access String Compression"](https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf) by Peter Boncz, Thomas Neumann, and Viktor Leis (VLDB 2020).

Key features:
* **Learned symbol tables** with up to 255 symbols (1-8 bytes each)
* **Fast decompression** via direct table lookup (~1-2 GB/s)
* **Compact representation** with 2-8 KB symbol tables
* **Zero-allocation decoding** for high-throughput scenarios
* **Efficient encoding** using hash tables and optimized lookup paths
* **Binary serialization** for persistent storage and reuse
* **Order-independent training** for deterministic results

This implementation provides a balance between compression ratio, decompression speed, and memory efficiency, making it ideal for structured data workloads.

## Usage

```go
// Train on representative data
inputs := [][]byte{
    []byte(`{"id":123,"name":"Alice"}`),
    []byte(`{"id":456,"name":"Bob"}`),
}
tbl := fsst.Train(inputs)

// Compress and decompress
compressed := tbl.Encode([]byte(`{"id":789,"name":"Charlie"}`))
original := tbl.DecodeAll(compressed)

// Or decode into a fixed buffer for zero-allocation
dst := make([]byte, 1024)
n := tbl.Decode(dst, compressed)
_ = dst[:n] // decompressed data

// Serialize table for reuse
data, _ := tbl.MarshalBinary()
var tbl2 fsst.Table
tbl2.UnmarshalBinary(data)
```

## Performance Characteristics

FSST is optimized for workloads requiring fast decompression:

* **Training**: O(n × k) where n is input size, k is number of training rounds (5)
* **Encoding**: ~200-500 MB/s
* **Decoding**: ~1-2 GB/s (table lookup)
* **Compression ratio**: 1.5x to 3x on structured/repetitive text
* **Table size**: 2-8 KB

Best suited for:
* Structured text: JSON, CSV, logs, XML
* Repetitive strings: database dumps, API responses
* Text with patterns: URLs, timestamps, common fields

## Installation

```sh
go get github.com/axiomhq/fsst
```

## Contributing

Kindly check our [contributing guide](https://github.com/axiomhq/fsst/blob/main/Contributing.md) on how to propose bugfixes and improvements, and submitting pull requests to the project.

## License

&copy; Axiom, Inc., 2025

Distributed under MIT License (`The MIT License`).

See [LICENSE](LICENSE) for more information.