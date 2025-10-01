// Package fsst provides fast string compression via learned symbol tables.
//
// # Overview
//
// FSST (Fast Static Symbol Table) is a compression algorithm optimized for
// strings with repetitive patterns. It learns up to 255 symbols (1-8 bytes each)
// from training data and encodes text by replacing matches with single-byte codes.
//
// # When to Use FSST
//
// FSST excels at compressing:
//   - Structured text: JSON, CSV, logs, XML
//   - Repetitive strings: database dumps, API responses
//   - Text with common patterns: URLs, email addresses, timestamps
//
// Typical compression ratios: 1.5x to 3x, depending on repetitiveness.
//
// # When NOT to Use FSST
//
// FSST is not suitable for:
//   - Binary data (use gzip, zstd, or specialized codecs)
//   - Random or encrypted data (incompressible)
//   - Datasets without shared patterns across records
//   - Single-use compression (training cost exceeds benefit)
//
// # Tradeoffs vs Other Compression
//
// Compared to gzip/zstd:
//   - Much faster decompression (~5-10x faster)
//   - Smaller model size (~2-8KB vs 32KB+ dictionaries)
//   - Deterministic and cache-friendly
//   - Lower compression ratio
//   - Requires training phase
//
// Compared to LZ4:
//   - Better compression on structured text
//   - Smaller model
//   - Requires training phase
//   - Slower than LZ4 for generic data
//
// # Basic Usage
//
//	// Train on representative data
//	inputs := [][]byte{
//	    []byte(`{"id":123,"name":"Alice"}`),
//	    []byte(`{"id":456,"name":"Bob"}`),
//	}
//	tbl := fsst.Train(inputs)
//
//	// Compress and decompress
//	compressed := tbl.Encode([]byte(`{"id":789,"name":"Charlie"}`))
//	original := tbl.DecodeAll(compressed)
//
//	// Or decode into a fixed buffer
//	dst := make([]byte, 1024)
//	n := tbl.Decode(dst, compressed)
//	_ = dst[:n] // decompressed data
//
//	// Serialize table for reuse
//	data, _ := tbl.MarshalBinary()
//	var tbl2 fsst.Table
//	tbl2.UnmarshalBinary(data)
//
// # Performance Characteristics
//
// Training: O(n × k) where n is input size, k is number of rounds (5)
// Encoding: O(m) where m is output size, ~200-500 MB/s
// Decoding: O(m) where m is output size, ~1-2 GB/s (table lookup)
//
// The table is ~2-8KB and encodes/decodes millions of strings per second.
package fsst
