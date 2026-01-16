// Package fsst provides fast string compression via learned symbol tables.
//
// # Overview
//
// FSST (Fast Static Symbol Table) is a compression algorithm optimized for
// strings with repetitive patterns. It learns symbols (1-8 bytes each)
// from training data and encodes text by replacing matches with codes.
//
// This package provides two implementations:
//   - FSST (8-bit codes): Up to 255 symbols, escape mechanism for literals
//   - FSST12 (12-bit codes): Up to 3840 symbols, no escape needed
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
// # FSST vs FSST12
//
// FSST (8-bit codes):
//   - Faster encoding and decoding
//   - Uses escape codes for literal bytes
//   - Best for text with good symbol coverage
//
// FSST12 (12-bit codes):
//   - More symbols available (3840 vs 255)
//   - No escape overhead (all bytes have codes)
//   - Better for diverse text with many patterns
//   - Outputs 12-bit packed codes (1.5 bytes per code)
//
// # Basic Usage
//
//	// FSST (8-bit)
//	inputs := [][]byte{
//	    []byte(`{"id":123,"name":"Alice"}`),
//	    []byte(`{"id":456,"name":"Bob"}`),
//	}
//	tbl := fsst.Train(inputs)
//	compressed := tbl.EncodeAll([]byte(`{"id":789,"name":"Charlie"}`))
//	original := tbl.DecodeAll(compressed)
//
//	// FSST12 (12-bit)
//	tbl12 := fsst.Train12(inputs)
//	compressed12 := tbl12.EncodeAll([]byte(`{"id":789,"name":"Charlie"}`))
//	original12 := tbl12.DecodeAll(compressed12)
//
//	// Serialize for reuse
//	data, _ := tbl.MarshalBinary()
//	var tbl2 fsst.Table
//	tbl2.UnmarshalBinary(data)
//
// # Performance Characteristics
//
// Training: O(n × k) where n is input size, k is number of rounds (5)
//
// FSST (8-bit):
//   - Encoding: ~130-185 MB/s
//   - Decoding: ~1.1-1.4 GB/s with buffer reuse
//   - Table size: ~140KB in memory
//
// FSST12 (12-bit):
//   - Encoding: ~250 MB/s
//   - Decoding: ~500 MB/s
//   - Table size: ~550KB in memory
package fsst
