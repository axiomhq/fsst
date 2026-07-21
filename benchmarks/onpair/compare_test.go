package onpairbench

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/axiomhq/fsst"
	"github.com/seiflotfy/onpair"
)

const (
	maxCorpusRows     = 5000
	maxCorpusRowBytes = 1 << 20
	maxCorpusBytes    = 64 << 20
)

var (
	sinkFSSTTable     *fsst.Table
	sinkOnPairModel   *onpair.Model
	sinkFSSTArchive   fsstArchive
	sinkOnPairArchive *onpair.Archive
	sinkBytes         []byte
	sinkInt           int
)

type dataset struct {
	name       string
	byteRows   [][]byte
	stringRows []string
	original   []byte
	totalSize  int
	maxRowLen  int
}

type fsstArchive struct {
	data    []byte
	offsets []int
}

type archiveSize struct {
	model   int
	payload int
	index   int
}

func (s archiveSize) total() int {
	return s.model + s.payload + s.index
}

func BenchmarkCompare(b *testing.B) {
	for _, ds := range benchmarkDatasets(b) {
		ds := ds
		b.Run(ds.name, func(b *testing.B) {
			if len(ds.byteRows) == 0 || ds.totalSize == 0 {
				b.Fatal("benchmark dataset must contain at least one non-empty row")
			}
			fsstTable := fsst.Train(ds.byteRows)
			fsstEncoded := encodeFSSTRows(fsstTable, ds.byteRows)
			onPairModel := mustTrainOnPair(b, ds.stringRows)
			onPairEncoded := mustEncodeOnPair(b, onPairModel, ds.stringRows)
			onPair16Model := mustTrainOnPair(b, ds.stringRows, onpair.WithMaxTokenLength(16))
			onPair16Encoded := mustEncodeOnPair(b, onPair16Model, ds.stringRows)

			verifyArchives(b, ds, fsstTable, fsstEncoded, onPairEncoded, onPair16Encoded)

			b.Run("train", func(b *testing.B) {
				benchmarkTrainFSST(b, ds)
				benchmarkTrainOnPair(b, ds, "OnPair")
				benchmarkTrainOnPair(b, ds, "OnPair16", onpair.WithMaxTokenLength(16))
			})

			b.Run("encode", func(b *testing.B) {
				benchmarkEncodeFSST(b, ds, fsstTable)
				benchmarkEncodeOnPair(b, ds, "OnPair", onPairModel, onPairEncoded)
				benchmarkEncodeOnPair(b, ds, "OnPair16", onPair16Model, onPair16Encoded)
			})

			b.Run("decode_all", func(b *testing.B) {
				benchmarkDecodeAllFSST(b, ds, fsstTable, fsstEncoded)
				benchmarkDecodeAllOnPair(b, ds, "OnPair", onPairEncoded)
				benchmarkDecodeAllOnPair(b, ds, "OnPair16", onPair16Encoded)
			})

			b.Run("decode_row", func(b *testing.B) {
				benchmarkDecodeRowFSST(b, ds, fsstTable, fsstEncoded)
				benchmarkDecodeRowOnPair(b, ds, "OnPair", onPairEncoded)
				benchmarkDecodeRowOnPair(b, ds, "OnPair16", onPair16Encoded)
			})
		})
	}
}

func benchmarkTrainFSST(b *testing.B, ds dataset) {
	b.Run("FSST", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(ds.totalSize))
		for b.Loop() {
			sinkFSSTTable = fsst.Train(ds.byteRows)
		}
	})
}

func benchmarkTrainOnPair(b *testing.B, ds dataset, name string, opts ...onpair.Option) {
	b.Run(name, func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(ds.totalSize))
		for b.Loop() {
			var err error
			sinkOnPairModel, err = onpair.TrainModel(ds.stringRows, opts...)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

func benchmarkEncodeFSST(b *testing.B, ds dataset, table *fsst.Table) {
	encoded := encodeFSSTRows(table, ds.byteRows)
	size := fsstArchiveSize(b, table, encoded)
	b.Run("FSST", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(ds.totalSize))
		for b.Loop() {
			sinkFSSTArchive = encodeFSSTRows(table, ds.byteRows)
		}
		reportArchiveSize(b, ds.totalSize, size)
	})
}

func benchmarkEncodeOnPair(b *testing.B, ds dataset, name string, model *onpair.Model, encoded *onpair.Archive) {
	size := onPairArchiveSize(encoded)
	b.Run(name, func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(ds.totalSize))
		for b.Loop() {
			var err error
			sinkOnPairArchive, err = model.Encode(ds.stringRows)
			if err != nil {
				b.Fatal(err)
			}
		}
		reportArchiveSize(b, ds.totalSize, size)
	})
}

func benchmarkDecodeAllFSST(b *testing.B, ds dataset, table *fsst.Table, encoded fsstArchive) {
	b.Run("FSST", func(b *testing.B) {
		dst := make([]byte, 0, ds.totalSize)
		dstOffsets := make([]int, 0, len(encoded.offsets))
		b.ReportAllocs()
		b.SetBytes(int64(ds.totalSize))
		for b.Loop() {
			var err error
			dst, dstOffsets, err = table.DecodeBatch(dst, dstOffsets, encoded.data, encoded.offsets)
			if err != nil {
				b.Fatal(err)
			}
		}
		sinkBytes = dst
		sinkInt = len(dstOffsets)
	})
}

func benchmarkDecodeAllOnPair(b *testing.B, ds dataset, name string, archive *onpair.Archive) {
	b.Run(name, func(b *testing.B) {
		dst := make([]byte, ds.totalSize)
		b.ReportAllocs()
		b.SetBytes(int64(ds.totalSize))
		for b.Loop() {
			var err error
			sinkInt, err = archive.DecompressAllChecked(dst)
			if err != nil {
				b.Fatal(err)
			}
		}
		sinkBytes = dst
	})
}

func benchmarkDecodeRowFSST(b *testing.B, ds dataset, table *fsst.Table, encoded fsstArchive) {
	b.Run("FSST", func(b *testing.B) {
		bufSize := ds.maxRowLen
		for i := range ds.byteRows {
			compressedLen := encoded.offsets[i+1] - encoded.offsets[i]
			bufSize = max(bufSize, compressedLen*4+8)
		}
		dst := make([]byte, 0, bufSize)
		row := 0
		b.ReportAllocs()
		b.SetBytes(int64(ds.totalSize / len(ds.byteRows)))
		for b.Loop() {
			start, end := encoded.offsets[row], encoded.offsets[row+1]
			dst = table.Decode(dst, encoded.data[start:end])
			row++
			if row == len(ds.byteRows) {
				row = 0
			}
		}
		sinkBytes = dst
	})
}

func benchmarkDecodeRowOnPair(b *testing.B, ds dataset, name string, archive *onpair.Archive) {
	b.Run(name, func(b *testing.B) {
		dst := make([]byte, ds.maxRowLen+16)
		row := 0
		b.ReportAllocs()
		b.SetBytes(int64(ds.totalSize / len(ds.byteRows)))
		for b.Loop() {
			var err error
			sinkInt, err = archive.DecompressString(row, dst)
			if err != nil {
				b.Fatal(err)
			}
			row++
			if row == len(ds.byteRows) {
				row = 0
			}
		}
		sinkBytes = dst
	})
}

func benchmarkDatasets(b *testing.B) []dataset {
	b.Helper()
	return []dataset{
		makeSyntheticDataset(4096),
		loadDataset(b, "logs_apache_2k.log"),
		loadDataset(b, "logs_hdfs_2k.log"),
		loadDataset(b, "en_shakespeare.txt"),
	}
}

func makeSyntheticDataset(rows int) dataset {
	values := make([][]byte, rows)
	for i := range values {
		values[i] = []byte(fmt.Sprintf(
			`{"timestamp":"2026-07-21T12:%02d:%02dZ","level":"info","service":"api-%02d","user_id":"user_%08d","message":"request completed"}`,
			i%60, (i/60)%60, i%32, i,
		))
	}
	return newDataset("synthetic_json_4k", values)
}

func loadDataset(b *testing.B, name string) dataset {
	b.Helper()
	path := filepath.Join("..", "..", "testdata", name)
	file, err := os.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	defer file.Close()

	rows := make([][]byte, 0, maxCorpusRows)
	totalSize := 0
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64<<10), maxCorpusRowBytes)
	for len(rows) < maxCorpusRows && scanner.Scan() {
		row := bytes.Clone(scanner.Bytes())
		if totalSize+len(row) > maxCorpusBytes {
			b.Fatalf("%s exceeds the %d-byte benchmark corpus limit", path, maxCorpusBytes)
		}
		rows = append(rows, row)
		totalSize += len(row)
	}
	if err := scanner.Err(); err != nil {
		b.Fatalf("scan %s: %v", path, err)
	}
	if len(rows) == 0 || totalSize == 0 {
		b.Fatalf("%s contains no non-empty benchmark data", path)
	}
	return newDataset(name, rows)
}

func newDataset(name string, rows [][]byte) dataset {
	ds := dataset{
		name:       name,
		byteRows:   rows,
		stringRows: make([]string, len(rows)),
	}
	for i, row := range rows {
		ds.stringRows[i] = string(row)
		ds.original = append(ds.original, row...)
		ds.totalSize += len(row)
		ds.maxRowLen = max(ds.maxRowLen, len(row))
	}
	return ds
}

func encodeFSSTRows(table *fsst.Table, rows [][]byte) fsstArchive {
	totalSize := 0
	maxRowLen := 0
	for _, row := range rows {
		totalSize += len(row)
		maxRowLen = max(maxRowLen, len(row))
	}
	archive := fsstArchive{
		data:    make([]byte, 0, totalSize),
		offsets: make([]int, len(rows)+1),
	}
	rowBuf := make([]byte, 0, 2*maxRowLen+8)
	for i, row := range rows {
		rowBuf = table.Encode(rowBuf, row)
		archive.data = append(archive.data, rowBuf...)
		archive.offsets[i+1] = len(archive.data)
	}
	return archive
}

func mustTrainOnPair(b *testing.B, rows []string, opts ...onpair.Option) *onpair.Model {
	b.Helper()
	model, err := onpair.TrainModel(rows, opts...)
	if err != nil {
		b.Fatal(err)
	}
	return model
}

func mustEncodeOnPair(b *testing.B, model *onpair.Model, rows []string) *onpair.Archive {
	b.Helper()
	archive, err := model.Encode(rows)
	if err != nil {
		b.Fatal(err)
	}
	return archive
}

func fsstArchiveSize(b *testing.B, table *fsst.Table, archive fsstArchive) archiveSize {
	b.Helper()
	model, err := table.MarshalBinary()
	if err != nil {
		b.Fatal(err)
	}
	return archiveSize{
		model:   len(model),
		payload: len(archive.data),
		index:   4 * len(archive.offsets),
	}
}

func onPairArchiveSize(archive *onpair.Archive) archiveSize {
	return archiveSize{
		model:   len(archive.Dictionary) + 4*len(archive.TokenBoundaries),
		payload: 2 * len(archive.CompressedData),
		index:   4 * len(archive.StringBoundaries),
	}
}

func reportArchiveSize(b *testing.B, original int, size archiveSize) {
	b.Helper()
	b.ReportMetric(float64(size.model), "model_B")
	b.ReportMetric(float64(size.payload), "payload_B")
	b.ReportMetric(float64(size.index), "index_B")
	b.ReportMetric(float64(size.total()), "archive_B")
	b.ReportMetric(float64(original)/float64(size.total()), "comp_ratio")
}

func verifyArchives(b *testing.B, ds dataset, table *fsst.Table, fsstEncoded fsstArchive, archives ...*onpair.Archive) {
	b.Helper()
	decoded, decodedOffsets, err := table.DecodeBatch(nil, nil, fsstEncoded.data, fsstEncoded.offsets)
	if err != nil {
		b.Fatalf("FSST DecodeBatch: %v", err)
	}
	if !bytes.Equal(decoded, ds.original) || len(decodedOffsets) != len(ds.byteRows)+1 {
		b.Fatal("FSST roundtrip mismatch")
	}
	for i, want := range ds.byteRows {
		if got := decoded[decodedOffsets[i]:decodedOffsets[i+1]]; !bytes.Equal(got, want) {
			b.Fatalf("FSST row %d roundtrip mismatch", i)
		}
	}
	for _, archive := range archives {
		out := make([]byte, ds.totalSize)
		n, err := archive.DecompressAllChecked(out)
		if err != nil {
			b.Fatalf("OnPair DecompressAllChecked: %v", err)
		}
		if n != ds.totalSize || !bytes.Equal(out[:n], ds.original) {
			b.Fatal("OnPair roundtrip mismatch")
		}
		rowBuf := make([]byte, ds.maxRowLen+16)
		for i, want := range ds.byteRows {
			n, err := archive.DecompressString(i, rowBuf)
			if err != nil {
				b.Fatalf("OnPair row %d: %v", i, err)
			}
			if !bytes.Equal(rowBuf[:n], want) {
				b.Fatalf("OnPair row %d roundtrip mismatch", i)
			}
		}
	}
}
