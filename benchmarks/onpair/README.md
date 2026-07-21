# FSST vs OnPair benchmarks

This nested module compares FSST, OnPair, and OnPair16 on identical sequences
of short strings. It keeps the local OnPair dependency out of FSST's production
module.

The suite measures:

- dictionary/model training;
- encoding with an already-trained model;
- bulk decoding;
- random-access row decoding;
- allocation counts and throughput;
- logical archive size.

`archive_B` is computed consistently as model/dictionary bytes, encoded payload
bytes, and one 32-bit offset per row plus the terminal offset. `comp_ratio` is
original bytes divided by `archive_B`, so larger values are better. This avoids
comparing OnPair's additional serialized-data compression with FSST's raw
encoded representation.

The text corpora are split into lines because OnPair is a field-level codec.
At most 5,000 rows are used from each corpus to keep the default suite practical.
Newline separators are not included in the input size for either codec.

Run from this directory:

```sh
go test -run='^$' -bench=BenchmarkCompare -benchmem -count=5
```

For a quick smoke run:

```sh
go test -run='^$' -bench=BenchmarkCompare -benchmem -benchtime=1x
```

The `go.mod` replacements expect sibling checkouts at `fsst/` and `onpair/`.
