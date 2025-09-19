# FSST
FSST: Fast Random Access String Compression

```sh
go get github.com/axiomhq/fsst
```

```go
tbl := fsst.Train(inputs)
compressed := tbl.Encode(data)
original := tbl.Decode(compressed)
```

Serialize:

```go
tbl.WriteTo(w)
tbl.ReadFrom(r)
```

Best on repetitive text. 1.5-3x typical. No-allocation decode.

https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf
