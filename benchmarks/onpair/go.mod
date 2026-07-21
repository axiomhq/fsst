module github.com/axiomhq/fsst/benchmarks/onpair

go 1.24.5

require (
	github.com/axiomhq/fsst v0.0.0
	github.com/seiflotfy/onpair v0.0.0
)

replace github.com/axiomhq/fsst => ../..

replace github.com/seiflotfy/onpair => ../../../onpair
