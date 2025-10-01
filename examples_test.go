package fsst

import (
	"fmt"
)

func Example() {
	inputs := [][]byte{
		[]byte("hello world"),
		[]byte("hello there"),
	}
	tbl := Train(inputs)
	for _, input := range inputs {
		comp := tbl.Encode(input)
		orig := tbl.DecodeAll(comp)
		fmt.Println(string(orig))
	}
	// Output:
	// hello world
	// hello there
}
