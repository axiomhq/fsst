//go:build !amd64 && !arm64

package fsst

func (t *Table) decodeIntoExactKernel(dst, src []byte) (int, error) {
	return t.decodeIntoExactSafe(dst, src)
}
