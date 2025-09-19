package fsst

import "testing"

func TestSymbolBasics(t *testing.T) {
	s := newSymbolFromByte('A', 123)
	if s.length() != 1 {
		t.Fatalf("length=1 got %d", s.length())
	}
	if s.first() != 'A' {
		t.Fatalf("first byte mismatch")
	}
	if s.code() != 123 {
		t.Fatalf("code mismatch")
	}
	if s.ignoredBits() != 56 {
		t.Fatalf("ignoredBits mismatch")
	}

	b := []byte("ABCDEFGH")
	s2 := newSymbolFromBytes(b)
	if s2.length() != 8 {
		t.Fatalf("len=8 got %d", s2.length())
	}
	if s2.first() != 'A' || s2.first2() != uint16('A')|(uint16('B')<<8) {
		t.Fatalf("first/first2 mismatch")
	}

	// setCodeLen
	s2.setCodeLen(42, 3)
	if s2.code() != 42 || s2.length() != 3 {
		t.Fatalf("setCodeLen failed")
	}

	// fsstConcat truncation
	a := newSymbolFromBytes([]byte("abcd"))
	b2 := newSymbolFromBytes([]byte("WXYZ"))
	c := fsstConcat(a, b2)
	if c.length() != 8 {
		t.Fatalf("fsstConcat length=%d", c.length())
	}
	if c.first() != 'a' {
		t.Fatalf("fsstConcat content")
	}
}
