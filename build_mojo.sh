#!/bin/bash
# Build Mojo decoder as static library for CGo (arm64 + amd64 with full SIMD)

set -e

cd mojo

echo "Building FSST Mojo decoder for arm64 and amd64..."

# Detect OS
OS=$(uname -s)
case "$OS" in
    Darwin)
        # macOS: Build for current architecture (universal build requires cross-compile)
        ARCH=$(uname -m)
        echo "Building for macOS ($ARCH)..."

        if [ "$ARCH" = "arm64" ]; then
            # Build arm64 with Apple Silicon optimizations (NEON)
            magic run mojo build --emit object \
                --target-triple aarch64-apple-darwin \
                --target-cpu apple-m1 \
                --target-features +neon,+fp-armv8,+crypto \
                fsst_decoder.mojo -o libfsst_decoder.o

            # Static library (for CGo)
            ar rcs libfsst_decoder.a libfsst_decoder.o
            rm libfsst_decoder.o

            echo "Static library built: mojo/libfsst_decoder.a (arm64 with NEON)"
        else
            # Build x86_64 with AVX2/AVX-512
            magic run mojo build --emit object \
                --target-triple x86_64-apple-darwin \
                --target-cpu skylake-avx512 \
                --target-features +avx2,+fma,+bmi2,+avx512f,+avx512cd,+avx512bw,+avx512dq,+avx512vl \
                fsst_decoder.mojo -o libfsst_decoder.o

            # Static library (for CGo)
            ar rcs libfsst_decoder.a libfsst_decoder.o
            rm libfsst_decoder.o

            echo "Static library built: mojo/libfsst_decoder.a (x86_64 with AVX-512)"
        fi
        ;;

    Linux)
        # Linux: Build for current architecture with max SIMD
        ARCH=$(uname -m)
        echo "Building for Linux ($ARCH)..."

        if [ "$ARCH" = "aarch64" ]; then
            # ARM64 with NEON + SVE if available
            magic run mojo build --emit object \
                --target-triple aarch64-unknown-linux-gnu \
                --target-cpu neoverse-n1 \
                --target-features +neon,+fp-armv8,+crypto,+sve \
                fsst_decoder.mojo -o libfsst_decoder.o

            # Static library (for CGo)
            ar rcs libfsst_decoder.a libfsst_decoder.o
            rm libfsst_decoder.o

            echo "Static library built: mojo/libfsst_decoder.a (arm64 with NEON+SVE)"
        else
            # x86_64 with AVX2/AVX-512
            magic run mojo build --emit object \
                --target-triple x86_64-unknown-linux-gnu \
                --target-cpu skylake-avx512 \
                --target-features +avx2,+fma,+bmi2,+avx512f,+avx512cd,+avx512bw,+avx512dq,+avx512vl \
                fsst_decoder.mojo -o libfsst_decoder.o

            # Static library (for CGo)
            ar rcs libfsst_decoder.a libfsst_decoder.o
            rm libfsst_decoder.o

            echo "Static library built: mojo/libfsst_decoder.a (x86_64 with AVX-512)"
        fi
        ;;

    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

echo ""
echo "To use the Mojo SIMD decoder:"
echo "  Build with CGo enabled: CGO_ENABLED=1 go build"
echo "  The Go code uses mojo/libfsst_decoder.a (static library)"
