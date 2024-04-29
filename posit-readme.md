# LLVM for Posit

## Build

Build with cmake, adjust the cmake variables if needed.

```bash
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local/posit \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLLVM_BUILD_UTILS=ON \
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_INSTALL_GTEST=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -S llvm -B build-posit
cmake --build build-posit
```

Posit support is supported under a new AArch64 subtarget named **Xiangjiang**,
use option `-mcpu=xiangjiang` while calling `clang` and `llc` to select the right target.

## Posit Extension to C

In C language level, posit types are added as new primitive types,
including `__posit16`, `__posit32`, `__posit64`, `__posit16_1`, and `__posit32_3`.
Vector posit types are not supported in C level.

Supported operations are (1) arithmetic operations, `+`, `-`, `*`, `/`,
and (2) convertion from/to IEEE `float` and `double`. See `posit-demo/posit-demo.c` for an example.

To generate llvm IR, run

```bash
./build-posit/bin/clang -O2 -S emit-llvm -mcpu=xiangjiang posit-demo/posit-demo.c
```

## Posit Extension to LLVM IR

In LLVM IR level, both scalar and vector posit types are supported.
Fused multiply operations are supported in addition to those supported in C.
See `posit-demo/posit-vector-demo.ll` for an example.

To generate assembly, run

```bash
./build-posit/bin/llc -mcpu=xiangjiang posit-demo/posit-vector-demo.ll
```

Use the `clang` driver to call the assembler.

```bash
./build-posit/bin/clang -c -mcpu=xiangjiang posit-vector-demo.s
```
