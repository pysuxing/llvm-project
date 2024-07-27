//===--- Precision.h - Types for Precision ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_PRECISION_H
#define LLVM_CLANG_PARSE_PRECISION_H

#include "clang/Lex/Token.h"

namespace clang {

class Expr;
struct IdentifierLoc;

class PragmaPrecisionInfo {
public:
  using VecTy = SmallVector<Token, 2>;
  using SpecTy = std::pair<VecTy, VecTy>;
  Token Pragma;
  SmallVector<SpecTy> TuneSpecs;
};

struct PragmaPrecisionSpec {
  using VecTy = SmallVector<IdentifierLoc *, 2>;
  using SpecTy = std::pair<VecTy, VecTy>;
  // Source range of the directive.
  // SourceRange Range;
  // Identifier corresponding to the name of the pragma. "precision" for
  // "#pragma precision" directives
  IdentifierLoc *PragmaNameLoc = nullptr;
  // Tuning specs in form of "(var1, var2, ...)[type1, type2, ...]"
  SmallVector<SpecTy> TuneSpecs;

  PragmaPrecisionSpec() = default;
};

} // end namespace clang

#endif // LLVM_CLANG_PARSE_PRECISION_H
