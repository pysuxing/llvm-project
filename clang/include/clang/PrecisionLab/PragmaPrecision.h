//===--- PragmaPrecision.h - Parse Precision pragmas ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PRECISIONLAB_PRAGMAPRECISION_H
#define LLVM_CLANG_PRECISIONLAB_PRAGMAPRECISION_H

#include "llvm/ADT/StringSwitch.h"
#include "clang/Lex/Token.h"

namespace clang {

class Expr;
struct IdentifierLoc;

class PragmaPrecision {
public:
  enum CommandKind {
    kRegion,
    kRange,
    kAbsError,
    kRelError,
    kInvalid,
  };
  static bool isSupportedType(tok::TokenKind Kind) {
    // See Token::isSimpleTypeSpecifier
    switch (Kind) {
    case tok::kw_half:
    case tok::kw_float:
    case tok::kw_double:
    case tok::kw___bf16:
    case tok::kw__Float16:
    case tok::kw___float128:
    case tok::kw___ibm128:
    // PLABFIXME add user-customized types
      return true;
    default:
      return false;
    }
  }
  static CommandKind parseCommand(StringRef S) {
    return llvm::StringSwitch<CommandKind>(S)
        .Case("region", kRegion)
        .Case("range", kRange)
        .Case("abserror", kAbsError)
        .Case("relerror", kRelError)
        .Default(kInvalid);
  }
public:
  Token Pragma;
  Token Command;
  SmallVector<unsigned> Segments;
  SmallVector<Token, 8> Data;
};

} // end namespace clang

#endif // LLVM_CLANG_PRECISIONLAB_PRAGMAPRECISION_H