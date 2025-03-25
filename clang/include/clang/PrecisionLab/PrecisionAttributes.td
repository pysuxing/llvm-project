//===- PrecisionAttrs.td - Precision dialect attributes ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Precision dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PRECISIONLAB_PRECISIONATTRIBUTES
#define LLVM_CLANG_PRECISIONLAB_PRECISIONATTRIBUTES

include "clang/PrecisionLab/PrecisionDialect.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/AttrTypeBase.td"

class Precision_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Precision_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif LLVM_CLANG_PRECISIONLAB_PRECISIONATTRIBUTES
