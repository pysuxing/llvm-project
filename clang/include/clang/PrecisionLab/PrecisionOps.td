//===-- PrecisionOps.td - Precision dialect operations -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the Precision dialect
///
//===----------------------------------------------------------------------===//


#ifndef LLVM_CLANG_PRECISIONLAB_PRECISIONOPS
#define LLVM_CLANG_PRECISIONLAB_PRECISIONOPS

include "clang/PrecisionLab/PrecisionDialect.td"
include "clang/PrecisionLab/PrecisionTypes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/BuiltinTypes.td"

class Precision_Op<string mnemonic, list<Trait> traits = []> :
    Op<Precision_Dialect, mnemonic, traits>;

#endif LLVM_CLANG_PRECISIONLAB_PRECISIONOPS