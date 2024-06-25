//===- Precision.h - MLIR Precision IR Classes ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef CLANG_CIR_DIALECT_PRECISION_PRECISION_H
#define CLANG_CIR_DIALECT_PRECISION_PRECISION_H

#include "llvm/ADT/APInt.h"
#include "mlir/IR/Types.h"
// #include "mlir/IR/BuiltinTypes.h"
// #include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "clang/CIR/Dialect/Precision/PrecisionOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "clang/CIR/Dialect/Precision/PrecisionOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "clang/CIR/Dialect/Precision/PrecisionAttributes.h.inc"

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/Precision/PrecisionOps.h.inc"

#endif // CLANG_CIR_DIALECT_PRECISION_PRECISION_H