//===- Precision.h - MLIR Precision IR Classes ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef MLIR_DIALECT_PRECISION_IR_PRECISION_H
#define MLIR_DIALECT_PRECISION_IR_PRECISION_H

#include "llvm/ADT/APInt.h"
#include "mlir/IR/Types.h"
// #include "mlir/IR/BuiltinTypes.h"
// #include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Dialect.h"

#include "mlir/Dialect/Precision/IR/PrecisionOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Precision/IR/PrecisionOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Precision/IR/PrecisionAttributes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Precision/IR/PrecisionOps.h.inc"

#endif // MLIR_DIALECT_PRECISION_IR_PRECISION_H