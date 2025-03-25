//===-- PrecisionDialect.td - Precision dialect ------------*- tablegen -*-===//
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

#ifndef LLVM_CLANG_PRECISIONLAB_PRECISIONDIALECT
#define LLVM_CLANG_PRECISIONLAB_PRECISIONDIALECT

include "mlir/IR/OpBase.td"

def Precision_Dialect : Dialect {
  let name = "precision";
  let summary = "A high-level dialect for custom precision types";
  let description = [{}];
  let dependentDialects = [];
  let cppNamespace = "::plab";
  // let hasConstantMaterializer = 0;
  // let hasNonDefaultDestructor = 0;
  // let hasOperationAttrVerify = 0;
  // let hasRegionArgAttrVerify = 0;
  // let hasRegionResultAttrVerify = 0;
  // let hasOperationInterfaceFallback = 0;
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  // let hasCanonicalizer = 0;
  // let isExtensible = 0;
  // let usePropertiesForAttributes = 1;
  let extraClassDeclaration = [{
    // Attributes for #pragma precision ...
    static llvm::StringRef getTargetRegionAttrName() { return "precision.region"; }
    static llvm::StringRef getTypeRangeAttrName() { return "precision.range"; }
    static llvm::StringRef getAbsoluteErrorAttrName() { return "precision.abserror"; }
    static llvm::StringRef getRelativeErrorAttrName() { return "precision.relerror"; }
  }];
}

#endif // LLVM_CLANG_PRECISIONLAB_PRECISIONDIALECT