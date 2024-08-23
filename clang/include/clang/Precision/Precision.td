//===-- PrecisionOps.td - Precision dialect definition ------*- tablegen -*-===//
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

#ifndef LLVM_CLANG_PRECISION_PRECISION
#define LLVM_CLANG_PRECISION_PRECISION

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "clang/Precision/PrecisionTypeInterface.td"

def Precision_Dialect : Dialect {
  let name = "precision";
  let summary = "A high-level dialect for custom precision real types";
  let description = [{}];
  let dependentDialects = [];
  let cppNamespace = "::mlir::precision";

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
  // let extraClassDeclaration = [{}];
}

class Precision_Type<string name, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<Precision_Dialect, name, traits # [DeclareTypeInterfaceMethods<PrecisionTypeInterface>]> {
  // let cppBaseClassName = "::mlir::Type";
  // let description = "";
  // let storageClass = name # "TypeStorage";
  // let storageNamespace = "detail";
  // let genStorageClass = 1;
  // let hasStorageCustomConstructor = 0;
  // let parameters = (ins);
  // let builders = ?;
  // let traits = traits;
  let mnemonic = typeMnemonic;
  // let assemblyFormat = ?;
  // let hasCustomAssemblyFormat = 0;
  // let genAccessors = 1;
  // let skipDefaultBuilders = 0;
  // let genVerifyDecl = 0;
  // let extraClassDeclaration = [{}];
  // let extraClassDefinition = [{}];
  // let cppType = dialect.cppNamespace # "::" # cppClassName;
  // let typeName = dialect.name # "." # mnemonic;
}

def Precision_GenericType : Precision_Type<"Generic", "generic"> {
  let description = "The generic real type using scientific notation, i.e., the (sign, significants, exponent) triple.";
  let parameters = (ins "int":$precision);
  let assemblyFormat = "`<` $precision `>`";
}

include "clang/Precision/PrecisionMLIRTypes.inc"

#endif // LLVM_CLANG_PRECISION_PRECISION