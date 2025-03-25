//===- PrecisionTypes.td - Precision dialect types ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Precision dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PRECISIONLAB_PRECISIONTYPES
#define LLVM_CLANG_PRECISIONLAB_PRECISIONTYPES

include "clang/PrecisionLab/PrecisionDialect.td"
// include "clang/Precision/PrecisionTypeInterface.td"
include "mlir/IR/AttrTypeBase.td"

class Precision_Type<string name, string typeMnemonic, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Type">
  : TypeDef<Precision_Dialect, name, traits, baseCppClass> {
  // : TypeDef<Precision_Dialect, name, traits # [DeclareTypeInterfaceMethods<PrecisionTypeInterface>]> {
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

// def Precision_GenericType : Precision_Type<"Generic", "generic"> {
//   let description = "The generic real type using scientific notation, i.e., the (sign, significants, exponent) triple.";
//   let parameters = (ins "int":$precision);
//   let assemblyFormat = "`<` $precision `>`";
// }

// include "clang/Precision/PrecisionMLIRTypes.inc"

#endif LLVM_CLANG_PRECISIONLAB_PRECISIONTYPES