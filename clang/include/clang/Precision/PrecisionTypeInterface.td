//===- PrecisionTypeInterface.td - Precision Type Interface -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PRECISION_PRECISIONTYPEINTERFACE
#define LLVM_CLANG_PRECISION_PRECISIONTYPEINTERFACE

include "mlir/IR/OpBase.td"

def PrecisionTypeInterface : TypeInterface<"PrecisionTypeInterface"> {
  let description = [{
    Interface for all custom precision types.
  }];
  let cppNamespace = "::mlir::precision";

  let methods = [
    InterfaceMethod<[{
        Returns the bit width of this type.
      }],
      /*retTy=*/"int",
      /*methodName=*/"getBitWidth",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{}]
    >,
    InterfaceMethod<[{
        Return the minimum exponent this type can represent.
      }],
      /*retTy=*/"int",
      /*methodName=*/"getMinExponent",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{}]
    >,
    InterfaceMethod<[{
        Return the maximum exponent this type can represent.
      }],
      /*retTy=*/"int",
      /*methodName=*/"getMaxExponent",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{}]
    >,
    InterfaceMethod<[{
        Return if the precision of this type is value-independent.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"getValueIndependentPrecision",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{}]
    >,
    InterfaceMethod<[{
        Return the number of significants, including any implicit bit.
        If getValueIndependentPrecision() returns false, this function
        should the largest number of significants of all positive values.
      }],
      /*retTy=*/"int",
      /*methodName=*/"getPrecision",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{}]
    >,
  ];
}

#endif // LLVM_CLANG_PRECISION_PRECISIONTYPEINTERFACE
