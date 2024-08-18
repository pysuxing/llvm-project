//===-- Precision.td - Custom precision types --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Custom precision types
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PRECISION_POSIT
#define LLVM_CLANG_PRECISION_POSIT

include "clang/Precision/PrecisionBase.td"

def Width : TypeParameter<"Width", "n">;
def ExpSize : TypeParameter<"ExpSize", "es">;

def PositType : PrecisionType<"Posit", [Width, ExpSize]> {
  let width = (val Width);
  // defvar n = Width;
  // defvar es = ExpSize;
  // defvar sign = (msb Binary);
  // defvar binary = (cond sign, (neg Binary), Binary)
  // defvar binary1 = (shl binary, 1);
  // defvar k = (clb binary1, (msb binary1));
  // defvar binary2 = (shl binary1, (add k, 1));
  // defvar e = (msb binary2, es);
  // defvar binary3 = (shl binary2, es)
  // defvar fs = (sub n, (add 1, (add es, (add k, 1))))
  // defvar fraction = (cond (gt fs, 0), (msb binary3, fs), 0)
  // defvar r = (cond sign, (sub 0, k), (sub k, 1))

  // let width = (val n);
}

def FloatingPointType : PrecisionType<"FloatingPoint", [Width, ExpSize]> {
  let keyword = "ieeefp";
  let width = (val Width);
  let extraASTTypeDecls = [{
    unsigned int getFractionSize() const { return getWidth() - 1 - getExpSize(); }
  }];
}

#endif // LLVM_CLANG_PRECISION_POSIT
