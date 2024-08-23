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

#ifndef LLVM_CLANG_PRECISION_PRECISIONDESCRIPTION
#define LLVM_CLANG_PRECISION_PRECISIONDESCRIPTION

include "clang/Precision/PrecisionDescriptionBase.td"

class IEEEFloatingPointDecoder<Value width, Value es> : Decoder<[width, es]> {
  defvar bias = ishr<sub<ishl<_1, es>, _1>, _1>;
  defvar mantissa = droph<__binary, add<es, _1>>;
  defvar biasedexponent = takeh<droph<__binary, _1>, es>;
  let sign = msb<__binary>;
  let zero = none<droph<__binary, _1>>;
  let poszero = none<__binary>;
  let negzero = and<sign, zero>;
  let inf = and<all<biasedexponent>, none<mantissa>>;
  let posinf = and<not<sign>, inf>;
  let neginf = and<sign, inf>;
  let nan = and<all<biasedexponent>, any<mantissa>>;
  let quietnan = and<all<biasedexponent>, msb<mantissa>>;
  let signalnan = and<and<all<biasedexponent>, not<msb<mantissa>>>, any<droph<mantissa, _1>>>;
  defvar unbiasedexponent = sub<biasedexponent, bias>;
  defvar subnormal = none<biasedexponent>;
  defvar numleadingzeros = clz<mantissa>;
  let significants = cond<subnormal, bshl<rext1<mantissa>, add<numleadingzeros, _1>>, lext1<mantissa>>;
  let exponent = cond<subnormal, sub<unbiasedexponent, numleadingzeros>, unbiasedexponent>;
}

class IEEEFloatingPointEncoder<Value width, Value es> : Encoder<[width, es]> {
  defvar bias = ishr<sub<ishl<_1, es>, _1>, _1>;
  defvar subnormalexponent = sub<_1, bias>;
  defvar normalexponent = add<__exponent, bias>;
  defvar subnormalshift = sub<_1, normalexponent>;
  defvar subnormalmantissa = bshr<dropl<__significants, _1>, subnormalshift>;
  defvar normalmantissa = droph<__significants, _1>;
  defvar subnormal = le<normalexponent, _0>;
  defvar biasedexponent = cond<subnormal, subnormalexponent, normalexponent>;
  defvar mantissa = cond<subnormal, subnormalmantissa, normalmantissa>;
  let binary = concat<__sign, concat<biasedexponent, mantissa>>;

  defvar fs = sub<width, add<es, _1>>;
  let poszero = zeros<width>;
  let negzero = lext1<zeros<sub<width, _1>>>;
  let zero = poszero;
  let posinf = lext0<concat<ones<es>, zeros<fs>>>;
  let neginf = lext1<concat<ones<es>, zeros<fs>>>;
  let inf = posinf;
  let signalnan = concat<ones<add<es, _1>>, lext0<ones<sub<fs, _1>>>>;
  let quietnan = ones<width>;
  let nan = quietnan;
}

class IEEEFloatingPointRounder<Value width, Value es> : Rounder<[width, es]> {
  defvar fs = sub<width, add<es, _1>>;
  defvar bias = ishr<sub<ishl<_1, es>, _1>, _1>;
  defvar normalexponent = add<__exponent, bias>;
  defvar subnormal = le<normalexponent, _0>;
  let precision = cond<subnormal, add<fs, normalexponent>, add<fs, _1>>;
}

// def PositType : PrecisionTypeDef<"Posit", [Width, ExpSize]> {
//   let width = (val Width);
// }

// Bitwidth of the binary representation
def BitSize : NamedValue<IntType, "BitSize", "n">;
// Bitwidth of the exponent field in the binary representation
def ExponentSize : NamedValue<IntType, "ExponentSize", "es">;

def IEEEFloatingPointType : PrecisionTypeDef<"FloatingPoint", [BitSize, ExponentSize]> {
  let keyword = "ieeefp";
  let mnemonic = "ieeefp";
  let bitsize = BitSize;
  let decoder = IEEEFloatingPointDecoder<BitSize, ExponentSize>;
  let encoder = IEEEFloatingPointEncoder<BitSize, ExponentSize>;
  let rounder = IEEEFloatingPointRounder<BitSize, ExponentSize>;
  // let extraASTTypeDecls = [{
  //   unsigned int getFractionSize() const { return getWidth() - 1 - getExpSize(); }
  // }];
}

#endif // LLVM_CLANG_PRECISION_PRECISIONDESCRIPTION
