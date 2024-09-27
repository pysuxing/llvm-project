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
  defvar bias = sub<pow2<sub<es, _1>>, _1>;
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
  defvar unbiasedexponent = sub<btoi<biasedexponent>, bias>;
  defvar subnormal = none<biasedexponent>;
  defvar numleadingzeros = clz<mantissa>;
  let significants = cond<subnormal, droph<mantissa, numleadingzeros>, lext1<mantissa>>;
  let exponent = cond<subnormal, sub<unbiasedexponent, numleadingzeros>, unbiasedexponent>;
}

class IEEEFloatingPointEncoder<Value width, Value es> : Encoder<[width, es]> {
  defvar bias = sub<pow2<sub<es, _1>>, _1>;
  defvar subnormalexponent = sub<_1, bias>;
  defvar normalexponent = add<__exponent, bias>;
  defvar subnormalshift = sub<_1, normalexponent>;
  defvar subnormalmantissa = shr<dropl<__significants, _1>, subnormalshift>;
  defvar normalmantissa = droph<__significants, _1>;
  defvar subnormal = le<normalexponent, _0>;
  defvar biasedexponent = itob<cond<subnormal, subnormalexponent, normalexponent>>;
  defvar mantissa = cond<subnormal, subnormalmantissa, normalmantissa>;
  let binary = concat<itob<__sign>, concat<biasedexponent, mantissa>>;

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
  defvar bias = sub<pow2<sub<es, _1>>, _1>;
  defvar normalexponent = add<__exponent, bias>;
  defvar subnormal = le<normalexponent, _0>;
  let precision = cond<subnormal, add<fs, normalexponent>, add<fs, _1>>;
}

class IEEEFloatingPointProperty<Value w, Value es> : Property<[w, es]> {
  let width = w;
  let precision = sub<width, es>;
  let maxsig = ones<precision>;
  let maxexp = sub<pow2<sub<es, _1>>, _2>;
  let minsig = ones<_1>;
  let minexp = sub<neg<maxexp>, sub<precision, _1>>;
}

def IEEEFloatingPointType : PrecisionTypeDef<"FloatingPoint", [__width, __expwidth]> {
  let keyword = "ieeefp";
  let mnemonic = "ieeefp";
  let decoder = IEEEFloatingPointDecoder<__width, __expwidth>;
  let encoder = IEEEFloatingPointEncoder<__width, __expwidth>;
  let rounder = IEEEFloatingPointRounder<__width, __expwidth>;
  let property = IEEEFloatingPointProperty<__width, __expwidth>;
  let assertions = [
    gt<__expwidth, _1>,
    lt<add<__expwidth, _1>, __width>,
    le<__width, __128>
  ];
}

#endif // LLVM_CLANG_PRECISION_PRECISIONDESCRIPTION
