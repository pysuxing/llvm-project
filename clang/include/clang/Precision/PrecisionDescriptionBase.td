//===-- PrecisionDescriptionBase.td - Base PDL definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Base definitions for PDL
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PRECISION_PRECISIONDESCRIPTIONBASE
#define LLVM_CLANG_PRECISION_PRECISIONDESCRIPTIONBASE

include "clang/Precision/PrecisionDescriptionLanguage.td"

// The Decoder describes how to decode a binary representation into a special
// value, e.g., 0, infinity or NaN., or the generic scientific notation form,
// i.e., a triple of (sign, significants, exponent).
// 
// It has one mandatory parameter, binary, and can take extra parameters to
// facilatate the decoding procedure, such as bitwidth of the type.
class Decoder<list<Value> params = []> {
  list<Value> parameters = params;

  // Each field below is defined as an PDL value expression, formed by 
  // (1) referencing predefined named value '__binary' and values in 'parameters', and
  // (2) applying PDL operators to other values.

  // [IntType] The sign, 0 for positive and 1 for negtive
  Value sign = ?;
  // [IntType] The exponent
  Value exponent = ?;
  // [BinType] The significants. The highest bit must be 1, thus having the form
  // of '0b1xxxx', interpreted as real number 1.xxxx. Any implicit bits in the
  // binary representation (like IEEE floating point format) should be made
  // explicit in significants.
  Value significants = ?;

  // Below are decoders for special values. If the type does not define a special
  // value, leave the corresponding field as '?'. For instance, the Posit type 
  // has no representation for infinity, then the 'posinf', 'neginf' and 'inf'
  // fields are all set to '?'.

  // [IntType] 1 if 'binary' represents +0
  Value poszero = ?;
  // [IntType] 1 if 'binary' represents -0
  Value negzero = ?;
  // [IntType] 1 if 'binary' represents +0/-0
  Value zero = ?;
  // [IntType] 1 if 'binary' represents +inf
  Value posinf = ?;
  // [IntType] 1 if 'binary' represents -inf
  Value neginf = ?;
  // [IntType] 1 if 'binary' represents +inf/-inf
  Value inf = ?;
  // [IntType] 1 if 'binary' represents signaling NaN
  Value signalnan = ?;
  // [IntType] 1 if 'binary' represents quiet NaN
  Value quietnan = ?;
  // [IntType] 1 if 'binary' represents any kind of NaN
  Value nan = ?;
}

// The Encoder describes how to encode a special value, e.g., 0, infinity or NaN.,
// or the generic scientific notation form, i.e., a triple of (sign, significants,
// exponent), into the binary representation.
//
// It has three mandatory parameters, (sign, significants, exponent), and can
// take extra parameters to facilatate the encoding procedure, such as bitwidth
// of the type.
//
// If the input value is not a special value, the (sign, significants, exponent)
// triple represents a value after rounding. That is, any non-special input value
// is guaranteed to fit in the binary format, without exceeding the precision
// limit or exponent range.
class Encoder<list<Value> params = []> {
  list<Value> parameters = params;

  // Each field below is defined as an PDL value expression, formed by 
  // (1) referencing predefined named values '__sign', '__exponent', '__significants'
  //     and values in 'parameters', and
  // (2) applying PDL operators to other values.

  // [BinType] The binary representation of the given generic scientifc notation
  Value binary = ?;

  // Below are encoders for special values. If the type does not define a special
  // value, leave the corresponding field as '?'. For instance, the posit type
  // has no representation for infinity, then the 'posinf', 'neginf' and 'inf'
  // fields are all set to '?'.

  // [BinType] binary representation of +0
  Value poszero = ?;
  // [BinType] binary representation of -0
  Value negzero = ?;
  // [BinType] binary representation of +0/-0
  Value zero = ?;
  // [BinType] binary representation of +inf
  Value posinf = ?;
  // [BinType] binary representation of -inf
  Value neginf = ?;
  // [BinType] binary representation of +inf/-inf
  Value inf = ?;
  // [BinType] binary representation of signaling NaN
  Value signalnan = ?;
  // [BinType] binary representation of quiet NaN
  Value quietnan = ?;
  // [BinType] binary representation of any kind of NaN
  Value nan = ?;
}

// The Rounder describes how many significants can be encoded into the binary representation.
// This is needed because for some type formats, the number of significants is value-dependent,
// e.g.,
// (1) For IEEE floating point types, subnormal numbers has fewer significants than normal numbers.
// (2) For Posit types, values around +1/-1 has longer fraction field than smaller or larger values.
//
// The input value is guaranteed to be a non-special value (sign, significants, exponent).
class Rounder<list<Value> params = []> {
  list<Value> parameters = params;

  // Each field below is defined as an PDL value expression, formed by 
  // (1) referencing predefined named values '__sign', '__exponent', '__significants'
  //     and values in 'parameters', and
  // (2) applying PDL operators to other values.

  // [IntType] computes the number of significants after rounding to fit into the binary
  // representation, including any implicit bits. For example, IEEE single format has a
  // 8-bit mantissa field, and for a normal number which has an implicit leading bit 1,
  // the number of significants is 8+1=9.
  Value precision = ?;
}

// The Property class describes static properties of the type. A property is considered
// static if it is value-independent.
class Property<list<Value> params = []> {
  list<Value> parameters = params;
  // [IntType] The bit-width of the type's binary representation. For non-parametric types,
  // it is a literal value, e.g., Literal<32> for IEEE single. For parametric types,
  // it must be the named value '__width'.
  Value width = ?;
  // [IntType] The maximum number of significants, including any implicit bits. This differs
  // to Rounder.precision that Rounder.precision is a value-dependent function dependent on
  // __sign, __exponent and __significants.
  Value precision = ?
  // [IntType] The minimum positive value this type can represent, in the form of
  // minpos = minsig * 2^minexp
  Value minsig = ?;
  Value minexp = ?;
  // [IntType] The maximum positive value this type can represent, in the form of
  // maxpos = maxsig * 2^maxexp
  Value maxsig = ?;
  Value maxexp = ?;

  // We do not define negtive counterparts of minpos/maxpos, and considers the following
  // equations to be valid,
  //   minneg = -maxpos
  //   maxneg = -minpos
  // that is, we only support types whose reprentation range is symmetry about 0.

  // The size (in bytes), defaulted to (width+7) / 8
  // Value size = ?;
  // The alignment (in bytes), defaulted to size. Should be a power of 2.
  // Value alignment = size;
}

class PrecisionTypeDef<string n, list<NamedValue> params = []> {
  // Name of the type, generally in title case, e.g., "MyFloat"
  string name = n;
  // C/C++ keyword used for this type, e.g. "myfloat"
  string keyword = !tolower(name);
  // MLIR type mnemonic,
  string mnemonic = !tolower(name);
  // Description of this type, used as auto-generated documentation
  string description = [{}];
  // Type parameters
  list<NamedValue> parameters = params;
  // The decoder, encoder, rounder and property, see above
  Decoder decoder = ?;
  Encoder encoder = ?;
  Rounder rounder = ?;
  Property property = ?;
  // Assertions on 'width', 'size', 'alignment', and 'parameters'
  list<Value> assertions = [];

  // see CXXNameMangler::mangleType(const BuiltinType *T) in ItaniumMangle.cpp
  string cppManglePrefix = ?;
}

// class InfinityKind;
// def IEEEInfinity : InfinityKind;
// def NoInfinity : InfinityKind;

// class NaNKind;
// def IEEENaN : NaNKind;
// def AllOnesNaN : NaNKind;
// def NegtiveZeroNaN : NaNKind;
// def NoNaN : NaNKind;

// class SubNormalKind;
// def IEEESubNormal : SubNormalKind;
// def NoSubNormal : SubNormalKind;

  // InfinityKind infinityKind;
  // NaNKind nanKind;
  // SubNormalKind  subNormalKind;
  // // Only used for IEEE-like floating point types
  // // Default to (2^es-1)/2, e.g., 127 for fp32(es=8)
  // dag expBias = ?;
  // // NoSubNormal   -> minExp = 0 - expBias
  // // IEEESubNormal -> minExp = 0 - expBias + 1
  // dag minExp = ?
  // // IEEENan       -> maxExp = 2^es - expBias - 1
  // // else          -> maxExp = 2^es - expBias
  // dag maxExp = ?
#endif // LLVM_CLANG_PRECISION_PRECISIONDESCRIPTIONBASE
