//===-- PrecisionDescriptionLanguage.td - The PDL language -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The Precision Description Language (PDL)
///
/// This is a mini-language for describing custom precision types and their
/// associating arithmetic operations.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PRECISION_PRECISIONDESCRIPTIONLANGUAGE
#define LLVM_CLANG_PRECISION_PRECISIONDESCRIPTIONLANGUAGE

//===----------------------------------------------------------------------===//
//
// 1. Type system
// 
// There are only two types, IntType and BinType, and BinType is a special kind of
// IntType with fixed bit width, forming the type hierachy below
//
//   IntType <- BinType
//
// BinType is used to represent (part of) the binary representation of cutom precision types.
// IntType is used to represent number of bits (lenth of BinType) or boolean value.
// The IntType is signed, so negtive values can be represented.
//
// BinType values and IntType values can be explictly converted to each other.
// 
//===----------------------------------------------------------------------===//

class Type;

def IntType : Type;
def BinType : Type;

//===----------------------------------------------------------------------===//
//
// 2. Values
// 
// There are 3 kinds of values.
// 
// Literals
//   Integer literal values can be created with 'Literal<2>'.
//   Binary literal is not supported yet. Literals for 0 and 1, '_0' and '_1',
//   are predefined for convinience.
// Named Values
//   These are values which haved a name. Several predefined named values exists
//   to facilatate type definition, see below. User defined Named values are
//   generally used as type parameters while defining parametric types, see
//   Precision.td for examples. Users are free to define other named values.
// Operators
//   Applying PDL operators to PDL values creates new PDL values,
//   e.g., add<x, y> creates a value equals to "x+y". See the Operator class and
//   its derived subclasses below.
//
//===----------------------------------------------------------------------===//

class Value<Type ty> {
  Type type = ty;
}

class Literal<int v> : Value<IntType> {
  int value = v;
}

// Integer literals 0-128 are predefined
// NOTE sync this with 'SmallIntegerLimit' in PrecisionDescriptionLanguage.h
foreach i = 0 ... 128 in {
  def _#i : Literal<i>;
}

class NamedValue<Type ty, string n, string abbr = ""> : Value<ty> {
  string name = n;
  string abbreviation = abbr;
}

// Predefined named values. All predefined named values starts with two underscores "__",
// which is reserved for predefined named values. User-defined named values should avoid
// this prefix.
//
// The four predefined values below are merely used to define encoders, decoers and rounders
// for custom precision types, see PrecisionBase.td.
// __binary       - Binary representation of a custom precision value
// __sign         - Sign in the scientifc notation triple (sign, exponent, significants)
// __exponent     - Exponent in the scientifc notation triple (sign, exponent, significants)
// __significants - Significants in the scientifc notation triple (sign, exponent, significants)
def __binary : NamedValue<BinType, "binary", "">;
def __sign : NamedValue<IntType, "sign", "">;
def __exponent : NamedValue<IntType, "exponent", "">;
def __significants : NamedValue<BinType, "significants", "">;
// The four predefined values below are meant to be used as type parameters in parametric
// precision types.
// __width        - Bit-width of a custom precision type
// __size         - Size (in bytes) of a custom precision type
// __expwidth     - Bit-width of the exponent field
// __fracwidth    - Bit-width of the fraction field
def __width : NamedValue<IntType, "Width", "n">;
def __size : NamedValue<IntType, "Size", "s">;
def __expwidth : NamedValue<IntType, "ExponentWidth", "es">;
def __fracwidth : NamedValue<IntType, "FractionWidth", "fs">;

//===----------------------------------------------------------------------===//
//
// 3. Operators
//
// A bunch of builtin operators on PDL values are supported. These are 
// mainly used to define the fields of customized precision types, e.g.,
// Encoder and Decoder. See PrecisionBase.td for detail.
//
// Operators can have variadic input values. If an operator is variadic, all of
// its input values are constrained to have the same type.
//
// All operators are single valued.
//
// All operators create new values and do not change any existing values.
// In other words, the PDL language is a pure functional language.
//
//===----------------------------------------------------------------------===//

class Operator<Type ty, list<Type> paramTypes, list<Value> args> : Value<ty> {
  // Parameter types. If the operator is variadic, the list contains only one element
  list<Type> parametersTypes = paramTypes;
  // Argument values
  list<Value> arguments = args;
}
class UnaryOperator<Type ty, Type paramType, Value arg>
  : Operator<ty, [paramType], [arg]> {
  assert !eq(paramType, arg.type), "Invalid argument " # !repr(arg);
}
class BinaryOperator<Type ty, Type lhsType, Type rhsType, Value lhsValue, Value rhsValue>
  : Operator<ty, [lhsType, rhsType], [lhsValue, rhsValue]> {
  assert !eq(lhsType, lhsValue.type), "Invalid argument " # !repr(lhsValue);
  assert !eq(rhsType, rhsValue.type), "Invalid argument " # !repr(rhsValue);
}
class TernaryOperator<Type ty, Type ty0, Type ty1, Type ty2, Value arg0, Value arg1, Value arg2>
  : Operator<ty, [ty0, ty1, ty2], [arg0, arg1, arg2]> {
  assert !eq(ty0, arg0.type), "Invalid argument type";
  assert !eq(ty1, arg1.type), "Invalid argument type";
  assert !eq(ty2, arg2.type), "Invalid argument type";
}

// Conversion operations
//
// btoi - convert binary to integer
// itob - convert integer to binary
class btoi<Value b> : UnaryOperator<IntType, BinType, b>;
class itob<Value i> : UnaryOperator<BinType, IntType, i>;

// Constant operations
//
// ones  - creates an fixed-length binary value with all bits set to 1
// zeros - creates an fixed-length binary value with all bits set to 0
class ones<Value n> : UnaryOperator<BinType, IntType, n>;
class zeros<Value n> : UnaryOperator<BinType, IntType, n>;

// Binary inspect operations
//
// none - true if all bits are 0
// all  - true if all bits are 1
// any  - true if 1 or more bits are 1
class none<Value bs> : UnaryOperator<IntType, BinType, bs>;
class all<Value bs> : UnaryOperator<IntType, BinType, bs>;
class any<Value bs> : UnaryOperator<IntType, BinType, bs>;

// Binary manipulate operations
//
// msb    - extract the most-significant-bit (highest)
// lsb    - extract the least-significant-bit (lowest)
// takel  - extract the lowest n bits
// takeh  - extract the highest n bits
// dropl  - drop the lowest n bits and return the remained bits
// droph  - drop the highest n bits and return the remained bits
// shl    - left shift the bits, keeping the bit length unchanged
// shr    - right shift the bits, keeping the bit length unchanged
// lext0  - left extend a 0-bit to the given binary
// rext0  - right extend a 0-bit to the given binary
// lext1  - left extend a 1-bit to the given binary
// rext1  - right extend a 1-bit to the given binary
// concat - concatenate two binary, e.g. (concat 0b01, 0b11) returns 0b0111
class msb<Value bs> : UnaryOperator<IntType, BinType, bs>;
class lsb<Value bs> : UnaryOperator<IntType, BinType, bs>;
class takel<Value bs, Value n> : BinaryOperator<BinType, BinType, IntType, bs, n>;
class takeh<Value bs, Value n> : BinaryOperator<BinType, BinType, IntType, bs, n>;
class dropl<Value bs, Value n> : BinaryOperator<BinType, BinType, IntType, bs, n>;
class droph<Value bs, Value n> : BinaryOperator<BinType, BinType, IntType, bs, n>;
class shl<Value bs, Value shift> : BinaryOperator<BinType, BinType, IntType, bs, shift>;
class shr<Value bs, Value shift> : BinaryOperator<BinType, BinType, IntType, bs, shift>;
class lext0<Value bs> : UnaryOperator<BinType, BinType, bs>;
class rext0<Value bs> : UnaryOperator<BinType, BinType, bs>;
class lext1<Value bs> : UnaryOperator<BinType, BinType, bs>;
class rext1<Value bs> : UnaryOperator<BinType, BinType, bs>;
class concat<Value lhs, Value rhs> : BinaryOperator<BinType, BinType, BinType, lhs, rhs>;

// Binary bit counting operations
//
// clz - count leading zeros
// ctz - count trailing zeros
// clo - count leading ones
// cto - count trailing ones
// clb - count leading bit, e.g., given 1-bit value b, (clb bits, b) counts the number of leading bits of value b
// ctb - count leading bit, e.g., given 1-bit value b, (ctb bits, b) counts the number of trailing bits of value b
class clz<Value bs> : UnaryOperator<IntType, BinType, bs>;
class ctz<Value bs> : UnaryOperator<IntType, BinType, bs>;
class clo<Value bs> : UnaryOperator<IntType, BinType, bs>;
class cto<Value bs> : UnaryOperator<IntType, BinType, bs>;
class clb<Value bs, Value b> : BinaryOperator<IntType, BinType, BinType, bs, b>;
class ctb<Value bs, Value b> : BinaryOperator<IntType, BinType, BinType, bs, b>;

// Integer arithmetic Operators
//
// add  - addition
// sub  - subtraction
// neg  - negation
// pow2 - power of 2, i.e., pow2(e) = (1 << e)
// scl2 - scale by a power of 2, i.e., scl2(v, f) = (v << f), in which 'f' must be a literal, and can be negtive
class add<Value lhs, Value rhs> : BinaryOperator<IntType, IntType, IntType, lhs, rhs>;
class sub<Value lhs, Value rhs> : BinaryOperator<IntType, IntType, IntType, lhs, rhs>;
class neg<Value v> : UnaryOperator<IntType, IntType, v>;
class pow2<Value e> : UnaryOperator<IntType, IntType, e>;
class scl2<Value v, Value f> : BinaryOperator<IntType, IntType, IntType, v, f>;

// Comparison operators
// 
// eq - equal
// ne - nonequal
// lt - less than
// le - less than or equal
// gt - greater than
// ge - greater than or equal
class eq<Value lhs, Value rhs> : BinaryOperator<IntType, IntType, IntType, lhs, rhs>;
class ne<Value lhs, Value rhs> : BinaryOperator<IntType, IntType, IntType, lhs, rhs>;
class lt<Value lhs, Value rhs> : BinaryOperator<IntType, IntType, IntType, lhs, rhs>;
class le<Value lhs, Value rhs> : BinaryOperator<IntType, IntType, IntType, lhs, rhs>;
class gt<Value lhs, Value rhs> : BinaryOperator<IntType, IntType, IntType, lhs, rhs>;
class ge<Value lhs, Value rhs> : BinaryOperator<IntType, IntType, IntType, lhs, rhs>;

// Logical operators
// not - logical not
// and - logical and
// or  - logical or
class not<Value v> : UnaryOperator<IntType, IntType, v>;
class and<Value lhs, Value rhs> : BinaryOperator<IntType, IntType, IntType, lhs, rhs>;
class or<Value lhs, Value rhs>  : BinaryOperator<IntType, IntType, IntType, lhs, rhs>;

// Conditional operator
// (cond b, t, f) behaves the same as the C tenary operator (b? t : f)
class cond<Value b, Value t, Value f> : TernaryOperator<t.type, IntType, t.type, f.type, b, t, f> {
  assert !eq(t.type, f.type), "Types of t and f do not match";
}

#endif // LLVM_CLANG_PRECISION_PRECISIONDESCRIPTIONLANGUAGE
