//===-- Precision.td - Base of precision definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Base definitions for precision difinitions
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PRECISION_PRECISIONBASE
#define LLVM_CLANG_PRECISION_PRECISIONBASE

class Value;

def Binary : Value;
def Sign : Value;
def Exponent : Value;
def Significants : Value;

class TypeParameter<string n, string abbr = n> : Value {
  string name = n;
  string abbrivation = abbr;
}

class Operator;

def val : Operator;
def neg : Operator;
def add : Operator;
def sub : Operator;
def msb : Operator;
def lsb : Operator;
def tst : Operator;
def shl : Operator;
def shr : Operator;
def clb : Operator;
def clo : Operator;
def cto : Operator;
def clz : Operator;
def ctz : Operator;
def clr : Operator;
def cond : Operator;


class PrecisionType<string n, list<TypeParameter> params = []> {
  string name = n;
  string cppClassName = n # "Type";
  string keyword = !tolower(n);
  // see CXXNameMangler::mangleType(const BuiltinType *T) in ItaniumMangle.cpp
  string cppManglePrefix = ?;
  list<TypeParameter> parameters = params;
  code extraASTTypeDecls = [{}];

  dag width = ?;
  dag size = ?;
  dag align = ?;

  dag decoder = ?;
  dag encoder = ?;
}

#endif // LLVM_CLANG_PRECISION_PRECISIONBASE
