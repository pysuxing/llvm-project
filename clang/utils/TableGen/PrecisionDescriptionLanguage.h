//=== PrecisionDescriptionLanugage.h - The PDL language ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_UTILS_TABLEGEN_PRECISIONDESCRIPTIONLANGUAGE_H
#define CLANG_UTILS_TABLEGEN_PRECISIONDESCRIPTIONLANGUAGE_H

#include "clang/AST/Type.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Record.h"
#include <array>
#include <memory>
#include <utility>

namespace clang {
namespace tblgen {

// class PDLType {
// public:
//   enum Kind { KInteger, kBinary };

// private:
//   Kind kind;

// protected:
//   PDLType(Kind kind) : kind(kind) {}

// public:
//   Kind getKind() const { return kind; }
// }; // class PDLType

// class PDLIntegerType : public PDLType {
// public:
//   PDLIntegerType() : PDLType(KInteger) {}
// }; // class PDLIntegerType

// class PDLBinaryType : public PDLType {
// public:
//   PDLBinaryType() : PDLType(kBinary) {}
// }; // class PDLBinaryType

enum class PDLTypeKind { Integer, Binary };

enum class PDLOperatorKind {
  ones,
  zeros,
  none,
  all,
  any,
  msb,
  lsb,
  takel,
  takeh,
  dropl,
  droph,
  bshl,
  bshr,
  lext0,
  rext0,
  lext1,
  rext1,
  concat,
  clz,
  ctz,
  clo,
  cto,
  clb,
  ctb,
  add,
  sub,
  ishl,
  ishr,
  eq,
  ne,
  lt,
  le,
  gt,
  ge,
  lnot,
  land,
  lor,
  cond,
  NumOperators
}; // enum class PDLOperator

// forward declaration
class PDLContext;

class PDLValue {
public:
  enum Kind { Literal, Named, Operator };

protected:
  llvm::PointerIntPair<PDLContext *, 3, Kind> ctx;
  union {
    int value;
    llvm::Record *record;
  };

protected:
  explicit PDLValue(Kind kind, PDLContext *ctx, llvm::Record *record)
      : ctx(ctx, kind), record(record) {}
  explicit PDLValue(PDLContext *ctx, int value)
      : ctx(ctx, Literal), value(value) {}

public:
  static bool classOf(PDLValue *value) { return true; }

public:
  PDLContext *getContext() const { return ctx.getPointer(); }
  llvm::Record *getRecord() const { return record; }
  Kind getKind() const { return ctx.getInt(); }
  PDLTypeKind getType() const {
    if (getKind() == Literal)
      return PDLTypeKind::Integer;
    if (getKind() == Operator) {
    }
    auto *type = getRecord()->getValueAsDef("type");
    return type->getName() == "IntType" ? PDLTypeKind::Integer
                                        : PDLTypeKind::Binary;
  }
  // bool operator==(const PDLValue &other) const;
  // bool operator!=(const PDLValue &other) const { return not operator==(other); }
  bool isDynamic() const;
};

class PDLLiteral : public PDLValue {
  friend class PDLContext;

private:
  PDLLiteral(PDLContext *ctx, int value) : PDLValue(ctx, value) {}

public:
  static bool classOf(PDLValue *value) {
    return value->getKind() == Literal;
  }

public:
  int getValue() const {
    return getKind() == Literal ? value : getRecord()->getValueAsInt("value");
  }
};

class PDLNamedValue : public PDLValue {
  friend class PDLContext;

private:
  PDLNamedValue(PDLContext *ctx, llvm::Record *record)
      : PDLValue(Named, ctx, record) {}

public:
  static bool classOf(PDLValue *value) { return value->getKind() == Named; }

public:
  llvm::StringRef getName() const {
    return getRecord()->getValueAsString("name");
  }
  llvm::StringRef getAbbrivation() const {
    return getRecord()->getValueAsString("abbreviation");
  }
  bool getDynamic() const { return getRecord()->getValueAsBit("dynamic"); }
  bool isPredefined() const {
    return not getRecord()->isAnonymous() and
           getRecord()->getName().starts_with("__");
  }
};

class PDLOperator : public PDLValue {
  friend class PDLContext;

private:
  PDLOperatorKind opkind;
  llvm::SmallVector<PDLValue *, 3> operands;

private:
  PDLOperator(PDLContext *ctx, llvm::Record *record);
  PDLOperator(PDLContext *ctx, PDLOperatorKind opkind,
              llvm::ArrayRef<PDLValue *> operands)
      : PDLValue(Operator, ctx, nullptr), opkind(opkind),
        operands(operands.begin(), operands.end()) {}

public:
  static bool classOf(PDLValue *value) {
    return value->getKind() == Operator;
  }

public:
  bool isArtificial() const { return getRecord() == nullptr; }
  PDLOperatorKind getOperatorKind() const { return opkind; }
  bool isCommutative() const {
    return opkind == PDLOperatorKind::add or opkind == PDLOperatorKind::land or
           opkind == PDLOperatorKind::lor;
  }
  int getNumOperands() const { return operands.size(); }
  llvm::ArrayRef<PDLValue *> getOperands() const { return operands; }
  PDLValue *getOperand(int Index) const { return operands[Index]; }
  template <typename T> T *getOperandAs(int Index) const {
    return llvm::dyn_cast<T>(operands[Index]);
  }
};

class alignas(8) PDLContext {
private:
  llvm::DenseMap<llvm::Record *, std::unique_ptr<PDLValue>> values;
  llvm::SmallVector<std::unique_ptr<PDLLiteral>, 0> literals;
  llvm::SmallVector<std::unique_ptr<PDLOperator>, 0> operators;
  // predefined named values
  PDLValue *__binary;
  PDLValue *__sign;
  PDLValue *__exponent;
  PDLValue *__significants;
  // small integer literals for convinience
  static constexpr int SmallIntegerLimit = 65;
  std::array<std::unique_ptr<PDLValue>, SmallIntegerLimit> smallIntegers;

public:
  explicit PDLContext() = default;
  ~PDLContext() = default;

public:
  // Get a PDLValue, create if not exists.
  PDLValue *get(llvm::Record *record);
  PDLLiteral *get(int value);
  PDLOperator *get(PDLOperatorKind opkind, llvm::ArrayRef<PDLValue *> operands);
};

class PDLAnalyzer {
private:
  PDLContext &ctx;
  llvm::DenseMap<PDLValue *, PDLValue *> canonicalized;
  llvm::DenseMap<PDLValue *, std::pair<PDLValue *, PDLValue *>> results;

public:
  explicit PDLAnalyzer(PDLContext &ctx)
      : ctx(ctx), canonicalized(), results() {}

public:
  PDLContext &getContext() const { return ctx; }
  int compare(PDLValue *lhs, PDLValue *rhs);
  const std::pair<PDLValue *, PDLValue *> &getRange(PDLValue *value);
  const std::pair<PDLValue *, PDLValue *> &getWidthRange(PDLValue *value);

private:
  // enum CanonicalStatus { Canonical, Noncanonical, Unknown };
  // CanonicalStatus getCanonicalStatus(PDLValue *value, PDLValue *&canonicalValue) const;
  PDLValue *canonicalize(PDLValue *value);
}; // class SemanticInfo

} // namespace tblgen
} // namespace clang

#endif // CLANG_UTILS_TABLEGEN_PRECISIONDESCRIPTIONLANGUAGE_H
