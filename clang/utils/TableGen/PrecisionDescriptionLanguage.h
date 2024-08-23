//=== PrecisionDescriptionLanugage.h - The PDL language ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_UTILS_TABLEGEN_PRECISIONDESCRIPTIONLANGUAGE_H
#define CLANG_UTILS_TABLEGEN_PRECISIONDESCRIPTIONLANGUAGE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Record.h"
#include <memory>

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
protected:
  PDLContext *ctx;
  llvm::Record *record;

protected:
  PDLValue() = default;
  PDLValue(PDLContext *ctx, llvm::Record *record) : record(record) {}

public:
  PDLContext *getContext() const { return ctx; }
  llvm::Record *getInit() const { return record; }
  PDLTypeKind getType() const;
};

class PDLLiteral : public PDLValue {
  friend class PDLContext;
private:
  using PDLValue::PDLValue;

public:
  int getValue() const;
};

class PDLNamedValue : public PDLValue {
  friend class PDLContext;
private:
  using PDLValue::PDLValue;

public:
  llvm::StringRef getName() const;
  llvm::StringRef getAbbrivation() const;
};

class PDLOperator : public PDLValue {
  friend class PDLContext;
private:
  PDLOperatorKind opkind;
  llvm::SmallVector<PDLValue *, 3> operands;

private:
  PDLOperator(PDLContext *ctx, llvm::Record *record);

public:
  PDLOperatorKind getOperatorKind() const { return opkind; }
  llvm::ArrayRef<PDLValue *> getOperands() const { return operands; }
};

class PDLContext {
private:
  llvm::DenseMap<llvm::Record *, std::unique_ptr<PDLValue>> values;

public:
  explicit PDLContext() = default;
  ~PDLContext() = default;
public:
  PDLValue *create(llvm::Record *record);
  PDLValue *find(llvm::Record *record) const {
    auto iter = values.find(record);
    return iter == values.end()? nullptr : iter->getSecond().get();
  }
};

} // namespace tblgen
} // namespace clang

#endif // CLANG_UTILS_TABLEGEN_PRECISIONDESCRIPTIONLANGUAGE_H
