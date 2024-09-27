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
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include <array>
#include <memory>
#include <utility>

namespace clang {
namespace tblgen {

enum class PDLType { Integer, Binary };

enum class PDLOpKind {
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
  shl,
  shr,
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
  neg,
  pow2,
  scl2,
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

class alignas(8) PDLValue {
public:
  enum Kind { Literal, Named, Operator };

protected:
  llvm::PointerIntPair<PDLContext *, 3, unsigned> ctxinfo;

protected:
  explicit PDLValue(PDLContext *ctx, PDLType type, Kind kind)
      : ctxinfo(ctx, composeTypeKind(type, kind)) {}

public:
  static bool classof(const PDLValue *value) { return true; }

public:
  PDLContext *getContext() const { return ctxinfo.getPointer(); }
  Kind getKind() const { return static_cast<Kind>(ctxinfo.getInt() & 0x03u); }
  PDLType getType() const {
    return static_cast<PDLType>((ctxinfo.getInt() >> 2) & 0x01u);
  }
  bool isInteger() const { return getType() == PDLType::Integer; }
  bool isBinary() const { return getType() == PDLType::Binary; }
  // bool isDynamic() const;

private:
  static unsigned composeTypeKind(PDLType type, Kind kind) {
    return static_cast<unsigned>(type) | (static_cast<unsigned>(kind) << 2);
  }
};

class PDLLiteral : public PDLValue {
  friend class PDLContext;

protected:
  int value;

private:
  PDLLiteral(PDLContext *ctx, int value)
      : PDLValue(ctx, PDLType::Integer, Literal), value(value) {}

public:
  static bool classof(const PDLValue *value) {
    return value->getKind() == Literal;
  }

public:
  int getValue() const { return value; }
};

class PDLRecordValue : public PDLValue {
  friend class PDLContext;

protected:
  llvm::PointerIntPair<llvm::Record *, 2, unsigned> record;

protected:
  PDLRecordValue(PDLContext *ctx, PDLType type, Kind kind, llvm::Record *record)
      : PDLValue(ctx, type, kind), record(record) {}

public:
  static bool classof(const PDLValue *value) {
    return value->getKind() == Named or value->getKind() == Operator;
  }

public:
  llvm::Record *getRecord() const { return record.getPointer(); }
  bool isPredefined() const {
    assert(getKind() == Named);
    return record.getInt() & 0x01u;
  }
  bool isArtificial() const {
    assert(getKind() == Operator);
    return record.getInt() & 0x02u;
  }
  void setPredefined() {
    assert(getKind() == Named);
    record.setInt(record.getInt() | 0x01u);
  }
  void setArtificial() {
    assert(getKind() == Operator);
    record.setInt(record.getInt() | 0x02u);
  }
};

class PDLNamedValue : public PDLRecordValue {
  friend class PDLContext;

private:
  PDLNamedValue(PDLContext *ctx, llvm::Record *record, bool predefined = false);

public:
  static bool classof(const PDLValue *value) {
    return value->getKind() == Named;
  }

public:
  llvm::StringRef getName() const {
    return getRecord()->getValueAsString("name");
  }
  llvm::StringRef getAbbrivation() const {
    return getRecord()->getValueAsString("abbreviation");
  }
};

class PDLOperator : public PDLRecordValue {
  friend class PDLContext;

protected:
  PDLOpKind opkind;
  llvm::SmallVector<PDLValue *, 3> operands;

private:
  PDLOperator(PDLContext *ctx, llvm::Record *record);
  PDLOperator(PDLContext *ctx, PDLOpKind opkind, llvm::ArrayRef<PDLValue *> operands);

public:
  static bool classof(const PDLValue *value) {
    return value->getKind() == Operator;
  }

public:
  PDLOpKind getOperatorKind() const { return opkind; }
  bool isCommutative() const {
    return opkind == PDLOpKind::add or opkind == PDLOpKind::land or
           opkind == PDLOpKind::lor;
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
  llvm::SmallVector<std::unique_ptr<PDLLiteral>, 0> literals;
  llvm::DenseMap<llvm::Record *, std::unique_ptr<PDLValue>> values;
  llvm::Record *ity;
  llvm::Record *bty;
  PDLNamedValue *__binary;
  PDLNamedValue *__sign;
  PDLNamedValue *__exponent;
  PDLNamedValue *__significants;
  PDLNamedValue *__width;
  PDLNamedValue *__size;
  PDLNamedValue *__expwidth;
  PDLNamedValue *__fracwidth;
  llvm::Record *operators[static_cast<int>(PDLOpKind::NumOperators)];

public:
  explicit PDLContext(llvm::RecordKeeper &records);

public:
  PDLLiteral *get(int value);
  PDLValue *get(llvm::Record *record);
  PDLOperator *get(PDLOpKind opkind, llvm::ArrayRef<PDLValue *> operands);
  PDLType getType(llvm::Record *record) const {
    return record->getValueAsDef("type") == ity ? PDLType::Integer
                                                : PDLType::Binary;
  }
  // void canonicalize(PDLOpKind &opkind, llvm::SmallVectorImpl<PDLValue *>
  // &operands);
  bool is__binary(PDLNamedValue *v) const { return v == __binary; }
  bool is__sign(PDLNamedValue *v) const { return v == __sign; }
  bool is__exponent(PDLNamedValue *v) const { return v == __exponent; }
  bool is__significants(PDLNamedValue *v) const { return v == __significants; }
  bool is__width(PDLNamedValue *v) const { return v == __width; }
  bool is__size(PDLNamedValue *v) const { return v == __size; }
  bool is__expwidth(PDLNamedValue *v) const { return v == __expwidth; }
  bool is__fracwidth(PDLNamedValue *v) const { return v == __fracwidth; }
  PDLNamedValue *get__binary() const { return __binary; }
  PDLNamedValue *get__sign() const { return __sign; }
  PDLNamedValue *get__exponent() const { return __exponent; }
  PDLNamedValue *get__significants() const { return __significants; }
  PDLNamedValue *get__width() const { return __width; }
  PDLNamedValue *get__size() const { return __size; }
  PDLNamedValue *get__expwidth() const { return __expwidth; }
  PDLNamedValue *get__fracwidth() const { return __fracwidth; }
  PDLType getOperatorType(PDLOpKind opkind, llvm::ArrayRef<PDLValue *> operands) const;
};

class PDLChecker {
private:
  PDLContext &ctx;
  PDLValue *root;
public:
  PDLChecker(PDLContext &ctx) : ctx(ctx), root(nullptr) {}

public:
  void checkDecoder(llvm::Record *decoder);
private:
  void check(PDLValue *value, llvm::ArrayRef<PDLValue *> parameters);
  void check(PDLValue *value, llvm::ArrayRef<PDLOpKind> excludedops);
};

class PDLAnalyzer {
public:
  using Range = std::pair<PDLValue *, PDLValue *>;
private:
  PDLContext &ctx;
  llvm::ArrayRef<PDLNamedValue *> parameters;
  llvm::DenseMap<PDLValue *, Range> ranges;

public:
  explicit PDLAnalyzer(PDLContext &ctx,
                       llvm::ArrayRef<PDLNamedValue *> parameters)
      : ctx(ctx), parameters(parameters), ranges() {}

public:
  Range &getRange(PDLValue *value);
};

class PrecisionType {
public:
  struct Decoder {
    llvm::SmallVector<PDLValue *, 4> parameters;
    PDLValue *sign;
    PDLValue *exponent;
    PDLValue *significants;
    PDLValue *poszero;
    PDLValue *negzero;
    PDLValue *zero;
    PDLValue *posinf;
    PDLValue *neginf;
    PDLValue *inf;
    PDLValue *signalnan;
    PDLValue *quietnan;
    PDLValue *nan;
  };
  struct Encoder {
    llvm::SmallVector<PDLNamedValue *, 4> parameters;
    PDLValue *binary;
    PDLValue *poszero;
    PDLValue *negzero;
    PDLValue *zero;
    PDLValue *posinf;
    PDLValue *neginf;
    PDLValue *inf;
    PDLValue *signalnan;
    PDLValue *quietnan;
    PDLValue *nan;
  };
  struct Rounder {
    llvm::SmallVector<PDLNamedValue *, 4> parameters;
    PDLValue *precision;
  };

private:
  llvm::Record *record;
  llvm::SmallVector<PDLNamedValue *, 4> parameters;
  PDLValue *width;
  Decoder decoder;
  Encoder encoder;
  Rounder rounder;

public:
  explicit PrecisionType(PDLContext &ctx, llvm::Record *record);

public:
  llvm::Record *getRecord() const { return record; }
  PDLContext *getContext() const { return width->getContext(); }
  llvm::ArrayRef<PDLNamedValue *> getParameters() const { return parameters; }
  PDLValue *getWidth() const { return width; }
  const Decoder &getDecoder() const { return decoder; }
  const Encoder &getEncoder() const { return encoder; }
  const Rounder &getRounder() const { return rounder; }
};

class PDLCodeGenerator {
  using NameType = llvm::SmallString<8>;
private:
  PDLContext &ctx;
  PrecisionType &type;
  llvm::DenseMap<PDLValue *, NameType> varmap;
  int varid;

public:
  explicit PDLCodeGenerator(PDLContext &ctx, PrecisionType &type)
      : ctx(ctx), type(type), varmap(), varid(0) {}

public:
  PDLContext *getContext() const { return &ctx; }
  PrecisionType &getPrecisionType() const { return type; }

private:
  NameType genMaxIntOrBinWidth(PDLValue *value,
                               llvm::ArrayRef<PDLNamedValue *> parameters,
                               llvm::raw_ostream &os, unsigned indent);
  NameType varname(PDLValue *value);
};

} // namespace tblgen
} // namespace clang

#endif // CLANG_UTILS_TABLEGEN_PRECISIONDESCRIPTIONLANGUAGE_H
