#include "PrecisionDescriptionLanguage.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>

using namespace llvm;

namespace clang {
namespace tblgen {

static const char *PDLOperatorNames[]{
    "ones",  "zeros",  "none",  "all",  "any",  "msb",   "lsb",   "takel",
    "takeh", "dropl",  "droph", "shl",  "shr",  "lext0", "rext0", "lext1",
    "rext1", "concat", "clz",   "ctz",  "clo",  "cto",   "clb",   "ctb",
    "add",   "sub",    "neg",   "pow2", "scl2", "eq",    "ne",    "lt",
    "le",    "gt",     "ge",    "lnot", "land", "lor",   "cond",
};

static constexpr unsigned NumPDLOperators =
    static_cast<unsigned>(PDLOpKind::NumOperators);

static_assert(sizeof(PDLOperatorNames) / sizeof(const char *) ==
              NumPDLOperators);

// bool PDLValue::isDynamic() const {
//   if (isa<const PDLLiteral>(this))
//     return false;
//   return cast<const PDLRecordValue>(this)->getDynamic();
// }

PDLNamedValue::PDLNamedValue(PDLContext *ctx, llvm::Record *record,
                             bool predefined)
    : PDLRecordValue(ctx, ctx->getType(record), Named, record) {
  if (predefined)
    setPredefined();
}

PDLOperator::PDLOperator(PDLContext *ctx, Record *record)
    : PDLRecordValue(ctx, ctx->getType(record), Operator, record) {
  for (auto opname : llvm::enumerate(PDLOperatorNames)) {
    if (record->isSubClassOf(opname.value())) {
      opkind = static_cast<PDLOpKind>(opname.index());
      break;
    }
  }
  auto arguments = record->getValueAsListOfDefs("arguments");
  for (auto *arg : arguments) {
    auto *operand = ctx->get(arg);
    assert(operand);
    operands.push_back(operand);
  }
  // ctx->canonicalize(opkind, operands);
}
static llvm::Record *fakeRecord(PDLOpKind opkind, ArrayRef<PDLValue *> operands) {
  FoldingSetNodeID id;
  id.AddInteger(static_cast<int>(opkind));
  for (auto *value : operands)
    id.AddPointer(value);
  return reinterpret_cast<Record *>(id.ComputeHash());
}

PDLOperator::PDLOperator(PDLContext *ctx, PDLOpKind opkind,
                         llvm::ArrayRef<PDLValue *> operands)
    : PDLRecordValue(ctx, ctx->getOperatorType(opkind, operands), Operator,
                     fakeRecord(opkind, operands)),
      opkind(opkind), operands(operands) {
  setArtificial();
}

PDLContext::PDLContext(RecordKeeper &records) {
  ity = records.getDef("IntType");
  bty = records.getDef("BinType");
  assert(ity and bty);
  __binary = cast<PDLNamedValue>(get(records.getDef("__binary")));
  __sign = cast<PDLNamedValue>(get(records.getDef("__sign")));
  __exponent = cast<PDLNamedValue>(get(records.getDef("__exponent")));
  __significants = cast<PDLNamedValue>(get(records.getDef("__significants")));
  __width = cast<PDLNamedValue>(get(records.getDef("__width")));
  __size = cast<PDLNamedValue>(get(records.getDef("__size")));
  __expwidth = cast<PDLNamedValue>(get(records.getDef("__expwidth")));
  __fracwidth = cast<PDLNamedValue>(get(records.getDef("__fracwidth")));
  for (unsigned i = 0; i < NumPDLOperators; ++i) {
    operators[i] = records.getClass(PDLOperatorNames[i]);
    assert(operators[i]);
  }
}

PDLLiteral *PDLContext::get(int value) {
  for (auto &lit : literals) {
    if (lit->getValue() == value)
      return lit.get();
  }
  return literals.emplace_back(new PDLLiteral(this, value)).get();
}

PDLValue *PDLContext::get(llvm::Record *record) {
  // forward if this is a literal
  if (record->isSubClassOf("Literal"))
    return get(record->getValueAsInt("value"));

  // return if already exists
  auto iter = values.find(record);
  if (iter != values.end())
    return iter->getSecond().get();

  // create a new value
  PDLValue *value = nullptr;
  if (record->isSubClassOf("NamedValue")) {
    value = new PDLNamedValue(this, record);
  } else {
    assert(record->isSubClassOf("Operator"));
    value = new PDLOperator(this, record);
  }
  auto iterbool = values.try_emplace(record, value);
  assert(iterbool.second and iterbool.first->getFirst());
  return value;
}

PDLOperator *PDLContext::get(PDLOpKind opkind,
                             llvm::ArrayRef<PDLValue *> operands) {
  auto *record = fakeRecord(opkind, operands);
  // return if already exists
  auto iter = values.find(record);
  if (iter != values.end())
    return cast<PDLOperator>(iter->getSecond().get());
  
  auto *op = new PDLOperator(this, opkind, operands);
  values.try_emplace(record, op);
  return op;
}
  
PDLType PDLContext::getOperatorType(PDLOpKind opkind,
                                    llvm::ArrayRef<PDLValue *> operands) const {
  if (opkind == PDLOpKind::cond) {
    assert(operands[1]->getType() == operands[2]->getType());
    return operands[1]->getType();
  }
  return getType(operators[static_cast<int>(opkind)]);
}

void PDLChecker::check(PDLValue *value, ArrayRef<PDLValue *> parameters) {
  if (isa<PDLLiteral>(value))
    return;
  if (auto *named = dyn_cast<PDLNamedValue>(value)) {
    if (not is_contained(parameters, named))
      PrintFatalError(named->getRecord(), "unexpected named value");
  }
  for_each(cast<PDLOperator>(value)->getOperands(),
           [&](PDLValue *v) { check(v, parameters); });
}
  void PDLChecker::check(PDLValue *value, llvm::ArrayRef<PDLOpKind> excludedops) {}

void PDLChecker::checkDecoder(Record *decoder) {
  // Ensure that __sign, __exponent and __significants do not occur in parameters
  SmallVector<PDLValue *, 3> excludes = {ctx.get__sign(), ctx.get__exponent(),
                                         ctx.get__significants()};
  auto paramRecords = decoder->getValueAsListOfDefs("parameters");
  SmallVector<PDLNamedValue *> parameters;
  for (auto *param : paramRecords) {
    auto *named = cast<PDLNamedValue>(ctx.get(param));
    if (is_contained(excludes, named)) {
      SmallString<64> msg;
      format("Disallowed parameter for Decoder: %s", param->getName())
          .snprint(msg.data(), msg.capacity());
      PrintFatalError(decoder, msg);
    }
    parameters.push_back(named);
  }
  // Check sign, exponent, and significants
  
}

// void PDLChecker::checkEncoderOrRounder(PDLValue *value,
//                               ArrayRef<PDLValue *> parameters) {
//   root = value;
//   SmallVector<PDLValue *> params(parameters);
//   params.append({ctx.get__sign(), ctx.get__exponent(), ctx.get__significants()});
//   check(value, params);
// }

PDLAnalyzer::Range &PDLAnalyzer::getRange(PDLValue *value) {
  auto iter = ranges.find(value);
  if (iter != ranges.end())
    return iter->getSecond();

  if (isa<PDLLiteral>(value))
    return ranges.try_emplace(value, value, value).first->getSecond();
  if (auto *named = dyn_cast<PDLNamedValue>(value)) {
    if (ctx.is__binary(named))
      return getRange(ctx.get__width());
    if (ctx.is__sign(named))
      return ranges.try_emplace(named, ctx.get(0), ctx.get(1))
          .first->getSecond();
    if (ctx.is__exponent(named))
      llvm_unreachable("unexpected __exponent here");
    // __width, __size, __expwidth, __fracwidth and other integer named values
    if (named->isInteger())
      return ranges.try_emplace(named, named, named).first->getSecond();
    // __significants and other binary named values are not expected
    llvm_unreachable("unexpected binary named values");
  }

  auto *op = cast<PDLOperator>(value);
  auto opkind = op->getOperatorKind();
  switch (opkind) {
  default:
    llvm_unreachable("unexpected opkind");
  case PDLOpKind::ones:
  case PDLOpKind::zeros:
    return getRange(op->getOperand(0));
  case PDLOpKind::none:
  case PDLOpKind::all:
  case PDLOpKind::any:
  case PDLOpKind::msb:
  case PDLOpKind::lsb:
    llvm_unreachable("unexpected opkind");
  case PDLOpKind::takel:
  case PDLOpKind::takeh:
    return getRange(op->getOperand(1));
  case PDLOpKind::dropl:
  case PDLOpKind::droph: {
    auto &xrange = getRange(op->getOperand(0));
    auto &yrange = getRange(op->getOperand(1));
    auto *lb = ctx.get(PDLOpKind::sub, {xrange.first, yrange.second});
    auto *ub = ctx.get(PDLOpKind::sub, {xrange.second, yrange.first});
    return ranges.try_emplace(op, lb, ub).first->getSecond();
  }
  case PDLOpKind::shl:
  case PDLOpKind::shr:
    return getRange(op->getOperand(0));
  case PDLOpKind::lext0:
  case PDLOpKind::rext0:
  case PDLOpKind::lext1:
  case PDLOpKind::rext1: {
    auto &xrange = getRange(op->getOperand(0));
    auto *lb = ctx.get(PDLOpKind::add, {xrange.first, ctx.get(1)});
    auto *ub = ctx.get(PDLOpKind::add, {xrange.second, ctx.get(1)});
    return ranges.try_emplace(op, lb, ub).first->getSecond();
  }
  case PDLOpKind::concat: {
    auto &xrange = getRange(op->getOperand(0));
    auto &yrange = getRange(op->getOperand(1));
    auto *lb = ctx.get(PDLOpKind::add, {xrange.first, yrange.first});
    auto *ub = ctx.get(PDLOpKind::add, {xrange.second, yrange.second});
    return ranges.try_emplace(op, lb, ub).first->getSecond();
  }
  case PDLOpKind::clz:
  case PDLOpKind::ctz:
  case PDLOpKind::clo:
  case PDLOpKind::cto:
  case PDLOpKind::clb:
  case PDLOpKind::ctb:
    return getRange(op->getOperand(0));
  case PDLOpKind::add:
  case PDLOpKind::sub:
  case PDLOpKind::neg:
  case PDLOpKind::pow2:
  case PDLOpKind::scl2:
    break;
  case PDLOpKind::eq:
  case PDLOpKind::ne:
  case PDLOpKind::lt:
  case PDLOpKind::le:
  case PDLOpKind::gt:
  case PDLOpKind::ge:
  case PDLOpKind::lnot:
  case PDLOpKind::land:
  case PDLOpKind::lor:
    llvm_unreachable("unexpected opkind");
  case PDLOpKind::cond:
    break;
  }
}

inline static PDLValue *getValueOrNull(PDLContext &ctx, Record *record,
                                       StringRef field) {
  if (isa<UnsetInit>(record->getValueInit(field)))
    return nullptr;
  return ctx.get(record->getValueAsDef(field));
}

PrecisionType::PrecisionType(PDLContext &ctx, llvm::Record *record)
    : record(record) {
  assert(record);
  width = ctx.get(record->getValueAsDef("width"));
  transform(record->getValueAsListOfDefs("parameters"),
            std::back_inserter(parameters),
            [&](Record *r) { return ctx.get(r); });
  auto *drec = record->getValueAsDef("decoder");
  assert(drec);
  decoder = {{},
             getValueOrNull(ctx, drec, "sign"),
             getValueOrNull(ctx, drec, "exponent"),
             getValueOrNull(ctx, drec, "significants"),
             getValueOrNull(ctx, drec, "poszero"),
             getValueOrNull(ctx, drec, "negzero"),
             getValueOrNull(ctx, drec, "zero"),
             getValueOrNull(ctx, drec, "posinf"),
             getValueOrNull(ctx, drec, "neginf"),
             getValueOrNull(ctx, drec, "inf"),
             getValueOrNull(ctx, drec, "signalnan"),
             getValueOrNull(ctx, drec, "quietnan"),
             getValueOrNull(ctx, drec, "nan")};
  transform(drec->getValueAsListOfDefs("parameters"),
            std::back_inserter(decoder.parameters),
            [&](Record *r) { return ctx.get(r); });
  auto *erec = record->getValueAsDef("encoder");
  assert(erec);
  encoder = {{},
             getValueOrNull(ctx, erec, "binary"),
             getValueOrNull(ctx, erec, "poszero"),
             getValueOrNull(ctx, erec, "negzero"),
             getValueOrNull(ctx, erec, "zero"),
             getValueOrNull(ctx, erec, "posinf"),
             getValueOrNull(ctx, erec, "neginf"),
             getValueOrNull(ctx, erec, "inf"),
             getValueOrNull(ctx, erec, "signalnan"),
             getValueOrNull(ctx, erec, "quietnan"),
             getValueOrNull(ctx, erec, "nan")};
  transform(erec->getValueAsListOfDefs("parameters"),
            std::back_inserter(encoder.parameters),
            [&](Record *r) { return ctx.get(r); });
  auto *rrec = record->getValueAsDef("rounder");
  assert(rrec);
  rounder = {{}, getValueOrNull(ctx, rrec, "precision")};
  transform(rrec->getValueAsListOfDefs("parameters"),
            std::back_inserter(rounder.parameters),
            [&](Record *r) { return ctx.get(r); });
}

PDLCodeGenerator::NameType PDLCodeGenerator::genMaxIntOrBinWidth(
    PDLValue *value, llvm::ArrayRef<PDLNamedValue *> parameters,
    raw_ostream &os, unsigned indent) {
  if (auto *literal = dyn_cast<PDLLiteral>(value)) {
    return NameType(to_string(literal->getValue()));
  }
  if (auto *named = dyn_cast<PDLNamedValue>(value)) {
    if (ctx.is__binary(named))
      return genMaxIntOrBinWidth(type.getWidth(), type.getParameters(), os, indent);
    const auto *iter = find(parameters, named);
    if (iter == parameters.end()) {
      // PFIXME diagnostic here
      llvm_unreachable("unexpected named value");
    }
    return named->getName();
  }
  // This is an operator value
  auto iter = varmap.find(value);
  if (iter != varmap.end())
    return iter->getSecond();

  auto *op = cast<PDLOperator>(value);
  auto opkind = op->getOperatorKind();
  switch (opkind) {
  default:
    llvm_unreachable("unexpected opkind");
  case PDLOpKind::ones:
  case PDLOpKind::zeros:
    return varmap
        .try_emplace(
            op, genMaxIntOrBinWidth(op->getOperand(0), parameters, os, indent))
        .first->getSecond();
  case PDLOpKind::none:
  case PDLOpKind::all:
  case PDLOpKind::any:
  case PDLOpKind::msb:
  case PDLOpKind::lsb:
    return varmap.try_emplace(op, to_string(1)).first->getSecond();
  case PDLOpKind::takel:
  case PDLOpKind::takeh:
    return varmap
        .try_emplace(
            op, genMaxIntOrBinWidth(op->getOperand(1), parameters, os, indent))
        .first->getSecond();
  case PDLOpKind::dropl:
  case PDLOpKind::droph:
    break;
  case PDLOpKind::shl:
  case PDLOpKind::shr:
    return varmap
        .try_emplace(
            op, genMaxIntOrBinWidth(op->getOperand(0), parameters, os, indent))
        .first->getSecond();
  case PDLOpKind::lext0:
  case PDLOpKind::rext0:
  case PDLOpKind::lext1:
  case PDLOpKind::rext1:
  case PDLOpKind::concat:
    break;
  case PDLOpKind::clz:
  case PDLOpKind::ctz:
  case PDLOpKind::clo:
  case PDLOpKind::cto:
  case PDLOpKind::clb:
  case PDLOpKind::ctb:
    return varmap
        .try_emplace(
            op, genMaxIntOrBinWidth(op->getOperand(0), parameters, os, indent))
        .first->getSecond();
  case PDLOpKind::add:
  case PDLOpKind::sub:
  case PDLOpKind::neg:
  case PDLOpKind::pow2:
  case PDLOpKind::scl2:
    break;
  case PDLOpKind::eq:
  case PDLOpKind::ne:
  case PDLOpKind::lt:
  case PDLOpKind::le:
  case PDLOpKind::gt:
  case PDLOpKind::ge:
  case PDLOpKind::lnot:
  case PDLOpKind::land:
  case PDLOpKind::lor:
    return varmap.try_emplace(op, to_string(1)).first->getSecond();
  case PDLOpKind::cond:
    break;
  }
}

PDLCodeGenerator::NameType PDLCodeGenerator::varname(PDLValue *value) {
  auto varty = value->getType();
  NameType name("i");
}

// void PDLContext::canonicalize(PDLOpKind &opkind,
//                               SmallVectorImpl<PDLValue *> &operands) {
//   SmallVector<PDLValue *, 3> newOperands;
//   switch (opkind) {
//   default:
//     llvm_unreachable("unexpected opkind");
//   case PDLOpKind::ones:
//   case PDLOpKind::zeros:
//     break;
//   case PDLOpKind::none:
//     if (auto *op = dyn_cast<PDLOperator>(operands.front())) {
//       if (op->getOperatorKind() == PDLOpKind::ones) {
//         newOperands.push_back(get(0));
//       } else if (op->getOperatorKind() == PDLOpKind::zeros) {
//         newOperands.push_back(get(1));
//       }
//     }
//     break;
//   case PDLOpKind::all:
//   case PDLOpKind::any:
//     if (auto *op = dyn_cast<PDLOperator>(operands.front())) {
//       if (op->getOperatorKind() == PDLOpKind::ones) {
//         newOperands.push_back(get(1));
//       } else if (op->getOperatorKind() == PDLOpKind::zeros) {
//         newOperands.push_back(get(0));
//       }
//     }
//     break;
//   case PDLOpKind::msb:
//   case PDLOpKind::lsb:
//   case PDLOpKind::takel:
//   case PDLOpKind::takeh:
//   case PDLOpKind::dropl:
//   case PDLOpKind::droph:
//   case PDLOpKind::shl:
//   case PDLOpKind::shr:
//   case PDLOpKind::lext0:
//   case PDLOpKind::rext0:
//   case PDLOpKind::lext1:
//   case PDLOpKind::rext1:
//   case PDLOpKind::concat:
//   case PDLOpKind::clz:
//   case PDLOpKind::ctz:
//   case PDLOpKind::clo:
//   case PDLOpKind::cto:
//   case PDLOpKind::clb:
//   case PDLOpKind::ctb:
//   case PDLOpKind::add:
//   case PDLOpKind::sub:
//   case PDLOpKind::neg:
//   case PDLOpKind::pow2:
//   case PDLOpKind::scl2:
//   case PDLOpKind::eq:
//   case PDLOpKind::ne:
//   case PDLOpKind::lt:
//   case PDLOpKind::le:
//   case PDLOpKind::gt:
//   case PDLOpKind::ge:
//   case PDLOpKind::lnot:
//   case PDLOpKind::land:
//   case PDLOpKind::lor:
//   case PDLOpKind::cond:
//     break;
//   }
// }

} // namespace tblgen
} // namespace clang