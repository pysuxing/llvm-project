#include "PrecisionDescriptionLanguage.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/TableGen/Record.h"
#include <cstdint>
#include <memory>
#include <utility>

using namespace llvm;

namespace clang {
namespace tblgen {

static const char *PDLOperatorNames[]{
    "ones",  "zeros",  "none",  "all",  "any",  "msb",   "lsb",   "takel",
    "takeh", "dropl",  "droph", "bshl", "bshr", "lext0", "rext0", "lext1",
    "rext1", "concat", "clz",   "ctz",  "clo",  "cto",   "clb",   "ctb",
    "add",   "sub",    "ishl",  "ishr", "eq",   "ne",    "lt",    "le",
    "gt",    "ge",     "lnot",  "land", "lor",  "cond",
};

static constexpr unsigned NumPDLOperators =
    static_cast<unsigned>(PDLOperatorKind::NumOperators);

static_assert(sizeof(PDLOperatorNames) / sizeof(const char *) ==
              NumPDLOperators);

// bool PDLValue::operator==(const PDLValue &other) const {
//   assert(this != &other);
//   if (isa<PDLLiteral>(this) and isa<PDLLiteral>(&other))
//     return cast<PDLLiteral>(this)->getValue() ==
//            cast<PDLLiteral>(&other)->getValue();
//   if (isa<PDLNamedValue>(this))
//     return false;
//   auto *me = cast<PDLOperator>(this);
//   auto *you = cast<PDLOperator>(&other);
//   // PFIXME logic per operator
//   return llvm::all_of(
//       llvm::zip(me->getOperands(), you->getOperands()),
//       [](PDLValue *lhs, PDLValue *rhs) { return lhs == rhs or *lhs == *rhs; });
// }

bool PDLValue::isDynamic() const {
  if (isa<PDLLiteral>(this))
    return false;
  if (auto *namedValue = dyn_cast<PDLNamedValue>(this)) {
    return namedValue->getDynamic();
  }
  auto *op = cast<PDLOperator>(this);
  return llvm::any_of(op->getOperands(),
                      [](PDLValue *oprand) { return oprand->isDynamic(); });
}

PDLOperator::PDLOperator(PDLContext *ctx, llvm::Record *record)
    : PDLValue(Operator, ctx, record), opkind(PDLOperatorKind::NumOperators) {
  assert(record->isAnonymous());
  for (auto opname : llvm::enumerate(PDLOperatorNames)) {
    if (record->isSubClassOf(opname.value())) {
      opkind = static_cast<PDLOperatorKind>(opname.index());
      break;
    }
  }
  auto arguments = record->getValueAsListOfDefs("arguments");
  for (auto *arg : arguments) {
    auto *operand = ctx->get(arg);
    assert(operand);
    operands.push_back(operand);
  }
}

PDLValue *PDLContext::get(llvm::Record *record) {
  if (record->isSubClassOf("Literal"))
    return get(record->getValueAsInt("value"));

  auto iter = values.find(record);
  if (iter != values.end())
    return iter->getSecond().get();
  PDLValue *value = nullptr;
  if (record->isSubClassOf("NamedValue")) {
    value = new PDLNamedValue(this, record);
    if (not __binary and record->getName() == "__binary")
      __binary = value;
    if (not __sign and record->getName() == "__sign")
      __sign = value;
    if (not __exponent and record->getName() == "__exponent")
      __exponent = value;
    if (not __significants and record->getName() == "__significants")
      __significants = value;
  } else {
    assert(record->isSubClassOf("Operator"));
    value = new PDLOperator(this, record);
  }
  auto iterbool = values.try_emplace(record, value);
  assert(iterbool.second and iterbool.first->getFirst());
  return value;
}

PDLLiteral *PDLContext::get(int value) {
  for (auto &entry : values) {
    if (auto *lit = dyn_cast<PDLLiteral>(entry.getSecond().get()))
      if (lit->getValue() == value)
        return lit;
  }
  for (auto &lit : literals) {
    if (lit->getValue() == value)
      return lit.get();
  }
  return literals.emplace_back(new PDLLiteral(this, value)).get();
}

PDLOperator *PDLContext::get(PDLOperatorKind opkind,
                             llvm::ArrayRef<PDLValue *> operands) {
  auto op =
      std::unique_ptr<PDLOperator>(new PDLOperator(this, opkind, operands));
  // for (auto &aop : operators) {
  //   if (*op == *aop)
  //     return aop.get();
  // }
  return operators.emplace_back(std::move(op)).get();
}

const std::pair<PDLValue *, PDLValue *> &
PDLAnalyzer::getRange(PDLValue *value) {
  auto iter = results.find(value);
  if (iter != results.end())
    return iter->getSecond();
  if (isa<PDLLiteral>(value) or isa<PDLNamedValue>(value)) {
    return results.try_emplace(value, value, value).first->getSecond();
  }
  auto *op = cast<PDLOperator>(value);
  auto opkind = op->getOperatorKind();
}


PDLValue *PDLAnalyzer::canonicalize(PDLValue *value) {
  if (isa<PDLLiteral>(value) or isa<PDLNamedValue>(value))
    return value;
  auto iter = canonicalized.find(value);
  if (iter != canonicalized.end())
    return iter->getSecond();
  auto *op = cast<PDLOperator>(value);
  auto opkind = op->getOperatorKind();
  PDLValue *canonicalValue = value;
  // auto numOperands = op->getNumOperands();
  // if (numOperands == 1) {
  //   auto *operand = op->getOperand(0);
  //   auto *canonicalOperand = canonicalize(operand);
  //   if (operand != canonicalOperand)
  //     canonicalValue = ctx.get(opkind, operand);
  // } else if (numOperands == 2) {
  //   auto *lhs = op->getOperand(0), *rhs = op->getOperand(1);
  //   if (op->isCommutative() and compare(lhs, rhs) > 0)
  //     std::swap(lhs, rhs);
  //   auto *canonicalLhs = canonicalize(lhs), *canonicalRhs = canonicalize(rhs);
  //   if (lhs != canonicalLhs or rhs != canonicalRhs)
  //     canonicalValue = ctx.get(opkind, {canonicalLhs, canonicalRhs});
  // } else {
  //   assert(opkind == PDLOperatorKind::cond);
  //   auto *cond = op->getOperand(0), *tval = op->getOperand(1),
  //        *fval = op->getOperand(2);
    
  // }
  // return canonicalValue;

  auto operands = op->getOperands();
  SmallVector<PDLValue *, 3> canonicalOperands;
  for (auto *operand : operands)
    canonicalOperands.push_back(canonicalize(operand));

  switch (opkind) {
  default:
    llvm_unreachable("unexpected opkind");
  case PDLOperatorKind::ones:
  case PDLOperatorKind::zeros:
  case PDLOperatorKind::none:
  case PDLOperatorKind::all:
  case PDLOperatorKind::any:
  case PDLOperatorKind::msb:
  case PDLOperatorKind::lsb:
  case PDLOperatorKind::takel:
  case PDLOperatorKind::takeh:
  case PDLOperatorKind::dropl:
  case PDLOperatorKind::droph:
  case PDLOperatorKind::bshl:
  case PDLOperatorKind::bshr:
  case PDLOperatorKind::lext0:
  case PDLOperatorKind::rext0:
  case PDLOperatorKind::lext1:
  case PDLOperatorKind::rext1:
  case PDLOperatorKind::concat:
  case PDLOperatorKind::clz:
  case PDLOperatorKind::ctz:
  case PDLOperatorKind::clo:
  case PDLOperatorKind::cto:
  case PDLOperatorKind::clb:
  case PDLOperatorKind::ctb:
  case PDLOperatorKind::add:
  case PDLOperatorKind::sub:
  case PDLOperatorKind::ishl:
  case PDLOperatorKind::ishr:
  case PDLOperatorKind::eq:
  case PDLOperatorKind::ne:
  case PDLOperatorKind::lt:
  case PDLOperatorKind::le:
  case PDLOperatorKind::gt:
  case PDLOperatorKind::ge:
  case PDLOperatorKind::lnot:
  case PDLOperatorKind::land:
  case PDLOperatorKind::lor:
  case PDLOperatorKind::cond:
    break;
  }
}

} // namespace tblgen
} // namespace clang