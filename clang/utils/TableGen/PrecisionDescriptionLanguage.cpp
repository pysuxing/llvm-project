#include "PrecisionDescriptionLanguage.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

namespace clang {
namespace tblgen {

static const char *PDLOperatorNames[] {
    "ones",  "zeros",  "none",  "all",  "any",  "msb",   "lsb",   "takel",
    "takeh", "dropl",  "droph", "bshl", "bshr", "lext0", "rext0", "lext1",
    "rext1", "concat", "clz",   "ctz",  "clo",  "cto",   "clb",   "ctb",
    "add",   "sub",    "ishl",  "ishr", "eq",   "ne",    "lt",    "le",
    "gt",    "ge",     "lnot",  "land", "lor",  "cond",
};

static constexpr unsigned NumPDLOperators = static_cast<unsigned>(PDLOperatorKind::NumOperators);

static_assert(sizeof(PDLOperatorNames) / sizeof(const char *) ==
              NumPDLOperators);

PDLTypeKind PDLValue::getType() const {
  auto *type = record->getValueAsDef("type");
  return type->getName() == "IntType" ? PDLTypeKind::Integer : PDLTypeKind::Binary;
}

int PDLLiteral::getValue() const {
  return record->getValueAsInt("value");
}

StringRef PDLNamedValue::getName() const {
  return record->getValueAsString("name");
}

StringRef PDLNamedValue::getAbbrivation() const {
  return record->getValueAsString("abbreviation");
}

PDLOperator::PDLOperator(PDLContext *ctx, llvm::Record *record)
    : PDLValue(ctx, record), opkind(PDLOperatorKind::NumOperators) {
  assert(record->isAnonymous());
  for (auto opname : llvm::enumerate(PDLOperatorNames)) {
    if (record->isSubClassOf(opname.value())) {
      opkind = static_cast<PDLOperatorKind>(opname.index());
      break;
    }
  }
  auto arguments = record->getValueAsListOfDefs("arguments");
  for (auto *arg : arguments) {
    auto *operand = ctx->find(arg);
    assert(operand);
    operands.push_back(operand);
  }
}

PDLValue *PDLContext::create(llvm::Record *record) {
  assert(record->isClass());
  assert(not values.contains(record));
  PDLValue *value = nullptr;
  if (record->isSubClassOf("Literal")) {
    value = new PDLLiteral(this, record);
  } else if (record->isSubClassOf("NamedValue")) {
    value = new PDLNamedValue(this, record);
  } else {
    assert(record->isSubClassOf("Operator"));
    value = new PDLOperator(this, record);
  }
  auto iter = values.try_emplace(record, value);
  assert(iter.second and iter.first->getFirst());
  return value;
}

} // namespace tblgen
} // namespace clang 