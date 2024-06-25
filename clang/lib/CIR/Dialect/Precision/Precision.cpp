#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Precision/Precision.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"

namespace mlir {

AsmPrinter &operator<<(AsmPrinter &printer, const llvm::APInt &value) {
  llvm::SmallString<64> str;
  value.toString(str, 16, false);
  return printer << str;
}

template <> struct FieldParser<llvm::APInt> {
  static FailureOr<llvm::APInt> parse(AsmParser &parser) {
    llvm::APInt value;
    auto res = parser.parseOptionalInteger(value);
    if (res.has_value() and res.value().succeeded())
      return value;
    return failure();
  }
};

} // namespace mlir

namespace mlir {
namespace precision {

// static LogicalResult parseAPInt(llvm::StringRef str, llvm::APInt &value) {
//   return success(str.getAsInteger(0, value));
// }

// custom parser/printer used SignificantsType
// static LogicalResult parseIntegerWidth(AsmParser &parser, unsigned &width) {
//   if (parser.parseOptionalQuestion())
//     return parser.parseInteger(width);
//   width = IntegerType::kDynamic;
//   return success();
// }

// static void printIntegerWidth(AsmPrinter &printer, unsigned width) {
//   if (width == IntegerType::kDynamic)
//     printer << '?';
//   else
//     printer << width;
// }
//
static LogicalResult parseSignedness(AsmParser &parser, bool &isSigned) {
  llvm::StringRef sign;
  if (parser.parseKeyword(&sign))
    return failure();
  if (sign.equals("s")) {
    isSigned = true;
    return success();
  }
  if (sign.equals("u")) {
    isSigned = false;
    return success();
  }
  return failure();
}

static void printSignedness(AsmPrinter &printer, bool isSigned) {
  printer << (isSigned ? 's' : 'u');
}
// LogicalResult
// IntegerType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
//                     unsigned width) {
//   return width <= kMaxWidth ? success()
//                             : emitError()
//                                   << "Invalid integer configuration (" <<
//                                   width
//                                   << "), expecting width <= kMaxWidth";
// }

LogicalResult
FixedPointType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                       unsigned width, unsigned scale) {
  return width > scale and scale > 0
             ? success()
             : emitError() << "Invalid fixed-point configuration (" << width
                           << ", " << scale << "), expecting width > scale > 0";
}

LogicalResult
FloatingPointType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                          unsigned width, unsigned exponentSize) {
  return exponentSize + 1 < width and width <= 128
             ? success()
             : emitError() << "Invalid floating-point configuration (" << width
                           << ", " << exponentSize
                           << "), expecting exponentSize+1 < width <= 128";
}

LogicalResult
PositType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  unsigned width, unsigned exponentSize) {
  return exponentSize + 3 <= width and width <= 128
             ? success()
             : emitError() << "Invalid posit configuration (" << width << ", "
                           << exponentSize
                           << "), expecting exponentSize+3 <= width <= 128";
}

bool CIToIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  assert(inputs.size() == 1 and outputs.size() == 1);
  return llvm::isa<cir::IntType>(inputs.front()) and
         llvm::isa<IntegerType>(outputs.front());
}
bool IToCIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  assert(inputs.size() == 1 and outputs.size() == 1);
  return llvm::isa<IntegerType>(inputs.front()) and
         llvm::isa<cir::IntType>(outputs.front());
}
bool CFToIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  assert(inputs.size() == 1 and outputs.size() == 1);
  return llvm::isa<cir::CIRFPTypeInterface>(inputs.front()) and
         llvm::isa<IntegerType>(outputs.front());
}
bool IToCFOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  assert(inputs.size() == 1 and outputs.size() == 1);
  return llvm::isa<IntegerType>(inputs.front()) and
         llvm::isa<cir::CIRFPTypeInterface>(outputs.front());
}

OpFoldResult CIToIOp::fold(FoldAdaptor adaptor) {
  auto ty = getType();
  return constFoldCastOp<cir::IntAttr, precision::IntegerAttr, llvm::APInt,
                         precision::IntegerAttr::ValueType>(
      adaptor.getOperands(), ty, [](const llvm::APInt &api, bool &status) {
        status = true;
        return api;
      });
}
OpFoldResult IToCIOp::fold(FoldAdaptor adaptor) {
  auto ty = getType();
  return constFoldCastOp<precision::IntegerAttr, cir::IntAttr,
                         precision::IntegerAttr::ValueType, llvm::APInt>(
      adaptor.getOperands(), ty, [](const llvm::APInt &api, bool &status) {
        status = true;
        return api;
      });
}
OpFoldResult CFToIOp::fold(FoldAdaptor adaptor) {
  auto ty = getType();
  return constFoldCastOp<cir::FPAttr, precision::IntegerAttr, llvm::APFloat,
                         precision::IntegerAttr::ValueType>(
      adaptor.getOperands(), ty, [](const APFloat &apf, bool &castStatus) {
        bool isExact;
        APSInt api(APFloat::semanticsIntSizeInBits(apf.getSemantics(), true));
        castStatus = APFloat::opInvalidOp !=
                     apf.convertToInteger(api, APFloat::rmTowardZero, &isExact);
        return api;
      });
}
OpFoldResult IToCFOp::fold(FoldAdaptor adaptor) {
  auto ty = getType();
  return constFoldCastOp<precision::IntegerAttr, cir::FPAttr,
                         precision::IntegerAttr::ValueType, llvm::APFloat>(
      adaptor.getOperands(), ty, [&ty](const llvm::APInt &api, bool &status) {
        status = false;
        const llvm::fltSemantics *semantics =
            llvm::TypeSwitch<Type, const llvm::fltSemantics *>(ty)
                .Case<cir::SingleType>([](cir::SingleType) {
                  return &llvm::APFloat::IEEEsingle();
                })
                .Case<cir::DoubleType>([](cir::DoubleType) {
                  return &llvm::APFloat::IEEEdouble();
                })
                .Case<cir::FP80Type>([](cir::FP80Type) {
                  return &llvm::APFloat::x87DoubleExtended();
                })
                .Case<cir::LongDoubleType>([](cir::LongDoubleType t) {
                  // auto underlyingType = t.getUnderlying();
                  // RODSFIXME
                  return &llvm::APFloat::IEEEdouble();
                })
                .Default([](Type) { return nullptr; });
        if (semantics) {
          status = true;
          APFloat apf(*semantics);
          apf.convertFromAPInt(api, /*IsSigned=*/true, // RODSFIXME
                               APFloat::rmNearestTiesToEven);
          return apf;
        }
        return APFloat(0.0); // To suppress the compiler warning
      });
}
} // namespace precision
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "clang/CIR/Dialect/Precision/PrecisionOpsTypes.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "clang/CIR/Dialect/Precision/PrecisionAttributes.cpp.inc"
#define GET_OP_CLASSES
#include "clang/CIR/Dialect/Precision/PrecisionOps.cpp.inc"
#include "clang/CIR/Dialect/Precision/PrecisionOpsDialect.cpp.inc"

namespace mlir {
namespace precision {

void PrecisionDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "clang/CIR/Dialect/Precision/PrecisionAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "clang/CIR/Dialect/Precision/PrecisionOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "clang/CIR/Dialect/Precision/PrecisionOps.cpp.inc"
      >();
}

} // namespace precision
} // namespace mlir
