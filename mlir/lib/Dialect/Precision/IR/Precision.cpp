#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Precision/IR/Precision.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

namespace mlir {

AsmPrinter &operator<<(AsmPrinter &printer, const llvm::APInt &value) {
  llvm::SmallString<64> str;
  value.toString(str, 16, false);
  return printer << str;
}

template <>
struct FieldParser<llvm::APInt> {
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

OpFoldResult CIToIOp::fold(FoldAdaptor adaptor) {
  auto ty = getType();
  return constFoldCastOp<mlir::IntegerAttr, mlir::IntegerAttr>(
      adaptor.getOperands(), ty, [](const llvm::APInt &api, bool &status) {
        status = true;
        return api;
      });
}
OpFoldResult IToCIOp::fold(FoldAdaptor adaptor) {
  auto ty = getType();
  return constFoldCastOp<mlir::IntegerAttr, mlir::IntegerAttr>(
      adaptor.getOperands(), ty, [](const llvm::APInt &api, bool &status) {
        status = true;
        return api;
      });
}
OpFoldResult UIToCFOp::fold(FoldAdaptor adaptor) {
  auto ty = getType();
  return constFoldCastOp<mlir::IntegerAttr, mlir::FloatAttr>(
      adaptor.getOperands(), ty, [&ty](const llvm::APInt &api, bool &status) {
        status = true;
        APFloat apf(ty.getFloatSemantics(), APInt::getZero(ty.getWidth()));
        apf.convertFromAPInt(api, /*IsSigned=*/false,
                             APFloat::rmNearestTiesToEven);
        return apf;
      });
}
OpFoldResult SIToCFOp::fold(FoldAdaptor adaptor) {
  auto ty = getType();
  return constFoldCastOp<mlir::IntegerAttr, mlir::FloatAttr>(
      adaptor.getOperands(), ty, [&ty](const llvm::APInt &api, bool &status) {
        status = true;
        APFloat apf(ty.getFloatSemantics(), APInt::getZero(ty.getWidth()));
        apf.convertFromAPInt(api, /*IsSigned=*/true,
                             APFloat::rmNearestTiesToEven);
        return apf;
      });
}
OpFoldResult CFToUIOp::fold(FoldAdaptor adaptor) {
  auto ty = getType();
  return constFoldCastOp<FloatAttr, IntegerAttr>(
      adaptor.getOperands(), ty, [](const APFloat &mpf, bool &castStatus) {
        bool ignored;
        APSInt api(APFloat::semanticsIntSizeInBits(mpf.getSemantics(),
                                                   /*IsSigned=*/false));
        castStatus = APFloat::opInvalidOp !=
                     mpf.convertToInteger(api, APFloat::rmTowardZero, &ignored);
        return api;
      });
}
OpFoldResult CFToSIOp::fold(FoldAdaptor adaptor) {
  auto ty = getType();
  return constFoldCastOp<FloatAttr, IntegerAttr>(
      adaptor.getOperands(), ty, [](const APFloat &mpf, bool &castStatus) {
        bool ignored;
        APSInt api(APFloat::semanticsIntSizeInBits(mpf.getSemantics(),
                                                   /*IsSigned=*/true));
        castStatus = APFloat::opInvalidOp !=
                     mpf.convertToInteger(api, APFloat::rmTowardZero, &ignored);
        return api;
      });
}
} // namespace precision
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Precision/IR/PrecisionOpsTypes.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Precision/IR/PrecisionAttributes.cpp.inc"
#define GET_OP_CLASSES
#include "mlir/Dialect/Precision/IR/PrecisionOps.cpp.inc"
#include "mlir/Dialect/Precision/IR/PrecisionOpsDialect.cpp.inc"

namespace mlir {
namespace precision {

void PrecisionDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Precision/IR/PrecisionAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Precision/IR/PrecisionOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Precision/IR/PrecisionOps.cpp.inc"
      >();
}

} // namespace precision
} // namespace mlir
