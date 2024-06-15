#ifndef MLIR_DIALECT_PRECISION_IR_PRECISIONDIALECT
#define MLIR_DIALECT_PRECISION_IR_PRECISIONDIALECT

include "mlir/IR/DialectBase.td"

def Precision_Dialect : Dialect {
  let name = "precision";
  let summary = "An MLIR dialect for multi-precision computation";
  let description = "";
  let cppNamespace = "::mlir::precision";

  // FIXME we may need this
  // let hasConstantMaterializer = 1;
  // let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = ["arith::ArithDialect", "func::FuncDialect"];
}

#endif // MLIR_DIALECT_PRECISION_IR_PRECISIONDIALECT
