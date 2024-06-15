add_mlir_dialect_library(MLIRPrecisionDialect
  Precision.cpp

#   ADDITIONAL_HEADER_DIRS
#   ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

  DEPENDS
#   MLIRLinalgInterfacesIncGen
#   MLIRLinalgOpsAttributesIncGen
#   MLIRLinalgOpsEnumsIncGen
  MLIRPrecisionOpsIncGen
  MLIRPrecisionAttributesIncGen
#   MLIRLinalgStructuredOpsIncGen

  LINK_LIBS PUBLIC
#   MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
#   MLIRArithUtils
#   MLIRBufferizationDialect
#   MLIRDestinationStyleOpInterface
#   MLIRDialectUtils
#   MLIRFunctionInterfaces
#   MLIRInferTypeOpInterface
  MLIRIR
  MLIRParser
#   MLIRShardingInterface
#   MLIRSideEffectInterfaces
#   MLIRSparseTensorDialect
#   MLIRSCFDialect
#   MLIRMathDialect
#   MLIRMemRefDialect
#   MLIRTensorDialect
#   MLIRTilingInterface
#   MLIRValueBoundsOpInterface
#   MLIRViewLikeInterface
  )
