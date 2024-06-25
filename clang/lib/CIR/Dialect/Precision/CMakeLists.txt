add_clang_library(MLIRPrecision
  Precision.cpp

#   ADDITIONAL_HEADER_DIRS
#   ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

  DEPENDS
  MLIRPrecisionOpsIncGen
  MLIRPrecisionAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  # MLIRParser
)
