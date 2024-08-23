clang_tablegen(PrecisionTypeNodes.inc -gen-precision-type-nodes
  SOURCE PrecisionDescription.td
  TARGET ClangPrecisionNodesGen
)
clang_tablegen(PrecisionTypeProperties.inc -gen-precision-type-properties
  SOURCE PrecisionDescription.td
  TARGET ClangPrecisionPropertiesGen
)
clang_tablegen(PrecisionTypeList.inc -gen-precision-type-list
  SOURCE PrecisionDescription.td
  TARGET ClangPrecisionListGen
)
clang_tablegen(PrecisionTypeParsers.cpp.inc -gen-precision-parser
  SOURCE PrecisionDescription.td
  TARGET ClangPrecisionParserGen
)
clang_tablegen(PrecisionTypeAST.inc -gen-precision-ast
  SOURCE PrecisionDescription.td
  TARGET ClangPrecisionASTGen
)
clang_tablegen(PrecisionMLIRTypes.inc -gen-precision-mlir-types
  SOURCE PrecisionDescription.td
  TARGET ClangPrecisionMLIRTypesGen
)

set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include ) # --includedir
include_directories(${MLIR_INCLUDE_DIR})

# This replicates part of the add_mlir_dialect cmake function from MLIR that
# cannot be used here. This happens because it expects to be run inside MLIR
# directory which is not the case for CIR (and also FIR, both have similar
# workarounds). Moreover, we need these tablegen targets to depend on
# the ClangPrecisionMLIRTypesGen targets above

set(LLVM_TARGET_DEFINITIONS Precision.td)
mlir_tablegen(Precision.h.inc -gen-op-decls DEPENDS ClangPrecisionMLIRTypesGen)
mlir_tablegen(Precision.cpp.inc -gen-op-defs DEPENDS ClangPrecisionMLIRTypesGen)
mlir_tablegen(PrecisionTypes.h.inc -gen-typedef-decls -typedefs-dialect=precision DEPENDS ClangPrecisionMLIRTypesGen)
mlir_tablegen(PrecisionTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=precision DEPENDS ClangPrecisionMLIRTypesGen)
mlir_tablegen(PrecisionDialect.h.inc -gen-dialect-decls -dialect=precision DEPENDS ClangPrecisionMLIRTypesGen)
mlir_tablegen(PrecisionDialect.cpp.inc -gen-dialect-defs -dialect=precision DEPENDS ClangPrecisionMLIRTypesGen)
add_public_tablegen_target(MLIRPrecisionIncGen)
add_dependencies(mlir-headers MLIRPrecisionIncGen)
