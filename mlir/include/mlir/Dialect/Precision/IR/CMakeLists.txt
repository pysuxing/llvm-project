add_mlir_dialect(PrecisionOps precision)
add_mlir_doc(PrecisionOps PrecisionOps Dialects/ -gen-op-doc)

# Add Precision attributes
set(LLVM_TARGET_DEFINITIONS PrecisionAttributes.td)
# mlir_tablegen(PrecisionEnums.h.inc -gen-enum-decls)
# mlir_tablegen(PrecisionEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(PrecisionAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(PrecisionAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRPrecisionAttributesIncGen)
add_dependencies(mlir-generic-headers MLIRPrecisionAttributesIncGen)
