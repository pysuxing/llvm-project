add_llvm_component_library(LLVMMXP
	MPTune.cpp
  BF16CastPass.cpp
  FP128ToDDPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/MPTune

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  InstCombine
  Analysis
  Core
  Support
  TransformUtils
  )
