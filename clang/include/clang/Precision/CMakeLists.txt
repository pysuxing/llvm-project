clang_tablegen(PrecisionTypeNodes.inc -gen-precision-type-nodes
  SOURCE Precision.td
  TARGET ClangPrecisionNodesGen
)
clang_tablegen(PrecisionTypeProperties.inc -gen-precision-type-properties
  SOURCE Precision.td
  TARGET ClangPrecisionPropertiesGen
)
clang_tablegen(PrecisionTypeList.inc -gen-precision-type-list
  SOURCE Precision.td
  TARGET ClangPrecisionListGen
)
clang_tablegen(PrecisionTypeParsers.cpp.inc -gen-precision-parser
  SOURCE Precision.td
  TARGET ClangPrecisionParserGen
)
clang_tablegen(PrecisionTypeAST.inc -gen-precision-ast
  SOURCE Precision.td
  TARGET ClangPrecisionASTGen
)
