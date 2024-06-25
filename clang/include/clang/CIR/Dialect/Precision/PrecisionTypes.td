#ifndef MLIR_DIALECT_PRECISION_IR_PRECISIONTYPES
#define MLIR_DIALECT_PRECISION_IR_PRECISIONTYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "clang/CIR/Dialect/Precision/PrecisionDialect.td"

class PrecisionType<string name, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<Precision_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def IntegerType : PrecisionType<"Integer", "integer"> {
  // let cppBaseClassName = "mlir::IntegerType";
  let summary = "Integer type with arbitrary precision)";
  let description = [{}];
  let parameters = (ins "bool":$isSigned);
  let assemblyFormat = "`<` custom<Signedness>($isSigned) `>`";
  // let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    bool isSigned() const { return getIsSigned(); }
    bool isUnsigned() const { return not getIsSigned(); }
  }];
}

// def IntegerType : PrecisionType<"Integer", "integer"> {
//   let summary = "Integer type with arbitrary precision)";
//   let description = [{}];
//   let parameters = (ins);
// }

def RealType : PrecisionType<"Real", "real"> {
  let summary = "Real type with arbitrary precision up to a fixed limit)";
  let description = [{}];
  let parameters = (ins);
}

def FixedPointType : PrecisionType<"FixedPoint", "fixed"> {
  // FIXME add saturation semantics
  let summary = "Fixed-point type with fixed width and scale";
  let description = [{}];
  let parameters = (ins "unsigned":$width, "unsigned":$scale);
  let assemblyFormat = "`<` $width `,` $scale `>`";
  let genVerifyDecl = 1;
}

def FloatingPointType : PrecisionType<"FloatingPoint", "float"> {
  let summary = "Floating-point type with fixed width and exponent size";
  let description = [{}];
  let parameters = (ins "unsigned":$width, "unsigned":$exponentSize);
  let assemblyFormat = "`<` $width `,` $exponentSize `>`";
  let genVerifyDecl = 1;
}

def PositType : PrecisionType<"Posit", "posit"> {
  let summary = "Posit type with fixed width and exponent size";
  let description = [{}];
  let parameters = (ins "unsigned":$width, "unsigned":$exponentSize);
  let assemblyFormat = "`<` $width `,` $exponentSize `>`";
  let genVerifyDecl = 1;
}

#endif // MLIR_DIALECT_PRECISION_IR_PRECISIONTYPES
