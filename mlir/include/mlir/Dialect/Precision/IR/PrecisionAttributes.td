#ifndef MLIR_DIALECT_PRECISION_IR_PRECISIONATTRIBUTES
#define MLIR_DIALECT_PRECISION_IR_PRECISIONATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Dialect/Precision/IR/PrecisionDialect.td"

class PrecisionAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Precision_Dialect, name, traits # [TypedAttrInterface]> {
  let mnemonic = attrMnemonic;
}

// def IntegerAttr : PrecisionAttr<"Integer", "integer"> {
//   let summary = "An attribute containing an arbitarty precision integer value";
//   let description = [{}];
//   let parameters = (ins AttributeSelfTypeParameter<"">:$type, "APInt":$value);
//   let assemblyFormat = "`<` $value `>`";
//   let extraClassDeclaration = [{
//     using ValueType = APInt;
//   }];
//   let builders = [
//     AttrBuilder<(ins "const APInt &":$value), [{
//       auto type = precision::IntegerType::get($_ctxt);
//       return $_get($_ctxt, type, value);
//     }]>,
//     AttrBuilder<(ins "int64_t":$value), [{
//       APInt api(64, value);
//       return get($_ctxt, api);
//     }]>,
//     AttrBuilder<(ins "ArrayRef<uint64_t>":$limbs), [{
//       APInt api(64 * limbs.size(), limbs);
//       return get($_ctxt, api);
//     }]>
//   ];
// }

// def RealAttr : PrecisionAttr<"Real", "real"> {
//   let summary = "An attribute containing an arbitarty precision real value";
//   let description = [{}];
//   let parameters = (ins AttributeSelfTypeParameter<"", "precision::RealType">:$type, APFloatParameter<"">:$value);
//   let assemblyFormat = "`<` $value `>`";
//   let builders = [
//     AttrBuilder<(ins "const APFloat &":$value), [{
//       auto type = precision::RealType::get($_ctxt);
//       return $_get($_ctxt, type, value);
//     }]>,
//     // AttrBuilder<(ins "double":$value), [{
//     //   APInt api(64, value);
//     //   return get($_ctxt, api);
//     // }]>,
//     // AttrBuilder<(ins "float":$value), [{
//     //   APInt api(64, value);
//     //   return get($_ctxt, api);
//     // }]>,
//   ];
// }

#endif // MLIR_DIALECT_PRECISION_IR_PRECISIONATTRIBUTES
