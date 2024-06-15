
#ifndef MLIR_DIALECT_PRECISION_IR_PRECISIONOPS
#define MLIR_DIALECT_PRECISION_IR_PRECISIONOPS

include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Precision/IR/PrecisionTypes.td"

def AnyIntegralType : AnyTypeOf<[IntegerType]>;
def AnyRealType : AnyTypeOf<[RealType, FloatingPointType, PositType]>;
def AnyArithmeticType : AnyTypeOf<[IntegerType, RealType, FixedPointType, FloatingPointType, PositType]>;

// Base class for Precision dialect ops.
class Precision_Op<string mnemonic, list<Trait> traits = []> :
  Op<Precision_Dialect, mnemonic, traits # [Pure]> {
  let summary = mnemonic;
  let description = [{}];
}
//  DeclareOpInterfaceMethods<InferTypeOpInterface>
class SpecialValueOp<string mnemonic, Type Ty, list<Trait> traits = []>
  : Precision_Op<mnemonic, traits # [ConstantLike]> {
  let arguments = (ins Builtin_TypeAttr:$type);
  let results = (outs Ty:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [
     OpBuilder<(ins "TypeAttr":$type), [{
      $_state.addAttribute("type", type);
      $_state.addTypes(type.getValue());
    }]>
  ];
  let extraClassDefinition = [{
    // LogicalResult $cppClass::inferReturnTypes(
    //     MLIRContext *context, ::std::optional<Location> location,
    //     ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    //     RegionRange regions, ::llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
    //   auto typeAttr = attributes.getAs<TypeAttr>("type");
    //   if (not typeAttr)
    //     return failure();
    //   inferredReturnTypes.push_back(typeAttr.getValue());
    //   return success();
    // }
    LogicalResult $cppClass::verify() {
      return success(getType().getValue() == getResult().getType());
    }
  }];
}
class ConstantOp<string mnemonic, Type Ty, list<Trait> traits = []>
  : Precision_Op<mnemonic, traits # [ConstantLike, AllTypesMatch<["value", "result"]>]> {
  // FIXME we need attribute of type real/posit/float to use the AllTypesMatch trait
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs Ty:$result);
  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
  let extraClassDefinition = [{
    OpFoldResult $cppClass::fold(FoldAdaptor adaptor) {
      return getValue();
    }
  }];
}
class BinaryOp<string mnemonic, Type Ty, list<Trait> traits = []>
  : Precision_Op<mnemonic, traits> {
  let arguments = (ins Ty:$lhs, Ty:$rhs);
  let results = (outs Ty:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type(operands)";
}
class UnaryOp<string mnemonic, Type Ty, list<Trait> traits = []>
  : Precision_Op<mnemonic, traits> {
  let arguments = (ins Ty:$value);
  let results = (outs Ty:$result);
  let assemblyFormat = "$value attr-dict `:` type(operands)";
}
class BitOp<string mnemonic, Type Ty, list<Trait> traits = []>
  : Precision_Op<mnemonic, traits> {
  let arguments = (ins Ty:$value, AnySignlessIntegerOrIndex:$amount);
  let results = (outs Ty:$result);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}
class CountOp<string mnemonic, Type Ty, list<Trait> traits = []>
  : Precision_Op<mnemonic, traits> {
  let arguments = (ins Ty:$value);
  let results = (outs Builtin_Index:$result);
  let assemblyFormat = "$value attr-dict `:` type(operands)";
}
class CompareOp<string mnemonic, Type Ty, I64EnumAttr Predicate, list<Trait> traits = []>
  : Precision_Op<mnemonic, traits # [SameTypeOperands]> {
  let arguments = (ins Predicate:$predicate, Ty:$lhs, Ty:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}
class InspectOp<string mnemonic, Type Ty, Type ResTy, list<Trait> traits = []>
  : Precision_Op<mnemonic, traits> {
  let arguments = (ins Ty:$value);
  let results = (outs ResTy:$result);
  let assemblyFormat = "$value attr-dict `:` type(operands)";
}
class ResizeOp<string mnemonic, Type Ty, list<Trait> traits = []>
  : Precision_Op<mnemonic, traits> {
  let arguments = (ins Ty:$value, Builtin_Index:$width);
  let results = (outs Ty:$result);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}
class ConvertOp<string mnemonic, Type FromTy, Type ToTy, list<Trait> traits = []>
  : Precision_Op<mnemonic, traits # [DeclareOpInterfaceMethods<CastOpInterface>]> {
  let arguments = (ins FromTy:$in);
  let results = (outs ToTy:$out);
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";
  // NOTE TypeDef.cppType is qualified with C++ namespace, thus more suitable than Type.cppClassName,
  // however, there exists no TypeDef for mlir::FloatType
  let extraClassDefinition = !subst("$out", ToTy.cppClassName, !subst("$in", FromTy.cppClassName, [{
    bool $cppClass::areCastCompatible(TypeRange inputs, TypeRange outputs) {
      assert(inputs.size() == 1 and outputs.size() == 1);
      return llvm::isa<$in>(inputs.front()) and llvm::isa<$out>(outputs.front());
    }
  }]));
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
/// Type inspection operations
//===----------------------------------------------------------------------===//
def WidthOp : InspectOp<"width", IntegerType, Builtin_Index>;
def ActiveWidthOp : InspectOp<"activewidth", IntegerType, Builtin_Index>; // NOTE +/- are different
def ExponentSizeOp : InspectOp<"exponentsize", AnyRealType, Builtin_Index>;
def FractionSizeOp : InspectOp<"fractionsize", AnyRealType, Builtin_Index>;

//===----------------------------------------------------------------------===//
/// Special value operations
//===----------------------------------------------------------------------===//

def ZeroIOp : SpecialValueOp<"zeroi", IntegerType>;
def ZeroROp : SpecialValueOp<"zeror", RealType>;
def ZeroFOp : SpecialValueOp<"zerof", FloatingPointType>;
def ZeroPOp : SpecialValueOp<"zerop", PositType>;
def ZeroXOp : SpecialValueOp<"zerox", FixedPointType>;
def MaxUIOp : SpecialValueOp<"maxui", IntegerType>; // 0xffff...
def MaxSIOp : SpecialValueOp<"maxsi", IntegerType>; // 0x7fff...
def MinSIOp : SpecialValueOp<"minsi", IntegerType>; // 0x8000...
// def SignOp
def MinPosOp : SpecialValueOp<"minpos", AnyRealType>;
def MaxPosOp : SpecialValueOp<"maxpos", AnyRealType>;
def PosInfOp : SpecialValueOp<"posinf", FloatingPointType>;
def NegInfOp : SpecialValueOp<"neginf", FloatingPointType>;
def NanOp : SpecialValueOp<"nan", FloatingPointType>;
def NarOp : SpecialValueOp<"nar", PositType>;

//===----------------------------------------------------------------------===//
/// Constant operations
//===----------------------------------------------------------------------===//
def ConstantIOp : ConstantOp<"consti", IntegerType>;
def ConstantROp : ConstantOp<"constr", RealType>;
def ConstantFOp : ConstantOp<"constf", FloatingPointType>;
def ConstantPOp : ConstantOp<"constp", PositType>;
def ConstantXOp : ConstantOp<"constx", FixedPointType>;

//===----------------------------------------------------------------------===//
/// Compare operations
//===----------------------------------------------------------------------===//
def CmpIOp : CompareOp<"cmpi", IntegerType, Arith_CmpIPredicateAttr>;

//===----------------------------------------------------------------------===//
/// IntegerType operations
//===----------------------------------------------------------------------===//
def AddIOp : BinaryOp<"addi", IntegerType, [Commutative]>;
def SubIOp : BinaryOp<"subi", IntegerType>;
def MulIOp : BinaryOp<"muli", IntegerType, [Commutative]>;
def DivSIOp : BinaryOp<"divsi", IntegerType>;
def DivUIOp : BinaryOp<"divui", IntegerType>;
def CDivSIOp : BinaryOp<"cdivsi", IntegerType>;
def CDivUIOp : BinaryOp<"cdivui", IntegerType>;
def FDivSIOp : BinaryOp<"fdivsi", IntegerType>;
def RemSIOp : BinaryOp<"remsi", IntegerType>;
def RemUIOp : BinaryOp<"remui", IntegerType>;
def NegSIOp : UnaryOp<"negsi", IntegerType>;
def AbsSIOp : UnaryOp<"abssi", IntegerType>;
def AndOp : BinaryOp<"and", IntegerType, [Commutative]>;
def OrOp : BinaryOp<"or", IntegerType, [Commutative]>;
def XorOp : BinaryOp<"xor", IntegerType, [Commutative]>;
def NotOp : UnaryOp<"not", IntegerType>;
def ShLIOp : BitOp<"shli", IntegerType>;
def ShRUIOp : BitOp<"shrui", IntegerType>;
def ShRSIOp : BitOp<"shrsi", IntegerType>;
def RotLOp : BitOp<"rotl", IntegerType>;
def RotROp : BitOp<"rotr", IntegerType>;
// below are low-level ops, see APInt
// def SetBitOp : BitOp<"setbit", IntegerType>;
// def ClrBitOp : BitOp<"clrbit", IntegerType>;
// def FlipBitOp : BitOp<"flipbit", IntegerType>;
// def SetLowOp : BitOp<"setbit", IntegerType>;
// def ClrLowOp : BitOp<"clrbit", IntegerType>;
// def SetHighOp : BitOp<"sethigh", IntegerType>;
// def ClrHighOp : BitOp<"clrlow", IntegerType>;
// def ExtractLowOp : BitOp<"extractlow", IntegerType>;
// def ExtractHighOp : BitOp<"extracthigh", IntegerType>; // insert?
// def SExtOp : ResizeOp<"sext", IntegerType>;
// def UExtOp : ResizeOp<"sext", IntegerType>;
// def TruncOp : ResizeOp<"trunc", IntegerType>; // saturate?

// //===----------------------------------------------------------------------===//
// /// FloatintPointType operations
// //===----------------------------------------------------------------------===//
// def AddROp : BinaryOp<"addr", RealType>;
// def SubROp : BinaryOp<"subr", RealType>;
// def MulROp : BinaryOp<"mulr", RealType>;
// def DivROp : BinaryOp<"divr", RealType>;
// def RemROp : BinaryOp<"remr", RealType>;
// def NegROp : UnaryOp<"negr", RealType>;
// def CmpROp : CompareOp<"cmpr", RealType, Arith_CmpFPredicateAttr>;

// //===----------------------------------------------------------------------===//
// /// FloatintPointType operations
// //===----------------------------------------------------------------------===//
// def AddFOp : BinaryOp<"addf", FloatingPointType>;
// def SubFOp : BinaryOp<"subf", FloatingPointType>;
// def MulFOp : BinaryOp<"mulf", FloatingPointType>;
// def DivFOp : BinaryOp<"divf", FloatingPointType>;
// def RemFOp : BinaryOp<"remf", FloatingPointType>;
// def NegFOp : UnaryOp<"negf", FloatingPointType>;
// def CmpFOp : CompareOp<"cmpf", FloatingPointType, Arith_CmpFPredicateAttr>;

// //===----------------------------------------------------------------------===//
// /// PositType operations
// //===----------------------------------------------------------------------===//
// def AddPOp : BinaryOp<"addp", PositType>;
// def SubPOp : BinaryOp<"subp", PositType>;
// def MulPOp : BinaryOp<"mulp", PositType>;
// def DivPOp : BinaryOp<"divp", PositType>;
// def RemPOp : BinaryOp<"remp", PositType>;
// def NegPOp : UnaryOp<"negp", PositType>;
// def CmpPOp : CompareOp<"cmpp", PositType, Arith_CmpFPredicateAttr>;

//===----------------------------------------------------------------------===//
/// Convertion operations
//===----------------------------------------------------------------------===//
def CIToIOp : ConvertOp<"citoi", AnyInteger, IntegerType>;
def IToCIOp : ConvertOp<"itoci", IntegerType, AnyInteger>;
def CFToUIOp : ConvertOp<"cftoui", AnyFloat, IntegerType>;
def CFToSIOp : ConvertOp<"cftoui", AnyFloat, IntegerType>;
def UIToCFOp : ConvertOp<"uitocf", IntegerType, AnyFloat>;
def SIToCFOp : ConvertOp<"sitocf", IntegerType, AnyFloat>;
// def UIToFOp : ConvertOp<"uitof", IntegerType, FloatingPointType>;
// def UIToPOp : ConvertOp<"uitop", IntegerType, PositType>;
// def UIToXOp : ConvertOp<"uitox", IntegerType, FixedPointType>;
// def SIToFOp : ConvertOp<"sitof", IntegerType, FloatingPointType>;
// def SIToPOp : ConvertOp<"sitop", IntegerType, PositType>;
// def SIToXOp : ConvertOp<"sitox", IntegerType, FixedPointType>;
// def FToUIOp : ConvertOp<"ftoui", FloatingPointType, IntegerType>;
// def FToSIOp : ConvertOp<"ftosi", FloatingPointType, IntegerType>;
// def FToPOp : ConvertOp<"ftop", FloatingPointType, PositType>;
// def FToXOp : ConvertOp<"ftox", FloatingPointType, FixedPointType>;
// def PToUIOp : ConvertOp<"ptoui", PositType, IntegerType>;
// def PToSIOp : ConvertOp<"ptosi", PositType, IntegerType>;
// def PToFOp : ConvertOp<"ptof", PositType, FloatingPointType>;
// def PToXOp : ConvertOp<"ptox", PositType, FixedPointType>;
// def XToUIOp : ConvertOp<"xtoui", FixedPointType, IntegerType>;
// def XToSIOp : ConvertOp<"xtosi", FixedPointType, IntegerType>;
// def XToFOp : ConvertOp<"xtof", FixedPointType, FloatingPointType>;
// def XToPOp : ConvertOp<"xtop", FixedPointType, PositType>;

// fromi, froms, fromf, toi, tos, tof

// Operations for all arithmetic types
// - constant op: zero, ones,
// - binary op: add, sub, mul, div
// - unary op: neg, recp?, sqrt?

// Operations for integral types only
// - binary op: and, or, xor, gcd(ext)?
// - unary op: flip
// - bit op: lsl, lsr, asr, lsb, msb
// - count op: length, clz, ctz, popcount, scan0/1?

// Operations for real types only
// - constant op: pinf, ninf, nan, nar
// - inspect op: sign, sigificants, exponent, scale


#endif // MLIR_DIALECT_PRECISION_IR_PRECISIONOPS
