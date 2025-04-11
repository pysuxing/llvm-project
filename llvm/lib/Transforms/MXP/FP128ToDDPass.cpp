#include "llvm/IR/IRBuilder.h"
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include "llvm/IR/Constant.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/MXP/FP128ToDDPass.h"

#define CF1(I, s) ChangeInstToCall(I, s, {I->getOperand(0)}, I->getType())
#define CF2(I, s) ChangeInstToCall(I, s, {I->getOperand(0), I->getOperand(1)}, I->getType())
#define CF3(I, s) ChangeInstToCall(I, s, {I->getOperand(0), I->getOperand(1), I->getOperand(2)}, I->getType())
#define CB2(I, s) ChangeInstToCall(I, s, {I->getOperand(0), I->getOperand(1)}, Type::getInt1Ty(M.getContext()))
#define CT1(I, s) ChangeInstToCall(I, s, {I->getOperand(0)}, I->getType())

template <typename T>
inline void FP128ToDDPass::ChangeInstToCall(T *I, const std::string &FunctionName, const std::vector<Value*> &Args, Type *ReturnType) {
    Instruction *Inst = dyn_cast<Instruction>(I);
#ifdef DEBUG
    errs() << "Instruction : " << *Inst << "\n";
#endif
    Module *M = Inst->getModule();
    IRBuilder<> Builder(I);

    std::vector<Value*> newArgs;
    if (!(dyn_cast<StoreInst>(I))) {
        for (unsigned i = 0; i < Args.size(); i++) {
            auto *arg = Inst->getOperand(i);
#ifdef DEBUG
	    errs() << "Old operand " << i << " : " << *arg << "\n";
#endif
	    auto *constant = dyn_cast<Constant>(arg);
            if ( constant && arg->getType()->isFP128Ty()) {//如果函数参数中有fp128常量，先转成dd
                Function *FP128ToDD = M->getFunction("FP128ToDD");
                if (!FP128ToDD) {
                    Type *fp128Type = Type::getFP128Ty(M->getContext());
                    FunctionType *funType = FunctionType::get(Type::getVoidTy(M->getContext()), {fp128Type, fp128Type->getPointerTo()}, false);
            	    FP128ToDD = Function::Create(funType, Function::ExternalLinkage, "FP128ToDD", M);
                }
                AllocaInst *alloca = Builder.CreateAlloca(constant->getType());
                Value *callResult = Builder.CreateCall(FP128ToDD, {constant, alloca});
#ifdef DEBUG
		errs() << "New Alloca : " << *alloca << "\n";
		errs() << "New trunCall : " << *callResult << "\n";
#endif
		LoadInst *load = Builder.CreateLoad(arg->getType(), alloca);	
#ifdef DEBUG
		errs() << "Load : " << *load << "\n";
#endif
		newArgs.push_back(load);
                //Inst->setOperand(i, alloca);
    		//errs() << "Instruction : " << *Inst << "\n";
            } else {
	    	newArgs.push_back(arg);
	    }
        }
    } else {
    	newArgs = Args;
    }

    std::vector<Type*> aType;
    for (auto *arg : Args) {
    	aType.push_back(arg->getType());
    }
    FunctionType *FuncTy = FunctionType::get(ReturnType, aType, false);
    Function *Callee = M->getFunction(FunctionName);
    if (!Callee || Callee->getFunctionType() != FuncTy) {
    	Callee = Function::Create(FuncTy, Function::ExternalLinkage, FunctionName, M);
    }
    //Function *Callee = getOrCreateFunction(M, FunctionName, Args, ReturnType);
#ifdef DEBUG
    errs() << "New Function : " << *Callee << "\n";
#endif
    //CallInst *Call = Builder.CreateCall(Callee, Args);
    CallInst *Call = Builder.CreateCall(Callee, newArgs);
    Call->setDebugLoc(I->getDebugLoc());

    I->replaceAllUsesWith(Call);
    erase.push_back(I);
}

inline std::string getBOName(BinaryOperator *bo) {
	switch (bo->getOpcode()) {
		case Instruction::FAdd:
			return "fp128_add";
		case Instruction::FSub:
			return "fp128_sub";
		case Instruction::FMul:
			return "fp128_mul";
		case Instruction::FDiv:
			return "fp128_div";
		default:
			//errs() << "orther Binary Operator " << *bo << ".\n";
			return "";
	}
}

//template <typename T>
//inline void getFuncByI(T *I, string s) {
//    errs() << "s : " << s.c_str() << "\n";
//    IRBuilder<> Builder(I);
//    Function *Func = Builder.GetInsertBlock()->getModule()->getFunction(s);
//    errs() << "Func : " << Func << " from " << *I << "\n";
//    errs() << "O0 : " << *(I->getOperand(0)) << "\n";
//    errs() << "O1 : " << *(I->getOperand(1)) << "\n";
//    if (!Func) {
//        Func = Function::Create(
//        FunctionType::get(I->getType(), {I->getType(), I->getType()}, false),
//        Function::ExternalLinkage, s, Builder.GetInsertBlock()->getModule());
//        errs() << "new Func : " << Func << "\n";
//    }
//
//    // Create the function call
//    CallInst *Call = Builder.CreateCall(Func, {I->getOperand(0), I->getOperand(1)});
//    Call->setDebugLoc(I->getDebugLoc());
//
//    // Replace the original instruction with the function call
//    I->replaceAllUsesWith(Call);
//    erase.push_back(I);
//}

bool FP128ToDDPass::runOnModule(Module &M) {
    for (auto &F : M) {
        for (auto &BB : F) {
            for (auto &I : BB) {
                if (auto *SI = dyn_cast<StoreInst>(&I)) {
                    if (SI->getValueOperand()->getType()->isFP128Ty()) {
                        // Get the fp128 constant
                        auto *FP128Const = dyn_cast<ConstantFP>(SI->getValueOperand());
                        if (!FP128Const) continue;
			//errs() << "const fp128 store : " << *SI << "\n";
			//errs() << "Op0 : " << *(SI->getValueOperand()) << "\n";
			//errs() << "Op1 : " << *(SI->getPointerOperand()) << "\n";
			ChangeInstToCall(SI, "FP128ToDD", {SI->getValueOperand(), SI->getPointerOperand()}, SI->getType());
            
                        //// Convert fp128 to double
                        //APFloat fp128Val(FP128Const->getValueAPF());
                        //if (fp128Val.isNaN() || fp128Val.isInfinity()) {
                        //    // 处理异常情况
                        //    errs() << "fp128 constant get error\n";
                        //    return false;
                        //}
                        //APFloat doubleVal(fp128Val);
                        //bool losesInfo = false;
			////此处可能存在精度损失
                        //doubleVal.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
            
                        //// Create a new double constant
                        //Constant *DoubleConst = ConstantFP::get(
                        //    Type::getDoubleTy(F.getContext()), doubleVal);
            
                        //// Replace the fp128 constant with the double constant
                        //SI->setOperand(0, DoubleConst);
                    }
                } else if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) { // 处理计算指令
		    std::string funName = getBOName(BinOp);
		    if (BinOp->getType()->isFP128Ty() && !funName.empty()) {//处理fp128的add,sub,mul,div
			for (unsigned i = 0; i < BinOp->getNumOperands(); ++i) {
			    if (BinOp->getOperand(i)->getType()->isFP128Ty()) {
			    	funName += "2";
			    } else if (BinOp->getOperand(i)->getType()->isDoubleTy()) {
			    	funName += "1";
			    }
			}
			CF2(BinOp, funName);
		    }
                } else if (I.getOpcode() == Instruction::FNeg) {
			Instruction *II = &I;
			CF1(II, "fp128_neg");	
		} else if (auto *FCmp = dyn_cast<FCmpInst>(&I)) { // 处理比较指令
                    if (FCmp->getOperand(0)->getType()->isFP128Ty()) {
                    	switch (FCmp->getPredicate()) {
                    	    case FCmpInst::FCMP_OGT:
                    	    case FCmpInst::FCMP_UGT:
                    	        CB2(FCmp, "fp128_gt");
                    	        break;
                    	    case FCmpInst::FCMP_OGE:
                    	    case FCmpInst::FCMP_UGE:
                    	        CB2(FCmp, "fp128_ge");
                    	        break;
                    	    case FCmpInst::FCMP_OLT:
                    	    case FCmpInst::FCMP_ULT:
				CB2(FCmp, "fp128_lt");
                    	        break;
                    	    case FCmpInst::FCMP_OLE:
                    	    case FCmpInst::FCMP_ULE:
                    	        CB2(FCmp, "fp128_le");
                    	        break;
                    	    case FCmpInst::FCMP_OEQ:
                    	    case FCmpInst::FCMP_UEQ:
                    	        CB2(FCmp, "fp128_eq");
                    	        break;
                    	    case FCmpInst::FCMP_ONE:
                    	    case FCmpInst::FCMP_UNE:
                    	        CB2(FCmp, "fp128_ne");
                    	        break;
                    	    default:
                    	        continue;
                    	}
		    }
                } else if (auto *FCall = dyn_cast<CallInst>(&I)) { //处理函数调用
		    //errs() << "Call I : " << *FCall << "\n";	
		    auto *Callee = FCall->getCalledFunction();
		    if (Callee) {
		    	if (Callee->getReturnType()->isFP128Ty()) {
			    //errs() << "FP128 Call : " << I << "\n";
			    if (Callee->getName() == "llvm.fmuladd.f128") {
#ifdef DEBUG
			    	errs() << "This call (" << I << ") need to deal." << "\n"; 
#endif
		    		std::string funName = "fp128_muladd";
		    		for (unsigned i = 0; i < FCall->getNumOperands(); ++i) {
		    		    if (FCall->getOperand(i)->getType()->isFP128Ty()) {
		    		    	funName += "2";
		    		    } else if (FCall->getOperand(i)->getType()->isDoubleTy()) {
		    		    	funName += "1";
		    		    }
		    		}
				CF3(FCall, funName);
			    } else if (Callee->getName() == "sinl") {
			    	CF1(FCall, "fp128_sin");
			    } else if (Callee->getName() == "acosl") {
			    	CF1(FCall, "fp128_acos");
			    } else if (Callee->getName() == "sqrtl") {
			    	CF1(FCall, "fp128_sqrt");
			    } else if (Callee->getName() == "expl") {
				CF1(FCall, "fp128_exp");
			    }
			}
			if (Callee->getName() == "printf" && Callee->getFunctionType()->isVarArg()) {
#ifdef DEBUG
				errs() << "Call printf\n";
#endif
				//FunctionType *FT = Callee->getFunctionType();
				//errs() << "Call type : " << *FT << "\n";
				//errs() << "Call param num : " << FT->getNumParams() << "\n";
				//errs() << "Call num : " << FCall->getNumOperands() << "\n";
				//bool hasFP128Pa = false;
				for(unsigned i = 0; i < FCall->getNumOperands(); ++i) {
					auto *pa = FCall->getOperand(i);
#ifdef DEBUG
					errs() << "printf param : " << *pa << "\n";
#endif
					//errs() << "This printf : " << *FCall << " need to deal.\n";
					if (pa->getType()->isFP128Ty()) {
						//errs() << "fp128\n";
						IRBuilder<> Builder(&I);
#if 0
						Function *Fun = M.getFunction("DDToFP128");
						if (!Fun) {
							Fun = Function::Create(
									FunctionType::get(pa->getType(), {pa->getType()}, false),
									Function::ExternalLinkage, "DDToFP128", &M);
						}

						auto *newPa = Builder.CreateCall(Fun, pa);
						FCall->setArgOperand(i, newPa);
#else 
						Function *Fun = M.getFunction("fp128_printdd");
						if (!Fun) {
							Fun = Function::Create(
									FunctionType::get(Type::getInt32Ty(M.getContext()), {pa->getType()}, false),
									Function::ExternalLinkage, "fp128_printdd", &M);
						}

						auto *newPa = Builder.CreateCall(Fun, pa);
						newPa->setDebugLoc(I.getDebugLoc());
						
						I.replaceAllUsesWith(newPa);
						erase.push_back(&I);


#endif

						//ChangeInstToCall(FCall, "DDToFP128", {I->getOperand(0)}, I->getType());
						//if (auto *ap = dyn_cast<Instruction>(pa)) {
						//	errs() << "ap : " << *ap << "\n";
						//	auto *t = dyn_cast<Instruction>(ap->getOperand(0));
						//	//errs() << "op(0) : " << *(ap->getOperand(0)) << "\n";
						//	errs() << "op(0) : " << *(t-getType()) << "\n";
						//}
						//hasFP128Pa = true;
						break;
					} else if (pa->getType()->isDoubleTy()) {
						//errs() << "fp64\n";
						//IRBuilder<> Builder(&I);
						//Function *Fun = M.getFunction("fp128_printd");
						//if (!Fun) {
                                                //        Fun = Function::Create(
                                                //                        FunctionType::get(Type::getVoidTy(M.getContext()), {pa->getType()}, false),
                                                //                        Function::ExternalLinkage, "fp128_printd", &M);
                                                //}

						//auto *newPa = Builder.CreateCall(Fun, pa);
						//errs() << "old I : " << I << "\n";
						//errs() << "new func : " << *newPa << "\n";
                                                //newPa->setDebugLoc(I.getDebugLoc());

                                                //I.replaceAllUsesWith(newPa);
                                                //erase.push_back(&I);

						//break;
					} else if (pa->getType()->isFloatTy()) {
						//errs() << "fp32\n";
						//IRBuilder<> Builder(&I);
                                                //Function *Fun = M.getFunction("fp128_printf");
                                                //if (!Fun) {
                                                //        Fun = Function::Create(
                                                //                        FunctionType::get(Type::getVoidTy(M.getContext()), {pa->getType()}, false),
                                                //                        Function::ExternalLinkage, "fp128_printf", &M);
                                                //}

                                                //auto *newPa = Builder.CreateCall(Fun, pa);
                                                //newPa->setDebugLoc(I.getDebugLoc());

                                                //I.replaceAllUsesWith(newPa);
                                                //erase.push_back(&I);

                                                //break;
					}
				}
				//if (hasFP128Pa) {
				//	errs() << "This printf : " << *FCall << " need to deal.\n";
				//	
				//}
			}
		    }
		} else if (auto *CI = dyn_cast<CastInst>(&I)) {
			//Type *SrcType = CI->getOperand(0)->getType();
			//Type *DstType = CI->getType();
			Type *SrcType = CI->getSrcTy();
			Type *DstType = CI->getDestTy();
#ifdef DEBUG
			errs() << "Cast Instruction : " << I << "\n";
			errs() << "Src Type : " << *SrcType << "\n";
			errs() << "Dst Type : " << *DstType << "\n";
#endif
			switch(CI->getOpcode()) {
				case Instruction::FPTrunc:
					if (SrcType->isFP128Ty() && DstType->isDoubleTy()) {
						CT1(CI, "DDToDouble");
					} else if (SrcType->isFP128Ty() && DstType->isFloatTy()) {
						CT1(CI, "DDToFloat");
					} /*else if (SrcType->isFloatTy() && DstType->isBFloatTy()) {
						CT1(CI, "FP32ToBF16");
					} else if (SrcType->isDoubleTy() && DstType->isBFloatTy()) {
						CT1(CI, "FP64ToBF16");
					}*/ else {
						//errs() << "Don't neet to change.\n";
					}
					break;

				case Instruction::FPExt:
					if (SrcType->isDoubleTy() && DstType->isFP128Ty()) {
						CT1(CI, "DoubleToDD");
					} else if (SrcType->isFloatTy() && DstType->isFP128Ty()) {
						CT1(CI, "FloatToDD");
					} /*else if (SrcType->isBFloatTy() && DstType->isFloatTy()) {
						CT1(CI, "BF16ToFP32");
					} else if (SrcType->isBFloatTy() && DstType->isDoubleTy()) {
						CT1(CI, "BF16ToFP64");
					} */else {
						//errs() << "Don't neet to change.\n";
					}
					break;

				case Instruction::FPToUI:
					if (SrcType->isFP128Ty() && DstType->isIntegerTy()) {
						if (auto *intTy = dyn_cast<IntegerType>(DstType)){
							unsigned bw = intTy->getIntegerBitWidth();
							if (bw == 32) {
								CT1(CI, "DDToU32");
							} else if (bw == 64) {
								CT1(CI, "DDToU64");
							}
						}
					}
					break;

				case Instruction::FPToSI:
					if (SrcType->isFP128Ty() && DstType->isIntegerTy()) {
						if (auto *intTy = dyn_cast<IntegerType>(DstType)){
							unsigned bw = intTy->getIntegerBitWidth();
							if (bw == 32) {
								CT1(CI, "DDToS32");
							} else if (bw == 64) {
								CT1(CI, "DDToS64");
							}
						}
					}
					break;

				case Instruction::UIToFP:
					if (SrcType->isIntegerTy() && DstType->isFP128Ty()) {
						if (auto *intTy = dyn_cast<IntegerType>(SrcType)){
							unsigned bw = intTy->getIntegerBitWidth();
							if (bw == 32) {
								CT1(CI, "U32ToDD");
							} else if (bw == 64) {
								CT1(CI, "U64ToDD");
							}
						}
					}
					break;

				case Instruction::SIToFP:
					if (SrcType->isIntegerTy() && DstType->isFP128Ty()) {
						if (auto *intTy = dyn_cast<IntegerType>(SrcType)){
							unsigned bw = intTy->getIntegerBitWidth();
							if (bw == 32) {
								CT1(CI, "S32ToDD");
							} else if (bw == 64) {
								CT1(CI, "S64ToDD");
							}
						}
					}
					break;

				default:
					// BitCast Trunc ZExt SExt PtrToInt IntToPtr 
					break;

			}	
		}
#ifdef DEBUG
	        errs() << " BB : " << BB << "\n";
#endif
            }
        }
    }

    for(unsigned i = 0; i < erase.size(); i++) {
        erase[i]->eraseFromParent();
    }
    erase.clear();
#ifdef DEBUG
    errs() << "END M : \n" << M << "\n";
#endif
    return true;
}

PreservedAnalyses FP128ToDDPass::run(Module &M, ModuleAnalysisManager &) {
#ifdef DEBUG
    errs() << ">>> Perci-Tuner: Running FP128ToDD Pass...\n";
#endif
    if (!runOnModule(M))
        return PreservedAnalyses::all();
    return PreservedAnalyses::none();
}
