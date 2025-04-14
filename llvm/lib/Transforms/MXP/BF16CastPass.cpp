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
#include "llvm/Transforms/MXP/BF16CastPass.h"

#define CT1(I, s) ChangeInstToCall(I, s, {I->getOperand(0)}, I->getType())

template <typename T>
inline void BF16CastPass::ChangeInstToCall(T *I, const std::string &FunctionName, const std::vector<Value*> &Args, Type *ReturnType) {
    Instruction *Inst = dyn_cast<Instruction>(I);
    errs() << "Instruction : " << *Inst << "\n";
    Module *M = Inst->getModule();
    IRBuilder<> Builder(I);

    std::vector<Value*> newArgs;
    if (!(dyn_cast<StoreInst>(I))) {
        for (unsigned i = 0; i < Args.size(); i++) {
            auto *arg = Inst->getOperand(i);
	    errs() << "Old operand " << i << " : " << *arg << "\n";
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
		errs() << "New Alloca : " << *alloca << "\n";
		errs() << "New trunCall : " << *callResult << "\n";

		LoadInst *load = Builder.CreateLoad(arg->getType(), alloca);	
		errs() << "Load : " << *load << "\n";
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
    errs() << "New Function : " << *Callee << "\n";

    //CallInst *Call = Builder.CreateCall(Callee, Args);
    CallInst *Call = Builder.CreateCall(Callee, newArgs);
    Call->setDebugLoc(I->getDebugLoc());

    I->replaceAllUsesWith(Call);
    erase.push_back(I);
}

bool BF16CastPass::runOnModule(Module &M) {
    for (auto &F : M) {
        for (auto &BB : F) {
            for (auto &I : BB) {
		if (auto *CI = dyn_cast<CastInst>(&I)) {
			//Type *SrcType = CI->getOperand(0)->getType();
			//Type *DstType = CI->getType();
			Type *SrcType = CI->getSrcTy();
			Type *DstType = CI->getDestTy();
			errs() << "Cast Instruction : " << I << "\n";
			errs() << "Src Type : " << *SrcType << "\n";
			errs() << "Dst Type : " << *DstType << "\n";

			switch(CI->getOpcode()) {
				case Instruction::FPTrunc:
					if (SrcType->isFloatTy() && DstType->isBFloatTy()) {
						CT1(CI, "FP32ToBF16");
					} else if (SrcType->isDoubleTy() && DstType->isBFloatTy()) {
						CT1(CI, "FP64ToBF16");
					} else if (SrcType->isFP128Ty() && DstType->isBFloatTy()) {
						CT1(CI, "FP128ToBF16");
					} else {
						errs() << "Don't neet to change.\n";
					}
					break;

				case Instruction::FPExt:
					if (SrcType->isBFloatTy() && DstType->isFloatTy()) {
						CT1(CI, "BF16ToFP32");
					} else if (SrcType->isBFloatTy() && DstType->isDoubleTy()) {
						CT1(CI, "BF16ToFP64");
					} else if (SrcType->isBFloatTy() && DstType->isFP128Ty()) {
						CT1(CI, "BF16ToFP128");
					} else {
						errs() << "Don't neet to change.\n";
					}
					break;

				default:
					// BitCast Trunc ZExt SExt PtrToInt IntToPtr 
					break;

			}	
		}

	        errs() << " BB : " << BB << "\n";
            }
        }
    }

    for(unsigned i = 0; i < erase.size(); i++) {
        erase[i]->eraseFromParent();
    }
    erase.clear();

    errs() << "END M : \n" << M << "\n";
    return true;
}

PreservedAnalyses BF16CastPass::run(Module &M, ModuleAnalysisManager &) {
    errs() << ">>> Perci-Tuner: Running BF16Cast Pass...\n";
    if (!runOnModule(M))
        return PreservedAnalyses::all();
    return PreservedAnalyses::none();
}

namespace llvm {
    cl::opt<bool> EnableBF16ToFP32("bf16tofp32",
                               cl::desc("Convert bf16 ops to fp32 ops"),
                               cl::init(false));

void addBF16ToFP32Pass(ModulePassManager &MPM, OptimizationLevel Level) {
  MPM.addPass(BF16CastPass());
}
} // namespace llvm
