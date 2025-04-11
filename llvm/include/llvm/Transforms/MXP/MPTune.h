#ifndef LLVM_TRANSFORMS_PRECISION_H
#define LLVM_TRANSFORMS_PRECISION_H

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"
#include "llvm/Transforms/Utils/InstructionWorklist.h"
#include "llvm/Transforms/Utils/Local.h"

#include <cassert>

#include "llvm/Transforms/Utils/InstructionWorklist.h"

using namespace llvm::PatternMatch;

namespace llvm {

class MptunePass : public PassInfoMixin<MptunePass> {
public:
  InstructionWorklist Worklist;
  InstCombineOptions Options;
  explicit MptunePass(InstCombineOptions Opts = {});
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  Instruction *visit(Instruction &I);
};

class LLVM_LIBRARY_VISIBILITY MptuneImpl final
    : public InstCombiner,
      public InstVisitor<MptuneImpl, Instruction *> {
public:
  MptuneImpl(InstructionWorklist &Worklist, BuilderTy &Builder,
             bool MinimizeSize, AAResults *AA, AssumptionCache &AC,
             TargetLibraryInfo &TLI, TargetTransformInfo &TTI,
             DominatorTree &DT, OptimizationRemarkEmitter &ORE,
             BlockFrequencyInfo *BFI, ProfileSummaryInfo *PSI,
             const DataLayout &DL, LoopInfo *LI)
      : InstCombiner(Worklist, Builder, MinimizeSize, AA, AC, TLI, TTI, DT, ORE,
                     BFI, PSI, DL, LI) {}

  virtual ~MptuneImpl() = default;

  /// Run the combiner over the entire worklist until it is empty.
  ///
  /// \returns true if the IR is changed.
  bool run();

  Instruction *eraseInstFromFunction(Instruction &I) override {
    assert(I.use_empty() && "Cannot erase instruction that is used!");
    salvageDebugInfo(I);

    // Make sure that we reprocess all operands now that we reduced their
    // use counts.
    SmallVector<Value *> Ops(I.operands());
    Worklist.remove(&I);
    I.eraseFromParent();
    for (Value *Op : Ops)
      Worklist.handleUseCountDecrement(Op);
    MadeIRChange = true;
    return nullptr; // Don't do anything with FI
  }
  bool SimplifyDemandedBits(Instruction *I, unsigned Op,
                            const APInt &DemandedMask, KnownBits &Known,
                            unsigned Depth = 0) override;
  Value *SimplifyDemandedVectorElts(Value *V, APInt DemandedElts,
                                    APInt &UndefElts, unsigned Depth = 0,
                                    bool AllowMultipleUsers = false) override;
  Value *SimplifyDemandedUseBits(Value *V, APInt DemandedMask, KnownBits &Known,
                                 unsigned Depth, Instruction *CxtI);
  Value *SimplifyMultipleUseDemandedBits(Instruction *I,
                                         const APInt &DemandedMask,
                                         KnownBits &Known, unsigned Depth,
                                         Instruction *CxtI);
  Value *simplifyShrShlDemandedBits(Instruction *Shr, const APInt &ShrOp1,
                                    Instruction *Shl, const APInt &ShlOp1,
                                    const APInt &DemandedMask,
                                    KnownBits &Known);
  bool tryToSinkInstruction(Instruction *I, BasicBlock *DestBlock);
  Instruction *visit2peephole(Instruction &I);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_PRECISION_H
