#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LazyBlockFrequencyInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/Utils/Local.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/MXP/MPTune.h"
#include "llvm/Transforms/Utils/InstructionWorklist.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <optional>

#define DEBUG_TYPE "mptune"

using namespace llvm;
using namespace llvm::PatternMatch;

STATISTIC(NumWorklistIterations,
          "Number of instruction combining iterations performed");
STATISTIC(NumOneIteration, "Number of functions with one iteration");
STATISTIC(NumTwoIterations, "Number of functions with two iterations");
STATISTIC(NumThreeIterations, "Number of functions with three iterations");
STATISTIC(NumFourOrMoreIterations,
          "Number of functions with four or more iterations");

STATISTIC(NumCombined , "Number of insts combined");
STATISTIC(NumConstProp, "Number of constant folds");
STATISTIC(NumDeadInst , "Number of dead inst eliminated");
STATISTIC(NumSunkInst , "Number of instructions sunk");
STATISTIC(NumExpand,    "Number of expansions");
STATISTIC(NumFactor   , "Number of factorizations");
STATISTIC(NumReassoc  , "Number of reassociations");
DEBUG_COUNTER(VisitCounter, "instcombine-visit",
              "Controls which instructions are visited");


#ifndef NDEBUG
static constexpr unsigned InstCombineDefaultInfiniteLoopThreshold = 100;
#else
static constexpr unsigned InstCombineDefaultInfiniteLoopThreshold = 1000;
#endif

static cl::opt<bool>
EnableCodeSinking("pinstcombine-code-sinking", cl::desc("Enable code sinking"),
                                              cl::init(true));

static cl::opt<unsigned> MaxSinkNumUsers(
    "pinstcombine-max-sink-users", cl::init(32),
    cl::desc("Maximum number of undroppable users for instruction sinking"));

static cl::opt<unsigned> InfiniteLoopDetectionThreshold(
    "pinstcombine-infinite-loop-threshold",
    cl::desc("Number of instruction combining iterations considered an "
             "infinite loop"),
    cl::init(InstCombineDefaultInfiniteLoopThreshold), cl::Hidden);

static cl::opt<unsigned>
MaxArraySize("pinstcombine-maxarray-size", cl::init(1024),
             cl::desc("Maximum array size considered when doing a combine"));

// FIXME: Remove this flag when it is no longer necessary to convert
// llvm.dbg.declare to avoid inaccurate debug info. Setting this to false
// increases variable availability at the cost of accuracy. Variables that
// cannot be promoted by mem2reg or SROA will be described as living in memory
// for their entire lifetime. However, passes like DSE and instcombine can
// delete stores to the alloca, leading to misleading and inaccurate debug
// information. This flag can be removed when those passes are fixed.
static cl::opt<unsigned> ShouldLowerDbgDeclare("pinstcombine-lower-dbg-declare",
                                               cl::Hidden, cl::init(true));

static bool prepareICWorklistFromFunction(Function &F, const DataLayout &DL,
                                          const TargetLibraryInfo *TLI,
                                          InstructionWorklist &ICWorklist);

static bool combineInstructionsOverFunction(
    Function &F, InstructionWorklist &Worklist, AliasAnalysis *AA,
    AssumptionCache &AC, TargetLibraryInfo &TLI, TargetTransformInfo &TTI,
    DominatorTree &DT, OptimizationRemarkEmitter &ORE, BlockFrequencyInfo *BFI,
    ProfileSummaryInfo *PSI, unsigned MaxIterations, LoopInfo *LI){
  auto &DL = F.getParent()->getDataLayout();

  /// Builder - This is an IRBuilder that automatically inserts new
  /// instructions into the worklist when they are created.
  IRBuilder<TargetFolder, IRBuilderCallbackInserter> Builder(
      F.getContext(), TargetFolder(DL),
      IRBuilderCallbackInserter([&Worklist, &AC](Instruction *I) {
        Worklist.add(I);
        if (auto *Assume = dyn_cast<AssumeInst>(I))
          AC.registerAssumption(Assume);
      }));

  // Lower dbg.declare intrinsics otherwise their value may be clobbered
  // by instcombiner.
  bool MadeIRChange = false;
  if (ShouldLowerDbgDeclare)
    MadeIRChange = LowerDbgDeclare(F);

  // Iterate while there is work to do.
  unsigned Iteration = 0;
  while (true) {
    ++NumWorklistIterations;
    ++Iteration;

    if (Iteration > InfiniteLoopDetectionThreshold) {
      report_fatal_error(
          "Instruction Combining seems stuck in an infinite loop after " +
          Twine(InfiniteLoopDetectionThreshold) + " iterations.");
    }

    if (Iteration > MaxIterations) {
      LLVM_DEBUG(dbgs() << "\n\n[IC] Iteration limit #" << MaxIterations
                        << " on " << F.getName()
                        << " reached; stopping before reaching a fixpoint\n");
      break;
    }

    LLVM_DEBUG(dbgs() << "\n\nINSTCOMBINE ITERATION #" << Iteration << " on "
                      << F.getName() << "\n");

    MadeIRChange |= prepareICWorklistFromFunction(F, DL, &TLI, Worklist);

    MptuneImpl IC(Worklist, Builder, F.hasMinSize(), AA, AC, TLI, TTI, DT,
                        ORE, BFI, PSI, DL, LI);
    IC.MaxArraySizeForCombine = MaxArraySize;

    if (!IC.run())
      break;

    MadeIRChange = true;
  }

  if (Iteration == 1)
    ++NumOneIteration;
  else if (Iteration == 2)
    ++NumTwoIterations;
  else if (Iteration == 3)
    ++NumThreeIterations;
  else
    ++NumFourOrMoreIterations;

  return MadeIRChange;
}
static bool ShrinkDemandedConstant(Instruction *I, unsigned OpNo,
                                   const APInt &Demanded) {
  assert(I && "No instruction?");
  assert(OpNo < I->getNumOperands() && "Operand index too large");

  // The operand must be a constant integer or splat integer.
  Value *Op = I->getOperand(OpNo);
  const APInt *C;
  if (!match(Op, m_APInt(C)))
    return false;

  // If there are no bits set that aren't demanded, nothing to do.
  if (C->isSubsetOf(Demanded))
    return false;

  // This instruction is producing bits that are not demanded. Shrink the RHS.
  I->setOperand(OpNo, ConstantInt::get(Op->getType(), *C & Demanded));

  return true;
}
class AliasScopeTracker {
  SmallPtrSet<const MDNode *, 8> UsedAliasScopesAndLists;
  SmallPtrSet<const MDNode *, 8> UsedNoAliasScopesAndLists;

public:
  void analyse(Instruction *I) {
    // This seems to be faster than checking 'mayReadOrWriteMemory()'.
    if (!I->hasMetadataOtherThanDebugLoc())
      return;

    auto Track = [](Metadata *ScopeList, auto &Container) {
      const auto *MDScopeList = dyn_cast_or_null<MDNode>(ScopeList);
      if (!MDScopeList || !Container.insert(MDScopeList).second)
        return;
      for (const auto &MDOperand : MDScopeList->operands())
        if (auto *MDScope = dyn_cast<MDNode>(MDOperand))
          Container.insert(MDScope);
    };

    Track(I->getMetadata(LLVMContext::MD_alias_scope), UsedAliasScopesAndLists);
    Track(I->getMetadata(LLVMContext::MD_noalias), UsedNoAliasScopesAndLists);
  }

  bool isNoAliasScopeDeclDead(Instruction *Inst) {
    NoAliasScopeDeclInst *Decl = dyn_cast<NoAliasScopeDeclInst>(Inst);
    if (!Decl)
      return false;

    assert(Decl->use_empty() &&
           "llvm.experimental.noalias.scope.decl in use ?");
    const MDNode *MDSL = Decl->getScopeList();
    assert(MDSL->getNumOperands() == 1 &&
           "llvm.experimental.noalias.scope should refer to a single scope");
    auto &MDOperand = MDSL->getOperand(0);
    if (auto *MD = dyn_cast<MDNode>(MDOperand))
      return !UsedAliasScopesAndLists.contains(MD) ||
             !UsedNoAliasScopesAndLists.contains(MD);

    // Not an MDNode ? throw away.
    return true;
  }
};

static bool prepareICWorklistFromFunction(Function &F, const DataLayout &DL,
                                          const TargetLibraryInfo *TLI,
                                          InstructionWorklist &ICWorklist) {
  bool MadeIRChange = false;
  SmallPtrSet<BasicBlock *, 32> Visited;
  SmallVector<BasicBlock*, 256> Worklist;
  Worklist.push_back(&F.front());

  SmallVector<Instruction *, 128> InstrsForInstructionWorklist;
  DenseMap<Constant *, Constant *> FoldedConstants;
  AliasScopeTracker SeenAliasScopes;

  do {
    BasicBlock *BB = Worklist.pop_back_val();

    // We have now visited this block!  If we've already been here, ignore it.
    if (!Visited.insert(BB).second)
      continue;

    for (Instruction &Inst : llvm::make_early_inc_range(*BB)) {
      // ConstantProp instruction if trivially constant.
      if (!Inst.use_empty() &&
          (Inst.getNumOperands() == 0 || isa<Constant>(Inst.getOperand(0))))
        if (Constant *C = ConstantFoldInstruction(&Inst, DL, TLI)) {
          LLVM_DEBUG(dbgs() << "IC: ConstFold to: " << *C << " from: " << Inst
                            << '\n');
          Inst.replaceAllUsesWith(C);
          ++NumConstProp;
          if (isInstructionTriviallyDead(&Inst, TLI))
            Inst.eraseFromParent();
          MadeIRChange = true;
          continue;
        }

      // See if we can constant fold its operands.
      for (Use &U : Inst.operands()) {
        if (!isa<ConstantVector>(U) && !isa<ConstantExpr>(U))
          continue;

        auto *C = cast<Constant>(U);
        Constant *&FoldRes = FoldedConstants[C];
        if (!FoldRes)
          FoldRes = ConstantFoldConstant(C, DL, TLI);

        if (FoldRes != C) {
          LLVM_DEBUG(dbgs() << "IC: ConstFold operand of: " << Inst
                            << "\n    Old = " << *C
                            << "\n    New = " << *FoldRes << '\n');
          U = FoldRes;
          MadeIRChange = true;
        }
      }

      // Skip processing debug and pseudo intrinsics in InstCombine. Processing
      // these call instructions consumes non-trivial amount of time and
      // provides no value for the optimization.
      if (!Inst.isDebugOrPseudoInst()) {
        InstrsForInstructionWorklist.push_back(&Inst);
        SeenAliasScopes.analyse(&Inst);
      }
    }

    // Recursively visit successors.  If this is a branch or switch on a
    // constant, only visit the reachable successor.
    Instruction *TI = BB->getTerminator();
    if (BranchInst *BI = dyn_cast<BranchInst>(TI); BI && BI->isConditional()) {
      if (isa<UndefValue>(BI->getCondition()))
        // Branch on undef is UB.
        continue;
      if (auto *Cond = dyn_cast<ConstantInt>(BI->getCondition())) {
        bool CondVal = Cond->getZExtValue();
        BasicBlock *ReachableBB = BI->getSuccessor(!CondVal);
        Worklist.push_back(ReachableBB);
        continue;
      }
    } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
      if (isa<UndefValue>(SI->getCondition()))
        // Switch on undef is UB.
        continue;
      if (auto *Cond = dyn_cast<ConstantInt>(SI->getCondition())) {
        Worklist.push_back(SI->findCaseValue(Cond)->getCaseSuccessor());
        continue;
      }
    }

    append_range(Worklist, successors(TI));
  } while (!Worklist.empty());

  // Remove instructions inside unreachable blocks. This prevents the
  // instcombine code from having to deal with some bad special cases, and
  // reduces use counts of instructions.
  for (BasicBlock &BB : F) {
    if (Visited.count(&BB))
      continue;

    unsigned NumDeadInstInBB;
    unsigned NumDeadDbgInstInBB;
    std::tie(NumDeadInstInBB, NumDeadDbgInstInBB) =
        removeAllNonTerminatorAndEHPadInstructions(&BB);

    MadeIRChange |= NumDeadInstInBB + NumDeadDbgInstInBB > 0;
    NumDeadInst += NumDeadInstInBB;
  }

  // Once we've found all of the instructions to add to instcombine's worklist,
  // add them in reverse order.  This way instcombine will visit from the top
  // of the function down.  This jives well with the way that it adds all uses
  // of instructions to the worklist after doing a transformation, thus avoiding
  // some N^2 behavior in pathological cases.
  ICWorklist.reserve(InstrsForInstructionWorklist.size());
  for (Instruction *Inst : reverse(InstrsForInstructionWorklist)) {
    // DCE instruction if trivially dead. As we iterate in reverse program
    // order here, we will clean up whole chains of dead instructions.
    if (isInstructionTriviallyDead(Inst, TLI) ||
        SeenAliasScopes.isNoAliasScopeDeclDead(Inst)) {
      ++NumDeadInst;
      LLVM_DEBUG(dbgs() << "IC: DCE: " << *Inst << '\n');
      salvageDebugInfo(*Inst);
      Inst->eraseFromParent();
      MadeIRChange = true;
      continue;
    }

    ICWorklist.push(Inst);
  }

  return MadeIRChange;
}

static bool SoleWriteToDeadLocal(Instruction *I, TargetLibraryInfo &TLI) {
  auto *CB = dyn_cast<CallBase>(I);
  if (!CB)
    // TODO: handle e.g. store to alloca here - only worth doing if we extend
    // to allow reload along used path as described below.  Otherwise, this
    // is simply a store to a dead allocation which will be removed.
    return false;
  std::optional<MemoryLocation> Dest = MemoryLocation::getForDest(CB, TLI);
  if (!Dest)
    return false;
  auto *AI = dyn_cast<AllocaInst>(getUnderlyingObject(Dest->Ptr));
  if (!AI)
    // TODO: allow malloc?
    return false;
  // TODO: allow memory access dominated by move point?  Note that since AI
  // could have a reference to itself captured by the call, we would need to
  // account for cycles in doing so.
  SmallVector<const User *> AllocaUsers;
  SmallPtrSet<const User *, 4> Visited;
  auto pushUsers = [&](const Instruction &I) {
    for (const User *U : I.users()) {
      if (Visited.insert(U).second)
        AllocaUsers.push_back(U);
    }
  };
  pushUsers(*AI);
  while (!AllocaUsers.empty()) {
    auto *UserI = cast<Instruction>(AllocaUsers.pop_back_val());
    if (isa<BitCastInst>(UserI) || isa<GetElementPtrInst>(UserI) ||
        isa<AddrSpaceCastInst>(UserI)) {
      pushUsers(*UserI);
      continue;
    }
    if (UserI == CB)
      continue;
    // TODO: support lifetime.start/end here
    return false;
  }
  return true;
}

MptunePass::MptunePass(InstCombineOptions Opts) : Options(Opts) {}

PreservedAnalyses MptunePass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
																				
	
	auto &AC = AM.getResult<AssumptionAnalysis>(F);
	auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
	auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
	auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);
	auto &TTI = AM.getResult<TargetIRAnalysis>(F);

	// TODO: Only use LoopInfo when the option is set. This requires that the
	//       callers in the pass pipeline explicitly set the option.
	auto *LI = AM.getCachedResult<LoopAnalysis>(F);
	if (!LI && Options.UseLoopInfo)
		LI = &AM.getResult<LoopAnalysis>(F);

	auto *AA = &AM.getResult<AAManager>(F);
	auto &MAMProxy = AM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
	ProfileSummaryInfo *PSI =
		MAMProxy.getCachedResult<ProfileSummaryAnalysis>(*F.getParent());
	auto *BFI = (PSI && PSI->hasProfileSummary()) ?
		&AM.getResult<BlockFrequencyAnalysis>(F) : nullptr;
	
	if (!combineInstructionsOverFunction(F, Worklist, AA, AC, TLI, TTI, DT, ORE,
                                       BFI, PSI, Options.MaxIterations, LI))
    // No changes, all analyses are preserved.
    return PreservedAnalyses::all();

  // Mark all the analyses that instcombine updates as preserved.
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;

}
Value *MptuneImpl::SimplifyMultipleUseDemandedBits(
    Instruction *I, const APInt &DemandedMask, KnownBits &Known, unsigned Depth,
    Instruction *CxtI) {
  unsigned BitWidth = DemandedMask.getBitWidth();
  Type *ITy = I->getType();

  KnownBits LHSKnown(BitWidth);
  KnownBits RHSKnown(BitWidth);

  // Despite the fact that we can't simplify this instruction in all User's
  // context, we can at least compute the known bits, and we can
  // do simplifications that apply to *just* the one user if we know that
  // this instruction has a simpler value in that context.
  switch (I->getOpcode()) {
  case Instruction::And: {
    computeKnownBits(I->getOperand(1), RHSKnown, Depth + 1, CxtI);
    computeKnownBits(I->getOperand(0), LHSKnown, Depth + 1, CxtI);
    Known = LHSKnown & RHSKnown;
    computeKnownBitsFromAssume(I, Known, Depth, SQ.getWithInstruction(CxtI));

    // If the client is only demanding bits that we know, return the known
    // constant.
    if (DemandedMask.isSubsetOf(Known.Zero | Known.One))
      return Constant::getIntegerValue(ITy, Known.One);

    // If all of the demanded bits are known 1 on one side, return the other.
    // These bits cannot contribute to the result of the 'and' in this context.
    if (DemandedMask.isSubsetOf(LHSKnown.Zero | RHSKnown.One))
      return I->getOperand(0);
    if (DemandedMask.isSubsetOf(RHSKnown.Zero | LHSKnown.One))
      return I->getOperand(1);

    break;
  }
  case Instruction::Or: {
    computeKnownBits(I->getOperand(1), RHSKnown, Depth + 1, CxtI);
    computeKnownBits(I->getOperand(0), LHSKnown, Depth + 1, CxtI);
    Known = LHSKnown | RHSKnown;
    computeKnownBitsFromAssume(I, Known, Depth, SQ.getWithInstruction(CxtI));

    // If the client is only demanding bits that we know, return the known
    // constant.
    if (DemandedMask.isSubsetOf(Known.Zero | Known.One))
      return Constant::getIntegerValue(ITy, Known.One);

    // We can simplify (X|Y) -> X or Y in the user's context if we know that
    // only bits from X or Y are demanded.
    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'or' in this context.
    if (DemandedMask.isSubsetOf(LHSKnown.One | RHSKnown.Zero))
      return I->getOperand(0);
    if (DemandedMask.isSubsetOf(RHSKnown.One | LHSKnown.Zero))
      return I->getOperand(1);

    break;
  }
  case Instruction::Xor: {
    computeKnownBits(I->getOperand(1), RHSKnown, Depth + 1, CxtI);
    computeKnownBits(I->getOperand(0), LHSKnown, Depth + 1, CxtI);
    Known = LHSKnown ^ RHSKnown;
    computeKnownBitsFromAssume(I, Known, Depth, SQ.getWithInstruction(CxtI));

    // If the client is only demanding bits that we know, return the known
    // constant.
    if (DemandedMask.isSubsetOf(Known.Zero | Known.One))
      return Constant::getIntegerValue(ITy, Known.One);

    // We can simplify (X^Y) -> X or Y in the user's context if we know that
    // only bits from X or Y are demanded.
    // If all of the demanded bits are known zero on one side, return the other.
    if (DemandedMask.isSubsetOf(RHSKnown.Zero))
      return I->getOperand(0);
    if (DemandedMask.isSubsetOf(LHSKnown.Zero))
      return I->getOperand(1);

    break;
  }
  case Instruction::Add: {
    unsigned NLZ = DemandedMask.countl_zero();
    APInt DemandedFromOps = APInt::getLowBitsSet(BitWidth, BitWidth - NLZ);

    // If an operand adds zeros to every bit below the highest demanded bit,
    // that operand doesn't change the result. Return the other side.
    computeKnownBits(I->getOperand(1), RHSKnown, Depth + 1, CxtI);
    if (DemandedFromOps.isSubsetOf(RHSKnown.Zero))
      return I->getOperand(0);

    computeKnownBits(I->getOperand(0), LHSKnown, Depth + 1, CxtI);
    if (DemandedFromOps.isSubsetOf(LHSKnown.Zero))
      return I->getOperand(1);

    bool NSW = cast<OverflowingBinaryOperator>(I)->hasNoSignedWrap();
    Known = KnownBits::computeForAddSub(/*Add*/ true, NSW, LHSKnown, RHSKnown);
    computeKnownBitsFromAssume(I, Known, Depth, SQ.getWithInstruction(CxtI));
    break;
  }
  case Instruction::Sub: {
    unsigned NLZ = DemandedMask.countl_zero();
    APInt DemandedFromOps = APInt::getLowBitsSet(BitWidth, BitWidth - NLZ);

    // If an operand subtracts zeros from every bit below the highest demanded
    // bit, that operand doesn't change the result. Return the other side.
    computeKnownBits(I->getOperand(1), RHSKnown, Depth + 1, CxtI);
    if (DemandedFromOps.isSubsetOf(RHSKnown.Zero))
      return I->getOperand(0);

    bool NSW = cast<OverflowingBinaryOperator>(I)->hasNoSignedWrap();
    computeKnownBits(I->getOperand(0), LHSKnown, Depth + 1, CxtI);
    Known = KnownBits::computeForAddSub(/*Add*/ false, NSW, LHSKnown, RHSKnown);
    computeKnownBitsFromAssume(I, Known, Depth, SQ.getWithInstruction(CxtI));
    break;
  }
  case Instruction::AShr: {
    // Compute the Known bits to simplify things downstream.
    computeKnownBits(I, Known, Depth, CxtI);

    // If this user is only demanding bits that we know, return the known
    // constant.
    if (DemandedMask.isSubsetOf(Known.Zero | Known.One))
      return Constant::getIntegerValue(ITy, Known.One);

    // If the right shift operand 0 is a result of a left shift by the same
    // amount, this is probably a zero/sign extension, which may be unnecessary,
    // if we do not demand any of the new sign bits. So, return the original
    // operand instead.
    const APInt *ShiftRC;
    const APInt *ShiftLC;
    Value *X;
    unsigned BitWidth = DemandedMask.getBitWidth();
    if (match(I,
              m_AShr(m_Shl(m_Value(X), m_APInt(ShiftLC)), m_APInt(ShiftRC))) &&
        ShiftLC == ShiftRC && ShiftLC->ult(BitWidth) &&
        DemandedMask.isSubsetOf(APInt::getLowBitsSet(
            BitWidth, BitWidth - ShiftRC->getZExtValue()))) {
      return X;
    }

    break;
  }
  default:
    // Compute the Known bits to simplify things downstream.
    computeKnownBits(I, Known, Depth, CxtI);

    // If this user is only demanding bits that we know, return the known
    // constant.
    if (DemandedMask.isSubsetOf(Known.Zero|Known.One))
      return Constant::getIntegerValue(ITy, Known.One);

    break;
  }

  return nullptr;
}

Value *MptuneImpl::SimplifyDemandedUseBits(Value *V, APInt DemandedMask,
                                                 KnownBits &Known,
                                                 unsigned Depth,
                                                 Instruction *CxtI) {
  assert(V != nullptr && "Null pointer of Value???");
  assert(Depth <= MaxAnalysisRecursionDepth && "Limit Search Depth");
  uint32_t BitWidth = DemandedMask.getBitWidth();
  Type *VTy = V->getType();
  assert(
      (!VTy->isIntOrIntVectorTy() || VTy->getScalarSizeInBits() == BitWidth) &&
      Known.getBitWidth() == BitWidth &&
      "Value *V, DemandedMask and Known must have same BitWidth");

  if (isa<Constant>(V)) {
    computeKnownBits(V, Known, Depth, CxtI);
    return nullptr;
  }

  Known.resetAll();
  if (DemandedMask.isZero()) // Not demanding any bits from V.
    return UndefValue::get(VTy);

  if (Depth == MaxAnalysisRecursionDepth)
    return nullptr;

  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) {
    computeKnownBits(V, Known, Depth, CxtI);
    return nullptr;        // Only analyze instructions.
  }

  // If there are multiple uses of this value and we aren't at the root, then
  // we can't do any simplifications of the operands, because DemandedMask
  // only reflects the bits demanded by *one* of the users.
  if (Depth != 0 && !I->hasOneUse())
    return SimplifyMultipleUseDemandedBits(I, DemandedMask, Known, Depth, CxtI);

  KnownBits LHSKnown(BitWidth), RHSKnown(BitWidth);

  // If this is the root being simplified, allow it to have multiple uses,
  // just set the DemandedMask to all bits so that we can try to simplify the
  // operands.  This allows visitTruncInst (for example) to simplify the
  // operand of a trunc without duplicating all the logic below.
  if (Depth == 0 && !V->hasOneUse())
    DemandedMask.setAllBits();

  // Update flags after simplifying an operand based on the fact that some high
  // order bits are not demanded.
  auto disableWrapFlagsBasedOnUnusedHighBits = [](Instruction *I,
                                                  unsigned NLZ) {
    if (NLZ > 0) {
      // Disable the nsw and nuw flags here: We can no longer guarantee that
      // we won't wrap after simplification. Removing the nsw/nuw flags is
      // legal here because the top bit is not demanded.
      I->setHasNoSignedWrap(false);
      I->setHasNoUnsignedWrap(false);
    }
    return I;
  };

  // If the high-bits of an ADD/SUB/MUL are not demanded, then we do not care
  // about the high bits of the operands.
  auto simplifyOperandsBasedOnUnusedHighBits = [&](APInt &DemandedFromOps) {
    unsigned NLZ = DemandedMask.countl_zero();
    // Right fill the mask of bits for the operands to demand the most
    // significant bit and all those below it.
    DemandedFromOps = APInt::getLowBitsSet(BitWidth, BitWidth - NLZ);
    if (ShrinkDemandedConstant(I, 0, DemandedFromOps) ||
        SimplifyDemandedBits(I, 0, DemandedFromOps, LHSKnown, Depth + 1) ||
        ShrinkDemandedConstant(I, 1, DemandedFromOps) ||
        SimplifyDemandedBits(I, 1, DemandedFromOps, RHSKnown, Depth + 1)) {
      disableWrapFlagsBasedOnUnusedHighBits(I, NLZ);
      return true;
    }
    return false;
  };

  switch (I->getOpcode()) {
  default:
    computeKnownBits(I, Known, Depth, CxtI);
    break;
  case Instruction::And: {
    // If either the LHS or the RHS are Zero, the result is zero.
    if (SimplifyDemandedBits(I, 1, DemandedMask, RHSKnown, Depth + 1) ||
        SimplifyDemandedBits(I, 0, DemandedMask & ~RHSKnown.Zero, LHSKnown,
                             Depth + 1))
      return I;
    assert(!RHSKnown.hasConflict() && "Bits known to be one AND zero?");
    assert(!LHSKnown.hasConflict() && "Bits known to be one AND zero?");

    Known = analyzeKnownBitsFromAndXorOr(cast<Operator>(I), LHSKnown, RHSKnown,
                                         Depth, DL, &AC, CxtI, &DT);

    // If the client is only demanding bits that we know, return the known
    // constant.
    if (DemandedMask.isSubsetOf(Known.Zero | Known.One))
      return Constant::getIntegerValue(VTy, Known.One);

    // If all of the demanded bits are known 1 on one side, return the other.
    // These bits cannot contribute to the result of the 'and'.
    if (DemandedMask.isSubsetOf(LHSKnown.Zero | RHSKnown.One))
      return I->getOperand(0);
    if (DemandedMask.isSubsetOf(RHSKnown.Zero | LHSKnown.One))
      return I->getOperand(1);

    // If the RHS is a constant, see if we can simplify it.
    if (ShrinkDemandedConstant(I, 1, DemandedMask & ~LHSKnown.Zero))
      return I;

    break;
  }
  case Instruction::Or: {
    // If either the LHS or the RHS are One, the result is One.
    if (SimplifyDemandedBits(I, 1, DemandedMask, RHSKnown, Depth + 1) ||
        SimplifyDemandedBits(I, 0, DemandedMask & ~RHSKnown.One, LHSKnown,
                             Depth + 1))
      return I;
    assert(!RHSKnown.hasConflict() && "Bits known to be one AND zero?");
    assert(!LHSKnown.hasConflict() && "Bits known to be one AND zero?");

    Known = analyzeKnownBitsFromAndXorOr(cast<Operator>(I), LHSKnown, RHSKnown,
                                         Depth, DL, &AC, CxtI, &DT);

    // If the client is only demanding bits that we know, return the known
    // constant.
    if (DemandedMask.isSubsetOf(Known.Zero | Known.One))
      return Constant::getIntegerValue(VTy, Known.One);

    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'or'.
    if (DemandedMask.isSubsetOf(LHSKnown.One | RHSKnown.Zero))
      return I->getOperand(0);
    if (DemandedMask.isSubsetOf(RHSKnown.One | LHSKnown.Zero))
      return I->getOperand(1);

    // If the RHS is a constant, see if we can simplify it.
    if (ShrinkDemandedConstant(I, 1, DemandedMask))
      return I;

    break;
  }
  case Instruction::Xor: {
    if (SimplifyDemandedBits(I, 1, DemandedMask, RHSKnown, Depth + 1) ||
        SimplifyDemandedBits(I, 0, DemandedMask, LHSKnown, Depth + 1))
      return I;
    Value *LHS, *RHS;
    if (DemandedMask == 1 &&
        match(I->getOperand(0), m_Intrinsic<Intrinsic::ctpop>(m_Value(LHS))) &&
        match(I->getOperand(1), m_Intrinsic<Intrinsic::ctpop>(m_Value(RHS)))) {
      // (ctpop(X) ^ ctpop(Y)) & 1 --> ctpop(X^Y) & 1
      IRBuilderBase::InsertPointGuard Guard(Builder);
      Builder.SetInsertPoint(I);
      auto *Xor = Builder.CreateXor(LHS, RHS);
      return Builder.CreateUnaryIntrinsic(Intrinsic::ctpop, Xor);
    }

    assert(!RHSKnown.hasConflict() && "Bits known to be one AND zero?");
    assert(!LHSKnown.hasConflict() && "Bits known to be one AND zero?");

    Known = analyzeKnownBitsFromAndXorOr(cast<Operator>(I), LHSKnown, RHSKnown,
                                         Depth, DL, &AC, CxtI, &DT);

    // If the client is only demanding bits that we know, return the known
    // constant.
    if (DemandedMask.isSubsetOf(Known.Zero | Known.One))
      return Constant::getIntegerValue(VTy, Known.One);

    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'xor'.
    if (DemandedMask.isSubsetOf(RHSKnown.Zero))
      return I->getOperand(0);
    if (DemandedMask.isSubsetOf(LHSKnown.Zero))
      return I->getOperand(1);

    // If all of the demanded bits are known to be zero on one side or the
    // other, turn this into an *inclusive* or.
    //    e.g. (A & C1)^(B & C2) -> (A & C1)|(B & C2) iff C1&C2 == 0
    if (DemandedMask.isSubsetOf(RHSKnown.Zero | LHSKnown.Zero)) {
      Instruction *Or =
        BinaryOperator::CreateOr(I->getOperand(0), I->getOperand(1),
                                 I->getName());
      return InsertNewInstWith(Or, *I);
    }

    // If all of the demanded bits on one side are known, and all of the set
    // bits on that side are also known to be set on the other side, turn this
    // into an AND, as we know the bits will be cleared.
    //    e.g. (X | C1) ^ C2 --> (X | C1) & ~C2 iff (C1&C2) == C2
    if (DemandedMask.isSubsetOf(RHSKnown.Zero|RHSKnown.One) &&
        RHSKnown.One.isSubsetOf(LHSKnown.One)) {
      Constant *AndC = Constant::getIntegerValue(VTy,
                                                 ~RHSKnown.One & DemandedMask);
      Instruction *And = BinaryOperator::CreateAnd(I->getOperand(0), AndC);
      return InsertNewInstWith(And, *I);
    }

    // If the RHS is a constant, see if we can change it. Don't alter a -1
    // constant because that's a canonical 'not' op, and that is better for
    // combining, SCEV, and codegen.
    const APInt *C;
    if (match(I->getOperand(1), m_APInt(C)) && !C->isAllOnes()) {
      if ((*C | ~DemandedMask).isAllOnes()) {
        // Force bits to 1 to create a 'not' op.
        I->setOperand(1, ConstantInt::getAllOnesValue(VTy));
        return I;
      }
      // If we can't turn this into a 'not', try to shrink the constant.
      if (ShrinkDemandedConstant(I, 1, DemandedMask))
        return I;
    }

    // If our LHS is an 'and' and if it has one use, and if any of the bits we
    // are flipping are known to be set, then the xor is just resetting those
    // bits to zero.  We can just knock out bits from the 'and' and the 'xor',
    // simplifying both of them.
    if (Instruction *LHSInst = dyn_cast<Instruction>(I->getOperand(0))) {
      ConstantInt *AndRHS, *XorRHS;
      if (LHSInst->getOpcode() == Instruction::And && LHSInst->hasOneUse() &&
          match(I->getOperand(1), m_ConstantInt(XorRHS)) &&
          match(LHSInst->getOperand(1), m_ConstantInt(AndRHS)) &&
          (LHSKnown.One & RHSKnown.One & DemandedMask) != 0) {
        APInt NewMask = ~(LHSKnown.One & RHSKnown.One & DemandedMask);

        Constant *AndC = ConstantInt::get(VTy, NewMask & AndRHS->getValue());
        Instruction *NewAnd = BinaryOperator::CreateAnd(I->getOperand(0), AndC);
        InsertNewInstWith(NewAnd, *I);

        Constant *XorC = ConstantInt::get(VTy, NewMask & XorRHS->getValue());
        Instruction *NewXor = BinaryOperator::CreateXor(NewAnd, XorC);
        return InsertNewInstWith(NewXor, *I);
      }
    }
    break;
  }
  case Instruction::Select: {
    if (SimplifyDemandedBits(I, 2, DemandedMask, RHSKnown, Depth + 1) ||
        SimplifyDemandedBits(I, 1, DemandedMask, LHSKnown, Depth + 1))
      return I;
    assert(!RHSKnown.hasConflict() && "Bits known to be one AND zero?");
    assert(!LHSKnown.hasConflict() && "Bits known to be one AND zero?");

    // If the operands are constants, see if we can simplify them.
    // This is similar to ShrinkDemandedConstant, but for a select we want to
    // try to keep the selected constants the same as icmp value constants, if
    // we can. This helps not break apart (or helps put back together)
    // canonical patterns like min and max.
    auto CanonicalizeSelectConstant = [](Instruction *I, unsigned OpNo,
                                         const APInt &DemandedMask) {
      const APInt *SelC;
      if (!match(I->getOperand(OpNo), m_APInt(SelC)))
        return false;

      // Get the constant out of the ICmp, if there is one.
      // Only try this when exactly 1 operand is a constant (if both operands
      // are constant, the icmp should eventually simplify). Otherwise, we may
      // invert the transform that reduces set bits and infinite-loop.
      Value *X;
      const APInt *CmpC;
      ICmpInst::Predicate Pred;
      if (!match(I->getOperand(0), m_ICmp(Pred, m_Value(X), m_APInt(CmpC))) ||
          isa<Constant>(X) || CmpC->getBitWidth() != SelC->getBitWidth())
        return ShrinkDemandedConstant(I, OpNo, DemandedMask);

      // If the constant is already the same as the ICmp, leave it as-is.
      if (*CmpC == *SelC)
        return false;
      // If the constants are not already the same, but can be with the demand
      // mask, use the constant value from the ICmp.
      if ((*CmpC & DemandedMask) == (*SelC & DemandedMask)) {
        I->setOperand(OpNo, ConstantInt::get(I->getType(), *CmpC));
        return true;
      }
      return ShrinkDemandedConstant(I, OpNo, DemandedMask);
    };
    if (CanonicalizeSelectConstant(I, 1, DemandedMask) ||
        CanonicalizeSelectConstant(I, 2, DemandedMask))
      return I;

    // Only known if known in both the LHS and RHS.
    Known = LHSKnown.intersectWith(RHSKnown);
    break;
  }
  case Instruction::Trunc: {
    // If we do not demand the high bits of a right-shifted and truncated value,
    // then we may be able to truncate it before the shift.
    Value *X;
    const APInt *C;
    if (match(I->getOperand(0), m_OneUse(m_LShr(m_Value(X), m_APInt(C))))) {
      // The shift amount must be valid (not poison) in the narrow type, and
      // it must not be greater than the high bits demanded of the result.
      if (C->ult(VTy->getScalarSizeInBits()) &&
          C->ule(DemandedMask.countl_zero())) {
        // trunc (lshr X, C) --> lshr (trunc X), C
        IRBuilderBase::InsertPointGuard Guard(Builder);
        Builder.SetInsertPoint(I);
        Value *Trunc = Builder.CreateTrunc(X, VTy);
        return Builder.CreateLShr(Trunc, C->getZExtValue());
      }
    }
  }
    [[fallthrough]];
  case Instruction::ZExt: {
    unsigned SrcBitWidth = I->getOperand(0)->getType()->getScalarSizeInBits();

    APInt InputDemandedMask = DemandedMask.zextOrTrunc(SrcBitWidth);
    KnownBits InputKnown(SrcBitWidth);
    if (SimplifyDemandedBits(I, 0, InputDemandedMask, InputKnown, Depth + 1))
      return I;
    assert(InputKnown.getBitWidth() == SrcBitWidth && "Src width changed?");
    Known = InputKnown.zextOrTrunc(BitWidth);
    assert(!Known.hasConflict() && "Bits known to be one AND zero?");
    break;
  }
  case Instruction::BitCast:
    if (!I->getOperand(0)->getType()->isIntOrIntVectorTy())
      return nullptr;  // vector->int or fp->int?

    if (auto *DstVTy = dyn_cast<VectorType>(VTy)) {
      if (auto *SrcVTy = dyn_cast<VectorType>(I->getOperand(0)->getType())) {
        if (isa<ScalableVectorType>(DstVTy) ||
            isa<ScalableVectorType>(SrcVTy) ||
            cast<FixedVectorType>(DstVTy)->getNumElements() !=
            cast<FixedVectorType>(SrcVTy)->getNumElements())
          // Don't touch a bitcast between vectors of different element counts.
          return nullptr;
      } else
        // Don't touch a scalar-to-vector bitcast.
        return nullptr;
    } else if (I->getOperand(0)->getType()->isVectorTy())
      // Don't touch a vector-to-scalar bitcast.
      return nullptr;

    if (SimplifyDemandedBits(I, 0, DemandedMask, Known, Depth + 1))
      return I;
    assert(!Known.hasConflict() && "Bits known to be one AND zero?");
    break;
  case Instruction::SExt: {
    // Compute the bits in the result that are not present in the input.
    unsigned SrcBitWidth = I->getOperand(0)->getType()->getScalarSizeInBits();

    APInt InputDemandedBits = DemandedMask.trunc(SrcBitWidth);

    // If any of the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    if (DemandedMask.getActiveBits() > SrcBitWidth)
      InputDemandedBits.setBit(SrcBitWidth-1);

    KnownBits InputKnown(SrcBitWidth);
    if (SimplifyDemandedBits(I, 0, InputDemandedBits, InputKnown, Depth + 1))
      return I;

    // If the input sign bit is known zero, or if the NewBits are not demanded
    // convert this into a zero extension.
    if (InputKnown.isNonNegative() ||
        DemandedMask.getActiveBits() <= SrcBitWidth) {
      // Convert to ZExt cast.
      CastInst *NewCast = new ZExtInst(I->getOperand(0), VTy, I->getName());
      return InsertNewInstWith(NewCast, *I);
     }

    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.
    Known = InputKnown.sext(BitWidth);
    assert(!Known.hasConflict() && "Bits known to be one AND zero?");
    break;
  }
  case Instruction::Add: {
    if ((DemandedMask & 1) == 0) {
      // If we do not need the low bit, try to convert bool math to logic:
      // add iN (zext i1 X), (sext i1 Y) --> sext (~X & Y) to iN
      Value *X, *Y;
      if (match(I, m_c_Add(m_OneUse(m_ZExt(m_Value(X))),
                           m_OneUse(m_SExt(m_Value(Y))))) &&
          X->getType()->isIntOrIntVectorTy(1) && X->getType() == Y->getType()) {
        // Truth table for inputs and output signbits:
        //       X:0 | X:1
        //      ----------
        // Y:0  |  0 | 0 |
        // Y:1  | -1 | 0 |
        //      ----------
        IRBuilderBase::InsertPointGuard Guard(Builder);
        Builder.SetInsertPoint(I);
        Value *AndNot = Builder.CreateAnd(Builder.CreateNot(X), Y);
        return Builder.CreateSExt(AndNot, VTy);
      }

      // add iN (sext i1 X), (sext i1 Y) --> sext (X | Y) to iN
      // TODO: Relax the one-use checks because we are removing an instruction?
      if (match(I, m_Add(m_OneUse(m_SExt(m_Value(X))),
                         m_OneUse(m_SExt(m_Value(Y))))) &&
          X->getType()->isIntOrIntVectorTy(1) && X->getType() == Y->getType()) {
        // Truth table for inputs and output signbits:
        //       X:0 | X:1
        //      -----------
        // Y:0  | -1 | -1 |
        // Y:1  | -1 |  0 |
        //      -----------
        IRBuilderBase::InsertPointGuard Guard(Builder);
        Builder.SetInsertPoint(I);
        Value *Or = Builder.CreateOr(X, Y);
        return Builder.CreateSExt(Or, VTy);
      }
    }

    // Right fill the mask of bits for the operands to demand the most
    // significant bit and all those below it.
    unsigned NLZ = DemandedMask.countl_zero();
    APInt DemandedFromOps = APInt::getLowBitsSet(BitWidth, BitWidth - NLZ);
    if (ShrinkDemandedConstant(I, 1, DemandedFromOps) ||
        SimplifyDemandedBits(I, 1, DemandedFromOps, RHSKnown, Depth + 1))
      return disableWrapFlagsBasedOnUnusedHighBits(I, NLZ);

    // If low order bits are not demanded and known to be zero in one operand,
    // then we don't need to demand them from the other operand, since they
    // can't cause overflow into any bits that are demanded in the result.
    unsigned NTZ = (~DemandedMask & RHSKnown.Zero).countr_one();
    APInt DemandedFromLHS = DemandedFromOps;
    DemandedFromLHS.clearLowBits(NTZ);
    if (ShrinkDemandedConstant(I, 0, DemandedFromLHS) ||
        SimplifyDemandedBits(I, 0, DemandedFromLHS, LHSKnown, Depth + 1))
      return disableWrapFlagsBasedOnUnusedHighBits(I, NLZ);

    // If we are known to be adding zeros to every bit below
    // the highest demanded bit, we just return the other side.
    if (DemandedFromOps.isSubsetOf(RHSKnown.Zero))
      return I->getOperand(0);
    if (DemandedFromOps.isSubsetOf(LHSKnown.Zero))
      return I->getOperand(1);

    // Otherwise just compute the known bits of the result.
    bool NSW = cast<OverflowingBinaryOperator>(I)->hasNoSignedWrap();
    Known = KnownBits::computeForAddSub(true, NSW, LHSKnown, RHSKnown);
    break;
  }
  case Instruction::Sub: {
    // Right fill the mask of bits for the operands to demand the most
    // significant bit and all those below it.
    unsigned NLZ = DemandedMask.countl_zero();
    APInt DemandedFromOps = APInt::getLowBitsSet(BitWidth, BitWidth - NLZ);
    if (ShrinkDemandedConstant(I, 1, DemandedFromOps) ||
        SimplifyDemandedBits(I, 1, DemandedFromOps, RHSKnown, Depth + 1))
      return disableWrapFlagsBasedOnUnusedHighBits(I, NLZ);

    // If low order bits are not demanded and are known to be zero in RHS,
    // then we don't need to demand them from LHS, since they can't cause a
    // borrow from any bits that are demanded in the result.
    unsigned NTZ = (~DemandedMask & RHSKnown.Zero).countr_one();
    APInt DemandedFromLHS = DemandedFromOps;
    DemandedFromLHS.clearLowBits(NTZ);
    if (ShrinkDemandedConstant(I, 0, DemandedFromLHS) ||
        SimplifyDemandedBits(I, 0, DemandedFromLHS, LHSKnown, Depth + 1))
      return disableWrapFlagsBasedOnUnusedHighBits(I, NLZ);

    // If we are known to be subtracting zeros from every bit below
    // the highest demanded bit, we just return the other side.
    if (DemandedFromOps.isSubsetOf(RHSKnown.Zero))
      return I->getOperand(0);
    // We can't do this with the LHS for subtraction, unless we are only
    // demanding the LSB.
    if (DemandedFromOps.isOne() && DemandedFromOps.isSubsetOf(LHSKnown.Zero))
      return I->getOperand(1);

    // Otherwise just compute the known bits of the result.
    bool NSW = cast<OverflowingBinaryOperator>(I)->hasNoSignedWrap();
    Known = KnownBits::computeForAddSub(false, NSW, LHSKnown, RHSKnown);
    break;
  }
  case Instruction::Mul: {
    APInt DemandedFromOps;
    if (simplifyOperandsBasedOnUnusedHighBits(DemandedFromOps))
      return I;

    if (DemandedMask.isPowerOf2()) {
      // The LSB of X*Y is set only if (X & 1) == 1 and (Y & 1) == 1.
      // If we demand exactly one bit N and we have "X * (C' << N)" where C' is
      // odd (has LSB set), then the left-shifted low bit of X is the answer.
      unsigned CTZ = DemandedMask.countr_zero();
      const APInt *C;
      if (match(I->getOperand(1), m_APInt(C)) && C->countr_zero() == CTZ) {
        Constant *ShiftC = ConstantInt::get(VTy, CTZ);
        Instruction *Shl = BinaryOperator::CreateShl(I->getOperand(0), ShiftC);
        return InsertNewInstWith(Shl, *I);
      }
    }
    // For a squared value "X * X", the bottom 2 bits are 0 and X[0] because:
    // X * X is odd iff X is odd.
    // 'Quadratic Reciprocity': X * X -> 0 for bit[1]
    if (I->getOperand(0) == I->getOperand(1) && DemandedMask.ult(4)) {
      Constant *One = ConstantInt::get(VTy, 1);
      Instruction *And1 = BinaryOperator::CreateAnd(I->getOperand(0), One);
      return InsertNewInstWith(And1, *I);
    }

    computeKnownBits(I, Known, Depth, CxtI);
    break;
  }
  case Instruction::Shl: {
    const APInt *SA;
    if (match(I->getOperand(1), m_APInt(SA))) {
      const APInt *ShrAmt;
      if (match(I->getOperand(0), m_Shr(m_Value(), m_APInt(ShrAmt))))
        if (Instruction *Shr = dyn_cast<Instruction>(I->getOperand(0)))
          if (Value *R = simplifyShrShlDemandedBits(Shr, *ShrAmt, I, *SA,
                                                    DemandedMask, Known))
            return R;

      // TODO: If we only want bits that already match the signbit then we don't
      // need to shift.

      // If we can pre-shift a right-shifted constant to the left without
      // losing any high bits amd we don't demand the low bits, then eliminate
      // the left-shift:
      // (C >> X) << LeftShiftAmtC --> (C << RightShiftAmtC) >> X
      uint64_t ShiftAmt = SA->getLimitedValue(BitWidth-1);
      Value *X;
      Constant *C;
      if (DemandedMask.countr_zero() >= ShiftAmt &&
          match(I->getOperand(0), m_LShr(m_ImmConstant(C), m_Value(X)))) {
        Constant *LeftShiftAmtC = ConstantInt::get(VTy, ShiftAmt);
        Constant *NewC = ConstantExpr::getShl(C, LeftShiftAmtC);
        if (ConstantExpr::getLShr(NewC, LeftShiftAmtC) == C) {
          Instruction *Lshr = BinaryOperator::CreateLShr(NewC, X);
          return InsertNewInstWith(Lshr, *I);
        }
      }

      APInt DemandedMaskIn(DemandedMask.lshr(ShiftAmt));

      // If the shift is NUW/NSW, then it does demand the high bits.
      ShlOperator *IOp = cast<ShlOperator>(I);
      if (IOp->hasNoSignedWrap())
        DemandedMaskIn.setHighBits(ShiftAmt+1);
      else if (IOp->hasNoUnsignedWrap())
        DemandedMaskIn.setHighBits(ShiftAmt);

      if (SimplifyDemandedBits(I, 0, DemandedMaskIn, Known, Depth + 1))
        return I;
      assert(!Known.hasConflict() && "Bits known to be one AND zero?");

      Known = KnownBits::shl(Known,
                             KnownBits::makeConstant(APInt(BitWidth, ShiftAmt)),
                             /* NUW */ IOp->hasNoUnsignedWrap(),
                             /* NSW */ IOp->hasNoSignedWrap());
    } else {
      // This is a variable shift, so we can't shift the demand mask by a known
      // amount. But if we are not demanding high bits, then we are not
      // demanding those bits from the pre-shifted operand either.
      if (unsigned CTLZ = DemandedMask.countl_zero()) {
        APInt DemandedFromOp(APInt::getLowBitsSet(BitWidth, BitWidth - CTLZ));
        if (SimplifyDemandedBits(I, 0, DemandedFromOp, Known, Depth + 1)) {
          // We can't guarantee that nsw/nuw hold after simplifying the operand.
          I->dropPoisonGeneratingFlags();
          return I;
        }
      }
      computeKnownBits(I, Known, Depth, CxtI);
    }
    break;
  }
  case Instruction::LShr: {
    const APInt *SA;
    if (match(I->getOperand(1), m_APInt(SA))) {
      uint64_t ShiftAmt = SA->getLimitedValue(BitWidth-1);

      // If we are just demanding the shifted sign bit and below, then this can
      // be treated as an ASHR in disguise.
      if (DemandedMask.countl_zero() >= ShiftAmt) {
        // If we only want bits that already match the signbit then we don't
        // need to shift.
        unsigned NumHiDemandedBits = BitWidth - DemandedMask.countr_zero();
        unsigned SignBits =
            ComputeNumSignBits(I->getOperand(0), Depth + 1, CxtI);
        if (SignBits >= NumHiDemandedBits)
          return I->getOperand(0);

        // If we can pre-shift a left-shifted constant to the right without
        // losing any low bits (we already know we don't demand the high bits),
        // then eliminate the right-shift:
        // (C << X) >> RightShiftAmtC --> (C >> RightShiftAmtC) << X
        Value *X;
        Constant *C;
        if (match(I->getOperand(0), m_Shl(m_ImmConstant(C), m_Value(X)))) {
          Constant *RightShiftAmtC = ConstantInt::get(VTy, ShiftAmt);
          Constant *NewC = ConstantExpr::getLShr(C, RightShiftAmtC);
          if (ConstantExpr::getShl(NewC, RightShiftAmtC) == C) {
            Instruction *Shl = BinaryOperator::CreateShl(NewC, X);
            return InsertNewInstWith(Shl, *I);
          }
        }
      }

      // Unsigned shift right.
      APInt DemandedMaskIn(DemandedMask.shl(ShiftAmt));

      // If the shift is exact, then it does demand the low bits (and knows that
      // they are zero).
      if (cast<LShrOperator>(I)->isExact())
        DemandedMaskIn.setLowBits(ShiftAmt);

      if (SimplifyDemandedBits(I, 0, DemandedMaskIn, Known, Depth + 1))
        return I;
      assert(!Known.hasConflict() && "Bits known to be one AND zero?");
      Known.Zero.lshrInPlace(ShiftAmt);
      Known.One.lshrInPlace(ShiftAmt);
      if (ShiftAmt)
        Known.Zero.setHighBits(ShiftAmt);  // high bits known zero.
    } else {
      computeKnownBits(I, Known, Depth, CxtI);
    }
    break;
  }
  case Instruction::AShr: {
    unsigned SignBits = ComputeNumSignBits(I->getOperand(0), Depth + 1, CxtI);

    // If we only want bits that already match the signbit then we don't need
    // to shift.
    unsigned NumHiDemandedBits = BitWidth - DemandedMask.countr_zero();
    if (SignBits >= NumHiDemandedBits)
      return I->getOperand(0);

    // If this is an arithmetic shift right and only the low-bit is set, we can
    // always convert this into a logical shr, even if the shift amount is
    // variable.  The low bit of the shift cannot be an input sign bit unless
    // the shift amount is >= the size of the datatype, which is undefined.
    if (DemandedMask.isOne()) {
      // Perform the logical shift right.
      Instruction *NewVal = BinaryOperator::CreateLShr(
                        I->getOperand(0), I->getOperand(1), I->getName());
      return InsertNewInstWith(NewVal, *I);
    }

    const APInt *SA;
    if (match(I->getOperand(1), m_APInt(SA))) {
      uint32_t ShiftAmt = SA->getLimitedValue(BitWidth-1);

      // Signed shift right.
      APInt DemandedMaskIn(DemandedMask.shl(ShiftAmt));
      // If any of the high bits are demanded, we should set the sign bit as
      // demanded.
      if (DemandedMask.countl_zero() <= ShiftAmt)
        DemandedMaskIn.setSignBit();

      // If the shift is exact, then it does demand the low bits (and knows that
      // they are zero).
      if (cast<AShrOperator>(I)->isExact())
        DemandedMaskIn.setLowBits(ShiftAmt);

      if (SimplifyDemandedBits(I, 0, DemandedMaskIn, Known, Depth + 1))
        return I;

      assert(!Known.hasConflict() && "Bits known to be one AND zero?");
      // Compute the new bits that are at the top now plus sign bits.
      APInt HighBits(APInt::getHighBitsSet(
          BitWidth, std::min(SignBits + ShiftAmt - 1, BitWidth)));
      Known.Zero.lshrInPlace(ShiftAmt);
      Known.One.lshrInPlace(ShiftAmt);

      // If the input sign bit is known to be zero, or if none of the top bits
      // are demanded, turn this into an unsigned shift right.
      assert(BitWidth > ShiftAmt && "Shift amount not saturated?");
      if (Known.Zero[BitWidth-ShiftAmt-1] ||
          !DemandedMask.intersects(HighBits)) {
        BinaryOperator *LShr = BinaryOperator::CreateLShr(I->getOperand(0),
                                                          I->getOperand(1));
        LShr->setIsExact(cast<BinaryOperator>(I)->isExact());
        return InsertNewInstWith(LShr, *I);
      } else if (Known.One[BitWidth-ShiftAmt-1]) { // New bits are known one.
        Known.One |= HighBits;
      }
    } else {
      computeKnownBits(I, Known, Depth, CxtI);
    }
    break;
  }
  case Instruction::UDiv: {
    // UDiv doesn't demand low bits that are zero in the divisor.
    const APInt *SA;
    if (match(I->getOperand(1), m_APInt(SA))) {
      // TODO: Take the demanded mask of the result into account.
      unsigned RHSTrailingZeros = SA->countr_zero();
      APInt DemandedMaskIn =
          APInt::getHighBitsSet(BitWidth, BitWidth - RHSTrailingZeros);
      if (SimplifyDemandedBits(I, 0, DemandedMaskIn, LHSKnown, Depth + 1)) {
        // We can't guarantee that "exact" is still true after changing the
        // the dividend.
        I->dropPoisonGeneratingFlags();
        return I;
      }

      Known = KnownBits::udiv(LHSKnown, KnownBits::makeConstant(*SA),
                              cast<BinaryOperator>(I)->isExact());
    } else {
      computeKnownBits(I, Known, Depth, CxtI);
    }
    break;
  }
  case Instruction::SRem: {
    const APInt *Rem;
    if (match(I->getOperand(1), m_APInt(Rem))) {
      // X % -1 demands all the bits because we don't want to introduce
      // INT_MIN % -1 (== undef) by accident.
      if (Rem->isAllOnes())
        break;
      APInt RA = Rem->abs();
      if (RA.isPowerOf2()) {
        if (DemandedMask.ult(RA))    // srem won't affect demanded bits
          return I->getOperand(0);

        APInt LowBits = RA - 1;
        APInt Mask2 = LowBits | APInt::getSignMask(BitWidth);
        if (SimplifyDemandedBits(I, 0, Mask2, LHSKnown, Depth + 1))
          return I;

        // The low bits of LHS are unchanged by the srem.
        Known.Zero = LHSKnown.Zero & LowBits;
        Known.One = LHSKnown.One & LowBits;

        // If LHS is non-negative or has all low bits zero, then the upper bits
        // are all zero.
        if (LHSKnown.isNonNegative() || LowBits.isSubsetOf(LHSKnown.Zero))
          Known.Zero |= ~LowBits;

        // If LHS is negative and not all low bits are zero, then the upper bits
        // are all one.
        if (LHSKnown.isNegative() && LowBits.intersects(LHSKnown.One))
          Known.One |= ~LowBits;

        assert(!Known.hasConflict() && "Bits known to be one AND zero?");
        break;
      }
    }

    computeKnownBits(I, Known, Depth, CxtI);
    break;
  }
  case Instruction::URem: {
    APInt AllOnes = APInt::getAllOnes(BitWidth);
    if (SimplifyDemandedBits(I, 0, AllOnes, LHSKnown, Depth + 1) ||
        SimplifyDemandedBits(I, 1, AllOnes, RHSKnown, Depth + 1))
      return I;

    Known = KnownBits::urem(LHSKnown, RHSKnown);
    break;
  }
  case Instruction::Call: {
    bool KnownBitsComputed = false;
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::abs: {
        if (DemandedMask == 1)
          return II->getArgOperand(0);
        break;
      }
      case Intrinsic::ctpop: {
        // Checking if the number of clear bits is odd (parity)? If the type has
        // an even number of bits, that's the same as checking if the number of
        // set bits is odd, so we can eliminate the 'not' op.
        Value *X;
        if (DemandedMask == 1 && VTy->getScalarSizeInBits() % 2 == 0 &&
            match(II->getArgOperand(0), m_Not(m_Value(X)))) {
          Function *Ctpop = Intrinsic::getDeclaration(
              II->getModule(), Intrinsic::ctpop, VTy);
          return InsertNewInstWith(CallInst::Create(Ctpop, {X}), *I);
        }
        break;
      }
      case Intrinsic::bswap: {
        // If the only bits demanded come from one byte of the bswap result,
        // just shift the input byte into position to eliminate the bswap.
        unsigned NLZ = DemandedMask.countl_zero();
        unsigned NTZ = DemandedMask.countr_zero();

        // Round NTZ down to the next byte.  If we have 11 trailing zeros, then
        // we need all the bits down to bit 8.  Likewise, round NLZ.  If we
        // have 14 leading zeros, round to 8.
        NLZ = alignDown(NLZ, 8);
        NTZ = alignDown(NTZ, 8);
        // If we need exactly one byte, we can do this transformation.
        if (BitWidth - NLZ - NTZ == 8) {
          // Replace this with either a left or right shift to get the byte into
          // the right place.
          Instruction *NewVal;
          if (NLZ > NTZ)
            NewVal = BinaryOperator::CreateLShr(
                II->getArgOperand(0), ConstantInt::get(VTy, NLZ - NTZ));
          else
            NewVal = BinaryOperator::CreateShl(
                II->getArgOperand(0), ConstantInt::get(VTy, NTZ - NLZ));
          NewVal->takeName(I);
          return InsertNewInstWith(NewVal, *I);
        }
        break;
      }
      case Intrinsic::fshr:
      case Intrinsic::fshl: {
        const APInt *SA;
        if (!match(I->getOperand(2), m_APInt(SA)))
          break;

        // Normalize to funnel shift left. APInt shifts of BitWidth are well-
        // defined, so no need to special-case zero shifts here.
        uint64_t ShiftAmt = SA->urem(BitWidth);
        if (II->getIntrinsicID() == Intrinsic::fshr)
          ShiftAmt = BitWidth - ShiftAmt;

        APInt DemandedMaskLHS(DemandedMask.lshr(ShiftAmt));
        APInt DemandedMaskRHS(DemandedMask.shl(BitWidth - ShiftAmt));
        if (I->getOperand(0) != I->getOperand(1)) {
          if (SimplifyDemandedBits(I, 0, DemandedMaskLHS, LHSKnown,
                                   Depth + 1) ||
              SimplifyDemandedBits(I, 1, DemandedMaskRHS, RHSKnown, Depth + 1))
            return I;
        } else { // fshl is a rotate
          // Avoid converting rotate into funnel shift. 
          // Only simplify if one operand is constant.
          LHSKnown = computeKnownBits(I->getOperand(0), Depth + 1, I);
          if (DemandedMaskLHS.isSubsetOf(LHSKnown.Zero | LHSKnown.One) &&
              !match(I->getOperand(0), m_SpecificInt(LHSKnown.One))) {
            replaceOperand(*I, 0, Constant::getIntegerValue(VTy, LHSKnown.One));
            return I;
          }

          RHSKnown = computeKnownBits(I->getOperand(1), Depth + 1, I);
          if (DemandedMaskRHS.isSubsetOf(RHSKnown.Zero | RHSKnown.One) &&
              !match(I->getOperand(1), m_SpecificInt(RHSKnown.One))) {
            replaceOperand(*I, 1, Constant::getIntegerValue(VTy, RHSKnown.One));
            return I;
          }
        }

        Known.Zero = LHSKnown.Zero.shl(ShiftAmt) |
                     RHSKnown.Zero.lshr(BitWidth - ShiftAmt);
        Known.One = LHSKnown.One.shl(ShiftAmt) |
                    RHSKnown.One.lshr(BitWidth - ShiftAmt);
        KnownBitsComputed = true;
        break;
      }
      case Intrinsic::umax: {
        // UMax(A, C) == A if ...
        // The lowest non-zero bit of DemandMask is higher than the highest
        // non-zero bit of C.
        const APInt *C;
        unsigned CTZ = DemandedMask.countr_zero();
        if (match(II->getArgOperand(1), m_APInt(C)) &&
            CTZ >= C->getActiveBits())
          return II->getArgOperand(0);
        break;
      }
      case Intrinsic::umin: {
        // UMin(A, C) == A if ...
        // The lowest non-zero bit of DemandMask is higher than the highest
        // non-one bit of C.
        // This comes from using DeMorgans on the above umax example.
        const APInt *C;
        unsigned CTZ = DemandedMask.countr_zero();
        if (match(II->getArgOperand(1), m_APInt(C)) &&
            CTZ >= C->getBitWidth() - C->countl_one())
          return II->getArgOperand(0);
        break;
      }
      default: {
        // Handle target specific intrinsics
        std::optional<Value *> V = targetSimplifyDemandedUseBitsIntrinsic(
            *II, DemandedMask, Known, KnownBitsComputed);
        if (V)
          return *V;
        break;
      }
      }
    }

    if (!KnownBitsComputed)
      computeKnownBits(V, Known, Depth, CxtI);
    break;
  }
  }

  // If the client is only demanding bits that we know, return the known
  // constant.
  if (DemandedMask.isSubsetOf(Known.Zero|Known.One))
    return Constant::getIntegerValue(VTy, Known.One);
  return nullptr;
}
bool MptuneImpl::SimplifyDemandedBits(Instruction *I, unsigned OpNo,
                                            const APInt &DemandedMask,
                                            KnownBits &Known, unsigned Depth) {
  Use &U = I->getOperandUse(OpNo);
  Value *NewVal = SimplifyDemandedUseBits(U.get(), DemandedMask, Known,
                                          Depth, I);
  if (!NewVal) return false;
  if (Instruction* OpInst = dyn_cast<Instruction>(U))
    salvageDebugInfo(*OpInst);

  replaceUse(U, NewVal);
  return true;
}
Value *MptuneImpl::SimplifyDemandedVectorElts(Value *V,
                                                    APInt DemandedElts,
                                                    APInt &UndefElts,
                                                    unsigned Depth,
                                                    bool AllowMultipleUsers) {
  // Cannot analyze scalable type. The number of vector elements is not a
  // compile-time constant.
  if (isa<ScalableVectorType>(V->getType()))
    return nullptr;

  unsigned VWidth = cast<FixedVectorType>(V->getType())->getNumElements();
  APInt EltMask(APInt::getAllOnes(VWidth));
  assert((DemandedElts & ~EltMask) == 0 && "Invalid DemandedElts!");

  if (match(V, m_Undef())) {
    // If the entire vector is undef or poison, just return this info.
    UndefElts = EltMask;
    return nullptr;
  }

  if (DemandedElts.isZero()) { // If nothing is demanded, provide poison.
    UndefElts = EltMask;
    return PoisonValue::get(V->getType());
  }

  UndefElts = 0;

  if (auto *C = dyn_cast<Constant>(V)) {
    // Check if this is identity. If so, return 0 since we are not simplifying
    // anything.
    if (DemandedElts.isAllOnes())
      return nullptr;

    Type *EltTy = cast<VectorType>(V->getType())->getElementType();
    Constant *Poison = PoisonValue::get(EltTy);
    SmallVector<Constant*, 16> Elts;
    for (unsigned i = 0; i != VWidth; ++i) {
      if (!DemandedElts[i]) {   // If not demanded, set to poison.
        Elts.push_back(Poison);
        UndefElts.setBit(i);
        continue;
      }

      Constant *Elt = C->getAggregateElement(i);
      if (!Elt) return nullptr;

      Elts.push_back(Elt);
      if (isa<UndefValue>(Elt))   // Already undef or poison.
        UndefElts.setBit(i);
    }

    // If we changed the constant, return it.
    Constant *NewCV = ConstantVector::get(Elts);
    return NewCV != C ? NewCV : nullptr;
  }

  // Limit search depth.
  if (Depth == 10)
    return nullptr;

  if (!AllowMultipleUsers) {
    // If multiple users are using the root value, proceed with
    // simplification conservatively assuming that all elements
    // are needed.
    if (!V->hasOneUse()) {
      // Quit if we find multiple users of a non-root value though.
      // They'll be handled when it's their turn to be visited by
      // the main instcombine process.
      if (Depth != 0)
        // TODO: Just compute the UndefElts information recursively.
        return nullptr;

      // Conservatively assume that all elements are needed.
      DemandedElts = EltMask;
    }
  }

  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return nullptr;        // Only analyze instructions.

  bool MadeChange = false;
  auto simplifyAndSetOp = [&](Instruction *Inst, unsigned OpNum,
                              APInt Demanded, APInt &Undef) {
    auto *II = dyn_cast<IntrinsicInst>(Inst);
    Value *Op = II ? II->getArgOperand(OpNum) : Inst->getOperand(OpNum);
    if (Value *V = SimplifyDemandedVectorElts(Op, Demanded, Undef, Depth + 1)) {
      replaceOperand(*Inst, OpNum, V);
      MadeChange = true;
    }
  };

  APInt UndefElts2(VWidth, 0);
  APInt UndefElts3(VWidth, 0);
  switch (I->getOpcode()) {
  default: break;

  case Instruction::GetElementPtr: {
    // The LangRef requires that struct geps have all constant indices.  As
    // such, we can't convert any operand to partial undef.
    auto mayIndexStructType = [](GetElementPtrInst &GEP) {
      for (auto I = gep_type_begin(GEP), E = gep_type_end(GEP);
           I != E; I++)
        if (I.isStruct())
          return true;
      return false;
    };
    if (mayIndexStructType(cast<GetElementPtrInst>(*I)))
      break;

    // Conservatively track the demanded elements back through any vector
    // operands we may have.  We know there must be at least one, or we
    // wouldn't have a vector result to get here. Note that we intentionally
    // merge the undef bits here since gepping with either an poison base or
    // index results in poison.
    for (unsigned i = 0; i < I->getNumOperands(); i++) {
      if (i == 0 ? match(I->getOperand(i), m_Undef())
                 : match(I->getOperand(i), m_Poison())) {
        // If the entire vector is undefined, just return this info.
        UndefElts = EltMask;
        return nullptr;
      }
      if (I->getOperand(i)->getType()->isVectorTy()) {
        APInt UndefEltsOp(VWidth, 0);
        simplifyAndSetOp(I, i, DemandedElts, UndefEltsOp);
        // gep(x, undef) is not undef, so skip considering idx ops here
        // Note that we could propagate poison, but we can't distinguish between
        // undef & poison bits ATM
        if (i == 0)
          UndefElts |= UndefEltsOp;
      }
    }

    break;
  }
  case Instruction::InsertElement: {
    // If this is a variable index, we don't know which element it overwrites.
    // demand exactly the same input as we produce.
    ConstantInt *Idx = dyn_cast<ConstantInt>(I->getOperand(2));
    if (!Idx) {
      // Note that we can't propagate undef elt info, because we don't know
      // which elt is getting updated.
      simplifyAndSetOp(I, 0, DemandedElts, UndefElts2);
      break;
    }

    // The element inserted overwrites whatever was there, so the input demanded
    // set is simpler than the output set.
    unsigned IdxNo = Idx->getZExtValue();
    APInt PreInsertDemandedElts = DemandedElts;
    if (IdxNo < VWidth)
      PreInsertDemandedElts.clearBit(IdxNo);

    // If we only demand the element that is being inserted and that element
    // was extracted from the same index in another vector with the same type,
    // replace this insert with that other vector.
    // Note: This is attempted before the call to simplifyAndSetOp because that
    //       may change UndefElts to a value that does not match with Vec.
    Value *Vec;
    if (PreInsertDemandedElts == 0 &&
        match(I->getOperand(1),
              m_ExtractElt(m_Value(Vec), m_SpecificInt(IdxNo))) &&
        Vec->getType() == I->getType()) {
      return Vec;
    }

    simplifyAndSetOp(I, 0, PreInsertDemandedElts, UndefElts);

    // If this is inserting an element that isn't demanded, remove this
    // insertelement.
    if (IdxNo >= VWidth || !DemandedElts[IdxNo]) {
      Worklist.push(I);
      return I->getOperand(0);
    }

    // The inserted element is defined.
    UndefElts.clearBit(IdxNo);
    break;
  }
  case Instruction::ShuffleVector: {
    auto *Shuffle = cast<ShuffleVectorInst>(I);
    assert(Shuffle->getOperand(0)->getType() ==
           Shuffle->getOperand(1)->getType() &&
           "Expected shuffle operands to have same type");
    unsigned OpWidth = cast<FixedVectorType>(Shuffle->getOperand(0)->getType())
                           ->getNumElements();
    // Handle trivial case of a splat. Only check the first element of LHS
    // operand.
    if (all_of(Shuffle->getShuffleMask(), [](int Elt) { return Elt == 0; }) &&
        DemandedElts.isAllOnes()) {
      if (!match(I->getOperand(1), m_Undef())) {
        I->setOperand(1, PoisonValue::get(I->getOperand(1)->getType()));
        MadeChange = true;
      }
      APInt LeftDemanded(OpWidth, 1);
      APInt LHSUndefElts(OpWidth, 0);
      simplifyAndSetOp(I, 0, LeftDemanded, LHSUndefElts);
      if (LHSUndefElts[0])
        UndefElts = EltMask;
      else
        UndefElts.clearAllBits();
      break;
    }

    APInt LeftDemanded(OpWidth, 0), RightDemanded(OpWidth, 0);
    for (unsigned i = 0; i < VWidth; i++) {
      if (DemandedElts[i]) {
        unsigned MaskVal = Shuffle->getMaskValue(i);
        if (MaskVal != -1u) {
          assert(MaskVal < OpWidth * 2 &&
                 "shufflevector mask index out of range!");
          if (MaskVal < OpWidth)
            LeftDemanded.setBit(MaskVal);
          else
            RightDemanded.setBit(MaskVal - OpWidth);
        }
      }
    }

    APInt LHSUndefElts(OpWidth, 0);
    simplifyAndSetOp(I, 0, LeftDemanded, LHSUndefElts);

    APInt RHSUndefElts(OpWidth, 0);
    simplifyAndSetOp(I, 1, RightDemanded, RHSUndefElts);

    // If this shuffle does not change the vector length and the elements
    // demanded by this shuffle are an identity mask, then this shuffle is
    // unnecessary.
    //
    // We are assuming canonical form for the mask, so the source vector is
    // operand 0 and operand 1 is not used.
    //
    // Note that if an element is demanded and this shuffle mask is undefined
    // for that element, then the shuffle is not considered an identity
    // operation. The shuffle prevents poison from the operand vector from
    // leaking to the result by replacing poison with an undefined value.
    if (VWidth == OpWidth) {
      bool IsIdentityShuffle = true;
      for (unsigned i = 0; i < VWidth; i++) {
        unsigned MaskVal = Shuffle->getMaskValue(i);
        if (DemandedElts[i] && i != MaskVal) {
          IsIdentityShuffle = false;
          break;
        }
      }
      if (IsIdentityShuffle)
        return Shuffle->getOperand(0);
    }

    bool NewUndefElts = false;
    unsigned LHSIdx = -1u, LHSValIdx = -1u;
    unsigned RHSIdx = -1u, RHSValIdx = -1u;
    bool LHSUniform = true;
    bool RHSUniform = true;
    for (unsigned i = 0; i < VWidth; i++) {
      unsigned MaskVal = Shuffle->getMaskValue(i);
      if (MaskVal == -1u) {
        UndefElts.setBit(i);
      } else if (!DemandedElts[i]) {
        NewUndefElts = true;
        UndefElts.setBit(i);
      } else if (MaskVal < OpWidth) {
        if (LHSUndefElts[MaskVal]) {
          NewUndefElts = true;
          UndefElts.setBit(i);
        } else {
          LHSIdx = LHSIdx == -1u ? i : OpWidth;
          LHSValIdx = LHSValIdx == -1u ? MaskVal : OpWidth;
          LHSUniform = LHSUniform && (MaskVal == i);
        }
      } else {
        if (RHSUndefElts[MaskVal - OpWidth]) {
          NewUndefElts = true;
          UndefElts.setBit(i);
        } else {
          RHSIdx = RHSIdx == -1u ? i : OpWidth;
          RHSValIdx = RHSValIdx == -1u ? MaskVal - OpWidth : OpWidth;
          RHSUniform = RHSUniform && (MaskVal - OpWidth == i);
        }
      }
    }

    // Try to transform shuffle with constant vector and single element from
    // this constant vector to single insertelement instruction.
    // shufflevector V, C, <v1, v2, .., ci, .., vm> ->
    // insertelement V, C[ci], ci-n
    if (OpWidth ==
        cast<FixedVectorType>(Shuffle->getType())->getNumElements()) {
      Value *Op = nullptr;
      Constant *Value = nullptr;
      unsigned Idx = -1u;

      // Find constant vector with the single element in shuffle (LHS or RHS).
      if (LHSIdx < OpWidth && RHSUniform) {
        if (auto *CV = dyn_cast<ConstantVector>(Shuffle->getOperand(0))) {
          Op = Shuffle->getOperand(1);
          Value = CV->getOperand(LHSValIdx);
          Idx = LHSIdx;
        }
      }
      if (RHSIdx < OpWidth && LHSUniform) {
        if (auto *CV = dyn_cast<ConstantVector>(Shuffle->getOperand(1))) {
          Op = Shuffle->getOperand(0);
          Value = CV->getOperand(RHSValIdx);
          Idx = RHSIdx;
        }
      }
      // Found constant vector with single element - convert to insertelement.
      if (Op && Value) {
        Instruction *New = InsertElementInst::Create(
            Op, Value, ConstantInt::get(Type::getInt64Ty(I->getContext()), Idx),
            Shuffle->getName());
        InsertNewInstWith(New, *Shuffle);
        return New;
      }
    }
    if (NewUndefElts) {
      // Add additional discovered undefs.
      SmallVector<int, 16> Elts;
      for (unsigned i = 0; i < VWidth; ++i) {
        if (UndefElts[i])
          Elts.push_back(PoisonMaskElem);
        else
          Elts.push_back(Shuffle->getMaskValue(i));
      }
      Shuffle->setShuffleMask(Elts);
      MadeChange = true;
    }
    break;
  }
  case Instruction::Select: {
    // If this is a vector select, try to transform the select condition based
    // on the current demanded elements.
    SelectInst *Sel = cast<SelectInst>(I);
    if (Sel->getCondition()->getType()->isVectorTy()) {
      // TODO: We are not doing anything with UndefElts based on this call.
      // It is overwritten below based on the other select operands. If an
      // element of the select condition is known undef, then we are free to
      // choose the output value from either arm of the select. If we know that
      // one of those values is undef, then the output can be undef.
      simplifyAndSetOp(I, 0, DemandedElts, UndefElts);
    }

    // Next, see if we can transform the arms of the select.
    APInt DemandedLHS(DemandedElts), DemandedRHS(DemandedElts);
    if (auto *CV = dyn_cast<ConstantVector>(Sel->getCondition())) {
      for (unsigned i = 0; i < VWidth; i++) {
        // isNullValue() always returns false when called on a ConstantExpr.
        // Skip constant expressions to avoid propagating incorrect information.
        Constant *CElt = CV->getAggregateElement(i);
        if (isa<ConstantExpr>(CElt))
          continue;
        // TODO: If a select condition element is undef, we can demand from
        // either side. If one side is known undef, choosing that side would
        // propagate undef.
        if (CElt->isNullValue())
          DemandedLHS.clearBit(i);
        else
          DemandedRHS.clearBit(i);
      }
    }

    simplifyAndSetOp(I, 1, DemandedLHS, UndefElts2);
    simplifyAndSetOp(I, 2, DemandedRHS, UndefElts3);

    // Output elements are undefined if the element from each arm is undefined.
    // TODO: This can be improved. See comment in select condition handling.
    UndefElts = UndefElts2 & UndefElts3;
    break;
  }
  case Instruction::BitCast: {
    // Vector->vector casts only.
    VectorType *VTy = dyn_cast<VectorType>(I->getOperand(0)->getType());
    if (!VTy) break;
    unsigned InVWidth = cast<FixedVectorType>(VTy)->getNumElements();
    APInt InputDemandedElts(InVWidth, 0);
    UndefElts2 = APInt(InVWidth, 0);
    unsigned Ratio;

    if (VWidth == InVWidth) {
      // If we are converting from <4 x i32> -> <4 x f32>, we demand the same
      // elements as are demanded of us.
      Ratio = 1;
      InputDemandedElts = DemandedElts;
    } else if ((VWidth % InVWidth) == 0) {
      // If the number of elements in the output is a multiple of the number of
      // elements in the input then an input element is live if any of the
      // corresponding output elements are live.
      Ratio = VWidth / InVWidth;
      for (unsigned OutIdx = 0; OutIdx != VWidth; ++OutIdx)
        if (DemandedElts[OutIdx])
          InputDemandedElts.setBit(OutIdx / Ratio);
    } else if ((InVWidth % VWidth) == 0) {
      // If the number of elements in the input is a multiple of the number of
      // elements in the output then an input element is live if the
      // corresponding output element is live.
      Ratio = InVWidth / VWidth;
      for (unsigned InIdx = 0; InIdx != InVWidth; ++InIdx)
        if (DemandedElts[InIdx / Ratio])
          InputDemandedElts.setBit(InIdx);
    } else {
      // Unsupported so far.
      break;
    }

    simplifyAndSetOp(I, 0, InputDemandedElts, UndefElts2);

    if (VWidth == InVWidth) {
      UndefElts = UndefElts2;
    } else if ((VWidth % InVWidth) == 0) {
      // If the number of elements in the output is a multiple of the number of
      // elements in the input then an output element is undef if the
      // corresponding input element is undef.
      for (unsigned OutIdx = 0; OutIdx != VWidth; ++OutIdx)
        if (UndefElts2[OutIdx / Ratio])
          UndefElts.setBit(OutIdx);
    } else if ((InVWidth % VWidth) == 0) {
      // If the number of elements in the input is a multiple of the number of
      // elements in the output then an output element is undef if all of the
      // corresponding input elements are undef.
      for (unsigned OutIdx = 0; OutIdx != VWidth; ++OutIdx) {
        APInt SubUndef = UndefElts2.lshr(OutIdx * Ratio).zextOrTrunc(Ratio);
        if (SubUndef.popcount() == Ratio)
          UndefElts.setBit(OutIdx);
      }
    } else {
      llvm_unreachable("Unimp");
    }
    break;
  }
  case Instruction::FPTrunc:
  case Instruction::FPExt:
    simplifyAndSetOp(I, 0, DemandedElts, UndefElts);
    break;

  case Instruction::Call: {
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(I);
    if (!II) break;
    switch (II->getIntrinsicID()) {
    case Intrinsic::masked_gather: // fallthrough
    case Intrinsic::masked_load: {
      // Subtlety: If we load from a pointer, the pointer must be valid
      // regardless of whether the element is demanded.  Doing otherwise risks
      // segfaults which didn't exist in the original program.
      APInt DemandedPtrs(APInt::getAllOnes(VWidth)),
          DemandedPassThrough(DemandedElts);
      if (auto *CV = dyn_cast<ConstantVector>(II->getOperand(2)))
        for (unsigned i = 0; i < VWidth; i++) {
          Constant *CElt = CV->getAggregateElement(i);
          if (CElt->isNullValue())
            DemandedPtrs.clearBit(i);
          else if (CElt->isAllOnesValue())
            DemandedPassThrough.clearBit(i);
        }
      if (II->getIntrinsicID() == Intrinsic::masked_gather)
        simplifyAndSetOp(II, 0, DemandedPtrs, UndefElts2);
      simplifyAndSetOp(II, 3, DemandedPassThrough, UndefElts3);

      // Output elements are undefined if the element from both sources are.
      // TODO: can strengthen via mask as well.
      UndefElts = UndefElts2 & UndefElts3;
      break;
    }
    default: {
      // Handle target specific intrinsics
      std::optional<Value *> V = targetSimplifyDemandedVectorEltsIntrinsic(
          *II, DemandedElts, UndefElts, UndefElts2, UndefElts3,
          simplifyAndSetOp);
      if (V)
        return *V;
      break;
    }
    } // switch on IntrinsicID
    break;
  } // case Call
  } // switch on Opcode

  // TODO: We bail completely on integer div/rem and shifts because they have
  // UB/poison potential, but that should be refined.
  BinaryOperator *BO;
  if (match(I, m_BinOp(BO)) && !BO->isIntDivRem() && !BO->isShift()) {
    Value *X = BO->getOperand(0);
    Value *Y = BO->getOperand(1);

    // Look for an equivalent binop except that one operand has been shuffled.
    // If the demand for this binop only includes elements that are the same as
    // the other binop, then we may be able to replace this binop with a use of
    // the earlier one.
    //
    // Example:
    // %other_bo = bo (shuf X, {0}), Y
    // %this_extracted_bo = extelt (bo X, Y), 0
    // -->
    // %other_bo = bo (shuf X, {0}), Y
    // %this_extracted_bo = extelt %other_bo, 0
    //
    // TODO: Handle demand of an arbitrary single element or more than one
    //       element instead of just element 0.
    // TODO: Unlike general demanded elements transforms, this should be safe
    //       for any (div/rem/shift) opcode too.
    if (DemandedElts == 1 && !X->hasOneUse() && !Y->hasOneUse() &&
        BO->hasOneUse() ) {

      auto findShufBO = [&](bool MatchShufAsOp0) -> User * {
        // Try to use shuffle-of-operand in place of an operand:
        // bo X, Y --> bo (shuf X), Y
        // bo X, Y --> bo X, (shuf Y)
        BinaryOperator::BinaryOps Opcode = BO->getOpcode();
        Value *ShufOp = MatchShufAsOp0 ? X : Y;
        Value *OtherOp = MatchShufAsOp0 ? Y : X;
        for (User *U : OtherOp->users()) {
          auto Shuf = m_Shuffle(m_Specific(ShufOp), m_Value(), m_ZeroMask());
          if (BO->isCommutative()
                  ? match(U, m_c_BinOp(Opcode, Shuf, m_Specific(OtherOp)))
                  : MatchShufAsOp0
                        ? match(U, m_BinOp(Opcode, Shuf, m_Specific(OtherOp)))
                        : match(U, m_BinOp(Opcode, m_Specific(OtherOp), Shuf)))
            if (DT.dominates(U, I))
              return U;
        }
        return nullptr;
      };

      if (User *ShufBO = findShufBO(/* MatchShufAsOp0 */ true))
        return ShufBO;
      if (User *ShufBO = findShufBO(/* MatchShufAsOp0 */ false))
        return ShufBO;
    }

    simplifyAndSetOp(I, 0, DemandedElts, UndefElts);
    simplifyAndSetOp(I, 1, DemandedElts, UndefElts2);

    // Output elements are undefined if both are undefined. Consider things
    // like undef & 0. The result is known zero, not undef.
    UndefElts &= UndefElts2;
  }

  // If we've proven all of the lanes undef, return an undef value.
  // TODO: Intersect w/demanded lanes
  if (UndefElts.isAllOnes())
    return UndefValue::get(I->getType());

  return MadeChange ? I : nullptr;
}

Value *MptuneImpl::simplifyShrShlDemandedBits(
    Instruction *Shr, const APInt &ShrOp1, Instruction *Shl,
    const APInt &ShlOp1, const APInt &DemandedMask, KnownBits &Known) {
  if (!ShlOp1 || !ShrOp1)
    return nullptr; // No-op.

  Value *VarX = Shr->getOperand(0);
  Type *Ty = VarX->getType();
  unsigned BitWidth = Ty->getScalarSizeInBits();
  if (ShlOp1.uge(BitWidth) || ShrOp1.uge(BitWidth))
    return nullptr; // Undef.

  unsigned ShlAmt = ShlOp1.getZExtValue();
  unsigned ShrAmt = ShrOp1.getZExtValue();

  Known.One.clearAllBits();
  Known.Zero.setLowBits(ShlAmt - 1);
  Known.Zero &= DemandedMask;

  APInt BitMask1(APInt::getAllOnes(BitWidth));
  APInt BitMask2(APInt::getAllOnes(BitWidth));

  bool isLshr = (Shr->getOpcode() == Instruction::LShr);
  BitMask1 = isLshr ? (BitMask1.lshr(ShrAmt) << ShlAmt) :
                      (BitMask1.ashr(ShrAmt) << ShlAmt);

  if (ShrAmt <= ShlAmt) {
    BitMask2 <<= (ShlAmt - ShrAmt);
  } else {
    BitMask2 = isLshr ? BitMask2.lshr(ShrAmt - ShlAmt):
                        BitMask2.ashr(ShrAmt - ShlAmt);
  }

  // Check if condition-2 (see the comment to this function) is satified.
  if ((BitMask1 & DemandedMask) == (BitMask2 & DemandedMask)) {
    if (ShrAmt == ShlAmt)
      return VarX;

    if (!Shr->hasOneUse())
      return nullptr;

    BinaryOperator *New;
    if (ShrAmt < ShlAmt) {
      Constant *Amt = ConstantInt::get(VarX->getType(), ShlAmt - ShrAmt);
      New = BinaryOperator::CreateShl(VarX, Amt);
      BinaryOperator *Orig = cast<BinaryOperator>(Shl);
      New->setHasNoSignedWrap(Orig->hasNoSignedWrap());
      New->setHasNoUnsignedWrap(Orig->hasNoUnsignedWrap());
    } else {
      Constant *Amt = ConstantInt::get(VarX->getType(), ShrAmt - ShlAmt);
      New = isLshr ? BinaryOperator::CreateLShr(VarX, Amt) :
                     BinaryOperator::CreateAShr(VarX, Amt);
      if (cast<BinaryOperator>(Shr)->isExact())
        New->setIsExact(true);
    }

    return InsertNewInstWith(New, *Shl);
  }

  return nullptr;
}


bool MptuneImpl::tryToSinkInstruction(Instruction *I,
                                            BasicBlock *DestBlock) {
  BasicBlock *SrcBlock = I->getParent();

  // Cannot move control-flow-involving, volatile loads, vaarg, etc.
  if (isa<PHINode>(I) || I->isEHPad() || I->mayThrow() || !I->willReturn() ||
      I->isTerminator())
    return false;

  // Do not sink static or dynamic alloca instructions. Static allocas must
  // remain in the entry block, and dynamic allocas must not be sunk in between
  // a stacksave / stackrestore pair, which would incorrectly shorten its
  // lifetime.
  if (isa<AllocaInst>(I))
    return false;

  // Do not sink into catchswitch blocks.
  if (isa<CatchSwitchInst>(DestBlock->getTerminator()))
    return false;

  // Do not sink convergent call instructions.
  if (auto *CI = dyn_cast<CallInst>(I)) {
    if (CI->isConvergent())
      return false;
  }

  // Unless we can prove that the memory write isn't visibile except on the
  // path we're sinking to, we must bail.
  if (I->mayWriteToMemory()) {
    if (!SoleWriteToDeadLocal(I, TLI))
      return false;
  }

  // We can only sink load instructions if there is nothing between the load and
  // the end of block that could change the value.
  if (I->mayReadFromMemory()) {
    // We don't want to do any sophisticated alias analysis, so we only check
    // the instructions after I in I's parent block if we try to sink to its
    // successor block.
    if (DestBlock->getUniquePredecessor() != I->getParent())
      return false;
    for (BasicBlock::iterator Scan = std::next(I->getIterator()),
                              E = I->getParent()->end();
         Scan != E; ++Scan)
      if (Scan->mayWriteToMemory())
        return false;
  }

  I->dropDroppableUses([&](const Use *U) {
    auto *I = dyn_cast<Instruction>(U->getUser());
    if (I && I->getParent() != DestBlock) {
      Worklist.add(I);
      return true;
    }
    return false;
  });
  /// FIXME: We could remove droppable uses that are not dominated by
  /// the new position.

  BasicBlock::iterator InsertPos = DestBlock->getFirstInsertionPt();
  I->moveBefore(&*InsertPos);
  ++NumSunkInst;

  // Also sink all related debug uses from the source basic block. Otherwise we
  // get debug use before the def. Attempt to salvage debug uses first, to
  // maximise the range variables have location for. If we cannot salvage, then
  // mark the location undef: we know it was supposed to receive a new location
  // here, but that computation has been sunk.
  SmallVector<DbgVariableIntrinsic *, 2> DbgUsers;
  findDbgUsers(DbgUsers, I);
  // Process the sinking DbgUsers in reverse order, as we only want to clone the
  // last appearing debug intrinsic for each given variable.
  SmallVector<DbgVariableIntrinsic *, 2> DbgUsersToSink;
  for (DbgVariableIntrinsic *DVI : DbgUsers)
    if (DVI->getParent() == SrcBlock)
      DbgUsersToSink.push_back(DVI);
  llvm::sort(DbgUsersToSink,
             [](auto *A, auto *B) { return B->comesBefore(A); });

  SmallVector<DbgVariableIntrinsic *, 2> DIIClones;
  SmallSet<DebugVariable, 4> SunkVariables;
  for (auto *User : DbgUsersToSink) {
    // A dbg.declare instruction should not be cloned, since there can only be
    // one per variable fragment. It should be left in the original place
    // because the sunk instruction is not an alloca (otherwise we could not be
    // here).
    if (isa<DbgDeclareInst>(User))
      continue;

    DebugVariable DbgUserVariable =
        DebugVariable(User->getVariable(), User->getExpression(),
                      User->getDebugLoc()->getInlinedAt());

    if (!SunkVariables.insert(DbgUserVariable).second)
      continue;

    // Leave dbg.assign intrinsics in their original positions and there should
    // be no need to insert a clone.
    if (isa<DbgAssignIntrinsic>(User))
      continue;

    DIIClones.emplace_back(cast<DbgVariableIntrinsic>(User->clone()));
    if (isa<DbgDeclareInst>(User) && isa<CastInst>(I))
      DIIClones.back()->replaceVariableLocationOp(I, I->getOperand(0));
    LLVM_DEBUG(dbgs() << "CLONE: " << *DIIClones.back() << '\n');
  }

  // Perform salvaging without the clones, then sink the clones.
  if (!DIIClones.empty()) {
    salvageDebugInfoForDbgValues(*I, DbgUsers);
    // The clones are in reverse order of original appearance, reverse again to
    // maintain the original order.
    for (auto &DIIClone : llvm::reverse(DIIClones)) {
      DIIClone->insertBefore(&*InsertPos);
      LLVM_DEBUG(dbgs() << "SINK: " << *DIIClone << '\n');
    }
  }

  return true;
}

bool MptuneImpl::run() {
  while (!Worklist.isEmpty()) {
    // Walk deferred instructions in reverse order, and push them to the
    // worklist, which means they'll end up popped from the worklist in-order.
    while (Instruction *I = Worklist.popDeferred()) {
      // Check to see if we can DCE the instruction. We do this already here to
      // reduce the number of uses and thus allow other folds to trigger.
      // Note that eraseInstFromFunction() may push additional instructions on
      // the deferred worklist, so this will DCE whole instruction chains.
      if (isInstructionTriviallyDead(I, &TLI)) {
        eraseInstFromFunction(*I);
        ++NumDeadInst;
        continue;
      }

      Worklist.push(I);
    }

    Instruction *I = Worklist.removeOne();
    if (I == nullptr) continue;  // skip null values.

    // Check to see if we can DCE the instruction.
    if (isInstructionTriviallyDead(I, &TLI)) {
      eraseInstFromFunction(*I);
      ++NumDeadInst;
      continue;
    }

    if (!DebugCounter::shouldExecute(VisitCounter))
      continue;

    // See if we can trivially sink this instruction to its user if we can
    // prove that the successor is not executed more frequently than our block.
    // Return the UserBlock if successful.
    auto getOptionalSinkBlockForInst =
        [this](Instruction *I) -> std::optional<BasicBlock *> {
      if (!EnableCodeSinking)
        return std::nullopt;

      BasicBlock *BB = I->getParent();
      BasicBlock *UserParent = nullptr;
      unsigned NumUsers = 0;

      for (auto *U : I->users()) {
        if (U->isDroppable())
          continue;
        if (NumUsers > MaxSinkNumUsers)
          return std::nullopt;

        Instruction *UserInst = cast<Instruction>(U);
        // Special handling for Phi nodes - get the block the use occurs in.
        if (PHINode *PN = dyn_cast<PHINode>(UserInst)) {
          for (unsigned i = 0; i < PN->getNumIncomingValues(); i++) {
            if (PN->getIncomingValue(i) == I) {
              // Bail out if we have uses in different blocks. We don't do any
              // sophisticated analysis (i.e finding NearestCommonDominator of
              // these use blocks).
              if (UserParent && UserParent != PN->getIncomingBlock(i))
                return std::nullopt;
              UserParent = PN->getIncomingBlock(i);
            }
          }
          assert(UserParent && "expected to find user block!");
        } else {
          if (UserParent && UserParent != UserInst->getParent())
            return std::nullopt;
          UserParent = UserInst->getParent();
        }

        // Make sure these checks are done only once, naturally we do the checks
        // the first time we get the userparent, this will save compile time.
        if (NumUsers == 0) {
          // Try sinking to another block. If that block is unreachable, then do
          // not bother. SimplifyCFG should handle it.
          if (UserParent == BB || !DT.isReachableFromEntry(UserParent))
            return std::nullopt;

          auto *Term = UserParent->getTerminator();
          // See if the user is one of our successors that has only one
          // predecessor, so that we don't have to split the critical edge.
          // Another option where we can sink is a block that ends with a
          // terminator that does not pass control to other block (such as
          // return or unreachable or resume). In this case:
          //   - I dominates the User (by SSA form);
          //   - the User will be executed at most once.
          // So sinking I down to User is always profitable or neutral.
          if (UserParent->getUniquePredecessor() != BB && !succ_empty(Term))
            return std::nullopt;

          assert(DT.dominates(BB, UserParent) && "Dominance relation broken?");
        }

        NumUsers++;
      }

      // No user or only has droppable users.
      if (!UserParent)
        return std::nullopt;

      return UserParent;
    };

    auto OptBB = getOptionalSinkBlockForInst(I);
    if (OptBB) {
      auto *UserParent = *OptBB;
      // Okay, the CFG is simple enough, try to sink this instruction.
      if (tryToSinkInstruction(I, UserParent)) {
        LLVM_DEBUG(dbgs() << "IC: Sink: " << *I << '\n');
        MadeIRChange = true;
        // We'll add uses of the sunk instruction below, but since
        // sinking can expose opportunities for it's *operands* add
        // them to the worklist
        for (Use &U : I->operands())
          if (Instruction *OpI = dyn_cast<Instruction>(U.get()))
            Worklist.push(OpI);
      }
    }

    // Now that we have an instruction, try combining it to simplify it.
    Builder.SetInsertPoint(I);
    Builder.CollectMetadataToCopy(
        I, {LLVMContext::MD_dbg, LLVMContext::MD_annotation});

#ifndef NDEBUG
    std::string OrigI;
#endif
    LLVM_DEBUG(raw_string_ostream SS(OrigI); I->print(SS); OrigI = SS.str(););
    LLVM_DEBUG(dbgs() << "IC: Visiting: " << OrigI << '\n');

    //if (Instruction *Result = visit2peephole(static_cast<BinaryOperator&>(*I))) {
    if (Instruction *Result = visit2peephole(*I)){
      ++NumCombined;
      // Should we replace the old instruction with a new one?
      if (Result != I) {
        //errs()<<"result != I\n";
        LLVM_DEBUG(dbgs() << "IC: Old = " << *I << '\n'
                          << "    New = " << *Result << '\n');

        Result->copyMetadata(*I,
                             {LLVMContext::MD_dbg, LLVMContext::MD_annotation});
        // Everything uses the new instruction now.
        I->replaceAllUsesWith(Result);

        // Move the name to the new instruction first.
        Result->takeName(I);

        // Insert the new instruction into the basic block...
        BasicBlock *InstParent = I->getParent();
        BasicBlock::iterator InsertPos = I->getIterator();

        // Are we replace a PHI with something that isn't a PHI, or vice versa?
        if (isa<PHINode>(Result) != isa<PHINode>(I)) {
          // We need to fix up the insertion point.
          if (isa<PHINode>(I)) // PHI -> Non-PHI
            InsertPos = InstParent->getFirstInsertionPt();
          else // Non-PHI -> PHI
            InsertPos = InstParent->getFirstNonPHI()->getIterator();
        }

        Result->insertInto(InstParent, InsertPos);

        // Push the new instruction and any users onto the worklist.
        Worklist.pushUsersToWorkList(*Result);
        Worklist.push(Result);

        eraseInstFromFunction(*I);
      } else {
        LLVM_DEBUG(dbgs() << "IC: Mod = " << OrigI << '\n'
                          << "    New = " << *I << '\n');

        // If the instruction was modified, it's possible that it is now dead.
        // if so, remove it.
        if (isInstructionTriviallyDead(I, &TLI)) {
          eraseInstFromFunction(*I);
        } else {
          Worklist.pushUsersToWorkList(*I);
          Worklist.push(I);
        }
      }
      MadeIRChange = true;
    }
  }

  Worklist.zap();
  return MadeIRChange;
}

// 默认X是一个调用指令
static StringRef custom_getFuncName(Value *X){
  if(CallInst *ci = dyn_cast<CallInst>(X)){
    Value *calledValue = ci->getCalledOperand();
    Function *calledFunction = dyn_cast<Function>(calledValue);
    return calledFunction->getName();
  }
  return "";
}
// 默认X是一个调用指令，获取第一个参数。
static Value* custom_getFuncArg(Value *X){
  if(CallInst *ci = dyn_cast<CallInst>(X)){
    return ci->getArgOperand(0);
  }
  return nullptr;
}

// 默认old和neww是调用指令，保持调用指令的属性一致。
static void keepCallAttribute(Value *old, Value *neww){
  dyn_cast<CallInst>(neww)->setTailCall(dyn_cast<CallInst>(old)->getTailCallKind());
  dyn_cast<CallInst>(neww)->setAttributes(dyn_cast<CallInst>(old)->getAttributes());
  dyn_cast<CallInst>(neww)->setCallingConv(dyn_cast<CallInst>(old)->getCallingConv());
}

Instruction *MptuneImpl::visit2peephole(Instruction &I){
  //errs()<<I.getOpcodeName()<<'\n' ;
  Value *X, *Y, *W, *V;
  
  // float x;
  // sqrtf(x+1)-sqrtf(x) ---> 1/(sqrtf(x+1)+sqrtf(x))
  // 匹配llvm.sqrt.f32
  if(match(&I,m_FSub(m_Value(W),m_Value(V))) && 
     match(&I,
          m_FSub(m_OneUse(m_Sqrt(m_FAdd(m_Value(X),m_SpecificFP(1.0)))),
                 m_OneUse(m_Sqrt(m_Value(X)))))){
			Type* XTy=X->getType();
      if(XTy->isFloatTy()){
				//errs()<<"find llvm.sqrt : sqrtf(x+1)-sqrtf(x)\n";
				Value *add1 = Builder.CreateFAddFMF(X, ConstantFP::get(I.getType(), 1.0),&I);
				Value *Sqrt1 = Builder.CreateUnaryIntrinsic(Intrinsic::sqrt, add1, &I);
        keepCallAttribute(W,Sqrt1);
				Value *Sqrt2 = Builder.CreateUnaryIntrinsic(Intrinsic::sqrt, X, &I);
        keepCallAttribute(V,Sqrt2);
				Value *add2 = Builder.CreateFAddFMF(Sqrt1,Sqrt2,&I);
				Value *div1 = Builder.CreateFDivFMF(ConstantFP::get(I.getType(), 1.0),add2,&I);
				return replaceInstUsesWith(I,div1);
      }			
	  }

  // float x;
  // sqrtf(x+1)-sqrtf(x) ---> 1/(sqrtf(x+1)+sqrtf(x))
  // 匹配函数名sqrtf
  if(match(&I,m_FSub(m_Value(X),m_Value(Y))) && 
    custom_getFuncName(X)=="sqrtf" &&
    custom_getFuncName(Y)=="sqrtf" &&
    custom_getFuncArg(X) && 
    custom_getFuncArg(Y) && 
    match(custom_getFuncArg(X),m_FAdd(m_Value(W),m_SpecificFP(1.0))) && 
    W == custom_getFuncArg(Y)
    ){
			Type* WTy=W->getType();
      if(WTy->isFloatTy()){
				Value *add1 = Builder.CreateFAddFMF(W, ConstantFP::get(I.getType(), 1.0),&I);
        Module *module = I.getParent()->getModule();
        llvm::Function* sqrtFunc = module->getFunction("sqrtf");
        Value* Sqrt1 = Builder.CreateCall(sqrtFunc, {add1});
        keepCallAttribute(X,Sqrt1);
        Value* Sqrt2 = Builder.CreateCall(sqrtFunc, {W});
        keepCallAttribute(Y,Sqrt2);
				Value *add2 = Builder.CreateFAddFMF(Sqrt1,Sqrt2,&I);
				Value *div1 = Builder.CreateFDivFMF(ConstantFP::get(I.getType(), 1.0),add2,&I);
        dyn_cast<CallInst>(X)->setOnlyReadsMemory();
        dyn_cast<CallInst>(Y)->setOnlyReadsMemory();
				return replaceInstUsesWith(I,div1);
      }			
	  }

  // expf(x)-1 ----> x+ (1/2)x^2 + (1/6)x^3   --->      x(1+x(0.5+1/6*x))
  // 匹配内置函数llvm.exp.f32
  if(match(&I,m_FAdd(m_Intrinsic<Intrinsic::exp>(m_Value(X)),m_SpecificFP(-1.0)))){
      Type* XTy=X->getType();
      if(XTy->isFloatTy()){
        Value *div1=Builder.CreateFDivFMF(ConstantFP::get(I.getType(), 1.0),ConstantFP::get(I.getType(), 6.0),&I);
        Value *mul1=Builder.CreateFMulFMF(div1,X,&I);
        Value *add1=Builder.CreateFAddFMF(ConstantFP::get(I.getType(), 0.5),mul1,&I);
        Value *mul2=Builder.CreateFMulFMF(add1,X,&I);
        Value *add2=Builder.CreateFAddFMF(ConstantFP::get(I.getType(), 1.0),mul2,&I);
        Value *mul3=Builder.CreateFMulFMF(add2,X,&I);
        return replaceInstUsesWith(I,mul3);
      }
  }
  // expf(x)-1 ----> x+ (1/2)x^2 + (1/6)x^3   --->      x(1+x(0.5+1/6*x))
  // 匹配函数名expf,不需要判断参数类型。
  //-1在开启优化后，变成了+(-1)
  if(match(&I,m_FAdd(m_Value(X),m_SpecificFP(-1.0)))){
      if(CallInst *ci = dyn_cast<CallInst>(X)){
        Value *calledValue = ci->getCalledOperand();
        Function *calledFunction = dyn_cast<Function>(calledValue);
        if(calledFunction->getName() == "expf"){
          dyn_cast<CallInst>(X)->setOnlyReadsMemory();
          Value *Y=ci->getArgOperand(0);
          Value *div1=Builder.CreateFDivFMF(ConstantFP::get(I.getType(), 1.0),ConstantFP::get(I.getType(), 6.0),&I);
          Value *mul1=Builder.CreateFMulFMF(div1,Y,&I);
          Value *add1=Builder.CreateFAddFMF(ConstantFP::get(I.getType(), 0.5),mul1,&I);
          Value *mul2=Builder.CreateFMulFMF(add1,Y,&I);
          Value *add2=Builder.CreateFAddFMF(ConstantFP::get(I.getType(), 1.0),mul2,&I);
          Value *mul3=Builder.CreateFMulFMF(add2,Y,&I);
          return replaceInstUsesWith(I,mul3);
      //return BinaryOperator::CreateFMulFMF(add2, X,&I);
      }}}
			
  // float x,y;
  // sinf(x) * cosf(y) + cosf(x) * sinf(y) --> sinf(x+y)
  // if (match(&I, m_FAdd(
  //           m_c_FMul(m_Intrinsic<Intrinsic::sin>(m_Value(X)),
  //                  m_Intrinsic<Intrinsic::cos>(m_Value(Y))),
  //           m_c_FMul(m_Intrinsic<Intrinsic::cos>(m_Value(X)),
  //                  m_Intrinsic<Intrinsic::sin>(m_Value(Y)))))) {
  //   Type* XTy=X->getType();
  //   Type* YTy=Y->getType();
  //   if(XTy->isFloatTy() && YTy->isFloatTy()){
  //     errs()<<"find sinx * cosy + cosx * siny\n";
  //     Value *add1 = Builder.CreateFAddFMF(X, Y, &I);
  //     Value *sin_add = Builder.CreateUnaryIntrinsic(Intrinsic::sin, add1, &I);
  //     return replaceInstUsesWith(I,sin_add);
  //   }
  // }



  // float x,y;
  // cosf(x) * cosf(y) + sinf(x) * sinf(y) --> cosf(x-y)
  // if (match(&I, m_FAdd(
  //           m_c_FMul(m_Intrinsic<Intrinsic::cos>(m_Value(X)),
  //                  m_Intrinsic<Intrinsic::cos>(m_Value(Y))),
  //           m_c_FMul(m_Intrinsic<Intrinsic::sin>(m_Value(X)),
  //                  m_Intrinsic<Intrinsic::sin>(m_Value(Y)))))) {
  //   Type* XTy=X->getType();
  //   Type* YTy=Y->getType();
  //   if(XTy->isFloatTy() && YTy->isFloatTy()){
  //     errs()<<"find cosx * cosy + sinx * siny\n";
  //     Value *sub1 = Builder.CreateFSubFMF(X, Y, &I);
  //     Value *sin_sub = Builder.CreateUnaryIntrinsic(Intrinsic::cos, sub1, &I);
  //     return replaceInstUsesWith(I,sin_sub);
  //   }
  // }

  // float x,y;
  // sinf(x) * cosf(y) - cosf(x) * sinf(y) --> sinf(x-y)
  // if (match(&I, m_FSub(
  //           m_c_FMul(m_Intrinsic<Intrinsic::sin>(m_Value(X)),
  //                  m_Intrinsic<Intrinsic::cos>(m_Value(Y))),
  //           m_c_FMul(m_Intrinsic<Intrinsic::cos>(m_Value(X)),
  //                  m_Intrinsic<Intrinsic::sin>(m_Value(Y)))))) {
  //   Type* XTy=X->getType();
  //   Type* YTy=Y->getType();
  //   if(XTy->isFloatTy() && YTy->isFloatTy()){
  //     errs()<<"find sinx * cosy - cosx * siny\n";
  //     Value *sub1 = Builder.CreateFSubFMF(X, Y, &I);
  //     Value *sin_sub = Builder.CreateUnaryIntrinsic(Intrinsic::sin, sub1, &I);
  //     return replaceInstUsesWith(I,sin_sub);
  //   }
  // }

  // float x,y;
  // cosf(x) * cosf(y) - sinf(x) * sinf(y) --> cosf(x+y)
  // if (match(&I, m_FSub(
  //           m_c_FMul(m_Intrinsic<Intrinsic::cos>(m_Value(X)),
  //                  m_Intrinsic<Intrinsic::cos>(m_Value(Y))),
  //           m_c_FMul(m_Intrinsic<Intrinsic::sin>(m_Value(X)),
  //                  m_Intrinsic<Intrinsic::sin>(m_Value(Y)))))) {
  //   Type* XTy=X->getType();
  //   Type* YTy=Y->getType();
  //   if(XTy->isFloatTy() && YTy->isFloatTy()){
  //     errs()<<"find cosx * cosy - sinx * siny\n";
  //     Value *add1 = Builder.CreateFAddFMF(X, Y, &I);
  //     Value *cos_add = Builder.CreateUnaryIntrinsic(Intrinsic::cos, add1, &I);
  //     return replaceInstUsesWith(I,cos_add);
  //   }
  // }


  /*
  // double的加减乘除转flaot
  if (match(&I, m_FAdd(m_Value(X), m_Value(Y)))){
    Type* XTy=X->getType();
    Type* YTy=Y->getType();
    if(XTy->isDoubleTy()&&YTy->isDoubleTy()){
      //errs()<<"both double\n";
      Value *fptrunc1 = Builder.CreateFPTrunc(X, Type::getFloatTy(XTy->getContext()));
      Value *fptrunc2 = Builder.CreateFPTrunc(Y, Type::getFloatTy(YTy->getContext()));
      Value *faddFloat = Builder.CreateFAdd(fptrunc1, fptrunc2);
			//Value *faddExt = Builder.CreateFPExt(faddFloat, Type::getDoubleTy(YTy->getContext()));
      errs()<<"add double to add float success\n";
      //return replaceInstUsesWith(I,faddExt);
      return CastInst::CreateFPCast(faddFloat,Type::getDoubleTy(YTy->getContext()));
    }
  }

  if (match(&I, m_FSub(m_Value(X), m_Value(Y)))){
    Type* XTy=X->getType();
    Type* YTy=Y->getType();
    if(XTy->isDoubleTy()&&YTy->isDoubleTy()){
      errs()<<"both double\n";
      Value *fptrunc1 = Builder.CreateFPTrunc(X, Type::getFloatTy(XTy->getContext()));
      Value *fptrunc2 = Builder.CreateFPTrunc(Y, Type::getFloatTy(YTy->getContext()));
      Value *fsubFloat = Builder.CreateFSub(fptrunc1, fptrunc2);
			//Value *fsubExt = Builder.CreateFPExt(fsubFloat, Type::getDoubleTy(YTy->getContext()));
      errs()<<"sub double to sub float success\n";
      //return replaceInstUsesWith(I,fsubExt);
      return CastInst::CreateFPCast(fsubFloat, Type::getDoubleTy(YTy->getContext()));
    }
  }

  if (match(&I, m_FMul(m_Value(X), m_Value(Y)))){
    Type* XTy=X->getType();
    Type* YTy=Y->getType();
    if(XTy->isDoubleTy()&&YTy->isDoubleTy()){
      errs()<<"both double\n";
      Value *fptrunc1 = Builder.CreateFPTrunc(X, Type::getFloatTy(XTy->getContext()));
      Value *fptrunc2 = Builder.CreateFPTrunc(Y, Type::getFloatTy(YTy->getContext()));
      Value *fmulFloat = Builder.CreateFMul(fptrunc1, fptrunc2);
      //Value *fmulExt = Builder.CreateFPExt(fmulFloat, Type::getDoubleTy(YTy->getContext()));
      errs()<<"mul double to mul float success\n";
      //return replaceInstUsesWith(I,fmulExt);
      return CastInst::CreateFPCast(fmulFloat, Type::getDoubleTy(YTy->getContext()));
    }
  }

  if (match(&I, m_FDiv(m_Value(X), m_Value(Y)))){
    Type* XTy=X->getType();
    Type* YTy=Y->getType();
    if(XTy->isDoubleTy()&&YTy->isDoubleTy()){
      errs()<<"both double\n";
      Value *fptrunc1 = Builder.CreateFPTrunc(X, Type::getFloatTy(XTy->getContext()));
      Value *fptrunc2 = Builder.CreateFPTrunc(Y, Type::getFloatTy(YTy->getContext()));
      Value *fdivFloat = Builder.CreateFDiv(fptrunc1, fptrunc2);
      //Value *fdivExt = Builder.CreateFPExt(fdivFloat, Type::getDoubleTy(YTy->getContext()));
      errs()<<"div double to div float success\n";
      return CastInst::CreateFPCast(fdivFloat, Type::getDoubleTy(YTy->getContext()));
    }
  }
  */
  return nullptr;
}
