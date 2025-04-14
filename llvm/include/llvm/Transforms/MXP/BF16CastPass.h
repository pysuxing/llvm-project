#ifndef BF_CAST_PASS_H
#define BF_CAST_PASS_H

#include <llvm/Pass.h>
// #include <llvm/IR/Function.h>
// #include <llvm/Support/raw_ostream.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
// #include <llvm/Support/CommandLine.h>

#include <set>

using namespace std;
using namespace llvm;

namespace llvm {

extern cl::opt<bool> EnableBF16ToFP32;

class BF16CastPass : public PassInfoMixin<BF16CastPass> {
public:
  PreservedAnalyses run(Module &module,
                        ModuleAnalysisManager &); // for New Pass Manager
  virtual bool runOnModule(Module &module);

private:
  template <typename T>
  void ChangeInstToCall(T *I, const std::string &FunctionName,
                        const std::vector<Value *> &Args, Type *ReturnType);
  vector<Instruction *> erase;
};

void addBF16ToFP32Pass(ModulePassManager &MPM, OptimizationLevel Level);

} // namespace llvm

#endif // BF_CAST_PASS_H
