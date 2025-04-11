#ifndef FP128_TO_DD_PASS_H
#define FP128_TO_DD_PASS_H

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
    class FP128ToDDPass : public PassInfoMixin<FP128ToDDPass> {
    public:
        PreservedAnalyses run(Module &module, ModuleAnalysisManager &); // for New Pass Manager
        virtual bool runOnModule(Module &module);
    private:
	template <typename T> void ChangeInstToCall(T *I, const std::string &FunctionName, const std::vector<Value*> &Args, Type *ReturnType);
	vector<Instruction*> erase;
    };
    
} // namespace llvm

#endif // FP128_TO_DD_PASS_H
