// LLVM Introduction: Hello World IR Generator
// This example demonstrates basic LLVM IR generation using the LLVM C++ API
// Generates a simple "Hello, World!" program

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <vector>

using namespace llvm;

// This example generates IR equivalent to:
//
// @.str = private unnamed_addr constant [14 x i8] c"Hello, LLVM!\0A\00"
//
// declare i32 @puts(i8*)
//
// define i32 @main() {
// entry:
//   %call = call i32 @puts(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str, i32 0, i32 0))
//   ret i32 0
// }

int main() {
    // Step 1: Create LLVM context
    // Context holds type system and other global state
    LLVMContext context;
    
    // Step 2: Create a module
    // Module is a container for all IR (functions, globals, etc.)
    auto module = std::make_unique<Module>("hello_llvm", context);
    
    // Step 3: Create IRBuilder
    // IRBuilder is a helper for generating IR instructions
    IRBuilder<> builder(context);
    
    // Step 4: Create types we'll need
    Type* int32Type = Type::getInt32Ty(context);
    Type* int8PtrType = Type::getInt8PtrTy(context);
    
    // Step 5: Declare external puts function
    // int puts(const char* str)
    FunctionType* putsFuncType = FunctionType::get(
        int32Type,              // return type
        {int8PtrType},         // parameter types
        false                   // is variadic?
    );
    
    FunctionCallee putsFunc = module->getOrInsertFunction("puts", putsFuncType);
    
    // Step 6: Create string constant "Hello, LLVM!\n"
    Constant* helloStr = builder.CreateGlobalStringPtr("Hello, LLVM!\n", ".str");
    
    // Step 7: Create main function
    // int main()
    FunctionType* mainFuncType = FunctionType::get(
        int32Type,              // return type
        {},                     // no parameters
        false                   // not variadic
    );
    
    Function* mainFunc = Function::Create(
        mainFuncType,
        Function::ExternalLinkage,
        "main",
        module.get()
    );
    
    // Step 8: Create entry basic block
    BasicBlock* entryBlock = BasicBlock::Create(context, "entry", mainFunc);
    
    // Step 9: Position builder at end of entry block
    builder.SetInsertPoint(entryBlock);
    
    // Step 10: Create call to puts
    Value* putsCall = builder.CreateCall(putsFunc, {helloStr}, "call");
    
    // Step 11: Return 0
    builder.CreateRet(ConstantInt::get(int32Type, 0));
    
    // Step 12: Verify the module
    std::string errorInfo;
    raw_string_ostream errorStream(errorInfo);
    if (verifyModule(*module, &errorStream)) {
        errs() << "Error: Module verification failed\n";
        errs() << errorStream.str() << "\n";
        return 1;
    }
    
    // Step 13: Print the generated IR
    outs() << "=== Generated LLVM IR ===\n\n";
    module->print(outs(), nullptr);
    
    outs() << "\n=== IR Generation Complete ===\n";
    outs() << "You can save this to a .ll file and compile with:\n";
    outs() << "  llc output.ll -o output.s\n";
    outs() << "  gcc output.s -o output\n";
    
    return 0;
}
