# Exercise 01: Basic LLVM IR Generation

**Difficulty**: Intermediate  
**Estimated Time**: 2-3 hours  
**Topics**: LLVM IR, IRBuilder, Module Creation, Function Generation

## Learning Objectives

After completing this exercise, you will:
- Use LLVM C++ API to generate IR
- Create modules, functions, and basic blocks
- Generate arithmetic instructions
- Handle function calls and returns
- Understand SSA form in practice

## Problem Description

Write a C++ program that generates LLVM IR for various mathematical functions without parsing. You'll use the LLVM IRBuilder API directly to construct the IR.

### Functions to Generate

1. **add**: `int add(int a, int b) { return a + b; }`
2. **factorial**: Recursive factorial function
3. **fibonacci**: Iterative fibonacci with loop
4. **power**: `int power(int base, int exp)` using loops
5. **gcd**: Greatest common divisor using recursion

## Requirements

1. **Generate valid LLVM IR** for each function
2. **Verify** the IR using `verifyModule` and `verifyFunction`
3. **Print** the generated IR in human-readable format
4. **Create** a main function that calls these functions
5. **Handle** both recursive and iterative patterns

## Expected IR Output Examples

### Example 1: Add Function
```llvm
define i32 @add(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}
```

### Example 2: Factorial (Recursive)
```llvm
define i32 @factorial(i32 %n) {
entry:
  %cmp = icmp eq i32 %n, 0
  br i1 %cmp, label %base_case, label %recursive_case

base_case:
  ret i32 1

recursive_case:
  %n_minus_1 = sub i32 %n, 1
  %rec_result = call i32 @factorial(i32 %n_minus_1)
  %result = mul i32 %n, %rec_result
  ret i32 %result
}
```

### Example 3: Fibonacci (Iterative with Loop)
```llvm
define i32 @fibonacci(i32 %n) {
entry:
  %cmp = icmp sle i32 %n, 1
  br i1 %cmp, label %base, label %loop_init

base:
  ret i32 %n

loop_init:
  br label %loop

loop:
  %i = phi i32 [ 2, %loop_init ], [ %next_i, %loop_body ]
  %fib_prev = phi i32 [ 0, %loop_init ], [ %fib_curr, %loop_body ]
  %fib_curr = phi i32 [ 1, %loop_init ], [ %fib_next, %loop_body ]
  
  %cmp_loop = icmp slt i32 %i, %n
  br i1 %cmp_loop, label %loop_body, label %loop_end

loop_body:
  %fib_next = add i32 %fib_prev, %fib_curr
  %next_i = add i32 %i, 1
  br label %loop

loop_end:
  %fib_next_end = add i32 %fib_prev, %fib_curr
  ret i32 %fib_next_end
}
```

## Starter Code

```cpp
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace llvm;

class IRGenerator {
private:
    LLVMContext context;
    std::unique_ptr<Module> module;
    IRBuilder<> builder;
    
public:
    IRGenerator() 
        : module(std::make_unique<Module>("my_module", context)),
          builder(context) {}
    
    // Generate add function: int add(int a, int b)
    Function* generateAdd() {
        // TODO: Implement
        // 1. Create function type: i32 (i32, i32)
        // 2. Create function
        // 3. Create entry basic block
        // 4. Generate add instruction
        // 5. Generate return
        return nullptr;
    }
    
    // Generate factorial function (recursive)
    Function* generateFactorial() {
        // TODO: Implement
        // 1. Create function
        // 2. Create basic blocks: entry, base_case, recursive_case
        // 3. Check if n == 0
        // 4. Base case: return 1
        // 5. Recursive case: return n * factorial(n-1)
        return nullptr;
    }
    
    // Generate fibonacci function (iterative)
    Function* generateFibonacci() {
        // TODO: Implement
        // 1. Create function
        // 2. Handle base cases (n <= 1)
        // 3. Create loop structure
        // 4. Use phi nodes for loop variables
        // 5. Return result
        return nullptr;
    }
    
    // Generate power function: int power(int base, int exp)
    Function* generatePower() {
        // TODO: Implement
        // Use iterative multiplication in a loop
        return nullptr;
    }
    
    // Generate GCD function (recursive)
    Function* generateGCD() {
        // TODO: Implement
        // Use Euclidean algorithm: gcd(a, b) = gcd(b, a % b)
        return nullptr;
    }
    
    // Generate main function that calls the others
    Function* generateMain() {
        // TODO: Implement
        // Call each function with sample values
        // Print results (declare printf first)
        return nullptr;
    }
    
    // Verify and print the module
    void finalize() {
        // Verify module
        std::string errorInfo;
        raw_string_ostream errorStream(errorInfo);
        if (verifyModule(*module, &errorStream)) {
            errs() << "Module verification failed:\n";
            errs() << errorStream.str() << "\n";
            return;
        }
        
        // Print module
        module->print(outs(), nullptr);
    }
    
    Module* getModule() { return module.get(); }
};

int main() {
    IRGenerator generator;
    
    // Generate all functions
    generator.generateAdd();
    generator.generateFactorial();
    generator.generateFibonacci();
    generator.generatePower();
    generator.generateGCD();
    generator.generateMain();
    
    // Finalize and print
    generator.finalize();
    
    return 0;
}
```

## Detailed Steps

### Step 1: Create Add Function

```cpp
Function* IRGenerator::generateAdd() {
    // Define function type: i32 (i32, i32)
    Type* i32Type = Type::getInt32Ty(context);
    FunctionType* funcType = FunctionType::get(
        i32Type,                    // Return type
        {i32Type, i32Type},        // Parameter types
        false                       // Not variadic
    );
    
    // Create function
    Function* func = Function::Create(
        funcType,
        Function::ExternalLinkage,
        "add",
        module.get()
    );
    
    // Name the parameters
    func->getArg(0)->setName("a");
    func->getArg(1)->setName("b");
    
    // Create entry basic block
    BasicBlock* entry = BasicBlock::Create(context, "entry", func);
    builder.SetInsertPoint(entry);
    
    // Generate add instruction
    Value* sum = builder.CreateAdd(func->getArg(0), func->getArg(1), "sum");
    
    // Generate return
    builder.CreateRet(sum);
    
    // Verify function
    verifyFunction(*func);
    
    return func;
}
```

### Step 2: Hints for Other Functions

**Factorial**:
- Create three basic blocks: entry, base_case, recursive_case
- Use `CreateICmpEQ` for comparison
- Use `CreateCondBr` for conditional branch
- Use `CreateCall` for recursive call

**Fibonacci**:
- Use phi nodes (`CreatePHI`) for loop variables
- Create loop structure: loop_init, loop, loop_body, loop_end
- Track three values: counter, previous fib, current fib

**Power**:
- Initialize result to 1
- Loop from 0 to exp
- Multiply result by base each iteration

**GCD**:
- Base case: if b == 0, return a
- Recursive case: return gcd(b, a % b)
- Use `CreateSRem` for modulo operation

## Test Cases

After generating IR, you should be able to:

```bash
# Save output to file
./ir_generator > output.ll

# Verify with LLVM tools
llvm-as output.ll -o output.bc

# Run with lli
lli output.bc

# Compile to native code
llc output.ll -o output.s
gcc output.s -o output
./output
```

Expected outputs:
```
add(5, 3) = 8
factorial(5) = 120
fibonacci(10) = 55
power(2, 10) = 1024
gcd(48, 18) = 6
```

## Bonus Challenges

1. **Optimization**: Add function attributes like `alwaysinline`, `readonly`
2. **Debug Info**: Generate debug information for functions
3. **Tail Recursion**: Implement tail call optimization for factorial
4. **Array Operations**: Create functions that work with arrays
5. **String Operations**: Implement string length and concatenation
6. **JIT Execution**: Use LLVM JIT to execute the generated code directly

## Common Pitfalls

1. **Forgetting to set insert point**: Always call `SetInsertPoint` before creating instructions
2. **Not handling phi nodes correctly**: Remember incoming values must match predecessors
3. **Type mismatches**: Ensure all operations use compatible types
4. **Missing terminator**: Every basic block needs a terminator (ret, br, etc.)
5. **Not verifying**: Always verify functions and modules

## Debugging Tips

1. **Print IR incrementally**: Print after each function to catch errors early
2. **Use verifyFunction**: Check each function immediately after creation
3. **Check basic blocks**: Ensure all blocks have terminators
4. **Visualize CFG**: Use `opt -dot-cfg` to visualize control flow
5. **Test with opt**: Run optimization passes to ensure IR is valid

## Building and Running

```bash
# Compile (adjust LLVM paths as needed)
clang++ -std=c++17 $(llvm-config --cxxflags --ldflags --system-libs --libs core) \
    ir_generation_exercise.cpp -o ir_gen

# Run
./ir_gen > output.ll

# Verify
llvm-as output.ll

# Execute
lli output.ll
```

## Expected Learning Outcomes

- Understanding of LLVM IR structure
- Proficiency with IRBuilder API
- Knowledge of SSA form and phi nodes
- Ability to generate loops and recursion
- Skills in debugging generated IR

## Solution Location

Complete solution available in `solutions/01_ir_generation.cpp` with detailed comments.

## Next Exercise

Move on to **Exercise 02: Optimization Pass** to learn how to write LLVM transformation passes.
