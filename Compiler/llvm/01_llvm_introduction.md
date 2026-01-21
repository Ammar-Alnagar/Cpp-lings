# Chapter 1: Introduction to LLVM

## What is LLVM?

LLVM (Low Level Virtual Machine) is a modern compiler infrastructure designed with reusability and modularity in mind. Despite its name, LLVM is not a traditional virtual machine but rather a collection of modular compiler and toolchain technologies.

## History and Evolution

### Origins (2000-2003)
- Started by Chris Lattner at the University of Illinois
- Originally a research project for lifelong program analysis
- Goal: Enable sophisticated optimization throughout program lifetime

### Growth (2003-2010)
- Apple hired Chris Lattner in 2005
- Used to build Clang C/C++ compiler
- Became production-ready compiler infrastructure

### Modern Era (2010-Present)
- Foundation of many major compilers (Swift, Rust, Julia)
- Used by Apple, Google, Microsoft, Intel, AMD
- MLIR project added in 2019
- Foundation for ML compilers (TensorFlow, PyTorch)

## LLVM Architecture

LLVM follows a three-phase design:

```
Source Code
     ↓
[Frontend] → LLVM IR
     ↓
[Optimizer] → Optimized LLVM IR
     ↓
[Backend] → Machine Code
```

### Frontend
- Language-specific parsing and analysis
- Generates LLVM IR from source code
- Examples: Clang (C/C++), Swift frontend, Rust frontend

### Optimizer
- Language-agnostic optimization passes
- Works on LLVM IR
- Modular pass infrastructure
- Examples: Inlining, constant folding, loop optimization

### Backend
- Target-specific code generation
- Instruction selection, register allocation
- Supports multiple architectures: x86, ARM, RISC-V, etc.

## Key Components

### LLVM Core Libraries

1. **LLVM IR**
   - Intermediate representation
   - Low-level, RISC-like instruction set
   - Type system with static types
   - SSA (Static Single Assignment) form

2. **Pass Infrastructure**
   - Modular optimization framework
   - Analysis passes (compute information)
   - Transformation passes (modify IR)
   - Pass managers (control execution)

3. **Target Description**
   - TableGen-based target descriptions
   - Register definitions
   - Instruction patterns
   - Scheduling models

### Frontend Tools

1. **Clang**
   - C/C++/Objective-C compiler
   - Modern architecture, excellent diagnostics
   - Fast compilation, low memory usage
   - Compatible with GCC

2. **LLDB**
   - Debugger built on LLVM infrastructure
   - Supports C, C++, Objective-C, Swift
   - Expression evaluation using Clang

### Backend Tools

1. **llc**
   - LLVM static compiler
   - Converts LLVM IR to assembly or object code
   - Target selection and code generation

2. **lli**
   - LLVM interpreter and JIT compiler
   - Execute LLVM IR directly
   - Useful for prototyping

3. **opt**
   - LLVM optimizer driver
   - Runs optimization passes on LLVM IR
   - Testing and debugging passes

### Analysis and Utilities

1. **llvm-as / llvm-dis**
   - Assembler and disassembler for LLVM IR
   - Human-readable (.ll) ↔ Bitcode (.bc)

2. **llvm-link**
   - Linker for LLVM bitcode files
   - Combines multiple modules

3. **llvm-nm**
   - Symbol table dumper
   - List symbols in bitcode/object files

4. **llvm-objdump**
   - Object file dumper
   - Disassemble and inspect object files

## Why Use LLVM?

### Advantages

1. **Modularity**
   - Reusable components
   - Build only what you need
   - Easy to extend and customize

2. **Well-Designed IR**
   - Clear semantics
   - Language-independent
   - Optimizable representation

3. **Powerful Optimization**
   - State-of-the-art optimizations
   - Continuous improvement
   - Proven in production

4. **Multi-Target Support**
   - Write frontend once
   - Generate code for many architectures
   - Consistent optimization across targets

5. **Active Community**
   - Large, active developer community
   - Extensive documentation
   - Regular releases
   - Commercial and research use

6. **Permissive License**
   - Apache 2.0 with LLVM Exception
   - Commercial-friendly
   - No copyleft requirements

### Use Cases

1. **Building Programming Languages**
   - Swift, Rust, Julia, Crystal
   - Implement frontend, use LLVM backend
   - Focus on language design, not codegen

2. **JIT Compilation**
   - Dynamic languages (JavaScript V8, Python PyPy)
   - Database query engines
   - GPU shader compilation

3. **Static Analysis**
   - Code analysis tools
   - Sanitizers (AddressSanitizer, ThreadSanitizer)
   - Custom static analyzers

4. **Code Transformation**
   - Obfuscation
   - Instrumentation
   - Profile-guided optimization

5. **Research**
   - Compiler research
   - Program analysis
   - Novel optimization techniques

## LLVM IR Overview

### Design Principles

1. **SSA Form**
   - Each variable assigned exactly once
   - Phi nodes for merging values
   - Enables powerful analysis and optimization

2. **Type System**
   - Statically typed
   - Primitive types: integers, floats, pointers
   - Aggregate types: arrays, structures, vectors

3. **RISC-like Instructions**
   - Simple, orthogonal instruction set
   - Explicit data flow
   - Target-independent

4. **Infinite Registers**
   - Virtual registers in IR
   - Backend handles physical register allocation
   - Simplifies optimization

### IR Forms

1. **Text Format (.ll)**
```llvm
define i32 @add(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}
```

2. **Bitcode Format (.bc)**
- Binary representation
- Compact, fast to parse
- Used for distribution

3. **In-Memory Format**
- C++ objects (Module, Function, BasicBlock, Instruction)
- Used by compiler passes
- Manipulated through API

### Basic Structure

```
Module
  └── Function
        └── Basic Block
              └── Instruction
```

**Module**: Complete translation unit
- Global variables
- Function declarations and definitions
- Target information

**Function**: Corresponds to functions in source
- Function signature (parameters, return type)
- Attributes
- Basic blocks

**Basic Block**: Sequence of instructions with single entry and exit
- Label (optional)
- Instructions
- Terminator (branch, return, etc.)

**Instruction**: Single operation
- Arithmetic, logic, memory, control flow
- Uses and defines virtual registers

## LLVM API

LLVM provides C++ APIs for:

### IRBuilder
High-level interface for generating IR:
```cpp
IRBuilder<> builder(context);
Value* sum = builder.CreateAdd(a, b, "sum");
builder.CreateRet(sum);
```

### Module
Container for all IR:
```cpp
Module module("my_module", context);
Function* func = Function::Create(
    funcType, Function::ExternalLinkage, "my_func", module);
```

### Pass Infrastructure
Write custom optimization passes:
```cpp
struct MyPass : public FunctionPass {
    bool runOnFunction(Function &F) override {
        // Analyze or transform function
        return modified;
    }
};
```

## LLVM Compilation Process

### Example: Compiling C to Assembly

1. **Source to IR** (Frontend)
```bash
clang -S -emit-llvm hello.c -o hello.ll
```

2. **Optimize IR**
```bash
opt -O3 hello.ll -S -o hello.opt.ll
```

3. **IR to Assembly** (Backend)
```bash
llc hello.opt.ll -o hello.s
```

4. **Assembly to Object**
```bash
as hello.s -o hello.o
```

5. **Link**
```bash
ld hello.o -o hello
```

### All-in-One Compilation
```bash
clang hello.c -o hello
```
Clang handles all phases internally.

## LLVM Optimization Levels

### -O0 (No Optimization)
- Fast compilation
- Good debugging
- Large code size
- Slow execution

### -O1 (Basic Optimization)
- Balance between compile time and code quality
- Basic optimizations only
- Still debuggable

### -O2 (Standard Optimization)
- Aggressive optimization
- Default for production
- Most optimizations enabled
- Reasonable compile time

### -O3 (Aggressive Optimization)
- Maximum optimization
- May increase code size
- Longer compilation
- Potential for aggressive inlining

### -Os (Size Optimization)
- Optimize for code size
- Disable size-increasing optimizations
- Good for embedded systems

### -Oz (Aggressive Size Optimization)
- More aggressive than -Os
- Minimize binary size

## LLVM Pass Types

### Analysis Passes
Compute information without modifying IR:
- Dominator tree
- Loop information
- Alias analysis
- Call graph

### Transformation Passes
Modify IR:
- Dead code elimination
- Constant propagation
- Loop unrolling
- Function inlining

### Module Passes
Operate on entire module:
- Inter-procedural optimization
- Global constant propagation
- Whole-program analysis

### Function Passes
Operate on single function:
- Most optimizations
- Can't see across function boundaries
- Faster than module passes

### Basic Block Passes
Operate on single basic block:
- Local optimizations
- Fast, but limited scope

## LLVM Development

### Building LLVM from Source

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build && cd build

cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="clang;lldb;lld" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON

ninja
ninja install
```

### Using LLVM in Your Project

CMake configuration:
```cmake
find_package(LLVM REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs core support)
target_link_libraries(my_compiler ${llvm_libs})
```

## LLVM vs Other Compiler Infrastructures

### GCC
- Monolithic design
- Harder to reuse components
- Better for C/C++ specifically
- Excellent optimization quality

### JVM
- Higher-level bytecode
- Language-specific (Java ecosystem)
- Strong runtime support
- Different design goals

### WebAssembly
- Web-focused
- Portable bytecode
- Sandboxed execution
- LLVM can target WebAssembly

## Real-World LLVM Projects

### Programming Languages
- **Swift**: Apple's modern language
- **Rust**: Systems programming language
- **Julia**: High-performance technical computing
- **Kotlin/Native**: Kotlin for native platforms

### Tools
- **Clang Static Analyzer**: Bug finding
- **Clang-Tidy**: Linter and fix-it tool
- **Clang-Format**: Code formatter
- **LLDB**: Debugger

### Compilers
- **Emscripten**: C/C++ to WebAssembly
- **TensorFlow**: ML model compiler
- **PyTorch**: Dynamic neural network compiler
- **numba**: Python JIT compiler

## Getting Started

### Prerequisites
- C++ programming experience
- Understanding of compiler basics
- Familiarity with command line

### Learning Path
1. Install LLVM
2. Explore LLVM IR (write, compile, inspect)
3. Use IRBuilder to generate IR programmatically
4. Write simple transformations
5. Build a toy compiler frontend
6. Study existing frontends (Kaleidoscope tutorial)

### Resources
- Official LLVM Documentation
- Kaleidoscope Tutorial (in LLVM docs)
- LLVM Developer Meetings (videos online)
- "Getting Started with LLVM Core Libraries"
- LLVM Discourse forums

## Summary

Key concepts covered:
- LLVM is modular compiler infrastructure
- Three-phase design: frontend, optimizer, backend
- LLVM IR as universal intermediate representation
- Rich set of tools and libraries
- Wide adoption in industry and research
- Active development and community

## Next Steps

In Chapter 2, we will dive deep into LLVM IR, learning its syntax, semantics, and how to read and write LLVM IR code.

## Code Examples

See the `examples/` directory for:
- Basic LLVM IR examples
- IRBuilder usage
- Simple compiler frontend
