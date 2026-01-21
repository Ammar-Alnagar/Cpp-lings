# Chapter 1: Introduction to Compilers

## What is a Compiler?

A compiler is a program that translates source code written in a high-level programming language into a lower-level language, typically machine code or assembly language. The compiler enables programmers to write in human-readable languages while producing efficient executable programs.

## Compilers vs Interpreters

### Compiler
- Translates entire program before execution
- Produces standalone executable
- Generally faster execution
- Better optimization opportunities
- Examples: C, C++, Rust compilers

### Interpreter
- Executes code line-by-line or statement-by-statement
- No separate executable produced
- Slower execution but faster startup
- Easier debugging and dynamic features
- Examples: Python, Ruby, JavaScript (traditional)

### Hybrid Approaches
Modern languages often use both:
- Java: Compiles to bytecode, then JIT compiles to machine code
- JavaScript: V8 engine compiles hot code paths
- Python: PyPy uses JIT compilation

## Structure of a Compiler

A typical compiler consists of several phases:

```
Source Code
    ↓
[Lexical Analysis] → Tokens
    ↓
[Syntax Analysis] → Abstract Syntax Tree (AST)
    ↓
[Semantic Analysis] → Annotated AST
    ↓
[Intermediate Code Generation] → IR
    ↓
[Optimization] → Optimized IR
    ↓
[Code Generation] → Target Code
    ↓
Target Machine Code
```

### Frontend

The frontend analyzes the source code and builds an intermediate representation:

1. **Lexical Analysis (Scanner)**
   - Input: Character stream
   - Output: Token stream
   - Task: Group characters into meaningful lexemes
   - Example: "int x = 5;" → [INT, IDENTIFIER("x"), EQUALS, NUMBER(5), SEMICOLON]

2. **Syntax Analysis (Parser)**
   - Input: Token stream
   - Output: Abstract Syntax Tree (AST)
   - Task: Check grammatical structure
   - Example: Build tree representing "x = a + b * c"

3. **Semantic Analysis**
   - Input: AST
   - Output: Annotated AST with type information
   - Task: Check semantic rules (type checking, scope, etc.)
   - Example: Verify "x = a + b" where all have compatible types

### Middle End

Performs optimizations on intermediate representation:

1. **Intermediate Representation (IR)**
   - Platform-independent code representation
   - Examples: Three-address code, SSA form, LLVM IR

2. **Optimization**
   - Constant folding: 2 + 3 → 5
   - Dead code elimination: Remove unused code
   - Loop optimizations: Unrolling, invariant code motion
   - Inlining: Replace function calls with function body

### Backend

Generates target machine code:

1. **Instruction Selection**
   - Map IR operations to target machine instructions
   - Handle different addressing modes

2. **Register Allocation**
   - Assign variables to physical registers
   - Handle spills when registers run out

3. **Instruction Scheduling**
   - Reorder instructions for better performance
   - Minimize pipeline stalls

4. **Machine Code Emission**
   - Generate actual binary code
   - Handle relocations and linking

## Symbol Table

The symbol table is a data structure used throughout compilation:

- Stores information about identifiers (variables, functions, types)
- Tracks scope and visibility
- Stores type information
- Used for semantic checking and code generation

Example symbol table entry:
```
Name: "variable_x"
Type: int
Scope: function_main
Offset: 8 (from frame pointer)
```

## Error Handling

Compilers must handle errors gracefully:

### Lexical Errors
- Invalid characters
- Malformed tokens
- Example: 123abc (invalid identifier)

### Syntax Errors
- Grammatical mistakes
- Missing tokens
- Example: if (x > 0 { // missing closing parenthesis

### Semantic Errors
- Type mismatches
- Undefined variables
- Scope violations
- Example: int x = "hello"; // type error

### Error Recovery
Good compilers continue after errors to find multiple issues:
- Panic mode: Skip until synchronization point
- Phrase-level recovery: Local corrections
- Error productions: Handle common mistakes
- Global correction: Minimal changes to fix

## Compiler Architecture

### Single-Pass Compiler
- Processes source code in one traversal
- Generates code directly
- Fast but limited optimization
- Example: Early Pascal compilers

### Multi-Pass Compiler
- Multiple traversals of code/IR
- Better optimization opportunities
- Slower compilation but better code
- Example: Modern optimizing compilers

### Just-In-Time (JIT) Compiler
- Compiles code at runtime
- Can use runtime information for optimization
- Trade-off between compilation and execution time
- Example: Java HotSpot, V8 JavaScript engine

## Compilation Models

### Ahead-of-Time (AOT) Compilation
```
Source Code → Compiler → Native Executable
```
- Traditional compilation
- One-time cost
- Platform-specific output

### Bytecode Compilation
```
Source Code → Compiler → Bytecode → VM/Interpreter
```
- Platform-independent bytecode
- Smaller distribution size
- Interpreted or JIT compiled

### Incremental Compilation
```
Change Detection → Compile Changed Units → Link
```
- Only recompile what changed
- Faster development cycles
- Example: C/C++ separate compilation

## Historical Context

### Early Compilers (1950s-1960s)
- FORTRAN I (1957): First high-level language compiler
- Made programming practical for non-experts
- Manual optimization techniques

### Structured Programming Era (1970s-1980s)
- Formal language theory applied
- Parser generators (YACC, Lex)
- Optimization theory developed

### Modern Era (1990s-Present)
- LLVM: Modular compiler infrastructure
- JIT compilation widespread
- Advanced optimizations (vectorization, polyhedral)
- Domain-specific compilation (GPUs, ML accelerators)

## Why Study Compilers?

### Practical Benefits
1. **Better Programming Skills**
   - Understand language semantics deeply
   - Write more efficient code
   - Debug complex issues

2. **Tool Building**
   - Create domain-specific languages (DSLs)
   - Build code analysis tools
   - Develop transpilers

3. **Performance Optimization**
   - Understand compiler optimizations
   - Write compiler-friendly code
   - Profile and optimize effectively

### Theoretical Benefits
1. **Algorithms and Data Structures**
   - Graph algorithms (CFG, dominator trees)
   - Hash tables (symbol tables)
   - Complex tree operations (AST manipulation)

2. **Formal Languages**
   - Regular expressions
   - Context-free grammars
   - Automata theory

3. **Problem Solving**
   - Complex system design
   - Trade-off analysis
   - Optimization techniques

## Compiler Toolchains

### GCC (GNU Compiler Collection)
- Mature, widely-used
- Supports many languages (C, C++, Fortran, etc.)
- Multiple target architectures
- Heavy optimization focus

### Clang/LLVM
- Modern, modular architecture
- Excellent error messages
- Fast compilation
- Easy to extend and embed

### MSVC (Microsoft Visual C++)
- Windows-focused
- Integrated with Visual Studio
- Good debugging support
- Windows-specific optimizations

## Compilation Pipeline Example

Let's trace a simple program through compilation:

### Source Code
```c
int main() {
    int x = 5;
    int y = 10;
    return x + y;
}
```

### After Lexical Analysis
```
INT IDENTIFIER(main) LPAREN RPAREN LBRACE
INT IDENTIFIER(x) EQUALS NUMBER(5) SEMICOLON
INT IDENTIFIER(y) EQUALS NUMBER(10) SEMICOLON
RETURN IDENTIFIER(x) PLUS IDENTIFIER(y) SEMICOLON
RBRACE
```

### After Parsing (Simplified AST)
```
Function: main
  Body:
    VarDecl: x = 5
    VarDecl: y = 10
    Return: BinaryOp(+)
      Left: Identifier(x)
      Right: Identifier(y)
```

### After Semantic Analysis
```
Function: main returns int
  Body:
    VarDecl: x:int = 5:int
    VarDecl: y:int = 10:int
    Return: BinaryOp(+):int
      Left: Identifier(x):int
      Right: Identifier(y):int
```

### Intermediate Representation (Three-Address Code)
```
t1 = 5
t2 = 10
t3 = t1 + t2
return t3
```

### After Optimization (Constant Folding)
```
return 15
```

### Assembly Code (x86-64)
```asm
main:
    mov eax, 15
    ret
```

## Modern Compiler Challenges

### Performance
- Multi-core processors
- SIMD vectorization
- Memory hierarchy optimization
- Power efficiency

### Correctness
- Undefined behavior detection
- Memory safety (bounds checking, use-after-free)
- Concurrency bugs (race conditions)
- Formal verification

### Usability
- Fast compilation times
- Clear error messages
- Good debugging information
- IDE integration

### Emerging Domains
- Machine learning accelerators
- Quantum computing
- Heterogeneous systems
- Domain-specific architectures

## LLVM and MLIR Preview

### LLVM
- Modular compiler infrastructure
- Reusable components
- Well-defined IR
- Extensive optimization passes
- Multiple backend targets

Key concepts:
- Modules, functions, basic blocks
- SSA form
- Pass infrastructure
- Target descriptions

### MLIR
- Multi-Level Intermediate Representation
- Extensible dialect system
- Progressive lowering
- Domain-specific abstractions

Use cases:
- Machine learning compilers (TensorFlow, PyTorch)
- Hardware synthesis
- Domain-specific languages
- Heterogeneous computing

## Summary

Key concepts covered:
- Compiler structure and phases
- Frontend, middle-end, backend
- Compilation models (AOT, JIT, bytecode)
- Historical context and modern challenges
- Why studying compilers matters
- Preview of LLVM and MLIR

## Next Steps

In Chapter 2, we will dive into Lexical Analysis, learning how to transform character streams into tokens using regular expressions and finite automata. We will implement a complete lexer for a simple language.

## Further Reading

- "Compilers: Principles, Techniques, and Tools" by Aho et al. (Dragon Book)
- "Modern Compiler Implementation in C" by Appel (Tiger Book)
- LLVM documentation: https://llvm.org/docs/
- MLIR documentation: https://mlir.llvm.org/
