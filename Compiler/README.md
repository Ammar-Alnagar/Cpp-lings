# Compiler Design and Implementation

This directory contains comprehensive educational materials for understanding compiler design, with a focus on modern compiler infrastructure using LLVM and MLIR.

## Structure

The compiler curriculum is organized into three progressive modules:

### Module 1: Compiler Theory Fundamentals
Learn the theoretical foundations of compiler design, from lexical analysis to code generation.

### Module 2: LLVM Infrastructure
Practical introduction to LLVM, building custom compilers and understanding the LLVM IR.

### Module 3: MLIR (Multi-Level Intermediate Representation)
Advanced topics in MLIR for building domain-specific compilers and optimizations.

## Directory Organization

```
Compiler/
├── theory/                      # Compiler theory fundamentals
│   ├── examples/               # Complete implementations
│   ├── exercises/              # Practice problems
│   ├── 01_introduction.md
│   ├── 02_lexical_analysis.md
│   ├── 03_parsing.md
│   ├── 04_semantic_analysis.md
│   ├── 05_intermediate_representation.md
│   └── 06_code_generation.md
│
├── llvm/                        # LLVM infrastructure
│   ├── examples/               # LLVM code examples
│   ├── exercises/              # LLVM exercises
│   ├── 01_llvm_introduction.md
│   ├── 02_llvm_ir.md
│   ├── 03_building_frontend.md
│   ├── 04_optimization_passes.md
│   └── 05_backend_codegen.md
│
├── mlir/                        # MLIR framework
│   ├── examples/               # MLIR examples
│   ├── exercises/              # MLIR exercises
│   ├── 01_mlir_introduction.md
│   ├── 02_dialects.md
│   ├── 03_transformations.md
│   └── 04_lowering_strategies.md
│
├── Makefile                    # Build system
├── CMakeLists.txt             # CMake configuration
└── README.md                  # This file
```

## Prerequisites

Before starting this curriculum, you should have:

### Required Knowledge
- Strong C/C++ programming skills
- Understanding of data structures and algorithms
- Basic knowledge of assembly language
- Familiarity with command-line tools

### System Requirements
- C++17 or later compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- CMake 3.15 or later
- LLVM 14+ (for LLVM modules)
- MLIR (included with LLVM 14+)
- Python 3.8+ (for some tools)
- 8GB+ RAM recommended
- Linux, macOS, or Windows with WSL2

## Installing LLVM and MLIR

### Option 1: Pre-built Binaries (Recommended for Learning)

#### Ubuntu/Debian
```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 17
```

#### macOS
```bash
brew install llvm@17
```

#### Arch Linux
```bash
sudo pacman -S llvm mlir
```

### Option 2: Build from Source (Advanced)

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="clang;mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON
ninja
sudo ninja install
```

## Building Examples

### Using Make

```bash
# Build all examples
make all

# Build specific module
make theory
make llvm
make mlir

# Clean
make clean
```

### Using CMake

```bash
mkdir build && cd build
cmake ..
make
```

## Learning Path

### Recommended Order (Beginner to Expert)

1. **Compiler Theory Fundamentals** (2-3 weeks)
   - Start with introduction and history
   - Learn lexical analysis and implement a lexer
   - Study parsing techniques and build a parser
   - Understand semantic analysis
   - Learn about intermediate representations
   - Study code generation basics

2. **LLVM Infrastructure** (3-4 weeks)
   - Introduction to LLVM architecture
   - Understanding LLVM IR
   - Building a simple frontend
   - Writing optimization passes
   - Backend code generation

3. **MLIR Framework** (2-3 weeks)
   - MLIR concepts and motivation
   - Understanding dialects
   - Writing transformations
   - Lowering strategies
   - Building domain-specific compilers

### Alternative Paths

**For Practical Focus:**
- Quick theory overview (Chapter 1-2)
- Jump to LLVM practical examples
- Return to theory as needed

**For Research/Advanced:**
- Complete theory module thoroughly
- LLVM with focus on passes
- Deep dive into MLIR for research applications

## Module Descriptions

### Compiler Theory Fundamentals

This module provides a solid theoretical foundation:

- **Lexical Analysis**: Tokenization, regular expressions, finite automata
- **Parsing**: Context-free grammars, LL/LR parsing, AST construction
- **Semantic Analysis**: Type checking, symbol tables, scope resolution
- **Intermediate Representation**: Three-address code, SSA form, control flow graphs
- **Optimization**: Data flow analysis, common optimizations
- **Code Generation**: Instruction selection, register allocation, instruction scheduling

Each chapter includes:
- Detailed theoretical explanations
- Mathematical foundations
- Algorithm descriptions
- Complete working implementations
- Visual diagrams and examples

### LLVM Infrastructure

Practical guide to using LLVM:

- **LLVM Architecture**: Modular design, pass infrastructure, target descriptions
- **LLVM IR**: SSA form, type system, instructions, modules
- **Frontend Development**: Lexing, parsing, AST to IR translation
- **Optimization Passes**: Analysis passes, transformation passes, pass manager
- **Backend**: Instruction selection, register allocation, assembly emission

Features:
- Working compiler implementations
- Real-world examples
- Integration with existing tools
- Performance considerations
- Debugging techniques

### MLIR Framework

Modern compiler infrastructure:

- **MLIR Concepts**: Multi-level IR, dialects, operations, regions
- **Dialect Design**: Creating custom dialects, operations, types
- **Transformations**: Rewrite patterns, canonicalization, folding
- **Lowering**: Progressive lowering, conversion patterns
- **Integration**: Using MLIR with LLVM, JIT compilation

Advanced topics:
- Domain-specific languages
- Hardware synthesis
- Machine learning compilers
- Heterogeneous computing

## Code Quality Standards

All code in this curriculum follows:

- **Modern C++ Standards**: C++17/20 features
- **LLVM Coding Style**: Follows LLVM coding conventions
- **Comprehensive Comments**: Every non-trivial operation explained
- **Working Examples**: All code compiles and runs
- **Error Handling**: Proper error reporting and diagnostics
- **Testing**: Unit tests for critical components

## Teaching Philosophy

This curriculum is designed with these principles:

1. **Theory Meets Practice**: Understand why before implementing how
2. **Progressive Complexity**: Build knowledge incrementally
3. **Complete Examples**: No pseudo-code, everything works
4. **Real-World Focus**: Examples based on actual use cases
5. **Modern Tools**: Use current versions of LLVM/MLIR
6. **Visual Learning**: Diagrams and step-by-step transformations
7. **Hands-On**: Learn by building actual compilers

## Compiler Projects

As you progress, you will build:

### Theory Module Projects
1. Lexer for a simple language
2. Recursive descent parser
3. Type checker
4. Simple interpreter
5. Stack-based VM with bytecode compiler

### LLVM Module Projects
1. Calculator language compiler
2. Simple imperative language (like C subset)
3. Custom optimization pass
4. JIT compiler for expressions
5. Backend for custom architecture

### MLIR Module Projects
1. Toy language with MLIR
2. Custom dialect for domain-specific operations
3. Tensor computation dialect
4. Progressive lowering pipeline
5. Hardware description language frontend

## Resources and References

### Books
- "Compilers: Principles, Techniques, and Tools" (Dragon Book)
- "Modern Compiler Implementation in C" (Tiger Book)
- "Engineering a Compiler" by Cooper and Torczon
- "LLVM Cookbook" by Suyog Sarda and Mayur Pandey

### Online Resources
- LLVM Documentation: https://llvm.org/docs/
- MLIR Documentation: https://mlir.llvm.org/
- LLVM Blog: https://blog.llvm.org/
- Compiler Explorer: https://godbolt.org/

### Papers
- "LLVM: A Compilation Framework for Lifelong Program Analysis"
- "MLIR: A Compiler Infrastructure for the End of Moore's Law"
- "Simple and Efficient Construction of SSA Form"

## Performance Considerations

Compiler development involves understanding:
- Time complexity of algorithms (lexing, parsing, optimization)
- Space complexity (symbol tables, CFGs, SSA construction)
- Trade-offs between compile time and runtime performance
- Optimization levels and their impact

## Common Tools

You will work with:
- `clang`: C/C++ frontend for LLVM
- `llc`: LLVM static compiler
- `opt`: LLVM optimizer
- `lli`: LLVM interpreter and JIT
- `mlir-opt`: MLIR optimizer driver
- `mlir-translate`: MLIR translation tool

## Debugging Techniques

Learn to debug compilers:
- Dumping intermediate representations
- Visualization tools (CFG, dominator trees)
- Using LLVM/MLIR debugging flags
- Understanding error messages
- Regression testing

## Success Metrics

After completing this curriculum:

**Theory Module**: Understand all compilation phases and their algorithms
**LLVM Module**: Build a working compiler frontend and custom passes
**MLIR Module**: Create custom dialects and lowering pipelines

## Getting Help

When stuck:
1. Review the relevant chapter
2. Study the example code carefully
3. Check LLVM/MLIR documentation
4. Enable verbose/debug output
5. Start with minimal reproducible examples

## Next Steps

1. Verify LLVM/MLIR installation
2. Review prerequisites
3. Choose your learning path
4. Start with `theory/01_introduction.md`
5. Build and run examples as you progress

## License

This educational material is provided under the MIT License.

## Contributing

This is an educational curriculum. Focus on learning and understanding the concepts thoroughly.

---

**Ready to build compilers? Let's start with compiler theory fundamentals!**
