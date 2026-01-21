# Getting Started with Compiler Education

Welcome to the Compiler Design curriculum! This guide will help you get started learning compiler theory, LLVM, and MLIR.

## Quick Start

### Step 1: Build Theory Examples (No Dependencies)

```bash
cd Compiler
make theory
```

This builds compiler theory examples (lexer, parser, etc.) that don't require any external dependencies.

### Step 2: Run Theory Examples

```bash
# Run the lexer example
./bin/theory_01_lexer

# This will tokenize several code examples
```

### Step 3: Install LLVM (Optional, for LLVM/MLIR examples)

#### Ubuntu/Debian
```bash
# Install LLVM 17 (or latest available)
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 17

# Install development packages
sudo apt install llvm-17-dev libllvm17 llvm-17-tools
```

#### macOS
```bash
brew install llvm
export PATH="/usr/local/opt/llvm/bin:$PATH"
```

#### Arch Linux
```bash
sudo pacman -S llvm mlir
```

### Step 4: Build LLVM Examples

```bash
make llvm
```

If LLVM is installed correctly, this will build LLVM examples.

### Step 5: Build Everything

```bash
make all
```

## Learning Paths

### Path 1: Complete Beginner (Recommended)

Start from theory and progress to practical implementation:

1. **Week 1-2: Compiler Theory Basics**
   - Read `theory/01_introduction.md`
   - Read `theory/02_lexical_analysis.md`
   - Study and run `bin/theory_01_lexer`
   - Understand tokenization and regular expressions

2. **Week 3-4: Continue Theory**
   - Study parsing (when additional chapters are added)
   - Learn about ASTs and semantic analysis
   - Implement simple interpreters

3. **Week 5-6: Introduction to LLVM**
   - Read `llvm/01_llvm_introduction.md`
   - Install LLVM
   - Run `bin/llvm_01_hello_llvm`
   - Learn LLVM IR syntax

4. **Week 7-8: Advanced LLVM**
   - Study LLVM IR in depth
   - Write optimization passes
   - Build simple compiler frontend

5. **Week 9-10: MLIR**
   - Read `mlir/01_mlir_introduction.md`
   - Understand dialects and multi-level IR
   - Study MLIR examples

### Path 2: Experienced Programmer (Fast Track)

If you have compiler background:

1. **Day 1: Theory Review**
   - Skim theory chapters for refresh
   - Focus on any gaps in knowledge

2. **Day 2-3: LLVM Crash Course**
   - Read LLVM introduction
   - Build and run examples
   - Write simple LLVM IR by hand

3. **Day 4-5: LLVM API**
   - Study IRBuilder examples
   - Generate IR programmatically
   - Write optimization pass

4. **Day 6-7: MLIR**
   - Understand MLIR motivation
   - Learn dialect system
   - Study progressive lowering

### Path 3: Focused Learning (Specific Topics)

Jump to what you need:

**Need to tokenize code?**
- Go straight to `theory/02_lexical_analysis.md`
- Run the lexer example
- Adapt for your language

**Need to generate code?**
- Start with `llvm/01_llvm_introduction.md`
- Learn LLVM IR
- Use IRBuilder

**Building ML compiler?**
- Focus on `mlir/01_mlir_introduction.md`
- Study tensor dialects
- Learn progressive lowering

## Module Overview

### Theory Module

**What you'll learn:**
- How compilers work from source to machine code
- Lexical analysis and tokenization
- Parsing and AST construction
- Semantic analysis and type checking
- Intermediate representations
- Code generation techniques

**Examples:**
- Complete lexer implementation
- Parser examples (when added)
- Simple interpreters
- Code generators

**No external dependencies required!**

### LLVM Module

**What you'll learn:**
- LLVM architecture and design
- LLVM IR syntax and semantics
- Using LLVM C++ API
- Writing optimization passes
- Building compiler frontends
- Target-independent code generation

**Examples:**
- Hello World IR generator
- Simple language compiler
- Custom optimization passes
- JIT compilation

**Requires:** LLVM 14+ installed

### MLIR Module

**What you'll learn:**
- Multi-level IR concepts
- Dialect system
- Progressive lowering
- Pattern rewriting
- Building domain-specific compilers
- Integration with LLVM

**Examples:**
- Basic MLIR programs
- Custom dialect creation
- Transformation passes
- Lowering pipelines

**Requires:** MLIR (included with LLVM 14+)

## Directory Structure

```
Compiler/
├── theory/                  # Compiler theory fundamentals
│   ├── 01_introduction.md
│   ├── 02_lexical_analysis.md
│   ├── examples/
│   │   └── 01_lexer.cpp    # Complete lexer implementation
│   └── exercises/
│
├── llvm/                    # LLVM practical guide
│   ├── 01_llvm_introduction.md
│   ├── examples/
│   │   └── 01_hello_llvm.cpp
│   └── exercises/
│
├── mlir/                    # MLIR framework
│   ├── 01_mlir_introduction.md
│   ├── examples/
│   └── exercises/
│
├── Makefile                 # Build system
├── CMakeLists.txt          # Alternative build (CMake)
└── GETTING_STARTED.md      # This file
```

## Build System

### Using Make (Recommended)

```bash
# Show help
make help

# Build everything available
make all

# Build only theory examples
make theory

# Build only LLVM examples (requires LLVM)
make llvm

# Run all examples
make run-all

# Clean build artifacts
make clean

# Check if LLVM is available
make check-llvm
```

### Using CMake

```bash
mkdir build && cd build

# Configure
cmake ..

# Or specify LLVM location
cmake -DLLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm ..

# Build
make

# Run examples manually
./bin/theory_01_lexer
./bin/llvm_01_hello_llvm
```

## Troubleshooting

### Issue: "llvm-config not found"

**Solution:** Install LLVM development packages or add to PATH

```bash
# Find llvm-config
which llvm-config

# If found with version suffix
export PATH="/usr/lib/llvm-17/bin:$PATH"

# Or create symlink
sudo ln -s /usr/lib/llvm-17/bin/llvm-config /usr/local/bin/llvm-config
```

### Issue: "LLVM version mismatch"

**Solution:** Make sure you have LLVM 14 or later

```bash
llvm-config --version

# Should show 14.0.0 or higher
```

### Issue: "Cannot find LLVM libraries"

**Solution:** Install development packages

```bash
# Ubuntu/Debian
sudo apt install llvm-dev

# macOS
brew install llvm

# Check installation
llvm-config --libdir
```

### Issue: Theory examples compile but LLVM examples don't

**Solution:** This is expected if LLVM isn't installed. Theory examples work standalone.

```bash
# Build just theory examples
make theory

# They work without LLVM
```

## Testing Your Installation

### Test 1: Theory Examples

```bash
make theory
./bin/theory_01_lexer
```

Expected output: Token listings for various code examples

### Test 2: LLVM Examples (if LLVM installed)

```bash
make llvm
./bin/llvm_01_hello_llvm
```

Expected output: Generated LLVM IR for "Hello, World!"

### Test 3: Verify LLVM IR

```bash
./bin/llvm_01_hello_llvm > output.ll
cat output.ll

# Should see LLVM IR code
# Try to compile it
llc output.ll -o output.s
gcc output.s -o hello
./hello
# Should print "Hello, LLVM!"
```

## Learning Resources

### Recommended Reading Order

1. `theory/01_introduction.md` - Understand compiler phases
2. `theory/02_lexical_analysis.md` - Learn tokenization
3. Study `theory/examples/01_lexer.cpp` - See complete implementation
4. `llvm/01_llvm_introduction.md` - Learn LLVM basics
5. Study `llvm/examples/01_hello_llvm.cpp` - Generate IR
6. `mlir/01_mlir_introduction.md` - Multi-level IR concepts

### External Resources

**Books:**
- "Compilers: Principles, Techniques, and Tools" (Dragon Book)
- "Engineering a Compiler" by Cooper and Torczon
- "Getting Started with LLVM Core Libraries"

**Online:**
- LLVM Documentation: https://llvm.org/docs/
- MLIR Documentation: https://mlir.llvm.org/
- Kaleidoscope Tutorial: LLVM's toy language tutorial

**Videos:**
- LLVM Developers' Meetings (YouTube)
- Compiler talks from conferences

## Project Ideas

After completing modules, try these projects:

### Beginner Projects
1. Extend the lexer to support more tokens
2. Build a calculator compiler to LLVM IR
3. Implement a simple interpreter
4. Write a syntax highlighter

### Intermediate Projects
1. Complete compiler for subset of C
2. Custom optimization pass
3. DSL for specific domain
4. JIT compiler for expressions

### Advanced Projects
1. MLIR dialect for your domain
2. Hardware synthesis compiler
3. ML operator fusion pass
4. Quantum circuit compiler

## Tips for Success

### 1. Start Simple
Don't try to build a full compiler immediately. Start with:
- Tokenizing simple expressions
- Generating IR for constants
- Basic arithmetic operations

### 2. Build Incrementally
Add features one at a time:
- Variables, then operators
- Control flow, then functions
- Types, then arrays

### 3. Test Frequently
After each feature:
- Write test cases
- Verify output
- Check edge cases

### 4. Read the Output
When generating IR:
- Print and study the IR
- Understand each instruction
- Compare with Clang output

### 5. Use the Tools
LLVM provides excellent tools:
```bash
# Visualize CFG
opt -dot-cfg input.ll
dot -Tpng .main.dot -o cfg.png

# Check IR validity
llvm-as input.ll -o /dev/null

# Optimize and see result
opt -O3 -S input.ll
```

## Getting Help

### Self-Help Checklist
1. Read relevant chapter thoroughly
2. Check code examples
3. Verify LLVM installation (for LLVM examples)
4. Search official LLVM documentation
5. Try minimal reproducible example

### Debugging Tips

**Theory Examples:**
- Add print statements
- Use debugger (gdb/lldb)
- Check token positions

**LLVM Examples:**
- Verify module with `verifyModule()`
- Print IR with `module->print()`
- Enable LLVM debug output

## Next Steps

1. **Choose your learning path** (see above)
2. **Set up build environment**
   ```bash
   cd Compiler
   make all
   ```
3. **Start with first chapter**
   ```bash
   cat theory/01_introduction.md
   ```
4. **Build and run examples**
   ```bash
   make theory
   ./bin/theory_01_lexer
   ```
5. **Progress through modules systematically**

## Success Metrics

You're making good progress if you can:

**After Week 2:** Explain compiler phases and implement a lexer
**After Week 4:** Parse simple expressions and build AST
**After Week 6:** Generate LLVM IR for basic programs
**After Week 8:** Write optimization passes
**After Week 10:** Understand MLIR and build domain-specific compiler

## Final Notes

Compiler development is challenging but rewarding. Take your time, experiment often, and don't be afraid to make mistakes. The best way to learn is by building!

**Happy compiler hacking!**
