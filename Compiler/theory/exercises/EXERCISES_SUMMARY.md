# Compiler Exercises - Complete Summary

This document provides an overview of all exercises available for learning compiler construction.

## Exercise Categories

### 🔤 Lexical Analysis (Exercises 01-05)
Foundation exercises for tokenization and scanning.

### 🌳 Parsing (Exercises 06-10)
Building parsers and Abstract Syntax Trees.

### ✅ Semantic Analysis (Exercises 11-15)
Type checking and semantic validation.

### 🔄 Intermediate Representation (Exercises 16-20)
IR generation and transformations.

### 🚀 LLVM Exercises (Exercises 21-25)
Working with LLVM infrastructure.

### 🎯 MLIR Exercises (Exercises 26-30)
Multi-level IR and dialects.

### 🏗️ Final Project
Complete compiler implementation.

## Detailed Exercise List

### Lexical Analysis Exercises

#### Exercise 01: Basic Number Lexer ⭐
**Difficulty**: Beginner | **Time**: 30-45 min | **Status**: ✅ Complete
- Recognize integers, floats, hex, binary, octal
- Handle scientific notation
- Error detection and reporting
- **Location**: `01_basic_number_lexer.md`
- **Solution**: `solutions/01_basic_number_lexer.cpp`

#### Exercise 02: Identifier and Keyword Recognition
**Difficulty**: Beginner | **Time**: 30-45 min
- Implement identifier tokenization
- Handle keyword detection
- Case sensitivity
- Reserved word management

#### Exercise 03: String Literal Handling
**Difficulty**: Intermediate | **Time**: 1 hour
- Parse string literals with escape sequences
- Multi-line strings
- Raw strings
- Unicode support

#### Exercise 04: Complete Language Lexer
**Difficulty**: Intermediate | **Time**: 2 hours
- Combine all token types
- Handle all operators
- Complete error reporting
- Position tracking

#### Exercise 05: Lexer Optimization
**Difficulty**: Advanced | **Time**: 2 hours
- Optimize state machine
- Implement lookahead buffer
- String interning
- Performance benchmarking

### Parsing Exercises

#### Exercise 06: Expression Parser ⭐
**Difficulty**: Intermediate | **Time**: 1-2 hours | **Status**: ✅ Complete
- Recursive descent parsing
- Operator precedence
- AST construction
- **Location**: `02_expression_parser.md`

#### Exercise 07: Statement Parser
**Difficulty**: Intermediate | **Time**: 2 hours
- Parse declarations and statements
- Handle blocks and scope
- Control flow statements
- Error recovery

#### Exercise 08: Function Parser
**Difficulty**: Intermediate | **Time**: 2 hours
- Function declarations
- Parameter lists
- Return statements
- Function calls

#### Exercise 09: Complete Language Parser
**Difficulty**: Advanced | **Time**: 3 hours
- Full language grammar
- All statement types
- Complete AST
- Comprehensive error handling

#### Exercise 10: Parser Generator
**Difficulty**: Advanced | **Time**: 4 hours
- Implement simple parser generator
- Generate parsers from grammar
- Handle conflicts
- Optimize generated code

### Semantic Analysis Exercises

#### Exercise 11: Symbol Table Implementation
**Difficulty**: Intermediate | **Time**: 2 hours
- Implement multi-scope symbol table
- Name resolution
- Scope management
- Symbol attributes

#### Exercise 12: Type Checker
**Difficulty**: Intermediate | **Time**: 2 hours
- Type inference
- Type compatibility checking
- Implicit conversions
- Type error reporting

#### Exercise 13: Name Resolution
**Difficulty**: Intermediate | **Time**: 1.5 hours
- Resolve variable references
- Handle shadowing
- Check declaration before use
- Cross-function resolution

#### Exercise 14: Semantic Validator
**Difficulty**: Advanced | **Time**: 2 hours
- Validate control flow
- Check unreachable code
- Verify initialization
- Constant expression evaluation

#### Exercise 15: Advanced Type System
**Difficulty**: Advanced | **Time**: 3 hours
- Implement generic types
- Template/generics
- Type inference
- Complex type checking

### Intermediate Representation Exercises

#### Exercise 16: AST to Three-Address Code
**Difficulty**: Intermediate | **Time**: 2 hours
- Convert AST to linear IR
- Temporary variable generation
- Instruction selection
- Control flow handling

#### Exercise 17: Basic Block Construction
**Difficulty**: Intermediate | **Time**: 1.5 hours
- Identify basic blocks
- Build CFG
- Compute dominators
- Block optimization

#### Exercise 18: SSA Transformation
**Difficulty**: Advanced | **Time**: 3 hours
- Convert to SSA form
- Phi node insertion
- Variable renaming
- SSA destruction

#### Exercise 19: Control Flow Analysis
**Difficulty**: Advanced | **Time**: 2 hours
- Build control flow graph
- Dominator analysis
- Loop detection
- Dataflow analysis

#### Exercise 20: Basic Optimizations
**Difficulty**: Advanced | **Time**: 2 hours
- Constant folding
- Dead code elimination
- Common subexpression elimination
- Copy propagation

### LLVM Exercises

#### Exercise 21: Basic IR Generation ⭐
**Difficulty**: Intermediate | **Time**: 2-3 hours | **Status**: ✅ Complete
- Use LLVM IRBuilder
- Generate functions
- Handle loops and recursion
- **Location**: `../llvm/exercises/01_ir_generation.md`

#### Exercise 22: Optimization Pass
**Difficulty**: Advanced | **Time**: 3 hours
- Write LLVM pass
- Implement transformation
- Use analysis passes
- Pass registration

#### Exercise 23: Custom Intrinsics
**Difficulty**: Advanced | **Time**: 2 hours
- Define custom intrinsics
- Implement lowering
- Target-specific code
- Optimization integration

#### Exercise 24: JIT Compilation
**Difficulty**: Advanced | **Time**: 3 hours
- Set up LLVM JIT
- Compile and execute
- Handle dynamic code
- Optimization for JIT

#### Exercise 25: Backend Customization
**Difficulty**: Expert | **Time**: 4+ hours
- Custom target description
- Instruction selection
- Register allocation
- Assembly emission

### MLIR Exercises

#### Exercise 26: Dialect Basics ⭐
**Difficulty**: Intermediate | **Time**: 2-3 hours | **Status**: ✅ Complete
- Work with built-in dialects
- Progressive lowering
- Pattern rewriting
- **Location**: `../mlir/exercises/01_dialect_basics.md`

#### Exercise 27: Custom Dialect
**Difficulty**: Advanced | **Time**: 4 hours
- Define custom dialect
- Create operations
- Define types and attributes
- Implement canonicalization

#### Exercise 28: Dialect Conversion
**Difficulty**: Advanced | **Time**: 3 hours
- Write conversion patterns
- Implement lowering pass
- Handle type conversion
- Test conversions

#### Exercise 29: Affine Optimization
**Difficulty**: Advanced | **Time**: 3 hours
- Use affine dialect
- Apply loop transformations
- Tiling and fusion
- Polyhedral optimization

#### Exercise 30: GPU Code Generation
**Difficulty**: Expert | **Time**: 4+ hours
- Lower to GPU dialect
- Implement kernels
- Memory management
- Optimize for GPU

### Final Project

#### Complete SimpleLang Compiler ⭐
**Difficulty**: Advanced | **Time**: 8-12 hours | **Status**: ✅ Complete
- Full compiler implementation
- All phases integrated
- Complete test suite
- **Location**: `PROJECT_simple_language_compiler.md`

## Learning Paths

### Path 1: Sequential (Complete Beginner)
Complete exercises in numerical order for comprehensive learning.
- **Duration**: 8-10 weeks
- **Commitment**: 10-15 hours/week

### Path 2: Fast Track (Experienced)
Focus on harder exercises and skip basics.
- **Duration**: 4-6 weeks
- **Commitment**: 15-20 hours/week

### Path 3: Topic-Specific
Choose exercises based on specific interests.
- **Duration**: Variable
- **Focus**: Specific compiler aspects

### Path 4: Project-First
Start with final project, refer to exercises as needed.
- **Duration**: 2-3 weeks intensive
- **Style**: Learning by doing

## Difficulty Distribution

```
Beginner:      ████░░░░░░  6 exercises (20%)
Intermediate:  ████████░░  14 exercises (47%)
Advanced:      ██████░░░░  9 exercises (30%)
Expert:        █░░░░░░░░░  1 exercise (3%)
```

## Time Requirements

- **Total exercises**: ~100 hours
- **Final project**: ~12 hours
- **Review and practice**: ~20 hours
- **Grand total**: ~130 hours

## Completion Tracking

Use this checklist to track your progress:

### Lexical Analysis
- [ ] Exercise 01: Basic Number Lexer
- [ ] Exercise 02: Identifier Recognition
- [ ] Exercise 03: String Literals
- [ ] Exercise 04: Complete Lexer
- [ ] Exercise 05: Lexer Optimization

### Parsing
- [ ] Exercise 06: Expression Parser
- [ ] Exercise 07: Statement Parser
- [ ] Exercise 08: Function Parser
- [ ] Exercise 09: Complete Parser
- [ ] Exercise 10: Parser Generator

### Semantic Analysis
- [ ] Exercise 11: Symbol Table
- [ ] Exercise 12: Type Checker
- [ ] Exercise 13: Name Resolution
- [ ] Exercise 14: Semantic Validator
- [ ] Exercise 15: Advanced Types

### Intermediate Representation
- [ ] Exercise 16: Three-Address Code
- [ ] Exercise 17: Basic Blocks
- [ ] Exercise 18: SSA Transformation
- [ ] Exercise 19: Control Flow Analysis
- [ ] Exercise 20: Basic Optimizations

### LLVM
- [ ] Exercise 21: IR Generation
- [ ] Exercise 22: Optimization Pass
- [ ] Exercise 23: Custom Intrinsics
- [ ] Exercise 24: JIT Compilation
- [ ] Exercise 25: Backend Customization

### MLIR
- [ ] Exercise 26: Dialect Basics
- [ ] Exercise 27: Custom Dialect
- [ ] Exercise 28: Dialect Conversion
- [ ] Exercise 29: Affine Optimization
- [ ] Exercise 30: GPU Code Generation

### Final Project
- [ ] SimpleLang Compiler Complete

## Resources by Exercise

### All Exercises
- Theory chapters 1-3
- LLVM documentation
- MLIR documentation
- Example implementations

### Specific Resources
- **Lexer**: Chapter 2 (Lexical Analysis)
- **Parser**: Chapter 3 (Parsing)
- **LLVM**: LLVM chapters, Kaleidoscope tutorial
- **MLIR**: MLIR chapters, Toy tutorial

## Getting Help

### Self-Help Resources
1. Review relevant theory chapter
2. Check hints in exercise description
3. Study example implementations
4. Read LLVM/MLIR documentation
5. Review solution code (last resort)

### Common Issues
- Build problems: Check CMakeLists.txt, LLVM paths
- LLVM errors: Verify module after creation
- Parsing errors: Print tokens, check grammar
- Codegen errors: Verify IR, check types

## Status Legend

- ✅ Complete: Exercise fully written with solutions
- 🚧 In Progress: Exercise partially complete
- 📝 Planned: Exercise designed but not yet written

## Contributing

These exercises are designed to be comprehensive. Suggestions for improvements or additional exercises are welcome!

## Final Notes

Remember:
- Start with easier exercises
- Test frequently
- Read error messages carefully
- Build incrementally
- Review solutions only after attempting
- Practice is key to mastery

**Good luck with your compiler journey!**
