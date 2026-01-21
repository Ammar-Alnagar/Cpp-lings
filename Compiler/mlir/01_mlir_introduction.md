# Chapter 1: Introduction to MLIR

## What is MLIR?

MLIR (Multi-Level Intermediate Representation) is a compiler infrastructure project that provides a flexible framework for building domain-specific compilers. It extends LLVM's compiler infrastructure with the ability to represent computations at multiple levels of abstraction simultaneously.

## Motivation for MLIR

### The Heterogeneous Computing Challenge

Modern computing faces several challenges:

1. **Domain Proliferation**
   - Machine learning accelerators (TPU, GPU)
   - Quantum computers
   - FPGAs and ASICs
   - Specialized hardware for different domains

2. **Multiple Abstraction Levels**
   - High-level: TensorFlow operations, NumPy arrays
   - Mid-level: Linear algebra, loops, tiles
   - Low-level: LLVM IR, assembly
   - Need to represent all levels efficiently

3. **Compiler Fragmentation**
   - Each domain builds its own compiler
   - Code duplication and incompatibility
   - Difficulty in reusing optimizations
   - Hard to compose different compilation strategies

### Traditional LLVM Limitations

LLVM IR works well for CPU code generation but has limitations:

1. **Single Level of Abstraction**
   - LLVM IR is low-level, close to machine code
   - Difficult to represent high-level constructs (tensors, parallel loops)
   - High-level information lost early in compilation

2. **Optimization Trade-offs**
   - Once lowered to LLVM IR, hard to apply high-level optimizations
   - Example: Matrix multiplication represented as nested loops
   - LLVM can't see it's a matrix operation

3. **Domain-Specific Needs**
   - ML compilers need tensor operations
   - GPU compilers need parallelism primitives
   - Hardware synthesis needs structural information
   - One IR can't serve all domains optimally

## MLIR Key Concepts

### 1. Dialects

Dialects are extensible collections of operations that represent computations at a specific level of abstraction.

**Standard Dialects**:
- `arith`: Arithmetic operations (add, mul, etc.)
- `scf`: Structured control flow (for, if, while)
- `affine`: Affine loops and maps for polyhedral optimization
- `linalg`: Linear algebra operations
- `tensor`: Tensor operations
- `gpu`: GPU-specific operations
- `llvm`: LLVM IR dialect (for final lowering)

**Custom Dialects**: You can create domain-specific dialects
- Example: Toy language dialect
- Example: TensorFlow dialect
- Example: Hardware description dialect

### 2. Operations

Operations are the fundamental units of computation in MLIR.

Every operation has:
- **Opcode**: What operation it is
- **Operands**: Input values
- **Results**: Output values
- **Attributes**: Compile-time constants
- **Regions**: Nested blocks of code
- **Successors**: Control flow edges

Example operation:
```mlir
%result = arith.addi %a, %b : i32
```

### 3. Types

MLIR has a flexible type system:
- Built-in types: integers, floats, tensors
- Dialect-specific types
- Custom types for your domain

Examples:
```mlir
i32                          // 32-bit integer
f64                          // 64-bit float
tensor<10x20xf32>           // 10x20 tensor of floats
memref<100xi32>             // Memory reference to 100 integers
!custom.mytype              // Custom type
```

### 4. Regions and Blocks

Operations can contain regions, which contain blocks:

```mlir
scf.for %i = %lb to %ub step %step {
    // This is a region containing one block
    %val = arith.muli %i, %i : i32
    scf.yield %val : i32
}
```

### 5. Attributes

Compile-time constant data:
```mlir
func.func @foo() attributes {inline = true} {
    %c = arith.constant 42 : i32
    return
}
```

## MLIR Architecture

```
High-Level IR (Domain-Specific Dialect)
           ↓
    [Dialect Lowering]
           ↓
Mid-Level IR (Affine, SCF, LinAlg)
           ↓
    [Optimization Passes]
           ↓
Lower-Level IR (Standard, Math, MemRef)
           ↓
    [More Lowering]
           ↓
LLVM Dialect
           ↓
    [Translation to LLVM IR]
           ↓
LLVM IR
```

### Progressive Lowering

Key MLIR concept: Lower high-level abstractions gradually:

1. **High-Level**: `linalg.matmul` (matrix multiplication)
2. **Mid-Level**: Nested `scf.for` loops
3. **Low-Level**: `arith.mulf`, `arith.addf` operations
4. **LLVM Dialect**: `llvm.fmul`, `llvm.fadd`
5. **LLVM IR**: Machine operations

Benefits:
- Apply optimizations at appropriate level
- Preserve high-level information longer
- Mix representations at different levels

## MLIR vs LLVM

| Feature | MLVM IR | MLIR |
|---------|---------|------|
| Abstraction | Single level (low) | Multiple levels |
| Extensibility | Limited | Highly extensible |
| Type System | Fixed | Extensible |
| Operations | Fixed set | Dialects add operations |
| Use Case | General CPU codegen | Domain-specific compilers |
| Optimization | General purpose | Domain-specific + general |

MLIR includes LLVM dialect, can lower to LLVM IR.

## MLIR Components

### Core Infrastructure

1. **IR Data Structures**
   - Operation, Block, Region
   - Type, Attribute systems
   - Value and SSA representation

2. **Dialect Framework**
   - Define custom operations
   - Define custom types
   - Register with MLIR

3. **Pass Infrastructure**
   - Analysis passes
   - Transformation passes
   - Dialect conversion framework

4. **Pattern Rewriting**
   - Declarative transformations
   - Pattern matching on IR
   - Automatic application

### Transformation Framework

1. **Dialect Conversion**
   - Type conversion
   - Operation legalization
   - Partial lowering support

2. **Canonicalization**
   - Simplification patterns
   - Constant folding
   - Dead code elimination

3. **Greedy Pattern Rewriter**
   - Applies patterns until fixpoint
   - Used for peephole optimizations

### Code Generation

1. **Translation to LLVM IR**
   - Lower MLIR to LLVM dialect
   - Translate to LLVM IR
   - Use LLVM backend

2. **Custom Backends**
   - Generate code for custom targets
   - Emit specialized formats
   - Hardware synthesis

## MLIR Syntax

### Textual Format

MLIR has a human-readable textual format:

```mlir
// Function definition
func.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %result = arith.addi %arg0, %arg1 : i32
    return %result : i32
}

// Nested regions
func.func @conditional(%cond: i1, %a: i32, %b: i32) -> i32 {
    %result = scf.if %cond -> i32 {
        scf.yield %a : i32
    } else {
        scf.yield %b : i32
    }
    return %result : i32
}
```

### SSA Form

Like LLVM, MLIR uses SSA (Static Single Assignment):
- Each value defined exactly once
- Use before definition not allowed
- Block arguments for PHI nodes

### Types

Types follow values with colon:
```mlir
%0 = arith.constant 42 : i32
%1 = arith.addi %0, %0 : i32
```

## Use Cases

### 1. Machine Learning Compilers

**TensorFlow/XLA**:
```mlir
// High-level TensorFlow operations
%result = "tf.MatMul"(%a, %b) : (tensor<10x20xf32>, tensor<20x30xf32>) 
                                 -> tensor<10x30xf32>
```

Lower to optimized code for TPUs, GPUs, CPUs.

**PyTorch/TorchScript**:
- Represent PyTorch operations
- Apply graph optimizations
- Generate efficient code

### 2. Hardware Synthesis

**CIRCT (Circuit IR Compilers and Tools)**:
```mlir
// Hardware description
hw.module @adder(%a: i32, %b: i32) -> (%sum: i32) {
    %0 = comb.add %a, %b : i32
    hw.output %0 : i32
}
```

Generate Verilog, SystemVerilog, or FIRRTL.

### 3. Domain-Specific Languages

Create custom languages:
```mlir
// Custom dialect for your domain
%result = mydialect.special_op %input {attr = value}
```

Lower to efficient implementations.

### 4. Quantum Computing

Represent quantum circuits and operations:
```mlir
// Quantum operations
%q0 = quantum.allocate : !quantum.qubit
%q1 = quantum.h %q0 : !quantum.qubit
%q2 = quantum.cnot %q1, %q0 : !quantum.qubit
```

### 5. Polyhedral Optimization

Affine dialect for loop nests:
```mlir
affine.for %i = 0 to 100 {
    affine.for %j = 0 to 100 {
        // Operations with affine constraints
    }
}
```

Apply sophisticated loop transformations.

## MLIR Tools

### mlir-opt

Optimize and transform MLIR:
```bash
mlir-opt input.mlir -pass-pipeline='builtin.module(func.func(canonicalize))' -o output.mlir
```

### mlir-translate

Translate between MLIR and other formats:
```bash
mlir-translate input.mlir -mlir-to-llvmir -o output.ll
```

### mlir-tblgen

Generate C++ code from TableGen definitions:
- Define dialects declaratively
- Generate operation definitions
- Generate pattern rewrites

## Building with MLIR

### Project Structure

```cpp
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

MLIRContext context;
OpBuilder builder(&context);

// Create operations
auto funcOp = builder.create<FuncOp>(location, "my_func", funcType);
```

### Defining a Dialect

```cpp
class MyDialect : public Dialect {
public:
    explicit MyDialect(MLIRContext *context);
    static StringRef getDialectNamespace() { return "mydialect"; }
    
    // Register operations, types, attributes
    void initialize();
};
```

### Writing Transformations

```cpp
struct MyPass : public PassWrapper<MyPass, OperationPass<FuncOp>> {
    void runOnOperation() override {
        FuncOp func = getOperation();
        // Transform the function
    }
};
```

## MLIR Compilation Pipeline

Example: Compiling a toy language

```
Toy AST
   ↓ [AST to MLIR]
Toy Dialect IR
   ↓ [Inline, Shape Inference]
Optimized Toy IR
   ↓ [Lower to Affine]
Affine Dialect IR
   ↓ [Affine Optimizations]
Optimized Affine IR
   ↓ [Lower to SCF + Standard]
SCF + Standard IR
   ↓ [Lower to LLVM Dialect]
LLVM Dialect IR
   ↓ [Translate to LLVM IR]
LLVM IR
   ↓ [LLVM Optimizations]
Machine Code
```

## Advantages of MLIR

1. **Flexibility**
   - Custom dialects for any domain
   - Mix multiple abstraction levels
   - Compose different representations

2. **Reusability**
   - Share infrastructure across compilers
   - Reuse optimization passes
   - Common transformation framework

3. **Gradual Lowering**
   - Preserve information longer
   - Optimize at appropriate level
   - Better optimization opportunities

4. **Interoperability**
   - Different dialects can coexist
   - Easy to combine compilers
   - Standard conversion framework

5. **Extensibility**
   - Add new operations easily
   - Define custom types
   - Integrate new backends

## Learning MLIR

### Prerequisites
- LLVM knowledge helpful
- Understanding of compiler concepts
- C++ programming
- Familiarity with IR

### Learning Path
1. Understand MLIR concepts (dialects, operations, regions)
2. Read and write basic MLIR
3. Study existing dialects
4. Write simple transformations
5. Create custom dialect
6. Build end-to-end compiler

### Resources
- MLIR Documentation: https://mlir.llvm.org/
- Toy Tutorial: Step-by-step compiler building
- MLIR RFC documents: Design rationale
- LLVM Dev Meetings: MLIR talks

## Summary

Key concepts covered:
- MLIR enables multi-level IR representation
- Dialects provide extensibility
- Progressive lowering preserves information
- Used for ML compilers, hardware synthesis, DSLs
- Flexible and composable infrastructure
- Addresses modern heterogeneous computing challenges

## Next Steps

In Chapter 2, we will explore MLIR dialects in detail, learning how to use existing dialects and create custom ones for domain-specific needs.

## Code Examples

See the `examples/` directory for:
- Basic MLIR programs
- Dialect usage examples
- Simple transformations
- Toy language compiler
