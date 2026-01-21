# Chapter 2: MLIR Dialects

## Introduction to Dialects

Dialects are the fundamental extensibility mechanism in MLIR. A dialect is a namespace for operations, types, and attributes that represent a specific level of abstraction or domain.

## Why Dialects?

### The Abstraction Level Problem

Different compilation stages need different abstractions:

```
High-Level:   matrix_multiply(A, B)
Mid-Level:    nested loops over elements
Low-Level:    add, multiply instructions
Hardware:     MAC operations, DMA transfers
```

Traditional compilers either:
1. Lower too early (lose optimization opportunities)
2. Keep high-level (can't generate efficient code)

MLIR solves this with dialects at multiple levels.

## Built-in Dialects

MLIR provides many standard dialects:

### 1. Builtin Dialect

Core types and operations:
```mlir
module {
  func.func @example() {
    return
  }
}
```

### 2. Arithmetic Dialect (arith)

Basic arithmetic operations:
```mlir
%sum = arith.addi %a, %b : i32
%product = arith.muli %x, %y : i32
%quotient = arith.divsi %n, %d : i32  // Signed division
%quotient = arith.divui %n, %d : i32  // Unsigned division

%fsum = arith.addf %a, %b : f32
%fprod = arith.mulf %x, %y : f32
```

### 3. Standard Control Flow (scf)

Structured control flow:
```mlir
// For loop
scf.for %i = %lb to %ub step %step {
  // Loop body
  %val = arith.addi %i, %c1 : i32
}

// While loop
scf.while (%arg = %init) : (i32) -> i32 {
  %cond = arith.cmpi slt, %arg, %limit : i32
  scf.condition(%cond) %arg : i32
} do {
^bb0(%arg: i32):
  %next = arith.addi %arg, %c1 : i32
  scf.yield %next : i32
}

// If-else
%result = scf.if %condition -> i32 {
  scf.yield %true_value : i32
} else {
  scf.yield %false_value : i32
}
```

### 4. Affine Dialect

Polyhedral optimization for loops:
```mlir
// Affine for loop
affine.for %i = 0 to 100 {
  affine.for %j = 0 to 100 {
    // Perfect loop nest
    %val = affine.load %A[%i, %j] : memref<100x100xf32>
  }
}

// Affine map
#map = affine_map<(d0, d1) -> (d0 * 100 + d1)>
%idx = affine.apply #map(%i, %j)
```

### 5. Linear Algebra (linalg)

High-level linear algebra operations:
```mlir
// Matrix multiplication: C = A * B
linalg.matmul 
  ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
  outs(%C : tensor<?x?xf32>)

// Generic operation (custom computation)
linalg.generic {
  indexing_maps = [#map0, #map1, #map2],
  iterator_types = ["parallel", "parallel"]
} 
ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
outs(%C : tensor<?x?xf32>) {
^bb0(%a: f32, %b: f32, %c: f32):
  %sum = arith.addf %a, %b : f32
  linalg.yield %sum : f32
}
```

### 6. Tensor Dialect

Tensor operations:
```mlir
// Create tensor
%tensor = tensor.empty() : tensor<10x20xf32>

// Extract element
%elem = tensor.extract %tensor[%i, %j] : tensor<10x20xf32>

// Insert element
%new_tensor = tensor.insert %value into %tensor[%i, %j] 
              : tensor<10x20xf32>

// Reshape
%reshaped = tensor.reshape %tensor(%shape) 
            : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
```

### 7. MemRef Dialect

Memory buffer operations:
```mlir
// Allocate memory
%buffer = memref.alloc() : memref<10x20xf32>

// Load from memory
%value = memref.load %buffer[%i, %j] : memref<10x20xf32>

// Store to memory
memref.store %value, %buffer[%i, %j] : memref<10x20xf32>

// Deallocate
memref.dealloc %buffer : memref<10x20xf32>
```

### 8. GPU Dialect

GPU-specific operations:
```mlir
gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
           threads(%tx, %ty, %tz) in (%block_x = %c256, %block_y = %c1, %block_z = %c1) {
  // GPU kernel code
  %tid = arith.addi %tx, %ty : index
  gpu.terminator
}
```

### 9. Vector Dialect

SIMD vector operations:
```mlir
// Vector load
%vec = vector.load %buffer[%i] : memref<?xf32>, vector<4xf32>

// Vector arithmetic
%sum = vector.add %a, %b : vector<4xf32>

// Vector reduction
%result = vector.reduction <add>, %vec : vector<4xf32> into f32
```

### 10. LLVM Dialect

Maps to LLVM IR:
```mlir
// LLVM operations
%0 = llvm.add %arg0, %arg1 : i32
%1 = llvm.mul %0, %arg2 : i32

// LLVM intrinsics
%2 = llvm.intr.sqrt(%x) : (f32) -> f32

// Inline assembly
llvm.inline_asm "nop"
```

## Dialect Design

### Operation Definition

Operations are defined using TableGen (declarative):

```tablegen
def AddIOp : Arith_Op<"addi", [Pure, SameOperandsAndResultType]> {
  let summary = "Integer addition operation";
  let description = [{
    Adds two integer values.
    
    Example:
    ```mlir
    %sum = arith.addi %a, %b : i32
    ```
  }];
  
  let arguments = (ins SignlessInteger:$lhs, SignlessInteger:$rhs);
  let results = (outs SignlessInteger:$result);
  
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
```

### Custom Types

Dialects can define custom types:

```cpp
// In C++
class MyCustomType : public Type::TypeBase<MyCustomType, Type, TypeStorage> {
public:
  // Type implementation
  static MyCustomType get(MLIRContext *context);
};
```

### Custom Attributes

Dialects can define custom attributes:

```cpp
class MyCustomAttr : public Attribute::AttrBase<MyCustomAttr, Attribute> {
  // Attribute implementation
};
```

## Dialect Hierarchy

Dialects exist at different abstraction levels:

```
High-Level Dialects:
  ├── TensorFlow Dialect (TF operations)
  ├── Torch Dialect (PyTorch operations)
  └── TOSA Dialect (Tensor Operator Set Architecture)

Mid-Level Dialects:
  ├── Linalg Dialect (Linear algebra)
  ├── Tensor Dialect (Tensor operations)
  └── Affine Dialect (Polyhedral loops)

Low-Level Dialects:
  ├── SCF Dialect (Control flow)
  ├── Arith Dialect (Arithmetic)
  ├── MemRef Dialect (Memory operations)
  └── Vector Dialect (SIMD)

Target-Specific Dialects:
  ├── LLVM Dialect (LLVM IR)
  ├── GPU Dialect (GPU operations)
  ├── SPIRV Dialect (Vulkan/OpenCL)
  └── NVVM Dialect (NVIDIA PTX)
```

## Progressive Lowering

Dialects enable step-by-step lowering:

### Example: TensorFlow to LLVM

```
TensorFlow Dialect
  ↓ [tf-to-linalg]
Linalg + Tensor Dialect
  ↓ [linalg-to-loops]
SCF + MemRef Dialect
  ↓ [scf-to-cf, convert-to-llvm]
LLVM Dialect
  ↓ [mlir-translate]
LLVM IR
```

### Lowering Pass Example

```mlir
// Input: High-level
func.func @matmul(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %C = tensor.empty() : tensor<4x4xf32>
  %result = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
                          outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

// After lowering to loops
func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      scf.for %k = %c0 to %c4 step %c1 {
        %a = memref.load %A[%i, %k] : memref<4x4xf32>
        %b = memref.load %B[%k, %j] : memref<4x4xf32>
        %c = memref.load %C[%i, %j] : memref<4x4xf32>
        %prod = arith.mulf %a, %b : f32
        %sum = arith.addf %c, %prod : f32
        memref.store %sum, %C[%i, %j] : memref<4x4xf32>
      }
    }
  }
  return
}
```

## Creating Custom Dialects

### Step 1: Define Dialect

```cpp
class MyDialect : public Dialect {
public:
  explicit MyDialect(MLIRContext *context);
  
  static constexpr StringLiteral getDialectNamespace() {
    return StringLiteral("mydialect");
  }
  
  void initialize();
};
```

### Step 2: Define Operations

Using TableGen:

```tablegen
def MyDialect_AddOp : MyDialect_Op<"add"> {
  let summary = "My custom add operation";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
```

### Step 3: Register Dialect

```cpp
void MyDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "MyDialect/MyOps.cpp.inc"
  >();
}
```

### Step 4: Use in MLIR

```mlir
func.func @example(%a: i32, %b: i32) -> i32 {
  %result = mydialect.add %a, %b : i32
  return %result : i32
}
```

## Dialect Conversion

### Conversion Patterns

Define how to convert operations:

```cpp
struct MyAddOpLowering : public OpConversionPattern<MyAddOp> {
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      MyAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Replace MyAddOp with arith.addi
    rewriter.replaceOpWithNewOp<arith::AddIOp>(
      op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};
```

### Conversion Pass

```cpp
struct MyDialectLoweringPass 
    : public PassWrapper<MyDialectLoweringPass, OperationPass<ModuleOp>> {
  
  void runOnOperation() override {
    ConversionTarget target(getContext());
    
    // Mark illegal operations
    target.addIllegalDialect<MyDialect>();
    
    // Mark legal operations
    target.addLegalDialect<arith::ArithDialect>();
    
    // Define conversion patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<MyAddOpLowering>(&getContext());
    
    // Apply conversion
    if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
```

## Pattern Rewriting

### Rewrite Patterns

Transform IR without changing dialects:

```cpp
struct SimplifyAddZero : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;
  
  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    // Match: x + 0
    if (auto constOp = op.getRhs().getDefiningOp<arith::ConstantOp>()) {
      if (constOp.value() == 0) {
        // Replace with just x
        rewriter.replaceOp(op, op.getLhs());
        return success();
      }
    }
    return failure();
  }
};
```

### Canonicalization

Each operation can define canonicalization patterns:

```tablegen
def MyAddOp : MyDialect_Op<"add"> {
  // ...
  let hasCanonicalizer = 1;
}
```

```cpp
void MyAddOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<SimplifyAddZero>(context);
}
```

## Dialect Interfaces

Dialects can implement interfaces for common behavior:

### InlinerInterface

```cpp
struct MyDialectInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  
  bool isLegalToInline(Operation *op, Region *dest,
                      bool wouldBeCloned,
                      IRMapping &valueMapping) const final {
    // Define inlining rules
    return true;
  }
};
```

### FoldOpInterface

```cpp
OpFoldResult MyAddOp::fold(FoldAdaptor adaptor) {
  // Constant folding
  auto lhs = adaptor.getLhs().dyn_cast_or_null<IntegerAttr>();
  auto rhs = adaptor.getRhs().dyn_cast_or_null<IntegerAttr>();
  
  if (lhs && rhs) {
    return IntegerAttr::get(getType(), lhs.getInt() + rhs.getInt());
  }
  
  return {};
}
```

## Real-World Dialect Examples

### TensorFlow Dialect

```mlir
func.func @conv2d(%input: tensor<1x224x224x3xf32>,
                  %filter: tensor<64x3x3x3xf32>) 
                  -> tensor<1x224x224x64xf32> {
  %result = "tf.Conv2D"(%input, %filter) {
    strides = [1, 1, 1, 1],
    padding = "SAME"
  } : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>) 
      -> tensor<1x224x224x64xf32>
  return %result : tensor<1x224x224x64xf32>
}
```

### TOSA Dialect

Tensor Operator Set Architecture for hardware:

```mlir
func.func @tosa_conv2d(%input: tensor<1x224x224x3xf32>,
                       %weight: tensor<64x3x3x3xf32>,
                       %bias: tensor<64xf32>) 
                       -> tensor<1x224x224x64xf32> {
  %result = tosa.conv2d %input, %weight, %bias {
    dilation = [1, 1],
    pad = [1, 1, 1, 1],
    stride = [1, 1]
  } : (tensor<1x224x224x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) 
      -> tensor<1x224x224x64xf32>
  return %result : tensor<1x224x224x64xf32>
}
```

### Hardware Dialects (CIRCT)

For hardware synthesis:

```mlir
hw.module @adder(%a: i32, %b: i32) -> (sum: i32) {
  %0 = comb.add %a, %b : i32
  hw.output %0 : i32
}
```

## Best Practices

### 1. Design at Right Abstraction Level

```mlir
// Good: High-level operation
linalg.matmul ins(%A, %B : ...) outs(%C : ...)

// Bad: Too low-level too early
scf.for %i ... {
  scf.for %j ... {
    scf.for %k ... {
      // Lost high-level semantics
    }
  }
}
```

### 2. Use Standard Dialects When Possible

Prefer existing dialects over custom ones.

### 3. Define Clear Semantics

Document operation behavior precisely.

### 4. Provide Verification

```cpp
LogicalResult MyAddOp::verify() {
  if (getLhs().getType() != getRhs().getType())
    return emitError("operand types must match");
  return success();
}
```

### 5. Implement Folding

```cpp
OpFoldResult MyAddOp::fold(FoldAdaptor adaptor) {
  // Constant folding, identity elimination
  return {};
}
```

## Summary

Key concepts covered:
- Dialects as abstraction levels
- Built-in MLIR dialects
- Dialect hierarchy and organization
- Progressive lowering strategy
- Creating custom dialects
- Dialect conversion and pattern rewriting
- Dialect interfaces
- Real-world examples
- Best practices

## Next Steps

In Chapter 3, we would explore transformations and optimizations in MLIR, learning how to write passes that operate on multi-level IR.

Practice working with existing dialects and understand how they compose to build complete compilation pipelines!
