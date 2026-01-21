# Exercise 01: MLIR Dialect Basics

**Difficulty**: Intermediate  
**Estimated Time**: 2-3 hours  
**Topics**: MLIR Operations, Dialects, Pattern Rewriting, Progressive Lowering

## Learning Objectives

After completing this exercise, you will:
- Understand MLIR dialect system
- Work with built-in dialects
- Write simple MLIR programs
- Lower between dialect levels
- Apply pattern rewriting

## Problem Description

Write MLIR programs using different dialects and implement transformations to lower them progressively from high-level to low-level representations.

### Tasks

1. **High-Level**: Write matrix multiplication in `linalg` dialect
2. **Mid-Level**: Lower to `affine` loops
3. **Low-Level**: Lower to `scf` (structured control flow)
4. **Conversion**: Write patterns to perform lowering
5. **Execution**: Lower to LLVM dialect and execute

## Part 1: High-Level Matrix Multiplication

Write an MLIR function that performs matrix multiplication using `linalg` dialect.

### Expected MLIR Code

```mlir
// Part 1: High-level linalg matrix multiplication
#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @matmul_linalg(%A: tensor<4x4xf32>, 
                         %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // Initialize output tensor
  %C = tensor.empty() : tensor<4x4xf32>
  %zero = arith.constant 0.0 : f32
  %C_init = linalg.fill ins(%zero : f32) outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>
  
  // Perform matrix multiplication
  %result = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%C_init : tensor<4x4xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %mul = arith.mulf %a, %b : f32
      %add = arith.addf %c, %mul : f32
      linalg.yield %add : f32
  } -> tensor<4x4xf32>
  
  return %result : tensor<4x4xf32>
}
```

## Part 2: Lower to Affine Loops

Convert the linalg operation to explicit affine loops.

### Expected Output

```mlir
func.func @matmul_affine(%A: memref<4x4xf32>, 
                         %B: memref<4x4xf32>,
                         %C: memref<4x4xf32>) {
  // Initialize C to zero
  %zero = arith.constant 0.0 : f32
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      affine.store %zero, %C[%i, %j] : memref<4x4xf32>
    }
  }
  
  // Matrix multiplication
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      affine.for %k = 0 to 4 {
        %a = affine.load %A[%i, %k] : memref<4x4xf32>
        %b = affine.load %B[%k, %j] : memref<4x4xf32>
        %c = affine.load %C[%i, %j] : memref<4x4xf32>
        
        %mul = arith.mulf %a, %b : f32
        %add = arith.addf %c, %mul : f32
        
        affine.store %add, %C[%i, %j] : memref<4x4xf32>
      }
    }
  }
  
  return
}
```

## Part 3: Lower to SCF

Convert affine loops to structured control flow.

### Expected Output

```mlir
func.func @matmul_scf(%A: memref<4x4xf32>, 
                      %B: memref<4x4xf32>,
                      %C: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  
  // Initialize C
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      memref.store %zero, %C[%i, %j] : memref<4x4xf32>
    }
  }
  
  // Matrix multiplication
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      scf.for %k = %c0 to %c4 step %c1 {
        %a = memref.load %A[%i, %k] : memref<4x4xf32>
        %b = memref.load %B[%k, %j] : memref<4x4xf32>
        %c = memref.load %C[%i, %j] : memref<4x4xf32>
        
        %mul = arith.mulf %a, %b : f32
        %add = arith.addf %c, %mul : f32
        
        memref.store %add, %C[%i, %j] : memref<4x4xf32>
      }
    }
  }
  
  return
}
```

## Part 4: Implement Lowering Pass

Write C++ code to perform the lowering transformations.

### Conversion Framework

```cpp
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

// TODO: Implement lowering pass
struct LinalgToAffinePass 
    : public PassWrapper<LinalgToAffinePass, OperationPass<func::FuncOp>> {
  
  void runOnOperation() override {
    auto func = getOperation();
    
    // Set up conversion target
    ConversionTarget target(getContext());
    target.addLegalDialect<affine::AffineDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<linalg::LinalgDialect>();
    
    // Define conversion patterns
    RewritePatternSet patterns(&getContext());
    // TODO: Add patterns
    
    // Apply conversion
    if (failed(applyFullConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
```

## Requirements

1. **Write MLIR**: Create valid MLIR for each dialect level
2. **Verify**: Use `mlir-opt` to verify correctness
3. **Transform**: Apply lowering passes
4. **Test**: Execute the final LLVM IR

## Test Commands

```bash
# Verify high-level MLIR
mlir-opt --verify-diagnostics matmul_linalg.mlir

# Lower linalg to affine
mlir-opt --linalg-bufferize --convert-linalg-to-affine-loops \
    matmul_linalg.mlir -o matmul_affine.mlir

# Lower affine to SCF
mlir-opt --lower-affine matmul_affine.mlir -o matmul_scf.mlir

# Lower to LLVM dialect
mlir-opt --convert-scf-to-cf --convert-func-to-llvm \
    --reconcile-unrealized-casts matmul_scf.mlir -o matmul_llvm.mlir

# Translate to LLVM IR
mlir-translate --mlir-to-llvmir matmul_llvm.mlir -o matmul.ll

# Execute
lli matmul.ll
```

## Starter Files

### matmul_linalg.mlir
```mlir
module {
  func.func @main() {
    // TODO: Create test matrices
    // TODO: Call matmul_linalg
    // TODO: Print result
    return
  }
}
```

## Hints

### Hint 1: Tensor vs MemRef
- `tensor` is value-based (immutable, SSA)
- `memref` is reference-based (mutable)
- Use `-linalg-bufferize` to convert tensor to memref

### Hint 2: Affine Maps
Affine maps describe index transformations:
```mlir
#map = affine_map<(i, j, k) -> (i, j)>
```

### Hint 3: Iterator Types
- `parallel`: Can execute in parallel
- `reduction`: Has dependencies (like sum)

### Hint 4: Pattern Rewriting
```cpp
struct MyPattern : public OpRewritePattern<MyOp> {
  LogicalResult matchAndRewrite(MyOp op, PatternRewriter &rewriter) const {
    // Match and transform
    return success();
  }
};
```

## Bonus Challenges

1. **Tiling**: Add loop tiling for better cache performance
2. **Vectorization**: Use vector dialect for SIMD
3. **Parallelization**: Add parallel attributes for multi-threading
4. **GPU**: Lower to GPU dialect for GPU execution
5. **Optimization**: Implement loop fusion and other optimizations

## Common Issues

1. **Type mismatches**: Ensure tensor/memref dimensions match
2. **Missing verifier**: Always verify IR after transformations
3. **Unrealized casts**: May need `--reconcile-unrealized-casts`
4. **Dialect dependencies**: Ensure all required dialects are registered

## Expected Learning Outcomes

- Understanding of MLIR's multi-level approach
- Proficiency with major MLIR dialects
- Knowledge of progressive lowering strategy
- Ability to write MLIR transformations
- Skills in debugging MLIR programs

## Solution Location

Complete solution in `solutions/01_dialect_basics/` with:
- `matmul_linalg.mlir` - High-level implementation
- `lowering_pass.cpp` - Conversion pass
- `CMakeLists.txt` - Build configuration
- `README.md` - Detailed explanation

## Next Exercise

Proceed to **Exercise 02: Custom Dialect** to learn how to create your own MLIR dialect.
