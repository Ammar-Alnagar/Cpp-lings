# Chapter 2: LLVM IR in Depth

## Introduction to LLVM IR

LLVM Intermediate Representation (IR) is a low-level, strongly-typed, RISC-like instruction set that serves as the foundation of the LLVM compiler infrastructure. Understanding LLVM IR is essential for working with LLVM.

## Why LLVM IR?

### Design Goals

1. **Language Independence**: Not tied to any source language
2. **Target Independence**: Not specific to any architecture
3. **Optimization Friendly**: Designed for analysis and transformation
4. **Strongly Typed**: Catches errors early
5. **SSA Form**: Simplifies optimization

### Benefits

- Write compiler frontend once, target multiple architectures
- Reuse optimization passes across languages
- Clear semantics for transformations
- Human-readable text format

## LLVM IR Representations

LLVM IR has three isomorphic forms:

### 1. In-Memory Form
C++ objects used by compiler passes:
```cpp
Module* M = new Module("my_module", Context);
Function* F = Function::Create(...);
BasicBlock* BB = BasicBlock::Create(...);
```

### 2. Bitcode Format (.bc)
Binary representation:
```bash
clang -emit-llvm -c source.c -o source.bc
```
- Compact
- Fast to parse
- Used for distribution

### 3. Assembly Format (.ll)
Human-readable text:
```bash
clang -S -emit-llvm source.c -o source.ll
```
- Easier to debug
- Can be hand-written
- Good for learning

## Module Structure

A module is the top-level container in LLVM IR.

### Basic Structure

```llvm
; ModuleID = 'example.c'
source_filename = "example.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64..."
target triple = "x86_64-pc-linux-gnu"

; Global variables
@global_var = global i32 0

; Function declarations
declare i32 @external_func(i32)

; Function definitions
define i32 @my_func(i32 %arg) {
entry:
  ; Function body
  ret i32 %arg
}
```

### Module Components

1. **Target Information**: Data layout and triple
2. **Global Variables**: Module-level data
3. **Function Declarations**: External functions
4. **Function Definitions**: Implemented functions
5. **Metadata**: Debug info, optimization hints
6. **Type Definitions**: Custom types
7. **Module Flags**: Compilation settings

## Type System

LLVM has a rich, strongly-typed system.

### Primitive Types

```llvm
void            ; No value
i1              ; 1-bit integer (boolean)
i8              ; 8-bit integer (byte)
i16             ; 16-bit integer
i32             ; 32-bit integer
i64             ; 64-bit integer
i128            ; 128-bit integer
half            ; 16-bit floating point
float           ; 32-bit floating point
double          ; 64-bit floating point
x86_fp80        ; 80-bit x87 floating point
fp128           ; 128-bit floating point
```

### Derived Types

#### Pointers
```llvm
i32*            ; Pointer to 32-bit integer
i8**            ; Pointer to pointer to 8-bit integer
[10 x i32]*     ; Pointer to array
{i32, double}*  ; Pointer to structure
```

#### Arrays
```llvm
[10 x i32]      ; Array of 10 integers
[0 x i32]       ; Dynamically sized array
```

#### Structures
```llvm
{i32, double}               ; Packed structure
{i32, double, [10 x i8]}   ; Nested types

; Named structure
%MyStruct = type {i32, double}
```

#### Vectors
```llvm
<4 x i32>       ; Vector of 4 integers (SIMD)
<8 x float>     ; Vector of 8 floats
```

#### Functions
```llvm
i32 (i32, i32)          ; Function type: int(int, int)
void (i8*, i32)*        ; Pointer to function
i32 (i32, ...)          ; Variadic function
```

## Functions

Functions are the primary unit of code in LLVM IR.

### Function Definition

```llvm
define i32 @add(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}
```

### Function Attributes

```llvm
; Linkage types
define private i32 @internal_func() { ... }      ; Only visible in module
define internal i32 @static_func() { ... }       ; Like C static
define external i32 @public_func() { ... }       ; Default, visible externally

; Calling conventions
define fastcc i32 @fast_func() { ... }           ; Fast calling convention
define coldcc void @cold_func() { ... }          ; Optimized for rarely called

; Function attributes
define i32 @pure_func() readonly { ... }         ; Doesn't modify memory
define i32 @safe_func() nounwind { ... }         ; Never throws exceptions
define void @noret_func() noreturn { ... }       ; Never returns

; Optimization hints
define i32 @hot_func() alwaysinline { ... }      ; Always inline
define i32 @big_func() noinline { ... }          ; Never inline
```

### Parameters

```llvm
; By value
define void @func(i32 %x) { ... }

; By reference (pointer)
define void @func(i32* %ptr) { ... }

; By reference (read-only)
define void @func(i32* readonly %ptr) { ... }

; Attributes
define void @func(i32 signext %x) { ... }        ; Sign-extended
define void @func(i32 zeroext %x) { ... }        ; Zero-extended
define void @func(i32* noalias %p) { ... }       ; No aliasing
define void @func(i32* nocapture %p) { ... }     ; Not stored/captured
```

## Basic Blocks

Basic blocks are sequences of instructions with single entry and exit.

### Structure

```llvm
define i32 @example(i32 %x) {
entry:                          ; Label (entry block)
  %cmp = icmp sgt i32 %x, 0    ; Instructions
  br i1 %cmp, label %positive, label %negative

positive:                       ; Another basic block
  %result1 = add i32 %x, 10
  br label %end

negative:                       ; Yet another basic block
  %result2 = sub i32 %x, 10
  br label %end

end:                            ; Merge point
  %final = phi i32 [%result1, %positive], [%result2, %negative]
  ret i32 %final
}
```

### Properties

1. **Single Entry**: Only one way to enter (first instruction)
2. **Single Exit**: Only one way to leave (terminator)
3. **No Branches**: Except at the end (terminator instruction)
4. **Sequential**: Instructions execute in order

### Terminators

Every basic block must end with a terminator:

```llvm
ret i32 %value              ; Return from function
ret void                    ; Return void

br label %target            ; Unconditional branch
br i1 %cond, label %true, label %false  ; Conditional branch

switch i32 %val, label %default [
  i32 0, label %case0
  i32 1, label %case1
]                           ; Multi-way branch

unreachable                 ; Undefined behavior (never reached)
```

## Instructions

LLVM instructions are typed and in SSA form.

### Arithmetic Operations

```llvm
; Integer arithmetic
%sum = add i32 %a, %b           ; Addition
%diff = sub i32 %a, %b          ; Subtraction
%prod = mul i32 %a, %b          ; Multiplication
%quot = sdiv i32 %a, %b         ; Signed division
%quot = udiv i32 %a, %b         ; Unsigned division
%rem = srem i32 %a, %b          ; Signed remainder
%rem = urem i32 %a, %b          ; Unsigned remainder

; Floating point arithmetic
%sum = fadd float %a, %b        ; FP addition
%diff = fsub float %a, %b       ; FP subtraction
%prod = fmul float %a, %b       ; FP multiplication
%quot = fdiv float %a, %b       ; FP division
%rem = frem float %a, %b        ; FP remainder
```

### Bitwise Operations

```llvm
%and = and i32 %a, %b           ; Bitwise AND
%or = or i32 %a, %b             ; Bitwise OR
%xor = xor i32 %a, %b           ; Bitwise XOR
%shl = shl i32 %a, %b           ; Shift left
%lshr = lshr i32 %a, %b         ; Logical shift right
%ashr = ashr i32 %a, %b         ; Arithmetic shift right
```

### Comparison Operations

```llvm
; Integer comparisons
%eq = icmp eq i32 %a, %b        ; Equal
%ne = icmp ne i32 %a, %b        ; Not equal
%gt = icmp sgt i32 %a, %b       ; Signed greater than
%ge = icmp sge i32 %a, %b       ; Signed greater or equal
%lt = icmp slt i32 %a, %b       ; Signed less than
%le = icmp sle i32 %a, %b       ; Signed less or equal
%gt = icmp ugt i32 %a, %b       ; Unsigned greater than

; Floating point comparisons
%eq = fcmp oeq float %a, %b     ; Ordered equal
%ne = fcmp one float %a, %b     ; Ordered not equal
%gt = fcmp ogt float %a, %b     ; Ordered greater than
%eq = fcmp ueq float %a, %b     ; Unordered equal (NaN handling)
```

### Memory Operations

```llvm
; Allocation
%ptr = alloca i32               ; Allocate on stack
%ptr = alloca i32, i32 10       ; Allocate array of 10 elements

; Load
%val = load i32, i32* %ptr      ; Load from memory
%val = load volatile i32, i32* %ptr  ; Volatile load

; Store
store i32 %val, i32* %ptr       ; Store to memory
store volatile i32 %val, i32* %ptr   ; Volatile store

; Memory ordering
%val = load atomic i32, i32* %ptr acquire
store atomic i32 %val, i32* %ptr release
```

### Conversion Operations

```llvm
; Integer extensions
%ext = zext i8 %val to i32      ; Zero extend
%ext = sext i8 %val to i32      ; Sign extend

; Integer truncation
%trunc = trunc i32 %val to i8   ; Truncate

; Integer/Float conversions
%f = sitofp i32 %i to float     ; Signed int to float
%f = uitofp i32 %i to float     ; Unsigned int to float
%i = fptosi float %f to i32     ; Float to signed int
%i = fptoui float %f to i32     ; Float to unsigned int

; Float conversions
%d = fpext float %f to double   ; Float to double
%f = fptrunc double %d to float ; Double to float

; Pointer conversions
%p = inttoptr i64 %val to i32*  ; Integer to pointer
%i = ptrtoint i32* %p to i64    ; Pointer to integer
%p2 = bitcast i32* %p to i8*    ; Pointer cast
```

### Aggregate Operations

```llvm
; Structure access
%val = extractvalue {i32, double} %struct, 0    ; Get first field
%struct2 = insertvalue {i32, double} %struct, double 3.14, 1  ; Set second field

; Array/Vector access
%elem = extractelement <4 x i32> %vec, i32 0    ; Get element
%vec2 = insertelement <4 x i32> %vec, i32 5, i32 0  ; Set element
```

### Pointer Operations

```llvm
; Get element pointer (GEP)
%ptr = getelementptr i32, i32* %base, i32 5         ; base + 5
%ptr = getelementptr [10 x i32], [10 x i32]* %arr, i32 0, i32 3  ; arr[3]

; Structure member access
%field_ptr = getelementptr %MyStruct, %MyStruct* %s, i32 0, i32 1
```

## SSA Form

Static Single Assignment: Each variable assigned exactly once.

### Non-SSA vs SSA

```c
// C code
int x = 1;
x = x + 2;
x = x * 3;
return x;
```

```llvm
; SSA form
define i32 @func() {
entry:
  %x1 = add i32 1, 2        ; x1 = 1 + 2
  %x2 = mul i32 %x1, 3      ; x2 = x1 * 3
  ret i32 %x2
}
```

### Phi Nodes

Phi nodes select value based on predecessor block:

```llvm
define i32 @abs(i32 %x) {
entry:
  %cmp = icmp slt i32 %x, 0
  br i1 %cmp, label %negative, label %positive

negative:
  %neg = sub i32 0, %x
  br label %end

positive:
  br label %end

end:
  ; Phi: if came from negative, use %neg; if from positive, use %x
  %result = phi i32 [%neg, %negative], [%x, %positive]
  ret i32 %result
}
```

## Function Calls

```llvm
; Direct call
%result = call i32 @func(i32 %arg1, i32 %arg2)

; Call with attributes
%result = call i32 @func(i32 signext %arg)

; Tail call (optimization)
%result = tail call i32 @func(i32 %arg)

; Indirect call (function pointer)
%result = call i32 %funcptr(i32 %arg)

; Variadic call
declare i32 @printf(i8*, ...)
%result = call i32 (i8*, ...) @printf(i8* %fmt, i32 %arg1, double %arg2)
```

## Global Variables

```llvm
; Simple global
@global_int = global i32 42

; Constant global
@constant_string = constant [6 x i8] c"Hello\00"

; Private global (internal linkage)
@private_var = private global i32 0

; Thread-local
@thread_var = thread_local global i32 0

; With initializer
@array = global [3 x i32] [i32 1, i32 2, i32 3]

; External (declared elsewhere)
@external_var = external global i32
```

## Control Flow Examples

### If-Else

```llvm
define i32 @max(i32 %a, i32 %b) {
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %then, label %else

then:
  br label %end

else:
  br label %end

end:
  %result = phi i32 [%a, %then], [%b, %else]
  ret i32 %result
}
```

### While Loop

```llvm
define i32 @sum_to_n(i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%next_i, %loop]
  %sum = phi i32 [0, %entry], [%next_sum, %loop]
  
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %body, label %end

body:
  %next_sum = add i32 %sum, %i
  %next_i = add i32 %i, 1
  br label %loop

end:
  ret i32 %sum
}
```

### For Loop

```llvm
define void @print_array([10 x i32]* %arr) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%next_i, %body]
  
  %cmp = icmp slt i32 %i, 10
  br i1 %cmp, label %body, label %end

body:
  %ptr = getelementptr [10 x i32], [10 x i32]* %arr, i32 0, i32 %i
  %val = load i32, i32* %ptr
  call void @print_int(i32 %val)
  
  %next_i = add i32 %i, 1
  br label %loop

end:
  ret void
}
```

## Metadata

Metadata provides additional information without affecting semantics.

```llvm
; Debug information
!0 = !DILocation(line: 10, column: 5, scope: !1)

; Optimization hints
!llvm.loop !{!2}
!2 = !{!"llvm.loop.unroll.count", i32 4}

; Aliasing information
store i32 %val, i32* %ptr, !alias.scope !3
!3 = !{!"scope1"}
```

## Attributes

```llvm
; Function attributes
attributes #0 = { noinline nounwind optnone }

; Parameter attributes
define void @func(i32* noalias nocapture %ptr)

; Return attributes
declare zeroext i32 @func()
```

## Summary

Key concepts covered:
- LLVM IR representations (in-memory, bitcode, assembly)
- Type system (primitive and derived types)
- Functions, basic blocks, and instructions
- SSA form and phi nodes
- Memory operations
- Control flow patterns
- Global variables
- Metadata and attributes

## Next Steps

In Chapter 3, we will learn how to build a compiler frontend that generates LLVM IR from a high-level language.
