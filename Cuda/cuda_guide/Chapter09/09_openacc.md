# Chapter 9: OpenACC

## Overview
OpenACC is a directive-based parallel programming standard designed to simplify parallel programming on accelerators like GPUs. Unlike CUDA, which requires explicit kernel programming, OpenACC uses compiler directives to indicate regions of code that should be parallelized on accelerators. This chapter covers OpenACC fundamentals, directives, and best practices.

## OpenACC Basics

### What is OpenACC?
OpenACC is a specification for compiler directives that allows programmers to express parallelism in their code without having to manage low-level details. The directives guide the compiler to generate parallel code for accelerators.

### Key Concepts:
- **Directives**: Pragma-based annotations that guide the compiler
- **Kernels**: Regions of code to be executed in parallel on the accelerator
- **Data clauses**: Specify how data moves between host and device
- **Loop constructs**: Indicate which loops to parallelize

## OpenACC Directives

### Parallel Construct
The `parallel` construct indicates a region of code to be executed in parallel on the accelerator.

```c
#include <openacc.h>
#include <stdio.h>

void parallelExample() {
    const int N = 1024;
    float a[N], b[N], c[N];
    
    // Initialize arrays
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }
    
    // Parallel region with data movement
    #pragma acc parallel loop copyin(a[0:N], b[0:N]) copyout(c[0:N])
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
    
    // Verify results
    printf("First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", c[i]);
    }
    printf("\n");
}
```

### Kernels Construct
The `kernels` construct allows the compiler to determine what to parallelize automatically.

```c
void kernelsExample() {
    const int N = 1024;
    float a[N], b[N], c[N];
    
    // Initialize arrays
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }
    
    // Kernels region - compiler decides what to parallelize
    #pragma acc kernels copyin(a[0:N], b[0:N]) copyout(c[0:N])
    {
        for (int i = 0; i < N; i++) {
            c[i] = a[i] + b[i];
        }
        
        for (int i = 0; i < N; i++) {
            a[i] = c[i] * 2.0f;
        }
    }
    
    // Verify results
    printf("First 10 results after kernels: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", a[i]);
    }
    printf("\n");
}
```

## Data Management

### Data Clauses
OpenACC provides several clauses to manage data movement between host and device:

```c
void dataManagementExample() {
    const int N = 1024;
    float a[N], b[N], c[N];
    
    // Initialize arrays
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }
    
    // Create data region with different clauses
    #pragma acc data copyin(a[0:N], b[0:N]) copy(c[0:N])
    {
        // First computation
        #pragma acc parallel loop
        for (int i = 0; i < N; i++) {
            c[i] = a[i] + b[i];
        }
        
        // Second computation using result
        #pragma acc parallel loop
        for (int i = 0; i < N; i++) {
            c[i] = c[i] * 2.0f;
        }
    }
    
    // Results are automatically copied back due to 'copy' clause
    printf("First 10 results after data region: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", c[i]);
    }
    printf("\n");
}
```

### Common Data Clauses:
- `copyin(array[start:length])`: Copy data to device, don't copy back
- `copyout(array[start:length])`: Don't copy to device, copy back results
- `copy(array[start:length])`: Copy data to and from device
- `create(array[start:length])`: Create space on device, don't initialize
- `present(array[start:length])`: Assume data is already on device

## Loop Directives

### Independent Loops
Use the `loop` directive to indicate that iterations are independent:

```c
void independentLoops() {
    const int N = 1024;
    float a[N], b[N], c[N];
    
    // Initialize
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
    }
    
    // Independent loop - iterations can run in parallel
    #pragma acc parallel loop copy(a[0:N]) copyout(b[0:N])
    for (int i = 0; i < N; i++) {
        b[i] = a[i] * 2.0f;
    }
    
    // Another independent loop
    #pragma acc parallel loop copy(b[0:N]) copyout(c[0:N])
    for (int i = 0; i < N; i++) {
        c[i] = b[i] + 1.0f;
    }
    
    printf("Independent loops completed.\n");
}
```

### Gang, Worker, Vector Directives
These directives map to different levels of parallelism:

```c
void gangWorkerVectorExample() {
    const int N = 1024;
    float a[N], b[N];
    
    // Initialize
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
    }
    
    // gang: coarse-grain parallelism (blocks in CUDA)
    // worker: medium-grain parallelism (threads in a block)
    // vector: fine-grain parallelism (SIMD within a thread)
    #pragma acc parallel loop gang worker vector copyin(a[0:N]) copyout(b[0:N])
    for (int i = 0; i < N; i++) {
        b[i] = a[i] * a[i] + 2.0f * a[i] + 1.0f;
    }
    
    printf("Gang-worker-vector example completed.\n");
}
```

## Advanced OpenACC Features

### Conditional Parallelization
```c
void conditionalParallelization() {
    const int N = 1024;
    float a[N], b[N];
    
    // Initialize
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
    }
    
    // Only parallelize if accelerator is available
    #pragma acc parallel loop if(acc_on_device(acc_device_not_host))
    for (int i = 0; i < N; i++) {
        b[i] = a[i] * 2.0f;
    }
    
    printf("Conditional parallelization completed.\n");
}
```

### Async Execution
```c
void asyncExecution() {
    const int N = 1024;
    float a[N], b[N], c[N], d[N];
    
    // Initialize
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }
    
    // First async computation
    #pragma acc parallel loop async(1) copyin(a[0:N]) copyout(c[0:N])
    for (int i = 0; i < N; i++) {
        c[i] = a[i] * 2.0f;
    }
    
    // Second async computation
    #pragma acc parallel loop async(2) copyin(b[0:N]) copyout(d[0:N])
    for (int i = 0; i < N; i++) {
        d[i] = b[i] * 3.0f;
    }
    
    // Wait for both computations to complete
    #pragma acc wait(1, 2)
    
    printf("Async execution completed.\n");
}
```

### Update Directive
The `update` directive explicitly moves data between host and device:

```c
void updateDirectiveExample() {
    const int N = 1024;
    float a[N];
    
    // Initialize on host
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
    }
    
    // Create data region
    #pragma acc data create(a[0:N])
    {
        // Copy data to device
        #pragma acc update device(a[0:N])
        
        // Modify on device
        #pragma acc parallel loop
        for (int i = 0; i < N; i++) {
            a[i] = a[i] * 2.0f;
        }
        
        // Copy results back to host
        #pragma acc update self(a[0:N])
    }
    
    printf("Update directive example completed.\n");
    printf("First 10 values: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", a[i]);
    }
    printf("\n");
}
```

## Performance Optimization

### Loop Collapse
Combine nested loops for better parallelization:

```c
void loopCollapseExample() {
    const int ROWS = 64, COLS = 64;
    float matrix[ROWS][COLS];
    
    // Initialize
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            matrix[i][j] = i * COLS + j;
        }
    }
    
    // Collapse nested loops for better parallelization
    #pragma acc parallel loop collapse(2) copy(matrix[0:ROWS][0:COLS])
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            matrix[i][j] = matrix[i][j] * 2.0f;
        }
    }
    
    printf("Loop collapse example completed.\n");
}
```

### Cache Blocking
```c
void cacheBlockingExample() {
    const int N = 1024;
    float a[N][N], b[N][N];
    
    // Initialize
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = i * N + j;
            b[i][j] = 0.0f;
        }
    }
    
    const int BLOCK_SIZE = 32;
    
    // Blocked computation for better cache utilization
    #pragma acc parallel loop gang collapse(2) copyin(a[0:N][0:N]) copyout(b[0:N][0:N])
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            #pragma acc loop worker vector collapse(2)
            for (int i = ii; i < ii + BLOCK_SIZE && i < N; i++) {
                for (int j = jj; j < jj + BLOCK_SIZE && j < N; j++) {
                    b[i][j] = a[i][j] * 2.0f;
                }
            }
        }
    }
    
    printf("Cache blocking example completed.\n");
}
```

## OpenACC with C++

### OpenACC in C++ Code
```cpp
#include <openacc.h>
#include <vector>
#include <iostream>

class OpenACCAccelerator {
private:
    std::vector<float> data;
    int size;

public:
    OpenACCAccelerator(int n) : size(n) {
        data.resize(n);
    }
    
    void initialize() {
        #pragma acc parallel loop copyout(data[0:size])
        for (int i = 0; i < size; i++) {
            data[i] = i * 1.0f;
        }
    }
    
    void process() {
        #pragma acc parallel loop copy(data[0:size])
        for (int i = 0; i < size; i++) {
            data[i] = data[i] * data[i] + 1.0f;
        }
    }
    
    float getResult(int index) {
        return data[index];
    }
    
    void printFirstN(int n) {
        std::cout << "First " << n << " results: ";
        for (int i = 0; i < n && i < size; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
};

void cppExample() {
    OpenACCAccelerator acc(1024);
    acc.initialize();
    acc.process();
    acc.printFirstN(10);
}
```

## Comparison with CUDA

### OpenACC vs CUDA
| Feature | OpenACC | CUDA |
|---------|---------|------|
| Programming Model | Directive-based | Explicit kernel programming |
| Portability | High (works on different accelerators) | GPU-specific |
| Control | Limited | Fine-grained control |
| Learning Curve | Gentle | Steeper |
| Performance Tuning | Automatic optimization | Manual tuning required |

### When to Use OpenACC
- Rapid prototyping of parallel code
- Porting existing serial code to accelerators
- When portability across different accelerator types is important
- When developer productivity is prioritized over maximum performance

### When to Use CUDA
- When maximum performance is required
- When fine-grained control over GPU resources is needed
- When using GPU-specific features not available through OpenACC
- When developing specialized GPU algorithms

## OpenACC Runtime API

### Runtime Functions
```c
#include <openacc.h>

void runtimeApiExample() {
    // Check if running on device
    if (acc_on_device(acc_device_not_host)) {
        printf("Running on accelerator\n");
    } else {
        printf("Running on host\n");
    }
    
    // Get number of devices
    int num_devices = acc_get_num_devices(acc_device_nvidia);
    printf("Number of NVIDIA devices: %d\n", num_devices);
    
    // Set device
    if (num_devices > 0) {
        acc_set_device_num(0, acc_device_nvidia);
        printf("Set device to 0\n");
    }
    
    // Get device type
    acc_device_t device_type = acc_get_device_type();
    printf("Current device type: %d\n", device_type);
    
    // Get device number
    int device_num = acc_get_device_num(acc_device_nvidia);
    printf("Current device number: %d\n", device_num);
}
```

## Best Practices

### Data Movement Optimization
```c
void dataOptimizationExample() {
    const int N = 1024;
    float a[N], b[N], c[N], d[N];
    
    // Initialize
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
    }
    
    // Bad: Multiple data regions
    // #pragma acc data copyin(a[0:N]) copyout(b[0:N]) { ... }
    // #pragma acc data copyin(b[0:N]) copyout(c[0:N]) { ... }
    
    // Good: Single data region for multiple operations
    #pragma acc data copyin(a[0:N]) copyout(d[0:N]) create(b[0:N], c[0:N])
    {
        // First operation
        #pragma acc parallel loop
        for (int i = 0; i < N; i++) {
            b[i] = a[i] * 2.0f;
        }
        
        // Second operation
        #pragma acc parallel loop
        for (int i = 0; i < N; i++) {
            c[i] = b[i] + 1.0f;
        }
        
        // Third operation
        #pragma acc parallel loop
        for (int i = 0; i < N; i++) {
            d[i] = c[i] * 0.5f;
        }
    }
    
    printf("Data optimization example completed.\n");
}
```

### Loop Optimization
```c
void loopOptimizationExample() {
    const int N = 1024;
    float a[N], b[N];
    
    // Initialize
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
    }
    
    // Good: Indicate independence explicitly
    #pragma acc parallel loop independent copyin(a[0:N]) copyout(b[0:N])
    for (int i = 0; i < N; i++) {
        b[i] = a[i] * 2.0f;
    }
    
    // For reductions
    float sum = 0.0f;
    #pragma acc parallel loop reduction(+:sum) copyin(b[0:N])
    for (int i = 0; i < N; i++) {
        sum += b[i];
    }
    
    printf("Sum of array: %.2f\n", sum);
}
```

## Debugging OpenACC Code

### Debugging Tips
```c
void debuggingTips() {
    const int N = 1024;
    float a[N], b[N];
    
    // Initialize
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
    }
    
    // Use 'if' clause to disable acceleration for debugging
    #pragma acc parallel loop if(.false.) copyin(a[0:N]) copyout(b[0:N])
    for (int i = 0; i < N; i++) {
        b[i] = a[i] * 2.0f;
    }
    
    // Verify correctness on host
    for (int i = 0; i < 10; i++) {
        if (b[i] != a[i] * 2.0f) {
            printf("Error at index %d: expected %.2f, got %.2f\n", 
                   i, a[i] * 2.0f, b[i]);
        }
    }
    
    printf("Debugging tips example completed.\n");
}
```

## Summary
This chapter covered:
- OpenACC fundamentals and basic concepts
- Parallel and kernels constructs
- Data management with various clauses
- Loop directives and parallelization strategies
- Advanced features like async execution and updates
- Performance optimization techniques
- OpenACC with C++
- Comparison with CUDA
- Runtime API functions
- Best practices for optimization
- Debugging techniques

OpenACC provides a higher-level approach to GPU programming compared to CUDA, making it easier to parallelize existing code. The next chapter will explore deep learning with CUDA.