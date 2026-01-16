#include <stdio.h>
#include <openacc.h>
#include <math.h>

// Basic vector addition example
void vectorAddExample() {
    const int N = 1024;
    float a[N], b[N], c[N];
    
    // Initialize arrays
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }
    
    printf("=== OpenACC Vector Addition Example ===\n");
    
    // Parallel computation with OpenACC
    #pragma acc parallel loop copyin(a[0:N], b[0:N]) copyout(c[0:N])
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
    
    printf("First 10 results of vector addition: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", c[i]);
    }
    printf("\n");
}

// Matrix multiplication example
void matrixMultExample() {
    const int N = 64;
    float A[N][N], B[N][N], C[N][N];
    
    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = i * N + j;
            B[i][j] = (i * N + j) * 0.1f;
            C[i][j] = 0.0f;
        }
    }
    
    printf("\n=== OpenACC Matrix Multiplication Example ===\n");
    
    // Matrix multiplication using OpenACC
    #pragma acc parallel loop collapse(2) copyin(A[0:N][0:N], B[0:N][0:N]) copyout(C[0:N][0:N])
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            #pragma acc loop seq
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    printf("First 5x5 elements of result matrix:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%.2f ", C[i][j]);
        }
        printf("\n");
    }
}

// Reduction example
void reductionExample() {
    const int N = 1024;
    float a[N];
    float sum = 0.0f;
    
    // Initialize array
    for (int i = 0; i < N; i++) {
        a[i] = i * 0.1f;
    }
    
    printf("\n=== OpenACC Reduction Example ===\n");
    
    // Compute sum using reduction
    #pragma acc parallel loop reduction(+:sum) copyin(a[0:N])
    for (int i = 0; i < N; i++) {
        sum += a[i];
    }
    
    printf("Sum of array: %.2f\n", sum);
}

// Data region example
void dataRegionExample() {
    const int N = 1024;
    float a[N], b[N], c[N];
    
    // Initialize array
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
    }
    
    printf("\n=== OpenACC Data Region Example ===\n");
    
    // Use data region for multiple operations
    #pragma acc data copyin(a[0:N]) copyout(c[0:N]) create(b[0:N])
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
    }
    
    printf("First 10 results after data region operations: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", c[i]);
    }
    printf("\n");
}

// Async execution example
void asyncExample() {
    const int N = 1024;
    float a[N], b[N], c[N], d[N];
    
    // Initialize arrays
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }
    
    printf("\n=== OpenACC Async Execution Example ===\n");
    
    // First async computation
    #pragma acc parallel loop async(1) copyin(a[0:N]) copyout(c[0:N])
    for (int i = 0; i < N; i++) {
        c[i] = a[i] * 3.0f;
    }
    
    // Second async computation
    #pragma acc parallel loop async(2) copyin(b[0:N]) copyout(d[0:N])
    for (int i = 0; i < N; i++) {
        d[i] = b[i] * 4.0f;
    }
    
    // Wait for both computations to complete
    #pragma acc wait(1, 2)
    
    printf("First 10 results of async computation (c): ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", c[i]);
    }
    printf("\n");
    
    printf("First 10 results of async computation (d): ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", d[i]);
    }
    printf("\n");
}

// Update directive example
void updateExample() {
    const int N = 128;
    float a[N];
    
    // Initialize on host
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
    }
    
    printf("\n=== OpenACC Update Directive Example ===\n");
    
    // Create data region
    #pragma acc data create(a[0:N])
    {
        // Copy data to device
        #pragma acc update device(a[0:N])
        
        // Modify on device
        #pragma acc parallel loop
        for (int i = 0; i < N; i++) {
            a[i] = a[i] * 2.0f + 1.0f;
        }
        
        // Copy results back to host
        #pragma acc update self(a[0:N])
    }
    
    printf("First 10 values after update operations: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", a[i]);
    }
    printf("\n");
}

// Runtime API example
void runtimeApiExample() {
    printf("\n=== OpenACC Runtime API Example ===\n");
    
    // Check if running on device
    if (acc_on_device(acc_device_not_host)) {
        printf("Running on accelerator\n");
    } else {
        printf("Running on host\n");
    }
    
    // Get number of devices
    int num_devices = acc_get_num_devices(acc_device_nvidia);
    printf("Number of NVIDIA devices: %d\n", num_devices);
    
    // Set device if available
    if (num_devices > 0) {
        acc_set_device_num(0, acc_device_nvidia);
        printf("Set device to 0\n");
    }
    
    // Get device number
    int device_num = acc_get_device_num(acc_device_nvidia);
    printf("Current device number: %d\n", device_num);
}

int main() {
    printf("=== OpenACC Examples ===\n");
    
    // Check if OpenACC is available
    if (acc_get_num_devices(acc_device_not_host) > 0) {
        printf("OpenACC is available and devices are present.\n");
    } else {
        printf("OpenACC is not available or no devices found.\n");
    }
    
    // Run examples
    vectorAddExample();
    matrixMultExample();
    reductionExample();
    dataRegionExample();
    asyncExample();
    updateExample();
    runtimeApiExample();
    
    printf("\nAll OpenACC examples completed successfully.\n");
    printf("Compile with: gcc -fopenacc -o openacc_example openacc_example.c\n");
    
    return 0;
}