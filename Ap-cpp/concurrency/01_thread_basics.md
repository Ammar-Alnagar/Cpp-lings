# Chapter 1: Thread Basics

## Introduction to Concurrency

Concurrency is the ability of a program to execute multiple tasks simultaneously. In modern C++, the standard library provides robust support for multithreading, eliminating the need for platform-specific APIs in most cases.

### Why Concurrency?

1. **Performance**: Utilize multiple CPU cores for parallel computation
2. **Responsiveness**: Keep UI responsive while performing background tasks
3. **Throughput**: Process multiple requests simultaneously in servers
4. **Resource Utilization**: Better use of available hardware resources

### Concurrency vs Parallelism

- **Concurrency**: Multiple tasks making progress (may or may not run simultaneously)
- **Parallelism**: Multiple tasks actually running at the same time on different cores

## Thread Fundamentals

A thread is the smallest unit of execution that can be scheduled by the operating system. Every C++ program starts with at least one thread: the main thread.

### Creating Threads

The `std::thread` class (from `<thread>` header) represents a single thread of execution.

```cpp
#include <thread>
#include <iostream>

void hello() {
    std::cout << "Hello from thread!\n";
}

int main() {
    std::thread t(hello);  // Create and start thread
    t.join();              // Wait for thread to finish
    return 0;
}
```

### Thread Lifecycle

1. **Creation**: Thread object is constructed with a callable
2. **Execution**: Thread begins executing immediately
3. **Joinable**: Thread is running or has finished but not yet joined
4. **Joined/Detached**: Thread resources are released

## Joining and Detaching Threads

### join()

Blocks the calling thread until the thread represented by the thread object completes.

```cpp
std::thread t(work);
t.join();  // Wait for t to finish
// Now safe to proceed
```

### detach()

Separates the thread object from the thread of execution, allowing them to execute independently.

```cpp
std::thread t(background_work);
t.detach();  // Let it run independently
// Thread object is no longer associated with the execution thread
```

### Important Rules

1. A thread must be either joined or detached before destruction
2. Calling join() or detach() on a non-joinable thread is undefined behavior
3. Check `joinable()` before calling join() or detach()

## Passing Arguments to Threads

Arguments are passed by value by default. Use `std::ref` or `std::cref` for reference passing.

```cpp
void print_sum(int a, int b) {
    std::cout << a + b << '\n';
}

void modify_value(int& x) {
    x = 100;
}

int main() {
    std::thread t1(print_sum, 10, 20);  // Pass by value
    
    int value = 50;
    std::thread t2(modify_value, std::ref(value));  // Pass by reference
    
    t1.join();
    t2.join();
    // value is now 100
}
```

## Thread Identifiers

Each thread has a unique identifier accessible via `std::thread::get_id()` or `std::this_thread::get_id()`.

```cpp
std::thread::id main_thread_id = std::this_thread::get_id();
std::thread t([]() {
    std::thread::id worker_id = std::this_thread::get_id();
    // worker_id != main_thread_id
});
```

## Hardware Concurrency

Query the number of concurrent threads supported by the hardware:

```cpp
unsigned int n = std::thread::hardware_concurrency();
// Returns 0 if the value is not well-defined or computable
```

## Common Pitfalls

### 1. Forgetting to Join or Detach

```cpp
void bad_example() {
    std::thread t(work);
    // Forgot to join or detach!
}  // Destructor called on joinable thread -> std::terminate()
```

### 2. Accessing Local Variables After Thread Detachment

```cpp
void dangerous() {
    int local_var = 42;
    std::thread t([&]() {
        std::cout << local_var;  // DANGER!
    });
    t.detach();
}  // local_var destroyed, but thread may still access it
```

### 3. Exception Safety

```cpp
void exception_safe() {
    std::thread t(work);
    try {
        risky_operation();
    } catch (...) {
        t.join();  // Must join before exception propagates
        throw;
    }
    t.join();
}
```

Better approach: Use RAII wrapper (see examples).

## Thread Management Best Practices

1. Always ensure threads are joined or detached
2. Use RAII wrappers for automatic thread management
3. Be careful with thread lifetimes and data access
4. Prefer returning values over modifying shared state
5. Use thread pools for managing many short-lived tasks
6. Limit the number of threads to available hardware cores

## Performance Considerations

### Thread Creation Overhead

Creating threads is expensive:
- Kernel resources allocation
- Stack allocation (typically 1-2 MB)
- Context switching overhead

For many small tasks, consider thread pools.

### Cache Effects

Threads on different cores have separate caches. Sharing data between threads can cause cache coherency traffic, impacting performance.

### False Sharing

When different threads access different variables that share a cache line, performance degrades. Align and pad data structures to avoid this.

## Summary

Key concepts covered:
- Thread creation with `std::thread`
- Thread lifecycle: creation, execution, joining/detaching
- Passing arguments to threads
- Thread identifiers and hardware concurrency
- Common pitfalls and best practices
- Performance considerations

## Next Steps

In the next chapter, we will explore synchronization primitives (mutexes, locks, condition variables) that enable safe communication between threads.

## Code Examples

See the `examples/` directory for complete, runnable code demonstrating:
- Basic thread creation and management
- Passing arguments (by value and reference)
- RAII thread wrappers
- Thread pools (basic implementation)
- Exception-safe thread handling
