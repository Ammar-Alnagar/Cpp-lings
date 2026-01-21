# Chapter 3: Asynchronous Programming with Futures

## Introduction to Task-Based Parallelism

Instead of managing threads directly, task-based parallelism focuses on what tasks need to be done rather than how threads execute them. C++ provides powerful abstractions for asynchronous programming through futures and promises.

## Overview of Async Components

The `<future>` header provides:
- **std::async**: Launch asynchronous tasks
- **std::future**: Retrieve results from asynchronous operations
- **std::promise**: Set values that futures can retrieve
- **std::packaged_task**: Wrap callable objects for async execution

## std::async

std::async launches a function asynchronously and returns a std::future to retrieve the result.

### Basic Usage

```cpp
#include <future>
#include <iostream>

int compute() {
    // Expensive computation
    return 42;
}

int main() {
    // Launch async task
    std::future<int> result = std::async(compute);
    
    // Do other work while computation runs
    
    // Get the result (blocks if not ready)
    int value = result.get();
    std::cout << value << '\n';
}
```

### Launch Policies

std::async accepts a launch policy that controls execution:

```cpp
// May run asynchronously in a new thread
auto f1 = std::async(std::launch::async, compute);

// May run synchronously when get() is called
auto f2 = std::async(std::launch::deferred, compute);

// Implementation decides (default)
auto f3 = std::async(std::launch::async | std::launch::deferred, compute);
auto f4 = std::async(compute);  // Same as f3
```

### Launch Policy Details

**std::launch::async**
- Guarantees execution in a new thread
- Starts immediately
- Best for CPU-bound tasks that must run in parallel

**std::launch::deferred**
- Executes synchronously when get() or wait() is called
- No new thread created
- Best for lazy evaluation scenarios

**std::launch::async | std::launch::deferred (default)**
- Implementation chooses the strategy
- May create thread pool or defer execution
- Most flexible but behavior is implementation-defined

## std::future

A std::future is a handle to a value that will be available in the future.

### Key Operations

```cpp
std::future<int> fut = std::async(compute);

// Check if result is ready (non-blocking)
if (fut.valid()) {
    std::future_status status = fut.wait_for(std::chrono::seconds(0));
    if (status == std::future_status::ready) {
        int value = fut.get();
    }
}

// Wait for result to be ready (blocking)
fut.wait();

// Wait with timeout
std::future_status status = fut.wait_for(std::chrono::seconds(1));
switch (status) {
    case std::future_status::ready:
        // Result is ready
        break;
    case std::future_status::timeout:
        // Still computing
        break;
    case std::future_status::deferred:
        // Not started yet (deferred execution)
        break;
}

// Get result (blocking, can only call once)
int result = fut.get();
```

### Important Properties

1. **Single-use**: get() can only be called once
2. **Move-only**: futures cannot be copied, only moved
3. **Exception propagation**: Exceptions in async tasks are stored and rethrown on get()

```cpp
auto fut = std::async([]() {
    throw std::runtime_error("Error!");
});

try {
    fut.get();  // Exception is rethrown here
} catch (const std::exception& e) {
    std::cout << e.what() << '\n';
}
```

## std::promise

A promise is the "write-end" of a promise-future communication channel.

### Basic Promise-Future Pattern

```cpp
#include <future>
#include <thread>

void compute_value(std::promise<int> prom) {
    // Perform computation
    int result = 42;
    
    // Set the value (makes future ready)
    prom.set_value(result);
}

int main() {
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();
    
    std::thread t(compute_value, std::move(prom));
    
    // Wait for result
    int value = fut.get();
    
    t.join();
}
```

### Promise Operations

```cpp
std::promise<int> prom;

// Get associated future (can only call once)
std::future<int> fut = prom.get_future();

// Set value (makes future ready)
prom.set_value(42);

// Set exception (future will throw on get())
prom.set_exception(std::make_exception_ptr(std::runtime_error("Error")));

// Set value at thread exit
prom.set_value_at_thread_exit(100);
```

### Breaking the Promise

If a promise is destroyed without setting a value, the future receives a std::future_error with error code std::future_errc::broken_promise.

```cpp
std::future<int> fut;
{
    std::promise<int> prom;
    fut = prom.get_future();
    // Promise destroyed without set_value()
}

try {
    fut.get();  // Throws std::future_error
} catch (const std::future_error& e) {
    // e.code() == std::future_errc::broken_promise
}
```

## std::packaged_task

A packaged_task wraps any callable and provides a future for its return value.

### Basic Usage

```cpp
#include <future>

int compute(int x, int y) {
    return x + y;
}

int main() {
    // Wrap the function
    std::packaged_task<int(int, int)> task(compute);
    
    // Get the future
    std::future<int> result = task.get_future();
    
    // Execute the task (in current thread or another)
    task(10, 20);
    
    // Get result
    int value = result.get();  // 30
}
```

### Using with Threads

```cpp
std::packaged_task<int(int, int)> task(compute);
std::future<int> result = task.get_future();

// Execute in separate thread
std::thread t(std::move(task), 10, 20);

// Get result from main thread
int value = result.get();

t.join();
```

### Task Queues

packaged_task is useful for building task queues:

```cpp
std::queue<std::packaged_task<void()>> task_queue;
std::vector<std::future<void>> futures;

// Add tasks to queue
for (int i = 0; i < 10; ++i) {
    std::packaged_task<void()> task([i]() {
        process(i);
    });
    futures.push_back(task.get_future());
    task_queue.push(std::move(task));
}

// Worker thread processes tasks
std::thread worker([&]() {
    while (!task_queue.empty()) {
        auto task = std::move(task_queue.front());
        task_queue.pop();
        task();
    }
});
```

## std::shared_future

Unlike std::future, std::shared_future allows multiple threads to wait for the same result.

```cpp
std::promise<int> prom;
std::shared_future<int> shared_fut = prom.get_future().share();

// Multiple threads can call get()
std::thread t1([shared_fut]() {
    int value = shared_fut.get();  // OK
});

std::thread t2([shared_fut]() {
    int value = shared_fut.get();  // Also OK
});

prom.set_value(42);

t1.join();
t2.join();
```

### Converting future to shared_future

```cpp
std::future<int> fut = std::async(compute);

// Method 1: Call share()
std::shared_future<int> shared1 = fut.share();

// Method 2: Move construct
std::shared_future<int> shared2(std::move(fut));
```

## Exception Handling

Exceptions in async tasks are captured and rethrown when get() is called.

```cpp
auto fut = std::async([]() {
    throw std::runtime_error("Task failed!");
    return 42;
});

try {
    int result = fut.get();  // Exception rethrown here
} catch (const std::runtime_error& e) {
    std::cout << "Caught: " << e.what() << '\n';
}
```

### Exception with Promise

```cpp
std::promise<int> prom;
std::future<int> fut = prom.get_future();

std::thread t([&prom]() {
    try {
        throw std::runtime_error("Computation failed");
    } catch (...) {
        // Capture current exception
        prom.set_exception(std::current_exception());
    }
});

try {
    fut.get();  // Exception rethrown
} catch (const std::runtime_error& e) {
    std::cout << e.what() << '\n';
}

t.join();
```

## Parallel Algorithms with Futures

Combine multiple async tasks for parallel processing.

### Parallel Map

```cpp
template<typename Iterator, typename Func>
auto parallel_map(Iterator begin, Iterator end, Func func) {
    std::vector<std::future<decltype(func(*begin))>> futures;
    
    for (auto it = begin; it != end; ++it) {
        futures.push_back(std::async(std::launch::async, func, *it));
    }
    
    std::vector<decltype(func(*begin))> results;
    for (auto& fut : futures) {
        results.push_back(fut.get());
    }
    
    return results;
}
```

### Parallel Accumulate

```cpp
template<typename Iterator, typename T>
T parallel_accumulate(Iterator begin, Iterator end, T init) {
    size_t length = std::distance(begin, end);
    if (length == 0) return init;
    
    size_t num_threads = std::thread::hardware_concurrency();
    size_t chunk_size = length / num_threads;
    
    std::vector<std::future<T>> futures;
    
    auto chunk_begin = begin;
    for (size_t i = 0; i < num_threads - 1; ++i) {
        auto chunk_end = std::next(chunk_begin, chunk_size);
        futures.push_back(std::async(std::launch::async,
            [](Iterator b, Iterator e) {
                return std::accumulate(b, e, T{});
            }, chunk_begin, chunk_end));
        chunk_begin = chunk_end;
    }
    
    // Last chunk (may be larger)
    T last_result = std::accumulate(chunk_begin, end, T{});
    
    // Collect results
    T total = init + last_result;
    for (auto& fut : futures) {
        total += fut.get();
    }
    
    return total;
}
```

## Continuation and Composition

C++11/14/17 doesn't provide built-in continuation support (though C++20 adds some with coroutines). Here's a pattern for chaining:

```cpp
// Manual continuation pattern
auto future1 = std::async(task1);
auto future2 = std::async([fut = std::move(future1)]() mutable {
    auto result1 = fut.get();
    return task2(result1);
});
auto result = future2.get();
```

## Best Practices

### 1. Prefer std::async for Simple Cases

For simple parallel tasks, std::async is cleaner than manual thread management:

```cpp
// Instead of this:
std::thread t([]() { /* work */ });
t.join();

// Use this:
auto fut = std::async(std::launch::async, []() { /* work */ });
fut.get();
```

### 2. Check future validity

```cpp
std::future<int> fut = std::async(compute);
if (fut.valid()) {
    int result = fut.get();
}
```

### 3. Don't Ignore Returned Futures

```cpp
// BAD: Blocks immediately in destructor!
std::async(std::launch::async, expensive_task);

// GOOD: Store future to control when blocking occurs
auto fut = std::async(std::launch::async, expensive_task);
// Do other work
fut.get();  // Block here when you need result
```

### 4. Specify Launch Policy

The default launch policy can behave unexpectedly. Be explicit:

```cpp
// Guaranteed parallelism
auto fut = std::async(std::launch::async, task);
```

### 5. Use packaged_task for Task Queues

For fine-grained control over task execution:

```cpp
std::packaged_task<int()> task(compute);
auto fut = task.get_future();
// Control exactly when and where task executes
```

## Performance Considerations

### Thread Creation Overhead

std::async may create a new thread for each call, which has overhead. For many tasks, consider:
- Thread pools
- Batch processing
- Limiting parallelism to hardware_concurrency()

### Future Destruction Blocks

Destroying a future from std::async blocks until the task completes:

```cpp
{
    auto fut = std::async(std::launch::async, long_task);
    // Destructor blocks here until long_task completes!
}
```

### Deferred Execution Surprise

Default launch policy may defer execution:

```cpp
auto fut = std::async(task);  // Might not start yet!
// Task may not run until get() is called
```

## Common Pitfalls

### 1. Calling get() Multiple Times

```cpp
auto fut = std::async(compute);
int x = fut.get();  // OK
int y = fut.get();  // ERROR: undefined behavior
```

### 2. Not Storing the Future

```cpp
// BAD: Blocks immediately!
std::async(std::launch::async, task);
// Future destructor is called, blocking current thread
```

### 3. Moving Promises Incorrectly

```cpp
std::promise<int> prom;
auto fut = prom.get_future();

std::thread t([prom]() {  // ERROR: promise is not copyable
    prom.set_value(42);
});
```

Fix: Move the promise:
```cpp
std::thread t([prom = std::move(prom)]() mutable {
    prom.set_value(42);
});
```

## Summary

Key concepts covered:
- std::async for launching asynchronous tasks
- Launch policies: async, deferred, and default
- std::future for retrieving async results
- std::promise for setting values from other threads
- std::packaged_task for wrapping callables
- std::shared_future for multiple waiters
- Exception propagation through futures
- Parallel algorithms using futures
- Best practices and common pitfalls

## Next Steps

In the next chapter, we will explore atomic operations and lock-free programming for high-performance synchronization without mutexes.
