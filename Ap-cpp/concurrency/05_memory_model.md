# Chapter 5: The C++ Memory Model

## Introduction to Memory Models

The C++ memory model defines how threads interact through memory. Understanding it is crucial for writing correct concurrent code, especially when using atomic operations and lock-free programming.

## Why Memory Models Matter

### The Problem: Compiler and Hardware Reordering

Modern systems can reorder operations for performance:

```cpp
// Original code
int data = 0;
bool ready = false;

// Thread 1
data = 42;        // 1
ready = true;     // 2

// Thread 2
if (ready) {      // 3
    use(data);    // 4
}
```

Without proper synchronization, operations 1 and 2 might be reordered, causing Thread 2 to see `ready == true` but `data == 0`.

### Sources of Reordering

1. **Compiler Optimization**
   - Instruction reordering
   - Register allocation
   - Dead code elimination
   - Common subexpression elimination

2. **CPU Out-of-Order Execution**
   - Instruction-level parallelism
   - Speculative execution
   - Store buffers and write combining

3. **Cache Coherency Protocol**
   - Cache line invalidation delays
   - Store buffer forwarding
   - Memory barrier costs

## Sequential Consistency

### Definition

Sequential consistency provides the strongest guarantee: all operations appear to execute in some global sequential order, consistent with program order within each thread.

```cpp
// Sequential consistency guarantees:
// - Thread 1's operations appear in order: A before B
// - Thread 2's operations appear in order: C before D
// - There exists a total order consistent with these constraints

// Thread 1
A: x = 1;
B: y = 1;

// Thread 2
C: int r1 = y;
D: int r2 = x;

// Impossible: r1 == 1 && r2 == 0
// Because if C sees y==1, then B executed before C
// And A executed before B, so A must execute before D
```

### Cost of Sequential Consistency

Sequential consistency requires memory barriers, which are expensive:
- Flush store buffers
- Invalidate caches
- Prevent reordering

C++ default: `std::memory_order_seq_cst` provides this guarantee.

## The C++ Memory Model

### Memory Locations

A memory location is:
- An object of scalar type, OR
- A maximal sequence of adjacent bit-fields

Different memory locations can be accessed concurrently without synchronization (if at least one is read-only).

### Data Races

A data race occurs when:
1. Two threads access the same memory location
2. At least one access is a write
3. Accesses are not synchronized

**Data races cause undefined behavior!**

### Happens-Before Relationship

The happens-before relationship defines ordering between operations:

1. **Sequenced-Before** (within a thread)
   - Operations in the same thread have a sequenced-before order
   - Following program order

2. **Synchronizes-With** (between threads)
   - Release operation synchronizes-with acquire operation
   - Unlock synchronizes-with subsequent lock
   - Thread creation synchronizes-with thread start

3. **Happens-Before** (transitive)
   - If A sequenced-before B and B happens-before C, then A happens-before C
   - If A synchronizes-with B, then A happens-before B

### Example: Happens-Before

```cpp
std::atomic<bool> ready{false};
int data = 0;

// Thread 1
data = 42;                                    // A
ready.store(true, std::memory_order_release); // B (release)

// Thread 2
if (ready.load(std::memory_order_acquire)) {  // C (acquire)
    assert(data == 42);                       // D
}

// Relationships:
// A sequenced-before B (same thread)
// B synchronizes-with C (release-acquire)
// Therefore: A happens-before D
// D will see data == 42
```

## Memory Ordering

C++ provides six memory orderings:

### 1. memory_order_relaxed

**No synchronization, only atomicity**

```cpp
std::atomic<int> counter{0};

// Thread 1
counter.fetch_add(1, std::memory_order_relaxed);

// Thread 2
counter.fetch_add(1, std::memory_order_relaxed);

// Guarantee: Final value will be 2
// No guarantee: Order visible to other operations
```

Use case: Pure counters where ordering doesn't matter.

### 2. memory_order_acquire

**Load-acquire**: Prevents reordering of subsequent reads/writes before this load.

```cpp
// Thread 1
data = 42;
flag.store(true, std::memory_order_release);

// Thread 2
if (flag.load(std::memory_order_acquire)) {
    // All writes before the release are visible here
    assert(data == 42);
}
```

### 3. memory_order_release

**Store-release**: Prevents reordering of prior reads/writes after this store.

```cpp
// Thread 1
data = 42;                                    // Cannot move after release
flag.store(true, std::memory_order_release);

// Thread 2 (with acquire)
// Sees all writes before the release
```

### 4. memory_order_acq_rel

**Acquire-Release**: Combines acquire and release for read-modify-write operations.

```cpp
std::atomic<int> x{0};

// Read-modify-write with acquire-release
int old = x.fetch_add(1, std::memory_order_acq_rel);

// Acts as acquire for the read
// Acts as release for the write
```

### 5. memory_order_consume

**Load-consume**: Like acquire but only for dependent operations (rarely used, complex).

```cpp
std::atomic<int*> ptr;

// Thread 1
int* p = new int(42);
ptr.store(p, std::memory_order_release);

// Thread 2
int* p2 = ptr.load(std::memory_order_consume);
int value = *p2;  // Only this dependent read is synchronized
```

Note: Most implementations treat consume as acquire.

### 6. memory_order_seq_cst

**Sequential consistency**: Default, strongest guarantee.

```cpp
std::atomic<int> x{0}, y{0};
int r1, r2;

// Thread 1
x.store(1, std::memory_order_seq_cst);
r1 = y.load(std::memory_order_seq_cst);

// Thread 2
y.store(1, std::memory_order_seq_cst);
r2 = x.load(std::memory_order_seq_cst);

// Guaranteed: !(r1 == 0 && r2 == 0)
// At least one thread will see the other's store
```

## Memory Order Relationships

### Release-Acquire Synchronization

```cpp
std::atomic<bool> ready{false};
int data;

// Producer
data = 42;                                    // 1
ready.store(true, std::memory_order_release); // 2

// Consumer
while (!ready.load(std::memory_order_acquire)); // 3
use(data);                                     // 4

// Guarantee: 1 happens-before 4
```

### Relaxed Ordering

```cpp
std::atomic<int> x{0}, y{0};

// Thread 1
x.store(1, std::memory_order_relaxed);
y.store(1, std::memory_order_relaxed);

// Thread 2
int r1 = y.load(std::memory_order_relaxed);
int r2 = x.load(std::memory_order_relaxed);

// Possible: r1 == 1 && r2 == 0
// No synchronization between threads
```

### Sequential Consistency

```cpp
std::atomic<int> x{0}, y{0};

// Thread 1
x.store(1, std::memory_order_seq_cst);

// Thread 2
y.store(1, std::memory_order_seq_cst);

// Thread 3
int r1 = x.load(std::memory_order_seq_cst);
int r2 = y.load(std::memory_order_seq_cst);

// Thread 4
int r3 = y.load(std::memory_order_seq_cst);
int r4 = x.load(std::memory_order_seq_cst);

// Guarantee: If r1 == 0 && r2 == 1, then r3 == 1 || r4 == 1
// There is a global sequential order
```

## Fences (Memory Barriers)

Fences provide synchronization without atomic operations.

### Release Fence

```cpp
int data;
std::atomic<bool> ready{false};

// Thread 1
data = 42;
std::atomic_thread_fence(std::memory_order_release);
ready.store(true, std::memory_order_relaxed);

// Acts as if ready.store used release ordering
```

### Acquire Fence

```cpp
// Thread 2
if (ready.load(std::memory_order_relaxed)) {
    std::atomic_thread_fence(std::memory_order_acquire);
    use(data);
}

// Acts as if ready.load used acquire ordering
```

### Bidirectional Fence

```cpp
std::atomic_thread_fence(std::memory_order_seq_cst);
// Full memory barrier
```

## Common Patterns

### Double-Checked Locking (Correct)

```cpp
std::atomic<Widget*> instance{nullptr};
std::mutex mtx;

Widget* get_instance() {
    Widget* tmp = instance.load(std::memory_order_acquire);
    if (tmp == nullptr) {
        std::lock_guard<std::mutex> lock(mtx);
        tmp = instance.load(std::memory_order_relaxed);
        if (tmp == nullptr) {
            tmp = new Widget();
            instance.store(tmp, std::memory_order_release);
        }
    }
    return tmp;
}
```

### Message Passing

```cpp
struct Message {
    int data[100];
};

std::atomic<Message*> mailbox{nullptr};

// Sender
void send(Message* msg) {
    // ... fill message ...
    mailbox.store(msg, std::memory_order_release);
}

// Receiver
Message* receive() {
    Message* msg;
    while ((msg = mailbox.exchange(nullptr, std::memory_order_acquire)) == nullptr) {
        // Wait
    }
    return msg;
}
```

### Publication

```cpp
struct Data {
    int value1;
    int value2;
};

Data* data = nullptr;
std::atomic<bool> ready{false};

// Publisher
void publish() {
    Data* d = new Data{1, 2};
    data = d;  // Non-atomic write OK
    ready.store(true, std::memory_order_release);
}

// Subscriber
void subscribe() {
    while (!ready.load(std::memory_order_acquire));
    use(data->value1);  // Safe: happens-after publication
}
```

## Architecture-Specific Considerations

### x86/x64

- Strong memory model (almost sequential consistency)
- Stores are not reordered with stores
- Loads are not reordered with loads
- Loads are not reordered with stores
- Stores CAN be reordered with loads (store buffer)

### ARM/ARM64

- Weak memory model
- Extensive reordering possible
- Requires explicit barriers
- memory_order_relaxed essentially free
- memory_order_seq_cst expensive

### Power PC

- Very weak memory model
- Even more reordering than ARM
- Performance critical to use appropriate ordering

## Performance Guidelines

### Choose Minimal Ordering

```cpp
// BAD: Unnecessary seq_cst
std::atomic<int> counter{0};
counter.fetch_add(1, std::memory_order_seq_cst);

// GOOD: Relaxed is sufficient for counters
counter.fetch_add(1, std::memory_order_relaxed);
```

### Relative Costs (typical)

1. **Relaxed**: ~Free (regular atomic operation)
2. **Acquire/Release**: ~1-2 cycles overhead
3. **Seq_cst**: ~10-100 cycles (platform-dependent)

### Optimization Strategy

1. Start with seq_cst (safest)
2. Profile to find bottlenecks
3. Relax ordering where proven safe
4. Document reasoning thoroughly
5. Test on weakest target architecture

## Testing and Debugging

### Tools

1. **Thread Sanitizer** (TSan)
   - Detects data races
   - Compile with `-fsanitize=thread`
   - Catches most concurrency bugs

2. **Relacy Race Detector**
   - Systematic testing of all interleavings
   - Catches ordering issues
   - Research/testing tool

3. **Memory Order Simulator**
   - Test behavior under different architectures
   - Verify weak memory model scenarios

### Testing Strategy

```cpp
// Stress test with many threads
void stress_test() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 1000; ++i) {
        threads.emplace_back(worker_function);
    }
    for (auto& t : threads) {
        t.join();
    }
    verify_invariants();
}
```

## Common Pitfalls

### 1. Forgetting Acquire on Load

```cpp
// WRONG
if (flag.load(std::memory_order_relaxed)) {  // Relaxed!
    use(data);  // Data race!
}

// RIGHT
if (flag.load(std::memory_order_acquire)) {
    use(data);  // Safe
}
```

### 2. Mixing Atomic and Non-Atomic Access

```cpp
std::atomic<int> x{0};

// Thread 1
x.store(1, std::memory_order_release);

// Thread 2
int* p = reinterpret_cast<int*>(&x);
*p = 2;  // WRONG: Data race with atomic operations
```

### 3. Assuming Sequential Consistency

```cpp
// These operations might be reordered
x.store(1, std::memory_order_relaxed);
y.store(1, std::memory_order_relaxed);

// Don't assume another thread will see them in order
```

## Best Practices

1. **Default to seq_cst**
   - Start safe
   - Optimize only when needed
   - Document when using weaker ordering

2. **Use Higher-Level Abstractions**
   - Mutexes for complex synchronization
   - Atomics for simple flags and counters
   - Only use weak ordering when performance critical

3. **Document Memory Ordering**
```cpp
// Release: Publishes data to readers
flag.store(true, std::memory_order_release);
```

4. **Test on Weak Architectures**
   - ARM, not just x86
   - Weak memory models expose bugs
   - Use simulators if necessary

5. **Verify with Tools**
   - Always use ThreadSanitizer
   - Test under stress
   - Verify on target platform

## Summary

Key concepts covered:
- C++ memory model fundamentals
- Happens-before relationships
- Six memory orderings and their uses
- Release-acquire synchronization
- Sequential consistency
- Memory fences
- Common patterns and pitfalls
- Architecture-specific considerations
- Performance optimization strategies

## Next Steps

In Chapter 6, we will explore advanced concurrent patterns including lock-free data structures, hazard pointers, and practical applications of the memory model.
