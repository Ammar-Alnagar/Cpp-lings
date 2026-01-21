# Chapter 4: Atomic Operations and Lock-Free Programming

## Introduction to Atomics

Atomic operations are indivisible operations that complete without interruption. They provide lock-free synchronization for simple operations, offering better performance than mutexes for certain use cases.

## Why Atomic Operations?

Traditional synchronization with mutexes has overhead:
- Kernel calls for blocking
- Context switches
- Cache coherency traffic

Atomic operations provide:
- Lock-free synchronization
- No deadlock potential
- Better performance for simple operations
- Foundation for lock-free data structures

## std::atomic

The `<atomic>` header provides the std::atomic template for atomic operations.

### Basic Usage

```cpp
#include <atomic>

std::atomic<int> counter{0};

void increment() {
    counter++;  // Atomic increment
    // Equivalent to: counter.fetch_add(1);
}

int main() {
    std::thread t1(increment);
    std::thread t2(increment);
    t1.join();
    t2.join();
    // counter is guaranteed to be 2
}
```

### Supported Types

Atomic operations work with:
- Integral types: int, long, bool, char, etc.
- Pointer types
- Floating point types (C++20)
- User-defined types (if trivially copyable)

```cpp
std::atomic<int> atomic_int;
std::atomic<bool> atomic_bool;
std::atomic<float> atomic_float;  // C++20
std::atomic<MyStruct*> atomic_ptr;
```

## Atomic Operations

### Load and Store

```cpp
std::atomic<int> x{0};

// Store a value
x.store(42);
x = 42;  // Equivalent

// Load a value
int value = x.load();
value = x;  // Equivalent
```

### Read-Modify-Write Operations

```cpp
std::atomic<int> counter{0};

// Fetch and add (returns old value)
int old = counter.fetch_add(5);  // old = 0, counter = 5

// Fetch and subtract
old = counter.fetch_sub(2);  // old = 5, counter = 3

// Fetch and bitwise operations
counter.fetch_and(0xFF);
counter.fetch_or(0x01);
counter.fetch_xor(0x0F);

// Pre/post increment/decrement
counter++;  // Post-increment
++counter;  // Pre-increment
counter--;  // Post-decrement
--counter;  // Pre-decrement
```

### Exchange

```cpp
std::atomic<int> x{10};

// Exchange: set new value, return old value
int old = x.exchange(20);  // old = 10, x = 20
```

### Compare-And-Swap (CAS)

The most powerful atomic operation: compare_exchange_weak/strong.

```cpp
std::atomic<int> x{5};
int expected = 5;
int desired = 10;

// If x == expected, set x = desired and return true
// Otherwise, set expected = x and return false
bool success = x.compare_exchange_strong(expected, desired);

if (success) {
    // x is now 10
} else {
    // x was not 5, expected now contains actual value
}
```

### weak vs strong

- **compare_exchange_strong**: Never spurious failures
- **compare_exchange_weak**: May fail spuriously (even if values match)
  - Faster on some platforms
  - Use in loops where failure is expected

```cpp
// Typical pattern with weak
int expected = x.load();
int desired;
do {
    desired = compute_new_value(expected);
} while (!x.compare_exchange_weak(expected, desired));
```

## Memory Ordering

Memory ordering specifies how memory operations can be reordered around atomic operations.

### Memory Order Options

```cpp
namespace std {
    enum class memory_order {
        relaxed,    // No synchronization
        consume,    // Data dependency ordering (deprecated)
        acquire,    // Load-acquire
        release,    // Store-release
        acq_rel,    // Both acquire and release
        seq_cst     // Sequential consistency (default)
    };
}
```

### Sequential Consistency (Default)

Strongest guarantee: all operations appear in some global order.

```cpp
std::atomic<int> x{0}, y{0};

// Thread 1
x.store(1, std::memory_order_seq_cst);
int r1 = y.load(std::memory_order_seq_cst);

// Thread 2
y.store(1, std::memory_order_seq_cst);
int r2 = x.load(std::memory_order_seq_cst);

// Guaranteed: !(r1 == 0 && r2 == 0)
```

### Relaxed Ordering

No synchronization, only atomicity guaranteed.

```cpp
std::atomic<int> counter{0};

void increment() {
    counter.fetch_add(1, std::memory_order_relaxed);
}

// Counter will be correct, but no ordering guarantees
// with other operations
```

Use case: Counters where only final value matters.

### Acquire-Release Ordering

Synchronizes memory between threads.

```cpp
std::atomic<bool> ready{false};
int data = 0;

// Thread 1 (producer)
data = 42;
ready.store(true, std::memory_order_release);

// Thread 2 (consumer)
while (!ready.load(std::memory_order_acquire)) {
    // Wait
}
// Guaranteed: data == 42
```

- **Release**: All writes before this are visible after acquire
- **Acquire**: All reads after this see writes before release

### When to Use Each Ordering

- **seq_cst**: Default, safest, slight performance cost
- **relaxed**: Counters, statistics (no synchronization needed)
- **acquire/release**: Most lock-free algorithms (producer-consumer)
- **acq_rel**: Read-modify-write operations needing both

## Lock-Free Programming

### Lock-Free Stack

```cpp
template<typename T>
class LockFreeStack {
    struct Node {
        T data;
        Node* next;
        Node(const T& d) : data(d), next(nullptr) {}
    };
    
    std::atomic<Node*> head{nullptr};
    
public:
    void push(const T& value) {
        Node* new_node = new Node(value);
        new_node->next = head.load();
        
        // Keep trying until CAS succeeds
        while (!head.compare_exchange_weak(new_node->next, new_node)) {
            // new_node->next updated with current head on failure
        }
    }
    
    bool pop(T& value) {
        Node* old_head = head.load();
        
        while (old_head && 
               !head.compare_exchange_weak(old_head, old_head->next)) {
            // Retry with updated old_head
        }
        
        if (old_head) {
            value = old_head->data;
            delete old_head;  // Note: ABA problem!
            return true;
        }
        return false;
    }
};
```

### ABA Problem

The ABA problem occurs when:
1. Thread 1 reads A
2. Thread 2 changes A to B then back to A
3. Thread 1's CAS succeeds, but state has changed

Solution: Use tagged pointers or hazard pointers.

## Atomic Flags

std::atomic_flag is the most primitive atomic type, guaranteed lock-free.

```cpp
std::atomic_flag flag = ATOMIC_FLAG_INIT;

// Test and set
bool was_set = flag.test_and_set();

// Clear
flag.clear();
```

### Spinlock with atomic_flag

```cpp
class Spinlock {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
    
public:
    void lock() {
        while (flag.test_and_set(std::memory_order_acquire)) {
            // Spin
        }
    }
    
    void unlock() {
        flag.clear(std::memory_order_release);
    }
};
```

## Wait and Notify (C++20)

C++20 adds wait/notify operations to atomics.

```cpp
std::atomic<int> x{0};

// Thread 1: Wait for value to change
x.wait(0);  // Blocks until x != 0

// Thread 2: Update and notify
x.store(1);
x.notify_one();  // Wake one waiter
// or
x.notify_all();  // Wake all waiters
```

## Performance Considerations

### When to Use Atomics vs Mutexes

**Use atomics for:**
- Simple counters and flags
- Lock-free data structures
- High-contention scenarios with simple operations

**Use mutexes for:**
- Complex critical sections
- Multiple related operations
- When readability matters more than performance

### Memory Ordering Performance

From fastest to slowest:
1. relaxed (no synchronization)
2. acquire/release (one-way barrier)
3. seq_cst (full barrier)

### Cache Line Considerations

Avoid false sharing:

```cpp
// BAD: x and y on same cache line
struct {
    std::atomic<int> x;
    std::atomic<int> y;
} data;

// GOOD: Separate cache lines
struct alignas(64) {
    std::atomic<int> x;
} data1;

struct alignas(64) {
    std::atomic<int> y;
} data2;
```

## Common Patterns

### Double-Checked Locking

```cpp
std::atomic<bool> initialized{false};
std::mutex mtx;
Resource* resource = nullptr;

void init() {
    if (!initialized.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(mtx);
        if (!initialized.load(std::memory_order_relaxed)) {
            resource = new Resource();
            initialized.store(true, std::memory_order_release);
        }
    }
}
```

### Atomic Smart Pointers (C++20)

```cpp
std::atomic<std::shared_ptr<T>> ptr;

// Thread-safe operations
auto local = ptr.load();
ptr.store(new_ptr);
ptr.exchange(new_ptr);
```

## Best Practices

1. Start with mutexes, optimize to atomics if needed
2. Use seq_cst by default, optimize memory ordering carefully
3. Document memory ordering choices
4. Beware of the ABA problem in lock-free structures
5. Use std::atomic_ref for non-atomic objects needing atomic access
6. Align atomic variables to avoid false sharing
7. Test lock-free code extensively under high contention

## Summary

Key concepts covered:
- std::atomic for lock-free operations
- Atomic operations: load, store, exchange, CAS
- Memory ordering: relaxed, acquire/release, seq_cst
- Lock-free programming patterns
- The ABA problem and solutions
- Performance considerations
- When to use atomics vs mutexes

## Next Steps

In the next chapter, we will explore the C++ memory model in depth, understanding happens-before relationships, memory barriers, and the theoretical foundation of concurrent programming.
