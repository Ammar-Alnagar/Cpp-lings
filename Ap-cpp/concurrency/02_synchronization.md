# Chapter 2: Synchronization Primitives

## Introduction to Synchronization

When multiple threads access shared data, we need synchronization mechanisms to prevent race conditions and ensure data consistency. C++ provides several synchronization primitives in the `<mutex>` header.

## Race Conditions

A race condition occurs when the program's behavior depends on the relative timing of threads. This leads to non-deterministic behavior and bugs that are hard to reproduce.

```cpp
// DANGEROUS: Race condition
int counter = 0;

void increment() {
    for (int i = 0; i < 1000; ++i) {
        ++counter;  // NOT ATOMIC! Read-modify-write
    }
}

// Running two threads will not give counter == 2000
// The increment is actually three operations:
// 1. Read counter
// 2. Add 1
// 3. Write back
// These can interleave between threads
```

## Mutexes (Mutual Exclusion)

A mutex is a synchronization primitive that provides exclusive access to shared resources.

### std::mutex

The basic mutex type. Only one thread can lock it at a time.

```cpp
#include <mutex>

std::mutex mtx;
int shared_data = 0;

void safe_increment() {
    mtx.lock();
    ++shared_data;
    mtx.unlock();
}
```

### Problems with Manual Lock/Unlock

1. **Forgetting to unlock**: Causes deadlock
2. **Exception thrown**: Mutex remains locked
3. **Multiple return paths**: Easy to miss unlock

Solution: Use RAII lock guards!

## Lock Guards

Lock guards provide automatic mutex management through RAII.

### std::lock_guard

Simple RAII wrapper. Locks on construction, unlocks on destruction.

```cpp
void safe_increment() {
    std::lock_guard<std::mutex> lock(mtx);
    ++shared_data;
    // Automatically unlocked when lock goes out of scope
}
```

### std::unique_lock

More flexible than lock_guard. Supports deferred locking, timed locking, and manual unlock.

```cpp
void flexible_locking() {
    std::unique_lock<std::mutex> lock(mtx, std::defer_lock);
    // Mutex not yet locked
    
    // Do some work without holding the lock
    
    lock.lock();  // Now lock it
    // Critical section
    lock.unlock();  // Can manually unlock
    
    // Do more work
    
    lock.lock();  // Can re-lock
    // Another critical section
}  // Automatically unlocks if still locked
```

### std::scoped_lock (C++17)

Locks multiple mutexes simultaneously, avoiding deadlock.

```cpp
std::mutex m1, m2;

void transfer(Account& from, Account& to, int amount) {
    // Locks both mutexes in a deadlock-free manner
    std::scoped_lock lock(from.mutex, to.mutex);
    from.balance -= amount;
    to.balance += amount;
}
```

## Deadlock

Deadlock occurs when two or more threads are waiting for each other to release resources.

### Classic Deadlock Example

```cpp
// Thread 1
lock(m1);
lock(m2);  // Waits if thread 2 has m2
// work
unlock(m2);
unlock(m1);

// Thread 2
lock(m2);
lock(m1);  // Waits if thread 1 has m1 - DEADLOCK!
// work
unlock(m1);
unlock(m2);
```

### Deadlock Prevention

1. **Lock ordering**: Always acquire locks in the same order
2. **std::lock / std::scoped_lock**: Atomic multi-lock
3. **std::try_lock**: Non-blocking attempt
4. **Lock hierarchies**: Enforce ordering at compile time

```cpp
// Solution 1: Consistent ordering
void safe_transfer() {
    std::mutex& first = (m1 < m2) ? m1 : m2;
    std::mutex& second = (m1 < m2) ? m2 : m1;
    std::lock_guard<std::mutex> lock1(first);
    std::lock_guard<std::mutex> lock2(second);
    // work
}

// Solution 2: Use std::scoped_lock (preferred)
void safe_transfer_modern() {
    std::scoped_lock lock(m1, m2);  // Deadlock-free
    // work
}
```

## Recursive Mutex

std::recursive_mutex allows the same thread to lock multiple times.

```cpp
std::recursive_mutex rmtx;

void recursive_function(int depth) {
    std::lock_guard<std::recursive_mutex> lock(rmtx);
    if (depth > 0) {
        recursive_function(depth - 1);  // Can re-lock
    }
}
```

Note: Recursive mutexes are slower and usually indicate design issues. Prefer refactoring.

## Timed Mutexes

Allow timeout on lock acquisition.

### std::timed_mutex

```cpp
std::timed_mutex tmtx;

bool try_work() {
    // Try to acquire lock for 100ms
    if (tmtx.try_lock_for(std::chrono::milliseconds(100))) {
        // Got the lock
        std::lock_guard<std::timed_mutex> lock(tmtx, std::adopt_lock);
        // work
        return true;
    }
    return false;  // Timeout
}
```

## Shared Mutex (Reader-Writer Lock)

std::shared_mutex (C++17) allows multiple readers or one writer.

```cpp
#include <shared_mutex>

std::shared_mutex sh_mtx;
std::vector<int> shared_data;

// Multiple readers can hold shared lock simultaneously
void read_data() {
    std::shared_lock<std::shared_mutex> lock(sh_mtx);
    // Read from shared_data
    // Multiple threads can read concurrently
}

// Writer needs exclusive access
void write_data(int value) {
    std::unique_lock<std::shared_mutex> lock(sh_mtx);
    // Write to shared_data
    // No other readers or writers allowed
    shared_data.push_back(value);
}
```

## Condition Variables

Condition variables enable threads to wait for a condition to become true.

### Basic Usage

```cpp
#include <condition_variable>

std::mutex mtx;
std::condition_variable cv;
bool ready = false;

// Waiting thread
void wait_for_signal() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, []{ return ready; });  // Wait until ready is true
    // Proceed when notified and ready is true
}

// Signaling thread
void send_signal() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        ready = true;
    }  // Release lock before notifying
    cv.notify_one();  // Wake up one waiting thread
}
```

### Why the Predicate?

Condition variables can have spurious wakeups. The predicate protects against:
1. Spurious wakeups (waking without notification)
2. Lost wakeups (notification before wait)
3. Stolen wakeups (another thread consumed the condition)

### notify_one vs notify_all

- `notify_one()`: Wakes one waiting thread (arbitrary which one)
- `notify_all()`: Wakes all waiting threads

```cpp
cv.notify_one();   // Wake one thread
cv.notify_all();   // Wake all threads
```

## Semaphores (C++20)

Semaphores are lightweight synchronization primitives for signaling.

### Counting Semaphore

```cpp
#include <semaphore>

std::counting_semaphore<10> sem(3);  // Initial count of 3

void worker() {
    sem.acquire();  // Decrement, wait if zero
    // Critical section (max 3 threads simultaneously)
    sem.release();  // Increment
}
```

### Binary Semaphore

```cpp
std::binary_semaphore sem(1);  // Like a simpler mutex

void work() {
    sem.acquire();
    // Critical section
    sem.release();
}
```

## Barriers (C++20)

Barriers provide a synchronization point for a group of threads.

```cpp
#include <barrier>

std::barrier sync_point(3);  // For 3 threads

void parallel_work(int id) {
    // Phase 1
    do_work_phase1(id);
    
    sync_point.arrive_and_wait();  // Wait for all threads
    
    // Phase 2 (all threads start together)
    do_work_phase2(id);
}
```

## Latches (C++20)

Latches are single-use countdown synchronization primitives.

```cpp
#include <latch>

std::latch done(3);  // Count of 3

void worker() {
    // Do work
    done.count_down();  // Decrement
}

void coordinator() {
    done.wait();  // Wait until count reaches zero
    // All workers done
}
```

## Best Practices

### 1. Minimize Critical Sections

```cpp
// BAD: Holds lock too long
void bad_example() {
    std::lock_guard<std::mutex> lock(mtx);
    expensive_computation();  // Doesn't need lock!
    shared_data++;
}

// GOOD: Minimal critical section
void good_example() {
    int result = expensive_computation();
    std::lock_guard<std::mutex> lock(mtx);
    shared_data += result;
}
```

### 2. Avoid Nested Locks

Nested locks increase deadlock risk. If necessary, use std::scoped_lock.

### 3. Use RAII Locks

Always use lock_guard, unique_lock, or scoped_lock. Never use raw lock/unlock.

### 4. Prefer Higher-Level Abstractions

Consider using thread-safe containers or message passing instead of manual locking.

### 5. Document Lock Ordering

If multiple locks are necessary, document the required order clearly.

## Performance Considerations

### Lock Contention

High contention (many threads waiting for the same lock) degrades performance:
- Reduce critical section size
- Use finer-grained locking
- Consider lock-free alternatives for hot paths

### Mutex Types Performance

From fastest to slowest:
1. Atomic operations (no mutex)
2. Spinlock (for very short critical sections)
3. std::mutex
4. std::recursive_mutex
5. std::shared_mutex (reader lock)
6. std::shared_mutex (writer lock)

### Cache Effects

Mutex operations cause cache synchronization. Minimize lock frequency on hot paths.

## Common Pitfalls

### 1. Forgetting to Lock

```cpp
// WRONG: No synchronization
void increment() {
    ++counter;  // Race condition!
}
```

### 2. Locking the Wrong Mutex

```cpp
std::mutex m1, m2;
int data1, data2;

void update() {
    std::lock_guard<std::mutex> lock(m1);
    data2++;  // WRONG: Should lock m2!
}
```

### 3. Holding Locks Too Long

```cpp
void bad() {
    std::lock_guard<std::mutex> lock(mtx);
    network_call();  // VERY BAD: I/O while holding lock
}
```

### 4. Returning References to Protected Data

```cpp
class Bad {
    std::mutex mtx;
    std::vector<int> data;
public:
    std::vector<int>& get_data() {
        std::lock_guard<std::mutex> lock(mtx);
        return data;  // WRONG: Lock released, reference still accessible
    }
};
```

## Summary

Key concepts covered:
- Race conditions and why synchronization is needed
- Mutexes: std::mutex, std::recursive_mutex, std::timed_mutex, std::shared_mutex
- Lock guards: std::lock_guard, std::unique_lock, std::scoped_lock
- Deadlock prevention strategies
- Condition variables for thread coordination
- Modern C++20 features: semaphores, barriers, latches
- Best practices and common pitfalls

## Next Steps

In the next chapter, we will explore asynchronous programming with std::async, std::future, and std::promise for task-based parallelism.
