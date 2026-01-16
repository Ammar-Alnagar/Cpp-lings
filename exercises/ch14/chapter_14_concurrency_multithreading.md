# Chapter 14: Concurrency and Multithreading

## Overview

This chapter covers concurrency and multithreading in C++, which are essential for creating efficient, responsive applications. You'll learn about threads, synchronization mechanisms, atomic operations, and concurrent programming patterns.

## Learning Objectives

By the end of this chapter, you will:
- Understand the basics of concurrency and multithreading
- Learn to create and manage threads using std::thread
- Master synchronization mechanisms (mutexes, locks, condition variables)
- Understand atomic operations and memory ordering
- Learn about thread-safe data structures
- Understand futures and promises for asynchronous programming
- Learn about thread pools and concurrent algorithms
- Understand the challenges of concurrent programming (race conditions, deadlocks)
- Learn best practices for concurrent programming

## Introduction to Threading

Threading allows multiple execution paths to run concurrently within a single program.

### Exercise 1: Basic Threading

The following code has errors. Find and fix them:

```cpp
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <functional>
using namespace std;

void workerFunction(int id, int duration) {
    cout << "Worker " << id << " starting..." << endl;
    this_thread::sleep_for(chrono::seconds(duration));
    cout << "Worker " << id << " finished after " << duration << " seconds" << endl;
}

void printMessage(const string& msg) {
    cout << msg << endl;
}

int main() {
    cout << "=== Basic Threading Demo ===" << endl;
    
    // Creating and joining threads
    thread t1(workerFunction, 1, 2);  // Worker 1, sleeps for 2 seconds
    thread t2(workerFunction, 2, 1);  // Worker 2, sleeps for 1 second
    
    // Error: not joining threads before main ends
    // t1.join();  // This should be called
    // t2.join();  // This should be called
    
    // Correct way:
    if (t1.joinable()) {
        t1.join();
    }
    if (t2.joinable()) {
        t2.join();
    }
    
    // Creating threads with different callable objects
    thread t3(printMessage, "Hello from thread 3!");
    thread t4([]() {  // Lambda function
        cout << "Hello from lambda thread!" << endl;
    });
    
    // Join the threads
    t3.join();
    t4.join();
    
    // Thread with member function
    class Task {
    public:
        void execute(int value) {
            cout << "Task executed with value: " << value << endl;
        }
    };
    
    Task taskObj;
    thread t5(&Task::execute, &taskObj, 42);  // Pass object and arguments
    t5.join();
    
    // Detaching threads (be careful!)
    thread t6([]() {
        this_thread::sleep_for(chrono::milliseconds(500));
        cout << "Detached thread running" << endl;
    });
    
    // Error: detaching without proper synchronization
    // t6.detach();  // This could cause the program to terminate before thread completes
    
    // Better approach: join before detaching
    if (t6.joinable()) {
        t6.join();
    }
    
    // Getting thread information
    thread currentThread = thread([]() {
        cout << "Current thread ID: " << this_thread::get_id() << endl;
    });
    
    cout << "Main thread ID: " << this_thread::get_id() << endl;
    cout << "Created thread ID: " << currentThread.get_id() << endl;
    
    currentThread.join();
    
    // Thread hardware concurrency
    cout << "Hardware threads available: " << thread::hardware_concurrency() << endl;
    
    return 0;
}
```

### Exercise 2: Thread Management and RAII

Complete this thread management example with RAII:

```cpp
#include <iostream>
#include <thread>
#include <vector>
#include <memory>
#include <mutex>
#include <functional>
using namespace std;

class ThreadRAII {
private:
    thread t;
    
public:
    template<typename Function, typename... Args>
    explicit ThreadRAII(Function&& f, Args&&... args) 
        : t(forward<Function>(f), forward<Args>(args)...) {}
    
    ~ThreadRAII() {
        if (t.joinable()) {
            t.join();  // Always join to avoid std::terminate
        }
    }
    
    // Disable copy semantics
    ThreadRAII(const ThreadRAII&) = delete;
    ThreadRAII& operator=(const ThreadRAII&) = delete;
    
    // Enable move semantics
    ThreadRAII(ThreadRAII&& other) noexcept : t(move(other.t)) {}
    ThreadRAII& operator=(ThreadRAII&& other) noexcept {
        if (this != &other) {
            if (t.joinable()) {
                t.join();
            }
            t = move(other.t);
        }
        return *this;
    }
    
    thread::id get_id() const { return t.get_id(); }
    bool joinable() const { return t.joinable(); }
    void join() { t.join(); }
    void detach() { t.detach(); }
};

void workerTask(int id) {
    cout << "Worker task " << id << " running on thread " << this_thread::get_id() << endl;
    this_thread::sleep_for(chrono::milliseconds(100 * id));
    cout << "Worker task " << id << " completed" << endl;
}

int main() {
    cout << "=== Thread Management with RAII ===" << endl;
    
    // Using RAII wrapper for thread management
    vector<ThreadRAII> threads;
    
    for (int i = 1; i <= 5; i++) {
        threads.emplace_back(workerTask, i);
    }
    
    cout << "All threads created, about to join them..." << endl;
    
    // Threads will be automatically joined when they go out of scope
    // But we'll explicitly join them here for demonstration
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    cout << "All threads completed" << endl;
    
    // Exception safety with RAII threads
    cout << "\n=== Exception Safety Demo ===" << endl;
    
    try {
        vector<ThreadRAII> exceptionThreads;
        
        for (int i = 1; i <= 3; i++) {
            exceptionThreads.emplace_back([](int id) {
                this_thread::sleep_for(chrono::milliseconds(50 * id));
                cout << "Exception thread " << id << " running" << endl;
            }, i);
        }
        
        cout << "About to throw exception..." << endl;
        throw runtime_error("Simulated exception");
        
        // Even if exception is thrown, threads will be properly joined
    } catch (const exception& e) {
        cout << "Caught exception: " << e.what() << endl;
        cout << "But threads were properly cleaned up!" << endl;
    }
    
    // Thread with return value (using packaged_task and future)
    cout << "\n=== Thread with Return Value ===" << endl;
    
    packaged_task<int()> task([]() {
        cout << "Computing result in thread..." << endl;
        this_thread::sleep_for(chrono::milliseconds(200));
        return 42;
    });
    
    future<int> resultFuture = task.get_future();
    thread returnThread(move(task));
    
    cout << "Waiting for result..." << endl;
    int result = resultFuture.get();  // Blocks until result is available
    cout << "Result from thread: " << result << endl;
    
    returnThread.join();
    
    return 0;
}
```

## Synchronization Primitives

Synchronization is crucial for preventing race conditions in multithreaded programs.

### Exercise 3: Mutex and Locks

Complete this synchronization example:

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>
#include <atomic>
using namespace std;

class Counter {
private:
    int value;
    mutex mtx;  // Mutex for synchronization
    
public:
    Counter() : value(0) {}
    
    void increment() {
        lock_guard<mutex> lock(mtx);  // RAII lock
        value++;  // Critical section
    }
    
    int getValue() const {
        lock_guard<mutex> lock(mtx);  // RAII lock
        return value;  // Critical section
    }
    
    void incrementUnprotected() {
        value++;  // Race condition - unprotected access!
    }
};

// Shared resource with different synchronization methods
class SharedResource {
private:
    int data;
    mutable mutex dataMutex;
    mutable shared_mutex readWriteMutex;  // For reader-writer locking
    
public:
    SharedResource(int initial = 0) : data(initial) {}
    
    // Exclusive write access
    void writeData(int newValue) {
        lock_guard<mutex> lock(dataMutex);
        cout << "Writing " << newValue << " from thread " << this_thread::get_id() << endl;
        this_thread::sleep_for(chrono::milliseconds(100));  // Simulate work
        data = newValue;
    }
    
    // Exclusive read access
    int readDataExclusive() const {
        lock_guard<mutex> lock(dataMutex);
        cout << "Reading " << data << " from thread " << this_thread::get_id() << endl;
        this_thread::sleep_for(chrono::milliseconds(50));  // Simulate work
        return data;
    }
    
    // Shared read access (multiple readers allowed)
    int readDataShared() const {
        shared_lock<shared_mutex> lock(readWriteMutex);
        cout << "Shared reading " << data << " from thread " << this_thread::get_id() << endl;
        this_thread::sleep_for(chrono::milliseconds(50));  // Simulate work
        return data;
    }
    
    // Exclusive write access with shared_mutex
    void writeDataShared(int newValue) {
        unique_lock<shared_mutex> lock(readWriteMutex);
        cout << "Shared writing " << newValue << " from thread " << this_thread::get_id() << endl;
        this_thread::sleep_for(chrono::milliseconds(100));  // Simulate work
        data = newValue;
    }
};

void workerWithCounter(Counter& counter, int iterations) {
    for (int i = 0; i < iterations; i++) {
        counter.increment();
        this_thread::sleep_for(chrono::microseconds(10));  // Small delay
    }
}

void workerWithResource(SharedResource& resource, int id, bool isWriter) {
    if (isWriter) {
        resource.writeDataShared(id * 10);
    } else {
        resource.readDataShared();
    }
}

int main() {
    cout << "=== Synchronization Primitives Demo ===" << endl;
    
    // Race condition demonstration
    cout << "\n--- Race Condition Demo ---" << endl;
    Counter unprotectedCounter;
    Counter protectedCounter;
    
    const int numThreads = 10;
    const int iterationsPerThread = 1000;
    
    // Threads with unprotected counter (will have race condition)
    vector<thread> unprotectedThreads;
    for (int i = 0; i < numThreads; i++) {
        unprotectedThreads.emplace_back([&unprotectedCounter, iterationsPerThread]() {
            for (int j = 0; j < iterationsPerThread; j++) {
                unprotectedCounter.incrementUnprotected();
            }
        });
    }
    
    // Threads with protected counter (safe)
    vector<thread> protectedThreads;
    for (int i = 0; i < numThreads; i++) {
        protectedThreads.emplace_back([&protectedCounter, iterationsPerThread]() {
            for (int j = 0; j < iterationsPerThread; j++) {
                protectedCounter.increment();
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& t : unprotectedThreads) {
        t.join();
    }
    for (auto& t : protectedThreads) {
        t.join();
    }
    
    cout << "Expected result: " << numThreads * iterationsPerThread << endl;
    cout << "Unprotected counter result: " << unprotectedCounter.getValue() << endl;
    cout << "Protected counter result: " << protectedCounter.getValue() << endl;
    
    // Reader-writer lock demonstration
    cout << "\n--- Reader-Writer Lock Demo ---" << endl;
    
    SharedResource resource(0);
    vector<thread> rwThreads;
    
    // Create mixed readers and writers
    for (int i = 0; i < 6; i++) {
        if (i % 3 == 0) {  // Every third thread is a writer
            rwThreads.emplace_back([&resource, i]() {
                resource.writeDataShared((i + 1) * 100);
            });
        } else {  // Readers
            rwThreads.emplace_back([&resource, i]() {
                resource.readDataShared();
            });
        }
    }
    
    // Wait for all reader/writer threads
    for (auto& t : rwThreads) {
        t.join();
    }
    
    // Deadlock prevention with lock_guard
    cout << "\n--- Deadlock Prevention Demo ---" << endl;
    
    mutex m1, m2;
    
    thread deadlockAvoider1([&m1, &m2]() {
        lock(m1, m2);  // Acquire both locks in consistent order
        lock_guard<mutex> lock1(m1, adopt_lock);
        lock_guard<mutex> lock2(m2, adopt_lock);
        
        cout << "Thread 1: Got both locks" << endl;
        this_thread::sleep_for(chrono::milliseconds(100));
    });
    
    thread deadlockAvoider2([&m1, &m2]() {
        lock(m1, m2);  // Same order as thread 1 - prevents deadlock
        lock_guard<mutex> lock1(m1, adopt_lock);
        lock_guard<mutex> lock2(m2, adopt_lock);
        
        cout << "Thread 2: Got both locks" << endl;
        this_thread::sleep_for(chrono::milliseconds(100));
    });
    
    deadlockAvoider1.join();
    deadlockAvoider2.join();
    
    // Timed locks
    cout << "\n--- Timed Locks Demo ---" << endl;
    
    mutex timedMtx;
    timedMtx.lock();
    
    thread timedThread([&timedMtx]() {
        cout << "Timed thread trying to acquire lock..." << endl;
        
        unique_lock<mutex> lock(timedMtx, defer_lock);
        if (lock.try_lock_for(chrono::milliseconds(500))) {
            cout << "Lock acquired successfully" << endl;
        } else {
            cout << "Failed to acquire lock within timeout" << endl;
        }
    });
    
    this_thread::sleep_for(chrono::milliseconds(200));
    timedMtx.unlock();  // Release lock so other thread can proceed
    
    timedThread.join();
    
    return 0;
}
```

## Atomic Operations

Atomic operations provide lock-free synchronization for simple operations.

### Exercise 4: Atomic Operations

Complete this atomic operations example:

```cpp
#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
using namespace std;

class AtomicCounter {
private:
    atomic<int> value;
    
public:
    AtomicCounter(int initial = 0) : value(initial) {}
    
    void increment() {
        value.fetch_add(1, memory_order_relaxed);  // Atomic increment
    }
    
    int getValue() const {
        return value.load(memory_order_acquire);  // Atomic load with acquire semantics
    }
    
    int incrementAndGet() {
        return value.fetch_add(1, memory_order_acq_rel) + 1;  // Returns old value + 1
    }
    
    bool compareAndSwap(int expected, int desired) {
        return value.compare_exchange_strong(expected, desired, 
                                           memory_order_acq_rel, 
                                           memory_order_acquire);
    }
};

// Atomic flag for simple signaling
atomic_flag readyFlag = ATOMIC_FLAG_INIT;

void workerWithAtomic(AtomicCounter& counter, int iterations) {
    for (int i = 0; i < iterations; i++) {
        counter.increment();
        this_thread::sleep_for(chrono::microseconds(10));
    }
}

void signaler() {
    this_thread::sleep_for(chrono::milliseconds(500));
    cout << "Signaler: Setting ready flag" << endl;
    readyFlag.clear();  // Clear the flag (signal ready)
}

void waiter() {
    cout << "Waiter: Waiting for signal..." << endl;
    while (readyFlag.test_and_set(memory_order_acquire)) {
        this_thread::yield();  // Yield to other threads
    }
    cout << "Waiter: Signal received!" << endl;
}

int main() {
    cout << "=== Atomic Operations Demo ===" << endl;
    
    // Atomic counter demonstration
    AtomicCounter atomicCounter(0);
    const int numThreads = 8;
    const int iterationsPerThread = 1000;
    
    vector<thread> atomicThreads;
    auto startTime = chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numThreads; i++) {
        atomicThreads.emplace_back([&atomicCounter, iterationsPerThread]() {
            workerWithAtomic(atomicCounter, iterationsPerThread);
        });
    }
    
    for (auto& t : atomicThreads) {
        t.join();
    }
    
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
    
    cout << "Atomic counter result: " << atomicCounter.getValue() << endl;
    cout << "Expected: " << numThreads * iterationsPerThread << endl;
    cout << "Time taken: " << duration.count() << " ms" << endl;
    
    // Compare with regular counter and mutex
    Counter regularCounter;
    vector<thread> regularThreads;
    startTime = chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numThreads; i++) {
        regularThreads.emplace_back([&regularCounter, iterationsPerThread]() {
            for (int j = 0; j < iterationsPerThread; j++) {
                regularCounter.increment();
                this_thread::sleep_for(chrono::microseconds(10));
            }
        });
    }
    
    for (auto& t : regularThreads) {
        t.join();
    }
    
    endTime = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
    
    cout << "Regular counter result: " << regularCounter.getValue() << endl;
    cout << "Time taken with mutex: " << duration.count() << " ms" << endl;
    
    // Atomic flag demonstration
    cout << "\n--- Atomic Flag Demo ---" << endl;
    
    thread waitThread(waiter);
    thread signalThread(signaler);
    
    waitThread.join();
    signalThread.join();
    
    // Memory ordering demonstration
    cout << "\n--- Memory Ordering Demo ---" << endl;
    
    atomic<bool> ready(false);
    atomic<int> data(0);
    
    thread producer([&ready, &data]() {
        data.store(42, memory_order_relaxed);  // Store data first
        ready.store(true, memory_order_release);  // Signal that data is ready
    });
    
    thread consumer([&ready, &data]() {
        while (!ready.load(memory_order_acquire)) {  // Wait for ready signal
            this_thread::yield();
        }
        int result = data.load(memory_order_relaxed);  // Load data after ready
        cout << "Consumer got data: " << result << endl;
    });
    
    producer.join();
    consumer.join();
    
    // Atomic operations performance comparison
    cout << "\n--- Atomic Operations Performance ---" << endl;
    
    atomic<int> atomicOps(0);
    int regularOps = 0;
    mutex opsMutex;
    
    // Time atomic operations
    auto atomicStart = chrono::high_resolution_clock::now();
    vector<thread> atomicOpThreads;
    
    for (int i = 0; i < 4; i++) {
        atomicOpThreads.emplace_back([&atomicOps]() {
            for (int j = 0; j < 100000; j++) {
                atomicOps.fetch_add(1);
            }
        });
    }
    
    for (auto& t : atomicOpThreads) {
        t.join();
    }
    auto atomicEnd = chrono::high_resolution_clock::now();
    
    // Time regular operations with mutex
    auto regularStart = chrono::high_resolution_clock::now();
    vector<thread> regularOpThreads;
    
    for (int i = 0; i < 4; i++) {
        regularOpThreads.emplace_back([&regularOps, &opsMutex]() {
            for (int j = 0; j < 100000; j++) {
                lock_guard<mutex> lock(opsMutex);
                regularOps++;
            }
        });
    }
    
    for (auto& t : regularOpThreads) {
        t.join();
    }
    auto regularEnd = chrono::high_resolution_clock::now();
    
    cout << "Atomic operations result: " << atomicOps.load() << endl;
    cout << "Regular operations result: " << regularOps << endl;
    cout << "Atomic time: " << 
         chrono::duration_cast<chrono::milliseconds>(atomicEnd - atomicStart).count() << " ms" << endl;
    cout << "Regular time: " << 
         chrono::duration_cast<chrono::milliseconds>(regularEnd - regularStart).count() << " ms" << endl;
    
    // Atomic compare-and-swap
    cout << "\n--- Compare-and-Swap Demo ---" << endl;
    
    AtomicCounter casCounter(10);
    cout << "Initial value: " << casCounter.getValue() << endl;
    
    int expected = 10;
    bool success = casCounter.compareAndSwap(expected, 20);
    cout << "CAS from 10 to 20: " << (success ? "Success" : "Failed") << endl;
    cout << "Value after CAS: " << casCounter.getValue() << endl;
    
    expected = 10;  // Wrong expected value
    success = casCounter.compareAndSwap(expected, 30);
    cout << "CAS from 10 to 30 (should fail): " << (success ? "Success" : "Failed") << endl;
    cout << "Value after failed CAS: " << casCounter.getValue() << endl;
    
    return 0;
}
```

## Condition Variables

Condition variables allow threads to wait for specific conditions.

### Exercise 5: Condition Variables

Complete this condition variable example:

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <random>
using namespace std;

// Producer-consumer with condition variables
template<typename T>
class ThreadSafeQueue {
private:
    queue<T> dataQueue;
    mutable mutex mtx;
    condition_variable dataCondition;
    
public:
    void push(T item) {
        {
            lock_guard<mutex> lock(mtx);
            dataQueue.push(move(item));
        }
        dataCondition.notify_one();  // Notify waiting consumers
    }
    
    T pop() {
        unique_lock<mutex> lock(mtx);
        // Wait until queue is not empty
        dataCondition.wait(lock, [this] { return !dataQueue.empty(); });
        
        T item = move(dataQueue.front());
        dataQueue.pop();
        return item;
    }
    
    bool tryPop(T& item) {
        lock_guard<mutex> lock(mtx);
        if (dataQueue.empty()) {
            return false;
        }
        item = move(dataQueue.front());
        dataQueue.pop();
        return true;
    }
    
    bool empty() const {
        lock_guard<mutex> lock(mtx);
        return dataQueue.empty();
    }
    
    size_t size() const {
        lock_guard<mutex> lock(mtx);
        return dataQueue.size();
    }
};

void producer(ThreadSafeQueue<int>& queue, int id, int itemCount) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(100, 999);
    
    for (int i = 0; i < itemCount; i++) {
        int value = dis(gen);
        cout << "Producer " << id << " producing: " << value << endl;
        queue.push(value);
        this_thread::sleep_for(chrono::milliseconds(dis(gen) % 200));  // Random delay
    }
    cout << "Producer " << id << " finished" << endl;
}

void consumer(ThreadSafeQueue<int>& queue, int id, int maxItems) {
    int consumed = 0;
    while (consumed < maxItems) {
        try {
            int value = queue.pop();  // This will block if queue is empty
            cout << "Consumer " << id << " consuming: " << value << endl;
            consumed++;
            this_thread::sleep_for(chrono::milliseconds(150));  // Processing time
        } catch (const exception& e) {
            cout << "Consumer " << id << " exception: " << e.what() << endl;
            break;
        }
    }
    cout << "Consumer " << id << " finished" << endl;
}

// Barrier implementation using condition variables
class Barrier {
private:
    mutable mutex mtx;
    condition_variable cv;
    size_t count;
    size_t total;
    
public:
    explicit Barrier(size_t n) : count(0), total(n) {}
    
    void wait() {
        unique_lock<mutex> lock(mtx);
        count++;
        
        if (count == total) {
            count = 0;  // Reset for next barrier
            cv.notify_all();  // Wake up all waiting threads
        } else {
            cv.wait(lock, [this] { return count == 0; });  // Wait for barrier
        }
    }
};

void barrierWorker(int id, Barrier& barrier) {
    cout << "Worker " << id << " doing some work..." << endl;
    this_thread::sleep_for(chrono::milliseconds(100 * (id + 1)));  // Different work times
    
    cout << "Worker " << id << " waiting at barrier..." << endl;
    barrier.wait();  // Wait for all workers to reach this point
    
    cout << "Worker " << id << " passed barrier!" << endl;
}

int main() {
    cout << "=== Condition Variables Demo ===" << endl;
    
    // Producer-consumer example
    ThreadSafeQueue<int> sharedQueue;
    
    cout << "\n--- Producer-Consumer Demo ---" << endl;
    
    vector<thread> producerThreads;
    vector<thread> consumerThreads;
    
    // Create producers
    for (int i = 0; i < 2; i++) {
        producerThreads.emplace_back(producer, ref(sharedQueue), i, 5);
    }
    
    // Create consumers
    for (int i = 0; i < 3; i++) {
        consumerThreads.emplace_back(consumer, ref(sharedQueue), i, 3);
    }
    
    // Wait for all threads to complete
    for (auto& t : producerThreads) {
        t.join();
    }
    for (auto& t : consumerThreads) {
        t.join();
    }
    
    cout << "Final queue size: " << sharedQueue.size() << endl;
    
    // Barrier example
    cout << "\n--- Barrier Demo ---" << endl;
    
    Barrier barrier(4);  // 4 workers
    
    vector<thread> barrierWorkers;
    for (int i = 0; i < 4; i++) {
        barrierWorkers.emplace_back(barrierWorker, i, ref(barrier));
    }
    
    for (auto& t : barrierWorkers) {
        t.join();
    }
    
    // Event signaling with condition variables
    cout << "\n--- Event Signaling Demo ---" << endl;
    
    mutex eventMtx;
    condition_variable eventCV;
    bool eventTriggered = false;
    
    thread eventWaiter([&eventMtx, &eventCV, &eventTriggered]() {
        unique_lock<mutex> lock(eventMtx);
        cout << "Event waiter: Waiting for signal..." << endl;
        eventCV.wait(lock, [&eventTriggered] { return eventTriggered; });
        cout << "Event waiter: Event triggered!" << endl;
    });
    
    thread eventTrigger([&eventMtx, &eventCV, &eventTriggered]() {
        this_thread::sleep_for(chrono::milliseconds(500));
        {
            lock_guard<mutex> lock(eventMtx);
            eventTriggered = true;
            cout << "Event trigger: Signaling event..." << endl;
        }
        eventCV.notify_all();  // Notify waiting threads
    });
    
    eventWaiter.join();
    eventTrigger.join();
    
    // Timeout operations
    cout << "\n--- Timeout Operations Demo ---" << endl;
    
    mutex timeoutMtx;
    condition_variable timeoutCV;
    bool timeoutReady = false;
    
    thread timeoutWaiter([&timeoutMtx, &timeoutCV, &timeoutReady]() {
        unique_lock<mutex> lock(timeoutMtx);
        cout << "Timeout waiter: Waiting with timeout..." << endl;
        
        if (timeoutCV.wait_for(lock, chrono::milliseconds(300), 
                              [&timeoutReady] { return timeoutReady; })) {
            cout << "Timeout waiter: Condition met!" << endl;
        } else {
            cout << "Timeout waiter: Timeout occurred!" << endl;
        }
    });
    
    timeoutWaiter.join();
    
    // Another timeout example with absolute time
    thread absoluteTimeout([&timeoutMtx, &timeoutCV, &timeoutReady]() {
        unique_lock<mutex> lock(timeoutMtx);
        auto timeoutPoint = chrono::steady_clock::now() + chrono::milliseconds(1000);
        
        cout << "Absolute timeout: Waiting until specific time..." << endl;
        if (timeoutCV.wait_until(lock, timeoutPoint, 
                                [&timeoutReady] { return timeoutReady; })) {
            cout << "Absolute timeout: Condition met!" << endl;
        } else {
            cout << "Absolute timeout: Time point reached!" << endl;
        }
    });
    
    absoluteTimeout.join();
    
    cout << "\nCondition variables demonstration completed!" << endl;
    
    return 0;
}
```

## Futures and Promises

Futures and promises provide a way to work with asynchronous operations.

### Exercise 6: Futures and Promises

Complete this futures and promises example:

```cpp
#include <iostream>
#include <future>
#include <thread>
#include <chrono>
#include <vector>
#include <algorithm>
using namespace std;

// Function that simulates a long-running task
int intensiveComputation(int value) {
    cout << "Starting computation for " << value << " in thread " << this_thread::get_id() << endl;
    this_thread::sleep_for(chrono::milliseconds(1000));  // Simulate work
    int result = value * value;
    cout << "Computed " << value << "^2 = " << result << endl;
    return result;
}

// Function that might throw
int potentiallyThrowingFunction(int value) {
    if (value < 0) {
        throw runtime_error("Negative value not allowed!");
    }
    return value * 2;
}

int main() {
    cout << "=== Futures and Promises Demo ===" << endl;
    
    // async with future
    cout << "\n--- Async and Future Demo ---" << endl;
    
    future<int> future1 = async(launch::async, intensiveComputation, 5);
    future<int> future2 = async(launch::async, intensiveComputation, 7);
    
    cout << "Main thread continuing..." << endl;
    
    // Get results (blocks until computation is complete)
    int result1 = future1.get();
    int result2 = future2.get();
    
    cout << "Results: " << result1 << ", " << result2 << endl;
    
    // Using packaged_task
    cout << "\n--- Packaged Task Demo ---" << endl;
    
    packaged_task<int(int)> task(intensiveComputation);
    future<int> taskFuture = task.get_future();
    
    // Run task in a separate thread
    thread taskThread(move(task), 10);
    
    cout << "Waiting for packaged task result..." << endl;
    int taskResult = taskFuture.get();
    cout << "Packaged task result: " << taskResult << endl;
    
    taskThread.join();
    
    // Promise and future for custom synchronization
    cout << "\n--- Promise and Future Demo ---" << endl;
    
    promise<int> promiseObj;
    future<int> promiseFuture = promiseObj.get_future();
    
    thread promiseThread([&promiseObj]() {
        this_thread::sleep_for(chrono::milliseconds(500));
        try {
            int result = intensiveComputation(15);
            promiseObj.set_value(result);  // Set the result
        } catch (...) {
            promiseObj.set_exception(current_exception());  // Set exception
        }
    });
    
    cout << "Getting promise result..." << endl;
    int promiseResult = promiseFuture.get();
    cout << "Promise result: " << promiseResult << endl;
    
    promiseThread.join();
    
    // Exception handling with futures
    cout << "\n--- Exception Handling Demo ---" << endl;
    
    future<int> exceptionFuture = async(launch::async, potentiallyThrowingFunction, -5);
    
    try {
        int exceptionResult = exceptionFuture.get();  // This will re-throw the exception
        cout << "This won't be printed: " << exceptionResult << endl;
    } catch (const runtime_error& e) {
        cout << "Caught exception from future: " << e.what() << endl;
    }
    
    // Shared future (can be copied)
    cout << "\n--- Shared Future Demo ---" << endl;
    
    packaged_task<int()> sharedTask([]() {
        this_thread::sleep_for(chrono::milliseconds(300));
        cout << "Shared task running in thread " << this_thread::get_id() << endl;
        return 42;
    });
    
    future<int> sharedFuture = sharedTask.get_future();
    shared_future<int> sharedResult = sharedFuture.share();  // Convert to shared_future
    
    thread sharedThread(move(sharedTask));
    
    // Multiple threads can wait on the same shared future
    vector<thread> waitingThreads;
    for (int i = 0; i < 3; i++) {
        waitingThreads.emplace_back([sharedResult]() mutable {  // mutable to allow copying
            int result = sharedResult.get();
            cout << "Thread " << this_thread::get_id() << " got result: " << result << endl;
        });
    }
    
    sharedThread.join();
    for (auto& t : waitingThreads) {
        t.join();
    }
    
    // Waiting for multiple futures
    cout << "\n--- Multiple Futures Demo ---" << endl;
    
    vector<future<int>> futures;
    for (int i = 1; i <= 5; i++) {
        futures.push_back(async(launch::async, intensiveComputation, i));
    }
    
    cout << "Waiting for all futures..." << endl;
    vector<int> results;
    for (auto& f : futures) {
        results.push_back(f.get());
    }
    
    cout << "All results: ";
    for (int result : results) {
        cout << result << " ";
    }
    cout << endl;
    
    // Timeout with futures
    cout << "\n--- Future Timeout Demo ---" << endl;
    
    packaged_task<int()> timeoutTask([]() {
        this_thread::sleep_for(chrono::milliseconds(2000));  // Long operation
        return 999;
    });
    
    future<int> timeoutFuture = timeoutTask.get_future();
    thread timeoutThread(move(timeoutTask));
    
    // Wait with timeout
    auto status = timeoutFuture.wait_for(chrono::milliseconds(500));
    if (status == future_status::timeout) {
        cout << "Future timed out!" << endl;
    } else if (status == future_status::ready) {
        cout << "Future is ready: " << timeoutFuture.get() << endl;
    }
    
    // Wait for the long operation to complete
    timeoutFuture.wait();
    cout << "Long operation finally completed: " << timeoutFuture.get() << endl;
    
    timeoutThread.join();
    
    // When to use different launch policies
    cout << "\n--- Launch Policy Demo ---" << endl;
    
    // launch::async - guaranteed to run asynchronously
    auto asyncFuture = async(launch::async, []() {
        cout << "Async policy running in thread " << this_thread::get_id() << endl;
        this_thread::sleep_for(chrono::milliseconds(100));
        return 1;
    });
    
    // launch::deferred - run lazily when get() is called
    auto deferredFuture = async(launch::deferred, []() {
        cout << "Deferred policy running in thread " << this_thread::get_id() << endl;
        return 2;
    });
    
    // launch::async | launch::deferred - implementation chooses
    auto chooseFuture = async([]() {
        cout << "Choose policy running in thread " << this_thread::get_id() << endl;
        return 3;
    });
    
    cout << "Before getting deferred result" << endl;
    int deferredResult = deferredFuture.get();  // This actually runs the function
    cout << "After getting deferred result: " << deferredResult << endl;
    
    cout << "Results - Async: " << asyncFuture.get() 
         << ", Deferred: " << deferredResult 
         << ", Choose: " << chooseFuture.get() << endl;
    
    cout << "\nFutures and promises demonstration completed!" << endl;
    
    return 0;
}
```

## Thread Pools

Thread pools help manage and reuse threads efficiently.

### Exercise 7: Thread Pool Implementation

Complete this thread pool example:

```cpp
#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>
#include <memory>
using namespace std;

class ThreadPool {
private:
    vector<thread> workers;
    queue<function<void()>> tasks;
    
    mutex queueMutex;
    condition_variable condition;
    bool stop;
    
public:
    explicit ThreadPool(size_t numThreads) : stop(false) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    function<void()> task;
                    
                    {
                        unique_lock<mutex> lock(queueMutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        
                        if (stop && tasks.empty()) {
                            return;
                        }
                        
                        task = move(tasks.front());
                        tasks.pop();
                    }
                    
                    task();
                }
            });
        }
    }
    
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> future<typename result_of<F(Args...)>::type> {
        using return_type = typename result_of<F(Args...)>::type;
        
        auto task = make_shared<packaged_task<return_type()>>(
            bind(forward<F>(f), forward<Args>(args)...)
        );
        
        future<return_type> result = task->get_future();
        
        {
            unique_lock<mutex> lock(queueMutex);
            
            // Don't allow enqueueing after stopping the pool
            if (stop) {
                throw runtime_error("enqueue on stopped ThreadPool");
            }
            
            tasks.emplace([task]() { (*task)(); });
        }
        
        condition.notify_one();
        return result;
    }
    
    ~ThreadPool() {
        {
            unique_lock<mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (thread& worker : workers) {
            worker.join();
        }
    }
};

int main() {
    cout << "=== Thread Pool Demo ===" << endl;
    
    // Create a thread pool with 4 threads
    ThreadPool pool(4);
    
    vector<future<int>> results;
    
    // Submit tasks to the thread pool
    for (int i = 0; i < 8; i++) {
        results.emplace_back(pool.enqueue([i] {
            cout << "Task " << i << " running on thread " << this_thread::get_id() << endl;
            this_thread::sleep_for(chrono::milliseconds(500));
            return i * i;
        }));
    }
    
    cout << "All tasks submitted, waiting for results..." << endl;
    
    // Get all results
    for (auto& result : results) {
        cout << "Result: " << result.get() << endl;
    }
    
    cout << "All tasks completed!" << endl;
    
    // Demonstrate different types of tasks
    cout << "\n--- Different Task Types ---" << endl;
    
    // Void function
    auto voidFuture = pool.enqueue([]() {
        cout << "Void task executed" << endl;
        this_thread::sleep_for(chrono::milliseconds(200));
    });
    
    // Function with parameters
    auto paramFuture = pool.enqueue([](int a, int b) {
        cout << "Parameter task: " << a << " + " << b << " = " << (a + b) << endl;
        return a + b;
    }, 10, 20);
    
    // Function that returns complex type
    auto complexFuture = pool.enqueue([]() {
        vector<int> result = {1, 2, 3, 4, 5};
        cout << "Complex task returning vector of size: " << result.size() << endl;
        return result;
    });
    
    // Wait for void task
    voidFuture.wait();
    cout << "Void task completed" << endl;
    
    // Get result from parameter task
    int paramResult = paramFuture.get();
    cout << "Parameter task result: " << paramResult << endl;
    
    // Get result from complex task
    auto complexResult = complexFuture.get();
    cout << "Complex task result: ";
    for (int val : complexResult) {
        cout << val << " ";
    }
    cout << endl;
    
    // Exception handling in thread pool
    cout << "\n--- Exception Handling ---" << endl;
    
    auto exceptionFuture = pool.enqueue([]() -> int {
        throw runtime_error("Task exception occurred!");
        return 42;
    });
    
    try {
        int exceptionResult = exceptionFuture.get();
        cout << "This won't print: " << exceptionResult << endl;
    } catch (const runtime_error& e) {
        cout << "Caught exception from thread pool task: " << e.what() << endl;
    }
    
    cout << "\nThread pool demonstration completed!" << endl;
    
    return 0;
}
```

## Concurrent Algorithms (C++17)

C++17 introduced execution policies for STL algorithms.

### Exercise 8: Concurrent Algorithms

Complete this concurrent algorithms example:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <execution>
#include <numeric>
#include <chrono>
#include <random>
using namespace std;
using namespace std::execution;

// Function to measure execution time
template<typename Func>
auto measureTime(Func&& func) -> decltype(func()) {
    auto start = chrono::high_resolution_clock::now();
    auto result = func();
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count() << " ms" << endl;
    return result;
}

int main() {
    cout << "=== Concurrent Algorithms Demo ===" << endl;
    
    const size_t dataSize = 10000000;  // 10 million elements
    vector<int> data(dataSize);
    
    // Initialize with random values
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 1000);
    
    for (auto& val : data) {
        val = dis(gen);
    }
    
    cout << "Data size: " << dataSize << " elements" << endl;
    
    // Sequential execution
    cout << "\n--- Sequential Execution ---" << endl;
    auto sequentialResult = measureTime([&data]() {
        vector<int> temp = data;
        sort(temp.begin(), temp.end());
        return temp;
    });
    
    // Parallel execution
    cout << "\n--- Parallel Execution ---" << endl;
    auto parallelResult = measureTime([&data]() {
        vector<int> temp = data;
        sort(par(exec), temp.begin(), temp.end());
        return temp;
    });
    
    // Parallel + vectorized execution
    cout << "\n--- Parallel + Vectorized Execution ---" << endl;
    auto parVecResult = measureTime([&data]() {
        vector<int> temp = data;
        sort(par_unseq, temp.begin(), temp.end());
        return temp;
    });
    
    // Verify results are the same
    bool sequentialEqualsParallel = equal(sequentialResult.begin(), sequentialResult.end(), 
                                         parallelResult.begin());
    bool parallelEqualsParVec = equal(parallelResult.begin(), parallelResult.end(), 
                                     parVecResult.begin());
    
    cout << "\nResults verification:" << endl;
    cout << "Sequential == Parallel: " << (sequentialEqualsParallel ? "Yes" : "No") << endl;
    cout << "Parallel == Par+Vec: " << (parallelEqualsParVec ? "Yes" : "No") << endl;
    
    // Other concurrent algorithms
    cout << "\n--- Other Concurrent Algorithms ---" << endl;
    
    // Transform
    vector<int> transformedSeq(dataSize);
    vector<int> transformedPar(dataSize);
    vector<int> transformedParVec(dataSize);
    
    cout << "Transform (sequential): ";
    measureTime([&data, &transformedSeq]() {
        transform(seq, data.begin(), data.end(), transformedSeq.begin(),
                  [](int x) { return x * x; });
    });
    
    cout << "Transform (parallel): ";
    measureTime([&data, &transformedPar]() {
        transform(par, data.begin(), data.end(), transformedPar.begin(),
                  [](int x) { return x * x; });
    });
    
    cout << "Transform (par+vec): ";
    measureTime([&data, &transformedParVec]() {
        transform(par_unseq, data.begin(), data.end(), transformedParVec.begin(),
                  [](int x) { return x * x; });
    });
    
    // Accumulate (reduction)
    cout << "\nAccumulate (sequential): ";
    auto seqSum = measureTime([&data]() {
        return accumulate(seq, data.begin(), data.end(), 0LL);
    });
    
    cout << "Accumulate (parallel): ";
    auto parSum = measureTime([&data]() {
        return reduce(par, data.begin(), data.end(), 0LL);
    });
    
    cout << "Accumulate (par+vec): ";
    auto parVecSum = measureTime([&data]() {
        return reduce(par_unseq, data.begin(), data.end(), 0LL);
    });
    
    cout << "Sums - Seq: " << seqSum << ", Par: " << parSum << ", ParVec: " << parVecSum << endl;
    
    // Count if
    cout << "\nCount if (sequential): ";
    auto seqCount = measureTime([&data]() {
        return count_if(seq, data.begin(), data.end(),
                       [](int x) { return x > 500; });
    });
    
    cout << "Count if (parallel): ";
    auto parCount = measureTime([&data]() {
        return count_if(par, data.begin(), data.end(),
                       [](int x) { return x > 500; });
    });
    
    cout << "Count if (par+vec): ";
    auto parVecCount = measureTime([&data]() {
        return count_if(par_unseq, data.begin(), data.end(),
                       [](int x) { return x > 500; });
    });
    
    cout << "Counts - Seq: " << seqCount << ", Par: " << parCount << ", ParVec: " << parVecCount << endl;
    
    // Find
    int target = 500;
    cout << "\nFind (sequential): ";
    auto seqFind = measureTime([&data, target]() {
        return find(seq, data.begin(), data.end(), target);
    });
    
    cout << "Find (parallel): ";
    auto parFind = measureTime([&data, target]() {
        return find(par, data.begin(), data.end(), target);
    });
    
    cout << "Find (par+vec): ";
    auto parVecFind = measureTime([&data, target]() {
        return find(par_unseq, data.begin(), data.end(), target);
    });
    
    cout << "Find results - Seq: " << (seqFind != data.end() ? "Found" : "Not found")
         << ", Par: " << (parFind != data.end() ? "Found" : "Not found")
         << ", ParVec: " << (parVecFind != data.end() ? "Found" : "Not found") << endl;
    
    // For each (side effects)
    atomic<int> evenCounter{0};
    atomic<int> oddCounter{0};
    
    cout << "\nFor each (sequential): ";
    measureTime([&data, &evenCounter, &oddCounter]() {
        evenCounter = 0;
        oddCounter = 0;
        for_each(seq, data.begin(), data.end(), [&](int x) {
            if (x % 2 == 0) {
                evenCounter.fetch_add(1);
            } else {
                oddCounter.fetch_add(1);
            }
        });
    });
    cout << "Evens: " << evenCounter << ", Odds: " << oddCounter << endl;
    
    cout << "For each (parallel): ";
    measureTime([&data, &evenCounter, &oddCounter]() {
        evenCounter = 0;
        oddCounter = 0;
        for_each(par, data.begin(), data.end(), [&](int x) {
            if (x % 2 == 0) {
                evenCounter.fetch_add(1);
            } else {
                oddCounter.fetch_add(1);
            }
        });
    });
    cout << "Evens: " << evenCounter << ", Odds: " << oddCounter << endl;
    
    // Important note about execution policies
    cout << "\n=== Important Notes About Execution Policies ===" << endl;
    cout << "1. seq: Sequential execution, same as regular algorithms" << endl;
    cout << "2. par: Parallel execution, may execute in parallel" << endl;
    cout << "3. par_unseq: Parallel + vectorized, may execute in parallel and vectorized" << endl;
    cout << "4. Not all algorithms support all execution policies" << endl;
    cout << "5. Some operations must be thread-safe when used with parallel policies" << endl;
    cout << "6. Performance gains depend on algorithm complexity and data size" << endl;
    
    return 0;
}
```

## Best Practices and Common Pitfalls

### Exercise 9: Best Practices and Pitfalls

Demonstrate concurrency best practices:

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <future>
#include <vector>
#include <chrono>
using namespace std;

// Good: Thread-safe class design
class ThreadSafeCounter {
private:
    mutable shared_mutex counterMutex;  // Use shared_mutex for read-heavy workloads
    int value;
    
public:
    ThreadSafeCounter(int initial = 0) : value(initial) {}
    
    void increment() {
        unique_lock<shared_mutex> lock(counterMutex);
        ++value;
    }
    
    int getValue() const {
        shared_lock<shared_mutex> lock(counterMutex);
        return value;
    }
    
    void incrementBatch(int amount) {
        unique_lock<shared_mutex> lock(counterMutex);
        value += amount;
    }
};

// Bad: Class with race conditions
class UnsafeCounter {
private:
    int value;  // No synchronization!
    
public:
    UnsafeCounter(int initial = 0) : value(initial) {}
    
    void increment() {
        ++value;  // Race condition!
    }
    
    int getValue() const {
        return value;  // Race condition!
    }
};

// RAII wrapper for thread management
class ScopedThread {
    thread t;
public:
    explicit ScopedThread(thread t_) : t(move(t_)) {
        if (!t.joinable()) {
            throw logic_error("No thread");
        }
    }
    
    ~ScopedThread() {
        if (t.joinable()) {
            t.join();
        }
    }
    
    ScopedThread(const ScopedThread&) = delete;
    ScopedThread& operator=(const ScopedThread&) = delete;
    
    ScopedThread(ScopedThread&&) = default;
    ScopedThread& operator=(ScopedThread&&) = default;
};

int main() {
    cout << "=== Concurrency Best Practices ===" << endl;
    
    // 1. Always join or detach threads
    cout << "\n--- Thread Management ---" << endl;
    
    {
        // RAII approach - automatically joins
        ScopedThread st(thread([]() {
            this_thread::sleep_for(chrono::milliseconds(100));
            cout << "RAII thread completed" << endl;
        }));
    }  // Thread automatically joined when going out of scope
    
    // 2. Use appropriate synchronization primitives
    cout << "\n--- Synchronization Selection ---" << endl;
    
    // For simple counters: atomic is often better than mutex
    atomic<int> atomicCounter{0};
    vector<thread> atomicThreads;
    
    for (int i = 0; i < 4; i++) {
        atomicThreads.emplace_back([&atomicCounter]() {
            for (int j = 0; j < 1000; j++) {
                atomicCounter.fetch_add(1);
            }
        });
    }
    
    for (auto& t : atomicThreads) {
        t.join();
    }
    
    cout << "Atomic counter result: " << atomicCounter.load() << endl;
    
    // For complex objects: use mutex
    ThreadSafeCounter safeCounter(0);
    vector<thread> safeThreads;
    
    for (int i = 0; i < 4; i++) {
        safeThreads.emplace_back([&safeCounter]() {
            for (int j = 0; j < 1000; j++) {
                safeCounter.increment();
            }
        });
    }
    
    for (auto& t : safeThreads) {
        t.join();
    }
    
    cout << "Safe counter result: " << safeCounter.getValue() << endl;
    
    // 3. Avoid deadlock with consistent lock ordering
    cout << "\n--- Deadlock Prevention ---" << endl;
    
    mutex m1, m2;
    
    thread deadlockAvoider1([&m1, &m2]() {
        lock(m1, m2);  // Always acquire locks in same order
        lock_guard<mutex> lock1(m1, adopt_lock);
        lock_guard<mutex> lock2(m2, adopt_lock);
        cout << "Thread 1: Got both locks safely" << endl;
    });
    
    thread deadlockAvoider2([&m1, &m2]() {
        lock(m1, m2);  // Same order as thread 1
        lock_guard<mutex> lock1(m1, adopt_lock);
        lock_guard<mutex> lock2(m2, adopt_lock);
        cout << "Thread 2: Got both locks safely" << endl;
    });
    
    deadlockAvoider1.join();
    deadlockAvoider2.join();
    
    // 4. Use condition variables properly
    cout << "\n--- Condition Variable Best Practices ---" << endl;
    
    mutex cvMtx;
    condition_variable cv;
    bool ready = false;
    int data = 0;
    
    thread cvProducer([&cvMtx, &cv, &ready, &data]() {
        this_thread::sleep_for(chrono::milliseconds(200));
        {
            lock_guard<mutex> lock(cvMtx);
            data = 42;
            ready = true;
        }
        cv.notify_one();
    });
    
    thread cvConsumer([&cvMtx, &cv, &ready, &data]() {
        unique_lock<mutex> lock(cvMtx);
        cv.wait(lock, [&ready] { return ready; });  // Predicate prevents spurious wakeups
        cout << "Consumer: Received data = " << data << endl;
    });
    
    cvProducer.join();
    cvConsumer.join();
    
    // 5. Exception safety with RAII
    cout << "\n--- Exception Safety ---" << endl;
    
    try {
        vector<thread> exceptionThreads;
        
        for (int i = 0; i < 3; i++) {
            exceptionThreads.emplace_back([i]() {
                this_thread::sleep_for(chrono::milliseconds(100 * i));
                if (i == 1) {
                    throw runtime_error("Simulated error");
                }
                cout << "Thread " << i << " completed normally" << endl;
            });
        }
        
        for (auto& t : exceptionThreads) {
            if (t.joinable()) {
                t.join();  // Always join threads, even if exception occurs
            }
        }
        
    } catch (const exception& e) {
        cout << "Caught exception: " << e.what() << endl;
        cout << "But threads were properly cleaned up!" << endl;
    }
    
    // 6. Performance considerations
    cout << "\n--- Performance Considerations ---" << endl;
    
    const int iterations = 1000000;
    
    // Fine-grained locking (potentially slow due to overhead)
    mutex fineMtx;
    int fineCounter = 0;
    
    auto fineStart = chrono::high_resolution_clock::now();
    vector<thread> fineThreads;
    
    for (int i = 0; i < 4; i++) {
        fineThreads.emplace_back([&fineMtx, &fineCounter]() {
            for (int j = 0; j < iterations / 4; j++) {
                lock_guard<mutex> lock(fineMtx);
                fineCounter++;
            }
        });
    }
    
    for (auto& t : fineThreads) {
        t.join();
    }
    auto fineEnd = chrono::high_resolution_clock::now();
    
    cout << "Fine-grained locking time: " 
         << chrono::duration_cast<chrono::milliseconds>(fineEnd - fineStart).count() 
         << " ms" << endl;
    
    // Coarse-grained locking with batch operations
    mutex coarseMtx;
    int coarseCounter = 0;
    
    auto coarseStart = chrono::high_resolution_clock::now();
    vector<thread> coarseThreads;
    
    for (int i = 0; i < 4; i++) {
        coarseThreads.emplace_back([&coarseMtx, &coarseCounter, iterations]() {
            int localCount = 0;
            for (int j = 0; j < iterations / 4; j++) {
                localCount++;  // Do work locally
            }
            lock_guard<mutex> lock(coarseMtx);  // Lock only when updating shared data
            coarseCounter += localCount;
        });
    }
    
    for (auto& t : coarseThreads) {
        t.join();
    }
    auto coarseEnd = chrono::high_resolution_clock::now();
    
    cout << "Coarse-grained locking time: " 
         << chrono::duration_cast<chrono::milliseconds>(coarseEnd - coarseStart).count() 
         << " ms" << endl;
    
    cout << "Fine counter: " << fineCounter << ", Coarse counter: " << coarseCounter << endl;
    
    // 7. Thread-local storage
    cout << "\n--- Thread-Local Storage ---" << endl;
    
    thread_local int threadLocalValue = 0;
    
    vector<thread> tlsThreads;
    for (int i = 0; i < 3; i++) {
        tlsThreads.emplace_back([i, &threadLocalValue]() {
            threadLocalValue = i * 100;
            cout << "Thread " << this_thread::get_id() 
                 << " has thread-local value: " << threadLocalValue << endl;
        });
    }
    
    for (auto& t : tlsThreads) {
        t.join();
    }
    
    cout << "Main thread's thread-local value: " << threadLocalValue << endl;
    
    // 8. Future best practices
    cout << "\n--- Future Best Practices ---" << endl;
    
    // Always handle futures (don't let them go out of scope without getting result)
    vector<future<int>> futures;
    
    for (int i = 0; i < 3; i++) {
        futures.push_back(async(launch::async, [i]() {
            this_thread::sleep_for(chrono::milliseconds(100));
            return i * i;
        }));
    }
    
    cout << "Future results: ";
    for (auto& f : futures) {
        cout << f.get() << " ";  // Always call get() or wait() to avoid blocking destructor
    }
    cout << endl;
    
    cout << "\nConcurrency best practices demonstration completed!" << endl;
    
    return 0;
}
```

## Summary

In this chapter, you learned:
- Thread creation, management, and RAII wrappers
- Synchronization primitives: mutexes, locks, condition variables
- Atomic operations and memory ordering
- Futures, promises, and async programming
- Thread pools for efficient thread management
- Concurrent algorithms with execution policies
- Best practices and common pitfalls in concurrent programming

## Key Takeaways

- Always manage thread lifecycles properly (join or detach)
- Use RAII for automatic resource management
- Choose appropriate synchronization primitives for your use case
- Prefer atomic operations for simple shared data
- Use condition variables for thread coordination
- Apply consistent lock ordering to prevent deadlocks
- Use thread pools for efficient task management
- Leverage concurrent algorithms when beneficial
- Consider exception safety in concurrent code
- Be aware of performance implications of synchronization

## Common Mistakes to Avoid

1. Forgetting to join threads before they go out of scope
2. Using mutexes incorrectly (not using RAII, double-locking)
3. Creating race conditions by not synchronizing shared data access
4. Deadlocking by acquiring locks in different orders
5. Not using predicates with condition variables (causing spurious wakeups)
6. Letting futures go out of scope without retrieving results
7. Over-synchronizing (using fine-grained locks when coarse-grained would suffice)
8. Not considering exception safety in concurrent code
9. Using raw pointers for shared ownership in concurrent contexts
10. Not understanding memory ordering requirements for atomics

## Next Steps

Now that you understand concurrency and multithreading, you're ready to learn about modern C++ features in Chapter 15.