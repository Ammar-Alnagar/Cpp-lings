# Chapter 6: Advanced Concurrent Patterns

## Introduction

This chapter covers advanced patterns and techniques for concurrent programming, including lock-free data structures, practical design patterns, and real-world concurrent systems.

## Thread Pools

Thread pools reuse threads for multiple tasks, avoiding the overhead of thread creation.

### Basic Thread Pool

```cpp
class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
    
public:
    ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] {
                            return stop || !tasks.empty();
                        });
                        
                        if (stop && tasks.empty())
                            return;
                        
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    
                    task();
                }
            });
        }
    }
    
    template<typename F>
    void enqueue(F&& f) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }
    
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers)
            worker.join();
    }
};
```

### Work Stealing

More advanced: threads steal work from each other when idle.

```cpp
class WorkStealingQueue {
    std::deque<std::function<void()>> queue;
    mutable std::mutex mutex;
    
public:
    void push(std::function<void()> task) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push_front(std::move(task));
    }
    
    bool try_pop(std::function<void()>& task) {
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.empty())
            return false;
        task = std::move(queue.front());
        queue.pop_front();
        return true;
    }
    
    bool try_steal(std::function<void()>& task) {
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.empty())
            return false;
        task = std::move(queue.back());
        queue.pop_back();
        return true;
    }
};
```

## Producer-Consumer Patterns

### Single Producer, Single Consumer (SPSC)

Lock-free queue for one producer and one consumer:

```cpp
template<typename T, size_t Size>
class SPSCQueue {
    std::array<T, Size> buffer;
    std::atomic<size_t> read_pos{0};
    std::atomic<size_t> write_pos{0};
    
public:
    bool push(const T& item) {
        size_t current_write = write_pos.load(std::memory_order_relaxed);
        size_t next_write = (current_write + 1) % Size;
        
        if (next_write == read_pos.load(std::memory_order_acquire))
            return false;  // Queue full
        
        buffer[current_write] = item;
        write_pos.store(next_write, std::memory_order_release);
        return true;
    }
    
    bool pop(T& item) {
        size_t current_read = read_pos.load(std::memory_order_relaxed);
        
        if (current_read == write_pos.load(std::memory_order_acquire))
            return false;  // Queue empty
        
        item = buffer[current_read];
        read_pos.store((current_read + 1) % Size, std::memory_order_release);
        return true;
    }
};
```

### Multiple Producer, Multiple Consumer (MPMC)

More complex, requires synchronization:

```cpp
template<typename T>
class MPMCQueue {
    struct Node {
        std::atomic<Node*> next;
        T data;
    };
    
    std::atomic<Node*> head;
    std::atomic<Node*> tail;
    std::atomic<size_t> size{0};
    
public:
    MPMCQueue() {
        Node* dummy = new Node{nullptr, T{}};
        head.store(dummy);
        tail.store(dummy);
    }
    
    void push(const T& item) {
        Node* node = new Node{nullptr, item};
        Node* prev_tail = tail.exchange(node, std::memory_order_acq_rel);
        prev_tail->next.store(node, std::memory_order_release);
        size.fetch_add(1, std::memory_order_relaxed);
    }
    
    bool pop(T& item) {
        while (true) {
            Node* h = head.load(std::memory_order_acquire);
            Node* next = h->next.load(std::memory_order_acquire);
            
            if (next == nullptr)
                return false;  // Queue empty
            
            if (head.compare_exchange_weak(h, next, 
                                          std::memory_order_release,
                                          std::memory_order_relaxed)) {
                item = next->data;
                delete h;
                size.fetch_sub(1, std::memory_order_relaxed);
                return true;
            }
        }
    }
};
```

## Lock-Free Data Structures

### Lock-Free Stack

Using compare-and-swap for thread-safe push/pop:

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
    void push(const T& data) {
        Node* new_node = new Node(data);
        new_node->next = head.load(std::memory_order_relaxed);
        
        while (!head.compare_exchange_weak(
            new_node->next, new_node,
            std::memory_order_release,
            std::memory_order_relaxed));
    }
    
    bool pop(T& result) {
        Node* old_head = head.load(std::memory_order_acquire);
        
        while (old_head && !head.compare_exchange_weak(
            old_head, old_head->next,
            std::memory_order_release,
            std::memory_order_acquire));
        
        if (old_head) {
            result = old_head->data;
            delete old_head;  // ABA problem!
            return true;
        }
        return false;
    }
};
```

### The ABA Problem

The ABA problem occurs when:
1. Thread 1 reads A
2. Thread 2 changes A→B→A
3. Thread 1's CAS succeeds, but structure has changed

**Solution 1: Tagged Pointers**

```cpp
template<typename T>
class TaggedPointer {
    struct Pointer {
        T* ptr;
        uintptr_t tag;
    };
    
    std::atomic<Pointer> data;
    
public:
    void set(T* ptr) {
        Pointer old = data.load(std::memory_order_relaxed);
        Pointer new_ptr{ptr, old.tag + 1};
        
        while (!data.compare_exchange_weak(
            old, new_ptr,
            std::memory_order_release,
            std::memory_order_relaxed)) {
            new_ptr.tag = old.tag + 1;
        }
    }
};
```

**Solution 2: Hazard Pointers**

```cpp
class HazardPointer {
    std::atomic<void*> pointer{nullptr};
    
public:
    void set(void* ptr) {
        pointer.store(ptr, std::memory_order_release);
    }
    
    void* get() const {
        return pointer.load(std::memory_order_acquire);
    }
    
    void clear() {
        pointer.store(nullptr, std::memory_order_release);
    }
};

// Usage: Before accessing pointer, set hazard pointer
// This prevents other threads from deleting it
```

## Read-Copy-Update (RCU)

RCU allows lock-free reads with infrequent updates:

```cpp
template<typename T>
class RCUPtr {
    std::atomic<T*> ptr;
    
public:
    class ReadGuard {
        T* ptr_;
    public:
        ReadGuard(std::atomic<T*>& p) 
            : ptr_(p.load(std::memory_order_acquire)) {}
        
        T* operator->() const { return ptr_; }
        T& operator*() const { return *ptr_; }
    };
    
    ReadGuard read() {
        return ReadGuard(ptr);
    }
    
    void update(T* new_ptr) {
        T* old = ptr.exchange(new_ptr, std::memory_order_acq_rel);
        
        // Wait for all readers to finish
        // In practice, use epoch-based reclamation
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        delete old;
    }
};
```

## Concurrent Hash Table

Lock-free hash table with separate chaining:

```cpp
template<typename K, typename V>
class ConcurrentHashMap {
    struct Node {
        K key;
        V value;
        std::atomic<Node*> next;
        
        Node(const K& k, const V& v) 
            : key(k), value(v), next(nullptr) {}
    };
    
    std::vector<std::atomic<Node*>> buckets;
    std::atomic<size_t> size_{0};
    
    size_t hash(const K& key) const {
        return std::hash<K>{}(key) % buckets.size();
    }
    
public:
    ConcurrentHashMap(size_t bucket_count = 1024) 
        : buckets(bucket_count) {
        for (auto& bucket : buckets) {
            bucket.store(nullptr);
        }
    }
    
    void insert(const K& key, const V& value) {
        size_t idx = hash(key);
        Node* new_node = new Node(key, value);
        
        Node* head = buckets[idx].load(std::memory_order_acquire);
        new_node->next.store(head, std::memory_order_relaxed);
        
        while (!buckets[idx].compare_exchange_weak(
            head, new_node,
            std::memory_order_release,
            std::memory_order_acquire)) {
            new_node->next.store(head, std::memory_order_relaxed);
        }
        
        size_.fetch_add(1, std::memory_order_relaxed);
    }
    
    bool find(const K& key, V& value) {
        size_t idx = hash(key);
        Node* current = buckets[idx].load(std::memory_order_acquire);
        
        while (current) {
            if (current->key == key) {
                value = current->value;
                return true;
            }
            current = current->next.load(std::memory_order_acquire);
        }
        return false;
    }
};
```

## Double-Checked Locking Pattern

Lazy initialization with minimal synchronization:

```cpp
class Singleton {
    static std::atomic<Singleton*> instance;
    static std::mutex mutex;
    
    Singleton() = default;
    
public:
    static Singleton* get_instance() {
        Singleton* tmp = instance.load(std::memory_order_acquire);
        
        if (tmp == nullptr) {
            std::lock_guard<std::mutex> lock(mutex);
            tmp = instance.load(std::memory_order_relaxed);
            
            if (tmp == nullptr) {
                tmp = new Singleton();
                instance.store(tmp, std::memory_order_release);
            }
        }
        
        return tmp;
    }
};
```

## Futures and Continuations

Building on std::future for complex workflows:

```cpp
template<typename T>
class Future {
    std::shared_ptr<std::promise<T>> promise_;
    std::shared_future<T> future_;
    
public:
    Future() : promise_(std::make_shared<std::promise<T>>()),
               future_(promise_->get_future()) {}
    
    void set_value(const T& value) {
        promise_->set_value(value);
    }
    
    T get() {
        return future_.get();
    }
    
    template<typename F>
    auto then(F&& func) -> Future<decltype(func(std::declval<T>()))> {
        using ResultType = decltype(func(std::declval<T>()));
        Future<ResultType> next;
        
        std::thread([f = std::forward<F>(func), 
                    fut = future_, 
                    prom = next.promise_]() mutable {
            try {
                T value = fut.get();
                prom->set_value(f(value));
            } catch (...) {
                prom->set_exception(std::current_exception());
            }
        }).detach();
        
        return next;
    }
};
```

## Actor Model

Message-passing concurrency:

```cpp
class Actor {
    std::queue<std::function<void()>> mailbox;
    std::mutex mailbox_mutex;
    std::condition_variable mailbox_cv;
    std::atomic<bool> running{true};
    std::thread worker;
    
    void process_messages() {
        while (running) {
            std::function<void()> message;
            
            {
                std::unique_lock<std::mutex> lock(mailbox_mutex);
                mailbox_cv.wait(lock, [this] {
                    return !mailbox.empty() || !running;
                });
                
                if (!running && mailbox.empty())
                    break;
                
                message = std::move(mailbox.front());
                mailbox.pop();
            }
            
            message();
        }
    }
    
public:
    Actor() : worker(&Actor::process_messages, this) {}
    
    ~Actor() {
        running = false;
        mailbox_cv.notify_all();
        worker.join();
    }
    
    template<typename F>
    void send(F&& message) {
        {
            std::lock_guard<std::mutex> lock(mailbox_mutex);
            mailbox.emplace(std::forward<F>(message));
        }
        mailbox_cv.notify_one();
    }
};
```

## Transactional Memory (Conceptual)

Software transactional memory provides atomic transactions:

```cpp
// Conceptual interface (not standard C++)
template<typename T>
class TVar {
    T value;
    std::mutex mutex;
    
public:
    T read() const {
        std::lock_guard<std::mutex> lock(mutex);
        return value;
    }
    
    void write(const T& new_value) {
        std::lock_guard<std::mutex> lock(mutex);
        value = new_value;
    }
};

// Transaction: all or nothing
template<typename F>
void atomic_transaction(F&& func) {
    // In real STM, would retry on conflict
    func();
}
```

## Parallel Algorithms

### Parallel For

```cpp
template<typename Iterator, typename Func>
void parallel_for(Iterator begin, Iterator end, Func func) {
    size_t length = std::distance(begin, end);
    if (length == 0) return;
    
    size_t num_threads = std::thread::hardware_concurrency();
    size_t chunk_size = length / num_threads;
    
    std::vector<std::thread> threads;
    
    auto chunk_begin = begin;
    for (size_t i = 0; i < num_threads - 1; ++i) {
        auto chunk_end = std::next(chunk_begin, chunk_size);
        
        threads.emplace_back([=]() {
            std::for_each(chunk_begin, chunk_end, func);
        });
        
        chunk_begin = chunk_end;
    }
    
    // Process last chunk in main thread
    std::for_each(chunk_begin, end, func);
    
    for (auto& t : threads) {
        t.join();
    }
}
```

### Parallel Reduce

```cpp
template<typename Iterator, typename T, typename BinaryOp>
T parallel_reduce(Iterator begin, Iterator end, T init, BinaryOp op) {
    size_t length = std::distance(begin, end);
    if (length == 0) return init;
    
    size_t num_threads = std::thread::hardware_concurrency();
    size_t chunk_size = length / num_threads;
    
    std::vector<std::future<T>> futures;
    
    auto chunk_begin = begin;
    for (size_t i = 0; i < num_threads - 1; ++i) {
        auto chunk_end = std::next(chunk_begin, chunk_size);
        
        futures.push_back(std::async(std::launch::async, [=]() {
            return std::accumulate(chunk_begin, chunk_end, T{}, op);
        }));
        
        chunk_begin = chunk_end;
    }
    
    T result = std::accumulate(chunk_begin, end, init, op);
    
    for (auto& fut : futures) {
        result = op(result, fut.get());
    }
    
    return result;
}
```

## Performance Optimization

### False Sharing Prevention

```cpp
// BAD: x and y on same cache line
struct Counters {
    std::atomic<int> x;
    std::atomic<int> y;
};

// GOOD: Separate cache lines
struct alignas(64) Counter {
    std::atomic<int> value;
};

Counter x, y;  // On different cache lines
```

### Lock-Free When Possible

```cpp
// Prefer atomic over mutex for simple operations
std::atomic<int> counter{0};  // Better than mutex-protected int
```

### Batching

```cpp
// Reduce synchronization overhead by batching operations
class BatchedQueue {
    std::vector<int> batch;
    std::mutex mutex;
    
public:
    void add_batch(const std::vector<int>& items) {
        std::lock_guard<std::mutex> lock(mutex);
        batch.insert(batch.end(), items.begin(), items.end());
    }
};
```

## Summary

Key concepts covered:
- Thread pools and work stealing
- Producer-consumer patterns (SPSC, MPMC)
- Lock-free data structures
- ABA problem and solutions
- RCU pattern
- Concurrent hash tables
- Actor model
- Parallel algorithms
- Performance optimization techniques

## Next Steps

Practice implementing these patterns, measure their performance, and understand when each is appropriate. Consider studying production implementations like Intel TBB, Folly, and libcds for real-world examples.
