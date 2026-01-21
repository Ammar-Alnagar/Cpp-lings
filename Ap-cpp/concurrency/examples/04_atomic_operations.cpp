// Advanced C++ Concurrency: Atomic Operations and Lock-Free Programming
// This example demonstrates std::atomic, memory ordering, and lock-free patterns

#include <atomic>
#include <thread>
#include <vector>
#include <iostream>
#include <chrono>
#include <string>
#include <sstream>
#include <mutex>

// Thread-safe output
std::mutex cout_mutex;
void safe_print(const std::string& msg) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << msg << std::flush;
}

// Example 1: Basic atomic operations
namespace example1 {
    std::atomic<int> counter{0};
    
    void increment(int iterations) {
        for (int i = 0; i < iterations; ++i) {
            counter++;  // Atomic increment
        }
    }
    
    void demonstrate() {
        safe_print("\n=== Example 1: Basic Atomic Operations ===\n");
        
        const int iterations = 10000;
        const int num_threads = 4;
        
        counter = 0;
        std::vector<std::thread> threads;
        
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(increment, iterations);
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        std::stringstream ss;
        ss << "Expected: " << (num_threads * iterations) << "\n";
        ss << "Actual: " << counter.load() << "\n";
        ss << "Match: " << (counter == num_threads * iterations ? "Yes" : "No") << "\n";
        safe_print(ss.str());
    }
}

// Example 2: Atomic load and store
namespace example2 {
    std::atomic<int> shared_value{0};
    
    void writer(int id) {
        for (int i = 0; i < 5; ++i) {
            shared_value.store(id * 100 + i);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    void reader(int id) {
        for (int i = 0; i < 10; ++i) {
            int value = shared_value.load();
            std::stringstream ss;
            ss << "Reader " << id << " read: " << value << "\n";
            safe_print(ss.str());
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    
    void demonstrate() {
        safe_print("\n=== Example 2: Atomic Load and Store ===\n");
        
        std::thread w1(writer, 1);
        std::thread w2(writer, 2);
        std::thread r1(reader, 1);
        
        w1.join();
        w2.join();
        r1.join();
    }
}

// Example 3: Fetch-and-modify operations
namespace example3 {
    void demonstrate() {
        safe_print("\n=== Example 3: Fetch-and-Modify Operations ===\n");
        
        std::atomic<int> x{100};
        
        std::stringstream ss;
        ss << "Initial value: " << x.load() << "\n";
        
        // fetch_add returns old value
        int old = x.fetch_add(50);
        ss << "After fetch_add(50): old=" << old << ", new=" << x.load() << "\n";
        
        // fetch_sub returns old value
        old = x.fetch_sub(30);
        ss << "After fetch_sub(30): old=" << old << ", new=" << x.load() << "\n";
        
        // Bitwise operations
        x = 0xFF;
        old = x.fetch_and(0x0F);
        ss << "After fetch_and(0x0F): old=" << std::hex << old 
           << ", new=" << x.load() << std::dec << "\n";
        
        x = 0x0F;
        old = x.fetch_or(0xF0);
        ss << "After fetch_or(0xF0): old=" << std::hex << old 
           << ", new=" << x.load() << std::dec << "\n";
        
        safe_print(ss.str());
    }
}

// Example 4: Exchange operation
namespace example4 {
    std::atomic<int> shared{0};
    
    void worker(int id, int value) {
        int old = shared.exchange(value);
        std::stringstream ss;
        ss << "Thread " << id << " exchanged " << old << " with " << value << "\n";
        safe_print(ss.str());
    }
    
    void demonstrate() {
        safe_print("\n=== Example 4: Exchange Operation ===\n");
        
        std::vector<std::thread> threads;
        
        for (int i = 1; i <= 5; ++i) {
            threads.emplace_back(worker, i, i * 10);
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        std::stringstream ss;
        ss << "Final value: " << shared.load() << "\n";
        safe_print(ss.str());
    }
}

// Example 5: Compare-and-Exchange (CAS)
namespace example5 {
    std::atomic<int> value{0};
    
    void increment_with_cas() {
        int expected = value.load();
        int desired;
        
        do {
            desired = expected + 1;
        } while (!value.compare_exchange_weak(expected, desired));
        
        std::stringstream ss;
        ss << "Incremented to " << desired << "\n";
        safe_print(ss.str());
    }
    
    void demonstrate() {
        safe_print("\n=== Example 5: Compare-and-Exchange ===\n");
        
        std::vector<std::thread> threads;
        
        for (int i = 0; i < 5; ++i) {
            threads.emplace_back(increment_with_cas);
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        std::stringstream ss;
        ss << "Final value: " << value.load() << "\n";
        safe_print(ss.str());
    }
}

// Example 6: Lock-free stack
namespace example6 {
    template<typename T>
    class LockFreeStack {
        struct Node {
            T data;
            Node* next;
            Node(const T& d) : data(d), next(nullptr) {}
        };
        
        std::atomic<Node*> head{nullptr};
        
    public:
        ~LockFreeStack() {
            while (Node* node = head.load()) {
                head.store(node->next);
                delete node;
            }
        }
        
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
                delete old_head;
                return true;
            }
            return false;
        }
    };
    
    void demonstrate() {
        safe_print("\n=== Example 6: Lock-Free Stack ===\n");
        
        LockFreeStack<int> stack;
        
        // Push from multiple threads
        std::vector<std::thread> pushers;
        for (int i = 0; i < 3; ++i) {
            pushers.emplace_back([&stack, i]() {
                for (int j = 0; j < 5; ++j) {
                    stack.push(i * 10 + j);
                    std::stringstream ss;
                    ss << "Pushed: " << (i * 10 + j) << "\n";
                    safe_print(ss.str());
                }
            });
        }
        
        for (auto& t : pushers) {
            t.join();
        }
        
        // Pop from multiple threads
        std::vector<std::thread> poppers;
        for (int i = 0; i < 3; ++i) {
            poppers.emplace_back([&stack, i]() {
                for (int j = 0; j < 5; ++j) {
                    int value;
                    if (stack.pop(value)) {
                        std::stringstream ss;
                        ss << "Thread " << i << " popped: " << value << "\n";
                        safe_print(ss.str());
                    }
                }
            });
        }
        
        for (auto& t : poppers) {
            t.join();
        }
    }
}

// Example 7: Memory ordering - relaxed
namespace example7 {
    std::atomic<int> x{0}, y{0};
    std::atomic<int> z{0};
    
    void write_x() {
        x.store(1, std::memory_order_relaxed);
    }
    
    void write_y() {
        y.store(1, std::memory_order_relaxed);
    }
    
    void read_xy() {
        while (x.load(std::memory_order_relaxed) == 0) {
            // Wait
        }
        if (y.load(std::memory_order_relaxed) == 1) {
            z.fetch_add(1, std::memory_order_relaxed);
        }
    }
    
    void demonstrate() {
        safe_print("\n=== Example 7: Relaxed Memory Ordering ===\n");
        
        for (int i = 0; i < 100; ++i) {
            x = 0; y = 0; z = 0;
            
            std::thread t1(write_x);
            std::thread t2(write_y);
            std::thread t3(read_xy);
            
            t1.join();
            t2.join();
            t3.join();
        }
        
        std::stringstream ss;
        ss << "With relaxed ordering, synchronization is not guaranteed\n";
        ss << "z incremented in " << z.load() << " out of 100 iterations\n";
        safe_print(ss.str());
    }
}

// Example 8: Memory ordering - acquire-release
namespace example8 {
    std::atomic<bool> ready{false};
    int data = 0;
    
    void producer() {
        data = 42;  // Non-atomic write
        ready.store(true, std::memory_order_release);  // Release
    }
    
    void consumer() {
        while (!ready.load(std::memory_order_acquire)) {  // Acquire
            // Wait
        }
        // Guaranteed to see data == 42
        std::stringstream ss;
        ss << "Consumer read data: " << data << "\n";
        safe_print(ss.str());
    }
    
    void demonstrate() {
        safe_print("\n=== Example 8: Acquire-Release Ordering ===\n");
        
        std::thread t1(producer);
        std::thread t2(consumer);
        
        t1.join();
        t2.join();
    }
}

// Example 9: Spinlock using atomic_flag
namespace example9 {
    class Spinlock {
        std::atomic_flag flag = ATOMIC_FLAG_INIT;
        
    public:
        void lock() {
            while (flag.test_and_set(std::memory_order_acquire)) {
                // Spin - busy wait
                // Can add std::this_thread::yield() here
            }
        }
        
        void unlock() {
            flag.clear(std::memory_order_release);
        }
    };
    
    Spinlock spinlock;
    int shared_counter = 0;
    
    void increment_with_spinlock(int iterations) {
        for (int i = 0; i < iterations; ++i) {
            spinlock.lock();
            ++shared_counter;
            spinlock.unlock();
        }
    }
    
    void demonstrate() {
        safe_print("\n=== Example 9: Spinlock with atomic_flag ===\n");
        
        const int iterations = 10000;
        const int num_threads = 4;
        
        shared_counter = 0;
        std::vector<std::thread> threads;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(increment_with_spinlock, iterations);
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        auto duration = std::chrono::high_resolution_clock::now() - start;
        
        std::stringstream ss;
        ss << "Final counter: " << shared_counter << "\n";
        ss << "Expected: " << (num_threads * iterations) << "\n";
        ss << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(duration).count() << " us\n";
        safe_print(ss.str());
    }
}

// Example 10: Performance comparison - atomic vs mutex
namespace example10 {
    const int ITERATIONS = 100000;
    
    void test_atomic() {
        std::atomic<int> counter{0};
        std::vector<std::thread> threads;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([&counter]() {
                for (int j = 0; j < ITERATIONS; ++j) {
                    counter.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        auto duration = std::chrono::high_resolution_clock::now() - start;
        
        std::stringstream ss;
        ss << "Atomic: " << counter.load() 
           << " in " << std::chrono::duration_cast<std::chrono::microseconds>(duration).count() << " us\n";
        safe_print(ss.str());
    }
    
    void test_mutex() {
        std::mutex mtx;
        int counter = 0;
        std::vector<std::thread> threads;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([&counter, &mtx]() {
                for (int j = 0; j < ITERATIONS; ++j) {
                    std::lock_guard<std::mutex> lock(mtx);
                    counter++;
                }
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        auto duration = std::chrono::high_resolution_clock::now() - start;
        
        std::stringstream ss;
        ss << "Mutex: " << counter 
           << " in " << std::chrono::duration_cast<std::chrono::microseconds>(duration).count() << " us\n";
        safe_print(ss.str());
    }
    
    void demonstrate() {
        safe_print("\n=== Example 10: Atomic vs Mutex Performance ===\n");
        
        test_atomic();
        test_mutex();
        
        safe_print("Note: Atomics are typically faster for simple operations\n");
    }
}

int main() {
    safe_print("=== Advanced C++ Concurrency: Atomic Operations ===\n");
    
    example1::demonstrate();
    example2::demonstrate();
    example3::demonstrate();
    example4::demonstrate();
    example5::demonstrate();
    example6::demonstrate();
    example7::demonstrate();
    example8::demonstrate();
    example9::demonstrate();
    example10::demonstrate();
    
    safe_print("\n=== All examples completed successfully ===\n");
    
    return 0;
}
