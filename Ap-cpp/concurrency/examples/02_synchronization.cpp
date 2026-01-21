// Advanced C++ Concurrency: Synchronization Primitives
// This example demonstrates mutexes, locks, condition variables, and synchronization

#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <sstream>
#include <queue>
#include <random>

// Thread-safe output
std::mutex cout_mutex;
void safe_print(const std::string& msg) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << msg << std::flush;
}

// Example 1: Race Condition Demonstration
namespace example1 {
    int unsafe_counter = 0;
    int safe_counter = 0;
    std::mutex counter_mutex;
    
    void unsafe_increment(int iterations) {
        for (int i = 0; i < iterations; ++i) {
            // This is NOT thread-safe!
            // The ++ operator is actually three operations:
            // 1. Read value
            // 2. Increment
            // 3. Write back
            // These can interleave between threads
            ++unsafe_counter;
        }
    }
    
    void safe_increment(int iterations) {
        for (int i = 0; i < iterations; ++i) {
            // Protect the increment with a mutex
            std::lock_guard<std::mutex> lock(counter_mutex);
            ++safe_counter;
        }
    }
    
    void demonstrate() {
        safe_print("\n=== Example 1: Race Condition ===\n");
        
        const int iterations = 10000;
        const int num_threads = 4;
        
        // Test unsafe version
        unsafe_counter = 0;
        std::vector<std::thread> unsafe_threads;
        for (int i = 0; i < num_threads; ++i) {
            unsafe_threads.emplace_back(unsafe_increment, iterations);
        }
        for (auto& t : unsafe_threads) {
            t.join();
        }
        
        // Test safe version
        safe_counter = 0;
        std::vector<std::thread> safe_threads;
        for (int i = 0; i < num_threads; ++i) {
            safe_threads.emplace_back(safe_increment, iterations);
        }
        for (auto& t : safe_threads) {
            t.join();
        }
        
        std::stringstream ss;
        ss << "Expected value: " << (num_threads * iterations) << "\n";
        ss << "Unsafe counter: " << unsafe_counter << " (likely wrong due to race condition)\n";
        ss << "Safe counter: " << safe_counter << " (correct with mutex protection)\n";
        safe_print(ss.str());
    }
}

// Example 2: Different Lock Types
namespace example2 {
    std::mutex mtx;
    int shared_value = 0;
    
    void demonstrate_lock_guard() {
        // std::lock_guard - simplest RAII lock
        // Locks on construction, unlocks on destruction
        // Cannot be unlocked manually or moved
        std::lock_guard<std::mutex> lock(mtx);
        shared_value++;
        // Automatically unlocks here
    }
    
    void demonstrate_unique_lock() {
        // std::unique_lock - more flexible than lock_guard
        // Can be unlocked and relocked manually
        // Can be moved (but not copied)
        // Supports deferred and timed locking
        
        std::unique_lock<std::mutex> lock(mtx, std::defer_lock);
        // Mutex is not yet locked
        
        // Do some work that doesn't need the lock
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        
        lock.lock();  // Now acquire the lock
        shared_value++;
        lock.unlock();  // Can manually unlock
        
        // Do more work without holding the lock
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        
        lock.lock();  // Can re-acquire
        shared_value--;
        // Automatically unlocks here if still locked
    }
    
    void demonstrate() {
        safe_print("\n=== Example 2: Lock Types ===\n");
        
        std::thread t1(demonstrate_lock_guard);
        std::thread t2(demonstrate_unique_lock);
        
        t1.join();
        t2.join();
        
        std::stringstream ss;
        ss << "Shared value after lock operations: " << shared_value << "\n";
        safe_print(ss.str());
    }
}

// Example 3: Deadlock Prevention with std::scoped_lock
namespace example3 {
    struct Account {
        std::mutex mtx;
        int balance;
        std::string name;
        
        Account(std::string n, int b) : balance(b), name(std::move(n)) {}
    };
    
    // DANGEROUS: Can cause deadlock if called simultaneously
    // from different threads with opposite account order
    void unsafe_transfer(Account& from, Account& to, int amount) {
        std::lock_guard<std::mutex> lock1(from.mtx);
        // If another thread locks 'to' first, deadlock occurs
        std::this_thread::sleep_for(std::chrono::milliseconds(1));  // Increase deadlock chance
        std::lock_guard<std::mutex> lock2(to.mtx);
        
        if (from.balance >= amount) {
            from.balance -= amount;
            to.balance += amount;
        }
    }
    
    // SAFE: std::scoped_lock prevents deadlock
    void safe_transfer(Account& from, Account& to, int amount) {
        // Locks both mutexes simultaneously in a deadlock-free manner
        std::scoped_lock lock(from.mtx, to.mtx);
        
        if (from.balance >= amount) {
            from.balance -= amount;
            to.balance += amount;
            
            std::stringstream ss;
            ss << "Transferred " << amount << " from " << from.name 
               << " to " << to.name << "\n";
            safe_print(ss.str());
        }
    }
    
    void demonstrate() {
        safe_print("\n=== Example 3: Deadlock Prevention ===\n");
        
        Account alice("Alice", 1000);
        Account bob("Bob", 1000);
        
        std::vector<std::thread> threads;
        
        // Create multiple threads doing transfers in different directions
        // With unsafe_transfer, this would likely deadlock
        // With safe_transfer, it always works
        for (int i = 0; i < 5; ++i) {
            threads.emplace_back(safe_transfer, std::ref(alice), std::ref(bob), 10);
            threads.emplace_back(safe_transfer, std::ref(bob), std::ref(alice), 10);
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        std::stringstream ss;
        ss << "Final balances - Alice: " << alice.balance 
           << ", Bob: " << bob.balance << "\n";
        ss << "Total (should be 2000): " << (alice.balance + bob.balance) << "\n";
        safe_print(ss.str());
    }
}

// Example 4: Reader-Writer Lock (std::shared_mutex)
namespace example4 {
    class ThreadSafeCounter {
        mutable std::shared_mutex mtx_;  // mutable allows locking in const methods
        int value_ = 0;
        
    public:
        // Multiple readers can read simultaneously
        int read() const {
            std::shared_lock<std::shared_mutex> lock(mtx_);
            // Simulate some read processing
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            return value_;
        }
        
        // Writer needs exclusive access
        void write(int value) {
            std::unique_lock<std::shared_mutex> lock(mtx_);
            // Simulate some write processing
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            value_ = value;
        }
        
        void increment() {
            std::unique_lock<std::shared_mutex> lock(mtx_);
            ++value_;
        }
    };
    
    void demonstrate() {
        safe_print("\n=== Example 4: Reader-Writer Lock ===\n");
        
        ThreadSafeCounter counter;
        std::vector<std::thread> threads;
        
        // Launch multiple reader threads (can run concurrently)
        for (int i = 0; i < 5; ++i) {
            threads.emplace_back([&counter, i]() {
                int value = counter.read();
                std::stringstream ss;
                ss << "Reader " << i << " read value: " << value << "\n";
                safe_print(ss.str());
            });
        }
        
        // Launch a few writer threads (require exclusive access)
        for (int i = 0; i < 3; ++i) {
            threads.emplace_back([&counter, i]() {
                counter.write(i * 10);
                std::stringstream ss;
                ss << "Writer " << i << " wrote value: " << (i * 10) << "\n";
                safe_print(ss.str());
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        std::stringstream ss;
        ss << "Final counter value: " << counter.read() << "\n";
        safe_print(ss.str());
    }
}

// Example 5: Condition Variables - Producer-Consumer Pattern
namespace example5 {
    class ThreadSafeQueue {
        std::queue<int> queue_;
        mutable std::mutex mtx_;
        std::condition_variable cv_not_empty_;
        std::condition_variable cv_not_full_;
        const size_t max_size_;
        bool done_ = false;
        
    public:
        explicit ThreadSafeQueue(size_t max_size) : max_size_(max_size) {}
        
        void push(int value) {
            std::unique_lock<std::mutex> lock(mtx_);
            
            // Wait until queue is not full
            cv_not_full_.wait(lock, [this]() {
                return queue_.size() < max_size_ || done_;
            });
            
            if (done_) return;
            
            queue_.push(value);
            
            // Notify one waiting consumer
            cv_not_empty_.notify_one();
        }
        
        bool pop(int& value) {
            std::unique_lock<std::mutex> lock(mtx_);
            
            // Wait until queue is not empty or done
            cv_not_empty_.wait(lock, [this]() {
                return !queue_.empty() || done_;
            });
            
            if (queue_.empty()) {
                return false;  // Queue is done and empty
            }
            
            value = queue_.front();
            queue_.pop();
            
            // Notify one waiting producer
            cv_not_full_.notify_one();
            
            return true;
        }
        
        void set_done() {
            {
                std::lock_guard<std::mutex> lock(mtx_);
                done_ = true;
            }
            // Wake up all waiting threads
            cv_not_empty_.notify_all();
            cv_not_full_.notify_all();
        }
    };
    
    void demonstrate() {
        safe_print("\n=== Example 5: Producer-Consumer with Condition Variables ===\n");
        
        ThreadSafeQueue queue(5);  // Max size of 5
        
        // Producer thread
        std::thread producer([&queue]() {
            for (int i = 0; i < 20; ++i) {
                queue.push(i);
                std::stringstream ss;
                ss << "Produced: " << i << "\n";
                safe_print(ss.str());
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            queue.set_done();
            safe_print("Producer finished\n");
        });
        
        // Consumer threads
        std::vector<std::thread> consumers;
        for (int id = 0; id < 3; ++id) {
            consumers.emplace_back([&queue, id]() {
                int value;
                while (queue.pop(value)) {
                    std::stringstream ss;
                    ss << "Consumer " << id << " consumed: " << value << "\n";
                    safe_print(ss.str());
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                }
                std::stringstream ss;
                ss << "Consumer " << id << " finished\n";
                safe_print(ss.str());
            });
        }
        
        producer.join();
        for (auto& consumer : consumers) {
            consumer.join();
        }
        
        safe_print("All producers and consumers finished\n");
    }
}

// Example 6: Recursive Mutex
namespace example6 {
    class RecursiveCounter {
        mutable std::recursive_mutex mtx_;  // mutable allows locking in const methods
        int value_ = 0;
        
    public:
        void increment() {
            std::lock_guard<std::recursive_mutex> lock(mtx_);
            ++value_;
        }
        
        void increment_by(int n) {
            std::lock_guard<std::recursive_mutex> lock(mtx_);
            for (int i = 0; i < n; ++i) {
                increment();  // Can re-lock the same mutex
            }
        }
        
        int get_value() const {
            std::lock_guard<std::recursive_mutex> lock(mtx_);
            return value_;
        }
    };
    
    void demonstrate() {
        safe_print("\n=== Example 6: Recursive Mutex ===\n");
        
        RecursiveCounter counter;
        
        std::thread t1([&counter]() {
            counter.increment_by(5);  // Will recursively lock
        });
        
        std::thread t2([&counter]() {
            counter.increment_by(3);
        });
        
        t1.join();
        t2.join();
        
        std::stringstream ss;
        ss << "Counter value: " << counter.get_value() << "\n";
        safe_print(ss.str());
    }
}

// Example 7: Timed Mutex
namespace example7 {
    std::timed_mutex tmtx;
    
    void try_work(int id, int duration_ms) {
        // Try to acquire lock for 50ms
        if (tmtx.try_lock_for(std::chrono::milliseconds(50))) {
            std::stringstream ss;
            ss << "Thread " << id << " acquired lock\n";
            safe_print(ss.str());
            
            // Simulate work
            std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
            
            ss.str("");
            ss << "Thread " << id << " releasing lock\n";
            safe_print(ss.str());
            
            tmtx.unlock();
        } else {
            std::stringstream ss;
            ss << "Thread " << id << " timeout, couldn't acquire lock\n";
            safe_print(ss.str());
        }
    }
    
    void demonstrate() {
        safe_print("\n=== Example 7: Timed Mutex ===\n");
        
        std::vector<std::thread> threads;
        
        // First thread will hold lock for 100ms
        threads.emplace_back(try_work, 1, 100);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        // These threads will timeout waiting
        threads.emplace_back(try_work, 2, 10);
        threads.emplace_back(try_work, 3, 10);
        
        for (auto& t : threads) {
            t.join();
        }
    }
}

int main() {
    safe_print("=== Advanced C++ Concurrency: Synchronization Primitives ===\n");
    
    example1::demonstrate();
    example2::demonstrate();
    example3::demonstrate();
    example4::demonstrate();
    example5::demonstrate();
    example6::demonstrate();
    example7::demonstrate();
    
    safe_print("\n=== All examples completed successfully ===\n");
    
    return 0;
}
