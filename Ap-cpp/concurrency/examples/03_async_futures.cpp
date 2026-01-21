// Advanced C++ Concurrency: Asynchronous Programming with Futures
// This example demonstrates std::async, std::future, std::promise, and std::packaged_task

#include <future>
#include <thread>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <string>
#include <sstream>
#include <mutex>
#include <queue>
#include <functional>
#include <random>

// Thread-safe output
std::mutex cout_mutex;
void safe_print(const std::string& msg) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << msg << std::flush;
}

// Example 1: Basic std::async usage
namespace example1 {
    int expensive_computation(int x) {
        // Simulate expensive work
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return x * x;
    }
    
    void demonstrate() {
        safe_print("\n=== Example 1: Basic std::async ===\n");
        
        // Launch async task (may run in separate thread)
        std::future<int> result = std::async(expensive_computation, 42);
        
        // Do other work while computation runs
        safe_print("Main thread continues while computation runs...\n");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // Get the result (blocks if not ready)
        int value = result.get();
        
        std::stringstream ss;
        ss << "Result: " << value << "\n";
        safe_print(ss.str());
    }
}

// Example 2: Launch policies
namespace example2 {
    int compute(int id) {
        std::stringstream ss;
        ss << "Task " << id << " running on thread " 
           << std::this_thread::get_id() << "\n";
        safe_print(ss.str());
        return id * 2;
    }
    
    void demonstrate() {
        safe_print("\n=== Example 2: Launch Policies ===\n");
        
        std::thread::id main_id = std::this_thread::get_id();
        std::stringstream ss;
        ss << "Main thread ID: " << main_id << "\n";
        safe_print(ss.str());
        
        // std::launch::async - guaranteed to run in separate thread
        auto fut1 = std::async(std::launch::async, compute, 1);
        
        // std::launch::deferred - runs when get() is called (synchronously)
        auto fut2 = std::async(std::launch::deferred, compute, 2);
        
        // Default - implementation chooses
        auto fut3 = std::async(compute, 3);
        
        safe_print("Calling get() on deferred task...\n");
        int result2 = fut2.get();  // Runs synchronously here
        
        safe_print("Getting other results...\n");
        int result1 = fut1.get();
        int result3 = fut3.get();
        
        ss.str("");
        ss << "Results: " << result1 << ", " << result2 << ", " << result3 << "\n";
        safe_print(ss.str());
    }
}

// Example 3: std::promise and std::future communication
namespace example3 {
    void compute_value(std::promise<int> prom, int x) {
        // Simulate computation
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        safe_print("Worker thread: Computation complete, setting value\n");
        
        // Set the value (makes future ready)
        prom.set_value(x * x);
    }
    
    void demonstrate() {
        safe_print("\n=== Example 3: Promise-Future Communication ===\n");
        
        std::promise<int> prom;
        std::future<int> fut = prom.get_future();
        
        // Launch thread with promise
        std::thread t(compute_value, std::move(prom), 10);
        
        safe_print("Main thread: Waiting for result...\n");
        
        // Wait for result
        int result = fut.get();
        
        std::stringstream ss;
        ss << "Main thread: Received result: " << result << "\n";
        safe_print(ss.str());
        
        t.join();
    }
}

// Example 4: Exception handling with futures
namespace example4 {
    int risky_computation(int x) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        if (x < 0) {
            throw std::invalid_argument("Negative input not allowed");
        }
        
        return x * 2;
    }
    
    void demonstrate() {
        safe_print("\n=== Example 4: Exception Handling ===\n");
        
        // Task that succeeds
        auto fut1 = std::async(std::launch::async, risky_computation, 5);
        
        // Task that throws
        auto fut2 = std::async(std::launch::async, risky_computation, -5);
        
        try {
            int result1 = fut1.get();
            std::stringstream ss;
            ss << "First task succeeded: " << result1 << "\n";
            safe_print(ss.str());
        } catch (const std::exception& e) {
            safe_print("First task failed (unexpected)\n");
        }
        
        try {
            fut2.get();
            safe_print("Second task succeeded (unexpected)\n");
        } catch (const std::exception& e) {
            std::stringstream ss;
            ss << "Second task threw exception: " << e.what() << "\n";
            safe_print(ss.str());
        }
    }
}

// Example 5: std::packaged_task
namespace example5 {
    int multiply(int a, int b) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return a * b;
    }
    
    void demonstrate() {
        safe_print("\n=== Example 5: std::packaged_task ===\n");
        
        // Create a packaged task
        std::packaged_task<int(int, int)> task(multiply);
        
        // Get the future before moving the task
        std::future<int> result = task.get_future();
        
        // Execute task in a separate thread
        std::thread t(std::move(task), 6, 7);
        
        safe_print("Main thread: Task launched, waiting for result...\n");
        
        // Get result
        int value = result.get();
        
        std::stringstream ss;
        ss << "Result: " << value << "\n";
        safe_print(ss.str());
        
        t.join();
    }
}

// Example 6: Task queue with packaged_task
namespace example6 {
    class TaskQueue {
        std::queue<std::packaged_task<int()>> queue_;
        std::mutex mtx_;
        std::condition_variable cv_;
        bool done_ = false;
        
    public:
        void push(std::packaged_task<int()>&& task) {
            {
                std::lock_guard<std::mutex> lock(mtx_);
                queue_.push(std::move(task));
            }
            cv_.notify_one();
        }
        
        void worker_thread() {
            while (true) {
                std::packaged_task<int()> task;
                
                {
                    std::unique_lock<std::mutex> lock(mtx_);
                    cv_.wait(lock, [this]() { 
                        return !queue_.empty() || done_; 
                    });
                    
                    if (done_ && queue_.empty()) {
                        return;
                    }
                    
                    task = std::move(queue_.front());
                    queue_.pop();
                }
                
                // Execute task outside the lock
                task();
            }
        }
        
        void shutdown() {
            {
                std::lock_guard<std::mutex> lock(mtx_);
                done_ = true;
            }
            cv_.notify_all();
        }
    };
    
    void demonstrate() {
        safe_print("\n=== Example 6: Task Queue with packaged_task ===\n");
        
        TaskQueue queue;
        std::vector<std::future<int>> futures;
        
        // Start worker thread
        std::thread worker([&queue]() { queue.worker_thread(); });
        
        // Submit tasks
        for (int i = 0; i < 5; ++i) {
            std::packaged_task<int()> task([i]() {
                std::stringstream ss;
                ss << "Executing task " << i << "\n";
                safe_print(ss.str());
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                return i * 10;
            });
            
            futures.push_back(task.get_future());
            queue.push(std::move(task));
        }
        
        // Collect results
        safe_print("Collecting results...\n");
        for (size_t i = 0; i < futures.size(); ++i) {
            int result = futures[i].get();
            std::stringstream ss;
            ss << "Task " << i << " result: " << result << "\n";
            safe_print(ss.str());
        }
        
        queue.shutdown();
        worker.join();
    }
}

// Example 7: std::shared_future
namespace example7 {
    void demonstrate() {
        safe_print("\n=== Example 7: std::shared_future ===\n");
        
        std::promise<int> prom;
        std::shared_future<int> shared_fut = prom.get_future().share();
        
        // Multiple threads can wait on the same shared_future
        auto waiter = [shared_fut](int id) {
            std::stringstream ss;
            ss << "Thread " << id << " waiting...\n";
            safe_print(ss.str());
            
            int value = shared_fut.get();  // Multiple threads can call get()
            
            ss.str("");
            ss << "Thread " << id << " got value: " << value << "\n";
            safe_print(ss.str());
        };
        
        std::vector<std::thread> threads;
        for (int i = 0; i < 3; ++i) {
            threads.emplace_back(waiter, i);
        }
        
        // Give threads time to start waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        safe_print("Setting value...\n");
        prom.set_value(42);
        
        for (auto& t : threads) {
            t.join();
        }
    }
}

// Example 8: Parallel accumulate using futures
namespace example8 {
    template<typename Iterator>
    typename Iterator::value_type parallel_accumulate(Iterator begin, Iterator end) {
        using value_type = typename Iterator::value_type;
        
        size_t length = std::distance(begin, end);
        if (length == 0) return value_type{};
        
        size_t num_threads = std::min(
            static_cast<size_t>(std::thread::hardware_concurrency()),
            length
        );
        
        size_t chunk_size = length / num_threads;
        
        std::vector<std::future<value_type>> futures;
        
        auto chunk_begin = begin;
        for (size_t i = 0; i < num_threads - 1; ++i) {
            auto chunk_end = std::next(chunk_begin, chunk_size);
            
            futures.push_back(std::async(std::launch::async,
                [](Iterator b, Iterator e) {
                    return std::accumulate(b, e, value_type{});
                }, chunk_begin, chunk_end));
            
            chunk_begin = chunk_end;
        }
        
        // Process last chunk in current thread
        value_type last_result = std::accumulate(chunk_begin, end, value_type{});
        
        // Collect results from futures
        value_type total = last_result;
        for (auto& fut : futures) {
            total += fut.get();
        }
        
        return total;
    }
    
    void demonstrate() {
        safe_print("\n=== Example 8: Parallel Accumulate ===\n");
        
        std::vector<int> data(1000000);
        std::iota(data.begin(), data.end(), 1);  // Fill with 1, 2, 3, ...
        
        // Sequential version
        auto start = std::chrono::high_resolution_clock::now();
        long long seq_result = std::accumulate(data.begin(), data.end(), 0LL);
        auto seq_time = std::chrono::high_resolution_clock::now() - start;
        
        // Parallel version
        start = std::chrono::high_resolution_clock::now();
        long long par_result = parallel_accumulate(data.begin(), data.end());
        auto par_time = std::chrono::high_resolution_clock::now() - start;
        
        std::stringstream ss;
        ss << "Sequential result: " << seq_result 
           << " (time: " << std::chrono::duration_cast<std::chrono::microseconds>(seq_time).count() << " us)\n";
        ss << "Parallel result: " << par_result 
           << " (time: " << std::chrono::duration_cast<std::chrono::microseconds>(par_time).count() << " us)\n";
        ss << "Results match: " << (seq_result == par_result ? "Yes" : "No") << "\n";
        safe_print(ss.str());
    }
}

// Example 9: Wait with timeout
namespace example9 {
    void demonstrate() {
        safe_print("\n=== Example 9: Future Wait with Timeout ===\n");
        
        auto long_task = std::async(std::launch::async, []() {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            return 42;
        });
        
        safe_print("Checking if result is ready...\n");
        
        // Try to get result with timeout
        std::future_status status = long_task.wait_for(std::chrono::milliseconds(50));
        
        if (status == std::future_status::ready) {
            safe_print("Result is ready immediately\n");
        } else if (status == std::future_status::timeout) {
            safe_print("Task still running, waiting more...\n");
            
            // Wait again with longer timeout
            status = long_task.wait_for(std::chrono::milliseconds(200));
            
            if (status == std::future_status::ready) {
                int result = long_task.get();
                std::stringstream ss;
                ss << "Result ready now: " << result << "\n";
                safe_print(ss.str());
            }
        }
    }
}

// Example 10: Broken promise
namespace example10 {
    void demonstrate() {
        safe_print("\n=== Example 10: Broken Promise ===\n");
        
        std::future<int> fut;
        
        {
            std::promise<int> prom;
            fut = prom.get_future();
            // Promise goes out of scope without calling set_value()
        }
        
        try {
            fut.get();
            safe_print("Got value (unexpected)\n");
        } catch (const std::future_error& e) {
            std::stringstream ss;
            ss << "Caught future_error: " << e.what() << "\n";
            if (e.code() == std::future_errc::broken_promise) {
                ss << "Error code: broken_promise\n";
            }
            safe_print(ss.str());
        }
    }
}

int main() {
    safe_print("=== Advanced C++ Concurrency: Async Programming with Futures ===\n");
    
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
