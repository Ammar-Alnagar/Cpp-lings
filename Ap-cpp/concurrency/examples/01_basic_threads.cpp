// Advanced C++ Concurrency: Basic Thread Management
// This example demonstrates fundamental thread operations in modern C++

#include <thread>
#include <iostream>
#include <vector>
#include <chrono>
#include <functional>
#include <string>
#include <sstream>

// Thread-safe output helper to avoid interleaved output
// This uses a mutex which we'll explain in detail in the next chapter
#include <mutex>
std::mutex cout_mutex;

void safe_print(const std::string& message) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << message << std::flush;
}

// Example 1: Simple thread function
// This is the most basic example - a function that runs in a separate thread
void simple_worker(int id) {
    std::stringstream ss;
    ss << "Worker thread " << id << " is running on thread ID: " 
       << std::this_thread::get_id() << "\n";
    safe_print(ss.str());
    
    // Simulate some work
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    ss.str("");
    ss << "Worker thread " << id << " finished\n";
    safe_print(ss.str());
}

// Example 2: Passing arguments by value and reference
void demonstrate_argument_passing() {
    safe_print("\n=== Example 2: Argument Passing ===\n");
    
    // Function that takes arguments by value
    auto by_value = [](int x, double y, std::string s) {
        std::stringstream ss;
        ss << "By value: x=" << x << ", y=" << y << ", s=" << s << "\n";
        safe_print(ss.str());
    };
    
    // Function that modifies a reference
    auto by_reference = [](int& counter) {
        for (int i = 0; i < 5; ++i) {
            ++counter;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    };
    
    // Pass by value (arguments are copied)
    int value = 42;
    double pi = 3.14159;
    std::string name = "Thread";
    
    std::thread t1(by_value, value, pi, name);
    t1.join();
    
    // Pass by reference using std::ref
    // Without std::ref, the thread would receive a copy
    int counter = 0;
    std::thread t2(by_reference, std::ref(counter));
    t2.join();
    
    std::stringstream ss;
    ss << "Counter after thread execution: " << counter << "\n";
    safe_print(ss.str());
}

// Example 3: Thread lifecycle management
class ThreadGuard {
    // RAII wrapper that ensures thread is properly joined
    // This is a critical pattern for exception-safe thread management
    std::thread& t_;
    
public:
    explicit ThreadGuard(std::thread& t) : t_(t) {}
    
    // Destructor automatically joins the thread
    ~ThreadGuard() {
        if (t_.joinable()) {
            t_.join();
        }
    }
    
    // Prevent copying and moving
    ThreadGuard(const ThreadGuard&) = delete;
    ThreadGuard& operator=(const ThreadGuard&) = delete;
};

void demonstrate_raii_wrapper() {
    safe_print("\n=== Example 3: RAII Thread Management ===\n");
    
    auto worker = []() {
        safe_print("RAII-managed thread executing\n");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        safe_print("RAII-managed thread completed\n");
    };
    
    std::thread t(worker);
    ThreadGuard guard(t);
    
    // Even if an exception occurs here, the ThreadGuard destructor
    // will ensure the thread is joined
    safe_print("Main thread continues, thread will be auto-joined\n");
    
    // ThreadGuard's destructor is called here, joining the thread
}

// Example 4: Lambda expressions with threads
void demonstrate_lambdas() {
    safe_print("\n=== Example 4: Lambda Expressions ===\n");
    
    // Lambda with no captures
    std::thread t1([]() {
        safe_print("Lambda with no captures\n");
    });
    
    // Lambda with capture by value
    int x = 100;
    std::thread t2([x]() {
        std::stringstream ss;
        ss << "Lambda captured x by value: " << x << "\n";
        safe_print(ss.str());
    });
    
    // Lambda with capture by reference
    int counter = 0;
    std::thread t3([&counter]() {
        counter = 200;
    });
    
    // Lambda with mutable capture
    // Note: captures are const by default, use mutable to modify
    std::thread t4([y = 50]() mutable {
        y += 10;
        std::stringstream ss;
        ss << "Lambda mutable capture: y = " << y << "\n";
        safe_print(ss.str());
    });
    
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    
    std::stringstream ss;
    ss << "Counter after lambda execution: " << counter << "\n";
    safe_print(ss.str());
}

// Example 5: Hardware concurrency
void demonstrate_hardware_info() {
    safe_print("\n=== Example 5: Hardware Concurrency ===\n");
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    
    std::stringstream ss;
    if (num_threads == 0) {
        ss << "Hardware concurrency information not available\n";
    } else {
        ss << "Hardware supports " << num_threads << " concurrent threads\n";
        ss << "This typically corresponds to the number of logical CPU cores\n";
    }
    safe_print(ss.str());
    
    // Demonstrate creating optimal number of threads
    std::vector<std::thread> threads;
    
    // Reserve space to avoid reallocation (important!)
    threads.reserve(num_threads);
    
    for (unsigned int i = 0; i < num_threads; ++i) {
        threads.emplace_back([i]() {
            std::stringstream ss;
            ss << "Thread " << i << " on core (thread id: " 
               << std::this_thread::get_id() << ")\n";
            safe_print(ss.str());
        });
    }
    
    // Join all threads
    for (auto& t : threads) {
        t.join();
    }
}

// Example 6: Thread ID comparison
void demonstrate_thread_ids() {
    safe_print("\n=== Example 6: Thread IDs ===\n");
    
    std::thread::id main_id = std::this_thread::get_id();
    
    std::stringstream ss;
    ss << "Main thread ID: " << main_id << "\n";
    safe_print(ss.str());
    
    std::thread t([main_id]() {
        std::thread::id worker_id = std::this_thread::get_id();
        std::stringstream ss;
        ss << "Worker thread ID: " << worker_id << "\n";
        ss << "Are they the same? " << (main_id == worker_id ? "Yes" : "No") << "\n";
        safe_print(ss.str());
    });
    
    t.join();
}

// Example 7: Detached threads (use with caution!)
void demonstrate_detached_threads() {
    safe_print("\n=== Example 7: Detached Threads ===\n");
    
    // Create a thread that will run independently
    // IMPORTANT: Make sure no local variables are captured by reference!
    std::thread t([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        safe_print("Detached thread is still running\n");
    });
    
    safe_print("Main thread about to detach worker\n");
    t.detach();
    
    // The thread is now running independently
    // We cannot join it, and it will continue running even after main() exits
    // (though the OS will terminate it when the process ends)
    
    safe_print("Main thread continues (detached thread runs independently)\n");
    
    // Wait a bit to let the detached thread finish
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Example 8: Exception safety
void demonstrate_exception_safety() {
    safe_print("\n=== Example 8: Exception Safety ===\n");
    
    try {
        std::thread t([]() {
            safe_print("Thread working...\n");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        });
        
        // Use ThreadGuard for automatic cleanup
        ThreadGuard guard(t);
        
        // Simulate an exception
        // Even if this throws, ThreadGuard ensures the thread is joined
        safe_print("Simulating risky operation...\n");
        // throw std::runtime_error("Something went wrong!");
        
        safe_print("Operation completed successfully\n");
        
    } catch (const std::exception& e) {
        std::stringstream ss;
        ss << "Exception caught: " << e.what() << "\n";
        safe_print(ss.str());
    }
    
    safe_print("Thread was properly cleaned up\n");
}

// Example 9: Multiple threads working together
void demonstrate_multiple_threads() {
    safe_print("\n=== Example 9: Multiple Threads ===\n");
    
    const int num_workers = 4;
    std::vector<std::thread> workers;
    
    // Launch multiple worker threads
    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back(simple_worker, i);
    }
    
    // Main thread can do other work while workers are running
    safe_print("Main thread: All workers launched, doing other work...\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Wait for all workers to complete
    for (auto& worker : workers) {
        worker.join();
    }
    
    safe_print("Main thread: All workers completed\n");
}

int main() {
    safe_print("=== Advanced C++ Concurrency: Thread Basics ===\n");
    safe_print("This example demonstrates fundamental thread operations\n");
    
    // Run all examples
    demonstrate_argument_passing();
    demonstrate_raii_wrapper();
    demonstrate_lambdas();
    demonstrate_hardware_info();
    demonstrate_thread_ids();
    demonstrate_detached_threads();
    demonstrate_exception_safety();
    demonstrate_multiple_threads();
    
    safe_print("\n=== All examples completed successfully ===\n");
    
    return 0;
}
