/*
 * Chapter 14 Exercise: Concurrency and Multithreading
 * 
 * Complete the program that demonstrates multithreading and synchronization.
 * The program should implement a thread-safe producer-consumer pattern.
 */

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <chrono>
#include <random>
#include <functional>

// TODO: Implement a thread-safe queue for producer-consumer pattern
template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;

public:
    // TODO: Implement push method that notifies waiting consumers
    void push(T item) {
        // TODO: Lock mutex, add item to queue, notify one waiting thread
    }
    
    // TODO: Implement pop method that waits for items to be available
    T pop() {
        // TODO: Lock mutex, wait for item to be available, remove and return item
    }
    
    // TODO: Implement tryPop method that doesn't block
    bool tryPop(T& item) {
        // TODO: Try to lock mutex, pop item if available, return success status
    }
    
    // TODO: Implement empty method
    bool empty() const {
        // TODO: Check if queue is empty (with proper synchronization)
    }
    
    // TODO: Implement size method
    size_t size() const {
        // TODO: Return queue size (with proper synchronization)
    }
};

// TODO: Implement a thread-safe counter
class ThreadSafeCounter {
private:
    mutable std::mutex mutex_;
    int value_;
    
public:
    ThreadSafeCounter(int initial = 0) : value_(initial) {}
    
    // TODO: Implement increment method
    void increment() {
        // TODO: Lock mutex and increment value
    }
    
    // TODO: Implement getValue method
    int getValue() const {
        // TODO: Lock mutex and return value
    }
    
    // TODO: Implement setValue method
    void setValue(int value) {
        // TODO: Lock mutex and set value
    }
};

// TODO: Implement a simple thread pool
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
    
public:
    // TODO: Implement constructor that creates worker threads
    explicit ThreadPool(size_t numThreads) : stop(false) {
        // TODO: Create numThreads worker threads that process tasks from the queue
    }
    
    // TODO: Implement enqueue method to add tasks to the thread pool
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type> {
        // TODO: Create a packaged_task, add to queue, and return future
    }
    
    // TODO: Implement destructor
    ~ThreadPool() {
        // TODO: Stop all workers and join all threads
    }
};

// TODO: Implement a function that simulates work and can be run in threads
void workerFunction(int id, ThreadSafeCounter& counter, int iterations) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(10, 100);  // Random delay between 10-100ms
    
    for (int i = 0; i < iterations; i++) {
        // Simulate work with random delay
        std::this_thread::sleep_for(std::chrono::milliseconds(dis(gen)));
        
        // Increment shared counter
        counter.increment();
        
        std::cout << "Worker " << id << " completed task " << (i+1) 
                  << ", counter value: " << counter.getValue() << std::endl;
    }
}

// TODO: Implement producer function for the thread-safe queue
void producer(ThreadSafeQueue<int>& queue, int id, int itemCount) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
    
    for (int i = 0; i < itemCount; i++) {
        int value = dis(gen);
        std::cout << "Producer " << id << " producing: " << value << std::endl;
        queue.push(value);
        
        // Random delay between productions
        std::this_thread::sleep_for(std::chrono::milliseconds(dis(gen) % 100));
    }
    std::cout << "Producer " << id << " finished" << std::endl;
}

// TODO: Implement consumer function for the thread-safe queue
void consumer(ThreadSafeQueue<int>& queue, int id, int maxItems) {
    int consumed = 0;
    while (consumed < maxItems) {
        try {
            int value = queue.pop();  // This will block if queue is empty
            std::cout << "Consumer " << id << " consuming: " << value << std::endl;
            consumed++;
            
            // Simulate processing time
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        } catch (const std::exception& e) {
            std::cout << "Consumer " << id << " exception: " << e.what() << std::endl;
            break;
        }
    }
    std::cout << "Consumer " << id << " finished after consuming " << consumed << " items" << std::endl;
}

int main() {
    std::cout << "=== Concurrency and Multithreading Exercise ===" << std::endl;
    
    // TODO: Test thread-safe counter
    ThreadSafeCounter counter(0);
    
    std::vector<std::thread> counterThreads;
    const int numCounterThreads = 4;
    const int iterationsPerThread = 100;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numCounterThreads; i++) {
        counterThreads.emplace_back(workerFunction, i, std::ref(counter), iterationsPerThread);
    }
    
    for (auto& t : counterThreads) {
        t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Final counter value: " << counter.getValue() << std::endl;
    std::cout << "Expected value: " << numCounterThreads * iterationsPerThread << std::endl;
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
    
    // TODO: Test thread-safe queue with producer-consumer pattern
    ThreadSafeQueue<int> sharedQueue;
    
    std::vector<std::thread> producerThreads;
    std::vector<std::thread> consumerThreads;
    
    // Create producers
    for (int i = 0; i < 2; i++) {
        producerThreads.emplace_back(producer, std::ref(sharedQueue), i, 5);
    }
    
    // Create consumers
    for (int i = 0; i < 3; i++) {
        consumerThreads.emplace_back(consumer, std::ref(sharedQueue), i, 3);
    }
    
    // Wait for all threads to complete
    for (auto& t : producerThreads) {
        t.join();
    }
    for (auto& t : consumerThreads) {
        t.join();
    }
    
    std::cout << "Final queue size: " << sharedQueue.size() << std::endl;
    
    // TODO: Test thread pool
    std::cout << "\n=== Thread Pool Demo ===" << endl;
    
    ThreadPool pool(4);  // 4 threads in the pool
    
    std::vector<std::future<int>> results;
    
    // Submit tasks to the thread pool
    for (int i = 0; i < 8; i++) {
        results.emplace_back(pool.enqueue([i] {
            std::cout << "Task " << i << " running on thread " 
                      << std::this_thread::get_id() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return i * i;
        }));
    }
    
    std::cout << "All tasks submitted, waiting for results..." << std::endl;
    
    // Get all results
    for (auto& result : results) {
        std::cout << "Result: " << result.get() << std::endl;
    }
    
    // TODO: Demonstrate std::async for asynchronous execution
    std::cout << "\n=== Async Demo ===" << std::endl;
    
    auto future1 = std::async(std::launch::async, []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return "Hello from async";
    });
    
    auto future2 = std::async(std::launch::async, [](int x, int y) {
        return x + y;
    }, 10, 20);
    
    std::cout << "Async result 1: " << future1.get() << std::endl;
    std::cout << "Async result 2: " << future2.get() << std::endl;
    
    // TODO: Demonstrate atomic operations
    std::cout << "\n=== Atomic Operations Demo ===" << std::endl;
    
    std::atomic<int> atomicCounter{0};
    std::vector<std::thread> atomicThreads;
    
    for (int i = 0; i < 4; i++) {
        atomicThreads.emplace_back([&atomicCounter, iterationsPerThread]() {
            for (int j = 0; j < iterationsPerThread; j++) {
                atomicCounter.fetch_add(1);
            }
        });
    }
    
    for (auto& t : atomicThreads) {
        t.join();
    }
    
    std::cout << "Atomic counter result: " << atomicCounter.load() << std::endl;
    std::cout << "Expected: " << 4 * iterationsPerThread << std::endl;
    
    // TODO: Demonstrate shared_mutex for reader-writer locking
    std::cout << "\n=== Shared Mutex Demo ===" << std::endl;
    
    std::shared_mutex rwMutex;
    std::vector<int> sharedData = {1, 2, 3, 4, 5};
    
    auto reader = [&sharedData, &rwMutex](int id) {
        for (int i = 0; i < 3; i++) {
            std::shared_lock<std::shared_mutex> lock(rwMutex);
            std::cout << "Reader " << id << " sees: ";
            for (int val : sharedData) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    };
    
    auto writer = [&sharedData, &rwMutex](int id) {
        for (int i = 0; i < 2; i++) {
            std::unique_lock<std::shared_mutex> lock(rwMutex);
            sharedData.push_back(id * 10 + i);
            std::cout << "Writer " << id << " added element, size now: " 
                      << sharedData.size() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }
    };
    
    std::thread reader1(reader, 1);
    std::thread reader2(reader, 2);
    std::thread writer1(writer, 1);
    std::thread writer2(writer, 2);
    
    reader1.join();
    reader2.join();
    writer1.join();
    writer2.join();
    
    std::cout << "\nFinal shared data: ";
    for (int val : sharedData) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\nConcurrency exercise completed!" << std::endl;
    
    return 0;
}