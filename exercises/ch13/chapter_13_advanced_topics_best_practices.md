# Chapter 13: Advanced Topics and Best Practices

## Overview

This chapter covers advanced C++ topics and best practices that experienced developers should know. We'll explore design patterns, performance optimization, modern C++ features, and coding guidelines that lead to robust, maintainable code.

## Learning Objectives

By the end of this chapter, you will:
- Understand common design patterns in C++
- Learn performance optimization techniques
- Master modern C++ idioms and best practices
- Understand the Rule of Zero, Five, and Three
- Learn about move semantics optimization
- Understand const-correctness and its importance
- Learn about type safety and avoiding common pitfalls
- Understand modern C++ coding standards and guidelines

## The Rule of Zero, Three, and Five

Understanding when and how to implement special member functions.

### Exercise 1: The Rule of Zero

The following code has errors. Find and fix them:

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <string>
using namespace std;

// Good: Rule of Zero - rely on compiler-generated functions
class GoodClass {
private:
    unique_ptr<int> data;
    vector<string> strings;
    string name;
    
public:
    // No need to define destructor, copy/move constructors, or assignment operators
    // The compiler-generated ones work perfectly with RAII types
    explicit GoodClass(const string& n) : name(n) {
        data = make_unique<int>(42);
        strings = {"hello", "world", "cpp"};
    }
    
    // Accessors
    int getValue() const { return *data; }
    const string& getName() const { return name; }
    const vector<string>& getStrings() const { return strings; }
};

// Bad: Violating Rule of Zero by implementing unnecessary functions
class BadClass {
private:
    unique_ptr<int> data;
    vector<string> strings;
    string name;
    
public:
    explicit BadClass(const string& n) : name(n) {
        data = make_unique<int>(42);
        strings = {"hello", "world", "cpp"};
    }
    
    // Unnecessary implementations - violating Rule of Zero!
    ~BadClass() = default;  // Unnecessary - compiler would generate the same
    
    BadClass(const BadClass& other)  // Unnecessary - compiler would do the same
        : data(make_unique<int>(*other.data)), 
          strings(other.strings), 
          name(other.name) {}
    
    BadClass& operator=(const BadClass& other) {  // Unnecessary
        if (this != &other) {
            data = make_unique<int>(*other.data);
            strings = other.strings;
            name = other.name;
        }
        return *this;
    }
    
    BadClass(BadClass&& other) noexcept  // Unnecessary
        : data(move(other.data)), 
          strings(move(other.strings)), 
          name(move(other.name)) {}
    
    BadClass& operator=(BadClass&& other) noexcept {  // Unnecessary
        if (this != &other) {
            data = move(other.data);
            strings = move(other.strings);
            name = move(other.name);
        }
        return *this;
    }
    
    // Accessors
    int getValue() const { return *data; }
    const string& getName() const { return name; }
    const vector<string>& getStrings() const { return strings; }
};

// When you DO need custom implementations (Rule of Three/Five)
class ResourceManagingClass {
private:
    int* data;
    size_t size;
    
public:
    explicit ResourceManagingClass(size_t s) : size(s) {
        data = new int[size];
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<int>(i);
        }
    }
    
    // Need to implement copy semantics for raw pointer management
    ResourceManagingClass(const ResourceManagingClass& other) : size(other.size) {
        data = new int[size];
        for (size_t i = 0; i < size; i++) {
            data[i] = other.data[i];
        }
    }
    
    ResourceManagingClass& operator=(const ResourceManagingClass& other) {
        if (this != &other) {
            delete[] data;  // Clean up existing resource
            size = other.size;
            data = new int[size];
            for (size_t i = 0; i < size; i++) {
                data[i] = other.data[i];
            }
        }
        return *this;
    }
    
    // Move semantics for efficiency
    ResourceManagingClass(ResourceManagingClass&& other) noexcept 
        : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }
    
    ResourceManagingClass& operator=(ResourceManagingClass&& other) noexcept {
        if (this != &other) {
            delete[] data;  // Clean up existing resource
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
    
    ~ResourceManagingClass() {
        delete[] data;
    }
    
    int& operator[](size_t index) { return data[index]; }
    const int& operator[](size_t index) const { return data[index]; }
    size_t getSize() const { return size; }
};

int main() {
    cout << "=== Rule of Zero/Three/Five Demo ===" << endl;
    
    // Rule of Zero example
    cout << "\n--- Rule of Zero ---" << endl;
    GoodClass goodObj("GoodObject");
    cout << "Good object created: " << goodObj.getName() << endl;
    
    // Copy and move work automatically
    GoodClass copiedObj = goodObj;
    GoodClass movedObj = move(goodObj);
    
    cout << "Copied object: " << copiedObj.getName() << endl;
    cout << "Moved object: " << movedObj.getName() << endl;
    cout << "Original object is now: " << (goodObj.getName().empty() ? "empty" : goodObj.getName()) << endl;
    
    // Rule of Three/Five example
    cout << "\n--- Rule of Three/Five ---" << endl;
    ResourceManagingClass resourceObj(5);
    for (size_t i = 0; i < resourceObj.getSize(); i++) {
        cout << "resourceObj[" << i << "] = " << resourceObj[i] << endl;
    }
    
    // Copy and move work correctly
    ResourceManagingClass copiedResource = resourceObj;
    ResourceManagingClass movedResource = move(resourceObj);
    
    cout << "After copy/move:" << endl;
    cout << "Copied resource size: " << copiedResource.getSize() << endl;
    cout << "Moved resource size: " << movedResource.getSize() << endl;
    
    // Demonstrate the difference in exception safety
    cout << "\n--- Exception Safety Comparison ---" << endl;
    
    try {
        vector<GoodClass> goodVector;
        goodVector.reserve(1000);  // Reserve space to avoid reallocations
        
        for (int i = 0; i < 1000; i++) {
            goodVector.emplace_back("Object" + to_string(i));
        }
        
        cout << "Successfully created " << goodVector.size() << " objects with Rule of Zero class" << endl;
        
        // Simulate error condition
        if (true) {  // Always throw to demonstrate safety
            throw runtime_error("Simulated error");
        }
        
    } catch (const exception& e) {
        cout << "Exception caught: " << e.what() << endl;
        cout << "But all GoodClass objects were properly cleaned up!" << endl;
    }
    
    return 0;
}
```

## Move Semantics and Perfect Forwarding

### Exercise 2: Move Semantics Optimization

Complete this move semantics example:

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <memory>
using namespace std;

class HeavyObject {
private:
    vector<int> data;
    string description;
    
public:
    // Constructor
    explicit HeavyObject(size_t size, const string& desc) 
        : data(size), description(desc) {
        cout << "HeavyObject created: " << description << " with " << size << " elements" << endl;
        // Initialize with some values
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<int>(i);
        }
    }
    
    // Copy constructor
    HeavyObject(const HeavyObject& other) 
        : data(other.data), description(other.description + "_copy") {
        cout << "HeavyObject copied: " << description << endl;
    }
    
    // Copy assignment
    HeavyObject& operator=(const HeavyObject& other) {
        cout << "HeavyObject copy assigned: " << other.description << endl;
        if (this != &other) {
            data = other.data;
            description = other.description + "_assigned";
        }
        return *this;
    }
    
    // Move constructor
    HeavyObject(HeavyObject&& other) noexcept 
        : data(move(other.data)), description(move(other.description)) {
        other.description = "MOVED_FROM_OBJECT";
        cout << "HeavyObject moved: " << description << endl;
    }
    
    // Move assignment
    HeavyObject& operator=(HeavyObject&& other) noexcept {
        cout << "HeavyObject move assigned" << endl;
        if (this != &other) {
            data = move(other.data);
            description = move(other.description);
            other.description = "MOVED_FROM_OBJECT";
        }
        return *this;
    }
    
    // Destructor
    ~HeavyObject() {
        cout << "HeavyObject destroyed: " << description << endl;
    }
    
    // Accessors
    size_t size() const { return data.size(); }
    const string& getDescription() const { return description; }
    const vector<int>& getData() const { return data; }
};

// Function that demonstrates perfect forwarding
template<typename T>
void forwardToConstructor(T&& arg) {
    cout << "Forwarding to constructor..." << endl;
    HeavyObject obj(forward<T>(arg));
    cout << "Object created: " << obj.getDescription() << endl;
}

// Function that accepts any type and forwards it
template<typename T, typename... Args>
auto createAndProcess(Args&&... args) -> T {
    cout << "Creating object with forwarded arguments..." << endl;
    T obj(forward<Args>(args)...);
    cout << "Processing object..." << endl;
    return obj;
}

int main() {
    cout << "=== Move Semantics Demo ===" << endl;
    
    // Create heavy objects
    HeavyObject obj1(1000000, "Original");  // Large object
    
    cout << "\n--- Copy Semantics ---" << endl;
    HeavyObject obj2 = obj1;  // Copy constructor called
    
    cout << "\n--- Move Semantics ---" << endl;
    HeavyObject obj3 = move(obj1);  // Move constructor called
    cout << "After move, original object: " << obj1.getDescription() << endl;
    cout << "Moved object: " << obj3.getDescription() << endl;
    
    cout << "\n--- Move Assignment ---" << endl;
    HeavyObject obj4(1000, "Assignable");
    obj4 = move(obj2);  // Move assignment
    cout << "Assigned object: " << obj4.getDescription() << endl;
    
    cout << "\n--- Perfect Forwarding ---" << endl;
    
    // Forwarding lvalue
    string lvalueStr = "Forwarded LValue";
    forwardToConstructor(lvalueStr);
    cout << "Original string after forwarding: " << lvalueStr << endl;
    
    // Forwarding rvalue
    forwardToConstructor(string("Forwarded RValue"));
    
    cout << "\n--- Universal References ---" << endl;
    
    // Using universal references with perfect forwarding
    auto heavyObj = createAndProcess<HeavyObject>(500000, "Universal Forwarded");
    cout << "Created object size: " << heavyObj.size() << endl;
    
    cout << "\n--- Move Semantics in Containers ---" << endl;
    
    vector<HeavyObject> container;
    
    // Emplace back - constructs object in place (most efficient)
    cout << "\nEmplace back:" << endl;
    container.emplace_back(100000, "Emplaced");
    
    // Push back with rvalue - moves the object
    cout << "\nPush back with rvalue:" << endl;
    container.push_back(HeavyObject(100000, "Pushed"));
    
    // Push back with lvalue - copies the object
    cout << "\nPush back with lvalue:" << endl;
    HeavyObject tempObj(100000, "Temporary");
    container.push_back(tempObj);
    
    cout << "\nContainer size: " << container.size() << endl;
    
    // Demonstrating move-only types
    cout << "\n--- Move-Only Types ---" << endl;
    
    vector<unique_ptr<HeavyObject>> uniquePtrContainer;
    
    // Can only move unique_ptr, not copy
    uniquePtrContainer.push_back(make_unique<HeavyObject>(50000, "Unique1"));
    uniquePtrContainer.push_back(make_unique<HeavyObject>(50000, "Unique2"));
    
    cout << "Unique pointer container size: " << uniquePtrContainer.size() << endl;
    
    // Move the container
    auto movedContainer = move(uniquePtrContainer);
    cout << "After move, original size: " << uniquePtrContainer.size() << endl;
    cout << "Moved container size: " << movedContainer.size() << endl;
    
    cout << "\n--- Performance Comparison ---" << endl;
    
    // Measure copy vs move performance
    HeavyObject largeObj(2000000, "LargeObject");
    
    cout << "\nCopying large object..." << endl;
    HeavyObject copied = largeObj;  // Expensive copy
    
    cout << "\nMoving large object..." << endl;
    HeavyObject moved = move(largeObj);  // Cheap move
    
    cout << "\nProgram ending - all objects will be properly destroyed" << endl;
    
    return 0;
}
```

## Const-Correctness

### Exercise 3: Const-Correctness

Complete this const-correctness example:

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

class ConstCorrectClass {
private:
    vector<int> data;
    mutable int accessCount;  // mutable allows modification even in const methods
    
public:
    explicit ConstCorrectClass(const vector<int>& initialData) 
        : data(initialData), accessCount(0) {}
    
    // Const member functions - don't modify object state
    size_t size() const {
        ++accessCount;  // OK: mutable member can be modified in const function
        return data.size();
    }
    
    bool empty() const {
        ++accessCount;
        return data.empty();
    }
    
    // Const accessor
    const int& operator[](size_t index) const {
        ++accessCount;
        return data[index];
    }
    
    // Non-const accessor (for modification)
    int& operator[](size_t index) {
        ++accessCount;
        return data[index];
    }
    
    // Const begin/end for const objects
    const vector<int>::const_iterator begin() const {
        ++accessCount;
        return data.begin();
    }
    
    const vector<int>::const_iterator end() const {
        ++accessCount;
        return data.end();
    }
    
    // Non-const begin/end
    vector<int>::iterator begin() {
        ++accessCount;
        return data.begin();
    }
    
    vector<int>::iterator end() {
        ++accessCount;
        return data.end();
    }
    
    // Const method that returns const reference
    const vector<int>& getData() const {
        ++accessCount;
        return data;
    }
    
    // Non-const method that returns non-const reference
    vector<int>& getData() {
        ++accessCount;
        return data;
    }
    
    int getAccessCount() const { return accessCount; }
    
    // Method that should be const but modifies state (ERROR)
    void printStatus() const {
        // This is problematic - printing shouldn't modify state
        // But we want to update access count
        ++accessCount;  // OK: mutable member
        cout << "Object has " << data.size() << " elements" << endl;
    }
};

// Function that takes const reference - can accept both const and non-const
void processConst(const ConstCorrectClass& obj) {
    cout << "Processing const object:" << endl;
    cout << "Size: " << obj.size() << endl;
    cout << "First element: " << obj[0] << endl;
    cout << "Access count: " << obj.getAccessCount() << endl;
}

// Function that takes non-const reference - only accepts non-const
void processNonConst(ConstCorrectClass& obj) {
    cout << "Processing non-const object:" << endl;
    cout << "Size: " << obj.size() << endl;
    obj[0] = 999;  // Can modify
    cout << "Modified first element to: " << obj[0] << endl;
}

// Template function demonstrating const-correctness
template<typename Container>
void printContainer(const Container& container) {
    cout << "Container contents: ";
    for (const auto& element : container) {
        cout << element << " ";
    }
    cout << endl;
}

// Function that demonstrates const_cast (use sparingly)
void demonstrateConstCast() {
    const int constValue = 42;
    // int& ref = constValue;  // Error: cannot convert const int to int&
    
    // Sometimes you need to remove const (be very careful!)
    const int* constPtr = &constValue;
    int* mutablePtr = const_cast<int*>(constPtr);  // Removes const
    cout << "Value through const_cast: " << *mutablePtr << endl;
    
    // Better approach: use mutable or redesign
    cout << "Better approaches avoid const_cast when possible" << endl;
}

int main() {
    cout << "=== Const-Correctness Demo ===" << endl;
    
    // Create const and non-const objects
    ConstCorrectClass nonConstObj({1, 2, 3, 4, 5});
    const ConstCorrectClass constObj({10, 20, 30, 40, 50});
    
    cout << "\n--- Non-const object ---" << endl;
    cout << "Size: " << nonConstObj.size() << endl;
    cout << "First element: " << nonConstObj[0] << endl;
    
    // Modify through non-const reference
    nonConstObj[0] = 100;
    cout << "After modification, first element: " << nonConstObj[0] << endl;
    
    cout << "\n--- Const object ---" << endl;
    cout << "Size: " << constObj.size() << endl;
    cout << "First element: " << constObj[0] << endl;
    // constObj[0] = 200;  // Error: cannot modify const object
    
    cout << "\n--- Function calls ---" << endl;
    
    // Both const and non-const objects can be passed to function taking const reference
    processConst(constObj);
    processConst(nonConstObj);
    
    // Only non-const objects can be passed to function taking non-const reference
    processNonConst(nonConstObj);
    // processNonConst(constObj);  // Error: cannot bind const object to non-const reference
    
    cout << "\n--- Container iteration ---" << endl;
    
    // Iterate with const correctness
    cout << "Const iteration:" << endl;
    for (const auto& element : constObj) {
        cout << element << " ";
    }
    cout << endl;
    
    cout << "Non-const iteration (allows modification):" << endl;
    for (auto& element : nonConstObj.getData()) {  // Non-const reference
        element *= 2;  // Can modify
    }
    
    cout << "After modification:" << endl;
    for (const auto& element : nonConstObj) {
        cout << element << " ";
    }
    cout << endl;
    
    cout << "\n--- Template function ---" << endl;
    
    vector<int> vec = {1, 2, 3, 4, 5};
    printContainer(vec);
    
    const vector<string> strVec = {"hello", "world", "cpp"};
    printContainer(strVec);
    
    cout << "\n--- Const-correct algorithms ---" << endl;
    
    vector<int> numbers = {5, 2, 8, 1, 9, 3};
    
    // Use const-correct algorithms
    auto minIt = min_element(numbers.begin(), numbers.end());
    auto maxIt = max_element(numbers.begin(), numbers.end());
    
    cout << "Min: " << *minIt << ", Max: " << *maxIt << endl;
    
    // Count with const-correct predicate
    int evenCount = count_if(numbers.begin(), numbers.end(),
                            [](int n) { return n % 2 == 0; });
    cout << "Even numbers: " << evenCount << endl;
    
    cout << "\n--- Demonstrating const_cast ---" << endl;
    demonstrateConstCast();
    
    cout << "\n--- Access counts ---" << endl;
    cout << "Const object access count: " << constObj.getAccessCount() << endl;
    cout << "Non-const object access count: " << nonConstObj.getAccessCount() << endl;
    
    return 0;
}
```

## Design Patterns in C++

### Exercise 4: Common Design Patterns

Complete this design patterns example:

```cpp
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <mutex>
using namespace std;

// Singleton pattern
class Logger {
private:
    static unique_ptr<Logger> instance;
    static mutex mtx;
    
    Logger() = default;  // Private constructor
    
public:
    // Delete copy constructor and assignment to enforce singleton
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    static Logger* getInstance() {
        lock_guard<mutex> lock(mtx);
        if (!instance) {
            instance = make_unique<Logger>();
        }
        return instance.get();
    }
    
    void log(const string& message) {
        cout << "[LOG] " << message << endl;
    }
};

// Initialize static members
unique_ptr<Logger> Logger::instance = nullptr;
mutex Logger::mtx;

// Factory pattern
class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw() const = 0;
    virtual double area() const = 0;
};

class Circle : public Shape {
private:
    double radius;
    
public:
    Circle(double r) : radius(r) {}
    
    void draw() const override {
        cout << "Drawing Circle with radius " << radius << endl;
    }
    
    double area() const override {
        return 3.14159 * radius * radius;
    }
};

class Rectangle : public Shape {
private:
    double width, height;
    
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    
    void draw() const override {
        cout << "Drawing Rectangle " << width << "x" << height << endl;
    }
    
    double area() const override {
        return width * height;
    }
};

class ShapeFactory {
public:
    enum class ShapeType {
        CIRCLE,
        RECTANGLE
    };
    
    static unique_ptr<Shape> createShape(ShapeType type, double param1, double param2 = 0) {
        switch (type) {
            case ShapeType::CIRCLE:
                return make_unique<Circle>(param1);
            case ShapeType::RECTANGLE:
                return make_unique<Rectangle>(param1, param2);
            default:
                throw invalid_argument("Unknown shape type");
        }
    }
};

// Observer pattern
class Observer {
public:
    virtual ~Observer() = default;
    virtual void update(const string& message) = 0;
};

class Subject {
private:
    vector<weak_ptr<Observer>> observers;  // Use weak_ptr to avoid circular references
    
public:
    void attach(shared_ptr<Observer> observer) {
        observers.push_back(observer);
    }
    
    void notify(const string& message) {
        // Remove expired observers and notify active ones
        auto it = observers.begin();
        while (it != observers.end()) {
            if (auto obs = it->lock()) {
                obs->update(message);
                ++it;
            } else {
                it = observers.erase(it);  // Remove expired observer
            }
        }
    }
};

class NewsAgency : public Subject {
private:
    string news;
    
public:
    void setNews(const string& newNews) {
        news = newNews;
        notify(news);
    }
    
    const string& getNews() const { return news; }
};

class NewsChannel : public Observer {
private:
    string channelName;
    
public:
    NewsChannel(const string& name) : channelName(name) {}
    
    void update(const string& message) override {
        cout << channelName << " received news: " << message << endl;
    }
};

// Strategy pattern
class SortStrategy {
public:
    virtual ~SortStrategy() = default;
    virtual void sort(vector<int>& data) const = 0;
};

class BubbleSort : public SortStrategy {
public:
    void sort(vector<int>& data) const override {
        cout << "Sorting with Bubble Sort" << endl;
        size_t n = data.size();
        for (size_t i = 0; i < n - 1; i++) {
            for (size_t j = 0; j < n - i - 1; j++) {
                if (data[j] > data[j + 1]) {
                    swap(data[j], data[j + 1]);
                }
            }
        }
    }
};

class QuickSort : public SortStrategy {
public:
    void sort(vector<int>& data) const override {
        cout << "Sorting with Quick Sort (simplified)" << endl;
        sort(data.begin(), data.end());  // Using std::sort as proxy for quicksort
    }
};

class SortContext {
private:
    unique_ptr<SortStrategy> strategy;
    
public:
    void setStrategy(unique_ptr<SortStrategy> newStrategy) {
        strategy = move(newStrategy);
    }
    
    void executeSort(vector<int>& data) {
        if (strategy) {
            strategy->sort(data);
        }
    }
};

int main() {
    cout << "=== Design Patterns Demo ===" << endl;
    
    // Singleton pattern
    cout << "\n--- Singleton Pattern ---" << endl;
    Logger* logger1 = Logger::getInstance();
    Logger* logger2 = Logger::getInstance();
    
    cout << "Same instance? " << (logger1 == logger2 ? "Yes" : "No") << endl;
    logger1->log("This is a singleton log message");
    
    // Factory pattern
    cout << "\n--- Factory Pattern ---" << endl;
    
    auto circle = ShapeFactory::createShape(ShapeFactory::ShapeType::CIRCLE, 5.0);
    auto rectangle = ShapeFactory::createShape(ShapeFactory::ShapeType::RECTANGLE, 4.0, 6.0);
    
    circle->draw();
    cout << "Circle area: " << circle->area() << endl;
    
    rectangle->draw();
    cout << "Rectangle area: " << rectangle->area() << endl;
    
    // Observer pattern
    cout << "\n--- Observer Pattern ---" << endl;
    
    NewsAgency agency;
    
    auto cnn = make_shared<NewsChannel>("CNN");
    auto bbc = make_shared<NewsChannel>("BBC");
    
    agency.attach(cnn);
    agency.attach(bbc);
    
    agency.setNews("Breaking: New C++ Standard Released!");
    
    // Strategy pattern
    cout << "\n--- Strategy Pattern ---" << endl;
    
    SortContext context;
    vector<int> data = {64, 34, 25, 12, 22, 11, 90};
    
    cout << "Original data: ";
    for (int x : data) cout << x << " ";
    cout << endl;
    
    // Use bubble sort
    context.setStrategy(make_unique<BubbleSort>());
    vector<int> bubbleData = data;  // Copy for bubble sort
    context.executeSort(bubbleData);
    
    cout << "After bubble sort: ";
    for (int x : bubbleData) cout << x << " ";
    cout << endl;
    
    // Use quick sort
    context.setStrategy(make_unique<QuickSort>());
    vector<int> quickData = data;  // Copy for quick sort
    context.executeSort(quickData);
    
    cout << "After quick sort: ";
    for (int x : quickData) cout << x << " ";
    cout << endl;
    
    // Command pattern simulation using function objects
    cout << "\n--- Command Pattern (Function Objects) ---" << endl;
    
    vector<function<void()>> commands;
    
    commands.push_back([&agency]() { 
        agency.setNews("Technology Update: C++23 Features Announced");
    });
    
    commands.push_back([circle = circle.get()]() {  // Capture shape to use
        circle->draw();
    });
    
    commands.push_back([logger1]() {
        logger1->log("Command pattern demonstration completed");
    });
    
    cout << "Executing commands:" << endl;
    for (auto& cmd : commands) {
        cmd();
    }
    
    // RAII pattern (already demonstrated throughout)
    cout << "\n--- RAII Pattern ---" << endl;
    {
        auto resource = make_unique<string>("RAII Resource");
        cout << "Resource created: " << *resource << endl;
        // Resource automatically destroyed when going out of scope
    }
    cout << "Resource automatically cleaned up" << endl;
    
    cout << "\nAll design patterns demonstrated successfully!" << endl;
    
    return 0;
}
```

## Performance Optimization Techniques

### Exercise 5: Performance Optimization

Complete this performance optimization example:

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <memory>
#include <numeric>
using namespace std;
using namespace std::chrono;

// Class to demonstrate efficient string handling
class EfficientStringProcessor {
public:
    // Bad: Creating unnecessary copies
    string processStringBad(string input) {  // Pass by value - creates copy!
        // Process the string
        for (auto& c : input) {
            c = toupper(c);
        }
        return input;  // Return by value - another copy!
    }
    
    // Good: Pass by const reference, return by value (NRVO/RVO)
    string processStringGood(const string& input) {  // Pass by const reference - no copy!
        string result = input;  // Copy only once
        for (auto& c : result) {
            c = toupper(c);
        }
        return result;  // NRVO/RVO may eliminate this copy
    }
    
    // Better: Use string_view for read-only operations (C++17)
    string processStringView(string_view input) {  // string_view is lightweight
        string result;  // Start with empty string
        result.reserve(input.length());  // Reserve space to avoid reallocations
        
        for (char c : input) {
            result += static_cast<char>(toupper(c));
        }
        return result;
    }
    
    // Best: In-place modification when possible
    void processStringInPlace(string& input) {  // Modify in place
        for (auto& c : input) {
            c = static_cast<char>(toupper(c));
        }
    }
};

// Demonstrating efficient container usage
class ContainerPerformance {
public:
    // Inefficient: repeated reallocations
    vector<int> createVectorInefficient(size_t size) {
        vector<int> result;
        for (size_t i = 0; i < size; i++) {
            result.push_back(static_cast<int>(i));  // May cause reallocations
        }
        return result;
    }
    
    // Efficient: reserve space upfront
    vector<int> createVectorEfficient(size_t size) {
        vector<int> result;
        result.reserve(size);  // Reserve space to avoid reallocations
        for (size_t i = 0; i < size; i++) {
            result.push_back(static_cast<int>(i));
        }
        return result;
    }
    
    // Most efficient: construct with size
    vector<int> createVectorMostEfficient(size_t size) {
        vector<int> result(size);  // Create with size, default initialize
        iota(result.begin(), result.end(), 0);  // Fill with sequence
        return result;
    }
    
    // Efficient: use emplace instead of push when possible
    vector<pair<string, int>> createPairsEfficient(size_t count) {
        vector<pair<string, int>> result;
        result.reserve(count);
        
        for (size_t i = 0; i < count; i++) {
            result.emplace_back("Item" + to_string(i), static_cast<int>(i));  // Construct in place
            // vs push_back(pair<string, int>("Item" + to_string(i), i)) - creates temporary
        }
        return result;
    }
    
    // Efficient: use algorithms instead of manual loops
    void processWithAlgorithm(vector<int>& data) {
        // Good: use algorithm
        transform(data.begin(), data.end(), data.begin(),
                  [](int x) { return x * 2; });
        
        // Bad: manual loop (less efficient, more error-prone)
        /*
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = data[i] * 2;
        }
        */
    }
};

// Move semantics for performance
class PerformanceClass {
private:
    vector<int> data;
    
public:
    explicit PerformanceClass(size_t size) : data(size) {
        iota(data.begin(), data.end(), 1);  // Fill with 1, 2, 3, ...
    }
    
    // Copy constructor - expensive
    PerformanceClass(const PerformanceClass& other) : data(other.data) {
        cout << "Expensive copy constructor called" << endl;
    }
    
    // Move constructor - cheap
    PerformanceClass(PerformanceClass&& other) noexcept : data(move(other.data)) {
        cout << "Cheap move constructor called" << endl;
    }
    
    // Copy assignment - expensive
    PerformanceClass& operator=(const PerformanceClass& other) {
        cout << "Expensive copy assignment called" << endl;
        if (this != &other) {
            data = other.data;
        }
        return *this;
    }
    
    // Move assignment - cheap
    PerformanceClass& operator=(PerformanceClass&& other) noexcept {
        cout << "Cheap move assignment called" << endl;
        if (this != &other) {
            data = move(other.data);
        }
        return *this;
    }
    
    const vector<int>& getData() const { return data; }
    size_t size() const { return data.size(); }
};

// Performance measurement utility
template<typename Func>
auto measureTime(Func&& func) -> decltype(func()) {
    auto start = high_resolution_clock::now();
    auto result = func();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "Execution time: " << duration.count() << " microseconds" << endl;
    return result;
}

int main() {
    cout << "=== Performance Optimization Demo ===" << endl;
    
    // String processing performance
    cout << "\n--- String Processing Performance ---" << endl;
    
    EfficientStringProcessor processor;
    string testString(10000, 'a');  // Large string of 'a's
    
    cout << "Processing string of size: " << testString.length() << endl;
    
    // Measure inefficient approach
    cout << "Inefficient approach:" << endl;
    auto result1 = measureTime([&]() {
        return processor.processStringBad(testString);
    });
    
    // Measure efficient approach
    cout << "Efficient approach:" << endl;
    auto result2 = measureTime([&]() {
        return processor.processStringGood(testString);
    });
    
    // Measure string_view approach
    cout << "String view approach:" << endl;
    auto result3 = measureTime([&]() {
        return processor.processStringView(testString);
    });
    
    // Container performance
    cout << "\n--- Container Performance ---" << endl;
    
    ContainerPerformance perfTest;
    size_t testSize = 100000;
    
    cout << "Creating vector with " << testSize << " elements:" << endl;
    
    cout << "Inefficient (with reallocations):" << endl;
    auto vec1 = measureTime([&]() {
        return perfTest.createVectorInefficient(testSize);
    });
    
    cout << "Efficient (with reserve):" << endl;
    auto vec2 = measureTime([&]() {
        return perfTest.createVectorEfficient(testSize);
    });
    
    cout << "Most efficient (construct with size):" << endl;
    auto vec3 = measureTime([&]() {
        return perfTest.createVectorMostEfficient(testSize);
    });
    
    cout << "Vector sizes: " << vec1.size() << ", " << vec2.size() << ", " << vec3.size() << endl;
    
    // Move semantics performance
    cout << "\n--- Move Semantics Performance ---" << endl;
    
    PerformanceClass largeObj(500000);  // Large object
    cout << "Created large object with " << largeObj.size() << " elements" << endl;
    
    cout << "\nCopy operation:" << endl;
    auto copiedObj = measureTime([&]() {
        return PerformanceClass(largeObj);  // Copy constructor
    });
    
    cout << "\nMove operation:" << endl;
    auto movedObj = measureTime([&]() {
        return PerformanceClass(move(largeObj));  // Move constructor
    });
    
    cout << "After move, original size: " << largeObj.size() << endl;
    cout << "Moved object size: " << movedObj.size() << endl;
    
    // Algorithm vs manual loop performance
    cout << "\n--- Algorithm vs Manual Loop ---" << endl;
    
    vector<int> algoTest(100000);
    iota(algoTest.begin(), algoTest.end(), 1);
    
    cout << "Using transform algorithm:" << endl;
    auto algoResult = algoTest;
    measureTime([&]() {
        perfTest.processWithAlgorithm(algoResult);
    });
    
    // Memory pool simulation for performance
    cout << "\n--- Memory Pool Simulation ---" << endl;
    
    // Instead of allocating many small objects individually (slow)
    // Use a container that manages them efficiently
    vector<unique_ptr<PerformanceClass>> objectContainer;
    objectContainer.reserve(1000);  // Reserve space for efficiency
    
    cout << "Creating 1000 objects:" << endl;
    measureTime([&]() {
        for (int i = 0; i < 1000; i++) {
            objectContainer.push_back(make_unique<PerformanceClass>(1000));
        }
    });
    
    cout << "Created " << objectContainer.size() << " objects efficiently" << endl;
    
    // Cache-friendly data access
    cout << "\n--- Cache-Friendly Access ---" << endl;
    
    // Poor cache locality: accessing data in column-major order
    const size_t rows = 1000, cols = 1000;
    vector<vector<int>> matrix(rows, vector<int>(cols));
    
    cout << "Poor cache locality (column-major access):" << endl;
    measureTime([&]() {
        for (size_t j = 0; j < cols; j++) {  // Outer loop on columns
            for (size_t i = 0; i < rows; i++) {  // Inner loop on rows
                matrix[i][j] = static_cast<int>(i * cols + j);
            }
        }
    });
    
    // Good cache locality: accessing data in row-major order
    cout << "Good cache locality (row-major access):" << endl;
    measureTime([&]() {
        for (size_t i = 0; i < rows; i++) {  // Outer loop on rows
            for (size_t j = 0; j < cols; j++) {  // Inner loop on columns
                matrix[i][j] = static_cast<int>(i * cols + j);
            }
        }
    });
    
    cout << "\nPerformance optimizations completed!" << endl;
    
    return 0;
}
```

## Modern C++ Best Practices

### Exercise 6: Modern C++ Best Practices

Demonstrate modern C++ best practices:

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <type_traits>
#include <functional>
using namespace std;

// Best Practice 1: Use auto for complex types
class ModernBestPractices {
public:
    void demonstrateAuto() {
        // Old way - verbose and error-prone
        vector<pair<string, unique_ptr<int>>>::iterator oldIt;
        
        // Modern way - clear and maintainable
        vector<pair<string, unique_ptr<int>>> container;
        auto modernIt = container.begin();
        
        // Even better - use cbegin/cend for const correctness
        auto constIt = container.cbegin();
        
        cout << "auto makes code more readable and maintainable" << endl;
    }
    
    // Best Practice 2: Use range-based for loops
    void demonstrateRangeBasedLoops() {
        vector<int> numbers = {1, 2, 3, 4, 5};
        
        // Old way
        for (vector<int>::const_iterator it = numbers.begin(); 
             it != numbers.end(); ++it) {
            cout << *it << " ";
        }
        cout << endl;
        
        // Modern way
        for (const auto& num : numbers) {
            cout << num << " ";
        }
        cout << endl;
        
        // For modification
        for (auto& num : numbers) {
            num *= 2;
        }
        
        cout << "After modification: ";
        for (const auto& num : numbers) {
            cout << num << " ";
        }
        cout << endl;
    }
    
    // Best Practice 3: Use make_unique and make_shared
    void demonstrateMakeFunctions() {
        // Old way - potential exception safety issue
        // unique_ptr<MyClass> ptr(new MyClass(args));  // If MyClass constructor throws, memory leaks
        
        // Modern way - exception safe
        auto safePtr = make_unique<string>("Safely created");
        auto sharedPtr = make_shared<string>("Also safely created");
        
        cout << "Using make_* functions is exception-safe" << endl;
        cout << "Unique: " << *safePtr << ", Shared: " << *sharedPtr << endl;
    }
    
    // Best Practice 4: Use nullptr instead of NULL or 0
    void demonstrateNullptr() {
        int* ptr1 = nullptr;  // Clear, type-safe
        // int* ptr2 = NULL;    // Could be int 0 in some contexts
        // int* ptr3 = 0;       // Integer 0, not pointer
        
        if (ptr1 == nullptr) {
            cout << "Using nullptr is type-safe and clear" << endl;
        }
    }
    
    // Best Practice 5: Use override and final
    class Base {
    public:
        virtual void method() { cout << "Base method" << endl; }
        virtual ~Base() = default;
    };
    
    class Derived : public Base {
    public:
        void method() override { cout << "Derived method" << endl; }  // Clear intent
    };
    
    void demonstrateOverride() {
        Base* basePtr = new Derived();
        basePtr->method();  // Calls Derived::method() due to virtual dispatch
        delete basePtr;
    }
    
    // Best Practice 6: Use constexpr when possible
    static constexpr int square(int x) {
        return x * x;
    }
    
    void demonstrateConstexpr() {
        constexpr int compileTimeResult = square(5);  // Computed at compile time
        cout << "Compile-time computation: " << compileTimeResult << endl;
        
        int runtimeValue = 7;
        int runtimeResult = square(runtimeValue);  // Computed at runtime
        cout << "Runtime computation: " << runtimeResult << endl;
    }
    
    // Best Practice 7: Use [[nodiscard]] for important return values
    [[nodiscard]] bool importantOperation() {
        cout << "Performing important operation" << endl;
        return true;
    }
    
    void demonstrateNodiscard() {
        importantOperation();  // Compiler warning: ignoring return value
        // bool result = importantOperation();  // Better: use the result
    }
    
    // Best Practice 8: Use [[maybe_unused]] to suppress warnings
    void demonstrateMaybeUnused([[maybe_unused]] int unusedParam) {
        cout << "Using maybe_unused to suppress warnings" << endl;
        // unusedParam is intentionally unused
    }
};

// Best Practice 9: Use type traits and SFINAE
template<typename T>
void processValue(T&& value) {
    if constexpr (is_arithmetic_v<decay_t<T>>) {
        cout << "Processing arithmetic value: " << value << endl;
    } else if constexpr (is_same_v<decay_t<T>, string>) {
        cout << "Processing string: " << value << endl;
    } else {
        cout << "Processing other type" << endl;
    }
}

// Best Practice 10: Use structured bindings (C++17)
void demonstrateStructuredBindings() {
    pair<string, int> person = {"Alice", 30};
    
    // Old way
    string name = person.first;
    int age = person.second;
    
    // Modern way
    auto [personName, personAge] = person;
    cout << "Structured binding: " << personName << " is " << personAge << " years old" << endl;
    
    // With containers
    vector<pair<string, int>> people = {{"Bob", 25}, {"Charlie", 35}};
    cout << "People in vector:" << endl;
    for (const auto& [name, age] : people) {
        cout << "  " << name << " is " << age << " years old" << endl;
    }
}

int main() {
    cout << "=== Modern C++ Best Practices ===" << endl;
    
    ModernBestPractices practices;
    
    cout << "\n--- Auto Demo ---" << endl;
    practices.demonstrateAuto();
    
    cout << "\n--- Range-based For Loop Demo ---" << endl;
    practices.demonstrateRangeBasedLoops();
    
    cout << "\n--- Make Functions Demo ---" << endl;
    practices.demonstrateMakeFunctions();
    
    cout << "\n--- nullptr Demo ---" << endl;
    practices.demonstrateNullptr();
    
    cout << "\n--- Override Demo ---" << endl;
    practices.demonstrateOverride();
    
    cout << "\n--- constexpr Demo ---" << endl;
    practices.demonstrateConstexpr();
    
    cout << "\n--- [[nodiscard]] Demo ---" << endl;
    practices.demonstrateNodiscard();
    
    cout << "\n--- [[maybe_unused]] Demo ---" << endl;
    practices.demonstrateMaybeUnused(42);
    
    cout << "\n--- Type Traits Demo ---" << endl;
    processValue(42);
    processValue(3.14);
    processValue(string("Hello"));
    
    cout << "\n--- Structured Bindings Demo ---" << endl;
    practices.demonstrateStructuredBindings();
    
    // Best Practice 11: Use RAII and smart pointers
    cout << "\n--- RAII and Smart Pointers ---" << endl;
    
    {
        auto resource = make_unique<vector<int>>(1000000, 42);
        cout << "Resource created with " << resource->size() << " elements" << endl;
        // Resource automatically cleaned up when going out of scope
    }
    cout << "Resource automatically cleaned up" << endl;
    
    // Best Practice 12: Use algorithms instead of manual loops
    cout << "\n--- STL Algorithms ---" << endl;
    
    vector<int> numbers = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    
    cout << "Original: ";
    for (const auto& num : numbers) cout << num << " ";
    cout << endl;
    
    // Sort using algorithm
    sort(numbers.begin(), numbers.end());
    cout << "Sorted: ";
    for (const auto& num : numbers) cout << num << " ";
    cout << endl;
    
    // Find using algorithm
    auto it = find(numbers.begin(), numbers.end(), 7);
    if (it != numbers.end()) {
        cout << "Found 7 at position: " << (it - numbers.begin()) << endl;
    }
    
    // Count using algorithm
    int evenCount = count_if(numbers.begin(), numbers.end(),
                            [](int n) { return n % 2 == 0; });
    cout << "Even numbers: " << evenCount << endl;
    
    // Transform using algorithm
    transform(numbers.begin(), numbers.end(), numbers.begin(),
              [](int n) { return n * n; });
    cout << "Squared: ";
    for (const auto& num : numbers) cout << num << " ";
    cout << endl;
    
    // Best Practice 13: Use const and constexpr appropriately
    cout << "\n--- Const and Constexpr ---" << endl;
    
    const vector<int> constNumbers = {1, 2, 3, 4, 5};
    // constNumbers.push_back(6);  // Error: can't modify const container
    
    constexpr int compileTimeValue = 42;  // Computed at compile time
    const int runtimeValue = 42;          // Computed at runtime
    
    cout << "Compile-time value: " << compileTimeValue << endl;
    cout << "Runtime value: " << runtimeValue << endl;
    
    // Best Practice 14: Use move semantics appropriately
    cout << "\n--- Move Semantics ---" << endl;
    
    vector<string> source = {"hello", "world", "cpp", "modern"};
    vector<string> destination;
    
    cout << "Before move:" << endl;
    cout << "Source size: " << source.size() << endl;
    
    // Move elements efficiently
    for (auto& str : source) {
        destination.push_back(move(str));  // Move each string efficiently
    }
    
    cout << "After move:" << endl;
    cout << "Source size: " << source.size() << endl;
    cout << "Destination: ";
    for (const auto& str : destination) {
        cout << str << " ";
    }
    cout << endl;
    
    cout << "\nAll modern C++ best practices demonstrated!" << endl;
    
    return 0;
}
```

## Coding Standards and Guidelines

### Exercise 7: Coding Standards

Demonstrate proper coding standards and guidelines:

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
using namespace std;

// Good: Proper naming conventions
class UserManager {
private:
    // Good: Descriptive names
    vector<unique_ptr<string>> userNames;
    size_t maxUsers;
    
    // Good: Private helper function with descriptive name
    bool isValidUsername(const string& username) const {
        if (username.empty() || username.length() > 50) {
            return false;
        }
        
        // Check if username contains only valid characters
        for (char c : username) {
            if (!isalnum(c) && c != '_') {
                return false;
            }
        }
        return true;
    }
    
public:
    // Good: Constructor with explicit keyword for single-argument constructors
    explicit UserManager(size_t max) : maxUsers(max) {
        if (max == 0) {
            throw invalid_argument("Maximum users cannot be zero");
        }
    }
    
    // Good: Descriptive function name
    bool addUser(const string& username) {
        if (!isValidUsername(username)) {
            throw invalid_argument("Invalid username format");
        }
        
        if (userNames.size() >= maxUsers) {
            throw runtime_error("Maximum number of users reached");
        }
        
        // Good: Use emplace_back for efficiency
        userNames.emplace_back(make_unique<string>(username));
        return true;
    }
    
    // Good: Const member function when not modifying state
    size_t getUserCount() const {
        return userNames.size();
    }
    
    // Good: Return by const reference to avoid copying
    const vector<unique_ptr<string>>& getUsers() const {
        return userNames;
    }
    
    // Good: Proper error handling
    const string& getUser(size_t index) const {
        if (index >= userNames.size()) {
            throw out_of_range("User index out of bounds");
        }
        return *userNames[index];
    }
    
    // Good: Use override for virtual functions
    virtual ~UserManager() = default;
};

// Good: Namespace for related functionality
namespace Utilities {
    // Good: Template function with descriptive name
    template<typename Container, typename Predicate>
    size_t countIf(const Container& container, Predicate pred) {
        size_t count = 0;
        for (const auto& element : container) {
            if (pred(element)) {
                ++count;
            }
        }
        return count;
    }
    
    // Good: Function with clear, descriptive name
    string formatErrorMessage(const string& operation, const string& reason) {
        return "Error during " + operation + ": " + reason;
    }
}

// Good: Exception class following naming convention
class UserManagementException : public runtime_error {
public:
    explicit UserManagementException(const string& message) 
        : runtime_error(message) {}
};

int main() {
    cout << "=== Coding Standards and Guidelines Demo ===" << endl;
    
    try {
        // Good: Descriptive variable names
        const size_t maximumUsers = 100;
        UserManager userManager(maximumUsers);
        
        // Good: Meaningful variable names
        const vector<string> testUsernames = {
            "alice_smith",
            "bob_jones", 
            "charlie_brown",
            "diana_prince"
        };
        
        // Good: Clear, descriptive loop
        for (const auto& username : testUsernames) {
            if (userManager.addUser(username)) {
                cout << "Successfully added user: " << username << endl;
            }
        }
        
        cout << "Total users: " << userManager.getUserCount() << endl;
        
        // Good: Use range-based for loop with const auto&
        const auto& users = userManager.getUsers();
        cout << "Users in system:" << endl;
        for (const auto& user : users) {
            cout << "  - " << *user << endl;
        }
        
        // Good: Proper error handling
        try {
            userManager.getUser(1000);  // This will throw
        } catch (const out_of_range& e) {
            cout << "Caught expected error: " << e.what() << endl;
        }
        
        // Good: Using utility functions
        auto evenCounter = [](int n) { return n % 2 == 0; };
        vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        size_t evenCount = Utilities::countIf(numbers, evenCounter);
        cout << "Even numbers in range 1-10: " << evenCount << endl;
        
        // Good: Proper exception usage
        throw UserManagementException("Demonstrating custom exception");
        
    } catch (const UserManagementException& e) {
        cout << "Caught custom exception: " << e.what() << endl;
    } catch (const exception& e) {
        cout << "Caught standard exception: " << e.what() << endl;
    }
    
    // Good: Consistent formatting and indentation
    cout << "\n=== Formatting and Style ===" << endl;
    
    // Good: Consistent brace style
    if (true) {
        cout << "Consistent formatting improves readability" << endl;
    } else {
        cout << "This won't print" << endl;
    }
    
    // Good: Proper spacing around operators
    int value1 = 10;
    int value2 = 20;
    int result = value1 + value2;
    
    // Good: Use of whitespace for readability
    vector<int> spacedVector = {1, 2, 3, 4, 5};
    
    cout << "Calculated result: " << result << endl;
    
    // Good: Comments for complex logic
    // Calculate the average, handling potential division by zero
    if (!spacedVector.empty()) {
        int sum = 0;
        for (int num : spacedVector) {
            sum += num;
        }
        double average = static_cast<double>(sum) / spacedVector.size();
        cout << "Average: " << average << endl;
    }
    
    cout << "\nCoding standards demonstration completed successfully!" << endl;
    
    return 0;
}
```

## Summary

In this chapter, you learned:
- The Rule of Zero, Three, and Five for special member functions
- Move semantics and perfect forwarding
- Const-correctness and its importance
- Common design patterns in C++
- Performance optimization techniques
- Modern C++ best practices and coding standards
- Memory management best practices
- Exception safety guidelines

## Key Takeaways

- Prefer the Rule of Zero when possible (let compiler generate special functions)
- Use move semantics for performance when transferring ownership
- Apply const-correctness consistently throughout your code
- Use smart pointers for automatic memory management
- Follow modern C++ idioms like auto, range-based for loops, and make functions
- Use algorithms instead of manual loops when possible
- Apply design patterns appropriately to solve common problems
- Optimize for performance while maintaining code clarity
- Follow consistent coding standards and naming conventions

## Common Mistakes to Avoid

1. Implementing unnecessary special member functions (violating Rule of Zero)
2. Forgetting to use move semantics when transferring ownership
3. Not applying const-correctness consistently
4. Using raw pointers for ownership instead of smart pointers
5. Not following RAII principles
6. Using NULL or 0 instead of nullptr
7. Forgetting to use override for virtual functions
8. Not handling exceptions properly
9. Writing inefficient code when STL algorithms would be better
10. Inconsistent naming and formatting

## Next Steps

Now that you understand advanced topics and best practices, you're ready to learn about concurrency and multithreading in Chapter 14.