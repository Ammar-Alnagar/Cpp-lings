# Chapter 11: Memory Management and Smart Pointers

## Overview

This chapter covers memory management in C++, including both traditional raw pointer management and modern smart pointer techniques. You'll learn about the RAII principle, different types of smart pointers, and best practices for memory management.

## Learning Objectives

By the end of this chapter, you will:
- Understand the different types of memory in C++ (stack, heap, static)
- Master manual memory management with new/delete
- Learn about memory leaks and dangling pointers
- Understand the RAII (Resource Acquisition Is Initialization) principle
- Master unique_ptr, shared_ptr, and weak_ptr
- Learn about custom deleters and allocators
- Understand the differences between scoped_ptr and unique_ptr
- Learn best practices for memory management
- Understand move semantics in the context of smart pointers

## Memory Layout in C++

C++ programs have different areas of memory: stack, heap, and static storage.

### Exercise 1: Memory Layout

The following code has errors. Find and fix them:

```cpp
#include <iostream>
#include <string>
using namespace std;

// Global variable - static storage duration
int globalVar = 10;

void demonstrateMemoryLayout() {
    // Local variable - stored on stack
    int stackVar = 20;
    
    // Dynamic allocation - stored on heap
    int* heapVar = new int(30);  // Allocated on heap
    
    cout << "Stack variable: " << stackVar << endl;
    cout << "Heap variable: " << *heapVar << endl;
    cout << "Global variable: " << globalVar << endl;
    
    // Error: forgetting to delete heap-allocated memory
    // delete heapVar;  // Uncomment this line to fix the memory leak
    
    // After delete, heapVar becomes a dangling pointer
    // cout << "After delete: " << *heapVar << endl;  // Undefined behavior!
    
    // Safe way: set pointer to nullptr after delete
    delete heapVar;
    heapVar = nullptr;  // Prevent dangling pointer
}

class Resource {
private:
    int* data;
    size_t size;
    
public:
    // Constructor - acquire resource
    Resource(size_t s) : size(s) {
        data = new int[size];  // Acquire memory
        cout << "Resource acquired: " << size << " integers" << endl;
    }
    
    // Destructor - release resource
    ~Resource() {
        delete[] data;  // Release memory
        cout << "Resource released" << endl;
    }
    
    // Copy constructor
    Resource(const Resource& other) : size(other.size) {
        data = new int[size];
        for (size_t i = 0; i < size; i++) {
            data[i] = other.data[i];
        }
        cout << "Resource copied" << endl;
    }
    
    // Assignment operator
    Resource& operator=(const Resource& other) {
        if (this != &other) {
            delete[] data;  // Release old resource
            size = other.size;
            data = new int[size];
            for (size_t i = 0; i < size; i++) {
                data[i] = other.data[i];
            }
        }
        cout << "Resource assigned" << endl;
        return *this;
    }
    
    // Move constructor
    Resource(Resource&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;  // Transfer ownership
        other.size = 0;
        cout << "Resource moved" << endl;
    }
    
    // Move assignment operator
    Resource& operator=(Resource&& other) noexcept {
        if (this != &other) {
            delete[] data;  // Release current resource
            data = other.data;  // Transfer ownership
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        cout << "Resource move-assigned" << endl;
        return *this;
    }
    
    int& operator[](size_t index) { return data[index]; }
    const int& operator[](size_t index) const { return data[index]; }
    size_t getSize() const { return size; }
};

int main() {
    cout << "=== Memory Layout Demo ===" << endl;
    
    // Stack allocation
    Resource stackResource(5);
    stackResource[0] = 100;
    cout << "Stack resource first element: " << stackResource[0] << endl;
    
    // Heap allocation
    Resource* heapResource = new Resource(3);
    (*heapResource)[0] = 200;
    cout << "Heap resource first element: " << (*heapResource)[0] << endl;
    
    // Error: potential memory leak if exception occurs
    // delete heapResource;  // Should be called to prevent leak
    
    // Better approach: use RAII
    {
        Resource tempResource(2);
        tempResource[0] = 300;
        cout << "Temporary resource: " << tempResource[0] << endl;
        // Destructor automatically called when tempResource goes out of scope
    }
    
    // Clean up heap resource
    delete heapResource;  // Prevent memory leak
    heapResource = nullptr;
    
    demonstrateMemoryLayout();
    
    return 0;
}
```

## Manual Memory Management

Understanding manual memory management is crucial for appreciating smart pointers.

### Exercise 2: Manual Memory Management Patterns

Complete this manual memory management example with errors:

```cpp
#include <iostream>
#include <vector>
#include <string>
using namespace std;

class ManualMemoryExample {
private:
    int* data;
    size_t size;
    
public:
    // Constructor
    ManualMemoryExample(size_t s) : size(s) {
        data = new int[size];
        for (size_t i = 0; i < size; i++) {
            data[i] = i + 1;
        }
        cout << "ManualMemoryExample constructed with size " << size << endl;
    }
    
    // Destructor
    ~ManualMemoryExample() {
        cout << "ManualMemoryExample destructor called" << endl;
        delete[] data;  // Important: release memory
        data = nullptr; // Prevent dangling pointer
    }
    
    // Copy constructor - deep copy
    ManualMemoryExample(const ManualMemoryExample& other) : size(other.size) {
        data = new int[size];
        for (size_t i = 0; i < size; i++) {
            data[i] = other.data[i];  // Copy each element
        }
        cout << "ManualMemoryExample copied" << endl;
    }
    
    // Copy assignment operator
    ManualMemoryExample& operator=(const ManualMemoryExample& other) {
        if (this != &other) {  // Self-assignment check
            // Clean up existing resource
            delete[] data;
            
            // Copy from other
            size = other.size;
            data = new int[size];
            for (size_t i = 0; i < size; i++) {
                data[i] = other.data[i];
            }
        }
        cout << "ManualMemoryExample assigned" << endl;
        return *this;
    }
    
    // Move constructor
    ManualMemoryExample(ManualMemoryExample&& other) noexcept 
        : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
        cout << "ManualMemoryExample moved" << endl;
    }
    
    // Move assignment operator
    ManualMemoryExample& operator=(ManualMemoryExample&& other) noexcept {
        if (this != &other) {
            // Clean up existing resource
            delete[] data;
            
            // Transfer ownership
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        cout << "ManualMemoryExample move-assigned" << endl;
        return *this;
    }
    
    // Accessor methods
    int& operator[](size_t index) {
        if (index >= size) {
            throw out_of_range("Index out of bounds");
        }
        return data[index];
    }
    
    const int& operator[](size_t index) const {
        if (index >= size) {
            throw out_of_range("Index out of bounds");
        }
        return data[index];
    }
    
    size_t getSize() const { return size; }
    
    void print() const {
        cout << "Data: ";
        for (size_t i = 0; i < size; i++) {
            cout << data[i] << " ";
        }
        cout << endl;
    }
};

// Function that might cause memory leak
ManualMemoryExample* potentiallyLeakyFunction(bool shouldThrow) {
    ManualMemoryExample* obj = new ManualMemoryExample(10);
    
    if (shouldThrow) {
        throw runtime_error("Something went wrong!");  // Memory leak!
    }
    
    return obj;
}

int main() {
    cout << "=== Manual Memory Management Demo ===" << endl;
    
    // Basic usage
    ManualMemoryExample mme1(5);
    mme1.print();
    
    // Copy
    ManualMemoryExample mme2 = mme1;  // Copy constructor
    mme2.print();
    
    // Assignment
    ManualMemoryExample mme3(3);
    mme3 = mme1;  // Copy assignment
    mme3.print();
    
    // Move
    ManualMemoryExample mme4 = move(mme1);  // Move constructor
    mme4.print();
    // mme1 is now in a valid but unspecified state
    
    // Dynamic allocation
    ManualMemoryExample* dynamicObj = new ManualMemoryExample(4);
    dynamicObj->print();
    
    // Error: potential memory leak scenario
    try {
        ManualMemoryExample* leakyObj = potentiallyLeakyFunction(true);
        delete leakyObj;  // This line won't be reached
    } catch (const exception& e) {
        cout << "Caught exception: " << e.what() << endl;
        cout << "Memory was leaked!" << endl;
    }
    
    // Safe approach: RAII
    try {
        ManualMemoryExample safeObj(5);  // Automatically cleaned up
        if (true) {  // Simulate error condition
            throw runtime_error("Error occurred");
        }
        // This line won't be reached, but safeObj is still properly destroyed
    } catch (const exception& e) {
        cout << "Caught exception: " << e.what() << endl;
        cout << "But no memory was leaked!" << endl;
    }
    
    // Clean up dynamic allocation
    delete dynamicObj;
    dynamicObj = nullptr;
    
    return 0;
}
```

## RAII (Resource Acquisition Is Initialization)

RAII is a fundamental C++ idiom for resource management.

### Exercise 3: RAII Principle

Complete this RAII example:

```cpp
#include <iostream>
#include <fstream>
#include <memory>
using namespace std;

// RAII wrapper for file handling
class FileHandler {
private:
    fstream file;
    string filename;
    
public:
    explicit FileHandler(const string& fname) : filename(fname) {
        file.open(filename, ios::in | ios::out | ios::app);
        if (!file.is_open()) {
            throw runtime_error("Could not open file: " + filename);
        }
        cout << "File opened: " << filename << endl;
    }
    
    // Destructor automatically closes file
    ~FileHandler() {
        if (file.is_open()) {
            file.close();
            cout << "File closed: " << filename << endl;
        }
    }
    
    // Disable copy to prevent resource duplication
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;
    
    // Enable move
    FileHandler(FileHandler&& other) noexcept 
        : file(move(other.file)), filename(move(other.filename)) {
        cout << "File handler moved" << endl;
    }
    
    FileHandler& operator=(FileHandler&& other) noexcept {
        if (this != &other) {
            if (file.is_open()) {
                file.close();
            }
            file = move(other.file);
            filename = move(other.filename);
        }
        cout << "File handler move-assigned" << endl;
        return *this;
    }
    
    void write(const string& data) {
        if (file.is_open()) {
            file << data << endl;
        }
    }
    
    bool isOpen() const {
        return file.is_open();
    }
};

// RAII wrapper for dynamic memory
template<typename T>
class RAIIPtr {
private:
    T* ptr;
    
public:
    explicit RAIIPtr(T* p = nullptr) : ptr(p) {
        cout << "RAIIPtr acquired resource" << endl;
    }
    
    ~RAIIPtr() {
        delete ptr;
        cout << "RAIIPtr released resource" << endl;
    }
    
    // Disable copy
    RAIIPtr(const RAIIPtr&) = delete;
    RAIIPtr& operator=(const RAIIPtr&) = delete;
    
    // Enable move
    RAIIPtr(RAIIPtr&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
        cout << "RAIIPtr moved" << endl;
    }
    
    RAIIPtr& operator=(RAIIPtr&& other) noexcept {
        if (this != &other) {
            delete ptr;
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        cout << "RAIIPtr move-assigned" << endl;
        return *this;
    }
    
    T& operator*() const { return *ptr; }
    T* operator->() const { return ptr; }
    T* get() const { return ptr; }
};

int main() {
    cout << "=== RAII Demo ===" << endl;
    
    // RAII with file handling
    {
        FileHandler fileHandler("test.txt");
        fileHandler.write("Hello, RAII!");
        fileHandler.write("This file will be automatically closed.");
        // File automatically closed when fileHandler goes out of scope
    }
    
    // RAII with dynamic memory
    {
        RAIIPtr<int> ptr(new int(42));
        cout << "Value: " << *ptr << endl;
        // Memory automatically deleted when ptr goes out of scope
    }
    
    // RAII with exception safety
    try {
        FileHandler safeFile("safe.txt");
        safeFile.write("This will be written safely.");
        
        if (true) {  // Simulate error condition
            throw runtime_error("Error occurred");
        }
        
        safeFile.write("This won't be written.");
    } catch (const exception& e) {
        cout << "Exception caught: " << e.what() << endl;
        cout << "But file was properly closed!" << endl;
    }
    
    // RAII with standard library smart pointers
    {
        unique_ptr<int> stdPtr = make_unique<int>(100);
        cout << "Standard smart pointer value: " << *stdPtr << endl;
        // Memory automatically freed
    }
    
    return 0;
}
```

## Smart Pointers

Smart pointers automate memory management and prevent common errors.

### Exercise 4: unique_ptr

Complete this unique_ptr example:

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <string>
using namespace std;

class Widget {
private:
    string name;
    int value;
    
public:
    Widget(const string& n, int v) : name(n), value(v) {
        cout << "Widget created: " << name << " with value " << value << endl;
    }
    
    ~Widget() {
        cout << "Widget destroyed: " << name << endl;
    }
    
    void display() const {
        cout << "Widget: " << name << ", Value: " << value << endl;
    }
    
    string getName() const { return name; }
    int getValue() const { return value; }
    void setValue(int v) { value = v; }
};

int main() {
    cout << "=== unique_ptr Demo ===" << endl;
    
    // Creating unique_ptr
    unique_ptr<Widget> ptr1 = make_unique<Widget>("First", 10);
    unique_ptr<Widget> ptr2 = make_unique<Widget>("Second", 20);
    
    // Accessing object
    ptr1->display();
    cout << "Value via *: " << (*ptr2).getValue() << endl;
    
    // unique_ptr cannot be copied (only moved)
    // unique_ptr<Widget> ptr3 = ptr1;  // Error: copy constructor deleted
    
    // Moving unique_ptr
    unique_ptr<Widget> ptr3 = move(ptr1);  // Transfer ownership
    cout << "After move:" << endl;
    cout << "ptr1 is " << (ptr1 ? "valid" : "nullptr") << endl;
    cout << "ptr3 is " << (ptr3 ? "valid" : "nullptr") << endl;
    
    if (ptr3) {
        ptr3->display();
    }
    
    // Array version of unique_ptr
    unique_ptr<int[]> arrayPtr = make_unique<int[]>(5);
    for (int i = 0; i < 5; i++) {
        arrayPtr[i] = (i + 1) * 10;
    }
    
    cout << "Array contents: ";
    for (int i = 0; i < 5; i++) {
        cout << arrayPtr[i] << " ";
    }
    cout << endl;
    
    // Release ownership
    Widget* rawPtr = ptr2.release();  // ptr2 now holds nullptr
    cout << "After release - ptr2 is " << (ptr2 ? "valid" : "nullptr") << endl;
    rawPtr->display();
    
    // Reset to take ownership of different object
    ptr2.reset(rawPtr);  // Now ptr2 owns the object
    cout << "After reset:" << endl;
    ptr2->display();
    
    // Custom deleter example
    auto customDeleter = [](Widget* w) {
        cout << "Custom deleter called for: " << w->getName() << endl;
        delete w;
    };
    
    unique_ptr<Widget, decltype(customDeleter)> customPtr(
        new Widget("Custom", 99), customDeleter);
    
    customPtr->display();
    
    // Using unique_ptr in containers
    vector<unique_ptr<Widget>> widgetVector;
    widgetVector.push_back(make_unique<Widget>("Vector1", 100));
    widgetVector.push_back(make_unique<Widget>("Vector2", 200));
    
    cout << "\nWidgets in vector:" << endl;
    for (const auto& widget : widgetVector) {
        widget->display();
    }
    
    // Moving unique_ptr into vector
    widgetVector.push_back(move(ptr3));
    cout << "After moving ptr3 to vector:" << endl;
    cout << "ptr3 is " << (ptr3 ? "valid" : "nullptr") << endl;
    cout << "Vector size: " << widgetVector.size() << endl;
    
    // Emplace back with unique_ptr
    widgetVector.emplace_back(make_unique<Widget>("Emplaced", 300));
    
    cout << "\nAll widgets after additions:" << endl;
    for (const auto& widget : widgetVector) {
        widget->display();
    }
    
    // unique_ptr automatically cleans up when going out of scope
    cout << "\nEnd of main function - cleanup will happen automatically" << endl;
    
    return 0;
}
```

### Exercise 5: shared_ptr

Complete this shared_ptr example:

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <string>
using namespace std;

class SharedResource {
private:
    string name;
    int data;
    
public:
    SharedResource(const string& n, int d) : name(n), data(d) {
        cout << "SharedResource created: " << name << " with data " << data << endl;
    }
    
    ~SharedResource() {
        cout << "SharedResource destroyed: " << name << endl;
    }
    
    void display() const {
        cout << "Resource: " << name << ", Data: " << data << endl;
    }
    
    string getName() const { return name; }
    int getData() const { return data; }
    void setData(int d) { data = d; }
};

int main() {
    cout << "=== shared_ptr Demo ===" << endl;
    
    // Creating shared_ptr
    shared_ptr<SharedResource> ptr1 = make_shared<SharedResource>("Resource1", 100);
    cout << "Reference count after creation: " << ptr1.use_count() << endl;
    
    // Sharing the same resource
    shared_ptr<SharedResource> ptr2 = ptr1;  // Both point to same object
    cout << "Reference count after sharing: " << ptr1.use_count() << endl;
    cout << "Reference count from ptr2: " << ptr2.use_count() << endl;
    
    // Another shared pointer
    shared_ptr<SharedResource> ptr3 = ptr1;
    cout << "Reference count after third share: " << ptr1.use_count() << endl;
    
    // Accessing the shared object
    ptr1->display();
    ptr2->display();
    ptr3->display();
    
    // Modifying through different pointers
    ptr2->setData(200);
    cout << "After modification through ptr2:" << endl;
    ptr1->display();
    
    // Releasing one reference
    ptr2.reset();  // Decrements reference count
    cout << "Reference count after ptr2.reset(): " << ptr1.use_count() << endl;
    
    // Creating shared_ptr from raw pointer
    SharedResource* rawPtr = new SharedResource("RawResource", 300);
    shared_ptr<SharedResource> ptrFromRaw(rawPtr);  // Takes ownership
    cout << "Reference count for raw pointer: " << ptrFromRaw.use_count() << endl;
    ptrFromRaw->display();
    
    // Array version of shared_ptr (not directly supported, but can use custom deleter)
    shared_ptr<int> arrayPtr(new int[5], [](int* p) { 
        cout << "Deleting array..." << endl; 
        delete[] p; 
    });
    
    // Initialize array
    for (int i = 0; i < 5; i++) {
        arrayPtr.get()[i] = (i + 1) * 10;
    }
    
    cout << "Array contents: ";
    for (int i = 0; i < 5; i++) {
        cout << arrayPtr.get()[i] << " ";
    }
    cout << endl;
    
    // Sharing the array
    shared_ptr<int> arrayPtr2 = arrayPtr;  // Shares the same array
    cout << "Array reference count: " << arrayPtr.use_count() << endl;
    
    // Using shared_ptr in containers
    vector<shared_ptr<SharedResource>> resourceVector;
    resourceVector.push_back(make_shared<SharedResource>("Vec1", 1000));
    resourceVector.push_back(make_shared<SharedResource>("Vec2", 2000));
    
    // Share existing resource with vector
    resourceVector.push_back(ptr1);
    
    cout << "\nResources in vector:" << endl;
    for (const auto& resource : resourceVector) {
        cout << "Count: " << resource.use_count() << " - ";
        resource->display();
    }
    
    // Weak pointer example
    cout << "\n=== weak_ptr Demo ===" << endl;
    shared_ptr<SharedResource> shared = make_shared<SharedResource>("Shared", 500);
    weak_ptr<SharedResource> weak = shared;  // Doesn't increment reference count
    
    cout << "Shared count: " << shared.use_count() << endl;
    cout << "Weak expired: " << weak.expired() << endl;
    
    // Lock to access the object safely
    if (auto locked = weak.lock()) {  // Creates temporary shared_ptr
        cout << "Locked resource: ";
        locked->display();
        cout << "During lock, count: " << locked.use_count() << endl;
    }
    
    // Reset shared pointer - now weak pointer should be expired
    shared.reset();
    cout << "After shared.reset(), weak expired: " << weak.expired() << endl;
    
    // Try to lock expired weak pointer
    if (auto locked = weak.lock()) {
        cout << "This won't print" << endl;
    } else {
        cout << "Cannot lock expired weak pointer" << endl;
    }
    
    // Circular reference demonstration (avoid in practice)
    cout << "\n=== Circular Reference Demo ===" << endl;
    struct Node {
        int value;
        shared_ptr<Node> next;
        weak_ptr<Node> parent;  // Use weak_ptr to break cycle
        
        Node(int v) : value(v) {
            cout << "Node " << value << " created" << endl;
        }
        
        ~Node() {
            cout << "Node " << value << " destroyed" << endl;
        }
    };
    
    auto node1 = make_shared<Node>(1);
    auto node2 = make_shared<Node>(2);
    
    node1->next = node2;  // Shared reference
    node2->parent = node1;  // Weak reference to avoid circular reference
    
    cout << "Node1 count: " << node1.use_count() << endl;
    cout << "Node2 count: " << node2.use_count() << endl;
    
    // Clean up
    node1->next.reset();
    node2->parent.reset();
    
    cout << "\nEnd of main - all resources will be cleaned up" << endl;
    
    return 0;
}
```

## Custom Deleters

Custom deleters allow you to specify custom cleanup logic.

### Exercise 6: Custom Deleters

Complete this custom deleter example:

```cpp
#include <iostream>
#include <memory>
#include <functional>
#include <cstdio>
using namespace std;

// Custom deleter for FILE*
struct FileDeleter {
    void operator()(FILE* file) const {
        if (file) {
            cout << "Closing file in custom deleter" << endl;
            fclose(file);
        }
    }
};

// Function-based custom deleter
void closeFile(FILE* file) {
    if (file) {
        cout << "Closing file in function deleter" << endl;
        fclose(file);
    }
}

int main() {
    cout << "=== Custom Deleters Demo ===" << endl;
    
    // Using function object as deleter
    unique_ptr<FILE, FileDeleter> filePtr(fopen("test.txt", "w"));
    if (filePtr) {
        fprintf(filePtr.get(), "Hello from custom deleter!\n");
        cout << "Data written to file" << endl;
    }
    
    // Using function pointer as deleter
    unique_ptr<FILE, decltype(&closeFile)> funcFilePtr(
        fopen("test2.txt", "w"), &closeFile);
    if (funcFilePtr) {
        fprintf(funcFilePtr.get(), "Hello from function deleter!\n");
        cout << "Data written to second file" << endl;
    }
    
    // Using lambda as deleter
    auto lambdaDeleter = [](int* p) {
        cout << "Deleting int array in lambda deleter" << endl;
        delete[] p;
    };
    
    unique_ptr<int, decltype(lambdaDeleter)> arrayPtr(
        new int[5]{1, 2, 3, 4, 5}, lambdaDeleter);
    
    cout << "Array contents: ";
    for (int i = 0; i < 5; i++) {
        cout << arrayPtr.get()[i] << " ";
    }
    cout << endl;
    
    // Using std::function for more flexibility
    std::function<void(int*)> funcDeleter = [](int* p) {
        cout << "Deleting in std::function deleter" << endl;
        delete p;
    };
    
    unique_ptr<int, std::function<void(int*)>> funcPtr(
        new int(42), funcDeleter);
    
    cout << "Single value: " << *funcPtr << endl;
    
    // Custom deleter that does logging
    struct LoggingDeleter {
        string name;
        
        LoggingDeleter(const string& n) : name(n) {}
        
        void operator()(int* p) const {
            cout << "LoggingDeleter '" << name << "' deleting resource" << endl;
            delete p;
        }
    };
    
    unique_ptr<int, LoggingDeleter> loggedPtr(
        new int(100), LoggingDeleter("MyLogger"));
    
    cout << "Logged value: " << *loggedPtr << endl;
    
    // Shared pointer with custom deleter
    shared_ptr<int> sharedWithDeleter(
        new int(200), 
        [](int* p) {
            cout << "Custom deleter for shared_ptr" << endl;
            delete p;
        });
    
    cout << "Shared with custom deleter: " << *sharedWithDeleter << endl;
    cout << "Reference count: " << sharedWithDeleter.use_count() << endl;
    
    // Array deleter for shared_ptr
    shared_ptr<int> sharedArray(
        new int[3]{10, 20, 30},
        [](int* p) {
            cout << "Deleting array in shared_ptr custom deleter" << endl;
            delete[] p;
        });
    
    cout << "Shared array: ";
    for (int i = 0; i < 3; i++) {
        cout << sharedArray.get()[i] << " ";
    }
    cout << endl;
    
    // Share the array
    shared_ptr<int> sharedArray2 = sharedArray;
    cout << "After sharing, count: " << sharedArray.use_count() << endl;
    
    return 0;
}
```

## Memory Management Best Practices

### Exercise 7: Best Practices

Demonstrate best practices for memory management:

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

// Good: RAII wrapper for resource management
class ResourceManager {
private:
    unique_ptr<int[]> data;
    size_t size;
    
public:
    explicit ResourceManager(size_t s) : size(s), data(make_unique<int[]>(s)) {
        cout << "ResourceManager created with size " << size << endl;
    }
    
    // Rule of Zero: compiler-generated special functions are sufficient
    // No need to define destructor, copy/move constructors, or assignment operators
    // because unique_ptr handles resource management automatically
    
    int& operator[](size_t index) { 
        if (index >= size) throw out_of_range("Index out of bounds");
        return data[index]; 
    }
    
    const int& operator[](size_t index) const { 
        if (index >= size) throw out_of_range("Index out of bounds");
        return data[index]; 
    }
    
    size_t getSize() const { return size; }
    
    void fill(int value) {
        for (size_t i = 0; i < size; i++) {
            data[i] = value;
        }
    }
};

// Factory function returning smart pointer
unique_ptr<ResourceManager> createManager(size_t size) {
    return make_unique<ResourceManager>(size);
}

// Function taking smart pointer by value (transfers ownership)
void takeOwnership(unique_ptr<ResourceManager> manager) {
    cout << "Taking ownership of manager with size " << manager->getSize() << endl;
    manager->fill(99);
    cout << "Filled with 99s" << endl;
    // manager automatically destroyed when function ends
}

// Function taking smart pointer by reference (doesn't transfer ownership)
void useManager(const unique_ptr<ResourceManager>& manager) {
    cout << "Using manager with size " << manager->getSize() << endl;
    cout << "First element: " << (*manager)[0] << endl;
}

// Function returning shared_ptr
shared_ptr<ResourceManager> createSharedManager(size_t size) {
    return make_shared<ResourceManager>(size);
}

int main() {
    cout << "=== Memory Management Best Practices ===" << endl;
    
    // 1. Prefer smart pointers over raw pointers
    auto manager = make_unique<ResourceManager>(10);
    manager->fill(42);
    cout << "First element: " << (*manager)[0] << endl;
    
    // 2. Use make_unique and make_shared
    auto anotherManager = make_unique<ResourceManager>(5);
    anotherManager->fill(77);
    
    // 3. Transfer ownership with move
    unique_ptr<ResourceManager> transferred = move(anotherManager);
    cout << "After transfer, original is " << (anotherManager ? "valid" : "nullptr") << endl;
    cout << "Transferred manager size: " << transferred->getSize() << endl;
    
    // 4. Use factory functions
    auto factoryManager = createManager(8);
    factoryManager->fill(55);
    
    // 5. Pass by reference when not transferring ownership
    useManager(factoryManager);
    cout << "After useManager, factoryManager still valid: " << factoryManager->getSize() << endl;
    
    // 6. Transfer ownership to function
    takeOwnership(move(factoryManager));  // factoryManager is now nullptr
    cout << "After takeOwnership, factoryManager is " << (factoryManager ? "valid" : "nullptr") << endl;
    
    // 7. Use shared_ptr for shared ownership
    auto sharedMgr1 = createSharedManager(6);
    {
        auto sharedMgr2 = sharedMgr1;  // Share ownership
        cout << "Within scope, count: " << sharedMgr1.use_count() << endl;
        sharedMgr2->fill(11);
    }  // sharedMgr2 goes out of scope
    cout << "After scope, count: " << sharedMgr1.use_count() << endl;
    cout << "First element after sharing: " << (*sharedMgr1)[0] << endl;
    
    // 8. Use weak_ptr to break cycles
    shared_ptr<ResourceManager> parent = createSharedManager(4);
    weak_ptr<ResourceManager> parentWeak = parent;
    
    cout << "Weak pointer valid: " << !parentWeak.expired() << endl;
    
    // 9. Exception safety with RAII
    try {
        auto safeManager = make_unique<ResourceManager>(3);
        safeManager->fill(88);
        
        if (true) {  // Simulate error condition
            throw runtime_error("Something went wrong");
        }
        
        // This line won't be reached, but safeManager is still properly destroyed
    } catch (const exception& e) {
        cout << "Exception caught: " << e.what() << endl;
        cout << "But ResourceManager was properly destroyed!" << endl;
    }
    
    // 10. Use containers with smart pointers for collections
    vector<unique_ptr<ResourceManager>> managers;
    
    managers.push_back(make_unique<ResourceManager>(2));
    managers.push_back(make_unique<ResourceManager>(4));
    managers.push_back(createManager(6));  // From factory function
    
    cout << "\nManagers in vector:" << endl;
    for (size_t i = 0; i < managers.size(); i++) {
        managers[i]->fill(static_cast<int>(i + 1) * 10);
        cout << "Manager " << i << " size: " << managers[i]->getSize() 
             << ", first element: " << (*managers[i])[0] << endl;
    }
    
    // 11. Use emplace_back for efficient insertion
    managers.emplace_back(make_unique<ResourceManager>(3));
    managers.back()->fill(999);
    cout << "Last manager: size " << managers.back()->getSize() 
         << ", value " << (*managers.back())[0] << endl;
    
    // 12. Avoid raw pointers for ownership
    // Good: Use smart pointers
    unique_ptr<int> goodPtr = make_unique<int>(42);
    
    // Avoid: Raw pointers for ownership
    // int* badPtr = new int(42);  // Potential memory leak
    // delete badPtr;  // Easy to forget
    
    cout << "\nAll resources will be automatically cleaned up" << endl;
    
    return 0;
}
```

## Advanced Smart Pointer Techniques

### Exercise 8: Advanced Techniques

Complete this advanced smart pointer example:

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <functional>
using namespace std;

// Polymorphic hierarchy
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

// Custom allocator example
template<typename T>
class DebugAllocator {
public:
    using value_type = T;
    
    DebugAllocator() = default;
    
    template<typename U>
    DebugAllocator(const DebugAllocator<U>&) {}
    
    T* allocate(size_t n) {
        cout << "Allocating " << n << " objects of size " << sizeof(T) << endl;
        return static_cast<T*>(std::malloc(n * sizeof(T)));
    }
    
    void deallocate(T* p, size_t n) {
        cout << "Deallocating " << n << " objects" << endl;
        std::free(p);
    }
    
    template<typename U>
    bool operator==(const DebugAllocator<U>&) const { return true; }
    
    template<typename U>
    bool operator!=(const DebugAllocator<U>&) const { return false; }
};

int main() {
    cout << "=== Advanced Smart Pointer Techniques ===" << endl;
    
    // 1. Polymorphic smart pointers
    vector<unique_ptr<Shape>> shapes;
    
    shapes.push_back(make_unique<Circle>(5.0));
    shapes.push_back(make_unique<Rectangle>(4.0, 6.0));
    shapes.push_back(make_unique<Circle>(3.0));
    
    cout << "Drawing shapes:" << endl;
    for (const auto& shape : shapes) {
        shape->draw();
        cout << "Area: " << shape->area() << endl;
    }
    
    // 2. Custom allocator with smart pointers
    // Note: This is more complex and usually not done in practice
    // But demonstrates the concept
    vector<int, DebugAllocator<int>> debugVec(DebugAllocator<int>{});
    debugVec.push_back(1);
    debugVec.push_back(2);
    debugVec.push_back(3);
    
    cout << "Debug vector contents: ";
    for (const auto& val : debugVec) {
        cout << val << " ";
    }
    cout << endl;
    
    // 3. Observer pattern with weak_ptr
    class Subject {
    private:
        vector<weak_ptr<Shape>> observers;
        
    public:
        void attach(shared_ptr<Shape> observer) {
            observers.push_back(observer);
        }
        
        void notify() {
            cout << "Notifying observers:" << endl;
            auto it = observers.begin();
            while (it != observers.end()) {
                if (auto obs = it->lock()) {  // Check if observer still exists
                    obs->draw();
                    ++it;
                } else {
                    cout << "Observer expired, removing..." << endl;
                    it = observers.erase(it);  // Remove expired observer
                }
            }
        }
    };
    
    Subject subject;
    
    auto observer1 = make_shared<Circle>(2.0);
    auto observer2 = make_shared<Rectangle>(3.0, 4.0);
    
    subject.attach(observer1);
    subject.attach(observer2);
    
    subject.notify();
    
    // Remove one observer
    observer1.reset();
    subject.notify();  // Should detect that observer1 is gone
    
    // 4. Shared pointer with custom deleter and allocator
    auto customDeleter = [](Shape* s) {
        cout << "Custom deleter for shape" << endl;
        delete s;
    };
    
    shared_ptr<Shape> customShape(new Rectangle(5.0, 7.0), customDeleter);
    customShape->draw();
    cout << "Area: " << customShape->area() << endl;
    
    // 5. Aliasing constructor (advanced)
    struct DataWithMetadata {
        int data;
        string metadata;
        
        DataWithMetadata(int d, const string& m) : data(d), metadata(m) {}
    };
    
    shared_ptr<DataWithMetadata> fullData = make_shared<DataWithMetadata>(42, "important");
    
    // Create a shared_ptr that points to the data member but shares ownership
    shared_ptr<int> dataPtr(fullData, &fullData->data);
    
    cout << "Aliased data: " << *dataPtr << endl;
    cout << "Full data metadata: " << fullData->metadata << endl;
    cout << "Shared count: " << fullData.use_count() << endl;
    
    // 6. enable_shared_from_this for objects that need to create shared_ptr to themselves
    class SelfReferencing : public enable_shared_from_this<SelfReferencing> {
    public:
        shared_ptr<SelfReferencing> getSharedPtr() {
            return shared_from_this();
        }
        
        void doSomething() {
            cout << "Doing something in SelfReferencing object" << endl;
        }
    };
    
    auto selfRef = make_shared<SelfReferencing>();
    auto anotherRef = selfRef->getSharedPtr();
    cout << "Self-referencing count: " << selfRef.use_count() << endl;
    anotherRef->doSomething();
    
    // 7. Combining smart pointers with STL algorithms
    vector<shared_ptr<Shape>> sharedShapes;
    sharedShapes.push_back(make_shared<Circle>(1.0));
    sharedShapes.push_back(make_shared<Rectangle>(2.0, 3.0));
    sharedShapes.push_back(make_shared<Circle>(4.0));
    
    // Find shapes with area > 10
    auto it = find_if(sharedShapes.begin(), sharedShapes.end(),
                      [](const shared_ptr<Shape>& s) { return s->area() > 10.0; });
    
    if (it != sharedShapes.end()) {
        cout << "Found shape with area > 10: ";
        (*it)->draw();
    }
    
    // Remove shapes with area < 5
    sharedShapes.erase(
        remove_if(sharedShapes.begin(), sharedShapes.end(),
                  [](const shared_ptr<Shape>& s) { return s->area() < 5.0; }),
        sharedShapes.end());
    
    cout << "Remaining shapes after filtering:" << endl;
    for (const auto& shape : sharedShapes) {
        shape->draw();
    }
    
    return 0;
}
```

## Memory Leak Detection and Prevention

### Exercise 9: Memory Leak Prevention

Demonstrate techniques for preventing memory leaks:

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <thread>
using namespace std;

// RAII wrapper for timing operations
class Timer {
private:
    chrono::high_resolution_clock::time_point start_time;
    string operation_name;
    
public:
    explicit Timer(const string& name) : operation_name(name) {
        start_time = chrono::high_resolution_clock::now();
        cout << "Starting timer for: " << operation_name << endl;
    }
    
    ~Timer() {
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        cout << "Timer for '" << operation_name << "' completed in " 
             << duration.count() << " microseconds" << endl;
    }
};

// Safe wrapper for potentially leaky operations
template<typename T>
class SafeContainer {
private:
    unique_ptr<T[]> data;
    size_t size;
    
public:
    explicit SafeContainer(size_t s) : size(s), data(make_unique<T[]>(s)) {
        cout << "SafeContainer created with " << size << " elements" << endl;
    }
    
    // No need for explicit destructor - unique_ptr handles cleanup
    
    T& operator[](size_t index) {
        if (index >= size) throw out_of_range("Index out of bounds");
        return data[index];
    }
    
    const T& operator[](size_t index) const {
        if (index >= size) throw out_of_range("Index out of bounds");
        return data[index];
    }
    
    size_t getSize() const { return size; }
    
    // Function that might throw but won't leak
    void riskyOperation() {
        cout << "Performing risky operation..." << endl;
        // Simulate some work
        this_thread::sleep_for(chrono::milliseconds(100));
        
        // Simulate potential error
        if (size > 100) {
            throw runtime_error("Container too large!");
        }
        
        cout << "Risky operation completed successfully" << endl;
    }
};

int main() {
    cout << "=== Memory Leak Prevention Demo ===" << endl;
    
    {
        Timer timer("Main Operations");
        
        // 1. Use RAII for automatic cleanup
        SafeContainer<int> container(10);
        for (size_t i = 0; i < container.getSize(); i++) {
            container[i] = static_cast<int>(i + 1);
        }
        
        cout << "Container contents: ";
        for (size_t i = 0; i < container.getSize(); i++) {
            cout << container[i] << " ";
        }
        cout << endl;
        
        // 2. Exception safety with RAII
        try {
            SafeContainer<int> largeContainer(150);  // This will cause exception
            largeContainer.riskyOperation();  // Won't reach here
        } catch (const exception& e) {
            cout << "Caught exception: " << e.what() << endl;
            cout << "But container was properly destroyed!" << endl;
        }
        
        // 3. Smart pointers in containers
        vector<unique_ptr<SafeContainer<double>>> containerVector;
        
        for (int i = 0; i < 3; i++) {
            auto container = make_unique<SafeContainer<double>>(5);
            for (size_t j = 0; j < container->getSize(); j++) {
                (*container)[j] = (i + 1) * (j + 1) * 1.5;
            }
            containerVector.push_back(move(container));
        }
        
        cout << "\nContainers in vector:" << endl;
        for (size_t i = 0; i < containerVector.size(); i++) {
            cout << "Container " << i << ": ";
            for (size_t j = 0; j < containerVector[i]->getSize(); j++) {
                cout << (*containerVector[i])[j] << " ";
            }
            cout << endl;
        }
        
        // 4. Shared ownership when needed
        vector<shared_ptr<SafeContainer<int>>> sharedContainers;
        
        auto sharedContainer = make_shared<SafeContainer<int>>(4);
        for (size_t i = 0; i < sharedContainer->getSize(); i++) {
            (*sharedContainer)[i] = static_cast<int>(i * i);
        }
        
        sharedContainers.push_back(sharedContainer);
        
        // Share the same container
        auto anotherRef = sharedContainer;
        sharedContainers.push_back(anotherRef);
        
        cout << "\nShared container (accessed through different references):" << endl;
        cout << "Reference count: " << sharedContainer.use_count() << endl;
        cout << "Values: ";
        for (size_t i = 0; i < sharedContainer->getSize(); i++) {
            cout << (*sharedContainer)[i] << " ";
        }
        cout << endl;
    }  // Timer automatically reports elapsed time
    
    // 5. Demonstrate proper cleanup order
    cout << "\n=== Cleanup Order Demo ===" << endl;
    
    {
        cout << "Entering scope..." << endl;
        
        auto ptr1 = make_unique<SafeContainer<int>>(2);
        auto ptr2 = make_unique<SafeContainer<int>>(3);
        auto ptr3 = make_shared<SafeContainer<int>>(4);
        
        cout << "All resources created" << endl;
        
        // Resources will be destroyed in reverse order of creation
        // when leaving scope
    }  // All resources automatically cleaned up here
    
    cout << "Exited scope - all resources cleaned up" << endl;
    
    // 6. Weak pointer to prevent circular references
    cout << "\n=== Weak Pointer Demo ===" << endl;
    
    struct Parent {
        string name;
        vector<shared_ptr<struct Child>> children;
        
        Parent(const string& n) : name(n) {
            cout << "Parent " << name << " created" << endl;
        }
        
        ~Parent() {
            cout << "Parent " << name << " destroyed" << endl;
        }
    };
    
    struct Child {
        string name;
        weak_ptr<Parent> parent;  // Use weak_ptr to avoid circular reference
        
        Child(const string& n) : name(n) {
            cout << "Child " << name << " created" << endl;
        }
        
        ~Child() {
            cout << "Child " << name << " destroyed" << endl;
        }
        
        void setParent(const shared_ptr<Parent>& p) {
            parent = p;
        }
        
        void printParent() {
            if (auto p = parent.lock()) {  // Check if parent still exists
                cout << "Child " << name << " belongs to parent " << p->name << endl;
            } else {
                cout << "Child " << name << " has no parent (parent was destroyed)" << endl;
            }
        }
    };
    
    auto parent = make_shared<Parent>("FamilyHead");
    auto child1 = make_shared<Child>("Child1");
    auto child2 = make_shared<Child>("Child2");
    
    child1->setParent(parent);
    child2->setParent(parent);
    parent->children.push_back(child1);
    parent->children.push_back(child2);
    
    child1->printParent();
    child2->printParent();
    
    // Destroy parent - children should still exist but parent reference should be expired
    parent.reset();
    cout << "After destroying parent:" << endl;
    child1->printParent();
    child2->printParent();
    
    cout << "\nProgram ending - all resources cleaned up" << endl;
    
    return 0;
}
```

## Best Practices Summary

### Exercise 10: Best Practices Checklist

Demonstrate the key best practices for memory management:

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <string>
using namespace std;

// Example of good practices
class BestPracticeExample {
private:
    unique_ptr<vector<string>> data;
    
public:
    BestPracticeExample() : data(make_unique<vector<string>>()) {
        // RAII: Resource acquired in constructor
    }
    
    // Rule of Zero: No need to define destructor, copy/move ops
    // because unique_ptr handles resource management
    
    void addData(const string& item) {
        data->push_back(item);
    }
    
    size_t size() const {
        return data->size();
    }
    
    const vector<string>& getData() const {
        return *data;
    }
};

int main() {
    cout << "=== Memory Management Best Practices Checklist ===" << endl;
    
    // ✓ Use RAII (Resource Acquisition Is Initialization)
    BestPracticeExample example;
    example.addData("Item 1");
    example.addData("Item 2");
    cout << "Data size: " << example.size() << endl;
    
    // ✓ Prefer smart pointers over raw pointers
    auto smartInt = make_unique<int>(42);
    auto sharedStr = make_shared<string>("Hello");
    cout << "Smart pointer values: " << *smartInt << ", " << *sharedStr << endl;
    
    // ✓ Use make_unique and make_shared
    auto container = make_unique<vector<int>>();
    container->push_back(100);
    cout << "Container value: " << container->front() << endl;
    
    // ✓ Use move semantics when transferring ownership
    auto movedContainer = move(container);
    cout << "After move, original is " << (container ? "valid" : "nullptr") << endl;
    cout << "Moved container value: " << movedContainer->front() << endl;
    
    // ✓ Use const references for non-owning parameters
    auto useResource = [](const unique_ptr<int>& ptr) {
        if (ptr) {
            cout << "Using resource: " << *ptr << endl;
        }
    };
    
    useResource(smartInt);
    
    // ✓ Use shared_ptr for shared ownership
    auto sharedResource = make_shared<int>(200);
    vector<shared_ptr<int>> sharedOwners;
    sharedOwners.push_back(sharedResource);
    sharedOwners.push_back(sharedResource);
    cout << "Shared resource count: " << sharedResource.use_count() << endl;
    
    // ✓ Use weak_ptr to break cycles
    weak_ptr<int> weakRef = sharedResource;
    cout << "Weak reference expired: " << weakRef.expired() << endl;
    
    // ✓ Exception safety with RAII
    try {
        auto safeResource = make_unique<string>("Safe Resource");
        cout << "Resource created: " << *safeResource << endl;
        
        if (false) {  // Normal execution path
            throw runtime_error("Simulated error");
        }
        
        cout << "This line would be reached if no exception" << endl;
    } catch (const exception& e) {
        cout << "Exception handled safely" << endl;
    }
    // safeResource automatically cleaned up even if exception occurred
    
    // ✓ Use containers with smart pointers for collections
    vector<unique_ptr<string>> stringCollection;
    stringCollection.push_back(make_unique<string>("First"));
    stringCollection.push_back(make_unique<string>("Second"));
    stringCollection.push_back(make_unique<string>("Third"));
    
    cout << "String collection: ";
    for (const auto& str : stringCollection) {
        cout << *str << " ";
    }
    cout << endl;
    
    // ✓ Avoid raw pointers for ownership
    // Good: Smart pointers manage ownership
    unique_ptr<int> goodPtr = make_unique<int>(300);
    
    // Bad: Raw pointers require manual management
    // int* badPtr = new int(300);  // Potential leak
    // delete badPtr;  // Must remember to delete
    
    cout << "Final value: " << *goodPtr << endl;
    
    cout << "\nAll resources properly managed and cleaned up!" << endl;
    
    return 0;
}
```

## Summary

In this chapter, you learned:
- Memory layout in C++ (stack, heap, static storage)
- Manual memory management with new/delete
- The RAII (Resource Acquisition Is Initialization) principle
- Smart pointers: unique_ptr, shared_ptr, weak_ptr
- Custom deleters and allocators
- Best practices for memory management
- Exception safety with smart pointers
- How to prevent memory leaks and dangling pointers

## Key Takeaways

- RAII is fundamental to C++ resource management
- Smart pointers automate memory management and prevent common errors
- unique_ptr provides exclusive ownership
- shared_ptr provides shared ownership with reference counting
- weak_ptr breaks circular references
- Always prefer smart pointers over raw pointers for ownership
- Use make_unique and make_shared for exception safety
- Move semantics enable efficient resource transfers
- Exception safety is built into smart pointer design

## Common Mistakes to Avoid

1. Using raw pointers for ownership instead of smart pointers
2. Forgetting to delete dynamically allocated memory
3. Using deleted pointers (dangling pointers)
4. Creating circular references with shared_ptr
5. Not using make_unique/make_shared for exception safety
6. Copying unique_ptr instead of moving it
7. Not checking weak_ptr before using it
8. Using shared_ptr when unique_ptr would suffice
9. Forgetting that custom deleters change the type of smart pointers
10. Not understanding the performance implications of reference counting

## Next Steps

Now that you understand memory management and smart pointers, you're ready to learn about exception handling in Chapter 12.