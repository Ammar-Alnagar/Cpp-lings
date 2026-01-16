# Chapter 15: Modern C++ Features

## Overview

This chapter covers the major modern C++ features introduced in C++11 and later standards (C++14, C++17, C++20, C++23). You'll learn about language improvements, library enhancements, and new paradigms that make C++ more expressive, safer, and easier to use.

## Learning Objectives

By the end of this chapter, you will:
- Master auto type deduction and trailing return types
- Understand and use range-based for loops
- Learn about lambda expressions and closures
- Master move semantics and perfect forwarding
- Understand smart pointers and RAII
- Learn about constexpr and consteval
- Understand concepts and constraints (C++20)
- Learn about modules (C++20)
- Understand coroutines (C++20)
- Learn about concepts like designated initializers and aggregate initialization
- Understand new STL features and algorithms
- Learn about concepts like spaceship operator and modules

## Auto Type Deduction

The `auto` keyword allows the compiler to deduce variable types automatically.

### Exercise 1: Auto Type Deduction

The following code has errors. Find and fix them:

```cpp
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <typeinfo>
using namespace std;

int main() {
    cout << "=== Auto Type Deduction Demo ===" << endl;
    
    // Basic auto usage
    auto x = 42;          // int
    auto y = 3.14;        // double
    auto z = 'A';         // char
    auto s = "Hello";     // const char* (not string!)
    
    cout << "x: " << x << " (type: " << typeid(x).name() << ")" << endl;
    cout << "y: " << y << " (type: " << typeid(y).name() << ")" << endl;
    cout << "z: " << z << " (type: " << typeid(z).name() << ")" << endl;
    cout << "s: " << s << " (type: " << typeid(s).name() << ")" << endl;
    
    // Error: auto cannot be used without initialization
    // auto uninitialized;  // Error!
    // uninitialized = 42;  // Too late!
    
    // Correct way: initialize at declaration
    auto initialized = 42;
    
    // Complex types become much cleaner with auto
    vector<pair<string, int>> complexData = {{"Alice", 25}, {"Bob", 30}};
    
    // Without auto (verbose):
    vector<pair<string, int>>::iterator verboseIt = complexData.begin();
    
    // With auto (clean):
    auto cleanIt = complexData.begin();
    
    cout << "\nIterating with auto:" << endl;
    for (auto it = complexData.begin(); it != complexData.end(); ++it) {
        cout << it->first << " is " << it->second << " years old" << endl;
    }
    
    // Error: incorrect type deduction
    auto wrongType = 42;  // This is int, not double
    // wrongType = 3.14;  // Precision loss, but no error
    
    // Better: be explicit about type when needed
    double correctType = 42;
    correctType = 3.14;  // Maintains precision
    
    // Auto with references and const
    int value = 100;
    auto& ref = value;        // int&
    const auto& constRef = value;  // const int&
    auto* ptr = &value;       // int*
    
    cout << "\nAuto with qualifiers:" << endl;
    cout << "ref: " << ref << endl;
    cout << "constRef: " << constRef << endl;
    cout << "ptr: " << *ptr << endl;
    
    ref = 200;  // Modifies original value
    cout << "After modifying ref, value is: " << value << endl;
    
    // Decltype - deduce type from expression
    auto deductedType = decltype(value)(50);  // Creates int with value 50
    cout << "Deducted type variable: " << deductedType << endl;
    
    // Trailing return type (useful for complex types)
    auto getComplexValue() -> int {  // Trailing return type
        return 42;
    }
    
    // Lambda with auto return type (C++14)
    auto multiplier = [](auto a, auto b) {  // Generic lambda (C++14)
        return a * b;
    };
    
    cout << "\nGeneric lambda results:" << endl;
    cout << "int * int: " << multiplier(5, 3) << endl;
    cout << "double * double: " << multiplier(2.5, 4.0) << endl;
    cout << "int * double: " << multiplier(10, 3.5) << endl;
    
    // Auto in range-based for loops
    vector<string> names = {"Charlie", "Diana", "Eve"};
    
    cout << "\nRange-based for with auto:" << endl;
    for (const auto& name : names) {  // const auto& - efficient and safe
        cout << name << endl;
    }
    
    // Error: copying strings in loop (inefficient)
    cout << "\nInefficient version (copies each string):" << endl;
    for (auto name : names) {  // Copies each string!
        cout << name << endl;
    }
    
    // Correct: use const reference for efficiency
    cout << "\nEfficient version (const reference):" << endl;
    for (const auto& name : names) {  // No copy, no modification
        cout << name << endl;
    }
    
    // Using auto with STL containers
    map<string, vector<int>> dataMap;
    dataMap["numbers"] = {1, 2, 3, 4, 5};
    dataMap["primes"] = {2, 3, 5, 7, 11};
    
    cout << "\nMap iteration with auto:" << endl;
    for (const auto& [key, values] : dataMap) {  // C++17 structured bindings
        cout << key << ": ";
        for (const auto& value : values) {
            cout << value << " ";
        }
        cout << endl;
    }
    
    return 0;
}
```

## Range-Based For Loops

Range-based for loops provide a cleaner way to iterate over containers.

### Exercise 2: Range-Based For Loops

Complete this range-based for loop example with errors:

```cpp
#include <iostream>
#include <vector>
#include <array>
#include <list>
#include <string>
using namespace std;

int main() {
    cout << "=== Range-Based For Loops Demo ===" << endl;
    
    // Array iteration
    int arr[] = {1, 2, 3, 4, 5};
    cout << "Array elements: ";
    for (const auto& element : arr) {  // const auto& to avoid copying
        cout << element << " ";
    }
    cout << endl;
    
    // Vector iteration
    vector<string> words = {"Hello", "Modern", "C++", "World"};
    cout << "Vector elements: ";
    for (const auto& word : words) {  // const reference for efficiency
        cout << word << " ";
    }
    cout << endl;
    
    // Modifying elements
    vector<int> numbers = {10, 20, 30, 40, 50};
    cout << "Before modification: ";
    for (const auto& num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    // Modify elements using non-const reference
    for (auto& num : numbers) {  // auto& to allow modification
        num *= 2;
    }
    
    cout << "After modification: ";
    for (const auto& num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    // Error: trying to modify with const reference
    // for (const auto& num : numbers) {
    //     num = 100;  // Error: cannot modify through const reference
    // }
    
    // Error: copying large objects unnecessarily
    vector<string> largeStrings = {"Very long string 1", "Very long string 2", "Very long string 3"};
    
    // Inefficient: copies each string
    cout << "\nInefficient iteration (copies strings):" << endl;
    for (auto str : largeStrings) {  // Copies each string!
        cout << "Processing: " << str << endl;
    }
    
    // Efficient: uses const reference
    cout << "\nEfficient iteration (const reference):" << endl;
    for (const auto& str : largeStrings) {  // No copy, no modification
        cout << "Processing: " << str << endl;
    }
    
    // Using with different container types
    array<int, 3> stdArray = {100, 200, 300};
    cout << "\nstd::array elements: ";
    for (const auto& element : stdArray) {
        cout << element << " ";
    }
    cout << endl;
    
    list<double> linkedList = {1.1, 2.2, 3.3, 4.4};
    cout << "std::list elements: ";
    for (const auto& element : linkedList) {
        cout << element << " ";
    }
    cout << endl;
    
    // String iteration
    string text = "Modern C++";
    cout << "String characters: ";
    for (const auto& ch : text) {
        cout << ch << "-";
    }
    cout << endl;
    
    // Error: range-based for with non-container
    int singleValue = 42;
    // for (const auto& element : singleValue) {  // Error: int is not iterable
    //     cout << element << endl;
    // }
    
    // Working with complex nested structures
    vector<vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    
    cout << "\nMatrix elements:" << endl;
    for (const auto& row : matrix) {
        for (const auto& element : row) {
            cout << element << " ";
        }
        cout << endl;
    }
    
    // Using with structured bindings (C++17)
    vector<pair<string, int>> people = {{"Alice", 25}, {"Bob", 30}, {"Charlie", 35}};
    
    cout << "\nPeople with structured bindings:" << endl;
    for (const auto& [name, age] : people) {  // C++17 structured bindings
        cout << name << " is " << age << " years old" << endl;
    }
    
    // Error: structured bindings with non-pair
    // vector<int> simpleVec = {1, 2, 3};
    // for (const auto& [first, second] : simpleVec) {  // Error: int is not decomposable
    //     cout << first << ", " << second << endl;
    // }
    
    // Range-based for with initialization (C++20)
    if (auto vec = vector<int>{1, 2, 3, 4, 5}; !vec.empty()) {
        cout << "\nC++20 range-for with initializer: ";
        for (const auto& element : vec) {
            cout << element << " ";
        }
        cout << endl;
    }
    
    return 0;
}
```

## Lambda Expressions

Lambda expressions provide a concise way to create anonymous function objects.

### Exercise 3: Lambda Expressions

Complete this lambda expressions example with errors:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
using namespace std;

int main() {
    cout << "=== Lambda Expressions Demo ===" << endl;
    
    // Basic lambda
    auto greet = []() { cout << "Hello from lambda!" << endl; };
    greet();
    
    // Lambda with parameters
    auto add = [](int a, int b) { return a + b; };
    cout << "5 + 3 = " << add(5, 3) << endl;
    
    // Lambda with return type specification
    auto divide = [](double a, double b) -> double {
        if (b != 0) return a / b;
        return 0.0;
    };
    cout << "10.0 / 3.0 = " << divide(10.0, 3.0) << endl;
    
    // Capturing by value
    int multiplier = 10;
    auto multiplyBy = [multiplier](int x) {  // Capture by value
        return x * multiplier;
    };
    cout << "5 * 10 = " << multiplyBy(5) << endl;
    
    // Capturing by reference
    int counter = 0;
    auto incrementer = [&counter]() {  // Capture by reference
        return ++counter;
    };
    cout << "Increment 1: " << incrementer() << endl;
    cout << "Increment 2: " << incrementer() << endl;
    cout << "Increment 3: " << incrementer() << endl;
    cout << "Counter value: " << counter << endl;
    
    // Mixed captures
    int base = 5;
    auto complexLambda = [multiplier, &counter, base](int x) {
        counter += x;  // Modifies captured reference
        return x * multiplier + base;  // Uses captured values
    };
    
    cout << "Complex lambda result: " << complexLambda(3) << endl;
    cout << "Counter after complex lambda: " << counter << endl;
    
    // Capture all by value [=]
    auto captureAllByValue = [=](int x) {
        return x * multiplier + counter + base;  // Uses captured values
    };
    
    // Capture all by reference [&]
    auto captureAllByReference = [&]() {
        counter++;
        return counter * base;
    };
    
    cout << "Capture all by value: " << captureAllByValue(2) << endl;
    cout << "Capture all by reference: " << captureAllByReference() << endl;
    
    // Error: capturing 'this' in regular function (only allowed in member functions)
    // auto thisCapture = [this]() {  // Error: 'this' not available
    //     return multiplier;
    // };
    
    // Mutable lambda - allows modification of captured values by value
    int mutableValue = 100;
    auto mutableLambda = [mutableValue]() mutable {
        mutableValue += 10;  // OK: modifies the captured copy
        return mutableValue;
    };
    
    cout << "Original mutableValue: " << mutableValue << endl;
    cout << "Lambda result: " << mutableLambda() << endl;
    cout << "Original after lambda: " << mutableValue << endl;  // Still 100
    
    // Generic lambdas (C++14) - accept auto parameters
    auto genericLambda = [](auto x, auto y) {
        return x + y;
    };
    
    cout << "\nGeneric lambda results:" << endl;
    cout << "int + int: " << genericLambda(5, 3) << endl;
    cout << "double + double: " << genericLambda(2.5, 1.5) << endl;
    cout << "string + string: " << genericLambda(string("Hello "), string("World")) << endl;
    
    // Using lambdas with STL algorithms
    vector<int> numbers = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    
    cout << "\nOriginal numbers: ";
    for (const auto& num : numbers) cout << num << " ";
    cout << endl;
    
    // Sort with custom comparator
    sort(numbers.begin(), numbers.end(), [](int a, int b) {
        return a > b;  // Descending order
    });
    
    cout << "Sorted descending: ";
    for (const auto& num : numbers) cout << num << " ";
    cout << endl;
    
    // Find with predicate
    auto it = find_if(numbers.begin(), numbers.end(), [](int n) {
        return n % 2 == 0;  // Find first even number
    });
    
    if (it != numbers.end()) {
        cout << "First even number: " << *it << endl;
    }
    
    // Count with predicate
    int evenCount = count_if(numbers.begin(), numbers.end(), [](int n) {
        return n % 2 == 0;
    });
    cout << "Count of even numbers: " << evenCount << endl;
    
    // Transform with lambda
    vector<int> doubled(numbers.size());
    transform(numbers.begin(), numbers.end(), doubled.begin(), [](int n) {
        return n * 2;
    });
    
    cout << "Doubled values: ";
    for (const auto& num : doubled) cout << num << " ";
    cout << endl;
    
    // Remove with lambda
    numbers.erase(remove_if(numbers.begin(), numbers.end(), [](int n) {
        return n > 5;  // Remove numbers greater than 5
    }), numbers.end());
    
    cout << "After removing numbers > 5: ";
    for (const auto& num : numbers) cout << num << " ";
    cout << endl;
    
    // Storing lambdas in std::function
    function<int(int, int)> func = [](int a, int b) { return a * b; };
    cout << "Stored lambda result: " << func(6, 7) << endl;
    
    // Function that returns a lambda
    auto makeMultiplier = [](int factor) {
        return [factor](int x) { return x * factor; };  // Returns a lambda
    };
    
    auto timesTwo = makeMultiplier(2);
    auto timesFive = makeMultiplier(5);
    
    cout << "10 * 2 = " << timesTwo(10) << endl;
    cout << "10 * 5 = " << timesFive(10) << endl;
    
    // Recursive lambda
    function<int(int)> factorial = [&factorial](int n) -> int {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    };
    
    cout << "Factorial of 5: " << factorial(5) << endl;
    
    // Lambda with perfect forwarding (advanced)
    auto forwarder = [](auto&&... args) {
        return make_tuple(forward<decltype(args)>(args)...);
    };
    
    auto packed = forwarder(42, string("Hello"), 3.14);
    cout << "Forwarded tuple size: " << tuple_size_v<decltype(packed)> << endl;
    
    return 0;
}
```

## Move Semantics and Rvalue References

Move semantics enable efficient resource transfers.

### Exercise 4: Move Semantics

Complete this move semantics example with errors:

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <memory>
using namespace std;

class MoveableResource {
private:
    int* data;
    size_t size;
    string name;
    
public:
    // Constructor
    MoveableResource(size_t s, const string& n) : size(s), name(n) {
        data = new int[size];
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<int>(i);
        }
        cout << "Constructed: " << name << " with " << size << " elements" << endl;
    }
    
    // Destructor
    ~MoveableResource() {
        delete[] data;
        cout << "Destroyed: " << name << endl;
    }
    
    // Copy constructor
    MoveableResource(const MoveableResource& other) : size(other.size), name(other.name + "_copy") {
        data = new int[size];
        for (size_t i = 0; i < size; i++) {
            data[i] = other.data[i];
        }
        cout << "Copied: " << name << endl;
    }
    
    // Copy assignment operator
    MoveableResource& operator=(const MoveableResource& other) {
        cout << "Copy assignment: " << other.name << endl;
        if (this != &other) {
            delete[] data;  // Clean up existing resource
            size = other.size;
            name = other.name + "_assigned";
            data = new int[size];
            for (size_t i = 0; i < size; i++) {
                data[i] = other.data[i];
            }
        }
        return *this;
    }
    
    // Move constructor
    MoveableResource(MoveableResource&& other) noexcept 
        : data(other.data), size(other.size), name(move(other.name)) {
        other.data = nullptr;  // Transfer ownership
        other.size = 0;
        cout << "Moved: " << name << " from source" << endl;
    }
    
    // Move assignment operator
    MoveableResource& operator=(MoveableResource&& other) noexcept {
        cout << "Move assignment" << endl;
        if (this != &other) {
            delete[] data;  // Clean up existing resource
            data = other.data;  // Transfer ownership
            size = other.size;
            name = move(other.name);
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
    
    // Accessor
    size_t getSize() const { return size; }
    const string& getName() const { return name; }
    int& operator[](size_t index) { return data[index]; }
    const int& operator[](size_t index) const { return data[index]; }
    
    void printFirst() const {
        if (size > 0) {
            cout << name << "[0] = " << data[0] << endl;
        }
    }
};

// Function that demonstrates move semantics
MoveableResource createResource(size_t size) {
    MoveableResource temp(size, "Temporary");
    // NRVO (Named Return Value Optimization) may occur here
    // If not, move constructor will be called
    return temp;
}

// Function that accepts both lvalues and rvalues
void processResource(MoveableResource&& resource) {  // Accept rvalue reference
    cout << "Processing rvalue: " << resource.getName() << endl;
    resource[0] = 999;
}

void processResource(const MoveableResource& resource) {  // Accept lvalue reference
    cout << "Processing lvalue: " << resource.getName() << endl;
}

int main() {
    cout << "=== Move Semantics Demo ===" << endl;
    
    // Create resources
    MoveableResource resource1(5, "Resource1");
    resource1.printFirst();
    
    // Move constructor
    MoveableResource resource2 = move(resource1);  // Explicit move
    cout << "After move:" << endl;
    cout << "Resource1 size: " << resource1.getSize() << endl;  // Should be 0
    cout << "Resource2 size: " << resource2.getSize() << endl;  // Should be 5
    resource2.printFirst();
    
    // Move assignment
    MoveableResource resource3(3, "Resource3");
    resource3 = move(resource2);
    cout << "\nAfter move assignment:" << endl;
    cout << "Resource2 size: " << resource2.getSize() << endl;  // Should be 0
    cout << "Resource3 size: " << resource3.getSize() << endl;  // Should be 5
    resource3.printFirst();
    
    // Return value optimization vs move
    cout << "\n--- Return Value Optimization ---" << endl;
    MoveableResource returnedResource = createResource(4);
    cout << "Returned resource size: " << returnedResource.getSize() << endl;
    
    // Move semantics with containers
    cout << "\n--- Move Semantics with Containers ---" << endl;
    
    vector<MoveableResource> container;
    
    // Emplace back - constructs object in place (most efficient)
    cout << "Emplace back:" << endl;
    container.emplace_back(2, "Emplaced");
    
    // Push back with rvalue - moves the object
    cout << "Push back with rvalue:" << endl;
    container.push_back(MoveableResource(3, "PushedRValue"));
    
    // Push back with lvalue - copies the object
    cout << "Push back with lvalue:" << endl;
    MoveableResource tempResource(4, "Temp");
    container.push_back(tempResource);  // Copies
    container.push_back(move(tempResource));  // Moves (tempResource is now in valid but unspecified state)
    
    cout << "\nContainer size: " << container.size() << endl;
    
    // Perfect forwarding example
    cout << "\n--- Perfect Forwarding ---" << endl;
    
    // Universal references with std::forward
    auto createAndStore = [](auto&&... args) {
        vector<MoveableResource> tempVec;
        tempVec.emplace_back(forward<decltype(args)>(args)...);
        return tempVec;
    };
    
    auto result = createAndStore(6, string("Forwarded"));
    cout << "Forwarded resource size: " << result[0].getSize() << endl;
    
    // Move-only types
    cout << "\n--- Move-Only Types ---" << endl;
    
    vector<unique_ptr<MoveableResource>> uniqueContainer;
    
    // Can only move unique_ptr, not copy
    uniqueContainer.push_back(make_unique<MoveableResource>(7, "Unique1"));
    uniqueContainer.push_back(make_unique<MoveableResource>(8, "Unique2"));
    
    cout << "Unique container size: " << uniqueContainer.size() << endl;
    
    // Move the container
    auto movedContainer = move(uniqueContainer);
    cout << "After move - original size: " << uniqueContainer.size() << endl;
    cout << "After move - moved size: " << movedContainer.size() << endl;
    
    // Forwarding references (universal references)
    cout << "\n--- Forwarding References ---" << endl;
    
    auto universalFunc = [](auto&& param) {
        // Perfectly forward param to another function
        MoveableResource forwardedResource(forward<decltype(param)>(param));
        return forwardedResource;
    };
    
    // This would work with both lvalues and rvalues if we had a proper constructor
    // For now, demonstrating the concept with our existing resources
    
    // Move semantics performance comparison
    cout << "\n--- Performance Comparison ---" << endl;
    
    // Create a large resource
    MoveableResource largeResource(1000000, "LargeResource");
    
    cout << "Copying large resource..." << endl;
    auto start = chrono::high_resolution_clock::now();
    MoveableResource copiedResource = largeResource;  // Expensive copy
    auto end = chrono::high_resolution_clock::now();
    cout << "Copy time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
    
    cout << "Moving large resource..." << endl;
    start = chrono::high_resolution_clock::now();
    MoveableResource movedResource = move(largeResource);  // Cheap move
    end = chrono::high_resolution_clock::now();
    cout << "Move time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
    
    cout << "Large resource size after move: " << largeResource.getSize() << endl;
    cout << "Moved resource size: " << movedResource.getSize() << endl;
    
    return 0;
}
```

## Smart Pointers

Smart pointers automate memory management and prevent common errors.

### Exercise 5: Smart Pointers

Complete this smart pointers example with errors:

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <string>
using namespace std;

class Resource {
private:
    string name;
    vector<int> data;
    
public:
    explicit Resource(const string& n) : name(n), data(1000, 42) {  // Large data for demonstration
        cout << "Resource created: " << name << endl;
    }
    
    ~Resource() {
        cout << "Resource destroyed: " << name << endl;
    }
    
    const string& getName() const { return name; }
    size_t getDataSize() const { return data.size(); }
};

// Function that might throw to demonstrate exception safety
unique_ptr<Resource> potentiallyThrowingFunction(bool shouldThrow) {
    auto resource = make_unique<Resource>("Temporary");
    
    if (shouldThrow) {
        throw runtime_error("Simulated error");
    }
    
    return resource;
}

int main() {
    cout << "=== Smart Pointers Demo ===" << endl;
    
    // unique_ptr - exclusive ownership
    cout << "\n--- unique_ptr ---" << endl;
    
    // Creating unique_ptr
    unique_ptr<Resource> ptr1 = make_unique<Resource>("Unique1");
    unique_ptr<Resource> ptr2 = make_unique<Resource>("Unique2");
    
    cout << "ptr1 name: " << ptr1->getName() << endl;
    cout << "ptr2 name: " << ptr2->getName() << endl;
    
    // unique_ptr cannot be copied, only moved
    // unique_ptr<Resource> ptr3 = ptr1;  // Error: copy constructor deleted
    
    // Move semantics
    unique_ptr<Resource> ptr3 = move(ptr1);  // Transfer ownership
    cout << "After move:" << endl;
    cout << "ptr1 is " << (ptr1 ? "valid" : "nullptr") << endl;
    cout << "ptr3 name: " << ptr3->getName() << endl;
    
    // Accessing through unique_ptr
    cout << "ptr2 data size: " << ptr2->getDataSize() << endl;
    cout << "ptr3 data size: " << ptr3->getDataSize() << endl;
    
    // get() method to access raw pointer (use carefully)
    Resource* rawPtr = ptr2.get();
    cout << "Raw pointer access: " << rawPtr->getName() << endl;
    
    // release() method to release ownership
    Resource* releasedPtr = ptr2.release();  // ptr2 becomes nullptr
    cout << "After release, ptr2 is " << (ptr2 ? "valid" : "nullptr") << endl;
    cout << "Released pointer name: " << releasedPtr->getName() << endl;
    
    // Must manually delete released pointer
    delete releasedPtr;
    
    // reset() method
    ptr3.reset();  // Deletes the resource and sets ptr3 to nullptr
    cout << "After reset, ptr3 is " << (ptr3 ? "valid" : "nullptr") << endl;
    
    // Reset with new resource
    ptr3.reset(new Resource("ResetResource"));
    cout << "After reset with new resource: " << ptr3->getName() << endl;
    
    // Array version of unique_ptr
    cout << "\n--- unique_ptr with Arrays ---" << endl;
    
    unique_ptr<int[]> arrayPtr = make_unique<int[]>(5);
    for (int i = 0; i < 5; i++) {
        arrayPtr[i] = (i + 1) * 10;
    }
    
    cout << "Array contents: ";
    for (int i = 0; i < 5; i++) {
        cout << arrayPtr[i] << " ";
    }
    cout << endl;
    
    // shared_ptr - shared ownership
    cout << "\n--- shared_ptr ---" << endl;
    
    shared_ptr<Resource> shared1 = make_shared<Resource>("Shared1");
    cout << "Reference count after creation: " << shared1.use_count() << endl;
    
    // Share ownership
    shared_ptr<Resource> shared2 = shared1;
    cout << "Reference count after sharing: " << shared1.use_count() << endl;
    cout << "Reference count from shared2: " << shared2.use_count() << endl;
    
    // Another shared pointer
    shared_ptr<Resource> shared3 = shared1;
    cout << "Reference count with 3 owners: " << shared1.use_count() << endl;
    
    cout << "All shared pointers point to: " << shared1->getName() << endl;
    
    // Reset one shared pointer
    shared2.reset();
    cout << "After resetting shared2, count: " << shared1.use_count() << endl;
    
    // Custom deleter example
    cout << "\n--- Custom Deleters ---" << endl;
    
    auto customDeleter = [](Resource* r) {
        cout << "Custom deleter called for: " << r->getName() << endl;
        delete r;
    };
    
    unique_ptr<Resource, decltype(customDeleter)> customPtr(
        new Resource("CustomDeleter"), customDeleter);
    
    cout << "Custom deleter resource: " << customPtr->getName() << endl;
    
    // weak_ptr - breaks circular references
    cout << "\n--- weak_ptr ---" << endl;
    
    shared_ptr<Resource> sharedResource = make_shared<Resource>("SharedForWeak");
    weak_ptr<Resource> weakResource = sharedResource;
    
    cout << "Weak pointer reference count: " << sharedResource.use_count() << endl;
    cout << "Weak pointer expired: " << weakResource.expired() << endl;
    
    // Lock to access the resource safely
    if (auto locked = weakResource.lock()) {
        cout << "Locked resource: " << locked->getName() << endl;
    }
    
    // Reset shared pointer - weak pointer should now be expired
    sharedResource.reset();
    cout << "After shared reset, weak expired: " << weakResource.expired() << endl;
    
    if (auto locked = weakResource.lock()) {
        cout << "This won't print" << endl;
    } else {
        cout << "Cannot lock expired weak pointer" << endl;
    }
    
    // Exception safety demonstration
    cout << "\n--- Exception Safety ---" << endl;
    
    try {
        auto safeResource = potentiallyThrowingFunction(true);  // Will throw
        cout << "This won't be printed" << endl;
    } catch (const exception& e) {
        cout << "Caught exception: " << e.what() << endl;
        cout << "But Resource was automatically cleaned up!" << endl;
    }
    
    // Smart pointers in containers
    cout << "\n--- Smart Pointers in Containers ---" << endl;
    
    vector<unique_ptr<Resource>> uniqueContainer;
    uniqueContainer.push_back(make_unique<Resource>("Vec1"));
    uniqueContainer.push_back(make_unique<Resource>("Vec2"));
    uniqueContainer.push_back(make_unique<Resource>("Vec3"));
    
    cout << "Unique container size: " << uniqueContainer.size() << endl;
    cout << "Contents:" << endl;
    for (const auto& ptr : uniqueContainer) {
        cout << "  " << ptr->getName() << endl;
    }
    
    vector<shared_ptr<Resource>> sharedContainer;
    sharedContainer.push_back(make_shared<Resource>("SharedVec1"));
    sharedContainer.push_back(make_shared<Resource>("SharedVec2"));
    
    // Share resources between containers
    sharedContainer.push_back(sharedContainer[0]);  // Share first resource
    cout << "Shared container reference counts:" << endl;
    for (size_t i = 0; i < sharedContainer.size(); i++) {
        cout << "  " << sharedContainer[i]->getName() 
             << " count: " << sharedContainer[i].use_count() << endl;
    }
    
    // Dynamic cast with smart pointers
    cout << "\n--- Dynamic Cast with Smart Pointers ---" << endl;
    
    class Base {
    public:
        virtual ~Base() = default;
        virtual void baseMethod() { cout << "Base method" << endl; }
    };
    
    class Derived : public Base {
    public:
        void derivedMethod() { cout << "Derived method" << endl; }
    };
    
    shared_ptr<Base> basePtr = make_shared<Derived>();
    
    // Dynamic pointer cast
    shared_ptr<Derived> derivedPtr = dynamic_pointer_cast<Derived>(basePtr);
    if (derivedPtr) {
        cout << "Dynamic cast successful" << endl;
        derivedPtr->derivedMethod();
    }
    
    // Static pointer cast (faster, less safe)
    shared_ptr<Derived> staticPtr = static_pointer_cast<Derived>(basePtr);
    if (staticPtr) {
        cout << "Static cast successful" << endl;
        staticPtr->derivedMethod();
    }
    
    // Smart pointer comparisons
    cout << "\n--- Smart Pointer Comparisons ---" << endl;
    
    unique_ptr<int> ptrA = make_unique<int>(10);
    unique_ptr<int> ptrB = make_unique<int>(20);
    unique_ptr<int> ptrC = ptrA;  // Move ptrA to ptrC
    
    cout << "ptrA == ptrB: " << (ptrA == ptrB) << endl;  // Both are nullptr now
    cout << "ptrC == ptrA: " << (ptrC == ptrA) << endl;  // ptrA is nullptr, ptrC has value
    
    // Using smart pointers with custom allocators (simplified example)
    cout << "\n--- Smart Pointer Best Practices ---" << endl;
    
    // 1. Prefer make_unique and make_shared
    auto bestPtr = make_unique<Resource>("BestPractice");
    
    // 2. Don't mix smart pointers and raw pointers for ownership
    Resource* raw = new Resource("RawPointer");  // Bad: potential leak
    delete raw;  // Must remember to delete
    
    // 3. Use unique_ptr for exclusive ownership
    auto exclusivePtr = make_unique<Resource>("Exclusive");
    
    // 4. Use shared_ptr for shared ownership
    auto sharedOwnership1 = make_shared<Resource>("SharedOwner1");
    auto sharedOwnership2 = sharedOwnership1;  // Shared ownership
    
    // 5. Use weak_ptr to break cycles
    weak_ptr<Resource> breaker = sharedOwnership1;
    
    cout << "Final reference count: " << sharedOwnership1.use_count() << endl;
    
    // All resources automatically cleaned up when going out of scope
    cout << "\nEnd of main - all resources cleaned up automatically!" << endl;
    
    return 0;
}
```

## constexpr and consteval

Compile-time computation features.

### Exercise 6: Compile-Time Computation

Complete this constexpr example with errors:

```cpp
#include <iostream>
#include <array>
using namespace std;

// constexpr function - can be evaluated at compile time
constexpr int square(int x) {
    return x * x;
}

// More complex constexpr function
constexpr int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// C++20 consteval - always evaluated at compile time
consteval int compileTimeOnly(int x) {
    return x * 2;
}

// Class with constexpr members
class constexprClass {
private:
    int value;
    
public:
    constexpr constexprClass(int v) : value(v) {}
    
    constexpr int getValue() const { return value; }
    constexpr void setValue(int v) { value = v; }
    
    constexpr int compute() const { return value * value + 1; }
};

// constexpr with templates
template<int N>
struct Fibonacci {
    static constexpr int value = Fibonacci<N-1>::value + Fibonacci<N-2>::value;
};

template<>
struct Fibonacci<0> {
    static constexpr int value = 0;
};

template<>
struct Fibonacci<1> {
    static constexpr int value = 1;
};

int main() {
    cout << "=== Compile-Time Computation Demo ===" << endl;
    
    // Compile-time evaluation
    constexpr int compileTimeResult = square(5);  // Evaluated at compile time
    cout << "Compile-time square of 5: " << compileTimeResult << endl;
    
    constexpr int factorialResult = factorial(5);  // 120
    cout << "Compile-time factorial of 5: " << factorialResult << endl;
    
    // Runtime evaluation (still works)
    int runtimeValue = 7;
    int runtimeResult = square(runtimeValue);  // Evaluated at runtime
    cout << "Runtime square of 7: " << runtimeResult << endl;
    
    // Error: consteval function must be called at compile time
    // int runtimeConsteval = compileTimeOnly(runtimeValue);  // Error!
    
    // Correct: consteval function called with compile-time value
    constexpr int constevalResult = compileTimeOnly(10);
    cout << "Compile-time only result: " << constevalResult << endl;
    
    // constexpr class usage
    constexpr constexprObj(42);
    constexpr int objValue = constexprObj.getValue();
    constexpr int computed = constexprObj.compute();
    
    cout << "Constexpr object value: " << objValue << endl;
    cout << "Constexpr computed value: " << computed << endl;
    
    // Non-constexpr usage (runtime)
    constexprClass runtimeObj(100);
    cout << "Runtime object value: " << runtimeObj.getValue() << endl;
    
    // constexpr with arrays
    constexpr size_t arraySize = factorial(4);  // 24
    array<int, arraySize> compileTimeArray;  // Size determined at compile time
    
    for (size_t i = 0; i < arraySize; i++) {
        compileTimeArray[i] = static_cast<int>(i);
    }
    
    cout << "First few elements of compile-time sized array: ";
    for (size_t i = 0; i < 5; i++) {
        cout << compileTimeArray[i] << " ";
    }
    cout << endl;
    
    // Template with constexpr values
    constexpr int fib5 = Fibonacci<5>::value;
    cout << "Compile-time Fibonacci(5): " << fib5 << endl;
    
    // if constexpr (C++17) - compile-time conditional
    auto processValue = [](auto value) {
        if constexpr (is_integral_v<decltype(value)>) {
            cout << "Processing integer: " << value * 2 << endl;
        } else if constexpr (is_floating_point_v<decltype(value)>) {
            cout << "Processing float: " << value * 1.5 << endl;
        } else {
            cout << "Processing other type" << endl;
        }
    };
    
    processValue(42);      // Calls integer branch
    processValue(3.14);    // Calls float branch
    processValue("text");  // Calls other branch
    
    // constexpr with user-defined literals
    class Length {
    private:
        double meters;
        
    public:
        constexpr Length(double m) : meters(m) {}
        
        constexpr double getMeters() const { return meters; }
        constexpr double getCentimeters() const { return meters * 100; }
        
        constexpr Length operator+(const Length& other) const {
            return Length(meters + other.meters);
        }
    };
    
    // User-defined literal for Length
    consteval Length operator""_m(long double meters) {
        return Length(static_cast<double>(meters));
    }
    
    consteval Length operator""_cm(long double cm) {
        return Length(static_cast<double>(cm) / 100.0);
    }
    
    // Using user-defined literals with compile-time evaluation
    constexpr auto length1 = 5.0_m;
    constexpr auto length2 = 100.0_cm;
    constexpr auto totalLength = length1 + length2;
    
    cout << "\nUser-defined literals with constexpr:" << endl;
    cout << "Length 1: " << length1.getMeters() << "m" << endl;
    cout << "Length 2: " << length2.getMeters() << "m" << endl;
    cout << "Total: " << totalLength.getMeters() << "m" << endl;
    cout << "Total in cm: " << totalLength.getCentimeters() << "cm" << endl;
    
    // Compile-time string operations (C++20 concepts)
    constexpr auto stringLength = [](const char* str) {
        size_t len = 0;
        while (str[len] != '\0') ++len;
        return len;
    };
    
    constexpr size_t helloLen = stringLength("Hello");
    cout << "Compile-time string length of 'Hello': " << helloLen << endl;
    
    // Error handling in constexpr functions
    constexpr auto safeDivide = [](int a, int b) -> int {
        if (b == 0) {
            // In constexpr context, this would cause compilation error
            // In runtime context, this would throw
            return 0;  // Simplified for demo
        }
        return a / b;
    };
    
    constexpr int safeResult = safeDivide(10, 2);
    cout << "Safe divide result: " << safeResult << endl;
    
    // Complex constexpr example: compile-time array processing
    constexpr auto processArray = [](const array<int, 5>& input) {
        array<int, 5> result{};
        for (size_t i = 0; i < input.size(); ++i) {
            result[i] = input[i] * input[i];  // Square each element
        }
        return result;
    };
    
    constexpr array<int, 5> inputArray = {1, 2, 3, 4, 5};
    constexpr auto processedArray = processArray(inputArray);
    
    cout << "Compile-time processed array: ";
    for (const auto& val : processedArray) {
        cout << val << " ";
    }
    cout << endl;
    
    // Using constexpr values for template parameters
    template<size_t N>
    constexpr size_t getCompileTimeValue() {
        return N * 2;
    }
    
    constexpr size_t templateValue = getCompileTimeValue<factorial(3)>();
    cout << "Template value (factorial(3) * 2): " << templateValue << endl;
    
    cout << "\nCompile-time computation completed successfully!" << endl;
    
    return 0;
}
```

## Concepts (C++20)

Concepts provide a way to constrain templates with requirements.

### Exercise 7: Concepts

Complete this concepts example:

```cpp
#include <iostream>
#include <type_traits>
#include <concepts>
#include <vector>
#include <string>
using namespace std;

// Define a concept for integral types
template<typename T>
concept Integral = is_integral_v<T>;

// Define a concept for floating-point types
template<typename T>
concept FloatingPoint = is_floating_point_v<T>;

// Define a concept for arithmetic types
template<typename T>
concept Arithmetic = Integral<T> || FloatingPoint<T>;

// Define a concept for types that are comparable
template<typename T>
concept Comparable = requires(T a, T b) {
    { a < b } -> same_as<bool>;
    { a > b } -> same_as<bool>;
    { a == b } -> same_as<bool>;
};

// Define a concept for types that support addition
template<typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> same_as<T>;
};

// Define a concept for types that have a size() method
template<typename T>
concept HasSize = requires(const T& t) {
    { t.size() } -> same_as<size_t>;
};

// Function constrained by concepts
template<Integral T>
T multiplyByTwo(T value) {
    return value * 2;
}

template<Arithmetic T>
T safeDivide(T numerator, T denominator) {
    if (denominator == T{}) {
        throw invalid_argument("Division by zero");
    }
    return numerator / denominator;
}

template<Comparable T>
T maximum(T a, T b) {
    return (a > b) ? a : b;
}

template<Addable T>
T sum(T a, T b) {
    return a + b;
}

// More complex concept combining multiple requirements
template<typename T>
concept PrintableAndArithmetic = Arithmetic<T> && requires(T t, ostream& os) {
    { os << t } -> same_as<ostream&>;
};

template<PrintableAndArithmetic T>
void printAndDouble(T value) {
    cout << "Value: " << value << ", Doubled: " << (value * 2) << endl;
}

// Concept for containers
template<typename Container>
concept SequenceContainer = HasSize<Container> &&
                           requires(Container c) {
                               typename Container::value_type;
                               { c.begin() } -> input_or_output_iterator;
                               { c.end() } -> input_or_output_iterator;
                           };

template<SequenceContainer Container>
void printContainer(const Container& container) {
    cout << "Container contents: ";
    for (const auto& element : container) {
        cout << element << " ";
    }
    cout << endl;
}

// Function that works with any type that supports indexing
template<typename Container>
concept Indexable = requires(Container c, size_t i) {
    { c[i] } -> same_as<typename Container::reference>;
};

template<Indexable Container>
auto getElement(Container& container, size_t index) -> typename Container::reference {
    return container[index];
}

int main() {
    cout << "=== Concepts Demo ===" << endl;
    
    // Using constrained functions
    cout << "multiplyByTwo(5): " << multiplyByTwo(5) << endl;
    cout << "multiplyByTwo(100L): " << multiplyByTwo(100L) << endl;
    
    cout << "safeDivide(10.0, 3.0): " << safeDivide(10.0, 3.0) << endl;
    cout << "safeDivide(15, 4): " << safeDivide(15, 4) << endl;
    
    cout << "maximum(5, 3): " << maximum(5, 3) << endl;
    cout << "maximum(3.14, 2.71): " << maximum(3.14, 2.71) << endl;
    
    cout << "sum(5, 10): " << sum(5, 10) << endl;
    cout << "sum(3.5, 2.5): " << sum(3.5, 2.5) << endl;
    
    // Using the printable and arithmetic concept
    printAndDouble(42);
    printAndDouble(3.14159);
    
    // Using with containers
    vector<int> numbers = {1, 2, 3, 4, 5};
    vector<double> doubles = {1.1, 2.2, 3.3};
    string text = "Hello";
    
    printContainer(numbers);
    printContainer(doubles);
    printContainer(text);
    
    // Using indexable concept
    cout << "Element at index 2 in numbers: " << getElement(numbers, 2) << endl;
    cout << "Element at index 1 in text: " << getElement(text, 1) << endl;
    
    // Demonstrating concept error messages
    cout << "\n--- Concept Error Demonstration ---" << endl;
    
    // This would cause a compilation error with clear message:
    // struct NonArithmetic {};
    // NonArithmetic obj;
    // auto result = multiplyByTwo(obj);  // Error: NonArithmetic doesn't satisfy Integral
    
    // Complex concept example: a function that works with sortable containers
    template<typename Container>
    concept Sortable = SequenceContainer<Container> && 
                      Comparable<typename Container::value_type> &&
                      requires(Container c) {
                          sort(c.begin(), c.end());
                      };
    
    // This function only works with sortable containers
    template<Sortable Container>
    void sortAndDisplay(Container& container) {
        sort(container.begin(), container.end());
        cout << "Sorted container: ";
        for (const auto& element : container) {
            cout << element << " ";
        }
        cout << endl;
    }
    
    vector<int> unsorted = {5, 2, 8, 1, 9, 3};
    cout << "Before sorting: ";
    for (const auto& element : unsorted) {
        cout << element << " ";
    }
    cout << endl;
    
    sortAndDisplay(unsorted);
    
    // Custom concept for a "mathematical" type
    template<typename T>
    concept Mathematical = Arithmetic<T> && 
                          requires(T a) {
                              { abs(a) } -> same_as<T>;
                              { sqrt(a) } -> same_as<T>;
                          };
    
    // Note: abs and sqrt are not universally available, so this is conceptual
    // In practice, you'd need to constrain this further or provide fallbacks
    
    cout << "\nConcepts demonstration completed!" << endl;
    
    return 0;
}
```

## Modules (C++20)

Modules provide a modern alternative to headers.

### Exercise 8: Modules

Create a simple module example:

```cpp
// math_operations.cppm
export module math_operations;

import <iostream>;
import <vector>;
import <string>;

export namespace math_utils {
    // Exported function
    int add(int a, int b) {
        return a + b;
    }
    
    // Exported function
    int multiply(int a, int b) {
        return a * b;
    }
    
    // Exported class
    export class Calculator {
    public:
        int add(int a, int b) const {
            return a + b;
        }
        
        int multiply(int a, int b) const {
            return a * b;
        }
        
        double divide(double a, double b) const {
            if (b != 0) {
                return a / b;
            }
            throw std::runtime_error("Division by zero");
        }
    };
    
    // Internal (non-exported) helper function
    namespace detail {
        bool is_prime_helper(int n) {
            if (n < 2) return false;
            for (int i = 2; i * i <= n; i++) {
                if (n % i == 0) return false;
            }
            return true;
        }
    }
    
    // Exported function using internal helper
    bool is_prime(int n) {
        return detail::is_prime_helper(n);
    }
}

// Export a partition (implementation part)
module : private;

namespace internal {
    // This is not visible to importers
    void internal_helper() {
        std::cout << "Internal helper function" << std::endl;
    }
}
```

```cpp
// main.cpp
import math_operations;
import <iostream>;
import <vector>;

int main() {
    std::cout << "=== Modules Demo ===" << endl;
    
    // Using exported functions
    std::cout << "5 + 3 = " << math_utils::add(5, 3) << std::endl;
    std::cout << "5 * 3 = " << math_utils::multiply(5, 3) << std::endl;
    
    // Using exported class
    math_utils::Calculator calc;
    std::cout << "Calculator add: " << calc.add(10, 20) << std::endl;
    std::cout << "Calculator multiply: " << calc.multiply(4, 5) << std::endl;
    
    // Using exported function
    std::cout << "Is 17 prime? " << (math_utils::is_prime(17) ? "Yes" : "No") << std::endl;
    std::cout << "Is 15 prime? " << (math_utils::is_prime(15) ? "Yes" : "No") << std::endl;
    
    // Internal functions are not accessible
    // internal::internal_helper();  // Error: not exported
    
    return 0;
}
```

## Coroutines (C++20)

Coroutines enable cooperative multitasking.

### Exercise 9: Coroutines

Complete this coroutine example:

```cpp
#include <iostream>
#include <coroutine>
#include <memory>
#include <exception>
using namespace std;

// Simple generator coroutine
template<typename T>
class Generator {
public:
    struct promise_type {
        T value;
        Generator<T> get_return_object() {
            return Generator{handle_type::from_promise(*this)};
        }
        
        suspend_always initial_suspend() { return {}; }
        suspend_always final_suspend() noexcept { return {}; }
        
        suspend_always yield_value(T v) {
            value = v;
            return {};
        }
        
        void return_void() {}
        
        void unhandled_exception() {
            std::terminate();
        }
    };
    
    using handle_type = coroutine_handle<promise_type>;
    
    Generator(handle_type h) : coro(h) {}
    ~Generator() {
        if (coro) coro.destroy();
    }
    
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;
    
    Generator(Generator&& other) noexcept : coro(other.coro) {
        other.coro = nullptr;
    }
    
    Generator& operator=(Generator&& other) noexcept {
        if (this != &other) {
            if (coro) coro.destroy();
            coro = other.coro;
            other.coro = nullptr;
        }
        return *this;
    }
    
    T operator()() {
        coro.resume();
        return coro.promise().value;
    }
    
    bool done() const {
        return coro.done();
    }
    
private:
    handle_type coro;
};

// Function that returns a generator
Generator<int> fibonacci_generator(int count) {
    int a = 0, b = 1;
    
    for (int i = 0; i < count; ++i) {
        co_yield a;
        int temp = a + b;
        a = b;
        b = temp;
    }
}

// Another generator example
Generator<string> word_generator() {
    co_yield "Hello";
    co_yield "Modern";
    co_yield "C++";
    co_yield "World";
}

int main() {
    cout << "=== Coroutines Demo ===" << endl;
    
    // Using fibonacci generator
    cout << "Fibonacci sequence: ";
    auto fib_gen = fibonacci_generator(10);
    
    while (!fib_gen.done()) {
        cout << fib_gen() << " ";
    }
    cout << endl;
    
    // Using word generator
    cout << "Word sequence: ";
    auto word_gen = word_generator();
    
    while (!word_gen.done()) {
        cout << word_gen() << " ";
    }
    cout << endl;
    
    // Simple async coroutine (conceptual)
    cout << "\nCoroutine demonstration completed!" << endl;
    
    return 0;
}
```

## Best Practices and Modern C++ Guidelines

### Exercise 10: Modern C++ Best Practices

Demonstrate modern C++ best practices:

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
using namespace std;

// Modern C++ class design with best practices
class ModernClass {
private:
    unique_ptr<vector<int>> data;
    string name;
    
public:
    // Constructor using member initializer list
    explicit ModernClass(const string& n, size_t size = 0) 
        : data(make_unique<vector<int>>(size)), name(n) {}
    
    // Rule of Zero - compiler-generated special functions are sufficient
    // because we're using RAII types (unique_ptr, string, vector)
    
    // Move operations are automatically optimal with RAII types
    ModernClass(ModernClass&&) = default;
    ModernClass& operator=(ModernClass&&) = default;
    
    // Accessors
    const vector<int>& getData() const { return *data; }
    vector<int>& getData() { return *data; }
    const string& getName() const { return name; }
    
    // Modern method with auto return type
    auto size() const -> size_t { return data->size(); }
    
    // Method with range-based for and auto
    void printData() const {
        cout << name << ": ";
        for (const auto& value : *data) {
            cout << value << " ";
        }
        cout << endl;
    }
    
    // Method using STL algorithms
    int sum() const {
        return accumulate(data->begin(), data->end(), 0);
    }
    
    // Method using lambda
    void applyFunction(const function<int(int)>& func) {
        transform(data->begin(), data->end(), data->begin(), func);
    }
};

// Function with modern parameter passing
void processContainer(auto&& container) {  // C++20 abbreviated function template
    for (auto& element : container) {
        element *= 2;  // Double each element
    }
}

// Template function with constraints (C++20)
template<typename T>
requires integral<T>  // C++20 concept requirement
T square(T value) {
    return value * value;
}

int main() {
    cout << "=== Modern C++ Best Practices ===" << endl;
    
    // 1. Use auto for complex types
    auto myVector = vector<int>{1, 2, 3, 4, 5};
    auto myString = string{"Modern C++"};
    
    // 2. Use range-based for loops
    cout << "Vector contents: ";
    for (const auto& element : myVector) {
        cout << element << " ";
    }
    cout << endl;
    
    // 3. Use algorithms instead of manual loops
    auto evenCount = count_if(myVector.begin(), myVector.end(),
                              [](int n) { return n % 2 == 0; });
    cout << "Even numbers: " << evenCount << endl;
    
    // 4. Use smart pointers
    auto myClass = make_unique<ModernClass>("TestClass", 5);
    iota(myClass->getData().begin(), myClass->getData().end(), 1);  // Fill with 1, 2, 3...
    myClass->printData();
    
    // 5. Use move semantics appropriately
    vector<string> source = {"hello", "world", "cpp"};
    vector<string> destination;
    
    // Move elements efficiently
    for (auto& element : source) {
        destination.push_back(move(element));  // Move instead of copy
    }
    
    cout << "After move - source: ";
    for (const auto& element : source) {
        cout << (element.empty() ? "(empty)" : element) << " ";
    }
    cout << endl;
    
    cout << "After move - destination: ";
    for (const auto& element : destination) {
        cout << element << " ";
    }
    cout << endl;
    
    // 6. Use make_unique and make_shared
    auto smartInt = make_unique<int>(42);
    auto smartString = make_shared<string>("Hello Smart Pointers");
    
    cout << "Smart int: " << *smartInt << endl;
    cout << "Smart string: " << *smartString << endl;
    
    // 7. Use const and constexpr appropriately
    constexpr int compileTimeValue = 100;
    const auto runtimeValue = myVector.size();
    
    cout << "Compile-time value: " << compileTimeValue << endl;
    cout << "Runtime value: " << runtimeValue << endl;
    
    // 8. Use structured bindings (C++17)
    auto [minIt, maxIt] = minmax_element(myVector.begin(), myVector.end());
    cout << "Min: " << *minIt << ", Max: " << *maxIt << endl;
    
    // 9. Use if/switch with initializers (C++17)
    if (auto pos = find(myVector.begin(), myVector.end(), 3); pos != myVector.end()) {
        cout << "Found 3 at position: " << (pos - myVector.begin()) << endl;
    }
    
    // 10. Use class template argument deduction (C++17)
    pair p(42, 3.14);  // Compiler deduces pair<int, double>
    cout << "Deduced pair: " << p.first << ", " << p.second << endl;
    
    // 11. Use designated initializers (C++20) for aggregates
    struct Point {
        int x = 0;
        int y = 0;
    };
    
    Point p1 = {.x = 5, .y = 10};  // C++20 designated initializers
    cout << "Point: (" << p1.x << ", " << p1.y << ")" << endl;
    
    // 12. Use concepts for better error messages (if C++20 available)
    cout << "Square of 5: " << square(5) << endl;
    
    // 13. Use string literals
    using namespace string_literals;
    auto str = "Modern C++"s;
    cout << "String literal: " << str << endl;
    
    // 14. Use [[maybe_unused]], [[nodiscard]], etc.
    [[maybe_unused]] int unusedVar = 42;
    
    [[nodiscard]] auto importantFunction() -> int {
        return 123;
    }
    
    // The compiler should warn if we ignore the return value
    importantFunction();  // This might generate a warning
    
    // 15. Use structured bindings with algorithms
    auto [it, count] = make_pair(find(myVector.begin(), myVector.end(), 5), 
                                count(myVector.begin(), myVector.end(), 5));
    if (it != myVector.end()) {
        cout << "Found element with count: " << count << endl;
    }
    
    cout << "\nModern C++ best practices demonstration completed!" << endl;
    
    return 0;
}
```

## Summary

In this chapter, you learned:
- Auto type deduction and its benefits
- Range-based for loops for cleaner iteration
- Lambda expressions and closures
- Move semantics and perfect forwarding
- Smart pointers for automatic memory management
- Compile-time computation with constexpr and consteval
- Concepts for constrained templates (C++20)
- Modules for better compilation (C++20)
- Coroutines for cooperative multitasking (C++20)
- Modern C++ best practices and guidelines

## Key Takeaways

- Use auto to reduce verbosity and improve maintainability
- Prefer range-based for loops over traditional loops
- Use lambda expressions for concise function objects
- Apply move semantics for efficient resource management
- Use smart pointers to prevent memory leaks
- Leverage constexpr for compile-time computation
- Use concepts to constrain templates and improve error messages
- Consider modules for better compilation performance
- Apply modern C++ idioms and best practices consistently

## Common Mistakes to Avoid

1. Overusing auto to the point of reducing code readability
2. Forgetting to use const auto& when iterating large objects
3. Not using move semantics when transferring ownership
4. Mixing smart pointers and raw pointers for ownership
5. Not using make_unique/make_shared consistently
6. Forgetting that constexpr functions must be safe to evaluate at compile time
7. Not constraining templates properly (before C++20 concepts)
8. Using headers instead of modules when available
9. Not following RAII principles consistently
10. Ignoring compiler warnings about unused return values

## Next Steps

Congratulations! You've completed this comprehensive C++ learning journey. You now have a solid foundation in modern C++ programming and are ready to tackle real-world projects. Consider exploring specialized areas like game development, systems programming, or high-performance computing based on your interests.