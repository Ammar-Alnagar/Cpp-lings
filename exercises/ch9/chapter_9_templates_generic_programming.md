# Chapter 9: Templates and Generic Programming

## Overview

This chapter covers templates and generic programming in C++, which allow you to write code that works with multiple data types. You'll learn about function templates, class templates, template specialization, and advanced template features.

## Learning Objectives

By the end of this chapter, you will:
- Understand function templates and how to create them
- Learn about class templates and their implementation
- Master template specialization (full and partial)
- Understand variadic templates and parameter packs
- Learn about template constraints and concepts (C++20)
- Understand SFINAE (Substitution Failure Is Not An Error)
- Learn about perfect forwarding and move semantics with templates
- Understand template metaprogramming basics
- Learn best practices for template design

## Function Templates

Function templates allow you to write functions that work with multiple types.

### Exercise 1: Basic Function Templates

The following code has errors. Find and fix them:

```cpp
#include <iostream>
#include <string>
using namespace std;

// Basic function template
template <typename T>
T max(T a, T b) {  // Error: potential issue with operator<
    if (a > b) {   // Uses operator>, which might not exist for all types
        return a;
    } else {
        return b;
    }
}

// Better implementation with proper comparison
template <typename T>
T maxFixed(T a, T b) {
    if (a < b) {   // Uses operator< (more commonly implemented)
        return b;
    } else {
        return a;
    }
}

// Function template with multiple parameters
template <typename T, typename U>
auto multiply(T a, U b) -> decltype(a * b) {  // Trailing return type
    return a * b;
}

int main() {
    // Using function template with different types
    cout << "Max of 5 and 10: " << max(5, 10) << endl;
    cout << "Max of 3.14 and 2.71: " << max(3.14, 2.71) << endl;
    cout << "Max of 'a' and 'z': " << max('a', 'z') << endl;
    
    // Error: trying to use max with different types
    // cout << max(5, 3.14) << endl;  // Error: can't deduce T
    
    // Correct way: explicit template instantiation
    cout << "Max of 5 and 3.14 (explicit): " << max<double>(5.0, 3.14) << endl;
    
    // Using template with multiple types
    cout << "Multiply int and double: " << multiply(5, 2.5) << endl;
    cout << "Multiply float and int: " << multiply(3.5f, 4) << endl;
    
    return 0;
}
```

### Exercise 2: Advanced Function Templates

Complete this function template example with errors:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// Template function to find minimum in a container
template <typename Container>
auto findMin(const Container& container) -> typename Container::value_type {
    if (container.empty()) {
        // Error: no way to indicate "no minimum" for arbitrary types
        throw runtime_error("Container is empty");
    }
    
    auto minElement = container.begin();
    for (auto it = container.begin(); it != container.end(); ++it) {
        if (*it < *minElement) {
            minElement = it;
        }
    }
    return *minElement;
}

// More efficient version using STL algorithm
template <typename Container>
auto findMinSTL(const Container& container) -> typename Container::value_type {
    if (container.empty()) {
        throw runtime_error("Container is empty");
    }
    
    auto minIt = min_element(container.begin(), container.end());
    return *minIt;
}

// Template function with constraints (C++20 concepts would be better)
template <typename T>
T power(T base, unsigned int exponent) {
    T result = 1;
    for (unsigned int i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}

// Swap function template
template <typename T>
void mySwap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

// Better swap with move semantics (C++11)
template <typename T>
void mySwapOptimized(T& a, T& b) {
    T temp = std::move(a);  // Move instead of copy when possible
    a = std::move(b);
    b = std::move(temp);
}

int main() {
    vector<int> numbers = {5, 2, 8, 1, 9, 3};
    cout << "Minimum in vector: " << findMin(numbers) << endl;
    cout << "Minimum in vector (STL): " << findMinSTL(numbers) << endl;
    
    vector<double> doubles = {3.14, 2.71, 1.41, 0.57};
    cout << "Minimum in doubles: " << findMin(doubles) << endl;
    
    // Power function
    cout << "2^10 = " << power(2, 10) << endl;
    cout << "3.5^3 = " << power(3.5, 3) << endl;
    
    // Swap function
    int x = 10, y = 20;
    cout << "Before swap: x=" << x << ", y=" << y << endl;
    mySwap(x, y);
    cout << "After swap: x=" << x << ", y=" << y << endl;
    
    // Error: what if types don't support required operations?
    // struct NoComparison {};
    // vector<NoComparison> noCompVec;
    // findMin(noCompVec);  // Would cause compilation error
    
    return 0;
}
```

## Class Templates

Class templates allow you to create classes that work with multiple types.

### Exercise 3: Basic Class Templates

Complete this class template example with errors:

```cpp
#include <iostream>
#include <string>
using namespace std;

// Generic Pair class template
template <typename T, typename U>
class Pair {
private:
    T first;
    U second;
    
public:
    // Constructor
    Pair(const T& f, const U& s) : first(f), second(s) {}
    
    // Getters
    T getFirst() const { return first; }
    U getSecond() const { return second; }
    
    // Setters
    void setFirst(const T& f) { first = f; }
    void setSecond(const U& s) { second = s; }
    
    // Display function
    void display() const {
        cout << "Pair: (" << first << ", " << second << ")" << endl;
    }
    
    // Equality operator
    bool operator==(const Pair<T, U>& other) const {
        return first == other.first && second == other.second;
    }
};

// Generic Container class template
template <typename T>
class SimpleContainer {
private:
    T data;
    bool hasValue;
    
public:
    SimpleContainer() : hasValue(false) {}
    
    SimpleContainer(const T& value) : data(value), hasValue(true) {}
    
    void setValue(const T& value) {
        data = value;
        hasValue = true;
    }
    
    T getValue() const {
        if (!hasValue) {
            throw runtime_error("No value set");
        }
        return data;
    }
    
    bool isEmpty() const { return !hasValue; }
    
    void clear() { hasValue = false; }
    
    // Equality operator
    bool operator==(const SimpleContainer<T>& other) const {
        if (isEmpty() && other.isEmpty()) return true;
        if (isEmpty() || other.isEmpty()) return false;
        return data == other.data;
    }
};

int main() {
    // Using Pair with different types
    Pair<int, string> intStringPair(42, "Hello");
    intStringPair.display();
    
    Pair<double, int> doubleIntPair(3.14, 100);
    doubleIntPair.display();
    
    // Using SimpleContainer
    SimpleContainer<string> stringContainer("C++ Templates");
    cout << "Container value: " << stringContainer.getValue() << endl;
    
    SimpleContainer<int> intContainer(123);
    cout << "Int container value: " << intContainer.getValue() << endl;
    
    // Testing equality
    Pair<int, string> pair1(1, "test");
    Pair<int, string> pair2(1, "test");
    Pair<int, string> pair3(2, "test");
    
    cout << "pair1 == pair2: " << (pair1 == pair2) << endl;
    cout << "pair1 == pair3: " << (pair1 == pair3) << endl;
    
    return 0;
}
```

### Exercise 4: Advanced Class Templates

Complete this advanced class template example:

```cpp
#include <iostream>
#include <vector>
#include <memory>
using namespace std;

// Generic Stack class template
template <typename T>
class Stack {
private:
    vector<T> elements;
    
public:
    // Default constructor
    Stack() = default;
    
    // Copy constructor
    Stack(const Stack<T>& other) : elements(other.elements) {}
    
    // Move constructor
    Stack(Stack<T>&& other) noexcept : elements(move(other.elements)) {}
    
    // Copy assignment operator
    Stack& operator=(const Stack<T>& other) {
        if (this != &other) {
            elements = other.elements;
        }
        return *this;
    }
    
    // Move assignment operator
    Stack& operator=(Stack<T>&& other) noexcept {
        if (this != &other) {
            elements = move(other.elements);
        }
        return *this;
    }
    
    // Push element onto stack
    void push(const T& element) {
        elements.push_back(element);
    }
    
    // Move version of push
    void push(T&& element) {
        elements.push_back(move(element));
    }
    
    // Pop element from stack
    void pop() {
        if (elements.empty()) {
            throw runtime_error("Stack is empty");
        }
        elements.pop_back();
    }
    
    // Get top element
    T& top() {
        if (elements.empty()) {
            throw runtime_error("Stack is empty");
        }
        return elements.back();
    }
    
    const T& top() const {
        if (elements.empty()) {
            throw runtime_error("Stack is empty");
        }
        return elements.back();
    }
    
    // Get size
    size_t size() const { return elements.size(); }
    
    // Check if empty
    bool empty() const { return elements.empty(); }
    
    // Clear the stack
    void clear() { elements.clear(); }
};

// Generic Smart Pointer class template
template <typename T>
class CustomSmartPointer {
private:
    T* ptr;
    
public:
    // Constructor
    explicit CustomSmartPointer(T* p = nullptr) : ptr(p) {}
    
    // Destructor
    ~CustomSmartPointer() { delete ptr; }
    
    // Copy constructor - implement move semantics instead
    // Copy constructor would create ownership issues
    CustomSmartPointer(const CustomSmartPointer&) = delete;
    
    // Copy assignment - implement move semantics instead
    CustomSmartPointer& operator=(const CustomSmartPointer&) = delete;
    
    // Move constructor
    CustomSmartPointer(CustomSmartPointer&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }
    
    // Move assignment operator
    CustomSmartPointer& operator=(CustomSmartPointer&& other) noexcept {
        if (this != &other) {
            delete ptr;  // Clean up current resource
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
    
    // Dereference operator
    T& operator*() const { return *ptr; }
    
    // Arrow operator
    T* operator->() const { return ptr; }
    
    // Get raw pointer
    T* get() const { return ptr; }
    
    // Reset pointer
    void reset(T* p = nullptr) {
        delete ptr;
        ptr = p;
    }
    
    // Release ownership
    T* release() {
        T* temp = ptr;
        ptr = nullptr;
        return temp;
    }
    
    // Boolean conversion
    explicit operator bool() const { return ptr != nullptr; }
};

int main() {
    cout << "=== Stack Template Demo ===" << endl;
    
    // Integer stack
    Stack<int> intStack;
    intStack.push(10);
    intStack.push(20);
    intStack.push(30);
    
    cout << "Integer stack size: " << intStack.size() << endl;
    while (!intStack.empty()) {
        cout << "Top: " << intStack.top() << endl;
        intStack.pop();
    }
    
    // String stack
    Stack<string> stringStack;
    stringStack.push("First");
    stringStack.push("Second");
    stringStack.push("Third");
    
    cout << "\nString stack size: " << stringStack.size() << endl;
    while (!stringStack.empty()) {
        cout << "Top: " << stringStack.top() << endl;
        stringStack.pop();
    }
    
    cout << "\n=== Smart Pointer Template Demo ===" << endl;
    
    // Using custom smart pointer
    {
        CustomSmartPointer<int> smartPtr(new int(42));
        if (smartPtr) {
            cout << "Smart pointer value: " << *smartPtr << endl;
        }
        
        // Move the smart pointer
        CustomSmartPointer<int> movedPtr = move(smartPtr);
        if (!smartPtr) {
            cout << "Original smart pointer is now null" << endl;
        }
        if (movedPtr) {
            cout << "Moved smart pointer value: " << *movedPtr << endl;
        }
    }  // Smart pointer automatically deletes memory
    
    return 0;
}
```

## Template Specialization

Template specialization allows you to provide specific implementations for particular types.

### Exercise 5: Template Specialization

Complete this template specialization example:

```cpp
#include <iostream>
#include <string>
#include <cstring>
using namespace std;

// Generic function template
template <typename T>
bool isEqual(const T& a, const T& b) {
    cout << "Using generic version" << endl;
    return a == b;
}

// Full specialization for char*
template <>
bool isEqual<char*>(char* const& a, char* const& b) {
    cout << "Using char* specialization" << endl;
    return strcmp(a, b) == 0;
}

// Full specialization for const char*
template <>
bool isEqual<const char*>(const char* const& a, const char* const& b) {
    cout << "Using const char* specialization" << endl;
    return strcmp(a, b) == 0;
}

// Class template
template <typename T>
class DataProcessor {
public:
    static void process(const T& data) {
        cout << "Generic processing for type T" << endl;
        cout << "Data: " << data << endl;
    }
};

// Full specialization for int
template <>
class DataProcessor<int> {
public:
    static void process(const int& data) {
        cout << "Specialized processing for int" << endl;
        cout << "Integer data: " << data << ", Square: " << data * data << endl;
    }
};

// Full specialization for bool
template <>
class DataProcessor<bool> {
public:
    static void process(const bool& data) {
        cout << "Specialized processing for bool" << endl;
        cout << "Boolean data: " << (data ? "true" : "false") << endl;
    }
};

// Full specialization for char*
template <>
class DataProcessor<char*> {
public:
    static void process(char* const& data) {
        cout << "Specialized processing for char*" << endl;
        cout << "C-string data: " << data << endl;
        cout << "Length: " << strlen(data) << endl;
    }
};

int main() {
    cout << "=== Function Template Specialization ===" << endl;
    
    // Generic version
    cout << "Generic: " << isEqual(5, 5) << endl;
    cout << "Generic: " << isEqual(3.14, 2.71) << endl;
    cout << "Generic: " << isEqual(string("hello"), string("hello")) << endl;
    
    // Specialized versions
    char str1[] = "hello";
    char str2[] = "hello";
    char str3[] = "world";
    
    cout << "Specialized: " << isEqual(str1, str2) << endl;  // Should be true
    cout << "Specialized: " << isEqual(str1, str3) << endl;  // Should be false
    
    const char* cstr1 = "test";
    const char* cstr2 = "test";
    cout << "Specialized const char*: " << isEqual(cstr1, cstr2) << endl;
    
    cout << "\n=== Class Template Specialization ===" << endl;
    
    // Generic version
    DataProcessor<double>::process(3.14159);
    DataProcessor<string>::process("Hello, World!");
    
    // Specialized versions
    DataProcessor<int>::process(42);
    DataProcessor<bool>::process(true);
    DataProcessor<bool>::process(false);
    
    char testStr[] = "Template Specialization";
    DataProcessor<char*>::process(testStr);
    
    return 0;
}
```

### Exercise 6: Partial Template Specialization

Complete this partial specialization example:

```cpp
#include <iostream>
#include <type_traits>
using namespace std;

// Primary template
template <typename T, typename U>
class Container {
public:
    void info() {
        cout << "Primary template: T=" << typeid(T).name() 
             << ", U=" << typeid(U).name() << endl;
    }
};

// Partial specialization for when U is int
template <typename T>
class Container<T, int> {
public:
    void info() {
        cout << "Partial specialization: T=" << typeid(T).name() 
             << ", U=int" << endl;
    }
};

// Partial specialization for when both are the same type
template <typename T>
class Container<T, T> {
public:
    void info() {
        cout << "Same type specialization: T=T=" << typeid(T).name() << endl;
    }
};

// Partial specialization for pointer types
template <typename T, typename U>
class Container<T*, U*> {
public:
    void info() {
        cout << "Pointer types specialization: T*=" << typeid(T*).name() 
             << ", U*=" << typeid(U*).name() << endl;
    }
};

// Template with enable_if for conditional compilation
template <typename T>
typename enable_if<is_arithmetic<T>::value, T>::type
conditionalProcess(T value) {
    cout << "Processing arithmetic type: " << value << endl;
    return value * 2;
}

template <typename T>
typename enable_if<!is_arithmetic<T>::value, T>::type
conditionalProcess(T value) {
    cout << "Processing non-arithmetic type" << endl;
    return value;
}

int main() {
    cout << "=== Partial Specialization Demo ===" << endl;
    
    Container<double, string> c1;
    c1.info();
    
    Container<double, int> c2;  // Uses partial specialization
    c2.info();
    
    Container<string, string> c3;  // Uses same type specialization
    c3.info();
    
    Container<int*, double*> c4;  // Uses pointer types specialization
    c4.info();
    
    cout << "\n=== Conditional Compilation Demo ===" << endl;
    
    int intVal = 5;
    string strVal = "hello";
    
    auto processedInt = conditionalProcess(intVal);
    auto processedStr = conditionalProcess(strVal);
    
    cout << "Processed int: " << processedInt << endl;
    cout << "Processed string: " << processedStr << endl;
    
    return 0;
}
```

## Variadic Templates

Variadic templates allow templates to take a variable number of arguments.

### Exercise 7: Variadic Templates

Complete this variadic template example:

```cpp
#include <iostream>
#include <string>
using namespace std;

// Base case for recursive variadic template
template <typename T>
void print(T&& t) {
    cout << t << endl;
}

// Recursive variadic template
template <typename T, typename... Args>
void print(T&& t, Args&&... args) {
    cout << t << " ";
    print(forward<Args>(args)...);  // Forward arguments recursively
}

// Count parameters using variadic templates
template <typename... Args>
constexpr size_t countParams(Args&&...) {
    return sizeof...(Args);  // Count the number of arguments
}

// Sum function using variadic templates
template <typename T>
T sum(T&& t) {
    return t;
}

template <typename T, typename... Args>
T sum(T&& t, Args&&... args) {
    return t + sum(forward<Args>(args)...);
}

// Perfect forwarding example
template <typename Func, typename... Args>
auto callFunction(Func&& func, Args&&... args) -> decltype(func(forward<Args>(args)...)) {
    return func(forward<Args>(args)...);
}

// Function to test perfect forwarding
void testFunction(int& x, const string& str, double d) {
    cout << "testFunction called with: " << x << ", " << str << ", " << d << endl;
    x = 42;  // Modify the reference
}

int main() {
    cout << "=== Variadic Template Demo ===" << endl;
    
    // Print with variable arguments
    print("Hello");
    print("Hello", "World");
    print("Number:", 42, "Pi:", 3.14159);
    print("Mixed types:", 100, 3.14, 'X', "end");
    
    // Count parameters
    cout << "\nParameter counts:" << endl;
    cout << "0 params: " << countParams() << endl;
    cout << "1 param: " << countParams(42) << endl;
    cout << "3 params: " << countParams("test", 3.14, 100) << endl;
    
    // Sum function
    cout << "\nSums:" << endl;
    cout << "Sum of 5: " << sum(5) << endl;
    cout << "Sum of 1, 2, 3: " << sum(1, 2, 3) << endl;
    cout << "Sum of 1.5, 2.5, 3.5: " << sum(1.5, 2.5, 3.5) << endl;
    cout << "Sum of 10, 20, 30, 40: " << sum(10, 20, 30, 40) << endl;
    
    // Perfect forwarding example
    cout << "\nPerfect forwarding demo:" << endl;
    int value = 10;
    string text = "forwarded";
    double pi = 3.14159;
    
    cout << "Before: value = " << value << endl;
    callFunction(testFunction, value, text, pi);
    cout << "After: value = " << value << endl;
    
    return 0;
}
```

## Template Constraints and Concepts (C++20)

Concepts provide a way to constrain templates with requirements.

### Exercise 8: Concepts (C++20)

Complete this concepts example:

```cpp
#include <iostream>
#include <concepts>
#include <type_traits>
using namespace std;

// Define a concept for integral types
template <typename T>
concept Integral = is_integral_v<T>;

// Define a concept for comparable types
template <typename T>
concept Comparable = requires(T a, T b) {
    { a < b } -> convertible_to<bool>;
    { a > b } -> convertible_to<bool>;
    { a == b } -> convertible_to<bool>;
};

// Define a concept for addable types
template <typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> same_as<T>;
};

// Function constrained by concepts
template <Integral T>
T multiplyByTwo(T value) {
    return value * 2;
}

template <Comparable T>
T maximum(T a, T b) {
    return (a > b) ? a : b;
}

template <Addable T>
T add(T a, T b) {
    return a + b;
}

// More complex concept combining multiple requirements
template <typename T>
concept Number = Integral<T> || is_floating_point_v<T>;

template <Number T>
T square(T x) {
    return x * x;
}

// Concept for types that have a size() method
template <typename T>
concept HasSize = requires(const T& t) {
    { t.size() } -> same_as<size_t>;
};

template <HasSize Container>
void printSize(const Container& c) {
    cout << "Container size: " << c.size() << endl;
}

int main() {
    cout << "=== Concepts Demo ===" << endl;
    
    // Using constrained functions
    cout << "multiplyByTwo(5): " << multiplyByTwo(5) << endl;
    cout << "multiplyByTwo(100L): " << multiplyByTwo(100L) << endl;
    
    cout << "maximum(10, 20): " << maximum(10, 20) << endl;
    cout << "maximum(3.14, 2.71): " << maximum(3.14, 2.71) << endl;
    
    cout << "add(5, 10): " << add(5, 10) << endl;
    cout << "add(3.5, 2.5): " << add(3.5, 2.5) << endl;
    
    cout << "square(5): " << square(5) << endl;
    cout << "square(3.14): " << square(3.14) << endl;
    
    // Using HasSize concept
    string str = "Hello, Concepts!";
    printSize(str);
    
    // Error: int doesn't have size() method
    // int x = 42;
    // printSize(x);  // Would cause compilation error
    
    return 0;
}
```

## Advanced Template Techniques

### Exercise 9: Template Metaprogramming

Complete this template metaprogramming example:

```cpp
#include <iostream>
#include <type_traits>
using namespace std;

// Compile-time factorial calculation
template <size_t N>
struct Factorial {
    static constexpr size_t value = N * Factorial<N-1>::value;
};

// Base case specialization
template <>
struct Factorial<0> {
    static constexpr size_t value = 1;
};

// Compile-time Fibonacci calculation
template <size_t N>
struct Fibonacci {
    static constexpr size_t value = Fibonacci<N-1>::value + Fibonacci<N-2>::value;
};

// Base cases
template <>
struct Fibonacci<0> {
    static constexpr size_t value = 0;
};

template <>
struct Fibonacci<1> {
    static constexpr size_t value = 1;
};

// Type traits example - check if type is a pointer
template <typename T>
struct IsPointer {
    static constexpr bool value = false;
};

template <typename T>
struct IsPointer<T*> {
    static constexpr bool value = true;
};

// Enable/Disable based on conditions
template <typename T>
auto processValue(T value) -> typename enable_if<is_arithmetic<T>::value, T>::type {
    cout << "Processing arithmetic value: " << value << endl;
    return value * 2;
}

template <typename T>
auto processValue(T value) -> typename enable_if<!is_arithmetic<T>::value, T>::type {
    cout << "Processing non-arithmetic value" << endl;
    return value;
}

// Conditional type selection
template <bool condition, typename TrueType, typename FalseType>
struct Conditional {
    using type = TrueType;
};

template <typename TrueType, typename FalseType>
struct Conditional<false, TrueType, FalseType> {
    using type = FalseType;
};

int main() {
    cout << "=== Template Metaprogramming Demo ===" << endl;
    
    // Compile-time calculations
    cout << "Factorial of 5: " << Factorial<5>::value << endl;
    cout << "Factorial of 0: " << Factorial<0>::value << endl;
    cout << "Fibonacci of 10: " << Fibonacci<10>::value << endl;
    cout << "Fibonacci of 0: " << Fibonacci<0>::value << endl;
    cout << "Fibonacci of 1: " << Fibonacci<1>::value << endl;
    
    // Type trait checks
    cout << "Is int a pointer? " << IsPointer<int>::value << endl;
    cout << "Is int* a pointer? " << IsPointer<int*>::value << endl;
    cout << "Is string a pointer? " << IsPointer<string>::value << endl;
    
    // Conditional processing
    int intVal = 10;
    string strVal = "hello";
    
    auto processedInt = processValue(intVal);
    auto processedStr = processValue(strVal);
    
    cout << "Processed int: " << processedInt << endl;
    cout << "Processed string: " << processedStr << endl;
    
    // Conditional type selection
    using ConditionalInt = Conditional<true, int, double>::type;
    using ConditionalDouble = Conditional<false, int, double>::type;
    
    ConditionalInt condInt = 42;
    ConditionalDouble condDouble = 3.14;
    
    cout << "Conditional int value: " << condInt << endl;
    cout << "Conditional double value: " << condDouble << endl;
    
    return 0;
}
```

## Perfect Forwarding and Move Semantics

### Exercise 10: Perfect Forwarding

Complete this perfect forwarding example:

```cpp
#include <iostream>
#include <utility>
#include <string>
#include <vector>
using namespace std;

// Function that accepts universal references and forwards perfectly
template <typename T>
void wrapper(T&& arg) {
    cout << "Forwarding argument to processor..." << endl;
    // Perfectly forward the argument to another function
    process(forward<T>(arg));
}

// Overloaded process functions to demonstrate forwarding
void process(int& x) {
    cout << "Processing int&: " << x << " (lvalue reference)" << endl;
    x = 100;  // Can modify lvalue reference
}

void process(const int& x) {
    cout << "Processing const int&: " << x << " (const lvalue reference)" << endl;
    // Cannot modify const reference
}

void process(int&& x) {
    cout << "Processing int&&: " << x << " (rvalue reference)" << endl;
    // Can modify rvalue reference
}

void process(const string& s) {
    cout << "Processing const string&: " << s << endl;
}

void process(string&& s) {
    cout << "Processing string&&: " << s << endl;
    // Can modify rvalue string
    s += " (modified)";
    cout << "Modified string: " << s << endl;
}

// More complex example with multiple arguments
template <typename... Args>
void forwardMultiple(Args&&... args) {
    cout << "Forwarding multiple arguments..." << endl;
    // Forward all arguments to another function
    consume(forward<Args>(args)...);
}

void consume(int x, string s, double d) {
    cout << "consume(int, string, double): " << x << ", " << s << ", " << d << endl;
}

void consume(int& x, string& s, double& d) {
    cout << "consume(int&, string&, double&): " << x << ", " << s << ", " << d << endl;
    x = 999;
    s = "Modified string";
    d = 9.99;
}

int main() {
    cout << "=== Perfect Forwarding Demo ===" << endl;
    
    // Test with lvalue
    int lvalue = 42;
    cout << "Before wrapper(lvalue): " << lvalue << endl;
    wrapper(lvalue);  // Passes int& to wrapper, which forwards as int&
    cout << "After wrapper(lvalue): " << lvalue << endl;
    
    // Test with rvalue
    cout << "\nTesting with rvalue:" << endl;
    wrapper(24);  // Passes int&& to wrapper, which forwards as int&&
    
    // Test with string
    string str = "Hello";
    cout << "\nBefore wrapper(str): " << str << endl;
    wrapper(str);
    cout << "After wrapper(str): " << str << endl;
    
    // Test with string rvalue
    cout << "\nTesting with string rvalue:" << endl;
    wrapper(string("Rvalue string"));
    
    // Test with multiple arguments
    cout << "\n=== Multiple Arguments Forwarding ===" << endl;
    int x = 1;
    string s = "test";
    double d = 2.5;
    
    cout << "Before forwardMultiple: x=" << x << ", s=" << s << ", d=" << d << endl;
    forwardMultiple(x, s, d);
    cout << "After forwardMultiple: x=" << x << ", s=" << s << ", d=" << d << endl;
    
    return 0;
}
```

## Practical Example: Generic Container

### Exercise 11: Complete Generic Container

Create a comprehensive generic container using templates:

```cpp
#include <iostream>
#include <memory>
#include <stdexcept>
#include <initializer_list>
using namespace std;

template <typename T>
class GenericContainer {
private:
    unique_ptr<T[]> data;
    size_t currentSize;
    size_t currentCapacity;
    
    void resize(size_t newCapacity) {
        unique_ptr<T[]> newData = make_unique<T[]>(newCapacity);
        
        for (size_t i = 0; i < currentSize; ++i) {
            newData[i] = move(data[i]);  // Move elements to new array
        }
        
        data = move(newData);
        currentCapacity = newCapacity;
    }
    
public:
    // Default constructor
    explicit GenericContainer(size_t initialCapacity = 10) 
        : currentSize(0), currentCapacity(initialCapacity) {
        data = make_unique<T[]>(currentCapacity);
    }
    
    // Constructor with initial size
    explicit GenericContainer(size_t size, const T& value = T{}) 
        : currentSize(size), currentCapacity(size) {
        data = make_unique<T[]>(currentCapacity);
        for (size_t i = 0; i < size; ++i) {
            data[i] = value;
        }
    }
    
    // Initializer list constructor
    GenericContainer(initializer_list<T> init) 
        : currentSize(init.size()), currentCapacity(init.size()) {
        data = make_unique<T[]>(currentCapacity);
        size_t i = 0;
        for (const auto& item : init) {
            data[i++] = item;
        }
    }
    
    // Copy constructor
    GenericContainer(const GenericContainer& other) 
        : currentSize(other.currentSize), currentCapacity(other.currentCapacity) {
        data = make_unique<T[]>(currentCapacity);
        for (size_t i = 0; i < currentSize; ++i) {
            data[i] = other.data[i];
        }
    }
    
    // Move constructor
    GenericContainer(GenericContainer&& other) noexcept
        : data(move(other.data)), currentSize(other.currentSize), 
          currentCapacity(other.currentCapacity) {
        other.currentSize = 0;
        other.currentCapacity = 0;
    }
    
    // Copy assignment operator
    GenericContainer& operator=(const GenericContainer& other) {
        if (this != &other) {
            currentSize = other.currentSize;
            currentCapacity = other.currentCapacity;
            data = make_unique<T[]>(currentCapacity);
            for (size_t i = 0; i < currentSize; ++i) {
                data[i] = other.data[i];
            }
        }
        return *this;
    }
    
    // Move assignment operator
    GenericContainer& operator=(GenericContainer&& other) noexcept {
        if (this != &other) {
            data = move(other.data);
            currentSize = other.currentSize;
            currentCapacity = other.currentCapacity;
            
            other.currentSize = 0;
            other.currentCapacity = 0;
        }
        return *this;
    }
    
    // Destructor
    ~GenericContainer() = default;
    
    // Element access
    T& operator[](size_t index) {
        if (index >= currentSize) {
            throw out_of_range("Index out of bounds");
        }
        return data[index];
    }
    
    const T& operator[](size_t index) const {
        if (index >= currentSize) {
            throw out_of_range("Index out of bounds");
        }
        return data[index];
    }
    
    T& at(size_t index) {
        if (index >= currentSize) {
            throw out_of_range("Index out of bounds");
        }
        return data[index];
    }
    
    const T& at(size_t index) const {
        if (index >= currentSize) {
            throw out_of_range("Index out of bounds");
        }
        return data[index];
    }
    
    T& front() {
        if (currentSize == 0) {
            throw out_of_range("Container is empty");
        }
        return data[0];
    }
    
    const T& front() const {
        if (currentSize == 0) {
            throw out_of_range("Container is empty");
        }
        return data[0];
    }
    
    T& back() {
        if (currentSize == 0) {
            throw out_of_range("Container is empty");
        }
        return data[currentSize - 1];
    }
    
    const T& back() const {
        if (currentSize == 0) {
            throw out_of_range("Container is empty");
        }
        return data[currentSize - 1];
    }
    
    T* begin() { return data.get(); }
    const T* begin() const { return data.get(); }
    T* end() { return data.get() + currentSize; }
    const T* end() const { return data.get() + currentSize; }
    
    // Capacity
    size_t size() const { return currentSize; }
    size_t capacity() const { return currentCapacity; }
    bool empty() const { return currentSize == 0; }
    
    void reserve(size_t newCapacity) {
        if (newCapacity > currentCapacity) {
            resize(newCapacity);
        }
    }
    
    // Modifiers
    void push_back(const T& value) {
        if (currentSize >= currentCapacity) {
            resize(currentCapacity == 0 ? 1 : currentCapacity * 2);
        }
        data[currentSize++] = value;
    }
    
    void push_back(T&& value) {
        if (currentSize >= currentCapacity) {
            resize(currentCapacity == 0 ? 1 : currentCapacity * 2);
        }
        data[currentSize++] = move(value);
    }
    
    template <typename... Args>
    void emplace_back(Args&&... args) {
        if (currentSize >= currentCapacity) {
            resize(currentCapacity == 0 ? 1 : currentCapacity * 2);
        }
        data[currentSize++] = T(forward<Args>(args)...);
    }
    
    void pop_back() {
        if (currentSize > 0) {
            --currentSize;
        }
    }
    
    void clear() {
        currentSize = 0;
    }
    
    void shrink_to_fit() {
        if (currentSize < currentCapacity) {
            resize(currentSize);
        }
    }
    
    // Utility function to print container
    void print() const {
        cout << "[";
        for (size_t i = 0; i < currentSize; ++i) {
            cout << data[i];
            if (i < currentSize - 1) cout << ", ";
        }
        cout << "]" << endl;
    }
};

int main() {
    cout << "=== Generic Container Demo ===" << endl;
    
    // Test with integers
    GenericContainer<int> intContainer{1, 2, 3, 4, 5};
    cout << "Initial container: ";
    intContainer.print();
    
    intContainer.push_back(6);
    intContainer.push_back(7);
    cout << "After adding 6, 7: ";
    intContainer.print();
    
    cout << "Element at index 2: " << intContainer[2] << endl;
    cout << "Front element: " << intContainer.front() << endl;
    cout << "Back element: " << intContainer.back() << endl;
    
    // Test with strings
    GenericContainer<string> stringContainer(3, "Hello");
    cout << "\nString container: ";
    stringContainer.print();
    
    stringContainer.push_back("World");
    stringContainer.push_back("C++");
    cout << "After adding 'World', 'C++': ";
    stringContainer.print();
    
    // Test move semantics
    string temp = "Temporary";
    stringContainer.push_back(move(temp));
    cout << "After moving 'Temporary': ";
    stringContainer.print();
    cout << "Moved-from string: '" << temp << "'" << endl;  // Should be empty or in valid state
    
    // Test emplace_back
    stringContainer.emplace_back("Emplaced");
    cout << "After emplacing 'Emplaced': ";
    stringContainer.print();
    
    // Test iteration
    cout << "\nIterating through string container: ";
    for (const auto& str : stringContainer) {
        cout << str << " ";
    }
    cout << endl;
    
    // Test copy and move operations
    auto copyContainer = stringContainer;
    cout << "Copy container: ";
    copyContainer.print();
    
    auto moveContainer = move(copyContainer);
    cout << "Move container: ";
    moveContainer.print();
    cout << "Moved-from container size: " << copyContainer.size() << endl;  // Should be 0
    
    return 0;
}
```

## Best Practices for Templates

### Exercise 12: Template Best Practices

Demonstrate best practices in template design:

```cpp
#include <iostream>
#include <type_traits>
#include <vector>
#include <memory>
using namespace std;

// 1. Use meaningful template parameter names
template <typename ValueType, typename AllocatorType = allocator<ValueType>>
class BestPracticeContainer {
    // Implementation using meaningful names
};

// 2. Use SFINAE for constraints (pre-C++20)
template <typename T>
typename enable_if<is_arithmetic<T>::value, T>::type
safeDivide(T a, T b) {
    if (b == T{}) {  // Use T{} instead of hardcoded 0
        throw invalid_argument("Division by zero");
    }
    return a / b;
}

// 3. Use concepts when available (C++20)
#ifdef __cpp_concepts
template <typename T>
concept Arithmetic = is_arithmetic_v<T>;

template <Arithmetic T>
T safeDivideConcepts(T a, T b) {
    if (b == T{}) {
        throw invalid_argument("Division by zero");
    }
    return a / b;
}
#endif

// 4. Perfect forwarding example
template <typename Func, typename... Args>
auto callWithTiming(Func&& func, Args&&... args) 
    -> decltype(func(forward<Args>(args)...)) {
    // Timing code would go here
    return func(forward<Args>(args)...);
}

// 5. Template alias for complex types
template <typename T>
using VecOfUniquePtr = vector<unique_ptr<T>>;

// 6. Conditional compilation based on type properties
template <typename T>
void process(T&& value) {
    if constexpr (is_integral_v<decay_t<T>>) {
        cout << "Processing integral: " << value << endl;
    } else if constexpr (is_floating_point_v<decay_t<T>>) {
        cout << "Processing floating point: " << value << endl;
    } else {
        cout << "Processing other type" << endl;
    }
}

// 7. CRTP (Curiously Recurring Template Pattern)
template <typename Derived>
class EqualityComparable {
public:
    bool operator!=(const Derived& other) const {
        return !static_cast<const Derived&>(*this).operator==(other);
    }
};

class MyInt : public EqualityComparable<MyInt> {
private:
    int value;
    
public:
    MyInt(int v) : value(v) {}
    
    bool operator==(const MyInt& other) const {
        return value == other.value;
    }
    
    int getValue() const { return value; }
};

int main() {
    cout << "=== Template Best Practices Demo ===" << endl;
    
    // Test safe division
    cout << "Safe division: " << safeDivide(10, 3) << endl;
    
    // Test conditional processing
    process(42);
    process(3.14);
    process(string("hello"));
    
    // Test CRTP
    MyInt a(5), b(10), c(5);
    cout << "a == b: " << (a == b) << endl;
    cout << "a != b: " << (a != b) << endl;
    cout << "a == c: " << (a == c) << endl;
    cout << "a != c: " << (a != c) << endl;
    
    // Test template alias
    VecOfUniquePtr<int> intVec;
    intVec.push_back(make_unique<int>(42));
    intVec.push_back(make_unique<int>(100));
    
    cout << "Vector of unique_ptrs: ";
    for (const auto& ptr : intVec) {
        cout << *ptr << " ";
    }
    cout << endl;
    
    return 0;
}
```

## Summary

In this chapter, you learned:
- Function templates and how to create generic functions
- Class templates for creating generic classes
- Template specialization (full and partial)
- Variadic templates for handling variable arguments
- Concepts for constraining templates (C++20)
- Template metaprogramming techniques
- Perfect forwarding and move semantics with templates
- Best practices for template design

## Key Takeaways

- Templates enable generic programming and code reuse
- Function templates allow single functions to work with multiple types
- Class templates create generic classes
- Template specialization provides type-specific implementations
- Variadic templates handle variable numbers of arguments
- Concepts provide better error messages and constraints
- Perfect forwarding preserves argument properties
- SFINAE enables conditional compilation based on type properties

## Common Mistakes to Avoid

1. Forgetting to implement move semantics in template classes
2. Not properly constraining templates (before C++20 concepts)
3. Incorrect use of enable_if without proper SFINAE patterns
4. Forgetting to handle both lvalues and rvalues in forwarding functions
5. Not considering the complexity of template error messages
6. Overusing templates when simpler solutions exist
7. Not testing templates with various types
8. Forgetting to implement proper copy/move semantics in template classes

## Next Steps

Now that you understand templates and generic programming, you're ready to learn about the Standard Template Library (STL) containers and algorithms in Chapter 10.