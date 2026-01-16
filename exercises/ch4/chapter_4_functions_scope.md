# Chapter 4: Functions and Scope

## Overview

This chapter explores functions in C++, which are essential for organizing code into reusable, manageable pieces. You'll learn how to define and call functions, understand parameter passing mechanisms, and master the concept of scope in C++.

## Learning Objectives

By the end of this chapter, you will:
- Understand how to define and call functions
- Learn about different parameter passing mechanisms
- Master function overloading
- Understand recursion and its applications
- Learn about scope and storage duration
- Explore default arguments and function templates
- Understand inline functions and lambda expressions

## Function Basics

A function is a block of code that performs a specific task and can be called from other parts of the program.

### Exercise 1: Basic Function Definition

The following code has errors in function definition and usage. Find and fix them:

```cpp
#include <iostream>
using namespace std;

// Function declaration
int add(int a, b);  // Error: missing type for parameter b

int main() {
    int x = 5, y = 3;
    int result = add(x, y);
    cout << x << " + " << y << " = " << result << endl;
    
    return 0;
}

// Function definition
int add(int a, int b) {  // This should match the declaration
    return a + b;
}

// Error: function defined after use without declaration
int multiply(int x, y) {  // Error: missing type for parameter y
    return x * y;
}
```

### Exercise 2: Function Prototypes and Definitions

Fix the errors in this function organization:

```cpp
#include <iostream>
using namespace std;

// Function prototypes
void printMessage();
int calculateSquare(int num);
double divide(double a, b);  // Error: missing type for parameter b

int main() {
    printMessage();
    
    int number = 4;
    int squared = calculateSquare(number);
    cout << number << " squared is " << squared << endl;
    
    double result = divide(10.0, 3.0);
    cout << "10.0 / 3.0 = " << result << endl;
    
    return 0;
}

// Function definitions
void printMessage() {
    cout << "Hello from function!" << endl;
}

int calculateSquare(int num) {
    return num * num;
}

double divide(double a, double b) {  // Fixed: added missing type
    if (b != 0) {
        return a / b;
    } else {
        cout << "Error: Division by zero!" << endl;
        return 0.0;  // Error: returning 0 for division by zero
    }
}
```

## Parameter Passing Mechanisms

C++ supports three main ways to pass parameters:
1. Pass by value
2. Pass by reference
3. Pass by pointer

### Exercise 3: Parameter Passing

Complete this example demonstrating different parameter passing methods:

```cpp
#include <iostream>
using namespace std;

// Pass by value - receives a copy
void passByValue(int x) {
    x = 100;  // Only affects the local copy
    cout << "Inside passByValue: x = " << x << endl;
}

// Pass by reference - receives the original variable
void passByReference(int& x) {
    x = 200;  // Modifies the original variable
    cout << "Inside passByReference: x = " << x << endl;
}

// Pass by pointer - receives address of the variable
void passByPointer(int* x) {
    if (x != nullptr) {
        *x = 300;  // Modifies the value at the address
        cout << "Inside passByPointer: *x = " << *x << endl;
    }
}

int main() {
    int value = 10;
    
    cout << "Original value: " << value << endl;
    
    passByValue(value);
    cout << "After passByValue: " << value << endl;  // Still 10
    
    passByReference(value);
    cout << "After passByReference: " << value << endl;  // Now 200
    
    passByPointer(&value);
    cout << "After passByPointer: " << value << endl;  // Now 300
    
    // Error: trying to pass a literal to a reference parameter
    // passByReference(42);  // This would cause a compilation error
    
    // Correct way: use a variable
    int temp = 42;
    passByReference(temp);
    
    // Error: null pointer dereference risk
    int* ptr = nullptr;
    passByPointer(ptr);  // Safe because function checks for nullptr
    
    return 0;
}
```

### Exercise 4: Const Parameters

Fix the const-correctness issues in this code:

```cpp
#include <iostream>
#include <string>
using namespace std;

// Function that should not modify the string
void printString(const string& str) {
    cout << "String: " << str << endl;
    // str += "modified";  // Error: cannot modify const reference
}

// Function that modifies the string
void modifyString(string& str) {
    str += " - Modified";
}

// Function that takes a pointer to const
void printConstPointer(const string* str) {
    if (str != nullptr) {
        cout << "Pointer string: " << *str << endl;
        // *str += "test";  // Error: cannot modify through const pointer
    }
}

int main() {
    string text = "Hello";
    
    printString(text);  // OK
    cout << "Before modification: " << text << endl;
    
    modifyString(text);  // OK
    cout << "After modification: " << text << endl;
    
    printConstPointer(&text);  // OK
    
    // Error: trying to pass const to non-const reference
    const string constText = "Constant";
    // modifyString(constText);  // Would cause compilation error
    
    // Correct way:
    printString(constText);  // Accepts both const and non-const
    
    return 0;
}
```

## Function Overloading

C++ allows multiple functions with the same name but different parameter lists.

### Exercise 5: Function Overloading

Complete this overloaded function example with errors:

```cpp
#include <iostream>
using namespace std;

// Overloaded functions
int add(int a, int b) {
    cout << "Adding integers: ";
    return a + b;
}

double add(double a, double b) {
    cout << "Adding doubles: ";
    return a + b;
}

string add(const string& a, const string& b) {
    cout << "Concatenating strings: ";
    return a + b;
}

// Error: cannot overload by return type only
// int add(int x, int y);        // Declaration
// double add(int x, int y);     // Error: only return type differs

int main() {
    cout << add(5, 3) << endl;           // Calls int version
    cout << add(2.5, 3.7) << endl;      // Calls double version
    cout << add(string("Hello"), string(" World")) << endl;  // Calls string version
    
    // Ambiguity error: which version to call?
    cout << add(5.0, 3) << endl;  // double and int - compiler may choose double version
    
    // Error: calling with wrong number of arguments
    // cout << add(1, 2, 3) << endl;  // No matching function
    
    return 0;
}
```

## Recursion

Recursion occurs when a function calls itself to solve a problem.

### Exercise 6: Recursive Functions

Complete these recursive functions with errors:

```cpp
#include <iostream>
using namespace std;

// Factorial using recursion
int factorial(int n) {
    // Error: missing base case
    if (n <= 1) {  // Base case
        return 1;
    }
    return n * factorial(n - 1);  // Recursive case
}

// Fibonacci using recursion
int fibonacci(int n) {
    // Error: inefficient recursive implementation
    if (n <= 1) {  // Base cases
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);  // Recursive case
}

// Error: infinite recursion
void infiniteRecursion(int n) {
    cout << n << " ";
    infiniteRecursion(n + 1);  // No base case - infinite recursion!
}

int main() {
    int num = 5;
    
    cout << "Factorial of " << num << " is " << factorial(num) << endl;
    
    cout << "First 10 Fibonacci numbers: ";
    for (int i = 0; i < 10; i++) {
        cout << fibonacci(i) << " ";
    }
    cout << endl;
    
    // Error: stack overflow risk
    // cout << factorial(-5) << endl;  // Infinite recursion for negative numbers
    
    // Safer factorial function
    int safeFactorial(int n) {
        if (n < 0) return -1;  // Error indicator
        if (n <= 1) return 1;
        return n * safeFactorial(n - 1);
    }
    
    return 0;
}
```

## Scope and Storage Duration

Understanding scope is crucial for writing correct C++ programs.

### Exercise 7: Variable Scope

Analyze and fix the scope issues in this code:

```cpp
#include <iostream>
using namespace std;

int globalVar = 10;  // Global scope

void function1() {
    int localVar = 20;  // Local scope (function scope)
    cout << "In function1 - globalVar: " << globalVar << endl;
    cout << "In function1 - localVar: " << localVar << endl;
}

void function2() {
    // This localVar is different from the one in function1
    int localVar = 30;
    cout << "In function2 - globalVar: " << globalVar << endl;
    cout << "In function2 - localVar: " << localVar << endl;
    
    // Error: accessing variable from different scope
    // cout << "Trying to access function1's localVar: " << localVar << endl;  // Ambiguous
}

int main() {
    cout << "In main - globalVar: " << globalVar << endl;
    // cout << "In main - localVar from function1: " << localVar << endl;  // Error: undefined
    
    function1();
    function2();
    
    // Block scope
    {
        int blockVar = 40;
        cout << "In block - blockVar: " << blockVar << endl;
        cout << "In block - globalVar: " << globalVar << endl;
    }
    // cout << "Outside block - blockVar: " << blockVar << endl;  // Error: undefined
    
    // Shadowing example
    int shadowed = 50;
    cout << "Before block - shadowed: " << shadowed << endl;
    
    {
        int shadowed = 60;  // Shadows outer variable
        cout << "Inside block - shadowed: " << shadowed << endl;
    }
    
    cout << "After block - shadowed: " << shadowed << endl;  // Back to original value
    
    return 0;
}
```

### Exercise 8: Storage Duration

Explore different storage durations in this example:

```cpp
#include <iostream>
using namespace std;

// Global variable - static storage duration
int globalCounter = 0;

void incrementGlobal() {
    globalCounter++;
    cout << "Global counter: " << globalCounter << endl;
}

void staticVariableFunction() {
    // Static local variable - retains value between calls
    static int staticCounter = 0;  // Initialized only once
    int regularCounter = 0;        // Initialized every time
    
    staticCounter++;
    regularCounter++;
    
    cout << "Static counter: " << staticCounter << ", Regular counter: " << regularCounter << endl;
}

int main() {
    cout << "Initial global counter: " << globalCounter << endl;
    
    incrementGlobal();
    incrementGlobal();
    incrementGlobal();
    
    cout << "Global counter after increments: " << globalCounter << endl;
    
    cout << "Testing static variables:" << endl;
    staticVariableFunction();
    staticVariableFunction();
    staticVariableFunction();
    
    // Error: uninitialized local variable
    int uninitialized;
    // cout << "Uninitialized: " << uninitialized << endl;  // Undefined behavior
    
    // Correct: initialized local variable
    int initialized = 0;
    cout << "Initialized: " << initialized << endl;
    
    return 0;
}
```

## Default Arguments

Functions can have default values for parameters.

### Exercise 9: Default Arguments

Complete this example with default arguments and errors:

```cpp
#include <iostream>
#include <string>
using namespace std;

// Function with default arguments
void printInfo(const string& name, int age = 18, const string& city = "Unknown") {
    cout << "Name: " << name << ", Age: " << age << ", City: " << city << endl;
}

// Error: default arguments must be at the end
// void badFunction(int x = 5, int y, int z = 10);  // Error: y has no default but comes after defaults

int multiply(int a, int b = 2, int c = 3) {  // OK: defaults at the end
    return a * b * c;
}

int main() {
    // Using default arguments
    printInfo("Alice");                    // Uses defaults for age and city
    printInfo("Bob", 25);                 // Uses default for city only
    printInfo("Charlie", 30, "New York"); // No defaults used
    
    // Calling with default arguments
    cout << "Multiply(5): " << multiply(5) << endl;           // 5 * 2 * 3 = 30
    cout << "Multiply(5, 4): " << multiply(5, 4) << endl;    // 5 * 4 * 3 = 60
    cout << "Multiply(5, 4, 2): " << multiply(5, 4, 2) << endl; // 5 * 4 * 2 = 40
    
    // Error: cannot specify middle argument without specifying others
    // cout << multiply(5, , 4) << endl;  // Syntax error
    
    return 0;
}
```

## Function Templates

Function templates allow writing generic functions.

### Exercise 10: Function Templates

Complete this template example with errors:

```cpp
#include <iostream>
using namespace std;

// Function template
template <typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

// Template with multiple types
template <typename T, typename U>
auto multiply(T a, U b) -> decltype(a * b) {  // Trailing return type
    return a * b;
}

// Error: template with constraints (C++20 style)
#ifdef HAS_CONCEPTS
#include <concepts>

template <std::integral T>
T square(T x) {
    return x * x;
}
#endif

int main() {
    // Using function template
    cout << "Max of 5 and 10: " << max(5, 10) << endl;
    cout << "Max of 3.14 and 2.71: " << max(3.14, 2.71) << endl;
    cout << "Max of 'a' and 'z': " << max('a', 'z') << endl;
    
    // Using template with different types
    cout << "Multiply int and double: " << multiply(5, 2.5) << endl;
    
    // Error: trying to use max with incompatible types
    // cout << max(5, 3.14) << endl;  // Error: different types, compiler can't deduce T
    
    // Correct way: explicit template instantiation
    cout << max<int>(5, static_cast<int>(3.14)) << endl;
    cout << max<double>(static_cast<double>(5), 3.14) << endl;
    
    return 0;
}
```

## Inline Functions

Inline functions suggest to the compiler to expand the function in-place.

### Exercise 11: Inline Functions

Work with inline functions in this example:

```cpp
#include <iostream>
using namespace std;

// Inline function
inline int square(int x) {
    return x * x;
}

// Regular function
int cube(int x) {
    return x * x * x;
}

// Error: recursive inline function (generally not recommended)
inline int factorial_inline(int n) {
    if (n <= 1) return 1;
    return n * factorial_inline(n - 1);  // Recursive call in inline function
}

int main() {
    int num = 5;
    
    cout << "Square of " << num << " is " << square(num) << endl;
    cout << "Cube of " << num << " is " << cube(num) << endl;
    
    // The inline keyword is just a suggestion to the compiler
    // The compiler may ignore it for complex functions
    
    // Error: function too complex to inline effectively
    inline int complexFunction(int a, int b) {  // Error: inline specifier not allowed here
        // This would cause a compilation error
        return a + b;
    }
    
    return 0;
}
```

## Lambda Expressions (C++11)

Lambda expressions allow defining anonymous functions.

### Exercise 12: Lambda Expressions

Complete this lambda example with errors:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    // Simple lambda
    auto greet = []() { cout << "Hello from lambda!" << endl; };
    greet();
    
    // Lambda with parameters
    auto add = [](int a, int b) { return a + b; };
    cout << "5 + 3 = " << add(5, 3) << endl;
    
    // Lambda with capture
    int multiplier = 10;
    auto multiply = [multiplier](int x) { return x * multiplier; };  // Capture by value
    cout << "5 * 10 = " << multiply(5) << endl;
    
    // Capture by reference
    int counter = 0;
    auto increment = [&counter]() { counter++; };
    increment();
    increment();
    cout << "Counter after increments: " << counter << endl;
    
    // Capture all by value
    auto all_by_value = [=]() { return multiplier * 5; };
    cout << "All by value result: " << all_by_value() << endl;
    
    // Capture all by reference
    auto all_by_reference = [&]() { return ++counter; };  // Modifies counter
    cout << "All by reference result: " << all_by_reference() << endl;
    
    // Error: using lambda with STL algorithms
    vector<int> numbers = {5, 2, 8, 1, 9};
    
    // Sort in descending order using lambda
    sort(numbers.begin(), numbers.end(), [](int a, int b) { return a > b; });
    
    cout << "Sorted in descending order: ";
    for (int n : numbers) {
        cout << n << " ";
    }
    cout << endl;
    
    // Find element using lambda
    auto it = find_if(numbers.begin(), numbers.end(), [](int x) { return x > 5; });
    if (it != numbers.end()) {
        cout << "First element > 5: " << *it << endl;
    }
    
    // Error: mutable lambda (when you need to modify captured values by value)
    int value = 5;
    auto mutable_lambda = [value]() mutable { 
        value += 10;  // OK: modifies the captured copy
        return value;
    };
    
    cout << "Original value: " << value << endl;
    cout << "Modified in lambda: " << mutable_lambda() << endl;
    cout << "Original after lambda: " << value << endl;  // Still 5
    
    return 0;
}
```

## Function Pointers

Functions can be stored in pointers and passed as parameters.

### Exercise 13: Function Pointers

Complete this function pointer example with errors:

```cpp
#include <iostream>
using namespace std;

// Regular functions
int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }
int multiply(int a, int b) { return a * b; }

// Function that takes another function as parameter
int calculate(int x, int y, int (*operation)(int, int)) {
    return operation(x, y);
}

int main() {
    // Function pointer
    int (*funcPtr)(int, int);
    
    funcPtr = add;  // Assign function address
    cout << "Using function pointer for addition: " << funcPtr(5, 3) << endl;
    
    funcPtr = subtract;  // Reassign to different function
    cout << "Using function pointer for subtraction: " << funcPtr(5, 3) << endl;
    
    // Using function pointer with the calculate function
    cout << "Calculate using add: " << calculate(10, 5, add) << endl;
    cout << "Calculate using subtract: " << calculate(10, 5, subtract) << endl;
    cout << "Calculate using multiply: " << calculate(10, 5, multiply) << endl;
    
    // Array of function pointers
    int (*operations[])(int, int) = {add, subtract, multiply};
    int numOperations = sizeof(operations) / sizeof(operations[0]);
    
    cout << "\nUsing array of function pointers:" << endl;
    for (int i = 0; i < numOperations; i++) {
        cout << "Operation " << i << ": " << operations[i](10, 5) << endl;
    }
    
    // Error: assigning wrong function type
    // double (*wrongPtr)(double, double) = add;  // Error: signature mismatch
    
    // Correct way: using typedef or using for cleaner syntax
    using Operation = int(*)(int, int);
    Operation op = multiply;
    cout << "Using typedef'd function pointer: " << op(4, 3) << endl;
    
    return 0;
}
```

## Hands-On Project: Calculator with Function Pointers

### Exercise 14: Complete Calculator

Create a calculator that uses function pointers and has intentional errors:

```cpp
#include <iostream>
#include <map>
#include <functional>
using namespace std;

// Calculator operations
double add(double a, double b) { return a + b; }
double subtract(double a, double b) { return a - b; }
double multiply(double a, double b) { return a * b; }
double divide(double a, double b) { 
    if (b != 0) return a / b;
    cout << "Error: Division by zero!" << endl;
    return 0.0;
}

int main() {
    map<char, function<double(double, double)>> operations = {
        {'+', add},
        {'-', subtract},
        {'*', multiply},
        {'/', divide}
    };
    
    double num1, num2, result;
    char op;
    
    cout << "Simple Calculator" << endl;
    cout << "Supported operations: +, -, *, /" << endl;
    cout << "Enter calculation (e.g., 5 + 3): ";
    cin >> num1 >> op >> num2;
    
    // Check if operation is supported
    if (operations.find(op) != operations.end()) {
        result = operations[op](num1, num2);
        cout << num1 << " " << op << " " << num2 << " = " << result << endl;
    } else {
        cout << "Unsupported operation: " << op << endl;
    }
    
    // Error: potential division by zero not handled in map
    // If user enters "5 / 0", the divide function handles it
    
    // Alternative implementation using raw function pointers
    double (*opFunc)(double, double) = nullptr;
    
    switch(op) {
        case '+': opFunc = add; break;
        case '-': opFunc = subtract; break;
        case '*': opFunc = multiply; break;
        case '/': opFunc = divide; break;
        default:
            cout << "Invalid operator!" << endl;
            return 1;
    }
    
    if (opFunc != nullptr) {
        result = opFunc(num1, num2);
        cout << "[Function pointer result] " << num1 << " " << op << " " << num2 << " = " << result << endl;
    }
    
    return 0;
}
```

## Best Practices

1. Use function overloading judiciously to enhance readability
2. Prefer pass-by-reference for large objects to avoid copying
3. Use const parameters when the function doesn't modify them
4. Keep functions focused on a single responsibility
5. Use meaningful function and parameter names
6. Document functions with comments when necessary
7. Consider using function templates for generic operations
8. Use lambdas for simple, local operations

## Summary

In this chapter, you learned:
- How to define and call functions with various parameter passing methods
- Function overloading and how the compiler resolves calls
- Recursion and its appropriate use cases
- Scope rules and storage duration in C++
- Default arguments and their limitations
- Function templates for generic programming
- Inline functions and lambda expressions
- Function pointers for dynamic behavior

## Key Takeaways

- Functions help organize code and promote reusability
- Parameter passing mechanism affects performance and behavior
- Scope determines where variables can be accessed
- Function overloading improves code readability
- Templates enable generic programming
- Lambdas provide convenient anonymous functions
- Function pointers enable dynamic behavior

## Common Mistakes to Avoid

1. Forgetting function declarations when defining functions after main()
2. Confusing pass-by-value with pass-by-reference
3. Creating infinite recursion without proper base cases
4. Accessing variables outside their scope
5. Modifying const parameters
6. Not handling edge cases in recursive functions
7. Using default arguments incorrectly
8. Ignoring return values when they contain important information

## Next Steps

Now that you understand functions and scope, you're ready to learn about arrays and strings in Chapter 5.