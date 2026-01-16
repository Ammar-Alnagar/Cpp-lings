# Chapter 12: Exception Handling

## Overview

This chapter covers exception handling in C++, which is a powerful mechanism for dealing with errors and exceptional conditions in C++ programs. You'll learn about try-catch blocks, exception specifications, standard exceptions, and best practices for robust error handling.

## Learning Objectives

By the end of this chapter, you will:
- Understand the basics of exception handling in C++
- Learn to use try, catch, and throw statements
- Master exception hierarchies and standard exceptions
- Understand exception specifications and noexcept
- Learn about stack unwinding and exception safety
- Understand RAII (Resource Acquisition Is Initialization) in exception contexts
- Learn best practices for exception handling
- Understand how to create custom exception classes
- Learn about exception handling in constructors and destructors

## Basic Exception Handling

The fundamental components of C++ exception handling are try, catch, and throw.

### Exercise 1: Basic Exception Handling

The following code has errors. Find and fix them:

```cpp
#include <iostream>
#include <string>
#include <stdexcept>
using namespace std;

int divide(int a, int b) {
    if (b == 0) {
        throw runtime_error("Division by zero!");  // Throw exception
    }
    return a / b;
}

int main() {
    cout << "=== Basic Exception Handling Demo ===" << endl;
    
    int x = 10, y = 0;
    
    // Error: not handling the exception that divide() might throw
    // int result = divide(x, y);  // This would terminate the program
    
    // Correct way: use try-catch
    try {
        int result = divide(x, y);
        cout << "Result: " << result << endl;
    } catch (runtime_error& e) {  // Catch by reference to avoid slicing
        cout << "Caught exception: " << e.what() << endl;
    }
    
    // Test with valid input
    try {
        int result = divide(10, 2);
        cout << "Valid division result: " << result << endl;
    } catch (runtime_error& e) {
        cout << "Unexpected error: " << e.what() << endl;
    }
    
    // Multiple catch blocks
    try {
        string str = "Hello";
        char ch = str.at(100);  // This will throw out_of_range
    } catch (const out_of_range& e) {
        cout << "Out of range exception: " << e.what() << endl;
    } catch (const runtime_error& e) {
        cout << "Runtime error: " << e.what() << endl;
    } catch (const exception& e) {  // General catch for all standard exceptions
        cout << "General exception: " << e.what() << endl;
    }
    
    return 0;
}
```

### Exercise 2: Exception Hierarchies

Complete this example showing exception hierarchies:

```cpp
#include <iostream>
#include <exception>
#include <stdexcept>
#include <string>
using namespace std;

// Custom exception class
class CustomException : public exception {
private:
    string message;
    
public:
    CustomException(const string& msg) : message(msg) {}
    
    const char* what() const noexcept override {
        return message.c_str();
    }
};

// More specific custom exception
class FileException : public CustomException {
public:
    FileException(const string& msg) : CustomException("File Error: " + msg) {}
};

int processData(int value) {
    if (value < 0) {
        throw invalid_argument("Negative value not allowed");
    }
    if (value == 0) {
        throw logic_error("Zero value is invalid");
    }
    if (value > 1000) {
        throw range_error("Value exceeds maximum allowed range");
    }
    return value * 2;
}

int main() {
    cout << "=== Exception Hierarchies Demo ===" << endl;
    
    // Test different exception types
    vector<int> testValues = {-5, 0, 1500, 50};
    
    for (int val : testValues) {
        try {
            cout << "Processing value: " << val << endl;
            int result = processData(val);
            cout << "Result: " << result << endl;
        } catch (const invalid_argument& e) {
            cout << "Invalid argument: " << e.what() << endl;
        } catch (const logic_error& e) {
            cout << "Logic error: " << e.what() << endl;
        } catch (const range_error& e) {
            cout << "Range error: " << e.what() << endl;
        } catch (const exception& e) {
            cout << "General standard exception: " << e.what() << endl;
        } catch (...) {  // Catch-all handler
            cout << "Unknown exception caught" << endl;
        }
        cout << "---" << endl;
    }
    
    // Demonstrate custom exceptions
    try {
        throw FileException("Could not open file");
    } catch (const FileException& e) {
        cout << "Caught FileException: " << e.what() << endl;
    } catch (const CustomException& e) {
        cout << "Caught CustomException: " << e.what() << endl;
    } catch (const exception& e) {
        cout << "Caught standard exception: " << e.what() << endl;
    }
    
    // Exception order matters - more specific first
    try {
        throw runtime_error("Test error");
    } catch (const exception& e) {        // This would catch everything
        cout << "General catch: " << e.what() << endl;
    } catch (const runtime_error& e) {    // This would never be reached
        cout << "Runtime error catch: " << e.what() << endl;
    }
    
    return 0;
}
```

## Standard Exception Hierarchy

C++ provides a hierarchy of standard exception classes.

### Exercise 3: Standard Exception Classes

Complete this example with standard exceptions:

```cpp
#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <string>
using namespace std;

void demonstrateStandardExceptions() {
    cout << "=== Standard Exception Classes Demo ===" << endl;
    
    // bad_alloc - thrown by new when allocation fails
    try {
        // Attempt to allocate a huge amount of memory
        // unique_ptr<int[]> hugeArray(new int[1000000000000ULL]);  // May throw bad_alloc
        // For safety, we'll simulate this
        throw bad_alloc();
    } catch (const bad_alloc& e) {
        cout << "bad_alloc caught: " << e.what() << endl;
    }
    
    // out_of_range - thrown by container access methods
    try {
        vector<int> vec = {1, 2, 3};
        cout << vec.at(10) << endl;  // Throws out_of_range
    } catch (const out_of_range& e) {
        cout << "out_of_range caught: " << e.what() << endl;
    }
    
    // invalid_argument - thrown when invalid argument is passed
    try {
        string str = "not_a_number";
        int num = stoi(str);  // Throws invalid_argument
    } catch (const invalid_argument& e) {
        cout << "invalid_argument caught: " << e.what() << endl;
    } catch (const out_of_range& e) {
        cout << "out_of_range caught: " << e.what() << endl;
    }
    
    // length_error - thrown when trying to exceed max size
    try {
        string str;
        str.replace(0, 1, 1000000000, 'a');  // May throw length_error
    } catch (const length_error& e) {
        cout << "length_error caught: " << e.what() << endl;
    } catch (const bad_alloc& e) {
        cout << "bad_alloc caught instead: " << e.what() << endl;
    }
    
    // domain_error - thrown when argument is outside domain
    try {
        // In practice, this would be thrown by mathematical functions
        throw domain_error("Argument outside function domain");
    } catch (const domain_error& e) {
        cout << "domain_error caught: " << e.what() << endl;
    }
    
    // overflow_error - thrown on arithmetic overflow
    try {
        throw overflow_error("Arithmetic overflow occurred");
    } catch (const overflow_error& e) {
        cout << "overflow_error caught: " << e.what() << endl;
    }
    
    // underflow_error - thrown on arithmetic underflow
    try {
        throw underflow_error("Arithmetic underflow occurred");
    } catch (const underflow_error& e) {
        cout << "underflow_error caught: " << e.what() << endl;
    }
}

int main() {
    demonstrateStandardExceptions();
    return 0;
}
```

## Exception Specifications and noexcept

C++11 introduced exception specifications to indicate whether functions can throw exceptions.

### Exercise 4: Exception Specifications

Complete this example with exception specifications:

```cpp
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace std;

// Function that may throw
int riskyFunction(int x) {
    if (x < 0) {
        throw invalid_argument("Negative input not allowed");
    }
    return x * x;
}

// Function that promises not to throw (C++11)
int safeFunction(int x) noexcept {
    return x + 10;
}

// Function that may throw but uses exception specification (deprecated in C++17)
int oldStyleFunction(int x) /* throw(int, bad_exception) */ {
    if (x < 0) {
        throw invalid_argument("Negative input");
    }
    return x;
}

// Function with conditional noexcept
template<typename T>
void conditionalNoexcept(T& container, size_t index) noexcept(noexcept(container.at(index))) {
    // This function is noexcept if container.at(index) is noexcept
    // In practice, this would be more complex
}

// Function that throws on purpose to demonstrate noexcept violation
void violateNoexcept() noexcept {
    cout << "About to violate noexcept guarantee..." << endl;
    throw runtime_error("This violates noexcept!");  // This will call terminate!
}

int main() {
    cout << "=== Exception Specifications Demo ===" << endl;
    
    // Test functions that may throw
    try {
        int result = riskyFunction(-5);
        cout << "Result: " << result << endl;
    } catch (const invalid_argument& e) {
        cout << "Caught: " << e.what() << endl;
    }
    
    // Test function marked noexcept
    int safeResult = safeFunction(5);
    cout << "Safe function result: " << safeResult << endl;
    
    // Test noexcept operator
    cout << "riskyFunction is noexcept: " << noexcept(riskyFunction(5)) << endl;
    cout << "safeFunction is noexcept: " << noexcept(safeFunction(5)) << endl;
    
    // Be careful with noexcept violations!
    // Uncommenting the next lines will cause the program to terminate:
    /*
    try {
        violateNoexcept();  // This will call terminate()!
    } catch (const runtime_error& e) {
        cout << "This will never be printed: " << e.what() << endl;
    }
    */
    
    // Demonstrating noexcept with templates
    vector<int> vec = {1, 2, 3};
    cout << "Vector access is noexcept: " << noexcept(vec[0]) << endl;
    cout << "Vector at() is noexcept: " << noexcept(vec.at(0)) << endl;
    
    // Practical example: swap operations
    vector<int> vec1 = {1, 2, 3};
    vector<int> vec2 = {4, 5, 6};
    
    cout << "Before swap - vec1: ";
    for (int x : vec1) cout << x << " ";
    cout << endl;
    
    cout << "Before swap - vec2: ";
    for (int x : vec2) cout << x << " ";
    cout << endl;
    
    // swap is typically noexcept for most standard containers
    swap(vec1, vec2);
    
    cout << "After swap - vec1: ";
    for (int x : vec1) cout << x << " ";
    cout << endl;
    
    cout << "After swap - vec2: ";
    for (int x : vec2) cout << x << " ";
    cout << endl;
    
    cout << "swap is noexcept: " << noexcept(swap(vec1, vec2)) << endl;
    
    return 0;
}
```

## Stack Unwinding

When an exception is thrown, the stack is unwound until a matching catch block is found.

### Exercise 5: Stack Unwinding

Complete this stack unwinding example:

```cpp
#include <iostream>
#include <stdexcept>
#include <string>
using namespace std;

class Resource {
private:
    string name;
    
public:
    Resource(const string& n) : name(n) {
        cout << "Acquiring resource: " << name << endl;
    }
    
    ~Resource() {
        cout << "Releasing resource: " << name << endl;
    }
    
    void use() {
        cout << "Using resource: " << name << endl;
    }
};

void functionC() {
    cout << "Entering functionC" << endl;
    Resource r("Resource in C");
    r.use();
    
    cout << "About to throw exception in functionC" << endl;
    throw runtime_error("Exception from functionC");
    
    cout << "This line will not be executed" << endl;
}

void functionB() {
    cout << "Entering functionB" << endl;
    Resource r("Resource in B");
    r.use();
    
    functionC();  // This will throw an exception
    
    cout << "This line will not be executed" << endl;
}

void functionA() {
    cout << "Entering functionA" << endl;
    Resource r("Resource in A");
    r.use();
    
    functionB();  // This will propagate the exception
    
    cout << "This line will not be executed" << endl;
}

int main() {
    cout << "=== Stack Unwinding Demo ===" << endl;
    
    try {
        cout << "About to call functionA" << endl;
        functionA();
        cout << "This line will not be executed" << endl;
    } catch (const runtime_error& e) {
        cout << "Caught exception: " << e.what() << endl;
    }
    
    cout << "Program continues after exception handling" << endl;
    
    // Demonstrate partial unwinding
    cout << "\n=== Partial Unwinding Demo ===" << endl;
    
    try {
        try {
            functionA();  // Will throw exception
        } catch (const logic_error& e) {
            // This won't catch runtime_error, so exception propagates
            cout << "Caught logic_error: " << e.what() << endl;
        }
    } catch (const runtime_error& e) {
        cout << "Finally caught in outer handler: " << e.what() << endl;
    }
    
    // RAII in action during stack unwinding
    cout << "\n=== RAII During Unwinding ===" << endl;
    
    try {
        Resource outer("Outer Resource");
        outer.use();
        
        try {
            Resource inner("Inner Resource");
            inner.use();
            
            throw runtime_error("Inner exception");
        } catch (const logic_error& e) {
            cout << "This won't catch the exception" << endl;
        }
        
        cout << "This won't be executed" << endl;
    } catch (const runtime_error& e) {
        cout << "Caught: " << e.what() << endl;
    }
    
    cout << "Both resources automatically cleaned up!" << endl;
    
    return 0;
}
```

## Custom Exception Classes

Creating custom exception classes allows for more specific error handling.

### Exercise 6: Custom Exception Classes

Complete this custom exception example:

```cpp
#include <iostream>
#include <exception>
#include <string>
#include <vector>
using namespace std;

// Base custom exception class
class MathException : public exception {
protected:
    string message;
    double operand1, operand2;
    
public:
    MathException(const string& msg, double op1 = 0, double op2 = 0)
        : message(msg), operand1(op1), operand2(op2) {}
    
    const char* what() const noexcept override {
        return message.c_str();
    }
    
    double getOperand1() const { return operand1; }
    double getOperand2() const { return operand2; }
};

// Specific math exceptions
class DivisionByZeroException : public MathException {
public:
    DivisionByZeroException(double dividend)
        : MathException("Division by zero", dividend, 0) {}
};

class NegativeSqrtException : public MathException {
public:
    NegativeSqrtException(double value)
        : MathException("Square root of negative number", value, 0) {}
};

class OverflowException : public MathException {
public:
    OverflowException(double op1, double op2, const string& operation)
        : MathException("Overflow in " + operation, op1, op2) {}
};

// Calculator class that throws custom exceptions
class Calculator {
public:
    static double divide(double a, double b) {
        if (b == 0) {
            throw DivisionByZeroException(a);
        }
        return a / b;
    }
    
    static double sqrt(double x) {
        if (x < 0) {
            throw NegativeSqrtException(x);
        }
        return std::sqrt(x);
    }
    
    static double multiply(double a, double b) {
        // Simplified overflow check (in reality, this would be more complex)
        if (a > 1e100 && b > 1e100) {
            throw OverflowException(a, b, "multiplication");
        }
        return a * b;
    }
    
    static double add(double a, double b) {
        // Simplified overflow check
        if ((a > 0 && b > 0 && a > numeric_limits<double>::max() - b) ||
            (a < 0 && b < 0 && a < numeric_limits<double>::lowest() - b)) {
            throw OverflowException(a, b, "addition");
        }
        return a + b;
    }
};

int main() {
    cout << "=== Custom Exception Classes Demo ===" << endl;
    
    // Test division by zero
    try {
        double result = Calculator::divide(10, 0);
        cout << "Result: " << result << endl;
    } catch (const DivisionByZeroException& e) {
        cout << "Division by zero: " << e.what() << endl;
        cout << "Dividend was: " << e.getOperand1() << endl;
    } catch (const MathException& e) {
        cout << "Math exception: " << e.what() << endl;
    }
    
    // Test negative square root
    try {
        double result = Calculator::sqrt(-5);
        cout << "Result: " << result << endl;
    } catch (const NegativeSqrtException& e) {
        cout << "Negative square root: " << e.what() << endl;
        cout << "Value was: " << e.getOperand1() << endl;
    } catch (const MathException& e) {
        cout << "Math exception: " << e.what() << endl;
    }
    
    // Test overflow
    try {
        double result = Calculator::multiply(1e200, 1e200);
        cout << "Result: " << result << endl;
    } catch (const OverflowException& e) {
        cout << "Overflow: " << e.what() << endl;
        cout << "Operands were: " << e.getOperand1() << " and " << e.getOperand2() << endl;
    } catch (const MathException& e) {
        cout << "Math exception: " << e.what() << endl;
    }
    
    // Generic math exception handling
    vector<pair<double, double>> testCases = {
        {10, 2}, {10, 0}, {4, -2}, {-5, 3}
    };
    
    for (const auto& testCase : testCases) {
        try {
            double result = Calculator::divide(testCase.first, testCase.second);
            cout << testCase.first << " / " << testCase.second << " = " << result << endl;
        } catch (const MathException& e) {
            cout << "Math error: " << e.what() 
                 << " (operands: " << e.getOperand1() << ", " << e.getOperand2() << ")" << endl;
        } catch (const exception& e) {
            cout << "Standard exception: " << e.what() << endl;
        }
    }
    
    return 0;
}
```

## Exception Safety

Exception safety ensures that programs maintain invariants even when exceptions occur.

### Exercise 7: Exception Safety Levels

Complete this exception safety example:

```cpp
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
using namespace std;

// Class that demonstrates exception safety
class SafeContainer {
private:
    vector<unique_ptr<string>> data;
    
public:
    void addElement(const string& str) {
        // Strong exception safety: if this fails, object remains unchanged
        auto newElement = make_unique<string>(str);  // This might throw
        data.push_back(move(newElement));  // This won't throw
        // If make_unique throws, data remains unchanged
    }
    
    void addElementWeak(const string& str) {
        // Weak exception safety: if this fails, object might be in invalid state
        data.emplace_back(make_unique<string>(str));  // If make_unique throws after emplace_back,
                                                      // data might have a null pointer
    }
    
    void addElementNoexcept(const string& str) noexcept {
        // No-throw guarantee: this function never throws
        // In practice, we'd need to handle allocation failures differently
        data.emplace_back(make_unique<string>(str));
    }
    
    size_t size() const { return data.size(); }
    
    const string& operator[](size_t index) const {
        if (index >= data.size()) {
            throw out_of_range("Index out of bounds");
        }
        return *data[index];
    }
    
    void print() const {
        cout << "Container contents: ";
        for (const auto& element : data) {
            cout << *element << " ";
        }
        cout << endl;
    }
};

// Function demonstrating exception safety
void unsafeFunction(vector<int>& vec, int value) {
    vec.push_back(value);  // If this throws, vec might be in inconsistent state
    // If vec.push_back throws, the vector might have partially added the element
}

void safeFunction(vector<int>& vec, int value) {
    vector<int> temp = vec;  // Copy current state
    temp.push_back(value);   // If this throws, original vec is unchanged
    vec = move(temp);        // Commit the change
    // This provides strong guarantee but is inefficient
}

void betterSafeFunction(vector<int>& vec, int value) {
    vec.push_back(value);  // vector::push_back provides strong exception safety guarantee
    // If push_back throws, vec remains unchanged
}

class BankAccount {
private:
    string accountNumber;
    double balance;
    
public:
    BankAccount(const string& number, double initialBalance) 
        : accountNumber(number), balance(initialBalance) {}
    
    // Strong exception safety: if transfer fails, both accounts remain unchanged
    void transfer(BankAccount& toAccount, double amount) {
        if (amount <= 0) {
            throw invalid_argument("Transfer amount must be positive");
        }
        if (balance < amount) {
            throw runtime_error("Insufficient funds");
        }
        
        // Perform the transfer
        this->balance -= amount;      // These operations don't throw
        toAccount.balance += amount;  // So strong guarantee is maintained
    }
    
    double getBalance() const { return balance; }
    string getAccountNumber() const { return accountNumber; }
    
    void deposit(double amount) {
        if (amount <= 0) {
            throw invalid_argument("Deposit amount must be positive");
        }
        balance += amount;
    }
    
    void withdraw(double amount) {
        if (amount <= 0) {
            throw invalid_argument("Withdrawal amount must be positive");
        }
        if (balance < amount) {
            throw runtime_error("Insufficient funds");
        }
        balance -= amount;
    }
};

int main() {
    cout << "=== Exception Safety Demo ===" << endl;
    
    // Test SafeContainer
    SafeContainer container;
    
    try {
        container.addElement("First");
        container.addElement("Second");
        container.addElement("Third");
        container.print();
        
        // This will throw and demonstrate strong exception safety
        container.addElement(string(1000000, 'x'));  // Very large string that might cause allocation failure
    } catch (const bad_alloc& e) {
        cout << "Allocation failed, but container remains intact:" << endl;
        container.print();
        cout << "Size: " << container.size() << endl;
    }
    
    // Test bank account transfer
    cout << "\n=== Bank Account Transfer Demo ===" << endl;
    
    BankAccount account1("001", 1000.0);
    BankAccount account2("002", 500.0);
    
    cout << "Before transfer:" << endl;
    cout << "Account 1 (" << account1.getAccountNumber() << "): $" << account1.getBalance() << endl;
    cout << "Account 2 (" << account2.getAccountNumber() << "): $" << account2.getBalance() << endl;
    
    try {
        account1.transfer(account2, 200.0);
        cout << "Transfer successful!" << endl;
    } catch (const exception& e) {
        cout << "Transfer failed: " << e.what() << endl;
    }
    
    cout << "After transfer:" << endl;
    cout << "Account 1 (" << account1.getAccountNumber() << "): $" << account1.getBalance() << endl;
    cout << "Account 2 (" << account2.getAccountNumber() << "): $" << account2.getBalance() << endl;
    
    // Test insufficient funds
    cout << "\nTesting insufficient funds:" << endl;
    try {
        account1.transfer(account2, 10000.0);  // More than available
    } catch (const exception& e) {
        cout << "Transfer failed as expected: " << e.what() << endl;
    }
    
    cout << "After failed transfer:" << endl;
    cout << "Account 1 (" << account1.getAccountNumber() << "): $" << account1.getBalance() << endl;
    cout << "Account 2 (" << account2.getAccountNumber() << "): $" << account2.getBalance() << endl;
    
    return 0;
}
```

## Exception Handling in Constructors and Destructors

Special considerations apply to exceptions in constructors and destructors.

### Exercise 8: Exceptions in Constructors and Destructors

Complete this example:

```cpp
#include <iostream>
#include <stdexcept>
#include <memory>
using namespace std;

class Resource {
private:
    int* data;
    size_t size;
    string name;
    
public:
    // Constructor that might throw
    Resource(size_t s, const string& n) : size(s), name(n) {
        cout << "Constructing Resource: " << name << endl;
        
        if (size == 0) {
            throw invalid_argument("Size cannot be zero");
        }
        
        data = new int[size];  // This might throw bad_alloc
        
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<int>(i);
        }
        
        cout << "Resource " << name << " constructed successfully" << endl;
    }
    
    // Destructor - should not throw!
    ~Resource() {
        cout << "Destroying Resource: " << name << endl;
        delete[] data;  // This should not throw
        data = nullptr;
    }
    
    // Copy constructor
    Resource(const Resource& other) : size(other.size), name(other.name + "_copy") {
        cout << "Copy constructing Resource: " << name << endl;
        
        data = new int[size];  // This might throw bad_alloc
        for (size_t i = 0; i < size; i++) {
            data[i] = other.data[i];
        }
        
        cout << "Resource " << name << " copy constructed successfully" << endl;
    }
    
    // Move constructor
    Resource(Resource&& other) noexcept : data(other.data), size(other.size), name(other.name + "_moved") {
        other.data = nullptr;
        other.size = 0;
        cout << "Move constructing Resource: " << name << endl;
    }
    
    int& operator[](size_t index) { return data[index]; }
    const int& operator[](size_t index) const { return data[index]; }
    size_t getSize() const { return size; }
};

// Class with throwing destructor (BAD PRACTICE!)
class BadResource {
public:
    ~BadResource() {
        cout << "BadResource destructor about to throw!" << endl;
        throw runtime_error("Exception from destructor");  // NEVER DO THIS!
    }
};

void demonstrateConstructorExceptions() {
    cout << "=== Constructor Exception Demo ===" << endl;
    
    // Test constructor with valid parameters
    try {
        auto goodResource = make_unique<Resource>(5, "GoodResource");
        cout << "Resource created with size: " << goodResource->getSize() << endl;
    } catch (const exception& e) {
        cout << "Exception caught: " << e.what() << endl;
    }
    
    // Test constructor with invalid parameters
    try {
        auto badResource = make_unique<Resource>(0, "BadResource");  // Will throw
        cout << "This won't be printed" << endl;
    } catch (const invalid_argument& e) {
        cout << "Caught invalid argument: " << e.what() << endl;
    }
    
    // Test constructor that might throw bad_alloc
    try {
        // This might throw bad_alloc depending on available memory
        // auto hugeResource = make_unique<Resource>(numeric_limits<size_t>::max() / sizeof(int), "HugeResource");
        // For safety, we'll simulate this:
        throw bad_alloc();
    } catch (const bad_alloc& e) {
        cout << "Caught bad_alloc: " << e.what() << endl;
    }
    
    // Test copy constructor exceptions
    try {
        Resource original(3, "Original");
        Resource copy = original;  // Copy constructor called
        cout << "Copy created successfully" << endl;
    } catch (const exception& e) {
        cout << "Exception in copy: " << e.what() << endl;
    }
}

void demonstrateDestructorExceptions() {
    cout << "\n=== Destructor Exception Demo ===" << endl;
    
    cout << "Creating object with throwing destructor..." << endl;
    try {
        BadResource badObj;  // This object will throw from its destructor
        cout << "Object created, about to go out of scope..." << endl;
    } catch (const exception& e) {
        cout << "Caught exception: " << e.what() << endl;
    }
    // The exception from the destructor will call terminate()!
    // This is why destructors should never throw!
    
    cout << "This line may not be reached if terminate() is called" << endl;
}

// Safe way to handle potential errors in destructors
class SafeResource {
private:
    int* data;
    bool errorOccurred;
    
public:
    SafeResource() : data(new int(42)), errorOccurred(false) {}
    
    ~SafeResource() noexcept {  // Mark as noexcept to prevent terminate()
        try {
            // Do cleanup that might fail
            if (data) {
                delete data;
                data = nullptr;
            }
            
            // If there was a previous error condition, log it but don't throw
            if (errorOccurred) {
                cerr << "Warning: Resource had previous error condition" << endl;
            }
        } catch (...) {
            // If cleanup throws, catch it and don't rethrow
            cerr << "Error during cleanup, but not rethrowing from destructor" << endl;
        }
    }
    
    void setError() { errorOccurred = true; }
};

int main() {
    demonstrateConstructorExceptions();
    
    // Be careful with the destructor example - it will terminate the program!
    // demonstrateDestructorExceptions();
    
    cout << "\n=== Safe Resource Demo ===" << endl;
    {
        SafeResource safeObj;
        // Even if something goes wrong, the destructor won't throw
    }
    cout << "Safe resource destroyed without terminating program" << endl;
    
    // RAII with exception safety
    cout << "\n=== RAII Exception Safety Demo ===" << endl;
    
    try {
        auto ptr1 = make_unique<Resource>(3, "RAII_1");
        auto ptr2 = make_unique<Resource>(4, "RAII_2");
        
        cout << "About to throw exception..." << endl;
        throw runtime_error("Simulated exception");
        
        cout << "This won't be printed" << endl;
    } catch (const exception& e) {
        cout << "Caught exception: " << e.what() << endl;
        cout << "But resources were automatically cleaned up!" << endl;
    }
    
    return 0;
}
```

## Advanced Exception Handling Techniques

### Exercise 9: Advanced Exception Handling

Complete this advanced exception handling example:

```cpp
#include <iostream>
#include <exception>
#include <stdexcept>
#include <memory>
#include <functional>
#include <vector>
using namespace std;

// Exception wrapper for functions
template<typename Func>
auto exceptionGuard(Func&& func) -> decltype(func()) {
    try {
        return func();
    } catch (const exception& e) {
        cerr << "Exception in guarded function: " << e.what() << endl;
        throw;  // Re-throw the same exception
    } catch (...) {
        cerr << "Unknown exception in guarded function" << endl;
        throw;  // Re-throw
    }
}

// Function that returns success/failure with exception handling
template<typename T>
struct Result {
    bool success;
    T value;
    string errorMessage;
    
    Result(T&& v) : success(true), value(std::forward<T>(v)) {}
    Result(const string& msg) : success(false), errorMessage(msg) {}
};

template<typename Func>
auto safeCall(Func&& func) -> Result<decltype(func())> {
    try {
        return Result<decltype(func())>(func());
    } catch (const exception& e) {
        return Result<decltype(func())>(e.what());
    } catch (...) {
        return Result<decltype(func())>("Unknown error occurred");
    }
}

// Exception handling with polymorphism
class Processor {
public:
    virtual ~Processor() = default;
    virtual void process() = 0;
};

class SafeProcessor : public Processor {
public:
    void process() override {
        cout << "SafeProcessor: Performing safe operations..." << endl;
        // Might throw
        if (rand() % 3 == 0) {  // Simulate occasional failure
            throw runtime_error("Processing failed in SafeProcessor");
        }
        cout << "SafeProcessor: Operations completed successfully" << endl;
    }
};

class UnsafeProcessor : public Processor {
public:
    void process() override {
        cout << "UnsafeProcessor: Performing operations..." << endl;
        throw logic_error("Critical error in UnsafeProcessor");
    }
};

int main() {
    cout << "=== Advanced Exception Handling Demo ===" << endl;
    
    // Exception guard example
    cout << "\n--- Exception Guard Demo ---" << endl;
    try {
        auto result = exceptionGuard([]() {
            cout << "Executing guarded function..." << endl;
            if (true) {  // Simulate condition that causes exception
                throw invalid_argument("Guarded function error");
            }
            return 42;
        });
        cout << "Result: " << result << endl;
    } catch (const invalid_argument& e) {
        cout << "Caught in main: " << e.what() << endl;
    }
    
    // Safe call example
    cout << "\n--- Safe Call Demo ---" << endl;
    
    auto safeResult = safeCall([]() -> int {
        cout << "Executing safe call function..." << endl;
        if (rand() % 2 == 0) {
            throw runtime_error("Random error in safe call");
        }
        return 100;
    });
    
    if (safeResult.success) {
        cout << "Safe call succeeded: " << safeResult.value << endl;
    } else {
        cout << "Safe call failed: " << safeResult.errorMessage << endl;
    }
    
    // Polymorphic exception handling
    cout << "\n--- Polymorphic Exception Handling ---" << endl;
    
    vector<unique_ptr<Processor>> processors;
    processors.push_back(make_unique<SafeProcessor>());
    processors.push_back(make_unique<UnsafeProcessor>());
    
    for (size_t i = 0; i < processors.size(); i++) {
        cout << "Processing with processor " << i << ":" << endl;
        try {
            processors[i]->process();
        } catch (const runtime_error& e) {
            cout << "Runtime error caught: " << e.what() << endl;
        } catch (const logic_error& e) {
            cout << "Logic error caught: " << e.what() << endl;
        } catch (const exception& e) {
            cout << "General exception caught: " << e.what() << endl;
        }
        cout << "---" << endl;
    }
    
    // Nested exception handling
    cout << "\n--- Nested Exception Demo ---" << endl;
    
    try {
        try {
            throw runtime_error("Inner exception");
        } catch (const runtime_error& e) {
            // Re-throw with additional context
            throw runtime_error(string("Outer exception wrapping: ") + e.what());
        }
    } catch (const runtime_error& e) {
        cout << "Caught outer exception: " << e.what() << endl;
    }
    
    // Exception handling with smart pointers
    cout << "\n--- Smart Pointer Exception Safety ---" << endl;
    
    try {
        auto ptr = make_unique<int[]>(1000000);  // Large allocation
        
        for (int i = 0; i < 1000000; i++) {
            ptr[i] = i;
        }
        
        // Simulate error condition
        throw runtime_error("Error after allocation");
        
    } catch (const exception& e) {
        cout << "Exception caught: " << e.what() << endl;
        cout << "But smart pointer automatically cleaned up memory!" << endl;
    }
    
    // Function try blocks (constructor example)
    cout << "\n--- Function Try Block Demo ---" << endl;
    
    class FunctionTryBlockDemo {
    public:
        int value;
        
        FunctionTryBlockDemo(int v) try : value(v) {
            if (v < 0) {
                throw invalid_argument("Negative value not allowed");
            }
            cout << "Constructor completed successfully" << endl;
        } catch (const invalid_argument& e) {
            cout << "Caught in constructor try block: " << e.what() << endl;
            throw;  // Re-throw to prevent object construction
        }
    };
    
    try {
        FunctionTryBlockDemo obj(-5);  // Will throw
    } catch (const invalid_argument& e) {
        cout << "Caught in main: " << e.what() << endl;
    }
    
    cout << "\nProgram completed successfully!" << endl;
    
    return 0;
}
```

## Best Practices for Exception Handling

### Exercise 10: Exception Handling Best Practices

Demonstrate best practices for exception handling:

```cpp
#include <iostream>
#include <stdexcept>
#include <memory>
#include <vector>
#include <string>
using namespace std;

// Good: Specific exception types
class FileOpenException : public runtime_error {
public:
    FileOpenException(const string& filename) 
        : runtime_error("Could not open file: " + filename) {}
};

class NetworkException : public runtime_error {
public:
    NetworkException(const string& msg) : runtime_error("Network error: " + msg) {}
};

// Good: RAII wrapper for resource management
class FileManager {
private:
    string filename;
    bool isOpen;
    
public:
    explicit FileManager(const string& fname) : filename(fname), isOpen(false) {
        // Simulate file opening
        if (filename.empty()) {
            throw FileOpenException(filename);
        }
        isOpen = true;
        cout << "File " << filename << " opened successfully" << endl;
    }
    
    ~FileManager() {
        if (isOpen) {
            cout << "File " << filename << " closed" << endl;
        }
    }
    
    // Prevent copying to avoid double-close issues
    FileManager(const FileManager&) = delete;
    FileManager& operator=(const FileManager&) = delete;
    
    // Allow moving
    FileManager(FileManager&& other) noexcept 
        : filename(move(other.filename)), isOpen(other.isOpen) {
        other.isOpen = false;
    }
    
    void read() {
        if (!isOpen) {
            throw runtime_error("File not open for reading");
        }
        cout << "Reading from file: " << filename << endl;
    }
};

// Good: Exception-safe function
vector<int> processVector(const vector<int>& input) {
    vector<int> result;
    result.reserve(input.size());  // Reserve to prevent reallocation exceptions
    
    for (int value : input) {
        if (value < 0) {
            throw invalid_argument("Negative values not allowed");
        }
        result.push_back(value * 2);  // vector::push_back has strong exception safety
    }
    
    return result;  // NRVO or move semantics
}

int main() {
    cout << "=== Exception Handling Best Practices ===" << endl;
    
    // 1. Be specific with exception types
    try {
        throw FileOpenException("nonexistent.txt");
    } catch (const FileOpenException& e) {
        cout << "Specific file exception: " << e.what() << endl;
    }
    
    // 2. Use RAII for resource management
    try {
        FileManager fileManager("example.txt");
        fileManager.read();
        
        // Exception occurs, but FileManager destructor still runs
        if (true) {  // Simulate error condition
            throw runtime_error("Something went wrong");
        }
    } catch (const exception& e) {
        cout << "Caught: " << e.what() << endl;
        cout << "File was automatically closed due to RAII!" << endl;
    }
    
    // 3. Use smart pointers for automatic memory management
    try {
        auto data = make_unique<vector<int>>(1000000, 42);  // Large vector
        
        if (data->size() > 500000) {
            throw length_error("Vector too large");
        }
        
    } catch (const length_error& e) {
        cout << "Caught: " << e.what() << endl;
        cout << "Memory was automatically freed!" << endl;
    }
    
    // 4. Handle exceptions at the right level
    vector<int> testData = {1, 2, 3, 4, 5};
    
    try {
        auto result = processVector(testData);
        cout << "Processing successful, result size: " << result.size() << endl;
    } catch (const invalid_argument& e) {
        cout << "Input validation failed: " << e.what() << endl;
    } catch (const exception& e) {
        cout << "Processing failed: " << e.what() << endl;
    }
    
    // Test with invalid data
    vector<int> badData = {1, -2, 3};  // Contains negative number
    
    try {
        auto result = processVector(badData);
        cout << "This won't be printed" << endl;
    } catch (const invalid_argument& e) {
        cout << "Caught expected error: " << e.what() << endl;
    }
    
    // 5. Don't throw from destructors
    // Demonstrated earlier - destructors should be marked noexcept
    
    // 6. Use const references for exception parameters
    // Demonstrated throughout this example
    
    // 7. Provide meaningful error messages
    try {
        vector<int> vec;
        cout << vec.at(100) << endl;  // Will throw with meaningful message
    } catch (const out_of_range& e) {
        cout << "Helpful error message: " << e.what() << endl;
    }
    
    // 8. Consider exception safety guarantees
    vector<int> container = {1, 2, 3, 4, 5};
    
    cout << "Before risky operation: ";
    for (int x : container) cout << x << " ";
    cout << endl;
    
    try {
        // This operation provides strong exception safety guarantee
        auto temp = container;  // Copy current state
        temp.push_back(42);     // If this throws, original is unchanged
        temp.push_back(43);     // If this throws, original is unchanged
        container = move(temp); // Commit the changes
    } catch (const exception& e) {
        cout << "Operation failed: " << e.what() << endl;
        cout << "Container preserved: ";
        for (int x : container) cout << x << " ";
        cout << endl;
    }
    
    cout << "After risky operation: ";
    for (int x : container) cout << x << " ";
    cout << endl;
    
    // 9. Use standard exceptions when appropriate
    try {
        int divisor = 0;
        if (divisor == 0) {
            throw runtime_error("Division by zero detected");
        }
    } catch (const runtime_error& e) {
        cout << "Standard exception used appropriately: " << e.what() << endl;
    }
    
    cout << "\nAll best practices demonstrated successfully!" << endl;
    
    return 0;
}
```

## Summary

In this chapter, you learned:
- Basic exception handling with try, catch, and throw
- Standard exception hierarchies and custom exception classes
- Exception specifications and noexcept
- Stack unwinding and resource cleanup
- Exception safety levels (basic, strong, no-throw guarantees)
- Exception handling in constructors and destructors
- Advanced exception handling techniques
- Best practices for robust error handling

## Key Takeaways

- Use exceptions for error conditions, not normal control flow
- Always clean up resources properly using RAII
- Destructors should never throw exceptions
- Use specific exception types for better error handling
- Consider exception safety guarantees in your designs
- Use standard exceptions when appropriate
- Provide meaningful error messages
- Catch exceptions by const reference to avoid slicing
- Be careful with exception handling in constructors

## Common Mistakes to Avoid

1. Throwing exceptions from destructors
2. Not cleaning up resources when exceptions occur
3. Using exceptions for normal program flow
4. Throwing raw pointers that won't be caught
5. Not providing meaningful error messages
6. Catching exceptions by value (causes slicing)
7. Using catch-all handlers too broadly
8. Not considering exception safety in class design
9. Throwing exceptions that don't inherit from std::exception
10. Ignoring exception specifications and noexcept

## Next Steps

Now that you understand exception handling, you're ready to learn about advanced topics and best practices in Chapter 13.