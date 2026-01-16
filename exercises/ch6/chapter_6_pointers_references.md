# Chapter 6: Pointers and References

## Overview

This chapter covers pointers and references in C++, which are fundamental concepts for memory management and efficient programming. You'll learn how to work with addresses, dynamically allocate memory, and understand the differences between pointers and references.

## Learning Objectives

By the end of this chapter, you will:
- Understand the concept of memory addresses and pointers
- Learn how to declare, initialize, and use pointers
- Master pointer arithmetic and array relationships
- Understand dynamic memory allocation with new/delete
- Learn about references and their differences from pointers
- Understand const correctness with pointers and references
- Learn about smart pointers for safer memory management
- Understand the relationship between arrays and pointers

## Memory Addresses and Pointer Basics

Every variable in C++ has a memory address that can be accessed using the address-of operator (&).

### Exercise 1: Basic Pointer Operations

The following code has errors. Find and fix them:

```cpp
#include <iostream>
using namespace std;

int main() {
    int value = 42;
    
    // Declare a pointer to int
    int* ptr;  // Uninitialized pointer - dangerous!
    
    // Error: assigning value to pointer instead of address
    ptr = value;  // Wrong! This assigns the value, not the address
    
    // Correct way: assign address of value to pointer
    ptr = &value;
    
    cout << "Value: " << value << endl;
    cout << "Address of value: " << &value << endl;
    cout << "Pointer value (address): " << ptr << endl;
    cout << "Value pointed to: " << *ptr << endl;  // Dereference the pointer
    
    // Change value through pointer
    *ptr = 100;
    cout << "New value: " << value << endl;
    
    // Error: uninitialized pointer
    int* bad_ptr;  // Contains garbage value
    // cout << *bad_ptr << endl;  // Dangerous! Dereferencing uninitialized pointer
    
    // Correct initialization
    int* good_ptr = nullptr;  // Initialize to null pointer
    cout << "Good pointer initialized to: " << good_ptr << endl;
    
    // Check for null before dereferencing
    if (good_ptr != nullptr) {
        cout << *good_ptr << endl;  // Won't execute
    } else {
        cout << "Pointer is null, cannot dereference." << endl;
    }
    
    return 0;
}
```

### Exercise 2: Pointer Declarations

Fix the confusion in these pointer declarations:

```cpp
#include <iostream>
using namespace std;

int main() {
    int a = 10, b = 20;
    
    // Correct way to declare multiple pointers
    int *p1, *p2;  // Both p1 and p2 are pointers to int
    p1 = &a;
    p2 = &b;
    
    cout << "*p1 = " << *p1 << ", *p2 = " << *p2 << endl;
    
    // Error: misleading declaration
    int* p3, p4;  // p3 is a pointer, p4 is an int!
    p3 = &a;
    // p4 = &b;  // Error: cannot assign address to int
    p4 = b;  // Correct assignment
    
    cout << "*p3 = " << *p3 << ", p4 = " << p4 << endl;
    
    // Error: pointer to wrong type
    double d = 3.14;
    // int* pd = &d;  // Error: cannot implicitly convert double* to int*
    int* pd = (int*)&d;  // C-style cast - dangerous!
    
    // Better approach: explicit cast if really needed
    int* pd_safe = reinterpret_cast<int*>(&d);  // C++-style cast
    
    cout << "Demonstrating different declaration styles:" << endl;
    int* ptr1, ptr2;  // ptr1 is pointer, ptr2 is int
    int *ptr3, *ptr4; // Both are pointers (preferred style)
    
    ptr3 = &a;
    ptr4 = &b;
    ptr2 = 50;  // ptr2 is an int, not a pointer
    
    cout << "*ptr3 = " << *ptr3 << ", *ptr4 = " << *ptr4 << ", ptr2 = " << ptr2 << endl;
    
    return 0;
}
```

## Pointer Arithmetic

Pointers support arithmetic operations that are scaled by the size of the data type they point to.

### Exercise 3: Pointer Arithmetic

Complete this pointer arithmetic example with errors:

```cpp
#include <iostream>
using namespace std;

int main() {
    int numbers[] = {10, 20, 30, 40, 50};
    int size = 5;
    
    // Pointer to the first element
    int* ptr = numbers;  // Same as &numbers[0]
    
    cout << "Array elements using pointer arithmetic:" << endl;
    for (int i = 0; i < size; i++) {
        cout << "*(ptr + " << i << ") = " << *(ptr + i) << endl;
    }
    
    // Moving the pointer
    cout << "\nMoving pointer:" << endl;
    cout << "Initial ptr points to: " << *ptr << endl;
    
    ptr++;  // Move to next element
    cout << "After ptr++, points to: " << *ptr << endl;
    
    ptr += 2;  // Move forward by 2 positions
    cout << "After ptr += 2, points to: " << *ptr << endl;
    
    ptr--;  // Move back one position
    cout << "After ptr--, points to: " << *ptr << endl;
    
    // Error: pointer arithmetic beyond array bounds
    int* end_ptr = numbers + size;  // Points to one past the end
    // cout << "Element at end_ptr: " << *end_ptr << endl;  // Undefined behavior!
    
    // Safe way to check bounds
    int* current = numbers;
    int* array_end = numbers + size;
    
    cout << "\nSafe iteration:" << endl;
    while (current != array_end) {
        cout << "Current value: " << *current << endl;
        current++;
    }
    
    // Calculating distance between pointers
    int* start = numbers;
    int* middle = numbers + 2;
    ptrdiff_t distance = middle - start;  // Safe way to calculate distance
    cout << "\nDistance between pointers: " << distance << endl;
    
    // Error: subtracting unrelated pointers
    int other_var = 100;
    int* other_ptr = &other_var;
    // ptrdiff_t invalid_dist = other_ptr - start;  // Undefined behavior!
    
    return 0;
}
```

## Dynamic Memory Allocation

C++ provides `new` and `delete` operators for dynamic memory allocation.

### Exercise 4: Dynamic Memory Allocation

Fix the memory management errors in this code:

```cpp
#include <iostream>
using namespace std;

int main() {
    // Allocate memory for a single integer
    int* ptr = new int;  // Allocates memory for one int
    *ptr = 42;
    cout << "Dynamically allocated value: " << *ptr << endl;
    
    // Initialize during allocation
    int* ptr2 = new int(100);  // Allocates and initializes
    cout << "Initialized value: " << *ptr2 << endl;
    
    // Allocate memory for an array
    int size = 5;
    int* arr = new int[size];  // Allocates array of 5 integers
    
    // Initialize the array
    for (int i = 0; i < size; i++) {
        arr[i] = (i + 1) * 10;
    }
    
    cout << "Dynamically allocated array: ";
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
    
    // Error: memory leak - forgetting to deallocate
    // The following line should come before return
    delete ptr;   // Free memory for single int
    delete ptr2;  // Free memory for single int
    delete[] arr; // Free memory for array (note the [])
    
    // Error: double deletion
    // delete ptr;  // Error: already deleted!
    
    // Error: deleting array with wrong operator
    int* array_ptr = new int[10];
    // delete array_ptr;  // Wrong! Should use delete[]
    delete[] array_ptr;  // Correct
    
    // Error: accessing deleted memory
    int* temp_ptr = new int(5);
    delete temp_ptr;
    // cout << *temp_ptr << endl;  // Error: accessing deleted memory (dangling pointer)
    
    // Safe approach: set pointer to nullptr after deletion
    int* safe_ptr = new int(15);
    delete safe_ptr;
    safe_ptr = nullptr;  // Prevent dangling pointer
    
    if (safe_ptr != nullptr) {
        cout << *safe_ptr << endl;  // Won't execute
    } else {
        cout << "Pointer is null, cannot dereference." << endl;
    }
    
    // Dynamic allocation with objects
    string* str_ptr = new string("Hello, Dynamic World!");
    cout << "Dynamically allocated string: " << *str_ptr << endl;
    delete str_ptr;
    
    return 0;
}
```

### Exercise 5: Memory Management Functions

Complete this example with proper memory management:

```cpp
#include <iostream>
#include <cstring>
using namespace std;

// Function to create a dynamic string
char* createString(const char* source) {
    if (source == nullptr) {
        return nullptr;
    }
    
    int len = strlen(source);
    char* newStr = new char[len + 1];  // +1 for null terminator
    
    strcpy(newStr, source);
    return newStr;
}

// Function to duplicate a string with error checking
char* safeDuplicate(const char* source) {
    if (source == nullptr) {
        return nullptr;
    }
    
    size_t len = strlen(source);
    char* result = new(nothrow) char[len + 1];  // nothrow version
    
    if (result == nullptr) {
        cout << "Memory allocation failed!" << endl;
        return nullptr;
    }
    
    strcpy(result, source);
    return result;
}

int main() {
    // Using the string creation function
    char* myString = createString("Hello, Dynamic Memory!");
    
    if (myString != nullptr) {
        cout << "Created string: " << myString << endl;
        delete[] myString;  // Don't forget to free!
        myString = nullptr; // Prevent dangling pointer
    }
    
    // Using safe duplication
    char* safeString = safeDuplicate("Safely duplicated string");
    
    if (safeString != nullptr) {
        cout << "Safely duplicated: " << safeString << endl;
        delete[] safeString;
        safeString = nullptr;
    }
    
    // Simulating memory allocation failure
    // This is hard to test, but the nothrow version handles it gracefully
    
    // Error: mismatched allocation/deallocation
    int* single_int = new int(42);
    int* array_int = new int[5];
    
    delete single_int;   // Correct
    delete[] array_int;  // Correct
    
    // Error demonstration (commented out to prevent undefined behavior):
    // int* ptr = new int(10);
    // delete[] ptr;  // Wrong! Should be delete
    // delete ptr;    // This would be correct for single allocation
    
    return 0;
}
```

## References

References provide an alias to an existing variable and are an alternative to pointers.

### Exercise 6: References Basics

Complete this reference example with errors:

```cpp
#include <iostream>
using namespace std;

int main() {
    int value = 42;
    
    // Reference declaration - must be initialized
    int& ref = value;  // ref is an alias for value
    
    cout << "Value: " << value << endl;
    cout << "Ref: " << ref << endl;
    cout << "Address of value: " << &value << endl;
    cout << "Address of ref: " << &ref << endl;  // Same as value's address
    
    // Modifying through reference
    ref = 100;  // Changes value
    cout << "After ref = 100, value is: " << value << endl;
    
    // Error: references must be initialized
    // int& bad_ref;  // Error: declaration of reference variable 'bad_ref' requires an initializer
    
    // References vs Pointers
    int x = 10, y = 20;
    int& r = x;  // r refers to x
    
    cout << "Initially: x=" << x << ", y=" << y << ", r=" << r << endl;
    
    r = y;  // This assigns y's VALUE to x (not making r refer to y!)
    cout << "After r = y: x=" << x << ", y=" << y << ", r=" << r << endl;
    // Notice x now equals y, but r still refers to x
    
    // Pointers can be reassigned
    int* ptr = &x;
    cout << "ptr points to x: " << *ptr << endl;
    ptr = &y;  // Now ptr points to y
    cout << "ptr now points to y: " << *ptr << endl;
    
    // Error: cannot reseat references
    // r = &y;  // Error: cannot assign address to reference
    
    // References to constants
    const int const_val = 50;
    const int& const_ref = const_val;  // Reference to const
    // const_ref = 60;  // Error: cannot modify through const reference
    
    cout << "Const reference: " << const_ref << endl;
    
    // Reference to pointer
    int* ptr_val = &x;
    int*& ptr_ref = ptr_val;  // Reference to pointer
    cout << "Value pointed to by ptr_val: " << *ptr_val << endl;
    cout << "Value pointed to by ptr_ref: " << *ptr_ref << endl;
    
    // Modifying through pointer reference
    *ptr_ref = 300;
    cout << "After *ptr_ref = 300, x = " << x << endl;
    
    return 0;
}
```

## Functions with Pointers and References

Functions can accept pointers and references as parameters.

### Exercise 7: Functions with Pointers

Fix the errors in these function examples:

```cpp
#include <iostream>
using namespace std;

// Function that modifies value through pointer
void modifyThroughPointer(int* ptr) {
    if (ptr != nullptr) {  // Always check for null
        *ptr = 100;
    }
}

// Function that swaps values using pointers
void swapWithPointers(int* a, int* b) {
    if (a != nullptr && b != nullptr) {  // Check both pointers
        int temp = *a;
        *a = *b;
        *b = temp;
    }
}

// Function that returns pointer (dangerous!)
int* dangerousFunction() {
    int local = 42;  // Local variable
    return &local;   // Error: returning address of local variable!
}

// Safer function that returns pointer
int* safeFunction() {
    int* heap_ptr = new int(42);  // Dynamically allocated
    return heap_ptr;  // Safe to return
}

int main() {
    int x = 10, y = 20;
    
    cout << "Before: x=" << x << ", y=" << y << endl;
    
    // Modify through pointer
    modifyThroughPointer(&x);
    cout << "After modifyThroughPointer(&x): x=" << x << ", y=" << y << endl;
    
    // Swap using pointers
    swapWithPointers(&x, &y);
    cout << "After swapWithPointers: x=" << x << ", y=" << y << endl;
    
    // Error: null pointer
    int* null_ptr = nullptr;
    modifyThroughPointer(null_ptr);  // Safe due to null check
    swapWithPointers(&x, null_ptr);  // Safe due to null check
    
    // Error: dangerous function
    // int* danger_ptr = dangerousFunction();  // Don't do this!
    // cout << "Dangerous result: " << *danger_ptr << endl;  // Undefined behavior!
    
    // Safe function
    int* safe_ptr = safeFunction();
    cout << "Safe result: " << *safe_ptr << endl;
    delete safe_ptr;  // Remember to free!
    
    return 0;
}
```

### Exercise 8: Functions with References

Complete this reference function example:

```cpp
#include <iostream>
#include <vector>
using namespace std;

// Function that modifies value through reference
void modifyThroughReference(int& ref) {
    ref = 500;
}

// Function that swaps values using references
void swapWithReferences(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

// Function that returns reference (can be dangerous)
int& dangerousReturnFunction(vector<int>& vec) {
    return vec[0];  // Returning reference to element in vector
}

// Function that accepts const reference (efficient and safe)
void printVector(const vector<int>& vec) {  // Pass by const reference - efficient!
    for (size_t i = 0; i < vec.size(); i++) {
        cout << vec[i] << " ";
    }
    cout << endl;
}

int main() {
    int x = 10, y = 20;
    
    cout << "Before: x=" << x << ", y=" << y << endl;
    
    // Modify through reference
    modifyThroughReference(x);
    cout << "After modifyThroughReference(x): x=" << x << ", y=" << y << endl;
    
    // Swap using references
    swapWithReferences(x, y);
    cout << "After swapWithReferences: x=" << x << ", y=" << y << endl;
    
    // Working with vectors
    vector<int> numbers = {1, 2, 3, 4, 5};
    cout << "Original vector: ";
    printVector(numbers);
    
    // Get reference to first element
    int& firstElement = dangerousReturnFunction(numbers);
    firstElement = 999;  // Modifies the vector
    cout << "After modifying through reference: ";
    printVector(numbers);
    
    // Error: cannot bind non-const reference to temporary
    // int& bad_ref = 42;  // Error: cannot bind non-const reference to rvalue
    const int& const_ref = 42;  // OK: const reference can bind to temporary
    
    // Error: returning reference to local variable
    // int& badReturn() {
    //     int local = 10;
    //     return local;  // Error: returning reference to local variable!
    // }
    
    return 0;
}
```

## Const Correctness

Understanding const with pointers and references is crucial for writing robust code.

### Exercise 9: Const Pointers and References

Complete this const correctness example:

```cpp
#include <iostream>
using namespace std;

int main() {
    int x = 10;
    int y = 20;
    const int z = 30;
    
    // Different const pointer declarations
    int* ptr1 = &x;              // Pointer to non-const int
    const int* ptr2 = &x;        // Pointer to const int
    int* const ptr3 = &x;        // Const pointer to non-const int
    const int* const ptr4 = &z;  // Const pointer to const int
    
    // Operations on ptr1 (pointer to non-const)
    *ptr1 = 100;  // OK: can modify value
    ptr1 = &y;    // OK: can change pointer
    
    // Operations on ptr2 (pointer to const)
    // *ptr2 = 200;  // Error: cannot modify value through const pointer
    ptr2 = &y;      // OK: can change pointer
    
    // Operations on ptr3 (const pointer to non-const)
    *ptr3 = 300;    // OK: can modify value
    // ptr3 = &y;   // Error: cannot change const pointer
    
    // Operations on ptr4 (const pointer to const)
    // *ptr4 = 400;  // Error: cannot modify value
    // ptr4 = &y;    // Error: cannot change pointer
    
    cout << "x = " << x << ", y = " << y << ", z = " << z << endl;
    
    // Const references
    int& ref1 = x;        // Non-const reference
    const int& ref2 = x;  // Const reference
    
    ref1 = 400;  // OK: can modify through non-const reference
    // ref2 = 500;  // Error: cannot modify through const reference
    
    cout << "After modifications: x = " << x << endl;
    
    // Const reference to temporary
    const int& temp_ref = x + y;  // OK: const reference can bind to temporary
    cout << "Temporary reference: " << temp_ref << endl;
    
    // Error: non-const reference to temporary
    // int& bad_ref = x + y;  // Error: cannot bind non-const reference to temporary
    
    return 0;
}
```

## Arrays and Pointers Relationship

Arrays and pointers are closely related in C++.

### Exercise 10: Array-Pointer Relationship

Complete this array-pointer relationship example:

```cpp
#include <iostream>
using namespace std;

int main() {
    int arr[] = {10, 20, 30, 40, 50};
    int size = 5;
    
    // Array name is a pointer to first element
    cout << "arr = " << arr << endl;
    cout << "&arr[0] = " << &arr[0] << endl;
    cout << "They are the same: " << (arr == &arr[0]) << endl;
    
    // Accessing elements
    cout << "\nAccessing elements:" << endl;
    for (int i = 0; i < size; i++) {
        cout << "arr[" << i << "] = " << arr[i] << " ";
        cout << "*(arr + " << i << ") = " << *(arr + i) << endl;
    }
    
    // Pointer to array
    int* ptr = arr;  // Points to first element
    
    cout << "\nUsing pointer to traverse:" << endl;
    for (int i = 0; i < size; i++) {
        cout << "Element " << i << ": " << *ptr << endl;
        ptr++;  // Move to next element
    }
    
    // Reset pointer
    ptr = arr;
    
    // Array of pointers
    int a = 1, b = 2, c = 3;
    int* ptrArr[] = {&a, &b, &c};  // Array of pointers
    
    cout << "\nArray of pointers:" << endl;
    for (size_t i = 0; i < 3; i++) {
        cout << "ptrArr[" << i << "] points to: " << *ptrArr[i] << endl;
    }
    
    // Pointer to array (different from array of pointers)
    int (*ptrToArray)[5] = &arr;  // Pointer to an array of 5 ints
    
    cout << "\nPointer to array:" << endl;
    cout << "(*ptrToArray)[2] = " << (*ptrToArray)[2] << endl;
    
    // Multidimensional arrays and pointers
    int matrix[3][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };
    
    cout << "\n2D array traversal with pointers:" << endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            cout << *(*(matrix + i) + j) << " ";  // Equivalent to matrix[i][j]
        }
        cout << endl;
    }
    
    // Error: array size in function parameter
    // When passing arrays to functions, they decay to pointers
    auto printArray = [](int arr[], int size) {  // arr[] is equivalent to int* arr
        for (int i = 0; i < size; i++) {
            cout << arr[i] << " ";
        }
        cout << endl;
    };
    
    printArray(arr, size);
    
    return 0;
}
```

## Smart Pointers (Modern C++)

Smart pointers provide automatic memory management.

### Exercise 11: Smart Pointers

Complete this smart pointer example:

```cpp
#include <iostream>
#include <memory>
#include <vector>
using namespace std;

class MyClass {
public:
    int value;
    
    MyClass(int v) : value(v) {
        cout << "MyClass constructor called, value = " << value << endl;
    }
    
    ~MyClass() {
        cout << "MyClass destructor called, value = " << value << endl;
    }
    
    void display() {
        cout << "Value: " << value << endl;
    }
};

int main() {
    // unique_ptr - exclusive ownership
    cout << "=== unique_ptr ===" << endl;
    {
        unique_ptr<MyClass> ptr1 = make_unique<MyClass>(42);
        ptr1->display();
        cout << "Value via *: " << (*ptr1).value << endl;
        
        // Transfer ownership (move semantics)
        unique_ptr<MyClass> ptr2 = move(ptr1);  // ptr1 becomes nullptr
        
        if (ptr1 == nullptr) {
            cout << "ptr1 is now nullptr" << endl;
        }
        
        ptr2->display();  // Only ptr2 can access the object now
    }  // ptr2 goes out of scope, object is automatically destroyed
    
    // shared_ptr - shared ownership
    cout << "\n=== shared_ptr ===" << endl;
    {
        shared_ptr<MyClass> sptr1 = make_shared<MyClass>(100);
        cout << "Reference count: " << sptr1.use_count() << endl;
        
        {
            shared_ptr<MyClass> sptr2 = sptr1;  // Share ownership
            cout << "Reference count: " << sptr1.use_count() << endl;
            cout << "Reference count: " << sptr2.use_count() << endl;
            
            sptr2->display();
        }  // sptr2 goes out of scope
        
        cout << "Reference count: " << sptr1.use_count() << endl;
        sptr1->display();
    }  // sptr1 goes out of scope, object destroyed when count reaches 0
    
    // weak_ptr - breaks circular references
    cout << "\n=== weak_ptr ===" << endl;
    {
        shared_ptr<MyClass> shared = make_shared<MyClass>(200);
        weak_ptr<MyClass> weak = shared;  // Does not increase reference count
        
        cout << "Shared count: " << shared.use_count() << endl;
        cout << "Weak expired: " << weak.expired() << endl;
        
        // Lock to access the object safely
        if (auto locked = weak.lock()) {
            cout << "Locked object value: " << locked->value << endl;
        }
        
        shared.reset();  // Release shared ownership
        cout << "After reset, weak expired: " << weak.expired() << endl;
        
        // Trying to lock now will return nullptr
        if (auto locked = weak.lock()) {
            cout << "This won't print" << endl;
        } else {
            cout << "Cannot lock expired weak_ptr" << endl;
        }
    }
    
    // Raw pointer vs smart pointer comparison
    cout << "\n=== Comparison ===" << endl;
    
    // Manual memory management (error-prone)
    MyClass* raw_ptr = new MyClass(300);
    raw_ptr->display();
    delete raw_ptr;  // Must remember to delete!
    
    // Smart pointer (automatic management)
    unique_ptr<MyClass> smart_ptr = make_unique<MyClass>(400);
    smart_ptr->display();
    // Automatic cleanup when going out of scope
    
    // Error: don't mix raw and smart pointers
    int* raw_int = new int(500);
    // unique_ptr<int> mixed_up(raw_int);  // OK, but risky
    // delete raw_int;  // Error: double delete!
    
    // Correct approach
    unique_ptr<int> correct_way = make_unique<int>(600);
    cout << "Smart pointer value: " << *correct_way << endl;
    
    return 0;
}
```

## Practical Examples

### Exercise 12: Pointer-Based String Functions

Implement string functions using pointers:

```cpp
#include <iostream>
using namespace std;

// Calculate string length using pointer arithmetic
size_t myStrlen(const char* str) {
    if (str == nullptr) return 0;
    
    const char* ptr = str;
    while (*ptr != '\0') {
        ptr++;
    }
    return ptr - str;  // Distance from start to end
}

// Copy string using pointers
char* myStrcpy(char* dest, const char* src) {
    if (dest == nullptr || src == nullptr) return dest;
    
    char* original_dest = dest;
    while ((*dest++ = *src++) != '\0') {
        // Copy each character including null terminator
    }
    return original_dest;
}

// Compare strings using pointers
int myStrcmp(const char* str1, const char* str2) {
    if (str1 == nullptr && str2 == nullptr) return 0;
    if (str1 == nullptr) return -1;
    if (str2 == nullptr) return 1;
    
    while (*str1 && (*str1 == *str2)) {
        str1++;
        str2++;
    }
    
    return *(unsigned char*)str1 - *(unsigned char*)str2;
}

int main() {
    // Test custom string functions
    const char* source = "Hello, World!";
    char buffer[50];
    
    cout << "Source string: " << source << endl;
    cout << "Length: " << myStrlen(source) << endl;
    
    myStrcpy(buffer, source);
    cout << "Copied string: " << buffer << endl;
    
    const char* test1 = "Apple";
    const char* test2 = "Banana";
    const char* test3 = "Apple";
    
    cout << "Comparisons:" << endl;
    cout << "Apple vs Banana: " << myStrcmp(test1, test2) << endl;
    cout << "Apple vs Apple: " << myStrcmp(test1, test3) << endl;
    cout << "Banana vs Apple: " << myStrcmp(test2, test1) << endl;
    
    // Error: buffer overflow
    char small_buffer[5];
    // myStrcpy(small_buffer, "This string is too long");  // Buffer overflow!
    
    // Safer approach would check lengths first
    const char* long_string = "This string is too long";
    if (myStrlen(long_string) < sizeof(small_buffer)) {
        myStrcpy(small_buffer, long_string);
    } else {
        cout << "Warning: string too long for buffer!" << endl;
    }
    
    return 0;
}
```

### Exercise 13: Dynamic Array Class

Create a simple dynamic array class:

```cpp
#include <iostream>
#include <stdexcept>
using namespace std;

template<typename T>
class DynamicArray {
private:
    T* data;
    size_t size_;
    size_t capacity_;

public:
    // Constructor
    explicit DynamicArray(size_t initial_capacity = 10) 
        : size_(0), capacity_(initial_capacity) {
        data = new T[capacity_];
    }
    
    // Destructor
    ~DynamicArray() {
        delete[] data;
    }
    
    // Copy constructor
    DynamicArray(const DynamicArray& other) 
        : size_(other.size_), capacity_(other.capacity_) {
        data = new T[capacity_];
        for (size_t i = 0; i < size_; i++) {
            data[i] = other.data[i];
        }
    }
    
    // Assignment operator
    DynamicArray& operator=(const DynamicArray& other) {
        if (this != &other) {  // Self-assignment check
            delete[] data;  // Clean up existing memory
            
            size_ = other.size_;
            capacity_ = other.capacity_;
            data = new T[capacity_];
            for (size_t i = 0; i < size_; i++) {
                data[i] = other.data[i];
            }
        }
        return *this;
    }
    
    // Move constructor
    DynamicArray(DynamicArray&& other) noexcept 
        : data(other.data), size_(other.size_), capacity_(other.capacity_) {
        other.data = nullptr;  // Prevent double deletion
        other.size_ = 0;
        other.capacity_ = 0;
    }
    
    // Move assignment operator
    DynamicArray& operator=(DynamicArray&& other) noexcept {
        if (this != &other) {
            delete[] data;  // Clean up existing memory
            
            data = other.data;
            size_ = other.size_;
            capacity_ = other.capacity_;
            
            other.data = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }
    
    // Access operators
    T& operator[](size_t index) {
        if (index >= size_) {
            throw out_of_range("Index out of bounds");
        }
        return data[index];
    }
    
    const T& operator[](size_t index) const {
        if (index >= size_) {
            throw out_of_range("Index out of bounds");
        }
        return data[index];
    }
    
    // Add element
    void push_back(const T& value) {
        if (size_ >= capacity_) {
            reserve(capacity_ * 2);  // Double capacity
        }
        data[size_++] = value;
    }
    
    // Remove last element
    void pop_back() {
        if (size_ > 0) {
            size_--;
        }
    }
    
    // Get size
    size_t size() const { return size_; }
    
    // Check if empty
    bool empty() const { return size_ == 0; }
    
    // Get capacity
    size_t capacity() const { return capacity_; }
    
    // Reserve more space
    void reserve(size_t new_capacity) {
        if (new_capacity > capacity_) {
            T* new_data = new T[new_capacity];
            for (size_t i = 0; i < size_; i++) {
                new_data[i] = data[i];
            }
            delete[] data;
            data = new_data;
            capacity_ = new_capacity;
        }
    }
    
    // Clear all elements
    void clear() { size_ = 0; }
};

int main() {
    // Test the dynamic array
    DynamicArray<int> arr;
    
    cout << "Initial size: " << arr.size() << ", capacity: " << arr.capacity() << endl;
    
    // Add elements
    for (int i = 1; i <= 5; i++) {
        arr.push_back(i * 10);
    }
    
    cout << "After adding elements:" << endl;
    cout << "Size: " << arr.size() << ", Capacity: " << arr.capacity() << endl;
    cout << "Elements: ";
    for (size_t i = 0; i < arr.size(); i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
    
    // Test copy constructor
    DynamicArray<int> arr2 = arr;  // Copy
    cout << "Copy - Elements: ";
    for (size_t i = 0; i < arr2.size(); i++) {
        cout << arr2[i] << " ";
    }
    cout << endl;
    
    // Test assignment
    DynamicArray<int> arr3;
    arr3 = arr;  // Assignment
    cout << "Assignment - Elements: ";
    for (size_t i = 0; i < arr3.size(); i++) {
        cout << arr3[i] << " ";
    }
    cout << endl;
    
    // Test bounds checking
    try {
        cout << arr[10] << endl;  // Should throw exception
    } catch (const out_of_range& e) {
        cout << "Caught exception: " << e.what() << endl;
    }
    
    return 0;
}
```

## Best Practices

1. Prefer references to pointers when you don't need reseating
2. Use smart pointers instead of raw pointers when possible
3. Always initialize pointers, preferably to nullptr
4. Check for null before dereferencing pointers
5. Match new with delete and new[] with delete[]
6. Follow RAII (Resource Acquisition Is Initialization) principle
7. Use const whenever possible for safety
8. Implement the Rule of Three/Five for classes managing resources

## Summary

In this chapter, you learned:
- How pointers work and how to use them safely
- Dynamic memory allocation with new/delete
- References and how they differ from pointers
- Const correctness with pointers and references
- The relationship between arrays and pointers
- Modern smart pointers for safer memory management
- Practical applications of pointers and references

## Key Takeaways

- Pointers store memory addresses and require careful management
- References provide aliases and cannot be null or reassigned
- Dynamic memory allocation requires matching allocation/deallocation
- Smart pointers automate memory management
- Const correctness prevents unintended modifications
- The Rule of Three/Five is important for resource management

## Common Mistakes to Avoid

1. Dereferencing null or uninitialized pointers
2. Memory leaks from forgetting to delete allocated memory
3. Double deletion of the same memory
4. Accessing deleted memory (dangling pointers)
5. Mismatching new/delete or new[]/delete[]
6. Returning references to local variables
7. Not implementing proper copy/move semantics for classes managing resources
8. Using raw pointers when smart pointers would be safer

## Next Steps

Now that you understand pointers and references, you're ready to learn about object-oriented programming basics in Chapter 7.