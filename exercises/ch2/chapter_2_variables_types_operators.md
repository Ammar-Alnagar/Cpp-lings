# Chapter 2: Variables, Data Types, and Operators

## Overview

This chapter covers the fundamental data types in C++, how to declare and initialize variables, and the various operators available in C++. Understanding these concepts is essential for writing any meaningful C++ program.

## Learning Objectives

By the end of this chapter, you will:
- Understand the basic data types in C++
- Know how to declare and initialize variables
- Learn about type modifiers and qualifiers
- Master arithmetic, relational, logical, and bitwise operators
- Understand operator precedence and associativity
- Learn about type conversions and casting

## Basic Data Types

C++ provides several fundamental data types:

- `bool` - Boolean values (true/false)
- `char` - Character values
- `int` - Integer values
- `float` - Single-precision floating-point
- `double` - Double-precision floating-point
- `void` - No type

### Exercise 1: Data Type Declaration

The following code has errors in variable declarations. Find and fix them:

```cpp
#include <iostream>
using namespace std;

int main() {
    bool flag = true;
    char letter = 'A';
    int count = 100;
    float price = 19.99;  // Potential precision loss
    double precise = 19.99;
    void nothing;         // Error: void cannot be used as variable type
    
    cout << "Flag: " << flag << endl;
    cout << "Letter: " << letter << endl;
    cout << "Count: " << count << endl;
    cout << "Price: " << price << endl;
    cout << "Precise: " << precise << endl;
    
    return 0;
}
```

## Type Modifiers

C++ provides modifiers that alter the meaning of basic data types:
- `signed`/`unsigned` - signedness of integers
- `short`/`long`/`long long` - size variations
- `const` - immutable variables
- `volatile` - may be changed by external sources

### Exercise 2: Type Modifiers

Fix the errors in this code using appropriate type modifiers:

```cpp
#include <iostream>
#include <climits>
using namespace std;

int main() {
    int small_num = -32768;
    int large_num = 2147483648;  // Error: too large for int
    unsigned int pos_only = -5;   // Warning: negative assigned to unsigned
    
    cout << "Small number: " << small_num << endl;
    cout << "Large number: " << large_num << endl;
    cout << "Positive only: " << pos_only << endl;
    
    // Print size information
    cout << "Size of int: " << sizeof(int) << " bytes" << endl;
    cout << "Size of long: " << sizeof(long) << " bytes" << endl;
    cout << "Size of short: " << sizeof(short) << " bytes" << endl;
    
    return 0;
}
```

## Variable Declaration and Initialization

Variables can be initialized in several ways:
- Copy initialization: `int x = 5;`
- Direct initialization: `int x(5);`
- Uniform initialization: `int x{5};` (C++11)

### Exercise 3: Initialization Methods

The following code demonstrates different initialization methods but has errors:

```cpp
#include <iostream>
using namespace std;

int main() {
    int a = 5;      // Copy initialization
    int b(10);      // Direct initialization  
    int c{15};      // Uniform initialization
    int d = {20};   // Copy list initialization
    
    // Error: narrowing conversion
    int e{3.14};    // Will cause error in most compilers
    
    cout << "a: " << a << endl;
    cout << "b: " << b << endl;
    cout << "c: " << c << endl;
    cout << "d: " << d << endl;
    cout << "e: " << e << endl;  // This won't compile if e causes error
    
    return 0;
}
```

## Constants and Immutability

Use `const` to create immutable variables:

### Exercise 4: Constant Variables

Fix the issues with constant variables in this code:

```cpp
#include <iostream>
using namespace std;

int main() {
    const int MAX_SIZE = 100;
    int arr[MAX_SIZE];
    
    // Error: trying to modify a const variable
    MAX_SIZE = 200;
    
    // Correct usage
    for (int i = 0; i < MAX_SIZE; ++i) {
        arr[i] = i;
    }
    
    cout << "Array size: " << MAX_SIZE << endl;
    
    return 0;
}
```

## Arithmetic Operators

C++ supports the standard arithmetic operators:
- `+` (addition)
- `-` (subtraction)
- `*` (multiplication)
- `/` (division)
- `%` (modulus)

### Exercise 5: Arithmetic Operations

Complete this calculator program with intentional errors:

```cpp
#include <iostream>
using namespace std;

int main() {
    int a = 10, b = 3;
    double da = 10.0, db = 3.0;
    
    cout << "Integer operations:" << endl;
    cout << a << " + " << b << " = " << a + b << endl;
    cout << a << " - " << b << " = " << a - b << endl;
    cout << a << " * " << b << " = " << a * b << endl;
    cout << a << " / " << b << " = " << a / b << endl;  // Integer division
    cout << a << " % " << b << " = " << a % b << endl;
    
    cout << "\nFloating-point operations:" << endl;
    cout << da << " / " << db << " = " << da / db << endl;
    
    // Error: modulus with floating-point
    cout << da << " % " << db << " = " << da % db << endl;
    
    return 0;
}
```

## Increment and Decrement Operators

C++ provides `++` and `--` operators with prefix and postfix forms.

### Exercise 6: Increment/Decrement Behavior

Understand the difference between prefix and postfix operators:

```cpp
#include <iostream>
using namespace std;

int main() {
    int x = 5, y = 5;
    
    // Prefix: increment then use
    cout << "Prefix ++x: " << ++x << endl;  // Should print 6
    cout << "x after prefix: " << x << endl;  // Should print 6
    
    // Postfix: use then increment
    cout << "Postfix y++: " << y++ << endl;  // Should print 5
    cout << "y after postfix: " << y << endl;  // Should print 6
    
    // More complex expressions
    int a = 10;
    int b = ++a + a++;  // What is the value of b?
    
    cout << "a: " << a << ", b: " << b << endl;
    
    return 0;
}
```

## Relational and Logical Operators

Relational: `==`, `!=`, `<`, `>`, `<=`, `>=`
Logical: `&&`, `||`, `!`

### Exercise 7: Boolean Logic

Fix the logical errors in this validation code:

```cpp
#include <iostream>
using namespace std;

int main() {
    int age;
    cout << "Enter your age: ";
    cin >> age;
    
    // Check if age is valid (between 0 and 150)
    if (age >= 0 && age <= 150) {
        cout << "Valid age entered." << endl;
        
        // Check if adult (18 or older)
        if (age => 18) {  // Error: wrong operator
            cout << "You are an adult." << endl;
        } else {
            cout << "You are a minor." << endl;
        }
    } else {
        cout << "Invalid age entered!" << endl;
    }
    
    // Logical operator practice
    bool has_license = true;
    bool is_employed = false;
    bool is_student = true;
    
    // Can get discount if student OR unemployed
    if (is_student || !is_employed) {  // Error in logic?
        cout << "Eligible for discount." << endl;
    }
    
    return 0;
}
```

## Bitwise Operators

C++ provides bitwise operators: `&`, `|`, `^`, `~`, `<<`, `>>`

### Exercise 8: Bitwise Operations

Complete this bit manipulation program with errors:

```cpp
#include <iostream>
#include <bitset>
using namespace std;

int main() {
    unsigned char a = 5;   // Binary: 00000101
    unsigned char b = 3;   // Binary: 00000011
    
    cout << "a = " << static_cast<int>(a) << " (" << bitset<8>(a) << ")" << endl;
    cout << "b = " << static_cast<int>(b) << " (" << bitset<8>(b) << ")" << endl;
    
    cout << "a & b = " << (a & b) << " (" << bitset<8>(a & b) << ")" << endl;
    cout << "a | b = " << (a | b) << " (" << bitset<8>(a | b) << ")" << endl;
    cout << "a ^ b = " << (a ^ b) << " (" << bitset<8>(a ^ b) << ")" << endl;
    cout << "~a = " << (~a) << " (" << bitset<8>(~a) << ")" << endl;  // Error: unexpected result?
    
    // Shift operations
    cout << "a << 1 = " << (a << 1) << " (" << bitset<8>(a << 1) << ")" << endl;
    cout << "a >> 1 = " << (a >> 1) << " (" << bitset<8>(a >> 1) << ")" << endl;
    
    return 0;
}
```

## Assignment Operators

C++ provides compound assignment operators: `+=`, `-=`, `*=`, `/=`, `%=`, etc.

### Exercise 9: Compound Assignment

Fix the assignment operator errors in this accumulation program:

```cpp
#include <iostream>
using namespace std;

int main() {
    int sum = 0;
    int product = 1;
    
    for (int i = 1; i <= 5; ++i) {
        sum += i;      // sum = sum + i
        product *= i;  // product = product * i
        
        cout << "i=" << i << ", sum=" << sum << ", product=" << product << endl;
    }
    
    // Error demonstration
    sum =+ 10;  // What does this do? (Hint: it's sum = +10)
    
    cout << "Final sum: " << sum << endl;
    
    return 0;
}
```

## Operator Precedence and Associativity

Understanding operator precedence is crucial for writing correct expressions.

### Exercise 10: Operator Precedence

Predict and verify the output of this expression evaluation:

```cpp
#include <iostream>
using namespace std;

int main() {
    int a = 5, b = 10, c = 3;
    
    // Without parentheses - depends on precedence
    int result1 = a + b * c;        // Should be 5 + (10 * 3) = 35
    int result2 = a * b + c;        // Should be (5 * 10) + 3 = 53
    int result3 = a + b < c;        // Should be (a + b) < c = false (0)
    int result4 = a == b && c > 0;  // Error in precedence?
    
    cout << "a + b * c = " << result1 << endl;
    cout << "a * b + c = " << result2 << endl;
    cout << "a + b < c = " << result3 << endl;
    cout << "a == b && c > 0 = " << result4 << endl;
    
    // Fix with parentheses to show intended behavior
    int correct_result = ((a == b) && (c > 0));
    cout << "Corrected: " << correct_result << endl;
    
    return 0;
}
```

## Type Conversion and Casting

C++ performs implicit conversions and supports explicit casting.

### Exercise 11: Type Conversions

Analyze and fix the type conversion issues in this code:

```cpp
#include <iostream>
using namespace std;

int main() {
    int int_val = 10;
    double double_val = 3.7;
    char char_val = 'A';
    
    // Implicit conversions
    double result1 = int_val + double_val;  // int promoted to double
    int result2 = int_val + char_val;       // char promoted to int
    int result3 = int_val / 3.0;           // int converted to double then back to int?
    
    cout << "int + double = " << result1 << endl;
    cout << "int + char = " << result2 << endl;
    cout << "int / 3.0 = " << result3 << endl;
    
    // Explicit casting
    int truncated = (int)double_val;              // C-style cast
    int also_truncated = static_cast<int>(double_val);  // C++-style cast
    
    cout << "Truncated (C-style): " << truncated << endl;
    cout << "Truncated (C++-style): " << also_truncated << endl;
    
    // Error: narrowing conversion in initialization
    int bad_conversion{double_val};  // Error in most compilers
    
    return 0;
}
```

## Special Literals

C++ provides special literals for different types:
- Integer: `123`, `0x7B` (hex), `0173` (octal), `0b1111011` (binary, C++14)
- Floating-point: `3.14`, `3.14f`, `3.14L`
- Character: `'A'`, `'\n'`, `'\x41'`
- String: `"Hello"`

### Exercise 12: Literal Types

Work with different literal types in this program:

```cpp
#include <iostream>
using namespace std;

int main() {
    // Integer literals
    int decimal = 123;
    int hex = 0x7B;
    int octal = 0173;
    int binary = 0b1111011;  // C++14 feature
    
    cout << "Decimal: " << decimal << endl;
    cout << "Hex: " << hex << endl;
    cout << "Octal: " << octal << endl;
    cout << "Binary: " << binary << endl;
    
    // Floating-point literals
    float f = 3.14f;
    double d = 3.14;
    long double ld = 3.14L;
    
    cout << "Float: " << f << endl;
    cout << "Double: " << d << endl;
    cout << "Long double: " << ld << endl;
    
    // Character literals
    char c1 = 'A';
    char c2 = 65;        // ASCII value
    char c3 = '\x41';    // Hexadecimal
    char c4 = '\101';    // Octal
    
    cout << "Char 1: " << c1 << endl;
    cout << "Char 2: " << c2 << endl;
    cout << "Char 3: " << c3 << endl;
    cout << "Char 4: " << c4 << endl;
    
    // Error: invalid character literal
    char bad_char = 'AB';  // Error: character literal too long
    
    return 0;
}
```

## Hands-On Project: Unit Converter

### Exercise 13: Complete Unit Converter

Create a program that converts between different units. The template below has intentional errors:

```cpp
#include <iostream>
using namespace std;

int main() {
    double value;
    int choice;
    
    cout << "Unit Converter" << endl;
    cout << "1. Inches to Centimeters" << endl;
    cout << "2. Pounds to Kilograms" << endl;
    cout << "3. Fahrenheit to Celsius" << endl;
    cout << "Enter your choice: ";
    cin >> choice;
    
    cout << "Enter value to convert: ";
    cin >> value;
    
    double result;
    switch(choice) {
        case 1:
            result = value * 2.54;  // inches to cm
            cout << value << " inches = " << result << " centimeters" << endl;
            break;
        case 2:
            result = value * 0.453592;  // pounds to kg
            cout << value << " pounds = " << result << " kilograms" << endl;
            break;
        case 3:
            result = (value - 32) * 5/9;  // F to C
            cout << value << " Fahrenheit = " << result << " Celsius" << endl;
            break;
        default:
            cout << "Invalid choice!" << endl;
    }
    
    // Error: potential division by zero in reverse calculation
    if (choice == 3) {
        double reverse = result * 9/5 + 32;  // Is this correct?
        cout << "Reverse check: " << reverse << " Fahrenheit" << endl;
    }
    
    return 0;
}
```

## Best Practices

1. Initialize variables when declaring them
2. Use appropriate data types for your needs
3. Be aware of integer overflow and underflow
4. Use `static_cast` for explicit conversions
5. Understand operator precedence to avoid errors
6. Use `const` for values that shouldn't change

## Summary

In this chapter, you learned:
- The fundamental data types in C++
- How to declare and initialize variables
- The various type modifiers and qualifiers
- All the arithmetic, relational, logical, and bitwise operators
- Operator precedence and associativity rules
- Type conversion and casting techniques
- Different literal types in C++

## Key Takeaways

- Choose the right data type for your variables
- Initialize variables to avoid undefined behavior
- Understand the difference between prefix and postfix increment/decrement
- Be careful with operator precedence in complex expressions
- Use appropriate casting methods for type conversions
- Constants help prevent accidental modifications

## Common Mistakes to Avoid

1. Using uninitialized variables
2. Integer division when floating-point is expected
3. Confusing assignment (=) with equality comparison (==)
4. Overflow with integer types
5. Precision issues with floating-point comparisons
6. Misunderstanding operator precedence

## Next Steps

Now that you understand variables, data types, and operators, you're ready to learn about control structures and loops in Chapter 3.