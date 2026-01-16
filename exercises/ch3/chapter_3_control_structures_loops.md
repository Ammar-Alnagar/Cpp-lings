# Chapter 3: Control Structures and Loops

## Overview

This chapter covers the control flow structures in C++ that allow you to make decisions and repeat actions. You'll learn about conditional statements, loops, and jump statements that control the execution flow of your programs.

## Learning Objectives

By the end of this chapter, you will:
- Understand conditional statements (if, if-else, switch)
- Master different loop constructs (for, while, do-while)
- Learn about range-based for loops (C++11)
- Understand jump statements (break, continue, goto)
- Apply control structures to solve programming problems
- Recognize common patterns and best practices

## Conditional Statements

### If Statement

The `if` statement executes a block of code if a condition is true.

### Exercise 1: Basic If Statement

The following code has errors. Find and fix them:

```cpp
#include <iostream>
using namespace std;

int main() {
    int score;
    cout << "Enter your score: ";
    cin >> score;
    
    if (score >= 90) {
        cout << "Grade: A" << endl;
    }
    else if (score >= 80) {
        cout << "Grade: B" << endl;
    }
    else if (score >= 70) {
        cout << "Grade: C" << endl;
    }
    else if (score >= 60) {
        cout << "Grade: D" << endl;
    }
    else {
        cout << "Grade: F" << endl;
    }
    
    // Error: incorrect comparison
    if (score = 100) {  // Assignment instead of comparison
        cout << "Perfect score!" << endl;
    }
    
    return 0;
}
```

### Exercise 2: Nested If Statements

Fix the logical errors in this nested if example:

```cpp
#include <iostream>
using namespace std;

int main() {
    int age;
    bool has_license;
    
    cout << "Enter your age: ";
    cin >> age;
    cout << "Do you have a license? (1 for yes, 0 for no): ";
    cin >> has_license;
    
    if (age >= 16) {
        if (has_license) {
            cout << "You can drive legally." << endl;
        }
        else {
            cout << "You are old enough but need a license." << endl;
        }
    }
    else {
        cout << "You are too young to drive." << endl;
    }
    
    // Error: incorrect logic
    if (age >= 18) {
        if (has_license) {
            cout << "You can rent a car." << endl;
        }
        else {
            cout << "You can drive but cannot rent a car." << endl;  // Wrong!
        }
    }
    
    return 0;
}
```

## Switch Statement

The `switch` statement provides multi-way branching based on a variable's value.

### Exercise 3: Switch Statement

Complete this menu-driven program with errors:

```cpp
#include <iostream>
using namespace std;

int main() {
    int choice;
    
    cout << "Menu:" << endl;
    cout << "1. Option 1" << endl;
    cout << "2. Option 2" << endl;
    cout << "3. Option 3" << endl;
    cout << "4. Exit" << endl;
    cout << "Enter your choice: ";
    cin >> choice;
    
    switch (choice) {
        case 1:
            cout << "You selected Option 1" << endl;
            // Missing break statement - intentional error
        case 2:
            cout << "You selected Option 2" << endl;
            break;
        case 3:
            cout << "You selected Option 3" << endl;
            break;
        case 4:
            cout << "Exiting..." << endl;
            break;
        default:
            cout << "Invalid choice!" << endl;
            break;
    }
    
    // Error: switch with non-integral type
    char grade = 'A';
    switch (grade) {  // This is actually valid in C++
        case 'A':
            cout << "Excellent!" << endl;
            break;
        case 'B':
            cout << "Good!" << endl;
            break;
        default:
            cout << "Need improvement." << endl;
            break;
    }
    
    return 0;
}
```

## Loop Constructs

### For Loop

The `for` loop is ideal when you know the number of iterations in advance.

### Exercise 4: For Loop Basics

Fix the errors in this for loop example:

```cpp
#include <iostream>
using namespace std;

int main() {
    // Count from 1 to 10
    for (int i = 1; i <= 10; i++) {
        cout << i << " ";
    }
    cout << endl;
    
    // Error: infinite loop
    for (int i = 0; i < 10; i--) {  // i decreases instead of increases
        cout << i << " ";
    }
    cout << endl;
    
    // Error: variable scope issue
    for (int i = 0; i < 3; i++) {
        cout << i << " ";
    }
    cout << "Last value of i: " << i << endl;  // Error: i not accessible here
    
    // Error: modifying loop variable inside loop
    for (int i = 0; i < 5; i++) {
        cout << i << " ";
        i++;  // This creates unexpected behavior
    }
    cout << endl;
    
    return 0;
}
```

### While Loop

The `while` loop continues as long as a condition is true.

### Exercise 5: While Loop

Complete this number guessing game with errors:

```cpp
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

int main() {
    srand(time(0));
    int secret_number = rand() % 100 + 1;
    int guess;
    int attempts = 0;
    
    cout << "Guess the number (1-100): ";
    
    // Error: do-while logic misplaced
    while (guess != secret_number) {  // guess is uninitialized!
        cin >> guess;
        attempts++;
        
        if (guess < secret_number) {
            cout << "Too low! Try again: ";
        }
        else if (guess > secret_number) {
            cout << "Too high! Try again: ";
        }
        else {
            cout << "Congratulations! You guessed it in " << attempts << " attempts." << endl;
        }
    }
    
    return 0;
}
```

### Do-While Loop

The `do-while` loop guarantees at least one execution of the loop body.

### Exercise 6: Do-While Loop

Fix the menu loop with errors:

```cpp
#include <iostream>
using namespace std;

int main() {
    int choice;
    
    do {
        cout << "\nMenu:" << endl;
        cout << "1. Say Hello" << endl;
        cout << "2. Say Goodbye" << endl;
        cout << "3. Exit" << endl;
        cout << "Enter choice: ";
        cin >> choice;
        
        switch (choice) {
            case 1:
                cout << "Hello!" << endl;
                break;
            case 2:
                cout << "Goodbye!" << endl;
                break;
            case 3:
                cout << "Exiting..." << endl;
                break;
            default:
                cout << "Invalid option!" << endl;
        }
    } while (choice != 3);  // Error: what if user enters invalid input first?
    
    // Error: infinite loop possibility
    int counter = 0;
    do {
        cout << "Counter: " << counter << endl;
        counter++;
    } while (counter > 10);  // Condition is initially false, but loop still runs once
    
    return 0;
}
```

## Range-Based For Loop (C++11)

The range-based for loop simplifies iteration over containers.

### Exercise 7: Range-Based For Loop

Work with range-based for loops in this example:

```cpp
#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main() {
    vector<int> numbers = {1, 2, 3, 4, 5};
    
    // Basic range-based for
    cout << "Numbers: ";
    for (int num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    // Error: trying to modify elements (unless using reference)
    for (int num : numbers) {
        num *= 2;  // This modifies a copy, not the original
    }
    
    // Correct way to modify
    for (int& num : numbers) {  // Use reference to modify original
        num *= 2;
    }
    
    cout << "Doubled: ";
    for (const int& num : numbers) {  // Use const reference for efficiency
        cout << num << " ";
    }
    cout << endl;
    
    // Working with strings
    string text = "Hello";
    cout << "Characters: ";
    for (char c : text) {
        cout << c << " ";
    }
    cout << endl;
    
    // Error: trying to iterate over a scalar
    int x = 42;
    for (int val : x) {  // Error: x is not a range
        cout << val << endl;
    }
    
    return 0;
}
```

## Jump Statements

### Break Statement

The `break` statement exits the nearest enclosing loop or switch statement.

### Exercise 8: Break Statement

Fix the search algorithm with errors:

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> numbers = {10, 20, 30, 40, 50};
    int target = 30;
    bool found = false;
    
    for (int i = 0; i < numbers.size(); i++) {
        if (numbers[i] == target) {
            cout << "Found " << target << " at index " << i << endl;
            found = true;
            break;  // Exit the loop once found
        }
    }
    
    if (!found) {
        cout << target << " not found." << endl;
    }
    
    // Break in nested loops
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == 1 && j == 1) {
                break;  // Only breaks inner loop!
            }
            cout << "(" << i << "," << j << ") ";
        }
        cout << endl;
    }
    
    // Error: break outside of loop/switch
    if (true) {
        break;  // Error: break not inside loop or switch
    }
    
    return 0;
}
```

### Continue Statement

The `continue` statement skips the rest of the current iteration and proceeds to the next.

### Exercise 9: Continue Statement

Complete this filtering example with errors:

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    cout << "Even numbers: ";
    for (int num : numbers) {
        if (num % 2 != 0) {
            continue;  // Skip odd numbers
        }
        cout << num << " ";
    }
    cout << endl;
    
    // Error: continue in inappropriate context
    if (true) {
        continue;  // Error: continue not inside loop
    }
    
    // Practical example: skip processing certain items
    vector<string> items = {"apple", "", "banana", "", "cherry"};
    cout << "Non-empty items: ";
    for (const string& item : items) {
        if (item.empty()) {
            continue;  // Skip empty strings
        }
        cout << item << " ";
    }
    cout << endl;
    
    return 0;
}
```

## Loop Optimization and Best Practices

### Exercise 10: Loop Efficiency

Improve the efficiency of these loops:

```cpp
#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main() {
    vector<string> words = {"hello", "world", "cpp", "programming"};
    
    // Inefficient: calling size() in each iteration
    for (int i = 0; i < words.size(); i++) {  // size() called each time
        cout << words[i] << " ";
    }
    cout << endl;
    
    // Better: cache the size
    for (int i = 0, n = words.size(); i < n; i++) {
        cout << words[i] << " ";
    }
    cout << endl;
    
    // Even better: use range-based for or iterators
    for (const string& word : words) {
        cout << word << " ";
    }
    cout << endl;
    
    // Error: off-by-one error
    for (int i = 0; i <= words.size(); i++) {  // <= instead of <, causes out-of-bounds
        cout << words[i] << " ";
    }
    cout << endl;
    
    // Error: modifying container during iteration
    vector<int> nums = {1, 2, 3, 4, 5};
    for (size_t i = 0; i < nums.size(); i++) {
        if (nums[i] % 2 == 0) {
            nums.push_back(nums[i] * 2);  // This changes size() and can cause issues
        }
    }
    
    return 0;
}
```

## Practical Examples

### Exercise 11: Prime Number Checker

Complete this prime number checker with errors:

```cpp
#include <iostream>
#include <cmath>
using namespace std;

bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    
    // Error: inefficient loop condition
    for (int i = 5; i < n; i += 6) {  // Should be i*i <= n
        if (n % i == 0 || n % (i + 2) == 0) {
            return false;
        }
    }
    return true;
}

int main() {
    int number;
    cout << "Enter a number: ";
    cin >> number;
    
    if (isPrime(number)) {
        cout << number << " is prime." << endl;
    } else {
        cout << number << " is not prime." << endl;
    }
    
    // Find primes up to a limit
    int limit;
    cout << "Find primes up to: ";
    cin >> limit;
    
    cout << "Primes up to " << limit << ": ";
    for (int i = 2; i <= limit; i++) {
        if (isPrime(i)) {
            cout << i << " ";
        }
    }
    cout << endl;
    
    return 0;
}
```

### Exercise 12: Pattern Printing

Create programs to print patterns with errors to fix:

```cpp
#include <iostream>
using namespace std;

int main() {
    int rows;
    cout << "Enter number of rows: ";
    cin >> rows;
    
    // Print right triangle
    cout << "Right triangle:" << endl;
    for (int i = 1; i <= rows; i++) {
        for (int j = 1; j <= i; j++) {
            cout << "* ";
        }
        cout << endl;
    }
    
    // Print pyramid
    cout << "\nPyramid:" << endl;
    for (int i = 1; i <= rows; i++) {
        // Spaces
        for (int j = 1; j <= rows - i; j++) {  // Error: incorrect spacing
            cout << " ";
        }
        // Stars
        for (int k = 1; k <= 2 * i - 1; k++) {
            cout << "*";
        }
        cout << endl;
    }
    
    // Error: infinite loop example
    int countdown = 10;
    while (countdown > 0) {
        cout << countdown << " ";
        countdown--;
        if (countdown == 5) {
            continue;  // This doesn't affect the countdown
        }
    }
    cout << "Blast off!" << endl;
    
    return 0;
}
```

## Hands-On Project: Simple Calculator with Menu

### Exercise 13: Complete Calculator

Create a calculator program with a menu that has intentional errors:

```cpp
#include <iostream>
using namespace std;

int main() {
    double num1, num2, result;
    char op;
    int choice;
    
    do {
        cout << "\nCalculator Menu:" << endl;
        cout << "1. Addition" << endl;
        cout << "2. Subtraction" << endl;
        cout << "3. Multiplication" << endl;
        cout << "4. Division" << endl;
        cout << "5. Exit" << endl;
        cout << "Enter your choice: ";
        cin >> choice;
        
        if (choice >= 1 && choice <= 4) {
            cout << "Enter two numbers: ";
            cin >> num1 >> num2;
        }
        
        switch (choice) {
            case 1:
                result = num1 + num2;
                cout << num1 << " + " << num2 << " = " << result << endl;
                break;
            case 2:
                result = num1 - num2;
                cout << num1 << " - " << num2 << " = " << result << endl;
                break;
            case 3:
                result = num1 * num2;
                cout << num1 << " * " << num2 << " = " << result << endl;
                break;
            case 4:
                if (num2 != 0) {
                    result = num1 / num2;
                    cout << num1 << " / " << num2 << " = " << result << endl;
                } else {
                    cout << "Error: Division by zero!" << endl;
                }
                break;
            case 5:
                cout << "Exiting calculator..." << endl;
                break;
            default:
                cout << "Invalid choice! Please try again." << endl;
        }
    } while (choice != 5);
    
    // Error: unreachable code after infinite loop condition
    cout << "Thanks for using the calculator!" << endl;
    
    return 0;
}
```

## Best Practices

1. Use braces `{}` even for single statements in if/else blocks
2. Initialize loop variables properly
3. Be careful with loop termination conditions
4. Use range-based for loops when possible for containers
5. Avoid deep nesting of control structures
6. Use meaningful variable names for conditions
7. Consider using switch instead of long if-else chains for discrete values

## Summary

In this chapter, you learned:
- How to use conditional statements (if, if-else, switch)
- Different loop constructs (for, while, do-while, range-based for)
- Jump statements (break, continue)
- How to avoid common control structure errors
- Best practices for writing clean, readable code

## Key Takeaways

- Control structures determine the flow of execution in your programs
- Always initialize variables before using them in conditions
- Be careful with loop bounds to avoid off-by-one errors
- Use appropriate loop types for different situations
- Break and continue can make loops more efficient but use them judiciously
- Range-based for loops simplify container iteration

## Common Mistakes to Avoid

1. Using assignment (=) instead of equality comparison (==) in conditions
2. Forgetting break statements in switch cases
3. Creating infinite loops due to incorrect conditions
4. Off-by-one errors in loop bounds
5. Modifying loop variables inside the loop unexpectedly
6. Accessing array/container elements beyond their bounds
7. Not considering all possible cases in conditional logic

## Next Steps

Now that you understand control structures and loops, you're ready to learn about functions and scope in Chapter 4.