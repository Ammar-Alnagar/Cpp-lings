# Chapter 5: Arrays and Strings

## Overview

This chapter covers arrays and strings in C++, which are fundamental for storing and manipulating collections of data. You'll learn about C-style arrays, C++ strings, and the modern std::array and std::vector containers.

## Learning Objectives

By the end of this chapter, you will:
- Understand C-style arrays and their limitations
- Learn to work with C++ strings effectively
- Master std::array and std::vector containers
- Understand string manipulation techniques
- Learn about multidimensional arrays
- Understand bounds checking and safety considerations
- Explore string streams for parsing and formatting

## C-Style Arrays

C-style arrays are fixed-size collections of elements of the same type.

### Exercise 1: Basic C-Style Arrays

The following code has errors. Find and fix them:

```cpp
#include <iostream>
using namespace std;

int main() {
    // Declaring and initializing arrays
    int numbers[5];  // Array of 5 integers
    int values[] = {10, 20, 30, 40, 50};  // Size determined by initializer
    
    // Error: accessing array out of bounds
    for (int i = 0; i <= 5; i++) {  // Should be i < 5
        numbers[i] = i * 10;
        cout << "numbers[" << i << "] = " << numbers[i] << endl;
    }
    
    // Error: uninitialized array elements
    int uninitialized[5];
    for (int i = 0; i < 5; i++) {
        cout << "uninitialized[" << i << "] = " << uninitialized[i] << endl;  // Undefined values
    }
    
    // Correct way: initialize all elements to 0
    int initialized[5] = {0};  // First element is 0, rest are automatically 0
    for (int i = 0; i < 5; i++) {
        cout << "initialized[" << i << "] = " << initialized[i] << endl;
    }
    
    // Partial initialization
    int partial[5] = {1, 2, 3};  // Remaining elements are 0
    for (int i = 0; i < 5; i++) {
        cout << "partial[" << i << "] = " << partial[i] << endl;
    }
    
    return 0;
}
```

### Exercise 2: Array Operations

Complete this array operations example with errors:

```cpp
#include <iostream>
#include <cstring>  // For C-style string functions
using namespace std;

int main() {
    // Array of characters (C-style string)
    char greeting[6] = {'H', 'e', 'l', 'l', 'o', '\0'};  // Null-terminated
    char greeting2[] = "Hello";  // Automatically null-terminated
    
    cout << "Greeting: " << greeting << endl;
    cout << "Greeting2: " << greeting2 << endl;
    
    // Error: buffer overflow
    char small_buffer[10];
    // strcpy(small_buffer, "This string is too long for the buffer");  // Buffer overflow!
    
    // Safer approach: use strncpy
    strncpy(small_buffer, "Short", sizeof(small_buffer) - 1);
    small_buffer[sizeof(small_buffer) - 1] = '\0';  // Ensure null termination
    cout << "Safe copy: " << small_buffer << endl;
    
    // Array operations
    int numbers[5] = {5, 2, 8, 1, 9};
    int size = 5;  // Need to manually track size
    
    // Find maximum
    int max = numbers[0];  // Error: what if array is empty?
    for (int i = 1; i < size; i++) {
        if (numbers[i] > max) {
            max = numbers[i];
        }
    }
    cout << "Maximum: " << max << endl;
    
    // Calculate average
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += numbers[i];
    }
    double average = static_cast<double>(sum) / size;
    cout << "Average: " << average << endl;
    
    return 0;
}
```

## C++ Strings

C++ strings provide a safer and more convenient way to work with text.

### Exercise 3: Basic String Operations

Fix the errors in this string operations code:

```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    // String creation and initialization
    string str1 = "Hello";
    string str2("World");
    string str3(5, '!');  // 5 exclamation marks
    string str4;  // Empty string
    
    cout << "str1: " << str1 << endl;
    cout << "str2: " << str2 << endl;
    cout << "str3: " << str3 << endl;
    cout << "str4: " << str4 << endl;
    
    // String concatenation
    string combined = str1 + " " + str2 + str3;
    cout << "Combined: " << combined << endl;
    
    // Error: mixing string and C-string incorrectly
    // string error = "Hello" + str2;  // Error: can't add string literal to string directly
    string correct = string("Hello") + str2;  // Correct way
    cout << "Correct: " << correct << endl;
    
    // String length
    cout << "Length of combined: " << combined.length() << endl;
    cout << "Size of combined: " << combined.size() << endl;
    
    // Accessing characters
    cout << "First character: " << combined[0] << endl;
    cout << "Last character: " << combined[combined.length() - 1] << endl;
    
    // Error: accessing out of bounds
    // cout << "Beyond end: " << combined[combined.length()] << endl;  // Undefined behavior
    
    // Safe access using at() which throws exception
    try {
        cout << "Safe access: " << combined.at(combined.length()) << endl;  // Throws exception
    } catch (const out_of_range& e) {
        cout << "Exception caught: " << e.what() << endl;
    }
    
    // Substrings
    string sub = combined.substr(0, 5);  // Start at 0, take 5 characters
    cout << "Substring: " << sub << endl;
    
    // Find substring
    size_t pos = combined.find("World");
    if (pos != string::npos) {
        cout << "'World' found at position: " << pos << endl;
    } else {
        cout << "'World' not found" << endl;
    }
    
    return 0;
}
```

### Exercise 4: String Manipulation

Complete this string manipulation example with errors:

```cpp
#include <iostream>
#include <string>
#include <algorithm>
using namespace std;

int main() {
    string text = "Hello, World!";
    
    // Convert to uppercase
    transform(text.begin(), text.end(), text.begin(), ::toupper);
    cout << "Uppercase: " << text << endl;
    
    // Convert back to lowercase
    transform(text.begin(), text.end(), text.begin(), ::tolower);
    cout << "Lowercase: " << text << endl;
    
    // Erasing characters
    string sentence = "Hello, World!";
    sentence.erase(5, 2);  // Remove 2 characters starting at position 5
    cout << "After erase: " << sentence << endl;
    
    // Inserting characters
    sentence.insert(5, " beautiful");
    cout << "After insert: " << sentence << endl;
    
    // Replacing characters
    sentence.replace(0, 5, "Hi");  // Replace first 5 characters with "Hi"
    cout << "After replace: " << sentence << endl;
    
    // Error: modifying string during iteration
    string modify_me = "Hello";
    for (char c : modify_me) {
        if (c == 'l') {
            modify_me += '!';  // Modifying the string being iterated
        }
    }
    cout << "Modified: " << modify_me << endl;
    
    // Correct way: iterate with index
    string correct_modify = "Hello";
    for (size_t i = 0; i < correct_modify.length(); i++) {
        if (correct_modify[i] == 'l') {
            correct_modify.insert(i + 1, "!");
            i++;  // Skip the inserted character
        }
    }
    cout << "Correctly modified: " << correct_modify << endl;
    
    return 0;
}
```

## Modern C++ Containers: std::array and std::vector

Modern C++ provides safer alternatives to C-style arrays.

### Exercise 5: std::array

Work with std::array in this example:

```cpp
#include <iostream>
#include <array>
#include <algorithm>
using namespace std;

int main() {
    // std::array - fixed size, type-safe
    array<int, 5> numbers = {5, 2, 8, 1, 9};
    
    // Safe access with size information
    cout << "Size: " << numbers.size() << endl;
    cout << "Max size: " << numbers.max_size() << endl;
    
    // Iteration
    cout << "Elements: ";
    for (const auto& num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    // Safe access with at() method
    try {
        cout << "Element at index 2: " << numbers.at(2) << endl;
        cout << "Element at index 10: " << numbers.at(10) << endl;  // Will throw exception
    } catch (const out_of_range& e) {
        cout << "Exception: " << e.what() << endl;
    }
    
    // Direct access with [] (no bounds checking)
    cout << "Element at index 2: " << numbers[2] << endl;
    // cout << "Element at index 10: " << numbers[10] << endl;  // Undefined behavior!
    
    // Front and back elements
    cout << "First element: " << numbers.front() << endl;
    cout << "Last element: " << numbers.back() << endl;
    
    // Data pointer (for C-style functions)
    int* raw_ptr = numbers.data();
    cout << "First element via pointer: " << *raw_ptr << endl;
    
    // Sorting
    array<int, 5> sorted_nums = numbers;  // Copy
    sort(sorted_nums.begin(), sorted_nums.end());
    cout << "Sorted: ";
    for (const auto& num : sorted_nums) {
        cout << num << " ";
    }
    cout << endl;
    
    // Error: trying to change size (std::array has fixed size)
    // numbers.resize(10);  // Error: std::array has no resize method
    
    return 0;
}
```

### Exercise 6: std::vector

Work with std::vector in this example:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    // std::vector - dynamic size
    vector<int> numbers = {5, 2, 8, 1, 9};
    
    cout << "Initial size: " << numbers.size() << endl;
    cout << "Capacity: " << numbers.capacity() << endl;
    
    // Adding elements
    numbers.push_back(15);
    numbers.push_back(3);
    cout << "After adding elements, size: " << numbers.size() << endl;
    cout << "Capacity may have increased: " << numbers.capacity() << endl;
    
    // Iteration
    cout << "Elements: ";
    for (const auto& num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    // Access elements
    cout << "Element at index 0: " << numbers[0] << endl;
    cout << "Element at index 0: " << numbers.at(0) << endl;
    
    // Error: accessing empty vector
    vector<int> empty_vec;
    // cout << empty_vec[0] << endl;  // Undefined behavior
    // cout << empty_vec.at(0) << endl;  // Throws exception
    
    // Safe check
    if (!empty_vec.empty()) {
        cout << "First element: " << empty_vec[0] << endl;
    } else {
        cout << "Vector is empty" << endl;
    }
    
    // Removing elements
    numbers.pop_back();  // Remove last element
    cout << "After pop_back, size: " << numbers.size() << endl;
    
    // Insert at specific position
    numbers.insert(numbers.begin() + 2, 99);  // Insert 99 at index 2
    cout << "After insertion: ";
    for (const auto& num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    // Erase element
    numbers.erase(numbers.begin() + 2);  // Remove element at index 2
    cout << "After erasure: ";
    for (const auto& num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    // Resizing
    numbers.resize(10, 0);  // Resize to 10, fill new elements with 0
    cout << "After resize: ";
    for (const auto& num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    cout << "New size: " << numbers.size() << endl;
    
    // Reserve capacity to avoid reallocations
    vector<int> vec;
    vec.reserve(100);  // Reserve space for 100 elements
    cout << "Reserved capacity: " << vec.capacity() << endl;
    
    return 0;
}
```

## Multidimensional Arrays

Arrays can have multiple dimensions.

### Exercise 7: Multidimensional Arrays

Complete this multidimensional array example with errors:

```cpp
#include <iostream>
#include <array>
using namespace std;

int main() {
    // 2D C-style array
    int matrix[3][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };
    
    // Print the matrix
    cout << "2D Matrix:" << endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    
    // Error: bounds checking
    // cout << matrix[3][0] << endl;  // Out of bounds!
    
    // Using std::array for 2D
    array<array<int, 4>, 3> std_matrix = {{
        {{1, 2, 3, 4}},
        {{5, 6, 7, 8}},
        {{9, 10, 11, 12}}
    }};
    
    cout << "\nStd::array 2D Matrix:" << endl;
    for (size_t i = 0; i < std_matrix.size(); i++) {
        for (size_t j = 0; j < std_matrix[i].size(); j++) {
            cout << std_matrix[i][j] << " ";
        }
        cout << endl;
    }
    
    // Using std::vector for dynamic 2D array
    vector<vector<int>> vec_matrix(3, vector<int>(4));  // 3x4 matrix
    
    // Initialize the vector matrix
    int value = 1;
    for (size_t i = 0; i < vec_matrix.size(); i++) {
        for (size_t j = 0; j < vec_matrix[i].size(); j++) {
            vec_matrix[i][j] = value++;
        }
    }
    
    cout << "\nVector 2D Matrix:" << endl;
    for (const auto& row : vec_matrix) {
        for (const auto& col : row) {
            cout << col << " ";
        }
        cout << endl;
    }
    
    // Jagged array (rows of different sizes)
    vector<vector<int>> jagged = {
        {1, 2, 3},
        {4, 5},
        {6, 7, 8, 9, 10}
    };
    
    cout << "\nJagged Array:" << endl;
    for (size_t i = 0; i < jagged.size(); i++) {
        cout << "Row " << i << ": ";
        for (size_t j = 0; j < jagged[i].size(); j++) {
            cout << jagged[i][j] << " ";
        }
        cout << endl;
    }
    
    return 0;
}
```

## String Streams

String streams allow treating strings as input/output streams.

### Exercise 8: String Stream Operations

Complete this string stream example with errors:

```cpp
#include <iostream>
#include <sstream>
#include <string>
using namespace std;

int main() {
    // Output string stream
    ostringstream oss;
    oss << "The answer is " << 42 << " and pi is approximately " << 3.14159;
    
    string result = oss.str();
    cout << "Output stream result: " << result << endl;
    
    // Input string stream
    istringstream iss("10 20 30");
    int a, b, c;
    iss >> a >> b >> c;
    cout << "Parsed values: " << a << ", " << b << ", " << c << endl;
    
    // Parsing mixed data types
    istringstream mixed_stream("John 25 85.5");
    string name;
    int age;
    double weight;
    
    mixed_stream >> name >> age >> weight;
    cout << "Name: " << name << ", Age: " << age << ", Weight: " << weight << endl;
    
    // Error: parsing failure
    istringstream bad_stream("abc def");
    int num1, num2;
    bad_stream >> num1 >> num2;  // Will fail to parse "abc" as int
    
    if (bad_stream.fail()) {
        cout << "Parsing failed!" << endl;
        bad_stream.clear();  // Clear error flags
        bad_stream.ignore(10000, '\n');  // Ignore remaining characters
    }
    
    // Line-by-line parsing
    string multiline = "Line 1\nLine 2\nLine 3";
    istringstream line_stream(multiline);
    string line;
    
    cout << "Lines:" << endl;
    while (getline(line_stream, line)) {
        cout << line << endl;
    }
    
    // Building formatted strings
    ostringstream formatted;
    formatted << "Name: " << name << ", Age: " << age;
    cout << "Formatted string: " << formatted.str() << endl;
    
    // Converting numbers to strings
    int number = 123;
    ostringstream num_stream;
    num_stream << number;
    string num_str = num_stream.str();
    cout << "Number as string: " << num_str << endl;
    
    // Converting strings to numbers
    string str_num = "456";
    istringstream str_stream(str_num);
    int converted_num;
    str_stream >> converted_num;
    cout << "String to number: " << converted_num << endl;
    
    return 0;
}
```

## Practical Examples

### Exercise 9: Array Search and Sort

Complete this array search and sorting example with errors:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
using namespace std;

// Linear search function
int linearSearch(const vector<int>& arr, int target) {
    for (size_t i = 0; i < arr.size(); i++) {
        if (arr[i] == target) {
            return static_cast<int>(i);  // Found at index i
        }
    }
    return -1;  // Not found
}

// Binary search function (requires sorted array)
int binarySearch(const vector<int>& arr, int target) {
    int left = 0;
    int right = static_cast<int>(arr.size()) - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;  // Prevent overflow
        
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;  // Not found
}

int main() {
    vector<int> numbers = {64, 34, 25, 12, 22, 11, 90};
    
    cout << "Original array: ";
    for (const auto& num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    // Linear search
    int target = 25;
    int pos = linearSearch(numbers, target);
    if (pos != -1) {
        cout << target << " found at index " << pos << endl;
    } else {
        cout << target << " not found" << endl;
    }
    
    // For binary search, we need a sorted array
    vector<int> sorted_numbers = numbers;
    sort(sorted_numbers.begin(), sorted_numbers.end());
    
    cout << "Sorted array: ";
    for (const auto& num : sorted_numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    // Binary search
    pos = binarySearch(sorted_numbers, target);
    if (pos != -1) {
        cout << target << " found at index " << pos << endl;
    } else {
        cout << target << " not found" << endl;
    }
    
    // Error: binary search on unsorted array
    pos = binarySearch(numbers, target);  // May not work correctly
    cout << "Binary search on unsorted array - result: " << pos << endl;
    
    // Sorting algorithms
    vector<int> bubble_sort_nums = {64, 34, 25, 12, 22, 11, 90};
    
    // Bubble sort implementation
    int n = bubble_sort_nums.size();
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;
        for (int j = 0; j < n - i - 1; j++) {
            if (bubble_sort_nums[j] > bubble_sort_nums[j + 1]) {
                swap(bubble_sort_nums[j], bubble_sort_nums[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;  // Array is already sorted
    }
    
    cout << "Bubble sorted: ";
    for (const auto& num : bubble_sort_nums) {
        cout << num << " ";
    }
    cout << endl;
    
    return 0;
}
```

### Exercise 10: String Processing

Complete this string processing example with errors:

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cctype>
using namespace std;

// Function to split a string by delimiter
vector<string> split(const string& str, char delimiter) {
    vector<string> tokens;
    stringstream ss(str);
    string token;
    
    while (getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}

// Function to remove whitespace from string
string trim(const string& str) {
    size_t start = str.find_first_not_of(" \t\n\r");
    if (start == string::npos) {
        return "";  // String is all whitespace
    }
    
    size_t end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}

int main() {
    string text = "  Hello,    world!   Welcome   to C++  ";
    
    cout << "Original: '" << text << "'" << endl;
    cout << "Trimmed: '" << trim(text) << "'" << endl;
    
    // Split string
    string sentence = "apple,banana,cherry,date";
    vector<string> fruits = split(sentence, ',');
    
    cout << "Split by comma:" << endl;
    for (const auto& fruit : fruits) {
        cout << "'" << trim(fruit) << "'" << endl;  // Also trim individual parts
    }
    
    // Word counting
    string paragraph = "This is a sample paragraph. This paragraph has multiple sentences.";
    stringstream ps(paragraph);
    string word;
    int word_count = 0;
    
    while (ps >> word) {  // Extract words separated by whitespace
        word_count++;
    }
    cout << "Word count: " << word_count << endl;
    
    // Character counting
    string sample = "Hello World!";
    int vowel_count = 0;
    int consonant_count = 0;
    
    for (char c : sample) {
        c = tolower(c);  // Convert to lowercase for comparison
        if (c >= 'a' && c <= 'z') {  // Only count alphabetic characters
            if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
                vowel_count++;
            } else {
                consonant_count++;
            }
        }
    }
    
    cout << "Vowels: " << vowel_count << ", Consonants: " << consonant_count << endl;
    
    // Palindrome check
    string test_palindrome = "racecar";
    string reversed = test_palindrome;
    reverse(reversed.begin(), reversed.end());
    
    if (test_palindrome == reversed) {
        cout << test_palindrome << " is a palindrome" << endl;
    } else {
        cout << test_palindrome << " is not a palindrome" << endl;
    }
    
    // Error: case-sensitive palindrome check
    string case_sensitive = "Racecar";
    string case_reversed = case_sensitive;
    reverse(case_reversed.begin(), case_reversed.end());
    
    if (case_sensitive == case_reversed) {
        cout << case_sensitive << " is a palindrome (case-sensitive)" << endl;
    } else {
        cout << case_sensitive << " is not a palindrome (case-sensitive)" << endl;
    }
    
    // Case-insensitive palindrome check
    string lower_original = case_sensitive;
    string lower_reversed = case_sensitive;
    transform(lower_original.begin(), lower_original.end(), lower_original.begin(), ::tolower);
    transform(lower_reversed.begin(), lower_reversed.end(), lower_reversed.begin(), ::tolower);
    reverse(lower_reversed.begin(), lower_reversed.end());
    
    if (lower_original == lower_reversed) {
        cout << case_sensitive << " is a palindrome (case-insensitive)" << endl;
    } else {
        cout << case_sensitive << " is not a palindrome (case-insensitive)" << endl;
    }
    
    return 0;
}
```

## Hands-On Project: Grade Book Manager

### Exercise 11: Complete Grade Book

Create a grade book manager that stores student grades in arrays/vectors:

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
using namespace std;

class Student {
public:
    string name;
    vector<double> grades;
    
    Student(const string& studentName) : name(studentName) {}
    
    void addGrade(double grade) {
        if (grade >= 0 && grade <= 100) {
            grades.push_back(grade);
        } else {
            cout << "Invalid grade: " << grade << ". Grades must be between 0 and 100." << endl;
        }
    }
    
    double getAverage() const {
        if (grades.empty()) return 0.0;
        
        // Calculate sum using accumulate
        double sum = accumulate(grades.begin(), grades.end(), 0.0);
        return sum / grades.size();
    }
    
    double getHighest() const {
        if (grades.empty()) return 0.0;
        
        // Find highest grade using max_element
        return *max_element(grades.begin(), grades.end());
    }
    
    double getLowest() const {
        if (grades.empty()) return 0.0;
        
        // Find lowest grade using min_element
        return *min_element(grades.begin(), grades.end());
    }
    
    void displayGrades() const {
        cout << name << "'s grades: ";
        for (size_t i = 0; i < grades.size(); i++) {
            cout << grades[i];
            if (i < grades.size() - 1) cout << ", ";
        }
        cout << endl;
        
        cout << "Average: " << getAverage() << endl;
        cout << "Highest: " << getHighest() << endl;
        cout << "Lowest: " << getLowest() << endl;
    }
};

int main() {
    vector<Student> students;
    
    // Add students
    students.emplace_back("Alice");
    students.emplace_back("Bob");
    students.emplace_back("Charlie");
    
    // Add grades for Alice
    students[0].addGrade(85.5);
    students[0].addGrade(92.0);
    students[0].addGrade(78.5);
    students[0].addGrade(96.0);
    
    // Add grades for Bob
    students[1].addGrade(76.0);
    students[1].addGrade(81.5);
    students[1].addGrade(89.0);
    students[1].addGrade(72.5);
    
    // Add grades for Charlie
    students[2].addGrade(94.5);
    students[2].addGrade(91.0);
    students[2].addGrade(98.5);
    students[2].addGrade(93.0);
    
    // Display all student information
    for (const auto& student : students) {
        student.displayGrades();
        cout << "---" << endl;
    }
    
    // Class statistics
    vector<double> classAverages;
    for (const auto& student : students) {
        classAverages.push_back(student.getAverage());
    }
    
    double classAverage = accumulate(classAverages.begin(), classAverages.end(), 0.0) / classAverages.size();
    double highestAvg = *max_element(classAverages.begin(), classAverages.end());
    double lowestAvg = *min_element(classAverages.begin(), classAverages.end());
    
    cout << "\nClass Statistics:" << endl;
    cout << "Class Average: " << classAverage << endl;
    cout << "Highest Student Average: " << highestAvg << endl;
    cout << "Lowest Student Average: " << lowestAvg << endl;
    
    // Error: potential division by zero if no students
    if (students.empty()) {
        cout << "No students in the grade book!" << endl;
    }
    
    return 0;
}
```

## Best Practices

1. Prefer std::vector over C-style arrays for dynamic sizing
2. Use std::array for fixed-size collections
3. Always check bounds when accessing arrays
4. Initialize arrays to avoid undefined behavior
5. Use string streams for parsing and formatting
6. Prefer algorithms from <algorithm> over manual loops
7. Use const references when passing large containers
8. Check for empty containers before accessing elements

## Summary

In this chapter, you learned:
- C-style arrays and their limitations
- C++ strings and string operations
- Modern containers: std::array and std::vector
- Multidimensional arrays
- String streams for parsing and formatting
- Practical applications of arrays and strings

## Key Takeaways

- C-style arrays lack bounds checking and size information
- C++ strings are safer and more feature-rich than C-strings
- std::vector provides dynamic sizing with safety features
- std::array provides fixed-size arrays with safety features
- String streams are powerful for parsing and formatting
- Algorithms from <algorithm> make code more expressive

## Common Mistakes to Avoid

1. Buffer overflows with C-style arrays and strings
2. Using uninitialized array elements
3. Forgetting to null-terminate C-strings
4. Mixing signed and unsigned indices incorrectly
5. Not checking for empty containers before access
6. Using raw arrays when std::vector or std::array would be safer
7. Not handling string parsing errors appropriately

## Next Steps

Now that you understand arrays and strings, you're ready to learn about pointers and references in Chapter 6.