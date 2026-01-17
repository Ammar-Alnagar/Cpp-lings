/*
 * Chapter 2 Exercise: Data Types and Operators
 *
 * Complete the program by implementing the missing functions and fixing errors.
 * The program should demonstrate different data types and operators.
 */

#include <cctype>
#include <iostream>
#include <limits>
#include <print>

int main() {
  // TODO: Declare variables of different types:
  // - an int named 'integerValue' initialized to 42
  // - a double named 'doubleValue' initialized to 3.14159
  // - a char named 'charValue' initialized to 'A'
  // - a bool named 'boolValue' initialized to true
  int integerValue = 42;
  double doubleValue = 3.14159;
  char charValue = 'A';
  bool boolValue = true;

  // TODO: Perform the following operations and store results:
  // - Add integer and double values, store in 'sum'
  // - Multiply integer by 2, store in 'product'
  // - Check if charValue is uppercase (hint: use isupper from <cctype>), store
  // in 'isUpper'
  auto sum = integerValue + doubleValue;
  auto product = integerValue * 2;
  auto isUpper = std::isupper(charValue);

  // TODO: Print all values with appropriate labels
  std::println("Sum: {}", sum);
  std::println("Product: {}", product);
  std::println("Is uppercase: {}", isUpper);
  std::println("bool Value: {}", boolValue);

  // TODO: Show the size of each data type using sizeof operator
  std::println("Size of int: {} bytes", sizeof(integerValue));
  std::println("Size of double: {} bytes", sizeof(doubleValue));
  std::println("Size of char: {} bytes", sizeof(charValue));
  std::println("Size of bool: {} bytes", sizeof(boolValue));

  return 0;
}