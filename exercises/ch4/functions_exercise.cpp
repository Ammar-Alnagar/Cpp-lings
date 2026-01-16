/*
 * Chapter 4 Exercise: Functions
 * 
 * Complete the program by implementing the missing functions.
 * The program should calculate the factorial of a number and check if a number is prime.
 */

#include <iostream>
#include <cmath>

// TODO: Implement the factorial function
// This function should calculate the factorial of n (n! = n * (n-1) * ... * 1)
// For n = 0, factorial should return 1
// For negative numbers, return -1 (as error indicator)
long long factorial(int n) {
    // TODO: Implement factorial calculation
}

// TODO: Implement the isPrime function
// This function should return true if n is a prime number, false otherwise
// A prime number is greater than 1 and divisible only by 1 and itself
bool isPrime(int n) {
    // TODO: Implement prime check
}

// TODO: Implement a function that calculates the nth Fibonacci number
// Use an efficient iterative approach
int fibonacci(int n) {
    // TODO: Implement fibonacci calculation
}

int main() {
    int number;
    
    std::cout << "Enter a number to calculate its factorial: ";
    std::cin >> number;
    
    long long fact = factorial(number);
    if (fact == -1) {
        std::cout << "Factorial is not defined for negative numbers." << std::endl;
    } else {
        std::cout << "Factorial of " << number << " is: " << fact << std::endl;
    }
    
    std::cout << "Enter a number to check if it's prime: ";
    std::cin >> number;
    
    if (isPrime(number)) {
        std::cout << number << " is a prime number." << std::endl;
    } else {
        std::cout << number << " is not a prime number." << std::endl;
    }
    
    std::cout << "Enter a position to get the Fibonacci number: ";
    std::cin >> number;
    
    if (number >= 0) {
        std::cout << "Fibonacci number at position " << number << " is: " 
                  << fibonacci(number) << std::endl;
    } else {
        std::cout << "Fibonacci is not defined for negative positions." << std::endl;
    }
    
    return 0;
}