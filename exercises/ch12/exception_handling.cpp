/*
 * Chapter 12 Exercise: Exception Handling
 * 
 * Complete the program that demonstrates proper exception handling techniques.
 * The program should handle various error conditions gracefully.
 */

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

class SafeArray {
private:
    std::unique_ptr<int[]> data;
    size_t size_;
    
public:
    explicit SafeArray(size_t size) : size_(size) {
        if (size == 0) {
            throw std::invalid_argument("Array size cannot be zero");
        }
        data = std::make_unique<int[]>(size);
    }
    
    // TODO: Implement at() function that throws std::out_of_range for invalid indices
    int& at(size_t index) {
        // TODO: Check bounds and throw exception if out of range
        // Otherwise return reference to element
    }
    
    const int& at(size_t index) const {
        // TODO: Implement const version
    }
    
    // TODO: Implement operator[] that does not perform bounds checking
    int& operator[](size_t index) {
        // TODO: Return element without bounds checking
    }
    
    const int& operator[](size_t index) const {
        // TODO: Implement const version
    }
    
    size_t size() const { return size_; }
};

// TODO: Create a custom exception class for division by zero
class DivisionByZeroException : public std::exception {
    // TODO: Implement the class with appropriate members and methods
};

// TODO: Implement a function that divides two numbers and throws an exception if dividing by zero
double safeDivide(double numerator, double denominator) {
    // TODO: Check for division by zero and throw appropriate exception
    // Otherwise return the result
}

// TODO: Implement a function that calculates factorial and throws exception for negative inputs
unsigned long long factorial(int n) {
    // TODO: Check for negative input and throw exception
    // Otherwise calculate and return factorial
}

class BankAccount {
private:
    std::string accountNumber;
    double balance;
    
public:
    BankAccount(const std::string& number, double initialBalance) 
        : accountNumber(number), balance(initialBalance) {
        if (initialBalance < 0) {
            throw std::invalid_argument("Initial balance cannot be negative");
        }
    }
    
    // TODO: Implement deposit function that throws exception for negative amounts
    void deposit(double amount) {
        // TODO: Check for negative amount and throw exception
        // Otherwise add to balance
    }
    
    // TODO: Implement withdraw function that throws exception for insufficient funds
    void withdraw(double amount) {
        // TODO: Check for negative amount and insufficient funds
        // Throw appropriate exceptions
        // Otherwise deduct from balance
    }
    
    double getBalance() const { return balance; }
    const std::string& getAccountNumber() const { return accountNumber; }
};

int main() {
    std::cout << "=== Exception Handling Exercise ===" << std::endl;
    
    // TODO: Test SafeArray with proper exception handling
    try {
        SafeArray arr(5);
        
        // TODO: Fill array with values
        for (size_t i = 0; i < arr.size(); ++i) {
            arr[i] = static_cast<int>(i * i);
        }
        
        // TODO: Print array contents
        std::cout << "Array contents: ";
        for (size_t i = 0; i < arr.size(); ++i) {
            std::cout << arr[i] << " ";
        }
        std::cout << std::endl;
        
        // TODO: Try to access invalid index (should throw exception)
        std::cout << "Attempting to access invalid index..." << std::endl;
        int value = arr.at(10);  // This should throw
        std::cout << "This line should not be printed: " << value << std::endl;
        
    } catch (const std::out_of_range& e) {
        std::cout << "Caught out_of_range: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }
    
    // TODO: Test safeDivide function
    std::vector<std::pair<double, double>> divisions = {
        {10.0, 2.0}, {5.0, 0.0}, {15.0, 3.0}, {-10.0, 2.0}
    };
    
    for (const auto& div : divisions) {
        try {
            double result = safeDivide(div.first, div.second);
            std::cout << div.first << " / " << div.second << " = " << result << std::endl;
        } catch (const DivisionByZeroException& e) {
            std::cout << "Division by zero error: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
    }
    
    // TODO: Test factorial function
    std::vector<int> factorials = {5, 0, -3, 10};
    
    for (int n : factorials) {
        try {
            unsigned long long result = factorial(n);
            std::cout << "Factorial of " << n << " = " << result << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Error calculating factorial of " << n << ": " << e.what() << std::endl;
        }
    }
    
    // TODO: Test BankAccount with exception handling
    try {
        BankAccount account("ACC-001", 1000.0);
        std::cout << "Created account " << account.getAccountNumber() 
                  << " with balance $" << account.getBalance() << std::endl;
        
        account.deposit(500.0);
        std::cout << "Deposited $500, new balance: $" << account.getBalance() << std::endl;
        
        account.withdraw(200.0);
        std::cout << "Withdrew $200, new balance: $" << account.getBalance() << std::endl;
        
        // TODO: Try to withdraw more than available balance
        std::cout << "Attempting to withdraw $2000..." << std::endl;
        account.withdraw(2000.0);
        
    } catch (const std::exception& e) {
        std::cout << "Banking error: " << e.what() << std::endl;
    }
    
    // TODO: Demonstrate exception safety with RAII
    std::cout << "\nTesting exception safety with RAII..." << std::endl;
    
    try {
        auto ptr = std::make_unique<SafeArray>(3);
        std::cout << "Created SafeArray with RAII protection" << std::endl;
        
        // TODO: Simulate an exception
        if (true) {  // Simulate error condition
            throw std::runtime_error("Simulated error");
        }
        
        std::cout << "This line won't be executed" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Caught: " << e.what() << std::endl;
        std::cout << "But SafeArray was properly cleaned up!" << std::endl;
    }
    
    // TODO: Demonstrate nested exception handling
    try {
        try {
            // TODO: Throw an exception inside this inner try block
            throw std::logic_error("Inner exception occurred");
        } catch (const std::logic_error& e) {
            std::cout << "Caught in inner handler: " << e.what() << std::endl;
            // TODO: Re-throw or throw a different exception
            throw std::runtime_error("Outer exception wrapping inner");
        }
    } catch (const std::runtime_error& e) {
        std::cout << "Caught in outer handler: " << e.what() << std::endl;
    }
    
    // TODO: Demonstrate noexcept specifications
    auto noThrowFunc = []() noexcept {
        return 42;
    };
    
    auto mayThrowFunc = []() {
        if (true) throw std::runtime_error("May throw");
        return 0;
    };
    
    std::cout << "noexcept function result: " << noThrowFunc() << std::endl;
    std::cout << "noexcept property of noThrowFunc: " << noexcept(noThrowFunc()) << std::endl;
    std::cout << "noexcept property of mayThrowFunc: " << noexcept(mayThrowFunc()) << std::endl;
    
    std::cout << "\nException handling exercise completed!" << std::endl;
    
    return 0;
}