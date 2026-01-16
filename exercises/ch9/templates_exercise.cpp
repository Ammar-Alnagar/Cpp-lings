/*
 * Chapter 9 Exercise: Templates and Generic Programming
 * 
 * Complete the generic Stack class template and implement a generic function to find maximum value.
 */

#include <iostream>
#include <vector>
#include <stdexcept>

template<typename T>
class Stack {
private:
    // TODO: Add private member to store elements (use std::vector<T>)
    
public:
    // TODO: Implement default constructor
    
    // TODO: Implement push function to add an element to the stack
    void push(const T& element) {
        // TODO: Add implementation
    }
    
    // TODO: Implement pop function to remove and return the top element
    T pop() {
        // TODO: Add implementation with error checking
        // Throw std::out_of_range if stack is empty
    }
    
    // TODO: Implement top function to return the top element without removing it
    T top() const {
        // TODO: Add implementation with error checking
        // Throw std::out_of_range if stack is empty
    }
    
    // TODO: Implement empty function to check if stack is empty
    bool empty() const {
        // TODO: Add implementation
    }
    
    // TODO: Implement size function to return the number of elements
    size_t size() const {
        // TODO: Add implementation
    }
    
    // TODO: Implement clear function to remove all elements
    void clear() {
        // TODO: Add implementation
    }
};

// TODO: Implement a generic function template to find the maximum of two values
template<typename T>
T max(const T& a, const T& b) {
    // TODO: Return the larger of a and b
}

// TODO: Implement a generic function template to find the maximum value in a container
template<typename Container>
typename Container::value_type maxValue(const Container& container) {
    // TODO: Return the maximum value in the container
    // Assume container is not empty
    // Use iterators to traverse the container
}

// TODO: Implement a generic function template that applies a function to each element
template<typename Container, typename Function>
void forEach(Container& container, Function func) {
    // TODO: Apply func to each element in the container
    // Use iterators to traverse the container
}

int main() {
    // TODO: Test the Stack template with different types (int, double, string)
    Stack<int> intStack;
    Stack<double> doubleStack;
    Stack<std::string> stringStack;
    
    // TODO: Push some values onto each stack
    // Pop and print values to verify functionality
    
    // TODO: Test the max function template with different types
    std::cout << "Max of 5 and 10: " << max(5, 10) << std::endl;
    std::cout << "Max of 3.14 and 2.71: " << max(3.14, 2.71) << std::endl;
    
    // TODO: Test the maxValue function with vectors
    std::vector<int> numbers = {5, 2, 8, 1, 9, 3};
    std::cout << "Max value in vector: " << maxValue(numbers) << std::endl;
    
    // TODO: Test the forEach function to double all values in a vector
    std::vector<int> values = {1, 2, 3, 4, 5};
    std::cout << "Original values: ";
    for (int v : values) std::cout << v << " ";
    std::cout << std::endl;
    
    // TODO: Use forEach to multiply each value by 2
    // Print the modified vector
    
    return 0;
}