/*
 * Chapter 6 Exercise: Pointers and References
 * 
 * Complete the program that demonstrates pointer operations and dynamic memory allocation.
 * The program should create a dynamic array, manipulate it using pointers, and properly clean up.
 */

#include <iostream>

// TODO: Implement a function that uses pointer arithmetic to find the maximum value in an array
// Parameters: pointer to the array, size of the array
// Return: pointer to the maximum element
double* findMax(double* arr, int size) {
    // TODO: Implement using pointer arithmetic
    // Don't use array indexing (arr[i]), use pointer arithmetic (*(ptr + offset))
}

// TODO: Implement a function that swaps two values using pointers
void swapWithPointers(double* a, double* b) {
    // TODO: Implement swap using pointers
}

// TODO: Implement a function that swaps two values using references
void swapWithReferences(double& a, double& b) {
    // TODO: Implement swap using references
}

int main() {
    int size;
    std::cout << "Enter the size of the array: ";
    std::cin >> size;
    
    if (size <= 0) {
        std::cout << "Invalid size!" << std::endl;
        return 1;
    }
    
    // TODO: Dynamically allocate an array of doubles with the given size
    // Use 'new' operator
    
    // TODO: Input values into the array using pointer arithmetic
    // Don't use array indexing, use pointer arithmetic
    
    // TODO: Print the original array using pointer arithmetic
    
    // TODO: Find and print the maximum value using findMax function
    
    // TODO: Demonstrate swapping using both pointer and reference functions
    
    // TODO: Print the array after swapping operations
    
    // TODO: Properly deallocate the dynamically allocated memory
    // Use 'delete[]' operator
    
    std::cout << "Memory cleaned up successfully!" << std::endl;
    
    return 0;
}