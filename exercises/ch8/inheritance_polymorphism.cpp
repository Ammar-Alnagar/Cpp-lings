/*
 * Chapter 8 Exercise: Advanced OOP Concepts
 * 
 * Complete the inheritance hierarchy with Shape as base class and Circle, Rectangle as derived classes.
 * Implement polymorphism with virtual functions.
 */

#include <iostream>
#include <vector>
#include <memory>

// TODO: Define a base class Shape with:
// - Pure virtual functions: calculateArea(), calculatePerimeter(), display()
// - Virtual destructor
// - Protected member variable for color (string)

class Shape {
    // TODO: Add protected member for color
    
public:
    // TODO: Constructor that takes color
    // TODO: Pure virtual functions for area, perimeter, and display
    // TODO: Virtual destructor
};

// TODO: Define Circle class that inherits from Shape
// - Should have radius as private member
// - Implement all pure virtual functions
// - Constructor should take radius and color

class Circle {
    // TODO: Add private member for radius
    
public:
    // TODO: Constructor, implement virtual functions
};

// TODO: Define Rectangle class that inherits from Shape
// - Should have width and height as private members
// - Implement all pure virtual functions
// - Constructor should take width, height and color

class Rectangle {
    // TODO: Add private members for width and height
    
public:
    // TODO: Constructor, implement virtual functions
};

// TODO: Define Triangle class that inherits from Shape
// - Should have three sides as private members
// - Implement all virtual functions
// - Constructor should take three sides and color

class Triangle {
    // TODO: Add private members for three sides
    
public:
    // TODO: Constructor, implement virtual functions
};

int main() {
    // TODO: Create vector of unique_ptr to Shape objects
    // Add different shapes (Circle, Rectangle, Triangle) to the vector
    
    // TODO: Use polymorphism to call virtual functions
    // Loop through the vector and call display(), calculateArea(), calculatePerimeter() for each shape
    
    // TODO: Demonstrate dynamic casting
    // Try to cast shapes back to their derived types and access specific properties
    
    return 0;
}