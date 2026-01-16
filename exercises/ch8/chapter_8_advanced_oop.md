# Chapter 8: Advanced Object-Oriented Programming Concepts

## Overview

This chapter covers advanced OOP concepts in C++, including inheritance, polymorphism, abstract classes, virtual functions, and multiple inheritance. These concepts are fundamental to creating flexible and extensible object-oriented designs.

## Learning Objectives

By the end of this chapter, you will:
- Understand inheritance and how to create derived classes
- Learn about access control in inheritance
- Master virtual functions and runtime polymorphism
- Understand abstract classes and pure virtual functions
- Learn about multiple inheritance and virtual inheritance
- Understand the virtual destructor pattern
- Learn about function overriding and hiding
- Understand the Liskov Substitution Principle

## Inheritance

Inheritance allows a class to inherit properties and methods from another class.

### Exercise 1: Basic Inheritance

The following code has errors. Find and fix them:

```cpp
#include <iostream>
#include <string>
using namespace std;

// Base class
class Vehicle {
protected:  // Changed from private to protected
    string brand;
    int year;
    
public:
    Vehicle(const string& b, int y) : brand(b), year(y) {
        cout << "Vehicle constructor called" << endl;
    }
    
    void displayInfo() const {
        cout << "Brand: " << brand << ", Year: " << year << endl;
    }
    
    string getBrand() const { return brand; }
    int getYear() const { return year; }
    
    // Virtual destructor for proper cleanup
    virtual ~Vehicle() {
        cout << "Vehicle destructor called" << endl;
    }
};

// Derived class
class Car : public Vehicle {  // Inherit publicly
private:
    int doors;
    
public:
    // Constructor that calls base class constructor
    Car(const string& b, int y, int d) : Vehicle(b, y), doors(d) {  // Call base constructor
        cout << "Car constructor called" << endl;
    }
    
    // Override the display function
    void displayInfo() const override {  // Error: missing 'virtual' in base class
        cout << "Car - Brand: " << brand << ", Year: " << year 
             << ", Doors: " << doors << endl;
    }
    
    int getDoors() const { return doors; }
    
    ~Car() {
        cout << "Car destructor called" << endl;
    }
};

int main() {
    Car myCar("Toyota", 2022, 4);
    myCar.displayInfo();
    
    // Base class pointer pointing to derived class object
    Vehicle* v = &myCar;
    v->displayInfo();  // Which function gets called?
    
    return 0;
}
```

### Exercise 2: Fixed Inheritance with Virtual Functions

Fix the virtual function issues in the previous example:

```cpp
#include <iostream>
#include <string>
using namespace std;

class Vehicle {
protected:
    string brand;
    int year;
    
public:
    Vehicle(const string& b, int y) : brand(b), year(y) {
        cout << "Vehicle constructor called" << endl;
    }
    
    // Make this function virtual for polymorphism
    virtual void displayInfo() const {
        cout << "Vehicle - Brand: " << brand << ", Year: " << year << endl;
    }
    
    string getBrand() const { return brand; }
    int getYear() const { return year; }
    
    // Virtual destructor is essential for proper cleanup
    virtual ~Vehicle() {
        cout << "Vehicle destructor called" << endl;
    }
};

class Car : public Vehicle {
private:
    int doors;
    
public:
    Car(const string& b, int y, int d) : Vehicle(b, y), doors(d) {
        cout << "Car constructor called" << endl;
    }
    
    // Override the virtual function
    void displayInfo() const override {
        cout << "Car - Brand: " << brand << ", Year: " << year 
             << ", Doors: " << doors << endl;
    }
    
    int getDoors() const { return doors; }
    
    ~Car() {
        cout << "Car destructor called" << endl;
    }
};

class Motorcycle : public Vehicle {
private:
    bool hasSidecar;
    
public:
    Motorcycle(const string& b, int y, bool sidecar) 
        : Vehicle(b, y), hasSidecar(sidecar) {
        cout << "Motorcycle constructor called" << endl;
    }
    
    void displayInfo() const override {
        cout << "Motorcycle - Brand: " << brand << ", Year: " << year 
             << ", Sidecar: " << (hasSidecar ? "Yes" : "No") << endl;
    }
    
    bool getHasSidecar() const { return hasSidecar; }
    
    ~Motorcycle() {
        cout << "Motorcycle destructor called" << endl;
    }
};

int main() {
    cout << "Creating objects:" << endl;
    
    Car car("Honda", 2023, 4);
    Motorcycle bike("Harley-Davidson", 2022, false);
    
    cout << "\nDirect calls:" << endl;
    car.displayInfo();
    bike.displayInfo();
    
    cout << "\nUsing base class pointers (polymorphism):" << endl;
    Vehicle* vehicles[] = {&car, &bike};
    
    for (int i = 0; i < 2; i++) {
        vehicles[i]->displayInfo();  // Calls the appropriate overridden function
    }
    
    cout << "\nDemonstrating virtual destructor:" << endl;
    Vehicle* dynamicCar = new Car("Ford", 2024, 2);
    dynamicCar->displayInfo();
    delete dynamicCar;  // Calls both Car and Vehicle destructors
    
    return 0;
}
```

## Access Control in Inheritance

Different inheritance access specifiers affect how base class members are inherited.

### Exercise 3: Inheritance Access Specifiers

Complete this example showing different inheritance access levels:

```cpp
#include <iostream>
#include <string>
using namespace std;

class Base {
private:
    int privateMember;
protected:
    int protectedMember;
public:
    int publicMember;
    
    Base(int priv, int prot, int pub) 
        : privateMember(priv), protectedMember(prot), publicMember(pub) {}
    
    void display() const {
        cout << "Base - Private: " << privateMember 
             << ", Protected: " << protectedMember 
             << ", Public: " << publicMember << endl;
    }
};

// Public inheritance - most common
class PublicDerived : public Base {
public:
    PublicDerived(int priv, int prot, int pub) : Base(priv, prot, pub) {}
    
    void accessMembers() {
        // Can access protected and public members from base
        cout << "PublicDerived accessing - Protected: " << protectedMember 
             << ", Public: " << publicMember << endl;
        // Cannot access privateMember - it's private in base
        // cout << privateMember;  // Error
    }
    
    void display() const {
        Base::display();
        cout << "PublicDerived - Protected: " << protectedMember 
             << ", Public: " << publicMember << endl;
    }
};

// Protected inheritance
class ProtectedDerived : protected Base {
public:
    ProtectedDerived(int priv, int prot, int pub) : Base(priv, prot, pub) {}
    
    void accessMembers() {
        // Can access protected and public members from base
        cout << "ProtectedDerived accessing - Protected: " << protectedMember 
             << ", Public: " << publicMember << endl;
    }
    
    void display() const {
        Base::display();
    }
};

// Private inheritance - default if no access specifier specified
class PrivateDerived : private Base {
public:
    PrivateDerived(int priv, int prot, int pub) : Base(priv, prot, pub) {}
    
    void accessMembers() {
        // Can access protected and public members from base
        cout << "PrivateDerived accessing - Protected: " << protectedMember 
             << ", Public: " << publicMember << endl;
    }
    
    // Need to make base functions public if needed
    using Base::publicMember;  // Make public member accessible
    void display() const { Base::display(); }
};

int main() {
    cout << "=== Public Inheritance ===" << endl;
    PublicDerived pubDer(1, 2, 3);
    pubDer.accessMembers();
    pubDer.display();
    cout << "Public member accessible: " << pubDer.publicMember << endl;
    // pubDer.protectedMember;  // Error: protected in derived class
    
    cout << "\n=== Protected Inheritance ===" << endl;
    ProtectedDerived protDer(4, 5, 6);
    protDer.accessMembers();
    protDer.display();
    // protDer.publicMember;  // Error: now protected in derived class
    
    cout << "\n=== Private Inheritance ===" << endl;
    PrivateDerived privDer(7, 8, 9);
    privDer.accessMembers();
    privDer.display();
    cout << "Public member accessible: " << privDer.publicMember << endl;
    
    return 0;
}
```

## Virtual Functions and Runtime Polymorphism

Virtual functions enable runtime polymorphism.

### Exercise 4: Virtual Functions and Polymorphism

Complete this polymorphism example with errors:

```cpp
#include <iostream>
#include <string>
#include <vector>
using namespace std;

class Shape {
protected:
    string color;
    
public:
    Shape(const string& c) : color(c) {}
    
    // Virtual function for polymorphism
    virtual double getArea() const {
        cout << "Shape::getArea() called" << endl;
        return 0.0;
    }
    
    virtual void draw() const {
        cout << "Drawing a shape with color " << color << endl;
    }
    
    // Pure virtual function makes this an abstract class
    virtual string getType() const = 0;  // Pure virtual function
    
    virtual ~Shape() {
        cout << "Shape destructor called" << endl;
    }
};

class Circle : public Shape {
private:
    double radius;
    
public:
    Circle(const string& c, double r) : Shape(c), radius(r) {}
    
    // Override virtual functions
    double getArea() const override {
        return 3.14159 * radius * radius;
    }
    
    void draw() const override {
        cout << "Drawing a circle with radius " << radius 
             << " and color " << color << endl;
    }
    
    string getType() const override {
        return "Circle";
    }
    
    double getRadius() const { return radius; }
};

class Rectangle : public Shape {
private:
    double width, height;
    
public:
    Rectangle(const string& c, double w, double h) : Shape(c), width(w), height(h) {}
    
    double getArea() const override {
        return width * height;
    }
    
    void draw() const override {
        cout << "Drawing a rectangle " << width << "x" << height 
             << " with color " << color << endl;
    }
    
    string getType() const override {
        return "Rectangle";
    }
    
    double getWidth() const { return width; }
    double getHeight() const { return height; }
};

class Triangle : public Shape {
private:
    double base, height;
    
public:
    Triangle(const string& c, double b, double h) : Shape(c), base(b), height(h) {}
    
    double getArea() const override {
        return 0.5 * base * height;
    }
    
    void draw() const override {
        cout << "Drawing a triangle with base " << base 
             << " and height " << height << " with color " << color << endl;
    }
    
    string getType() const override {
        return "Triangle";
    }
    
    double getBase() const { return base; }
    double getHeight() const { return height; }
};

int main() {
    // Cannot create abstract class object
    // Shape s("red");  // Error: cannot instantiate abstract class
    
    // Create concrete objects
    Circle circle("Red", 5.0);
    Rectangle rect("Blue", 4.0, 6.0);
    Triangle tri("Green", 3.0, 8.0);
    
    cout << "=== Direct calls ===" << endl;
    circle.draw();
    cout << "Area: " << circle.getArea() << endl;
    
    rect.draw();
    cout << "Area: " << rect.getArea() << endl;
    
    tri.draw();
    cout << "Area: " << tri.getArea() << endl;
    
    cout << "\n=== Polymorphism with pointers ===" << endl;
    vector<Shape*> shapes = {&circle, &rect, &tri};
    
    for (Shape* shape : shapes) {
        shape->draw();
        cout << "Area: " << shape->getArea() << endl;
        cout << "Type: " << shape->getType() << endl;
        cout << "---" << endl;
    }
    
    cout << "\n=== Polymorphism with dynamic allocation ===" << endl;
    vector<Shape*> dynamicShapes;
    dynamicShapes.push_back(new Circle("Yellow", 3.0));
    dynamicShapes.push_back(new Rectangle("Purple", 2.0, 7.0));
    dynamicShapes.push_back(new Triangle("Orange", 6.0, 4.0));
    
    for (Shape* shape : dynamicShapes) {
        shape->draw();
        cout << "Area: " << shape->getArea() << endl;
        cout << "Type: " << shape->getType() << endl;
        cout << "---" << endl;
    }
    
    // Clean up dynamic memory
    for (Shape* shape : dynamicShapes) {
        delete shape;  // Virtual destructor ensures proper cleanup
    }
    
    return 0;
}
```

## Abstract Classes and Pure Virtual Functions

Abstract classes contain at least one pure virtual function and cannot be instantiated.

### Exercise 5: Abstract Classes

Complete this abstract class example:

```cpp
#include <iostream>
#include <string>
#include <memory>
using namespace std;

// Abstract base class
class Animal {
protected:
    string name;
    int age;
    
public:
    Animal(const string& n, int a) : name(n), age(a) {}
    
    // Regular virtual function
    virtual void eat() const {
        cout << name << " is eating." << endl;
    }
    
    // Pure virtual functions - makes class abstract
    virtual void makeSound() const = 0;
    virtual void move() const = 0;
    virtual string getType() const = 0;
    
    // Virtual destructor
    virtual ~Animal() {
        cout << name << " the " << getType() << " is being destroyed." << endl;
    }
    
    string getName() const { return name; }
    int getAge() const { return age; }
};

class Dog : public Animal {
public:
    Dog(const string& n, int a) : Animal(n, a) {}
    
    void makeSound() const override {
        cout << name << " says Woof!" << endl;
    }
    
    void move() const override {
        cout << name << " runs around." << endl;
    }
    
    string getType() const override {
        return "Dog";
    }
    
    void fetch() const {
        cout << name << " fetches the ball!" << endl;
    }
};

class Cat : public Animal {
public:
    Cat(const string& n, int a) : Animal(n, a) {}
    
    void makeSound() const override {
        cout << name << " says Meow!" << endl;
    }
    
    void move() const override {
        cout << name << " prowls quietly." << endl;
    }
    
    string getType() const override {
        return "Cat";
    }
    
    void purr() const {
        cout << name << " purrs contentedly." << endl;
    }
};

class Bird : public Animal {
public:
    Bird(const string& n, int a) : Animal(n, a) {}
    
    void makeSound() const override {
        cout << name << " chirps melodiously!" << endl;
    }
    
    void move() const override {
        cout << name << " flies through the air." << endl;
    }
    
    string getType() const override {
        return "Bird";
    }
    
    void fly() const {
        cout << name << " soars high in the sky." << endl;
    }
};

int main() {
    cout << "=== Creating animals ===" << endl;
    
    // Cannot create abstract class object
    // Animal generic("Generic", 1);  // Error: abstract class
    
    Dog dog("Buddy", 3);
    Cat cat("Whiskers", 2);
    Bird bird("Tweety", 1);
    
    cout << "\n=== Direct calls ===" << endl;
    dog.makeSound();
    dog.move();
    dog.fetch();
    
    cat.makeSound();
    cat.move();
    cat.purr();
    
    bird.makeSound();
    bird.move();
    bird.fly();
    
    cout << "\n=== Polymorphic behavior ===" << endl;
    Animal* animals[] = {&dog, &cat, &bird};
    
    for (int i = 0; i < 3; i++) {
        animals[i]->eat();        // Calls base class implementation
        animals[i]->makeSound();  // Calls derived class implementation
        animals[i]->move();       // Calls derived class implementation
        cout << "Type: " << animals[i]->getType() << endl;
        cout << "---" << endl;
    }
    
    cout << "\n=== Using smart pointers ===" << endl;
    vector<unique_ptr<Animal>> smartAnimals;
    smartAnimals.push_back(make_unique<Dog>("Max", 4));
    smartAnimals.push_back(make_unique<Cat>("Fluffy", 3));
    smartAnimals.push_back(make_unique<Bird>("Robin", 2));
    
    for (const auto& animal : smartAnimals) {
        animal->makeSound();
        animal->move();
        cout << "Name: " << animal->getName() << endl;
        cout << "---" << endl;
    }
    
    // Smart pointers automatically handle cleanup
    return 0;
}
```

## Virtual Destructors

Virtual destructors ensure proper cleanup in inheritance hierarchies.

### Exercise 6: Virtual Destructors

Complete this example showing the importance of virtual destructors:

```cpp
#include <iostream>
#include <string>
using namespace std;

// Without virtual destructor - problematic
class BaseWithoutVirtualDtor {
protected:
    string name;
    
public:
    BaseWithoutVirtualDtor(const string& n) : name(n) {
        cout << "BaseWithoutVirtualDtor constructor called for " << name << endl;
    }
    
    // Non-virtual destructor - dangerous!
    ~BaseWithoutVirtualDtor() {
        cout << "BaseWithoutVirtualDtor destructor called for " << name << endl;
    }
    
    virtual void display() const {
        cout << "BaseWithoutVirtualDtor: " << name << endl;
    }
};

class DerivedWithoutVirtualDtor : public BaseWithoutVirtualDtor {
private:
    int value;
    
public:
    DerivedWithoutVirtualDtor(const string& n, int v) 
        : BaseWithoutVirtualDtor(n), value(v) {
        cout << "DerivedWithoutVirtualDtor constructor called with value " << value << endl;
    }
    
    ~DerivedWithoutVirtualDtor() {
        cout << "DerivedWithoutVirtualDtor destructor called for " << name 
             << " with value " << value << endl;
    }
    
    void display() const override {
        cout << "DerivedWithoutVirtualDtor: " << name << " with value " << value << endl;
    }
};

// With virtual destructor - correct
class BaseWithVirtualDtor {
protected:
    string name;
    
public:
    BaseWithVirtualDtor(const string& n) : name(n) {
        cout << "BaseWithVirtualDtor constructor called for " << name << endl;
    }
    
    // Virtual destructor - correct!
    virtual ~BaseWithVirtualDtor() {
        cout << "BaseWithVirtualDtor destructor called for " << name << endl;
    }
    
    virtual void display() const {
        cout << "BaseWithVirtualDtor: " << name << endl;
    }
};

class DerivedWithVirtualDtor : public BaseWithVirtualDtor {
private:
    int value;
    
public:
    DerivedWithVirtualDtor(const string& n, int v) 
        : BaseWithVirtualDtor(n), value(v) {
        cout << "DerivedWithVirtualDtor constructor called with value " << value << endl;
    }
    
    ~DerivedWithVirtualDtor() {
        cout << "DerivedWithVirtualDtor destructor called for " << name 
             << " with value " << value << endl;
    }
    
    void display() const override {
        cout << "DerivedWithVirtualDtor: " << name << " with value " << value << endl;
    }
};

int main() {
    cout << "=== Without Virtual Destructor (PROBLEMATIC) ===" << endl;
    {
        BaseWithoutVirtualDtor* ptr = new DerivedWithoutVirtualDtor("Problematic", 42);
        ptr->display();
        delete ptr;  // Only base destructor called! Derived destructor NOT called!
    }
    
    cout << "\n=== With Virtual Destructor (CORRECT) ===" << endl;
    {
        BaseWithVirtualDtor* ptr = new DerivedWithVirtualDtor("Correct", 123);
        ptr->display();
        delete ptr;  // Both derived and base destructors called!
    }
    
    cout << "\n=== Stack objects (both work fine) ===" << endl;
    {
        DerivedWithoutVirtualDtor stackObj("Stack", 999);
        stackObj.display();
        // Both destructors called properly for stack objects
    }
    
    return 0;
}
```

## Function Overriding vs Hiding

Understanding the difference between overriding and hiding is crucial.

### Exercise 7: Function Overriding vs Hiding

Complete this example showing the difference:

```cpp
#include <iostream>
#include <string>
using namespace std;

class Base {
public:
    // Virtual function - can be overridden
    virtual void display(int x) {
        cout << "Base::display(int): " << x << endl;
    }
    
    // Non-virtual function - will be hidden, not overridden
    void show(double x) {
        cout << "Base::show(double): " << x << endl;
    }
    
    // Function with different signature
    virtual void display(const string& s) {
        cout << "Base::display(string): " << s << endl;
    }
    
    // Another virtual function
    virtual void process() {
        cout << "Base::process()" << endl;
    }
};

class Derived : public Base {
public:
    // Override the virtual function
    void display(int x) override {
        cout << "Derived::display(int): " << x << endl;
    }
    
    // Hide the non-virtual function (not override!)
    void show(double x) {
        cout << "Derived::show(double): " << x << endl;
    }
    
    // Override the string version
    void display(const string& s) override {
        cout << "Derived::display(string): " << s << endl;
    }
    
    // Override process function
    void process() override {
        cout << "Derived::process()" << endl;
    }
    
    // Overload in derived class
    void display(float x) {
        cout << "Derived::display(float): " << x << endl;
    }
};

int main() {
    cout << "=== Polymorphic calls ===" << endl;
    Base* ptr = new Derived();
    
    // These will call derived versions due to virtual functions
    ptr->display(42);           // Calls Derived::display(int)
    ptr->process();             // Calls Derived::process()
    ptr->display(string("Hello")); // Calls Derived::display(string)
    
    // This calls base version because show() is not virtual
    ptr->show(3.14);            // Calls Base::show(double)
    
    cout << "\n=== Direct calls ===" << endl;
    Derived direct;
    
    direct.display(100);        // Calls Derived::display(int)
    direct.show(2.71);          // Calls Derived::show(double)
    direct.display(string("World")); // Calls Derived::display(string)
    direct.display(1.5f);       // Calls Derived::display(float) - overload
    
    cout << "\n=== Name hiding demonstration ===" << endl;
    Base base;
    base.display(50);           // Calls Base::display(int)
    base.show(1.23);            // Calls Base::show(double)
    
    // Derived hides Base::show, but not Base::display (because it's virtual)
    // All display functions from base are still accessible through derived
    direct.Base::display(200);  // Explicitly call base version
    direct.Base::show(9.99);    // Explicitly call base version
    
    delete ptr;
    return 0;
}
```

## Multiple Inheritance

C++ supports multiple inheritance, allowing a class to inherit from multiple base classes.

### Exercise 8: Multiple Inheritance

Complete this multiple inheritance example:

```cpp
#include <iostream>
#include <string>
using namespace std;

class Flyable {
protected:
    double maxAltitude;
    
public:
    Flyable(double altitude) : maxAltitude(altitude) {
        cout << "Flyable constructor called" << endl;
    }
    
    virtual void fly() const {
        cout << "Flying at altitude up to " << maxAltitude << " feet" << endl;
    }
    
    virtual ~Flyable() {
        cout << "Flyable destructor called" << endl;
    }
    
    virtual string getFlightCapability() const {
        return "Can fly";
    }
};

class Swimmable {
protected:
    double maxDepth;
    
public:
    Swimmable(double depth) : maxDepth(depth) {
        cout << "Swimmable constructor called" << endl;
    }
    
    virtual void swim() const {
        cout << "Swimming at depth up to " << maxDepth << " feet" << endl;
    }
    
    virtual ~Swimmable() {
        cout << "Swimmable destructor called" << endl;
    }
    
    virtual string getSwimCapability() const {
        return "Can swim";
    }
};

class Walkable {
protected:
    double maxSpeed;
    
public:
    Walkable(double speed) : maxSpeed(speed) {
        cout << "Walkable constructor called" << endl;
    }
    
    virtual void walk() const {
        cout << "Walking at speed up to " << maxSpeed << " mph" << endl;
    }
    
    virtual ~Walkable() {
        cout << "Walkable destructor called" << endl;
    }
    
    virtual string getWalkCapability() const {
        return "Can walk";
    }
};

// Multiple inheritance - inherits from all three
class Duck : public Flyable, public Swimmable, public Walkable {
private:
    string name;
    
public:
    Duck(const string& n, double alt, double depth, double speed) 
        : Flyable(alt), Swimmable(depth), Walkable(speed), name(n) {
        cout << "Duck constructor called for " << name << endl;
    }
    
    // Override all virtual functions
    void fly() const override {
        cout << name << " flies gracefully at altitude up to " << maxAltitude << " feet" << endl;
    }
    
    void swim() const override {
        cout << name << " swims elegantly at depth up to " << maxDepth << " feet" << endl;
    }
    
    void walk() const override {
        cout << name << " waddles at speed up to " << maxSpeed << " mph" << endl;
    }
    
    // Combined behavior
    void displayAbilities() const {
        cout << name << " can:" << endl;
        cout << "  - " << Flyable::getFlightCapability() << endl;
        cout << "  - " << Swimmable::getSwimCapability() << endl;
        cout << "  - " << Walkable::getWalkCapability() << endl;
    }
    
    ~Duck() {
        cout << "Duck destructor called for " << name << endl;
    }
};

int main() {
    cout << "=== Creating Duck object ===" << endl;
    Duck duck("Donald", 1000.0, 10.0, 3.0);
    
    cout << "\n=== Individual abilities ===" << endl;
    duck.fly();
    duck.swim();
    duck.walk();
    
    cout << "\n=== Combined abilities ===" << endl;
    duck.displayAbilities();
    
    cout << "\n=== Polymorphic behavior ===" << endl;
    Flyable* flyer = &duck;
    Swimmable* swimmer = &duck;
    Walkable* walker = &duck;
    
    flyer->fly();
    swimmer->swim();
    walker->walk();
    
    cout << "\n=== Diamond problem setup ===" << endl;
    // This demonstrates the classic diamond problem
    // We'll see how to solve it with virtual inheritance in the next example
    
    return 0;
}
```

## Virtual Inheritance and the Diamond Problem

The diamond problem occurs when a class inherits from two classes that both inherit from the same base class.

### Exercise 9: Virtual Inheritance

Complete this example solving the diamond problem:

```cpp
#include <iostream>
#include <string>
using namespace std;

// Base class
class Animal {
protected:
    string name;
    
public:
    Animal(const string& n) : name(n) {
        cout << "Animal constructor called for " << name << endl;
    }
    
    virtual void eat() const {
        cout << name << " is eating" << endl;
    }
    
    virtual ~Animal() {
        cout << "Animal destructor called for " << name << endl;
    }
    
    string getName() const { return name; }
};

// Two classes inheriting from Animal
class Mammal : virtual public Animal {  // Virtual inheritance
protected:
    bool warmBlooded;
    
public:
    Mammal(const string& n, bool warm) : Animal(n), warmBlooded(warm) {
        cout << "Mammal constructor called for " << name << endl;
    }
    
    virtual void giveBirth() const {
        cout << name << " gives birth to live young" << endl;
    }
    
    virtual ~Mammal() {
        cout << "Mammal destructor called for " << name << endl;
    }
};

class WingedAnimal : virtual public Animal {  // Virtual inheritance
protected:
    bool canFly;
    
public:
    WingedAnimal(const string& n, bool fly) : Animal(n), canFly(fly) {
        cout << "WingedAnimal constructor called for " << name << endl;
    }
    
    virtual void flapWings() const {
        if (canFly) {
            cout << name << " flaps wings" << endl;
        } else {
            cout << name << " cannot fly" << endl;
        }
    }
    
    virtual ~WingedAnimal() {
        cout << "WingedAnimal destructor called for " << name << endl;
    }
};

// Derived class inheriting from both Mammal and WingedAnimal
class Bat : public Mammal, public WingedAnimal {
private:
    double wingSpan;
    
public:
    // Constructor must initialize the virtual base class
    Bat(const string& n, double span) 
        : Animal(n), Mammal(n, true), WingedAnimal(n, true), wingSpan(span) {
        cout << "Bat constructor called for " << name << endl;
    }
    
    void eat() const override {
        cout << name << " eats insects while flying" << endl;
    }
    
    void giveBirth() const override {
        cout << name << " gives birth to live young and nurses them" << endl;
    }
    
    void flapWings() const override {
        cout << name << " flaps " << wingSpan << " foot wings" << endl;
    }
    
    void navigateWithEcholocation() const {
        cout << name << " uses echolocation to navigate" << endl;
    }
    
    ~Bat() {
        cout << "Bat destructor called for " << name << endl;
    }
};

int main() {
    cout << "=== Creating Bat object (Diamond Problem Solution) ===" << endl;
    Bat bat("Bruce", 1.5);
    
    cout << "\n=== Individual behaviors ===" << endl;
    bat.eat();  // From Bat
    bat.giveBirth();  // From Bat (overridden)
    bat.flapWings();  // From Bat (overridden)
    bat.navigateWithEcholocation();  // From Bat (specific to Bat)
    
    cout << "\n=== Polymorphic behavior ===" << endl;
    Animal* animal = &bat;
    Mammal* mammal = &bat;
    WingedAnimal* winged = &bat;
    
    animal->eat();  // Calls Bat's version
    mammal->giveBirth();  // Calls Bat's version
    winged->flapWings();  // Calls Bat's version
    
    cout << "\n=== Demonstrating virtual inheritance ===" << endl;
    cout << "Bat's name (from Animal): " << bat.getName() << endl;
    // Without virtual inheritance, we'd have two copies of Animal's members
    // With virtual inheritance, there's only one copy
    
    return 0;
}
```

## Advanced Polymorphism Techniques

### Exercise 10: Visitor Pattern Example

Implement a simple visitor pattern to demonstrate advanced polymorphism:

```cpp
#include <iostream>
#include <vector>
#include <memory>
using namespace std;

// Forward declaration
class Circle;
class Rectangle;
class Triangle;

// Visitor interface
class ShapeVisitor {
public:
    virtual ~ShapeVisitor() = default;
    virtual void visit(Circle& circle) = 0;
    virtual void visit(Rectangle& rectangle) = 0;
    virtual void visit(Triangle& triangle) = 0;
};

// Abstract base class for shapes
class Shape {
public:
    virtual ~Shape() = default;
    virtual void accept(ShapeVisitor& visitor) = 0;
    virtual void display() const = 0;
};

// Concrete shape classes
class Circle : public Shape {
private:
    double radius;
    
public:
    Circle(double r) : radius(r) {}
    
    double getRadius() const { return radius; }
    
    void accept(ShapeVisitor& visitor) override {
        visitor.visit(*this);
    }
    
    void display() const override {
        cout << "Circle with radius " << radius << endl;
    }
};

class Rectangle : public Shape {
private:
    double width, height;
    
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    
    double getWidth() const { return width; }
    double getHeight() const { return height; }
    
    void accept(ShapeVisitor& visitor) override {
        visitor.visit(*this);
    }
    
    void display() const override {
        cout << "Rectangle " << width << "x" << height << endl;
    }
};

class Triangle : public Shape {
private:
    double base, height;
    
public:
    Triangle(double b, double h) : base(b), height(h) {}
    
    double getBase() const { return base; }
    double getHeight() const { return height; }
    
    void accept(ShapeVisitor& visitor) override {
        visitor.visit(*this);
    }
    
    void display() const override {
        cout << "Triangle with base " << base << " and height " << height << endl;
    }
};

// Concrete visitors
class AreaCalculator : public ShapeVisitor {
public:
    double totalArea = 0.0;
    
    void visit(Circle& circle) override {
        double area = 3.14159 * circle.getRadius() * circle.getRadius();
        cout << "Circle area: " << area << endl;
        totalArea += area;
    }
    
    void visit(Rectangle& rectangle) override {
        double area = rectangle.getWidth() * rectangle.getHeight();
        cout << "Rectangle area: " << area << endl;
        totalArea += area;
    }
    
    void visit(Triangle& triangle) override {
        double area = 0.5 * triangle.getBase() * triangle.getHeight();
        cout << "Triangle area: " << area << endl;
        totalArea += area;
    }
};

class PerimeterCalculator : public ShapeVisitor {
public:
    double totalPerimeter = 0.0;
    
    void visit(Circle& circle) override {
        double perimeter = 2 * 3.14159 * circle.getRadius();
        cout << "Circle perimeter: " << perimeter << endl;
        totalPerimeter += perimeter;
    }
    
    void visit(Rectangle& rectangle) override {
        double perimeter = 2 * (rectangle.getWidth() + rectangle.getHeight());
        cout << "Rectangle perimeter: " << perimeter << endl;
        totalPerimeter += perimeter;
    }
    
    void visit(Triangle& triangle) override {
        // Assuming equilateral triangle for simplicity
        double perimeter = 3 * triangle.getBase();
        cout << "Triangle perimeter: " << perimeter << endl;
        totalPerimeter += perimeter;
    }
};

int main() {
    // Create shapes
    vector<unique_ptr<Shape>> shapes;
    shapes.push_back(make_unique<Circle>(5.0));
    shapes.push_back(make_unique<Rectangle>(4.0, 6.0));
    shapes.push_back(make_unique<Triangle>(3.0, 8.0));
    
    cout << "=== Shapes ===" << endl;
    for (const auto& shape : shapes) {
        shape->display();
    }
    
    cout << "\n=== Calculating Areas ===" << endl;
    AreaCalculator areaCalc;
    for (auto& shape : shapes) {
        shape->accept(areaCalc);
    }
    cout << "Total area: " << areaCalc.totalArea << endl;
    
    cout << "\n=== Calculating Perimeters ===" << endl;
    PerimeterCalculator perimCalc;
    for (auto& shape : shapes) {
        shape->accept(perimCalc);
    }
    cout << "Total perimeter: " << perimCalc.totalPerimeter << endl;
    
    return 0;
}
```

## Best Practices for Advanced OOP

### Exercise 11: Best Practices Example

Demonstrate best practices in advanced OOP:

```cpp
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
using namespace std;

// 1. Use virtual destructors in base classes
// 2. Use override keyword for clarity
// 3. Prefer composition over inheritance when possible
// 4. Use smart pointers for automatic memory management

// Abstract base class with best practices
class Drawable {
public:
    virtual ~Drawable() = default;  // Virtual destructor
    
    // Pure virtual functions
    virtual void draw() const = 0;
    virtual double getArea() const = 0;
    
    // Non-virtual interface pattern
    void displayInfo() const {
        cout << "Area: " << getArea() << endl;
        draw();
    }
};

// Composition example: instead of inheriting from multiple classes,
// we compose objects together
class Position {
private:
    double x, y;
    
public:
    Position(double x_pos = 0, double y_pos = 0) : x(x_pos), y(y_pos) {}
    
    double getX() const { return x; }
    double getY() const { return y; }
    
    void setPosition(double x_pos, double y_pos) {
        x = x_pos;
        y = y_pos;
    }
    
    void displayPosition() const {
        cout << "Position: (" << x << ", " << y << ")" << endl;
    }
};

class ColoredShape : public Drawable {
protected:
    unique_ptr<Position> position;  // Composition instead of inheritance
    string color;
    
public:
    ColoredShape(double x, double y, const string& c) 
        : position(make_unique<Position>(x, y)), color(c) {}
    
    // Composition: delegate to composed object
    void move(double x, double y) {
        position->setPosition(x, y);
    }
    
    void displayPosition() const {
        position->displayPosition();
    }
    
    string getColor() const { return color; }
    
    virtual ~ColoredShape() = default;
};

class ColoredCircle : public ColoredShape {
private:
    double radius;
    
public:
    ColoredCircle(double x, double y, const string& c, double r) 
        : ColoredShape(x, y, c), radius(r) {}
    
    void draw() const override {
        cout << "Drawing " << color << " circle at ";
        position->displayPosition();
        cout << "with radius " << radius << endl;
    }
    
    double getArea() const override {
        return 3.14159 * radius * radius;
    }
    
    double getRadius() const { return radius; }
};

class ColoredRectangle : public ColoredShape {
private:
    double width, height;
    
public:
    ColoredRectangle(double x, double y, const string& c, double w, double h) 
        : ColoredShape(x, y, c), width(w), height(h) {}
    
    void draw() const override {
        cout << "Drawing " << color << " rectangle at ";
        position->displayPosition();
        cout << "with dimensions " << width << "x" << height << endl;
    }
    
    double getArea() const override {
        return width * height;
    }
    
    double getWidth() const { return width; }
    double getHeight() const { return height; }
};

int main() {
    cout << "=== Best Practices Demo ===" << endl;
    
    // Use smart pointers for automatic memory management
    vector<unique_ptr<Drawable>> shapes;
    shapes.push_back(make_unique<ColoredCircle>(0, 0, "Red", 5.0));
    shapes.push_back(make_unique<ColoredRectangle>(10, 10, "Blue", 4.0, 6.0));
    shapes.push_back(make_unique<ColoredCircle>(5, 5, "Green", 3.0));
    
    cout << "\n=== Polymorphic Behavior ===" << endl;
    for (const auto& shape : shapes) {
        shape->displayInfo();  // Calls both getArea() and draw()
        cout << "---" << endl;
    }
    
    cout << "\n=== Specific Functionality ===" << endl;
    // Downcast when specific functionality is needed
    for (const auto& shape : shapes) {
        shape->displayInfo();
        
        // Dynamic casting for specific functionality
        if (auto* circle = dynamic_cast<ColoredCircle*>(shape.get())) {
            cout << "This is a circle with radius: " << circle->getRadius() << endl;
        } else if (auto* rect = dynamic_cast<ColoredRectangle*>(shape.get())) {
            cout << "This is a rectangle with dimensions: " 
                 << rect->getWidth() << "x" << rect->getHeight() << endl;
        }
        cout << "---" << endl;
    }
    
    // Smart pointers automatically handle cleanup
    return 0;
}
```

## Summary

In this chapter, you learned:
- Inheritance and how to create derived classes
- Access control in inheritance (public, protected, private)
- Virtual functions and runtime polymorphism
- Abstract classes and pure virtual functions
- Virtual destructors and proper cleanup
- Function overriding vs function hiding
- Multiple inheritance and the diamond problem
- Virtual inheritance to solve the diamond problem
- Advanced polymorphism patterns like the visitor pattern
- Best practices for advanced OOP

## Key Takeaways

- Inheritance enables code reuse and polymorphism
- Virtual functions enable runtime polymorphism
- Abstract classes provide interfaces without implementations
- Virtual destructors ensure proper cleanup in inheritance hierarchies
- Multiple inheritance can lead to the diamond problem, solved with virtual inheritance
- Composition is often preferable to inheritance
- Smart pointers help with automatic memory management
- The override keyword improves code safety and readability

## Common Mistakes to Avoid

1. Forgetting virtual destructors in base classes
2. Confusing function overriding with function hiding
3. Using multiple inheritance unnecessarily
4. Not using the override keyword for clarity
5. Forgetting that non-virtual functions are hidden, not overridden
6. Creating deep inheritance hierarchies
7. Using private inheritance when composition would be clearer
8. Not considering the diamond problem with multiple inheritance

## Next Steps

Now that you understand advanced OOP concepts, you're ready to learn about templates and generic programming in Chapter 9.