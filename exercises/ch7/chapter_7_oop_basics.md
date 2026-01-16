# Chapter 7: Object-Oriented Programming Basics

## Overview

This chapter introduces the fundamental concepts of object-oriented programming (OOP) in C++. You'll learn about classes, objects, constructors, destructors, encapsulation, and the basic principles of OOP.

## Learning Objectives

By the end of this chapter, you will:
- Understand the concept of classes and objects
- Learn how to define classes with data members and member functions
- Master constructors and destructors
- Understand access specifiers and encapsulation
- Learn about const member functions
- Understand the 'this' pointer
- Learn about static members
- Understand the basic principles of OOP

## Classes and Objects

A class is a blueprint for creating objects. An object is an instance of a class.

### Exercise 1: Basic Class Definition

The following code has errors. Find and fix them:

```cpp
#include <iostream>
#include <string>
using namespace std;

// Basic class definition
class Rectangle {
private:
    double width;   // Private data members
    double height;
    
public:
    // Public member functions
    void setDimensions(double w, double h) {
        width = w;   // Error: no validation
        height = h;  // Error: no validation
    }
    
    double getArea() {
        return width * height;
    }
    
    double getPerimeter() {
        return 2 * (width + height);
    }
    
    void display() {
        cout << "Rectangle: " << width << " x " << height << endl;
    }
};

int main() {
    // Creating objects
    Rectangle rect1;  // Default constructor (compiler-generated)
    Rectangle rect2;  // Another object
    
    // Setting dimensions
    rect1.setDimensions(5.0, 3.0);
    rect2.setDimensions(-2.0, 4.0);  // Error: negative dimensions!
    
    // Displaying information
    rect1.display();
    cout << "Area: " << rect1.getArea() << endl;
    cout << "Perimeter: " << rect1.getPerimeter() << endl;
    
    rect2.display();
    cout << "Area: " << rect2.getArea() << endl;  // Negative area!
    cout << "Perimeter: " << rect2.getPerimeter() << endl;
    
    return 0;
}
```

### Exercise 2: Improved Class with Validation

Fix the validation issues in the Rectangle class:

```cpp
#include <iostream>
#include <string>
using namespace std;

class Rectangle {
private:
    double width;
    double height;
    
public:
    // Constructor with validation
    Rectangle(double w = 0.0, double h = 0.0) {
        setDimensions(w, h);
    }
    
    // Setter with validation
    void setDimensions(double w, double h) {
        if (w >= 0) width = w;
        else width = 0;  // Default to 0 if negative
        
        if (h >= 0) height = h;
        else height = 0; // Default to 0 if negative
    }
    
    // Getter functions
    double getWidth() const { return width; }
    double getHeight() const { return height; }
    
    double getArea() const {
        return width * height;
    }
    
    double getPerimeter() const {
        return 2 * (width + height);
    }
    
    void display() const {
        cout << "Rectangle: " << width << " x " << height 
             << " (Area: " << getArea() << ", Perimeter: " << getPerimeter() << ")" << endl;
    }
};

int main() {
    // Creating objects with constructors
    Rectangle rect1(5.0, 3.0);  // Valid dimensions
    Rectangle rect2(-2.0, 4.0); // Invalid width, will be set to 0
    Rectangle rect3;            // Default constructor, both dimensions 0
    
    cout << "Rectangles with validation:" << endl;
    rect1.display();
    rect2.display();
    rect3.display();
    
    // Changing dimensions
    rect3.setDimensions(10.0, 8.0);
    rect3.display();
    
    return 0;
}
```

## Constructors and Destructors

Constructors initialize objects, while destructors clean up resources.

### Exercise 3: Constructor Types

Complete this example with different types of constructors:

```cpp
#include <iostream>
#include <string>
using namespace std;

class Student {
private:
    string name;
    int id;
    double gpa;
    
public:
    // Default constructor
    Student() {
        name = "Unknown";
        id = 0;
        gpa = 0.0;
        cout << "Default constructor called for " << name << endl;
    }
    
    // Parameterized constructor
    Student(const string& n, int i, double g) {
        name = n;
        id = i;
        gpa = g;
        cout << "Parameterized constructor called for " << name << endl;
    }
    
    // Copy constructor
    Student(const Student& other) {
        name = other.name;
        id = other.id;
        gpa = other.gpa;
        cout << "Copy constructor called for " << name << endl;
    }
    
    // Destructor
    ~Student() {
        cout << "Destructor called for " << name << endl;
    }
    
    // Getter functions
    string getName() const { return name; }
    int getId() const { return id; }
    double getGPA() const { return gpa; }
    
    // Setter functions
    void setName(const string& n) { name = n; }
    void setId(int i) { id = i; }
    void setGPA(double g) { gpa = g; }
    
    void display() const {
        cout << "Student: " << name << " (ID: " << id << ", GPA: " << gpa << ")" << endl;
    }
};

int main() {
    cout << "Creating students:" << endl;
    
    // Default constructor
    Student s1;
    s1.display();
    
    // Parameterized constructor
    Student s2("Alice Johnson", 12345, 3.85);
    s2.display();
    
    // Copy constructor
    Student s3 = s2;  // Copy constructor called
    s3.display();
    
    // Another way to copy
    Student s4(s2);   // Copy constructor called
    s4.display();
    
    cout << "\nEnd of main function:" << endl;
    
    return 0;  // Destructors will be called automatically
}
```

### Exercise 4: Constructor Initialization Lists

Fix the inefficiencies in this code using initialization lists:

```cpp
#include <iostream>
#include <string>
#include <vector>
using namespace std;

class Course {
private:
    string courseName;
    int courseId;
    vector<int> studentIds;  // Expensive to default construct
    
public:
    // Inefficient constructor - default constructs then assigns
    Course(string name, int id) {
        courseName = name;  // Assignment after default construction
        courseId = id;      // Assignment after default construction
        // studentIds is default constructed as empty vector
    }
    
    // Efficient constructor using initialization list
    Course(const string& name, int id, const vector<int>& ids) 
        : courseName(name), courseId(id), studentIds(ids) {
        // All members are initialized directly
    }
    
    // Best practice: using member initializer list
    Course(const string& name, int id) 
        : courseName(name), courseId(id), studentIds() {  // Explicit initialization
        // This is more efficient than default construction then assignment
    }
    
    void display() const {
        cout << "Course: " << courseName << " (ID: " << courseId << ")" << endl;
        cout << "Students: ";
        for (int id : studentIds) {
            cout << id << " ";
        }
        cout << endl;
    }
};

class StudentBetter {
private:
    string name;
    int id;
    double gpa;
    
public:
    // Good constructor with initialization list
    StudentBetter(const string& n, int i, double g) 
        : name(n), id(i), gpa(g) {
        cout << "StudentBetter constructor called for " << name << endl;
    }
    
    // Copy constructor with initialization list
    StudentBetter(const StudentBetter& other) 
        : name(other.name), id(other.id), gpa(other.gpa) {
        cout << "StudentBetter copy constructor called for " << name << endl;
    }
    
    // Assignment operator (Rule of Three/Five)
    StudentBetter& operator=(const StudentBetter& other) {
        if (this != &other) {  // Self-assignment check
            name = other.name;
            id = other.id;
            gpa = other.gpa;
        }
        return *this;
    }
    
    ~StudentBetter() {
        cout << "StudentBetter destructor called for " << name << endl;
    }
    
    void display() const {
        cout << "Student: " << name << " (ID: " << id << ", GPA: " << gpa << ")" << endl;
    }
};

int main() {
    cout << "Creating students with better constructors:" << endl;
    
    StudentBetter s1("Bob Smith", 67890, 3.92);
    s1.display();
    
    StudentBetter s2 = s1;  // Copy constructor
    s2.display();
    
    // Demonstrating assignment
    StudentBetter s3("Carol Davis", 54321, 3.75);
    s3 = s1;  // Assignment operator
    s3.display();
    
    return 0;
}
```

## Access Specifiers and Encapsulation

C++ provides three access specifiers: public, private, and protected.

### Exercise 5: Access Specifiers

Complete this example showing access control:

```cpp
#include <iostream>
#include <string>
using namespace std;

class BankAccount {
private:
    string accountNumber;
    double balance;
    string ownerName;
    
    // Private helper function
    bool isValidAmount(double amount) const {
        return amount >= 0;
    }
    
protected:
    // Protected members - accessible by derived classes
    string accountType;
    
public:
    // Public interface
    BankAccount(const string& accNum, const string& owner, double initialBalance = 0.0)
        : accountNumber(accNum), ownerName(owner), balance(initialBalance), accountType("Standard") {
    }
    
    // Public member functions - interface
    bool deposit(double amount) {
        if (isValidAmount(amount)) {
            balance += amount;
            return true;
        }
        return false;  // Invalid amount
    }
    
    bool withdraw(double amount) {
        if (isValidAmount(amount) && balance >= amount) {
            balance -= amount;
            return true;
        }
        return false;  // Insufficient funds or invalid amount
    }
    
    double getBalance() const {
        return balance;
    }
    
    string getOwnerName() const {
        return ownerName;
    }
    
    string getAccountNumber() const {
        return accountNumber;
    }
    
    void displayAccount() const {
        cout << "Account: " << accountNumber << endl;
        cout << "Owner: " << ownerName << endl;
        cout << "Balance: $" << balance << endl;
        cout << "Type: " << accountType << endl;
    }
    
    // Error: trying to access private members externally
    // void tryToAccessPrivate() {
    //     cout << balance;  // Error: 'balance' is private
    // }
};

int main() {
    BankAccount account("ACC-12345", "John Doe", 1000.0);
    
    cout << "Initial account info:" << endl;
    account.displayAccount();
    
    // Using public interface
    account.deposit(500.0);
    cout << "\nAfter deposit of $500:" << endl;
    account.displayAccount();
    
    account.withdraw(200.0);
    cout << "\nAfter withdrawal of $200:" << endl;
    account.displayAccount();
    
    // Error: cannot access private members directly
    // cout << account.balance << endl;  // Error: 'balance' is private
    // cout << account.accountNumber << endl;  // Error: 'accountNumber' is private
    
    // Access through public interface
    cout << "\nBalance through getter: $" << account.getBalance() << endl;
    cout << "Owner: " << account.getOwnerName() << endl;
    
    return 0;
}
```

## Const Member Functions

Const member functions promise not to modify the object's state.

### Exercise 6: Const Correctness

Fix the const-correctness issues in this code:

```cpp
#include <iostream>
#include <string>
#include <vector>
using namespace std;

class LibraryBook {
private:
    string title;
    string author;
    bool isAvailable;
    mutable int accessCount;  // mutable allows modification even in const functions
    
public:
    LibraryBook(const string& t, const string& a) 
        : title(t), author(a), isAvailable(true), accessCount(0) {
    }
    
    // Const member functions - don't modify object state
    string getTitle() const {
        return title;
    }
    
    string getAuthor() const {
        return author;
    }
    
    bool getAvailability() const {
        return isAvailable;
    }
    
    // This function should be const but modifies state - ERROR
    void displayInfo() const {
        accessCount++;  // OK: mutable member can be modified in const function
        cout << "Title: " << title << endl;
        cout << "Author: " << author << endl;
        cout << "Available: " << (isAvailable ? "Yes" : "No") << endl;
        cout << "Access count: " << accessCount << endl;
    }
    
    // Non-const member functions - can modify state
    bool borrowBook() {
        if (isAvailable) {
            isAvailable = false;
            return true;
        }
        return false;
    }
    
    void returnBook() {
        isAvailable = true;
    }
    
    // Function that should be const but isn't - POTENTIAL ERROR
    string getFullInfo() const {  // Should be const
        return title + " by " + author + " (" + 
               (isAvailable ? "Available" : "Checked out") + ")";
    }
    
    int getAccessCount() const {
        return accessCount;
    }
};

int main() {
    LibraryBook book("The C++ Programming Language", "Bjarne Stroustrup");
    
    // Using const object
    const LibraryBook constBook("Effective C++", "Scott Meyers");
    
    // Can call const member functions on const object
    cout << "Const book title: " << constBook.getTitle() << endl;
    cout << "Const book author: " << constBook.getAuthor() << endl;
    cout << "Const book info: " << constBook.getFullInfo() << endl;
    
    // Cannot call non-const functions on const object
    // constBook.borrowBook();  // Error: cannot call non-const function on const object
    // constBook.returnBook();  // Error: cannot call non-const function on const object
    
    // Working with non-const object
    book.displayInfo();
    cout << "Access count: " << book.getAccessCount() << endl;
    
    book.borrowBook();
    book.displayInfo();
    
    book.returnBook();
    book.displayInfo();
    
    // Demonstrate const reference parameter
    auto printBookInfo = [](const LibraryBook& b) {  // Const reference - efficient and safe
        cout << "Book: " << b.getTitle() << " by " << b.getAuthor() << endl;
        cout << "Available: " << (b.getAvailability() ? "Yes" : "No") << endl;
    };
    
    printBookInfo(book);
    printBookInfo(constBook);
    
    return 0;
}
```

## The 'this' Pointer

Every non-static member function has access to the 'this' pointer, which points to the current object.

### Exercise 7: Using the 'this' Pointer

Complete this example showing various uses of 'this':

```cpp
#include <iostream>
#include <string>
using namespace std;

class Person {
private:
    string name;
    int age;
    string email;
    
public:
    Person(const string& name, int age, const string& email) 
        : name(name), age(age), email(email) {
    }
    
    // Using 'this' to distinguish between member variables and parameters
    Person& setName(const string& name) {
        this->name = name;  // 'this->name' refers to member variable
        return *this;       // Return reference to current object for chaining
    }
    
    Person& setAge(int age) {
        this->age = age;
        return *this;
    }
    
    Person& setEmail(const string& email) {
        this->email = email;
        return *this;
    }
    
    // Method chaining using 'this'
    Person& setDetails(const string& name, int age, const string& email) {
        this->setName(name);
        this->setAge(age);
        this->setEmail(email);
        return *this;
    }
    
    // Self-comparison using 'this'
    bool isSamePerson(const Person& other) const {
        return this == &other;  // Compare addresses
    }
    
    // Function that returns current object
    Person& getThis() {
        return *this;
    }
    
    // Const version
    const Person& getThis() const {
        return *this;
    }
    
    void display() const {
        cout << "Name: " << name << ", Age: " << age << ", Email: " << email << endl;
    }
    
    // Function that could be called on temporary objects
    Person getCopy() const {
        return Person(name, age, email);  // Return by value
    }
};

int main() {
    Person person("John Smith", 30, "john@example.com");
    person.display();
    
    // Method chaining using 'this'
    person.setName("Jane Doe")
          .setAge(25)
          .setEmail("jane@example.com");
    
    cout << "\nAfter method chaining:" << endl;
    person.display();
    
    // Self-comparison
    Person anotherPerson = person;
    cout << "\nIs same person (person vs person): " << person.isSamePerson(person) << endl;
    cout << "Is same person (person vs anotherPerson): " << person.isSamePerson(anotherPerson) << endl;
    
    // Using getThis
    Person& ref = person.getThis();
    cout << "\nUsing getThis(): ";
    ref.display();
    
    // Demonstrating 'this' in action
    cout << "\nAddress of person: " << &person << endl;
    cout << "Address from getThis(): " << &person.getThis() << endl;
    
    // Working with temporary objects
    Person temp = Person("Temp", 20, "temp@example.com").getCopy();
    cout << "\nTemporary object copy: ";
    temp.display();
    
    return 0;
}
```

## Static Members

Static members belong to the class rather than to any specific object.

### Exercise 8: Static Members

Complete this example with static members:

```cpp
#include <iostream>
#include <string>
using namespace std;

class Counter {
private:
    int instanceValue;                    // Instance member
    static int staticCounter;            // Static member - shared among all instances
    static const int MAX_COUNT = 100;    // Static constant
    static string className;             // Static member
    
public:
    Counter(int value = 0) : instanceValue(value) {
        staticCounter++;  // Increment shared counter
        cout << "Constructor called. Total instances: " << staticCounter << endl;
    }
    
    ~Counter() {
        staticCounter--;  // Decrement shared counter
        cout << "Destructor called. Total instances: " << staticCounter << endl;
    }
    
    // Instance functions can access both instance and static members
    void display() const {
        cout << "Instance value: " << instanceValue 
             << ", Static counter: " << staticCounter << endl;
    }
    
    // Static functions can only access static members
    static void displayStaticInfo() {
        cout << "Static counter: " << staticCounter 
             << ", Class name: " << className << endl;
        // cout << instanceValue;  // Error: cannot access instance members
    }
    
    static int getStaticCounter() {
        return staticCounter;
    }
    
    static void resetCounter() {
        staticCounter = 0;
    }
    
    static bool isAtMaxCapacity() {
        return staticCounter >= MAX_COUNT;
    }
};

// Define static members outside the class
int Counter::staticCounter = 0;
string Counter::className = "Counter";

int main() {
    cout << "Initial static counter: " << Counter::getStaticCounter() << endl;
    
    // Create objects
    Counter c1(10);
    Counter c2(20);
    Counter c3(30);
    
    cout << "\nAfter creating 3 objects:" << endl;
    Counter::displayStaticInfo();
    
    // Individual objects can access static members
    c1.display();
    c2.display();
    
    // Static functions can be called without creating objects
    cout << "\nCalling static function:" << endl;
    Counter::displayStaticInfo();
    
    // Create more objects
    {
        Counter c4(40);
        Counter c5(50);
        cout << "\nInside scope with 2 more objects:" << endl;
        Counter::displayStaticInfo();
    }  // c4 and c5 are destroyed here
    
    cout << "\nAfter scope ends:" << endl;
    Counter::displayStaticInfo();
    
    // Demonstrate static member access
    cout << "\nDirect access to static counter: " << Counter::getStaticCounter() << endl;
    
    // Reset counter
    Counter::resetCounter();
    cout << "After reset: " << Counter::getStaticCounter() << endl;
    
    // Create new objects
    Counter c6(60);
    Counter::displayStaticInfo();
    
    return 0;
}
```

## Friend Functions and Classes

Friend functions and classes can access private members of a class.

### Exercise 9: Friend Functions and Classes

Complete this example with friend functions:

```cpp
#include <iostream>
#include <string>
using namespace std;

class Time {
private:
    int hours;
    int minutes;
    
public:
    Time(int h = 0, int m = 0) : hours(h), minutes(m) {
        normalize();  // Keep time valid
    }
    
    void normalize() {
        if (minutes >= 60) {
            hours += minutes / 60;
            minutes = minutes % 60;
        }
        if (hours >= 24) {
            hours = hours % 24;
        }
    }
    
    // Friend function declaration
    friend Time addTimes(const Time& t1, const Time& t2);
    
    // Friend class declaration
    friend class TimeComparator;
    
    void display() const {
        cout << hours << ":" << (minutes < 10 ? "0" : "") << minutes << endl;
    }
    
    int getHours() const { return hours; }
    int getMinutes() const { return minutes; }
};

// Friend function definition - can access private members
Time addTimes(const Time& t1, const Time& t2) {
    // Access private members directly
    int totalHours = t1.hours + t2.hours;  // Direct access to private members
    int totalMinutes = t1.minutes + t2.minutes;  // Direct access to private members
    
    return Time(totalHours, totalMinutes);
}

// Friend class
class TimeComparator {
public:
    static bool isEarlier(const Time& t1, const Time& t2) {
        // Access private members of both objects
        if (t1.hours < t2.hours) return true;
        if (t1.hours > t2.hours) return false;
        return t1.minutes < t2.minutes;  // Direct access to private members
    }
    
    static bool isLater(const Time& t1, const Time& t2) {
        return isEarlier(t2, t1);
    }
    
    static void compareTimes(const Time& t1, const Time& t2) {
        cout << "Time 1: ";
        t1.display();
        cout << "Time 2: ";
        t2.display();
        
        cout << "Time 1 is earlier than Time 2: " << isEarlier(t1, t2) << endl;
        cout << "Time 1 is later than Time 2: " << isLater(t1, t2) << endl;
    }
};

int main() {
    Time t1(10, 45);  // 10:45
    Time t2(3, 30);   // 3:30
    
    cout << "Time 1: ";
    t1.display();
    cout << "Time 2: ";
    t2.display();
    
    // Use friend function
    Time sum = addTimes(t1, t2);
    cout << "Sum (using friend function): ";
    sum.display();
    
    // Use friend class
    TimeComparator::compareTimes(t1, t2);
    
    // Create times that need normalization
    Time t3(25, 75);  // 25 hours, 75 minutes
    cout << "\nTime that needs normalization: ";
    t3.display();  // Should show 2:15 (next day)
    
    return 0;
}
```

## Practical Example: Student Management System

### Exercise 10: Complete Student Class

Create a comprehensive Student class with all OOP concepts:

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

class Student {
private:
    static int totalStudents;  // Track total number of students
    int studentId;
    string firstName;
    string lastName;
    vector<double> grades;
    static const int MAX_GRADES = 100;  // Maximum number of grades allowed
    
public:
    // Constructor
    Student(const string& first, const string& last, int id) 
        : firstName(first), lastName(last), studentId(id) {
        totalStudents++;
        cout << "Student " << fullName() << " created. Total students: " << totalStudents << endl;
    }
    
    // Destructor
    ~Student() {
        totalStudents--;
        cout << "Student " << fullName() << " destroyed. Total students: " << totalStudents << endl;
    }
    
    // Copy constructor
    Student(const Student& other) 
        : firstName(other.firstName), lastName(other.lastName), 
          studentId(other.studentId), grades(other.grades) {
        totalStudents++;
        cout << "Student " << fullName() << " copied. Total students: " << totalStudents << endl;
    }
    
    // Assignment operator
    Student& operator=(const Student& other) {
        if (this != &other) {
            firstName = other.firstName;
            lastName = other.lastName;
            studentId = other.studentId;
            grades = other.grades;
        }
        return *this;
    }
    
    // Getters
    string getFirstName() const { return firstName; }
    string getLastName() const { return lastName; }
    string fullName() const { return firstName + " " + lastName; }
    int getId() const { return studentId; }
    const vector<double>& getGrades() const { return grades; }
    
    // Setters
    void setFirstName(const string& first) { firstName = first; }
    void setLastName(const string& last) { lastName = last; }
    
    // Add grade with validation
    bool addGrade(double grade) {
        if (grade >= 0 && grade <= 100 && grades.size() < MAX_GRADES) {
            grades.push_back(grade);
            return true;
        }
        return false;  // Invalid grade or too many grades
    }
    
    // Calculate average
    double getAverage() const {
        if (grades.empty()) return 0.0;
        
        double sum = 0;
        for (double grade : grades) {
            sum += grade;
        }
        return sum / grades.size();
    }
    
    // Get highest grade
    double getHighestGrade() const {
        if (grades.empty()) return 0.0;
        
        return *max_element(grades.begin(), grades.end());
    }
    
    // Get lowest grade
    double getLowestGrade() const {
        if (grades.empty()) return 0.0;
        
        return *min_element(grades.begin(), grades.end());
    }
    
    // Get number of grades
    size_t getGradeCount() const {
        return grades.size();
    }
    
    // Display student information
    void display() const {
        cout << "Student ID: " << studentId << endl;
        cout << "Name: " << fullName() << endl;
        cout << "Grades: ";
        for (size_t i = 0; i < grades.size(); i++) {
            cout << grades[i];
            if (i < grades.size() - 1) cout << ", ";
        }
        cout << endl;
        cout << "Average: " << getAverage() << endl;
        cout << "Highest: " << getHighestGrade() << ", Lowest: " << getLowestGrade() << endl;
    }
    
    // Static member functions
    static int getTotalStudents() { return totalStudents; }
    static void resetStudentCount() { totalStudents = 0; }
    
    // Comparison operators
    bool operator==(const Student& other) const {
        return studentId == other.studentId;
    }
    
    bool operator<(const Student& other) const {
        return getAverage() < other.getAverage();
    }
    
    // Friend function for comparing students
    friend bool isHigherAchiever(const Student& s1, const Student& s2) {
        return s1.getAverage() > s2.getAverage();  // Access private grades through member function
    }
};

// Define static member
int Student::totalStudents = 0;

int main() {
    cout << "Starting student management system..." << endl;
    cout << "Total students initially: " << Student::getTotalStudents() << endl;
    
    // Create students
    Student s1("Alice", "Johnson", 1001);
    Student s2("Bob", "Smith", 1002);
    Student s3("Carol", "Davis", 1003);
    
    // Add grades
    s1.addGrade(85.5);
    s1.addGrade(92.0);
    s1.addGrade(78.5);
    s1.addGrade(96.0);
    
    s2.addGrade(76.0);
    s2.addGrade(81.5);
    s2.addGrade(89.0);
    s2.addGrade(72.5);
    
    s3.addGrade(94.5);
    s3.addGrade(91.0);
    s3.addGrade(98.5);
    s3.addGrade(93.0);
    
    // Display student information
    cout << "\n--- Student Information ---" << endl;
    s1.display();
    cout << endl;
    s2.display();
    cout << endl;
    s3.display();
    cout << endl;
    
    // Compare students
    cout << "--- Student Comparisons ---" << endl;
    cout << "Alice is higher achiever than Bob: " << isHigherAchiever(s1, s2) << endl;
    cout << "Carol is higher achiever than Alice: " << isHigherAchiever(s3, s1) << endl;
    
    // Create a copy
    Student s4 = s1;  // Copy constructor
    cout << "\nCopy of Alice:" << endl;
    s4.display();
    
    // Demonstrate static member
    cout << "\nTotal students in system: " << Student::getTotalStudents() << endl;
    
    // Sort students by average (would need to put them in a container)
    vector<Student> students = {s1, s2, s3};
    // Note: This would require a different approach since we can't copy non-copyable objects
    // This is just to demonstrate the concept
    
    return 0;
}
```

## Best Practices

1. Use access specifiers appropriately to enforce encapsulation
2. Make member functions const when they don't modify object state
3. Use initialization lists in constructors for efficiency
4. Follow the Rule of Three/Five when managing resources
5. Use const references for parameters when possible
6. Initialize static members outside the class
7. Use friend functions sparingly and only when necessary
8. Validate inputs in setter functions and constructors

## Summary

In this chapter, you learned:
- How to define classes with data members and member functions
- The different types of constructors and destructors
- Access specifiers and encapsulation principles
- Const member functions and const correctness
- The 'this' pointer and its uses
- Static members and their characteristics
- Friend functions and classes
- Best practices for class design

## Key Takeaways

- Classes are blueprints for creating objects
- Encapsulation hides implementation details
- Constructors initialize objects, destructors clean up
- Const correctness prevents unintended modifications
- Static members belong to the class, not individual objects
- The 'this' pointer refers to the current object
- Friend functions can access private members (use sparingly)

## Common Mistakes to Avoid

1. Forgetting to initialize member variables in constructors
2. Not following the Rule of Three/Five for resource management
3. Making too many members public instead of using proper encapsulation
4. Not using const when member functions don't modify state
5. Accessing private members from outside the class
6. Forgetting to define static members outside the class
7. Overusing friend functions/classes
8. Not validating inputs in setter functions

## Next Steps

Now that you understand object-oriented programming basics, you're ready to learn about advanced OOP concepts like inheritance and polymorphism in Chapter 8.