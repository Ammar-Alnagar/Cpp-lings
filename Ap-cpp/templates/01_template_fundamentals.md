# Chapter 1: Template Fundamentals

## Introduction to Templates

Templates are C++'s mechanism for generic programming, allowing code to work with any type. They enable compile-time polymorphism and are fundamental to modern C++ design.

## Why Templates?

Without templates, we need separate functions for each type:

```cpp
int max(int a, int b) { return a > b ? a : b; }
double max(double a, double b) { return a > b ? a : b; }
std::string max(std::string a, std::string b) { return a > b ? a : b; }
```

With templates, we write one generic version:

```cpp
template<typename T>
T max(T a, T b) {
    return a > b ? a : b;
}
```

## Function Templates

### Basic Syntax

```cpp
template<typename T>
T square(T x) {
    return x * x;
}

// Usage
int i = square(5);        // T = int
double d = square(3.14);  // T = double
```

### typename vs class

Both keywords work identically for template parameters:

```cpp
template<typename T>  // Preferred modern style
void foo(T x);

template<class T>     // Older style, still valid
void bar(T x);
```

Convention: Use `typename` for consistency with C++11+ features.

### Multiple Type Parameters

```cpp
template<typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;
}

// C++14: return type deduction
template<typename T, typename U>
auto add(T a, U b) {
    return a + b;
}
```

### Non-Type Template Parameters

Templates can take compile-time constant values:

```cpp
template<typename T, size_t N>
class Array {
    T data[N];
public:
    size_t size() const { return N; }
};

Array<int, 10> arr;  // Array of 10 ints
```

## Template Argument Deduction

The compiler infers template arguments from function arguments:

```cpp
template<typename T>
void print(T value) {
    std::cout << value << '\n';
}

print(42);      // T deduced as int
print(3.14);    // T deduced as double
print("hello"); // T deduced as const char*
```

### Explicit Template Arguments

You can specify template arguments explicitly:

```cpp
template<typename T>
T convert(const std::string& s);

int i = convert<int>("42");
double d = convert<double>("3.14");
```

### Deduction Failure

Deduction fails when types conflict:

```cpp
template<typename T>
void foo(T a, T b);

foo(42, 3.14);  // Error: T cannot be both int and double
```

## Class Templates

### Basic Class Template

```cpp
template<typename T>
class Box {
    T value_;
public:
    Box(T val) : value_(val) {}
    
    T get() const { return value_; }
    void set(T val) { value_ = val; }
};

// Usage - must specify template arguments
Box<int> int_box(42);
Box<std::string> str_box("hello");
```

### Member Function Definitions

Inside the class:

```cpp
template<typename T>
class Container {
public:
    void add(T item) {
        // Implementation
    }
};
```

Outside the class:

```cpp
template<typename T>
class Container {
public:
    void add(T item);
};

// Must repeat template declaration
template<typename T>
void Container<T>::add(T item) {
    // Implementation
}
```

### Static Members

Each template instantiation gets its own static members:

```cpp
template<typename T>
class Counter {
    static int count;
public:
    Counter() { ++count; }
    static int get_count() { return count; }
};

// Must define static member
template<typename T>
int Counter<T>::count = 0;

Counter<int> c1, c2;      // count for int = 2
Counter<double> c3;       // count for double = 1
```

## Template Instantiation

Templates are instantiated when used:

```cpp
template<typename T>
T square(T x) { return x * x; }

int main() {
    square(5);     // Instantiates square<int>
    square(3.14);  // Instantiates square<double>
}
```

### Implicit Instantiation

Occurs when template is used:

```cpp
std::vector<int> v;  // Instantiates vector<int>
```

### Explicit Instantiation

Force instantiation at a specific point:

```cpp
template class std::vector<int>;  // Instantiate all members

template int square<int>(int);    // Instantiate specific function
```

### Extern Template (C++11)

Prevent implicit instantiation in translation unit:

```cpp
extern template class std::vector<int>;  // Don't instantiate here
```

## Two-Phase Name Lookup

Template definitions undergo two-phase compilation:

### Phase 1: Definition Time

Non-dependent names are looked up:

```cpp
void foo() { std::cout << "foo\n"; }

template<typename T>
void bar() {
    foo();  // Looked up at definition - finds ::foo
}
```

### Phase 2: Instantiation Time

Dependent names are looked up:

```cpp
template<typename T>
void process(T x) {
    x.method();  // Dependent on T, looked up at instantiation
}
```

### The typename Keyword for Dependent Names

```cpp
template<typename T>
void foo() {
    // Without typename, compiler assumes value, not type
    typename T::value_type x;  // Tell compiler this is a type
}
```

### The template Keyword for Dependent Names

```cpp
template<typename T>
void foo(T obj) {
    // Need 'template' keyword for dependent template members
    obj.template method<int>();
}
```

## Template Template Parameters

Templates that take templates as parameters:

```cpp
template<typename T, template<typename> class Container>
class Stack {
    Container<T> data;
};

// Usage
Stack<int, std::vector> stack;
```

## Default Template Arguments

### Function Templates (C++11)

```cpp
template<typename T = int>
T create_default() {
    return T{};
}

auto x = create_default<>();     // T = int (default)
auto y = create_default<double>();  // T = double
```

### Class Templates

```cpp
template<typename T, typename Allocator = std::allocator<T>>
class Vector {
    // ...
};

Vector<int> v1;                           // Uses default allocator
Vector<int, MyAllocator<int>> v2;        // Custom allocator
```

## Variadic Templates (Introduction)

Templates can accept variable number of arguments (covered in detail later):

```cpp
template<typename... Args>
void print(Args... args) {
    (std::cout << ... << args) << '\n';  // C++17 fold expression
}

print(1, 2.5, "hello", 'c');  // Works with any number of arguments
```

## Common Patterns

### Type Traits

```cpp
template<typename T>
struct is_pointer {
    static constexpr bool value = false;
};

template<typename T>
struct is_pointer<T*> {
    static constexpr bool value = true;
};

static_assert(is_pointer<int*>::value);
static_assert(!is_pointer<int>::value);
```

### CRTP (Curiously Recurring Template Pattern)

```cpp
template<typename Derived>
class Base {
public:
    void interface() {
        static_cast<Derived*>(this)->implementation();
    }
};

class Derived : public Base<Derived> {
public:
    void implementation() {
        std::cout << "Derived implementation\n";
    }
};
```

## Template Compilation Model

### Inclusion Model (Standard)

Template definitions must be visible at point of use:

```cpp
// header.h
template<typename T>
class MyClass {
    void method();  // Declaration
};

template<typename T>
void MyClass<T>::method() {  // Definition in header
    // ...
}
```

### Separation Model (Removed in C++11)

The `export` keyword was removed. Templates require visible definitions.

## Best Practices

1. **Put template definitions in headers**
   - Templates need to be visible at instantiation point

2. **Use meaningful template parameter names**
   ```cpp
   template<typename Value, typename Allocator>  // Good
   template<typename T, typename U>              // Less clear
   ```

3. **Constrain templates where possible**
   - Use concepts (C++20) or SFINAE
   - Document requirements in comments

4. **Avoid unnecessary template parameters**
   ```cpp
   // Bad: N is redundant
   template<typename T, size_t N>
   void process(std::array<T, N>& arr);
   
   // Good: Deduce N
   template<typename T, size_t N>
   void process(std::array<T, N>& arr);
   ```

5. **Consider compilation time**
   - Templates increase compilation time
   - Use extern templates for common instantiations
   - Forward declare when possible

## Common Pitfalls

### 1. Forgetting typename for Dependent Types

```cpp
template<typename T>
void foo() {
    T::value_type x;  // Error: need typename
    typename T::value_type y;  // Correct
}
```

### 2. Two-Phase Lookup Issues

```cpp
template<typename T>
void foo() {
    bar();  // Must be visible at template definition
}

void bar();  // Too late!
```

### 3. Template Bloat

Each instantiation generates code:

```cpp
template<typename T>
void process(T x) { /* large function */ }

process(1);      // Instantiates for int
process(2.0);    // Instantiates for double
// Result: code duplication
```

## Summary

Key concepts covered:
- Function and class templates
- Template parameters: type and non-type
- Template argument deduction
- Template instantiation
- Two-phase name lookup
- Default template arguments
- Template compilation model
- Best practices and common pitfalls

## Next Steps

In the next chapter, we will explore template specialization, including full and partial specialization, and tag dispatching techniques.
