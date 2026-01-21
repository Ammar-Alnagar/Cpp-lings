# Chapter 5: Template Metaprogramming

## Introduction to Template Metaprogramming

Template metaprogramming (TMP) uses C++ templates to perform computations at compile time. It's a form of compile-time computation where templates serve as a functional programming language executed by the compiler.

## Why Template Metaprogramming?

### Benefits

1. **Zero Runtime Cost**: Computations happen at compile time
2. **Type Safety**: Errors caught at compile time
3. **Optimization**: Compiler can optimize based on compile-time knowledge
4. **Code Generation**: Generate specialized code for each use case

### Use Cases

- Compile-time calculations
- Type manipulation and generation
- Expression templates
- Policy-based design
- Compile-time loops and recursion

## Compile-Time Computation

### Classic Example: Factorial

```cpp
// Recursive template metaprogram
template<int N>
struct Factorial {
    static constexpr int value = N * Factorial<N - 1>::value;
};

// Base case (specialization)
template<>
struct Factorial<0> {
    static constexpr int value = 1;
};

// Usage
constexpr int fact5 = Factorial<5>::value;  // Computed at compile time
static_assert(Factorial<5>::value == 120);
```

### Modern Alternative: constexpr Functions

```cpp
// C++11 and later: constexpr functions
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

constexpr int fact5 = factorial(5);  // Compile-time
int runtime_fact = factorial(x);     // Runtime if x not constexpr
```

## Type Computation

### Type Selection

```cpp
// Select type based on condition
template<bool Condition, typename TrueType, typename FalseType>
struct conditional {
    using type = TrueType;
};

template<typename TrueType, typename FalseType>
struct conditional<false, TrueType, FalseType> {
    using type = FalseType;
};

// Helper alias
template<bool B, typename T, typename F>
using conditional_t = typename conditional<B, T, F>::type;

// Usage
using MyType = conditional_t<sizeof(int) == 4, int, long>;
```

### Type Lists

```cpp
// Compile-time list of types
template<typename... Types>
struct TypeList {};

// Get size of type list
template<typename List>
struct Length;

template<typename... Types>
struct Length<TypeList<Types...>> {
    static constexpr size_t value = sizeof...(Types);
};

// Get element at index
template<size_t N, typename List>
struct At;

template<typename Head, typename... Tail>
struct At<0, TypeList<Head, Tail...>> {
    using type = Head;
};

template<size_t N, typename Head, typename... Tail>
struct At<N, TypeList<Head, Tail...>> {
    using type = typename At<N - 1, TypeList<Tail...>>::type;
};

// Usage
using MyList = TypeList<int, double, char>;
static_assert(Length<MyList>::value == 3);
using SecondType = typename At<1, MyList>::type;  // double
```

### Type Transformations

```cpp
// Apply transformation to all types in list
template<template<typename> class Transform, typename List>
struct Map;

template<template<typename> class Transform, typename... Types>
struct Map<Transform, TypeList<Types...>> {
    using type = TypeList<Transform<Types>...>;
};

// Example transformation
template<typename T>
using AddPointer = T*;

using Original = TypeList<int, double, char>;
using Pointers = typename Map<AddPointer, Original>::type;
// Pointers = TypeList<int*, double*, char*>
```

## Compile-Time Recursion

### Sum of Integers

```cpp
template<int... Values>
struct Sum;

template<int First, int... Rest>
struct Sum<First, Rest...> {
    static constexpr int value = First + Sum<Rest...>::value;
};

template<>
struct Sum<> {
    static constexpr int value = 0;
};

static_assert(Sum<1, 2, 3, 4, 5>::value == 15);
```

### Greatest Common Divisor

```cpp
template<int A, int B>
struct GCD {
    static constexpr int value = GCD<B, A % B>::value;
};

template<int A>
struct GCD<A, 0> {
    static constexpr int value = A;
};

static_assert(GCD<48, 18>::value == 6);
```

### Compile-Time Fibonacci

```cpp
template<int N>
struct Fibonacci {
    static constexpr int value = 
        Fibonacci<N - 1>::value + Fibonacci<N - 2>::value;
};

template<>
struct Fibonacci<0> {
    static constexpr int value = 0;
};

template<>
struct Fibonacci<1> {
    static constexpr int value = 1;
};

static_assert(Fibonacci<10>::value == 55);
```

## Compile-Time Strings

### Fixed String

```cpp
template<char... Chars>
struct String {
    static constexpr char value[sizeof...(Chars) + 1] = {Chars..., '\0'};
    static constexpr size_t length = sizeof...(Chars);
};

// Usage requires user-defined literal or macro
// C++20 provides better alternatives
```

### String Manipulation

```cpp
// Concatenate strings
template<typename S1, typename S2>
struct Concat;

template<char... C1, char... C2>
struct Concat<String<C1...>, String<C2...>> {
    using type = String<C1..., C2...>;
};

// Reverse string
template<typename S>
struct Reverse;

template<char First, char... Rest>
struct Reverse<String<First, Rest...>> {
    using RestReversed = typename Reverse<String<Rest...>>::type;
    // Append First to end... (implementation omitted for brevity)
};
```

## Expression Templates

Expression templates enable efficient evaluation of complex expressions.

### Problem: Temporary Objects

```cpp
// Standard approach creates temporaries
Vector a, b, c, d;
Vector result = a + b + c + d;
// Creates: temp1 = a + b
//          temp2 = temp1 + c
//          temp3 = temp2 + d
//          result = temp3
```

### Solution: Expression Templates

```cpp
// Expression template
template<typename Left, typename Right>
struct VectorSum {
    const Left& left;
    const Right& right;
    
    VectorSum(const Left& l, const Right& r) : left(l), right(r) {}
    
    double operator[](size_t i) const {
        return left[i] + right[i];
    }
};

class Vector {
    std::vector<double> data;
    
public:
    double operator[](size_t i) const { return data[i]; }
    
    // Return expression template, not evaluated yet
    template<typename E>
    VectorSum<Vector, E> operator+(const E& expr) const {
        return VectorSum<Vector, E>(*this, expr);
    }
    
    // Assignment evaluates expression
    template<typename E>
    Vector& operator=(const E& expr) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = expr[i];
        }
        return *this;
    }
};

// Usage: No temporaries created!
// result = a + b + c + d;
// Creates single expression tree, evaluated once
```

## Policy-Based Design

Combine policies using templates for flexible, reusable code.

```cpp
// Storage policies
template<typename T>
class VectorStorage {
    std::vector<T> data_;
public:
    T& get(size_t i) { return data_[i]; }
    void resize(size_t n) { data_.resize(n); }
};

template<typename T>
class ArrayStorage {
    std::array<T, 1024> data_;
public:
    T& get(size_t i) { return data_[i]; }
    void resize(size_t n) { /* no-op for arrays */ }
};

// Checking policies
struct BoundsChecking {
    template<typename Container>
    static void check(const Container& c, size_t i) {
        if (i >= c.size()) throw std::out_of_range("Index out of bounds");
    }
};

struct NoBoundsChecking {
    template<typename Container>
    static void check(const Container&, size_t) { /* no check */ }
};

// Combine policies
template<typename T,
         template<typename> class Storage = VectorStorage,
         class Checking = NoBoundsChecking>
class SmartVector : private Storage<T> {
public:
    T& operator[](size_t i) {
        Checking::check(*this, i);
        return this->get(i);
    }
};

// Usage
SmartVector<int> v1;                                    // Default: vector storage, no checking
SmartVector<int, ArrayStorage> v2;                      // Array storage
SmartVector<int, VectorStorage, BoundsChecking> v3;    // With bounds checking
```

## Static Assertions and Constraints

### Compile-Time Validation

```cpp
template<typename T>
class NumericVector {
    static_assert(std::is_arithmetic_v<T>,
        "NumericVector requires arithmetic type");
    
    std::vector<T> data;
};

// Won't compile
// NumericVector<std::string> v;  // Error: static assertion failed
```

### Concept Emulation (Pre-C++20)

```cpp
// Require concept
template<typename T>
struct requires_arithmetic {
    static_assert(std::is_arithmetic_v<T>,
        "Type must be arithmetic");
    using type = T;
};

template<typename T>
class Calculator {
    using checked = typename requires_arithmetic<T>::type;
    // ...
};
```

## Template Template Parameters

Templates that take templates as parameters.

### Container Adapters

```cpp
// Generic stack that works with any sequence container
template<typename T,
         template<typename, typename> class Container = std::vector>
class Stack {
    Container<T, std::allocator<T>> container;
    
public:
    void push(const T& value) {
        container.push_back(value);
    }
    
    void pop() {
        container.pop_back();
    }
    
    T& top() {
        return container.back();
    }
};

// Usage
Stack<int> s1;                          // Uses std::vector
Stack<int, std::deque> s2;              // Uses std::deque
Stack<int, std::list> s3;               // Uses std::list
```

## Metafunction Classes

Classes used as compile-time functions.

```cpp
// Metafunction: Takes type, returns type
template<typename T>
struct AddPointer {
    using type = T*;
};

template<typename T>
struct AddConst {
    using type = const T;
};

// Compose metafunctions
template<typename T,
         template<typename> class... Transforms>
struct Apply;

template<typename T>
struct Apply<T> {
    using type = T;
};

template<typename T,
         template<typename> class First,
         template<typename> class... Rest>
struct Apply<T, First, Rest...> {
    using type = typename Apply<typename First<T>::type, Rest...>::type;
};

// Usage
using Result = typename Apply<int, AddPointer, AddConst>::type;
// Result = const int*
```

## Compile-Time Loops

### Unrolling with Recursion

```cpp
// Compile-time for loop
template<int Start, int End, typename Func>
struct ForLoop {
    static void apply(Func f) {
        f.template operator()<Start>();
        ForLoop<Start + 1, End, Func>::apply(f);
    }
};

template<int End, typename Func>
struct ForLoop<End, End, Func> {
    static void apply(Func) {}
};

// Usage
struct Printer {
    template<int I>
    void operator()() const {
        std::cout << I << " ";
    }
};

ForLoop<0, 5, Printer>::apply(Printer{});  // Prints: 0 1 2 3 4
```

### Modern Alternative: Fold Expressions

```cpp
template<typename Func, size_t... Is>
void for_each_index(Func f, std::index_sequence<Is...>) {
    (f.template operator()<Is>(), ...);
}

template<size_t N, typename Func>
void compile_time_for(Func f) {
    for_each_index(f, std::make_index_sequence<N>{});
}
```

## Practical Applications

### Unit Conversion

```cpp
// Compile-time dimensional analysis
template<int M, int L, int T>  // Mass, Length, Time exponents
struct Dimension {};

using Scalar = Dimension<0, 0, 0>;
using Length = Dimension<0, 1, 0>;
using Time = Dimension<0, 0, 1>;
using Velocity = Dimension<0, 1, -1>;  // L/T
using Acceleration = Dimension<0, 1, -2>;  // L/T^2

template<typename D, typename T = double>
class Quantity {
    T value;
public:
    explicit Quantity(T v) : value(v) {}
    T get() const { return value; }
    
    // Only allow addition of same dimensions
    Quantity operator+(Quantity other) const {
        return Quantity(value + other.value);
    }
};

// Multiplication produces correct dimensions
template<int M1, int L1, int T1, int M2, int L2, int T2, typename T>
Quantity<Dimension<M1+M2, L1+L2, T1+T2>, T>
operator*(Quantity<Dimension<M1, L1, T1>, T> a,
         Quantity<Dimension<M2, L2, T2>, T> b) {
    return Quantity<Dimension<M1+M2, L1+L2, T1+T2>, T>(a.get() * b.get());
}

// Usage
Quantity<Length> distance(100.0);
Quantity<Time> time(10.0);
Quantity<Velocity> speed = distance / time;  // Type-safe!
// distance + time;  // Compile error: incompatible dimensions
```

### Compile-Time State Machines

```cpp
// States
struct StateA {};
struct StateB {};
struct StateC {};

// Events
struct Event1 {};
struct Event2 {};

// Transition table
template<typename State, typename Event>
struct Transition;

template<>
struct Transition<StateA, Event1> {
    using next_state = StateB;
};

template<>
struct Transition<StateB, Event2> {
    using next_state = StateC;
};

// State machine
template<typename CurrentState>
class StateMachine {
    CurrentState state;
    
public:
    template<typename Event>
    auto handle_event(Event e) {
        using NextState = typename Transition<CurrentState, Event>::next_state;
        return StateMachine<NextState>{};
    }
};
```

### Type-Safe Format Strings

```cpp
// Compile-time format string parser
template<char... Chars>
struct FormatString {
    static constexpr size_t arg_count = /* count %d, %s, etc. */;
    
    template<typename... Args>
    static void check_args() {
        static_assert(sizeof...(Args) == arg_count,
            "Wrong number of arguments for format string");
    }
};

// Usage with user-defined literal
template<typename CharT, CharT... Chars>
constexpr FormatString<Chars...> operator""_fmt() {
    return {};
}

// auto fmt = "Hello %s, you are %d years old"_fmt;
// fmt.print("Alice", 30);  // OK
// fmt.print("Bob");  // Compile error: wrong number of args
```

## Advanced Techniques

### CRTP for Static Polymorphism

```cpp
template<typename Derived>
class Base {
public:
    void interface() {
        static_cast<Derived*>(this)->implementation();
    }
    
    void common_code() {
        // Shared code
    }
};

class Derived1 : public Base<Derived1> {
public:
    void implementation() {
        std::cout << "Derived1\n";
    }
};

class Derived2 : public Base<Derived2> {
public:
    void implementation() {
        std::cout << "Derived2\n";
    }
};
```

### Mixin Classes

```cpp
template<typename... Mixins>
class Object : public Mixins... {
public:
    using Mixins::Mixins...;  // Inherit constructors
};

struct Printable {
    void print() const { std::cout << "Object\n"; }
};

struct Serializable {
    std::string serialize() const { return "serialized"; }
};

using MyObject = Object<Printable, Serializable>;
```

## Performance Considerations

### Compilation Time

```cpp
// BAD: Deep template recursion
template<int N>
struct Slow {
    static constexpr int value = Slow<N-1>::value + 1;
};

// BETTER: Use constexpr functions
constexpr int fast(int n) {
    return n;
}
```

### Template Instantiation Limits

Most compilers limit recursion depth:
- Default: ~900-1024 levels
- Can increase with `-ftemplate-depth=N`

### Binary Size

Template instantiations increase binary size. Consider:
- Type erasure for non-performance-critical code
- Extern templates to reduce duplicates

## Best Practices

1. **Prefer constexpr over TMP**
   - Easier to read and write
   - Better error messages
   - More efficient compilation

2. **Use concepts (C++20)**
   - Clearer constraints
   - Better error messages

3. **Document metaprograms**
   - Complex TMP is hard to understand
   - Explain the "why" not just the "what"

4. **Provide type aliases**
```cpp
template<typename T>
using pointer_t = typename AddPointer<T>::type;
```

5. **Test compile-time code**
```cpp
static_assert(Factorial<5>::value == 120);
```

## Summary

Key concepts covered:
- Compile-time computation with templates
- Type manipulation and generation
- Type lists and metafunctions
- Compile-time recursion
- Expression templates
- Policy-based design
- Template template parameters
- Practical applications
- Modern alternatives (constexpr, concepts)

## Next Steps

In Chapter 6, we would explore C++20 Concepts in depth, providing modern alternatives to traditional template metaprogramming techniques.

Practice implementing metaprograms, but remember: prefer simpler solutions (constexpr functions, concepts) when they suffice!
