# Chapter 4: SFINAE and Template Constraints

## Introduction to SFINAE

SFINAE stands for "Substitution Failure Is Not An Error". It's a fundamental principle in C++ template metaprogramming that enables compile-time selection of template specializations.

## The SFINAE Principle

When the compiler substitutes template parameters and encounters an invalid type or expression, instead of producing an error, it removes that template from the overload set.

### Basic Example

```cpp
template<typename T>
typename T::value_type get_value(T container) {
    return container[0];
}

template<typename T>
T get_value(T value) {
    return value;
}

// Usage
std::vector<int> vec{1, 2, 3};
int x = 42;

auto v1 = get_value(vec);  // Calls first overload (has value_type)
auto v2 = get_value(x);    // Calls second overload (int has no value_type, SFINAE!)
```

## Why SFINAE Matters

### Problem Without SFINAE

```cpp
template<typename T>
void process(T value) {
    // Want different behavior for integers vs others
    // How to select at compile time?
}
```

### Solution With SFINAE

```cpp
// Only enabled for integral types
template<typename T>
std::enable_if_t<std::is_integral_v<T>, void>
process(T value) {
    std::cout << "Processing integer: " << value << "\n";
}

// Only enabled for floating point types
template<typename T>
std::enable_if_t<std::is_floating_point_v<T>, void>
process(T value) {
    std::cout << "Processing float: " << value << "\n";
}
```

## std::enable_if

The primary tool for SFINAE-based overload control.

### Basic Syntax

```cpp
template<bool Condition, typename T = void>
struct enable_if {
    // No type member if Condition is false
};

template<typename T>
struct enable_if<true, T> {
    using type = T;  // Has type member if Condition is true
};

// Helper alias
template<bool B, typename T = void>
using enable_if_t = typename enable_if<B, T>::type;
```

### Usage Patterns

#### 1. Return Type SFINAE

```cpp
// Enable only for integral types
template<typename T>
std::enable_if_t<std::is_integral_v<T>, T>
double_value(T x) {
    return x * 2;
}

// Enable only for floating point
template<typename T>
std::enable_if_t<std::is_floating_point_v<T>, T>
double_value(T x) {
    return x * 2.0;
}
```

#### 2. Template Parameter SFINAE

```cpp
template<typename T,
         typename = std::enable_if_t<std::is_integral_v<T>>>
void print_int(T value) {
    std::cout << "Integer: " << value << "\n";
}
```

#### 3. Function Parameter SFINAE

```cpp
template<typename T>
void print(T value,
          std::enable_if_t<std::is_integral_v<T>>* = nullptr) {
    std::cout << "Integer: " << value << "\n";
}
```

#### 4. Non-Type Template Parameter SFINAE

```cpp
template<typename T>
void print(T value,
          std::enable_if_t<std::is_integral_v<T>, int> = 0) {
    std::cout << "Integer: " << value << "\n";
}
```

## Type Traits

Type traits are the foundation of SFINAE-based metaprogramming.

### Standard Type Traits

#### Primary Type Categories

```cpp
std::is_integral_v<int>          // true
std::is_floating_point_v<double> // true
std::is_array_v<int[]>           // true
std::is_pointer_v<int*>          // true
std::is_reference_v<int&>        // true
std::is_class_v<std::string>     // true
std::is_function_v<int()>        // true
std::is_void_v<void>             // true
```

#### Composite Type Categories

```cpp
std::is_arithmetic_v<T>    // integral or floating point
std::is_fundamental_v<T>   // arithmetic or void
std::is_scalar_v<T>        // arithmetic, pointer, enum, member pointer, nullptr_t
std::is_object_v<T>        // not function, reference, or void
std::is_compound_v<T>      // array, function, pointer, reference, class, union, enum
```

#### Type Properties

```cpp
std::is_const_v<const int>       // true
std::is_volatile_v<volatile int> // true
std::is_signed_v<int>            // true
std::is_unsigned_v<unsigned>     // true
```

#### Type Relationships

```cpp
std::is_same_v<int, int>              // true
std::is_base_of_v<Base, Derived>      // true if Derived inherits from Base
std::is_convertible_v<From, To>       // true if From convertible to To
```

#### Type Modifications

```cpp
std::remove_const_t<const int>        // int
std::remove_reference_t<int&>         // int
std::remove_pointer_t<int*>           // int
std::add_const_t<int>                 // const int
std::add_pointer_t<int>               // int*
std::decay_t<int[10]>                 // int*
```

### Custom Type Traits

```cpp
// Check if type has member function
template<typename T, typename = void>
struct has_size : std::false_type {};

template<typename T>
struct has_size<T, std::void_t<decltype(std::declval<T>().size())>>
    : std::true_type {};

template<typename T>
inline constexpr bool has_size_v = has_size<T>::value;

// Usage
static_assert(has_size_v<std::vector<int>>);
static_assert(!has_size_v<int>);
```

## Detection Idiom

Modern approach to detecting members and operations.

### Basic Detection

```cpp
template<typename T>
using has_value_type = typename T::value_type;

template<typename T>
constexpr bool has_value_type_v = 
    std::experimental::is_detected_v<has_value_type, T>;
```

### Member Function Detection

```cpp
// Detect if type has begin() member
template<typename T>
using has_begin = decltype(std::declval<T>().begin());

template<typename T>
constexpr bool is_container_v = 
    std::experimental::is_detected_v<has_begin, T>;
```

### Custom Implementation

```cpp
namespace detail {
    template<typename...>
    using void_t = void;
    
    template<typename Default, typename, template<typename...> class, typename...>
    struct detector {
        using value_t = std::false_type;
        using type = Default;
    };
    
    template<typename Default, template<typename...> class Op, typename... Args>
    struct detector<Default, void_t<Op<Args...>>, Op, Args...> {
        using value_t = std::true_type;
        using type = Op<Args...>;
    };
}

template<template<typename...> class Op, typename... Args>
using is_detected = typename detail::detector<void, void, Op, Args...>::value_t;

template<template<typename...> class Op, typename... Args>
constexpr bool is_detected_v = is_detected<Op, Args...>::value;
```

## Expression SFINAE

SFINAE works with arbitrary expressions, not just types.

### Basic Expression SFINAE

```cpp
// Only enable if T supports operator+
template<typename T>
auto add(T a, T b) -> decltype(a + b) {
    return a + b;
}

// Won't compile for types without operator+
// But won't cause error if overload exists
```

### Complex Expressions

```cpp
// Enable if T has member function foo() returning int
template<typename T>
auto call_foo(T& obj) 
    -> std::enable_if_t<
        std::is_same_v<decltype(obj.foo()), int>, 
        int
    >
{
    return obj.foo();
}
```

## Tag Dispatching

Alternative to SFINAE using overload resolution.

### Basic Tag Dispatching

```cpp
// Tags
struct integral_tag {};
struct floating_point_tag {};
struct other_tag {};

// Tag selection
template<typename T>
using category_t = std::conditional_t<
    std::is_integral_v<T>, integral_tag,
    std::conditional_t<
        std::is_floating_point_v<T>, floating_point_tag,
        other_tag
    >
>;

// Implementations
template<typename T>
void process_impl(T value, integral_tag) {
    std::cout << "Integer: " << value << "\n";
}

template<typename T>
void process_impl(T value, floating_point_tag) {
    std::cout << "Float: " << value << "\n";
}

template<typename T>
void process_impl(T value, other_tag) {
    std::cout << "Other: " << value << "\n";
}

// Dispatcher
template<typename T>
void process(T value) {
    process_impl(value, category_t<T>{});
}
```

### Iterator Tag Dispatching

```cpp
// STL uses this for algorithms
template<typename Iterator>
void advance_impl(Iterator& it, int n, std::random_access_iterator_tag) {
    it += n;  // O(1) for random access
}

template<typename Iterator>
void advance_impl(Iterator& it, int n, std::input_iterator_tag) {
    for (int i = 0; i < n; ++i) {
        ++it;  // O(n) for input iterators
    }
}

template<typename Iterator>
void advance(Iterator& it, int n) {
    advance_impl(it, n, 
        typename std::iterator_traits<Iterator>::iterator_category{});
}
```

## if constexpr (C++17)

Modern alternative to SFINAE for compile-time branching.

### Basic Usage

```cpp
template<typename T>
void process(T value) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "Integer: " << value << "\n";
    } else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "Float: " << value << "\n";
    } else {
        std::cout << "Other\n";
    }
}
```

### Advantages Over SFINAE

```cpp
// SFINAE: Need separate functions
template<typename T>
std::enable_if_t<std::is_integral_v<T>, T>
double_value(T x) { return x * 2; }

template<typename T>
std::enable_if_t<std::is_floating_point_v<T>, T>
double_value(T x) { return x * 2.0; }

// if constexpr: Single function
template<typename T>
T double_value(T x) {
    if constexpr (std::is_integral_v<T>) {
        return x * 2;
    } else {
        return x * 2.0;
    }
}
```

## Concepts (C++20)

Modern, clean alternative to SFINAE.

### Basic Concepts

```cpp
// Define a concept
template<typename T>
concept Integral = std::is_integral_v<T>;

template<typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

// Use in template
template<Integral T>
void process(T value) {
    std::cout << "Integer: " << value << "\n";
}

template<FloatingPoint T>
void process(T value) {
    std::cout << "Float: " << value << "\n";
}
```

### Concept Syntax

```cpp
// Requires clause
template<typename T>
    requires std::is_integral_v<T>
void func(T x) {}

// Trailing requires
template<typename T>
void func(T x) requires std::is_integral_v<T> {}

// Abbreviated function template
void func(Integral auto x) {}
```

### Complex Concepts

```cpp
template<typename T>
concept Container = requires(T t) {
    typename T::value_type;
    typename T::iterator;
    { t.begin() } -> std::same_as<typename T::iterator>;
    { t.end() } -> std::same_as<typename T::iterator>;
    { t.size() } -> std::convertible_to<std::size_t>;
};

template<Container C>
void process_container(C& container) {
    for (auto& elem : container) {
        // ...
    }
}
```

### Standard Concepts

```cpp
#include <concepts>

std::same_as<T, U>
std::derived_from<Derived, Base>
std::convertible_to<From, To>
std::integral<T>
std::floating_point<T>
std::copyable<T>
std::movable<T>
std::default_initializable<T>
```

## Practical Examples

### Generic Container Operations

```cpp
// Works with any container having begin/end
template<typename Container>
auto sum_elements(const Container& c)
    -> std::enable_if_t<
        std::is_detected_v<has_begin, Container>,
        typename Container::value_type
    >
{
    typename Container::value_type sum{};
    for (const auto& elem : c) {
        sum += elem;
    }
    return sum;
}
```

### Generic Serialization

```cpp
// Serialize types with serialize() member
template<typename T>
auto serialize(const T& obj)
    -> decltype(obj.serialize(), std::string{})
{
    return obj.serialize();
}

// Fallback for types without serialize()
template<typename T>
std::enable_if_t<!is_detected_v<has_serialize, T>, std::string>
serialize(const T& obj) {
    return default_serialize(obj);
}
```

### Type-Safe Numeric Conversions

```cpp
template<typename To, typename From>
std::enable_if_t<
    std::is_arithmetic_v<From> && std::is_arithmetic_v<To> &&
    (sizeof(To) >= sizeof(From)) &&
    (std::is_signed_v<To> == std::is_signed_v<From>),
    To
>
safe_cast(From value) {
    return static_cast<To>(value);
}
```

## Common Patterns

### Enable Member Functions

```cpp
template<typename T>
class Optional {
    T value_;
    bool has_value_;
    
public:
    // Only enable operator* for copyable types
    template<typename U = T>
    std::enable_if_t<std::is_copy_constructible_v<U>, U>
    operator*() const {
        return value_;
    }
};
```

### Conditional Base Classes

```cpp
template<typename T, bool Optimize>
class Container;

// Optimized version
template<typename T>
class Container<T, true> : public OptimizedBase<T> {
    // ...
};

// Standard version
template<typename T>
class Container<T, false> : public StandardBase<T> {
    // ...
};
```

## Best Practices

### 1. Prefer Concepts (C++20)

```cpp
// Old way (SFINAE)
template<typename T>
std::enable_if_t<std::is_integral_v<T>, void>
func(T x);

// New way (Concepts)
template<std::integral T>
void func(T x);
```

### 2. Use if constexpr for Branching

```cpp
// Instead of multiple SFINAE overloads
template<typename T>
void process(T x) {
    if constexpr (condition) {
        // ...
    } else {
        // ...
    }
}
```

### 3. Document Constraints

```cpp
/// Requires: T must be copyable and have operator<
template<typename T>
void sort(std::vector<T>& vec);
```

### 4. Provide Clear Error Messages

```cpp
template<typename T>
void func(T x) {
    static_assert(std::is_arithmetic_v<T>,
        "T must be an arithmetic type");
}
```

## Summary

Key concepts covered:
- SFINAE principle and motivation
- std::enable_if and usage patterns
- Type traits for compile-time introspection
- Detection idiom
- Expression SFINAE
- Tag dispatching
- if constexpr (C++17)
- Concepts (C++20)
- Practical applications

## Next Steps

In Chapter 5, we will explore template metaprogramming, using templates for compile-time computation and advanced type manipulation.
