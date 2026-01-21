# Chapter 3: Variadic Templates

## Introduction to Variadic Templates

Variadic templates allow functions and classes to accept any number of template arguments. This powerful feature enables type-safe generic programming with variable argument lists.

## Basic Syntax

```cpp
// Function with variable number of type parameters
template<typename... Args>
void func(Args... args);

// Class with variable number of type parameters
template<typename... Types>
class Tuple;
```

The ellipsis (`...`) appears in three contexts:
1. In template parameter list: `typename... Args` (parameter pack)
2. In function parameter list: `Args... args` (function parameter pack)
3. In expressions: `args...` (pack expansion)

## Simple Variadic Function

```cpp
#include <iostream>

// Base case: no arguments
void print() {
    std::cout << "\n";
}

// Recursive case: process first argument, recurse on rest
template<typename T, typename... Args>
void print(T first, Args... rest) {
    std::cout << first << " ";
    print(rest...);  // Recursive call
}

// Usage
print(1, 2.5, "hello", 'x');  // Output: 1 2.5 hello x
```

## Parameter Pack Expansion

Pack expansion applies a pattern to each element in the pack.

### Basic Expansion

```cpp
template<typename... Args>
void forward_to_func(Args... args) {
    // Expand: func(arg1, arg2, arg3, ...)
    func(args...);
}

template<typename... Args>
void forward_with_transform(Args... args) {
    // Expand with transformation: func(transform(arg1), transform(arg2), ...)
    func(transform(args)...);
}
```

### Multiple Expansions

```cpp
template<typename... Args>
void multiple_calls(Args... args) {
    // Call func for each argument separately
    (func(args), ...);  // C++17 fold expression
    
    // Or using initializer list trick (pre-C++17)
    int dummy[] = { (func(args), 0)... };
    (void)dummy;  // Suppress unused warning
}
```

## Fold Expressions (C++17)

Fold expressions simplify parameter pack operations.

### Unary Folds

```cpp
// Left fold: (((arg1 op arg2) op arg3) op ...)
template<typename... Args>
auto sum_left(Args... args) {
    return (... + args);  // Left unary fold
}

// Right fold: (arg1 op (arg2 op (arg3 op ...)))
template<typename... Args>
auto sum_right(Args... args) {
    return (args + ...);  // Right unary fold
}
```

### Binary Folds

```cpp
// Left fold with init: (((init op arg1) op arg2) op ...)
template<typename... Args>
auto sum_with_init(Args... args) {
    return (0 + ... + args);  // Binary left fold
}

// Right fold with init: (arg1 op (arg2 op (... op init)))
template<typename... Args>
auto sum_with_init_right(Args... args) {
    return (args + ... + 0);  // Binary right fold
}
```

### Common Fold Operations

```cpp
// Logical operations
template<typename... Args>
bool all(Args... args) {
    return (args && ...);  // true if all args are true
}

template<typename... Args>
bool any(Args... args) {
    return (args || ...);  // true if any arg is true
}

// Print all arguments
template<typename... Args>
void print_all(Args... args) {
    (std::cout << ... << args) << '\n';
}

// Call function on all arguments
template<typename Func, typename... Args>
void apply_to_all(Func f, Args... args) {
    (f(args), ...);  // Comma fold
}
```

## sizeof... Operator

Get the number of elements in a parameter pack:

```cpp
template<typename... Args>
constexpr size_t count_args(Args... args) {
    return sizeof...(Args);  // Number of types
    // or sizeof...(args);   // Number of arguments (same value)
}

static_assert(count_args(1, 2, 3, 4, 5) == 5);
```

## Variadic Class Templates

### Basic Variadic Class

```cpp
template<typename... Types>
class Tuple;

// Empty tuple specialization
template<>
class Tuple<> {};

// Recursive definition
template<typename Head, typename... Tail>
class Tuple<Head, Tail...> {
    Head head_;
    Tuple<Tail...> tail_;
    
public:
    Tuple(Head h, Tail... t) : head_(h), tail_(t...) {}
    
    Head& head() { return head_; }
    Tuple<Tail...>& tail() { return tail_; }
};
```

### Accessing Elements

```cpp
// Get element at index (using recursion)
template<size_t Index, typename Head, typename... Tail>
struct TupleElement {
    using type = typename TupleElement<Index - 1, Tail...>::type;
};

template<typename Head, typename... Tail>
struct TupleElement<0, Head, Tail...> {
    using type = Head;
};

// Get function
template<size_t Index, typename... Types>
auto get(Tuple<Types...>& tuple) {
    if constexpr (Index == 0) {
        return tuple.head();
    } else {
        return get<Index - 1>(tuple.tail());
    }
}
```

## Perfect Forwarding with Variadic Templates

```cpp
template<typename... Args>
void wrapper(Args&&... args) {
    // Perfect forward all arguments
    actual_function(std::forward<Args>(args)...);
}

// Factory function
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
```

## Variadic Templates with Indices

### Index Sequence

```cpp
// Generate compile-time index sequence
template<typename... Args>
void print_with_indices(Args... args) {
    size_t index = 0;
    ((std::cout << index++ << ": " << args << "\n"), ...);
}

// Using std::index_sequence
template<typename Tuple, size_t... Is>
void print_tuple_impl(const Tuple& t, std::index_sequence<Is...>) {
    ((std::cout << std::get<Is>(t) << " "), ...);
}

template<typename... Args>
void print_tuple(const std::tuple<Args...>& t) {
    print_tuple_impl(t, std::index_sequence_for<Args...>{});
    std::cout << "\n";
}
```

## Type Manipulation with Variadic Templates

### Type List Operations

```cpp
// Count occurrences of type T in pack
template<typename T, typename... Types>
struct count_type {
    static constexpr size_t value = 
        (std::is_same_v<T, Types> + ...);
};

// Check if type T is in pack
template<typename T, typename... Types>
struct contains_type {
    static constexpr bool value = 
        (std::is_same_v<T, Types> || ...);
};

// Get first type in pack
template<typename... Types>
struct first_type;

template<typename Head, typename... Tail>
struct first_type<Head, Tail...> {
    using type = Head;
};
```

### Concatenate Type Lists

```cpp
template<typename...>
struct type_list {};

// Concatenate two type lists
template<typename List1, typename List2>
struct concat;

template<typename... Ts, typename... Us>
struct concat<type_list<Ts...>, type_list<Us...>> {
    using type = type_list<Ts..., Us...>;
};
```

## Practical Applications

### Generic print function

```cpp
template<typename... Args>
void println(Args&&... args) {
    ((std::cout << std::forward<Args>(args) << " "), ...);
    std::cout << "\n";
}
```

### Variadic sum

```cpp
template<typename... Args>
auto sum(Args... args) {
    return (args + ...);
}

// With type safety
template<typename T, typename... Args>
T typed_sum(T first, Args... rest) {
    return (first + ... + rest);
}
```

### Apply function to all arguments

```cpp
template<typename Func, typename... Args>
void for_each_arg(Func f, Args&&... args) {
    (f(std::forward<Args>(args)), ...);
}

// Usage
for_each_arg([](auto x) { std::cout << x << "\n"; }, 1, 2.5, "hello");
```

### Variadic min/max

```cpp
template<typename T, typename... Args>
T min(T first, Args... rest) {
    T result = first;
    ((result = result < rest ? result : rest), ...);
    return result;
}

template<typename T, typename... Args>
T max(T first, Args... rest) {
    T result = first;
    ((result = result > rest ? result : rest), ...);
    return result;
}
```

## Advanced Patterns

### Visitor Pattern

```cpp
template<typename... Visitors>
struct overloaded : Visitors... {
    using Visitors::operator()...;
};

template<typename... Visitors>
overloaded(Visitors...) -> overloaded<Visitors...>;

// Usage with std::variant
std::variant<int, double, std::string> v = 42;
std::visit(overloaded{
    [](int i) { std::cout << "int: " << i << "\n"; },
    [](double d) { std::cout << "double: " << d << "\n"; },
    [](const std::string& s) { std::cout << "string: " << s << "\n"; }
}, v);
```

### Compile-time string concatenation

```cpp
template<char... Chars>
struct string_literal {
    static constexpr char value[] = {Chars..., '\0'};
};

template<char... C1, char... C2>
constexpr auto operator+(string_literal<C1...>, string_literal<C2...>) {
    return string_literal<C1..., C2...>{};
}
```

## Best Practices

1. **Use fold expressions when possible (C++17+)**
   - Cleaner and more efficient than recursion
   
2. **Prefer constexpr if over recursion (C++17+)**
   - Better error messages
   - Faster compilation

3. **Document parameter pack requirements**
   - What types are expected
   - What operations are required

4. **Use perfect forwarding for wrapper functions**
   ```cpp
   template<typename... Args>
   void wrapper(Args&&... args) {
       func(std::forward<Args>(args)...);
   }
   ```

5. **Avoid excessive template recursion**
   - Can hit compiler limits
   - Slow compilation

## Summary

Key concepts covered:
- Variadic template syntax and parameter packs
- Pack expansion patterns
- Fold expressions (C++17)
- sizeof... operator
- Variadic class templates
- Perfect forwarding with variadic templates
- Index sequences
- Type manipulation
- Practical applications and patterns

## Next Steps

In the next chapter, we will explore SFINAE and template constraints for controlling template instantiation.
