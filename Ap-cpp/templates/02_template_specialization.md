# Chapter 2: Template Specialization

## Introduction to Specialization

Template specialization allows you to provide custom implementations for specific types or conditions. This enables optimization and handling of special cases while maintaining a generic interface.

## Full Specialization

Full specialization provides a complete alternative implementation for specific template arguments.

### Function Template Specialization

```cpp
// Primary template
template<typename T>
T max(T a, T b) {
    return a > b ? a : b;
}

// Full specialization for const char*
template<>
const char* max<const char*>(const char* a, const char* b) {
    return std::strcmp(a, b) > 0 ? a : b;
}

// Usage
max(10, 20);           // Uses primary template
max("abc", "def");     // Uses specialized version
```

### Class Template Specialization

```cpp
// Primary template
template<typename T>
class Storage {
    T data_;
public:
    void set(T value) { data_ = value; }
    T get() const { return data_; }
};

// Full specialization for bool
template<>
class Storage<bool> {
    unsigned char data_ : 1;  // Use bit field
public:
    void set(bool value) { data_ = value ? 1 : 0; }
    bool get() const { return data_ != 0; }
};
```

## Partial Specialization

Partial specialization allows specialization for a subset of template parameters or particular patterns.

### Partial Specialization for Pointers

```cpp
// Primary template
template<typename T>
class SmartPtr {
    T* ptr_;
public:
    T& operator*() { return *ptr_; }
};

// Partial specialization for array types
template<typename T>
class SmartPtr<T[]> {
    T* ptr_;
public:
    T& operator[](size_t index) { return ptr_[index]; }
};

// Usage
SmartPtr<int> ptr1;       // Uses primary template
SmartPtr<int[]> ptr2;     // Uses array specialization
```

### Partial Specialization by Pattern

```cpp
// Primary template
template<typename T, typename U>
class Pair {
    T first_;
    U second_;
};

// Specialization when both types are the same
template<typename T>
class Pair<T, T> {
    T data_[2];
public:
    T& first() { return data_[0]; }
    T& second() { return data_[1]; }
};

// Specialization for pointer types
template<typename T, typename U>
class Pair<T*, U*> {
    // Special implementation for pairs of pointers
};
```

## Specialization vs Overloading

For function templates, you can use either specialization or overloading.

### Function Overloading (Preferred)

```cpp
// Primary template
template<typename T>
void process(T value) {
    std::cout << "Generic: " << value << "\n";
}

// Overload (not specialization)
void process(int value) {
    std::cout << "Int: " << value << "\n";
}

// Another overload
template<typename T>
void process(T* ptr) {
    std::cout << "Pointer: " << *ptr << "\n";
}
```

### Why Overloading is Preferred

Overloading participates in overload resolution, while specialization does not:

```cpp
template<typename T>
void foo(T);           // #1

template<>
void foo<int*>(int*);  // #2 - specialization of #1

template<typename T>
void foo(T*);          // #3 - overload

int* p;
foo(p);  // Calls #3 (overload), not #2 (specialization)!
```

## Type Traits with Specialization

Specialization is the foundation of type traits:

```cpp
// Primary template - default case
template<typename T>
struct is_pointer {
    static constexpr bool value = false;
};

// Specialization for pointer types
template<typename T>
struct is_pointer<T*> {
    static constexpr bool value = true;
};

// Usage
static_assert(!is_pointer<int>::value);
static_assert(is_pointer<int*>::value);
```

### Multiple Trait Implementations

```cpp
// Remove reference
template<typename T>
struct remove_reference {
    using type = T;
};

template<typename T>
struct remove_reference<T&> {
    using type = T;
};

template<typename T>
struct remove_reference<T&&> {
    using type = T;
};

// Remove const
template<typename T>
struct remove_const {
    using type = T;
};

template<typename T>
struct remove_const<const T> {
    using type = T;
};
```

## Tag Dispatching

Use specialization to select implementations based on type properties:

```cpp
// Tags
struct integral_tag {};
struct floating_point_tag {};

// Tag selection
template<typename T>
struct number_category {
    using type = std::conditional_t<
        std::is_integral_v<T>,
        integral_tag,
        floating_point_tag
    >;
};

// Implementations
template<typename T>
void process_impl(T value, integral_tag) {
    std::cout << "Processing integer: " << value << "\n";
}

template<typename T>
void process_impl(T value, floating_point_tag) {
    std::cout << "Processing float: " << value << "\n";
}

// Dispatcher
template<typename T>
void process(T value) {
    process_impl(value, typename number_category<T>::type{});
}
```

## Specialization for STL Containers

### std::hash Specialization

```cpp
struct Point {
    int x, y;
};

namespace std {
    template<>
    struct hash<Point> {
        size_t operator()(const Point& p) const {
            size_t h1 = hash<int>{}(p.x);
            size_t h2 = hash<int>{}(p.y);
            return h1 ^ (h2 << 1);
        }
    };
}

// Now Point can be used in unordered containers
std::unordered_set<Point> points;
```

### std::swap Specialization

```cpp
class BigObject {
    std::vector<int> data_;
public:
    // ...
    friend void swap(BigObject& a, BigObject& b) noexcept {
        using std::swap;
        swap(a.data_, b.data_);
    }
};

// ADL will find this swap
BigObject a, b;
swap(a, b);  // Calls custom swap
```

## Member Function Specialization

You cannot partially specialize member functions, but you can use workarounds:

### Using Helper Class

```cpp
template<typename T>
class Container {
    template<typename U>
    struct Helper {
        static void process(T& value) {
            // Generic implementation
        }
    };
    
    template<>
    struct Helper<int> {
        static void process(T& value) {
            // Specialized for int
        }
    };
    
public:
    void process() {
        Helper<T>::process(data_);
    }
};
```

### Using SFINAE

```cpp
template<typename T>
class Container {
public:
    // Enabled only for integral types
    template<typename U = T>
    std::enable_if_t<std::is_integral_v<U>, void>
    process() {
        // Implementation for integers
    }
    
    // Enabled only for floating point types
    template<typename U = T>
    std::enable_if_t<std::is_floating_point_v<U>, void>
    process() {
        // Implementation for floats
    }
};
```

## Explicit vs Implicit Instantiation

### Explicit Specialization

```cpp
// Declare primary template
template<typename T>
class Widget;

// Define specialization
template<>
class Widget<int> {
    // Complete implementation
};
```

### Instantiation Order Matters

```cpp
// WRONG ORDER
template<>
void foo<int>(int) { }  // Error: primary template not yet declared

template<typename T>
void foo(T) { }

// CORRECT ORDER
template<typename T>
void foo(T) { }

template<>
void foo<int>(int) { }
```

## Variadic Template Specialization

```cpp
// Primary template
template<typename... Types>
class Tuple;

// Specialization for empty tuple
template<>
class Tuple<> {
    // Empty tuple implementation
};

// Specialization for one or more types
template<typename Head, typename... Tail>
class Tuple<Head, Tail...> {
    Head head_;
    Tuple<Tail...> tail_;
public:
    // Implementation
};
```

## Best Practices

1. **Prefer overloading to specialization for functions**
   - Overloading participates in overload resolution
   - More intuitive behavior

2. **Use partial specialization for classes**
   - Provides pattern matching on types
   - Essential for type traits

3. **Document specialization requirements**
   - Make it clear why specialization exists
   - Document any invariants

4. **Be careful with specialization order**
   - Primary template must be declared first
   - Specializations must be visible at instantiation point

5. **Consider tag dispatching instead of specialization**
   - More flexible
   - Easier to understand and maintain

6. **Use std::conditional for simple type selection**
   ```cpp
   template<typename T>
   using result_type = std::conditional_t<
       std::is_integral_v<T>,
       long long,
       double
   >;
   ```

## Common Pitfalls

### 1. Function Template Specialization Doesn't Participate in Overload Resolution

```cpp
template<typename T> void foo(T);     // #1
template<> void foo<int*>(int*);      // #2
template<typename T> void foo(T*);    // #3

int* p;
foo(p);  // Calls #3, not #2!
```

### 2. Partial Specialization Not Allowed for Functions

```cpp
template<typename T>
void process(T value);

// ERROR: Can't partially specialize function templates
template<typename T>
void process<T*>(T* ptr);

// Solution: Use overloading
template<typename T>
void process(T* ptr);  // Overload instead
```

### 3. Missing Primary Template

```cpp
// ERROR: Specialization without primary template
template<>
class Widget<int>;

// FIX: Declare primary template
template<typename T>
class Widget;

template<>
class Widget<int>;
```

## Summary

Key concepts covered:
- Full specialization for specific types
- Partial specialization for type patterns
- Specialization vs overloading
- Type traits implementation
- Tag dispatching
- STL specializations (hash, swap)
- Member function specialization workarounds
- Best practices and common pitfalls

## Next Steps

In the next chapter, we will explore variadic templates in depth, including parameter packs, fold expressions, and perfect forwarding.
