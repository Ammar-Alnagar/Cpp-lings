// Advanced C++ Templates: Template Fundamentals
// This example demonstrates basic template concepts, function and class templates

#include <iostream>
#include <string>
#include <vector>
#include <type_traits>
#include <array>

// Example 1: Basic function template
namespace example1 {
    // Simple function template that works with any type supporting operator>
    template<typename T>
    T maximum(T a, T b) {
        return a > b ? a : b;
    }
    
    // Function template with multiple type parameters
    template<typename T, typename U>
    auto add(T a, U b) -> decltype(a + b) {
        return a + b;
    }
    
    void demonstrate() {
        std::cout << "\n=== Example 1: Basic Function Templates ===\n";
        
        // Template argument deduction
        std::cout << "maximum(10, 20) = " << maximum(10, 20) << "\n";
        std::cout << "maximum(3.14, 2.71) = " << maximum(3.14, 2.71) << "\n";
        std::cout << "maximum(\"apple\", \"banana\") = " << maximum(std::string("apple"), std::string("banana")) << "\n";
        
        // Multiple type parameters
        std::cout << "add(10, 3.14) = " << add(10, 3.14) << "\n";
        std::cout << "add(2.5, 7) = " << add(2.5, 7) << "\n";
        
        // Explicit template arguments
        std::cout << "maximum<double>(10, 20) = " << maximum<double>(10, 20) << "\n";
    }
}

// Example 2: Non-type template parameters
namespace example2 {
    // Template with non-type parameter (compile-time constant)
    template<typename T, size_t N>
    class FixedArray {
        T data_[N];
        
    public:
        FixedArray() {
            for (size_t i = 0; i < N; ++i) {
                data_[i] = T{};
            }
        }
        
        constexpr size_t size() const { return N; }
        
        T& operator[](size_t index) { return data_[index]; }
        const T& operator[](size_t index) const { return data_[index]; }
        
        // Iterator support
        T* begin() { return data_; }
        T* end() { return data_ + N; }
    };
    
    // Function template using non-type parameter
    template<typename T, size_t N>
    void print_array(const FixedArray<T, N>& arr) {
        std::cout << "[";
        for (size_t i = 0; i < N; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << arr[i];
        }
        std::cout << "]\n";
    }
    
    void demonstrate() {
        std::cout << "\n=== Example 2: Non-Type Template Parameters ===\n";
        
        FixedArray<int, 5> int_array;
        for (size_t i = 0; i < int_array.size(); ++i) {
            int_array[i] = i * 10;
        }
        
        std::cout << "FixedArray<int, 5>: ";
        print_array(int_array);
        
        FixedArray<double, 3> double_array;
        double_array[0] = 1.1;
        double_array[1] = 2.2;
        double_array[2] = 3.3;
        
        std::cout << "FixedArray<double, 3>: ";
        print_array(double_array);
    }
}

// Example 3: Class templates
namespace example3 {
    // Generic container class
    template<typename T>
    class Box {
        T value_;
        
    public:
        // Constructor
        explicit Box(const T& value) : value_(value) {}
        
        // Getter
        T get() const { return value_; }
        
        // Setter
        void set(const T& value) { value_ = value; }
        
        // Display
        void display() const {
            std::cout << "Box contains: " << value_ << "\n";
        }
    };
    
    // Template class with multiple parameters
    template<typename Key, typename Value>
    class Pair {
        Key key_;
        Value value_;
        
    public:
        Pair(const Key& k, const Value& v) : key_(k), value_(v) {}
        
        Key key() const { return key_; }
        Value value() const { return value_; }
        
        void display() const {
            std::cout << "Pair(" << key_ << ", " << value_ << ")\n";
        }
    };
    
    void demonstrate() {
        std::cout << "\n=== Example 3: Class Templates ===\n";
        
        Box<int> int_box(42);
        int_box.display();
        
        Box<std::string> str_box("Hello, Templates!");
        str_box.display();
        
        Box<double> double_box(3.14159);
        double_box.display();
        
        Pair<std::string, int> age_pair("Alice", 30);
        age_pair.display();
        
        Pair<int, double> coord_pair(10, 3.14);
        coord_pair.display();
    }
}

// Example 4: Member function templates
namespace example4 {
    class Printer {
    public:
        // Member function template
        template<typename T>
        void print(const T& value) const {
            std::cout << "Printing: " << value << "\n";
        }
        
        // Member function template with multiple parameters
        template<typename T, typename U>
        void print_pair(const T& first, const U& second) const {
            std::cout << "Pair: (" << first << ", " << second << ")\n";
        }
    };
    
    // Template class with member function template
    template<typename T>
    class Converter {
        T value_;
        
    public:
        Converter(const T& val) : value_(val) {}
        
        // Member function template for conversion
        template<typename U>
        U convert_to() const {
            return static_cast<U>(value_);
        }
    };
    
    void demonstrate() {
        std::cout << "\n=== Example 4: Member Function Templates ===\n";
        
        Printer printer;
        printer.print(42);
        printer.print(3.14);
        printer.print(std::string("Hello"));
        printer.print_pair(10, "items");
        
        Converter<double> conv(3.14159);
        std::cout << "Convert 3.14159 to int: " << conv.convert_to<int>() << "\n";
        std::cout << "Convert 3.14159 to float: " << conv.convert_to<float>() << "\n";
    }
}

// Example 5: Default template arguments
namespace example5 {
    // Function template with default type
    template<typename T = int>
    T get_default() {
        return T{};
    }
    
    // Class template with default type
    template<typename T, typename Container = std::vector<T>>
    class Stack {
        Container data_;
        
    public:
        void push(const T& value) {
            data_.push_back(value);
        }
        
        void pop() {
            if (!data_.empty()) {
                data_.pop_back();
            }
        }
        
        T top() const {
            return data_.back();
        }
        
        bool empty() const {
            return data_.empty();
        }
        
        size_t size() const {
            return data_.size();
        }
    };
    
    void demonstrate() {
        std::cout << "\n=== Example 5: Default Template Arguments ===\n";
        
        // Use default template argument
        auto default_int = get_default<>();
        std::cout << "Default int: " << default_int << "\n";
        
        // Override default
        auto default_double = get_default<double>();
        std::cout << "Default double: " << default_double << "\n";
        
        // Stack with default container (std::vector)
        Stack<int> stack1;
        stack1.push(10);
        stack1.push(20);
        stack1.push(30);
        std::cout << "Stack top: " << stack1.top() << "\n";
        std::cout << "Stack size: " << stack1.size() << "\n";
    }
}

// Example 6: Template template parameters
namespace example6 {
    // Template that takes a template as parameter
    template<typename T, template<typename> class Container>
    class Wrapper {
        Container<T> data_;
        
    public:
        void add(const T& value) {
            data_.push_back(value);
        }
        
        size_t size() const {
            return data_.size();
        }
        
        void display() const {
            std::cout << "Container contains " << size() << " elements\n";
        }
    };
    
    // Simple container template for demonstration
    template<typename T>
    class SimpleVector {
        std::vector<T> data_;
    public:
        void push_back(const T& val) { data_.push_back(val); }
        size_t size() const { return data_.size(); }
    };
    
    void demonstrate() {
        std::cout << "\n=== Example 6: Template Template Parameters ===\n";
        
        Wrapper<int, std::vector> wrapper1;
        wrapper1.add(1);
        wrapper1.add(2);
        wrapper1.add(3);
        wrapper1.display();
        
        Wrapper<double, SimpleVector> wrapper2;
        wrapper2.add(1.1);
        wrapper2.add(2.2);
        wrapper2.display();
    }
}

// Example 7: Static members in templates
namespace example7 {
    template<typename T>
    class Counter {
        static int count_;
        
    public:
        Counter() { ++count_; }
        ~Counter() { --count_; }
        
        static int get_count() { return count_; }
    };
    
    // Static member definition
    template<typename T>
    int Counter<T>::count_ = 0;
    
    void demonstrate() {
        std::cout << "\n=== Example 7: Static Members in Templates ===\n";
        
        std::cout << "Initial Counter<int>: " << Counter<int>::get_count() << "\n";
        std::cout << "Initial Counter<double>: " << Counter<double>::get_count() << "\n";
        
        {
            Counter<int> c1, c2, c3;
            Counter<double> d1, d2;
            
            std::cout << "Counter<int> after creating 3: " << Counter<int>::get_count() << "\n";
            std::cout << "Counter<double> after creating 2: " << Counter<double>::get_count() << "\n";
        }
        
        std::cout << "Counter<int> after destruction: " << Counter<int>::get_count() << "\n";
        std::cout << "Counter<double> after destruction: " << Counter<double>::get_count() << "\n";
    }
}

// Example 8: typename keyword for dependent types
namespace example8 {
    template<typename Container>
    void print_container(const Container& c) {
        // typename is required for dependent type names
        typename Container::value_type sum{};
        
        for (const auto& elem : c) {
            sum += elem;
        }
        
        std::cout << "Sum: " << sum << "\n";
    }
    
    template<typename T>
    class MyContainer {
    public:
        using value_type = T;  // Type alias
        using iterator = T*;
        
    private:
        std::vector<T> data_;
        
    public:
        void add(const T& val) { data_.push_back(val); }
        
        typename std::vector<T>::iterator begin() { return data_.begin(); }
        typename std::vector<T>::iterator end() { return data_.end(); }
    };
    
    void demonstrate() {
        std::cout << "\n=== Example 8: typename for Dependent Types ===\n";
        
        std::vector<int> vec = {1, 2, 3, 4, 5};
        print_container(vec);
        
        std::vector<double> dvec = {1.1, 2.2, 3.3};
        print_container(dvec);
    }
}

// Example 9: CRTP (Curiously Recurring Template Pattern)
namespace example9 {
    // Base template using CRTP
    template<typename Derived>
    class Shape {
    public:
        void draw() const {
            // Call the derived class implementation
            static_cast<const Derived*>(this)->draw_impl();
        }
        
        double area() const {
            return static_cast<const Derived*>(this)->area_impl();
        }
    };
    
    // Derived class inherits from Base<Derived>
    class Circle : public Shape<Circle> {
        double radius_;
        
    public:
        Circle(double r) : radius_(r) {}
        
        void draw_impl() const {
            std::cout << "Drawing circle with radius " << radius_ << "\n";
        }
        
        double area_impl() const {
            return 3.14159 * radius_ * radius_;
        }
    };
    
    class Rectangle : public Shape<Rectangle> {
        double width_, height_;
        
    public:
        Rectangle(double w, double h) : width_(w), height_(h) {}
        
        void draw_impl() const {
            std::cout << "Drawing rectangle " << width_ << "x" << height_ << "\n";
        }
        
        double area_impl() const {
            return width_ * height_;
        }
    };
    
    void demonstrate() {
        std::cout << "\n=== Example 9: CRTP Pattern ===\n";
        
        Circle circle(5.0);
        circle.draw();
        std::cout << "Circle area: " << circle.area() << "\n";
        
        Rectangle rect(4.0, 6.0);
        rect.draw();
        std::cout << "Rectangle area: " << rect.area() << "\n";
    }
}

// Example 10: Variadic templates introduction
namespace example10 {
    // Base case for recursion
    void print() {
        std::cout << "\n";
    }
    
    // Recursive variadic template
    template<typename T, typename... Args>
    void print(T first, Args... rest) {
        std::cout << first << " ";
        print(rest...);  // Recursive call with remaining arguments
    }
    
    // C++17 fold expression version (more efficient)
    template<typename... Args>
    void print_fold(Args... args) {
        (std::cout << ... << args) << "\n";
    }
    
    void demonstrate() {
        std::cout << "\n=== Example 10: Variadic Templates Introduction ===\n";
        
        std::cout << "Recursive version: ";
        print(1, 2.5, "hello", 'x', 42);
        
        std::cout << "Fold expression version: ";
        print_fold(1, 2.5, "hello", 'x', 42);
    }
}

int main() {
    std::cout << "=== Advanced C++ Templates: Fundamentals ===\n";
    
    example1::demonstrate();
    example2::demonstrate();
    example3::demonstrate();
    example4::demonstrate();
    example5::demonstrate();
    example6::demonstrate();
    example7::demonstrate();
    example8::demonstrate();
    example9::demonstrate();
    example10::demonstrate();
    
    std::cout << "\n=== All examples completed successfully ===\n";
    
    return 0;
}
