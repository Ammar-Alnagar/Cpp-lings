/*
 * Chapter 15 Exercise: Modern C++ Features
 * 
 * Complete the program that demonstrates modern C++ features introduced in C++11 and later.
 * The program should showcase features like auto, lambdas, smart pointers, etc.
 */

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
#include <chrono>
#include <map>
#include <set>
#include <unordered_map>
#include <optional>
#include <variant>
#include <any>

// TODO: Implement a class that demonstrates move semantics and perfect forwarding
class ModernClass {
private:
    std::string name;
    std::vector<int> data;
    
public:
    // TODO: Implement constructor with member initializer list
    ModernClass(std::string n, std::vector<int> d) : name(std::move(n)), data(std::move(d)) {
        std::cout << "ModernClass constructed: " << name << std::endl;
    }
    
    // TODO: Implement move constructor
    ModernClass(ModernClass&& other) noexcept 
        : name(std::move(other.name)), data(std::move(other.data)) {
        std::cout << "ModernClass moved: " << name << std::endl;
    }
    
    // TODO: Implement move assignment operator
    ModernClass& operator=(ModernClass&& other) noexcept {
        if (this != &other) {
            name = std::move(other.name);
            data = std::move(other.data);
        }
        std::cout << "ModernClass move-assigned: " << name << std::endl;
        return *this;
    }
    
    // TODO: Delete copy operations to make this move-only
    ModernClass(const ModernClass&) = delete;
    ModernClass& operator=(const ModernClass&) = delete;
    
    // TODO: Implement getter methods
    const std::string& getName() const { return name; }
    const std::vector<int>& getData() const { return data; }
};

// TODO: Implement a function that uses auto return type deduction
template<typename T, typename U>
auto multiply(T&& t, U&& u) -> decltype(std::forward<T>(t) * std::forward<U>(u)) {
    return std::forward<T>(t) * std::forward<U>(u);
}

// Alternative C++14 syntax (auto return type without trailing return type)
template<typename T, typename U>
auto multiplySimple(T&& t, U&& u) {
    return std::forward<T>(t) * std::forward<U>(u);
}

// TODO: Implement a function that demonstrates generic lambdas (C++14)
void demonstrateGenericLambda() {
    auto genericLambda = [](auto x, auto y) {
        return x + y;
    };
    
    std::cout << "Generic lambda - int: " << genericLambda(5, 3) << std::endl;
    std::cout << "Generic lambda - double: " << genericLambda(3.14, 2.71) << std::endl;
    std::cout << "Generic lambda - string: " << genericLambda(std::string("Hello "), std::string("World")) << std::endl;
}

// TODO: Implement a function that demonstrates structured bindings (C++17)
void demonstrateStructuredBindings() {
    std::map<std::string, int> ages = {{"Alice", 25}, {"Bob", 30}, {"Charlie", 35}};
    
    std::cout << "Using structured bindings:" << std::endl;
    for (const auto& [name, age] : ages) {
        std::cout << name << " is " << age << " years old" << std::endl;
    }
    
    // TODO: Use structured bindings with std::pair
    auto result = std::make_pair(42, std::string("Answer"));
    auto [number, text] = result;
    std::cout << "Number: " << number << ", Text: " << text << std::endl;
    
    // TODO: Use structured bindings with std::minmax_element
    std::vector<int> numbers = {5, 2, 8, 1, 9, 3};
    auto [minIt, maxIt] = std::minmax_element(numbers.begin(), numbers.end());
    std::cout << "Min: " << *minIt << ", Max: " << *maxIt << std::endl;
}

// TODO: Implement a function that demonstrates if/switch with initializer (C++17)
void demonstrateIfSwitchWithInitializer() {
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // Using if with initializer to avoid variable pollution
    if (auto it = std::find(numbers.begin(), numbers.end(), 7); it != numbers.end()) {
        std::cout << "Found 7 at position: " << std::distance(numbers.begin(), it) << std::endl;
    } else {
        std::cout << "7 not found" << std::endl;
    }
    
    // TODO: Demonstrate switch with initializer
    switch (auto maxElement = *std::max_element(numbers.begin(), numbers.end()); maxElement) {
        case 10:
            std::cout << "Max element is 10" << std::endl;
            break;
        case 9:
            std::cout << "Max element is 9" << std::endl;
            break;
        default:
            std::cout << "Max element is " << maxElement << std::endl;
            break;
    }
}

// TODO: Implement a function that demonstrates class template argument deduction (C++17)
void demonstrateClassTemplateArgumentDeduction() {
    // Before C++17, we had to specify template arguments explicitly
    std::pair<int, std::string> oldPair(42, "Hello");
    
    // C++17: Class Template Argument Deduction (CTAD)
    std::pair newPair(42, "Hello");  // Compiler deduces std::pair<int, const char*>
    std::vector vec{1, 2, 3, 4, 5};  // Compiler deduces std::vector<int>
    std::array arr{1.0, 2.0, 3.0};  // Compiler deduces std::array<double, 3>
    
    std::cout << "CTAD pair: " << newPair.first << ", " << newPair.second << std::endl;
    std::cout << "CTAD vector size: " << vec.size() << std::endl;
    std::cout << "CTAD array size: " << arr.size() << std::endl;
}

// TODO: Implement a function that demonstrates std::optional (C++17)
void demonstrateOptional() {
    std::optional<int> opt1;  // Empty optional
    std::optional<int> opt2 = 42;  // Optional with value
    auto opt3 = std::make_optional<int>(100);  // Another way to create optional
    
    std::cout << "opt1 has value: " << opt1.has_value() << std::endl;
    std::cout << "opt2 value: " << opt2.value() << std::endl;
    std::cout << "opt3 value: " << opt3.value_or(0) << std::endl;  // Default value if empty
    
    // TODO: Use optional in a function that might fail
    auto divide = [](int a, int b) -> std::optional<double> {
        if (b == 0) return std::nullopt;
        return static_cast<double>(a) / b;
    };
    
    if (auto result = divide(10, 3); result) {
        std::cout << "10 / 3 = " << result.value() << std::endl;
    }
    
    if (auto result = divide(10, 0); !result) {
        std::cout << "Division by zero detected!" << std::endl;
    }
}

// TODO: Implement a function that demonstrates std::variant (C++17)
void demonstrateVariant() {
    std::variant<int, std::string, double> var;
    
    var = 42;  // Holds int
    std::cout << "Variant holds int: " << std::get<int>(var) << std::endl;
    
    var = std::string("Hello");  // Now holds string
    std::cout << "Variant holds string: " << std::get<std::string>(var) << std::endl;
    
    var = 3.14;  // Now holds double
    std::cout << "Variant holds double: " << std::get<double>(var) << std::endl;
    
    // TODO: Use std::visit to work with variant
    std::visit([](const auto& value) {
        std::cout << "Visited value: " << value << std::endl;
    }, var);
    
    // TODO: Check what type is currently held
    if (std::holds_alternative<double>(var)) {
        std::cout << "Variant currently holds a double" << std::endl;
    }
}

// TODO: Implement a function that demonstrates std::any (C++17)
void demonstrateAny() {
    std::any anything;
    
    anything = 42;  // Store int
    std::cout << "Any holds int: " << std::any_cast<int>(anything) << std::endl;
    
    anything = std::string("Hello Any");  // Store string
    std::cout << "Any holds string: " << std::any_cast<std::string>(anything) << std::endl;
    
    anything = 3.14159;  // Store double
    std::cout << "Any holds double: " << std::any_cast<double>(anything) << std::endl;
    
    // TODO: Safe casting with type checking
    if (anything.type() == typeid(double)) {
        std::cout << "Confirmed: any holds a double" << std::endl;
    }
}

// TODO: Implement constexpr functions (C++11/14/20)
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : (n * factorial(n - 1));
}

// TODO: Implement a function that demonstrates if constexpr (C++17)
template<typename T>
void processType(const T& value) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "Processing integral: " << value * 2 << std::endl;
    } else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "Processing floating point: " << value * 1.5 << std::endl;
    } else {
        std::cout << "Processing other type" << std::endl;
    }
}

int main() {
    std::cout << "=== Modern C++ Features Demo ===" << std::endl;
    
    // TODO: Demonstrate auto type deduction
    auto x = 42;  // int
    auto y = 3.14;  // double
    auto z = std::string("Modern C++");  // std::string
    
    std::cout << "Auto-deduced types: " << x << ", " << y << ", " << z << std::endl;
    
    // TODO: Demonstrate range-based for loops with auto
    std::vector<std::string> modernFeatures = {
        "auto", "lambdas", "smart pointers", "move semantics", 
        "constexpr", "templates", "STL", "concurrency"
    };
    
    std::cout << "\nModern C++ features:" << std::endl;
    for (const auto& feature : modernFeatures) {
        std::cout << "- " << feature << std::endl;
    }
    
    // TODO: Demonstrate lambda expressions
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // Filter even numbers using lambda
    auto evenNumbers = numbers;
    evenNumbers.erase(
        std::remove_if(evenNumbers.begin(), evenNumbers.end(),
                      [](int n) { return n % 2 != 0; }),
        evenNumbers.end());
    
    std::cout << "\nEven numbers: ";
    for (int n : evenNumbers) std::cout << n << " ";
    std::cout << std::endl;
    
    // Transform using lambda
    std::transform(numbers.begin(), numbers.end(), numbers.begin(),
                   [](int n) { return n * n; });
    
    std::cout << "Squared numbers: ";
    for (int n : numbers) std::cout << n << " ";
    std::cout << std::endl;
    
    // TODO: Demonstrate smart pointers
    auto smartInt = std::make_unique<int>(42);
    auto smartString = std::make_shared<std::string>("Hello Smart Pointers");
    
    std::cout << "\nSmart pointer values: " << *smartInt << ", " << *smartString << std::endl;
    
    // TODO: Demonstrate move semantics
    std::vector<ModernClass> container;
    
    // Emplace back with move semantics
    container.emplace_back("Object1", std::vector<int>{1, 2, 3});
    container.emplace_back("Object2", std::vector<int>{4, 5, 6});
    
    std::cout << "\nContainer contents:" << std::endl;
    for (const auto& obj : container) {
        std::cout << obj.getName() << ": ";
        for (int val : obj.getData()) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    
    // TODO: Demonstrate generic lambdas
    demonstrateGenericLambda();
    
    // TODO: Demonstrate structured bindings
    demonstrateStructuredBindings();
    
    // TODO: Demonstrate if/switch with initializer
    demonstrateIfSwitchWithInitializer();
    
    // TODO: Demonstrate class template argument deduction
    demonstrateClassTemplateArgumentDeduction();
    
    // TODO: Demonstrate optional
    demonstrateOptional();
    
    // TODO: Demonstrate variant
    demonstrateVariant();
    
    // TODO: Demonstrate any
    demonstrateAny();
    
    // TODO: Demonstrate constexpr
    constexpr int compileTimeFactorial = factorial(5);
    std::cout << "\nCompile-time factorial of 5: " << compileTimeFactorial << std::endl;
    
    // TODO: Demonstrate if constexpr
    processType(42);      // Will use integral branch
    processType(3.14);    // Will use floating point branch
    processType(std::string("test"));  // Will use other branch
    
    // TODO: Demonstrate fold expressions (C++17) - variadic template technique
    auto foldSum = [](auto... args) {
        return (args + ... + 0);  // Unary right fold
    };
    
    std::cout << "\nFold expression sum: " << foldSum(1, 2, 3, 4, 5) << std::endl;
    
    // TODO: Demonstrate string literals
    using namespace std::string_literals;
    auto str = "Modern C++"s;
    std::cout << "String literal: " << str << std::endl;
    
    using namespace std::chrono_literals;
    auto duration = 100ms;  // milliseconds
    std::cout << "Duration: " << duration.count() << "ms" << std::endl;
    
    // TODO: Demonstrate [[maybe_unused]], [[nodiscard]], etc.
    [[maybe_unused]] int unusedVar = 42;
    
    [[nodiscard]] auto importantFunction() -> int {
        return 123;
    }
    
    // The compiler should warn if we ignore the return value
    importantFunction();  // This might generate a warning
    
    // TODO: Demonstrate structured bindings with more complex types
    std::set<int> uniqueNumbers = {5, 2, 8, 1, 9, 3};
    auto [inserted, position] = uniqueNumbers.insert(7);
    std::cout << "Inserted " << (inserted ? "successfully" : "failed") 
              << ", size now: " << uniqueNumbers.size() << std::endl;
    
    // TODO: Demonstrate parallel algorithms (C++17)
    std::vector<int> largeVector(1000000);
    std::iota(largeVector.begin(), largeVector.end(), 1);
    
    // Shuffle the vector
    std::shuffle(largeVector.begin(), largeVector.end(), 
                 std::default_random_engine(std::random_device{}()));
    
    // Sort using execution policy (if available)
    #ifdef __cpp_lib_execution
    std::sort(std::execution::par_unseq, largeVector.begin(), largeVector.end());
    #else
    std::sort(largeVector.begin(), largeVector.end());
    #endif
    
    std::cout << "\nFirst 10 elements after sorting: ";
    for (int i = 0; i < 10; i++) {
        std::cout << largeVector[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\nModern C++ features exercise completed!" << std::endl;
    
    return 0;
}