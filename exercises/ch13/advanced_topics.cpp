/*
 * Chapter 13 Exercise: Advanced Topics and Best Practices
 * 
 * Complete the program that demonstrates advanced C++ concepts and best practices.
 * The program should showcase modern C++ idioms and techniques.
 */

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <algorithm>
#include <type_traits>

// TODO: Implement the Rule of Zero by using only RAII types
class RuleOfZeroExample {
private:
    std::string name;
    std::vector<int> data;
    std::unique_ptr<int> specialResource;
    
public:
    // TODO: Implement constructor that initializes all members
    RuleOfZeroExample(const std::string& n, const std::vector<int>& d) {
        // No need to define special member functions - compiler-generated ones are perfect!
    }
    
    // TODO: Add getter methods
    const std::string& getName() const { return name; }
    const std::vector<int>& getData() const { return data; }
    int getSpecialValue() const { return specialResource ? *specialResource : 0; }
};

// TODO: Implement a class that follows the Rule of Five (but prefer Rule of Zero!)
class RuleOfFiveExample {
private:
    std::string* name;
    int* data;
    size_t size;
    
public:
    // TODO: Implement constructor
    RuleOfFiveExample(const std::string& n, size_t s) : size(s) {
        name = new std::string(n);
        data = new int[size];
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<int>(i);
        }
    }
    
    // TODO: Implement destructor
    ~RuleOfFiveExample() {
        // TODO: Clean up resources
    }
    
    // TODO: Implement copy constructor
    RuleOfFiveExample(const RuleOfFiveExample& other) {
        // TODO: Deep copy all resources
    }
    
    // TODO: Implement copy assignment operator
    RuleOfFiveExample& operator=(const RuleOfFiveExample& other) {
        // TODO: Check for self-assignment, clean up, and deep copy
        return *this;
    }
    
    // TODO: Implement move constructor
    RuleOfFiveExample(RuleOfFiveExample&& other) noexcept {
        // TODO: Transfer ownership of resources
    }
    
    // TODO: Implement move assignment operator
    RuleOfFiveExample& operator=(RuleOfFiveExample&& other) noexcept {
        // TODO: Check for self-assignment, clean up current resources, transfer ownership
        return *this;
    }
    
    // TODO: Add getter methods
    const std::string& getName() const { return *name; }
    size_t getSize() const { return size; }
    int getData(size_t index) const { 
        if (index < size) return data[index];
        return -1; // Error value
    }
};

// TODO: Implement a function template with SFINAE to work only with arithmetic types
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
safeSqrt(T value) {
    // TODO: Implement square root function that only works with arithmetic types
    // Use std::sqrt from <cmath> for floating point types
    // For integer types, return integer square root
}

// TODO: Implement a generic function that works with any container
template<typename Container>
auto getContainerStats(const Container& container) -> std::pair<size_t, bool> {
    // TODO: Return a pair with size and whether container is empty
}

// TODO: Implement a function that uses perfect forwarding
template<typename Func, typename... Args>
auto timeFunction(Func&& func, Args&&... args) -> decltype(func(args...)) {
    // TODO: Time the execution of func with args
    // Return the result of func
    // Use <chrono> for timing
    return func(std::forward<Args>(args)...);
}

// TODO: Implement a class with move semantics and perfect forwarding
class MoveEnabledClass {
private:
    std::string data;
    std::vector<int> numbers;
    
public:
    // TODO: Implement constructor with perfect forwarding for the vector
    template<typename VecType>
    MoveEnabledClass(std::string str, VecType&& vec) 
        : data(std::move(str)), numbers(std::forward<VecType>(vec)) {
    }
    
    // TODO: Implement move constructor
    MoveEnabledClass(MoveEnabledClass&& other) noexcept
        : data(std::move(other.data)), numbers(std::move(other.numbers)) {
    }
    
    // TODO: Implement move assignment operator
    MoveEnabledClass& operator=(MoveEnabledClass&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            numbers = std::move(other.numbers);
        }
        return *this;
    }
    
    // TODO: Delete copy operations to make this move-only
    MoveEnabledClass(const MoveEnabledClass&) = delete;
    MoveEnabledClass& operator=(const MoveEnabledClass&) = delete;
    
    // TODO: Add getter methods
    const std::string& getData() const { return data; }
    const std::vector<int>& getNumbers() const { return numbers; }
};

int main() {
    std::cout << "=== Advanced Topics and Best Practices ===" << std::endl;
    
    // TODO: Demonstrate Rule of Zero
    std::vector<int> sampleData = {1, 2, 3, 4, 5};
    RuleOfZeroExample roo("Sample", sampleData);
    std::cout << "Rule of Zero - Name: " << roo.getName() << std::endl;
    
    // TODO: Create a copy and move to show automatic behavior
    RuleOfZeroExample rooCopy = roo;  // Copy
    RuleOfZeroExample rooMove = std::move(RuleOfZeroExample("Temp", {10, 20, 30}));  // Move
    
    // TODO: Demonstrate Rule of Five
    RuleOfFiveExample rofive("Test", 5);
    std::cout << "Rule of Five - Name: " << rofive.getName() << ", Size: " << rofive.getSize() << std::endl;
    
    // TODO: Test copy and move operations
    RuleOfFiveExample rofiveCopy = rofive;  // Copy
    RuleOfFiveExample rofiveMove = std::move(RuleOfFiveExample("Temp", 3));  // Move
    
    // TODO: Test SFINAE-enabled function
    std::cout << "Square root of 16: " << safeSqrt(16) << std::endl;
    std::cout << "Square root of 2.0: " << safeSqrt(2.0) << std::endl;
    
    // TODO: Test container stats function
    std::vector<int> vec = {1, 2, 3, 4, 5};
    auto [size, empty] = getContainerStats(vec);
    std::cout << "Vector size: " << size << ", Empty: " << empty << std::endl;
    
    std::string str = "Hello";
    auto [strSize, strEmpty] = getContainerStats(str);
    std::cout << "String size: " << strSize << ", Empty: " << strEmpty << std::endl;
    
    // TODO: Demonstrate perfect forwarding with timeFunction
    auto add = [](int a, int b) { return a + b; };
    int result = timeFunction(add, 5, 3);
    std::cout << "Result of timed function: " << result << std::endl;
    
    // TODO: Demonstrate move-enabled class
    std::vector<int> nums = {10, 20, 30, 40, 50};
    MoveEnabledClass mec("MoveMe", std::move(nums));  // nums is now empty
    
    std::cout << "Move-enabled class data: " << mec.getData() << std::endl;
    std::cout << "Numbers: ";
    for (int n : mec.getNumbers()) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    // TODO: Move the object to another variable
    MoveEnabledClass mec2 = std::move(mec);
    std::cout << "After move - Original is now empty: " << mec.getNumbers().empty() << std::endl;
    std::cout << "After move - Moved-to has data: " << !mec2.getNumbers().empty() << std::endl;
    
    // TODO: Demonstrate const-correctness
    const std::vector<int> constVec = {5, 4, 3, 2, 1};
    auto [constSize, constEmpty] = getContainerStats(constVec);
    std::cout << "Const vector size: " << constSize << std::endl;
    
    // TODO: Use STL algorithms with lambdas
    std::vector<int> numbers = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    
    // Sort in descending order
    std::sort(numbers.begin(), numbers.end(), [](int a, int b) { return a > b; });
    std::cout << "Sorted descending: ";
    for (int n : numbers) std::cout << n << " ";
    std::cout << std::endl;
    
    // Find first even number
    auto evenIt = std::find_if(numbers.begin(), numbers.end(), 
                              [](int n) { return n % 2 == 0; });
    if (evenIt != numbers.end()) {
        std::cout << "First even number: " << *evenIt << std::endl;
    }
    
    // Transform to squares
    std::vector<int> squares(numbers.size());
    std::transform(numbers.begin(), numbers.end(), squares.begin(),
                   [](int n) { return n * n; });
    std::cout << "Squares: ";
    for (int n : squares) std::cout << n << " ";
    std::cout << std::endl;
    
    // TODO: Demonstrate function objects and std::function
    std::function<int(int, int)> operation = [](int a, int b) { return a * b; };
    std::cout << "Function object result: " << operation(6, 7) << std::endl;
    
    // TODO: Create a function that returns different operations based on input
    auto getOperation = [](char op) -> std::function<int(int, int)> {
        switch(op) {
            case '+': return [](int a, int b) { return a + b; };
            case '-': return [](int a, int b) { return a - b; };
            case '*': return [](int a, int b) { return a * b; };
            case '/': return [](int a, int b) { return b != 0 ? a / b : 0; };
            default: return [](int a, int b) { return a + b; };
        }
    };
    
    auto multiplyOp = getOperation('*');
    std::cout << "Dynamic operation result: " << multiplyOp(4, 5) << std::endl;
    
    // TODO: Demonstrate move semantics with containers
    std::vector<MoveEnabledClass> container;
    
    // Add elements efficiently using emplace_back
    container.emplace_back("First", std::vector<int>{1, 2, 3});
    container.emplace_back("Second", std::vector<int>{4, 5, 6});
    
    std::cout << "\nContainer contents:" << std::endl;
    for (const auto& obj : container) {
        std::cout << obj.getData() << ": ";
        for (int n : obj.getNumbers()) {
            std::cout << n << " ";
        }
        std::cout << std::endl;
    }
    
    // TODO: Move elements between containers
    std::vector<MoveEnabledClass> destination;
    for (auto& obj : container) {
        destination.push_back(std::move(obj));  // Move each element
    }
    
    std::cout << "\nAfter moving to destination:" << std::endl;
    std::cout << "Original container size: " << container.size() << std::endl;
    std::cout << "Destination container size: " << destination.size() << std::endl;
    
    // TODO: Demonstrate type traits usage
    std::cout << "\nType traits demonstrations:" << std::endl;
    std::cout << "int is integral: " << std::is_integral<int>::value << std::endl;
    std::cout << "double is integral: " << std::is_integral<double>::value << std::endl;
    std::cout << "int is arithmetic: " << std::is_arithmetic<int>::value << std::endl;
    std::cout << "std::string is arithmetic: " << std::is_arithmetic<std::string>::value << std::endl;
    
    std::cout << "\nAdvanced C++ concepts exercise completed!" << std::endl;
    
    return 0;
}