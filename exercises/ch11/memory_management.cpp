/*
 * Chapter 11 Exercise: Memory Management and Smart Pointers
 * 
 * Complete the program that demonstrates proper memory management using smart pointers.
 * The program should manage a collection of dynamically allocated objects safely.
 */

#include <iostream>
#include <memory>
#include <vector>
#include <string>

class Resource {
private:
    std::string name;
    int* data;
    size_t size;
    
public:
    // TODO: Implement constructor that takes a name and size
    // Allocate memory for data array and initialize with sequential values
    Resource(const std::string& n, size_t s) {
        // TODO: Implementation
    }
    
    // TODO: Implement destructor that prints a message and deallocates memory
    ~Resource() {
        // TODO: Implementation
    }
    
    // TODO: Implement copy constructor with deep copy
    Resource(const Resource& other) {
        // TODO: Implementation
    }
    
    // TODO: Implement copy assignment operator with deep copy
    Resource& operator=(const Resource& other) {
        // TODO: Implementation
    }
    
    // TODO: Implement move constructor
    Resource(Resource&& other) noexcept {
        // TODO: Implementation
    }
    
    // TODO: Implement move assignment operator
    Resource& operator=(Resource&& other) noexcept {
        // TODO: Implementation
    }
    
    // TODO: Implement getters
    const std::string& getName() const { return name; }
    size_t getSize() const { return size; }
    const int* getData() const { return data; }
    
    void display() const {
        std::cout << "Resource: " << name << ", Size: " << size << ", Data: [";
        for (size_t i = 0; i < size && i < 5; ++i) {  // Print first 5 elements
            std::cout << data[i];
            if (i < size - 1 && i < 4) std::cout << ", ";
        }
        if (size > 5) std::cout << "...";
        std::cout << "]" << std::endl;
    }
};

// TODO: Implement a function that creates a unique_ptr<Resource>
std::unique_ptr<Resource> createResource(const std::string& name, size_t size) {
    // TODO: Use std::make_unique to create and return a Resource
}

// TODO: Implement a function that accepts a unique_ptr<Resource> and processes it
void processResource(std::unique_ptr<Resource> resource) {
    // TODO: Display the resource and then let it go out of scope
}

// TODO: Implement a function that accepts a shared_ptr<Resource> and shares it
void shareResource(std::shared_ptr<Resource> resource) {
    // TODO: Display the resource and show reference count
}

int main() {
    std::cout << "=== Smart Pointers Exercise ===" << std::endl;
    
    // TODO: Create a unique_ptr using make_unique
    auto uniqueRes = // TODO: Create unique pointer to Resource
    
    // TODO: Display the resource
    uniqueRes->display();
    
    // TODO: Transfer ownership to another function
    processResource(std::move(uniqueRes));  // uniqueRes is now nullptr
    
    // TODO: Verify that uniqueRes is now nullptr
    if (!uniqueRes) {
        std::cout << "uniqueRes is now nullptr after move" << std::endl;
    }
    
    // TODO: Create a shared_ptr using make_shared
    auto sharedRes1 = // TODO: Create shared pointer to Resource
    
    // TODO: Create another shared pointer that shares ownership
    auto sharedRes2 = // TODO: Share ownership with sharedRes1
    
    std::cout << "Reference count after creating sharedRes2: " << sharedRes1.use_count() << std::endl;
    
    // TODO: Display both shared resources
    sharedRes1->display();
    sharedRes2->display();
    
    // TODO: Pass shared resource to function that shares it
    shareResource(sharedRes1);
    std::cout << "Reference count after sharing: " << sharedRes1.use_count() << std::endl;
    
    // TODO: Create a vector of unique_ptr<Resource>
    std::vector<std::unique_ptr<Resource>> resourceVec;
    
    // TODO: Add resources to the vector using emplace_back with make_unique
    // Add at least 3 different resources
    
    std::cout << "\nResources in vector:" << std::endl;
    for (const auto& res : resourceVec) {
        res->display();
    }
    
    // TODO: Create a vector of shared_ptr<Resource>
    std::vector<std::shared_ptr<Resource>> sharedVec;
    
    // TODO: Add shared resources to the vector
    // Each resource should be shared among multiple elements
    
    // TODO: Demonstrate weak_ptr to break potential cycles
    std::vector<std::shared_ptr<Resource>> parentVec;
    std::vector<std::weak_ptr<Resource>> weakVec;  // To avoid circular references
    
    // TODO: Create parent resources and corresponding weak references
    // Add them to respective vectors
    
    // TODO: Safely access weak references using lock()
    for (size_t i = 0; i < weakVec.size(); ++i) {
        if (auto locked = weakVec[i].lock()) {
            std::cout << "Weak pointer " << i << " is valid: ";
            locked->display();
        } else {
            std::cout << "Weak pointer " << i << " is expired" << std::endl;
        }
    }
    
    // TODO: Create a resource with custom deleter
    auto customDeleter = [](Resource* r) {
        std::cout << "Custom deleter called for: " << r->getName() << std::endl;
        delete r;
    };
    
    std::unique_ptr<Resource, decltype(customDeleter)> customPtr(
        new Resource("Custom", 3), customDeleter);
    
    customPtr->display();
    
    // TODO: Demonstrate array version of smart pointers
    std::unique_ptr<int[]> arrayPtr = std::make_unique<int[]>(5);
    for (int i = 0; i < 5; ++i) {
        arrayPtr[i] = i * i;
    }
    
    std::cout << "Array contents: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << arrayPtr[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\nAll resources will be automatically cleaned up!" << std::endl;
    
    return 0;
}