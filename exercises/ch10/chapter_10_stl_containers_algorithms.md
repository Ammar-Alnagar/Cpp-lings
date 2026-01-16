# Chapter 10: STL Containers and Algorithms

## Overview

This chapter covers the Standard Template Library (STL) in C++, which provides a collection of template classes and functions that implement common data structures and algorithms. You'll learn about containers, iterators, algorithms, and how to use them effectively.

## Learning Objectives

By the end of this chapter, you will:
- Understand the different categories of STL containers
- Learn to use sequential containers (vector, list, deque, array)
- Learn to use associative containers (map, set, multimap, multiset)
- Learn to use unordered associative containers (unordered_map, unordered_set)
- Master STL iterators and their types
- Understand and use STL algorithms effectively
- Learn about function objects and lambdas with STL
- Understand container adaptors (stack, queue, priority_queue)
- Learn best practices for STL usage

## Sequential Containers

Sequential containers store elements in a linear sequence.

### Exercise 1: Vector Container

The following code has errors. Find and fix them:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    // Creating vectors
    vector<int> vec1;                    // Empty vector
    vector<int> vec2(5);                // Vector with 5 elements (all 0)
    vector<int> vec3(5, 10);            // Vector with 5 elements (all 10)
    vector<int> vec4 = {1, 2, 3, 4, 5}; // Initializer list
    
    cout << "vec2 size: " << vec2.size() << endl;
    cout << "vec3 size: " << vec3.size() << endl;
    cout << "vec4 size: " << vec4.size() << endl;
    
    // Adding elements
    vec1.push_back(100);
    vec1.push_back(200);
    vec1.push_back(300);
    
    cout << "vec1 after adding elements: ";
    for (size_t i = 0; i < vec1.size(); i++) {
        cout << vec1[i] << " ";  // Using operator[]
    }
    cout << endl;
    
    // Error: accessing out of bounds
    // cout << vec1[10] << endl;  // Undefined behavior!
    // cout << vec1.at(10) << endl;  // Throws exception
    
    // Using at() for bounds checking
    try {
        cout << "Element at index 2: " << vec1.at(2) << endl;
        cout << "Element at index 10: " << vec1.at(10) << endl;  // Will throw
    } catch (const out_of_range& e) {
        cout << "Exception caught: " << e.what() << endl;
    }
    
    // Iterating with iterators
    cout << "vec1 using iterators: ";
    for (auto it = vec1.begin(); it != vec1.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    
    // Range-based for loop
    cout << "vec1 using range-based for: ";
    for (const auto& element : vec1) {
        cout << element << " ";
    }
    cout << endl;
    
    // Modifying elements
    vec1[0] = 999;
    vec1.at(1) = 888;
    
    cout << "vec1 after modification: ";
    for (const auto& element : vec1) {
        cout << element << " ";
    }
    cout << endl;
    
    // Inserting elements
    vec1.insert(vec1.begin() + 1, 555);  // Insert at position 1
    cout << "After inserting 555 at position 1: ";
    for (const auto& element : vec1) {
        cout << element << " ";
    }
    cout << endl;
    
    // Removing elements
    vec1.pop_back();  // Remove last element
    vec1.erase(vec1.begin());  // Remove first element
    cout << "After removing elements: ";
    for (const auto& element : vec1) {
        cout << element << " ";
    }
    cout << endl;
    
    // Capacity operations
    cout << "Size: " << vec1.size() << endl;
    cout << "Capacity: " << vec1.capacity() << endl;
    cout << "Max size: " << vec1.max_size() << endl;
    
    // Reserve space to avoid reallocations
    vec1.reserve(100);
    cout << "Capacity after reserve(100): " << vec1.capacity() << endl;
    
    // Shrink to fit
    vec1.shrink_to_fit();
    cout << "Capacity after shrink_to_fit: " << vec1.capacity() << endl;
    
    // Check if empty
    cout << "vec1 is empty: " << vec1.empty() << endl;
    vector<int> emptyVec;
    cout << "emptyVec is empty: " << emptyVec.empty() << endl;
    
    return 0;
}
```

### Exercise 2: Other Sequential Containers

Complete this example with different sequential containers:

```cpp
#include <iostream>
#include <vector>
#include <list>
#include <deque>
#include <array>
#include <forward_list>
#include <algorithm>
using namespace std;

int main() {
    cout << "=== Vector Demo ===" << endl;
    vector<int> vec = {1, 2, 3, 4, 5};
    cout << "Vector: ";
    for (const auto& elem : vec) cout << elem << " ";
    cout << endl;
    
    cout << "\n=== List Demo ===" << endl;
    list<int> lst = {10, 20, 30, 40, 50};
    cout << "List: ";
    for (const auto& elem : lst) cout << elem << " ";
    cout << endl;
    
    // List-specific operations
    lst.push_front(5);    // Add to front
    lst.push_back(55);    // Add to back
    cout << "After push_front(5) and push_back(55): ";
    for (const auto& elem : lst) cout << elem << " ";
    cout << endl;
    
    lst.sort();  // Sort the list
    cout << "After sorting: ";
    for (const auto& elem : lst) cout << elem << " ";
    cout << endl;
    
    cout << "\n=== Deque Demo ===" << endl;
    deque<int> deq = {100, 200, 300};
    deq.push_front(50);
    deq.push_back(350);
    cout << "Deque: ";
    for (const auto& elem : deq) cout << elem << " ";
    cout << endl;
    
    cout << "\n=== Array Demo ===" << endl;
    array<int, 5> arr = {1, 2, 3, 4, 5};  // Fixed size
    cout << "Array: ";
    for (const auto& elem : arr) cout << elem << " ";
    cout << endl;
    cout << "Array size: " << arr.size() << endl;
    
    cout << "\n=== Forward List Demo ===" << endl;
    forward_list<int> flst = {1, 2, 3, 4, 5};
    cout << "Forward list: ";
    for (const auto& elem : flst) cout << elem << " ";
    cout << endl;
    
    // Forward list operations (singly linked list)
    flst.push_front(0);  // Only push_front is available
    cout << "After push_front(0): ";
    for (const auto& elem : flst) cout << elem << " ";
    cout << endl;
    
    // Error: forward_list doesn't have size() method (to maintain O(1) performance)
    // cout << "Forward list size: " << flst.size() << endl;  // Error!
    cout << "Forward list size: " << distance(flst.begin(), flst.end()) << endl;
    
    return 0;
}
```

## Associative Containers

Associative containers store elements in a sorted order based on keys.

### Exercise 3: Map Container

Complete this map example with errors:

```cpp
#include <iostream>
#include <map>
#include <string>
#include <algorithm>
using namespace std;

int main() {
    // Creating maps
    map<string, int> ages;
    map<int, string> numbers = {{1, "one"}, {2, "two"}, {3, "three"}};
    
    // Adding elements to map
    ages["Alice"] = 25;
    ages["Bob"] = 30;
    ages["Charlie"] = 35;
    
    cout << "Ages map: " << endl;
    for (const auto& pair : ages) {
        cout << pair.first << " -> " << pair.second << endl;
    }
    
    cout << "\nNumbers map: " << endl;
    for (const auto& pair : numbers) {
        cout << pair.first << " -> " << pair.second << endl;
    }
    
    // Accessing elements
    cout << "\nAlice's age: " << ages["Alice"] << endl;
    cout << "Bob's age: " << ages["Bob"] << endl;
    
    // Using at() for bounds checking
    try {
        cout << "David's age: " << ages.at("David") << endl;  // Will throw
    } catch (const out_of_range& e) {
        cout << "David not found in map: " << e.what() << endl;
    }
    
    // Inserting elements
    ages.insert({"David", 28});
    ages.insert(make_pair("Eve", 32));
    
    cout << "\nAfter adding David and Eve: " << endl;
    for (const auto& pair : ages) {
        cout << pair.first << " -> " << pair.second << endl;
    }
    
    // Checking if key exists
    string name = "Alice";
    if (ages.find(name) != ages.end()) {
        cout << name << " found with age " << ages[name] << endl;
    } else {
        cout << name << " not found" << endl;
    }
    
    // Using count to check existence (returns 0 or 1)
    if (ages.count("Bob")) {
        cout << "Bob exists in the map" << endl;
    }
    
    // Erasing elements
    ages.erase("Charlie");
    cout << "\nAfter erasing Charlie: " << endl;
    for (const auto& pair : ages) {
        cout << pair.first << " -> " << pair.second << endl;
    }
    
    // Iterating with iterators
    cout << "\nUsing iterators: " << endl;
    for (auto it = ages.begin(); it != ages.end(); ++it) {
        cout << it->first << " -> " << it->second << endl;
    }
    
    // Lower and upper bounds
    auto lower = ages.lower_bound("Bob");  // First element >= "Bob"
    auto upper = ages.upper_bound("Bob");  // First element > "Bob"
    
    cout << "\nLower bound for 'Bob': " << lower->first << " -> " << lower->second << endl;
    if (upper != ages.end()) {
        cout << "Upper bound for 'Bob': " << upper->first << " -> " << upper->second << endl;
    }
    
    // Equal range (returns pair of iterators)
    auto range = ages.equal_range("Bob");
    cout << "\nEqual range for 'Bob': ";
    for (auto it = range.first; it != range.second; ++it) {
        cout << it->first << " -> " << it->second << " ";
    }
    cout << endl;
    
    // Size and emptiness
    cout << "\nMap size: " << ages.size() << endl;
    cout << "Map empty: " << ages.empty() << endl;
    
    return 0;
}
```

### Exercise 4: Set Container

Complete this set example:

```cpp
#include <iostream>
#include <set>
#include <unordered_set>
#include <string>
using namespace std;

int main() {
    cout << "=== Set Demo ===" << endl;
    
    // Creating sets
    set<int> numbers = {5, 2, 8, 1, 9, 3};
    set<string> words = {"apple", "banana", "cherry", "date"};
    
    cout << "Numbers set (automatically sorted): ";
    for (const auto& num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    cout << "Words set (automatically sorted): ";
    for (const auto& word : words) {
        cout << word << " ";
    }
    cout << endl;
    
    // Adding elements
    numbers.insert(7);
    numbers.insert(2);  // Duplicate - won't be added
    cout << "After inserting 7 and 2: ";
    for (const auto& num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    // Checking existence
    if (numbers.count(5)) {
        cout << "5 exists in the set" << endl;
    }
    
    if (numbers.find(10) == numbers.end()) {
        cout << "10 does not exist in the set" << endl;
    }
    
    // Erasing elements
    numbers.erase(3);
    cout << "After erasing 3: ";
    for (const auto& num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    cout << "\n=== Unordered Set Demo ===" << endl;
    
    // Unordered set (hash-based, faster lookup but no ordering)
    unordered_set<int> unordered_nums = {5, 2, 8, 1, 9, 3};
    
    cout << "Unordered numbers set: ";
    for (const auto& num : unordered_nums) {
        cout << num << " ";
    }
    cout << endl;
    
    // Operations are similar
    unordered_nums.insert(7);
    unordered_nums.insert(2);  // Duplicate - won't be added
    
    cout << "After inserting 7 and 2: ";
    for (const auto& num : unordered_nums) {
        cout << num << " ";
    }
    cout << endl;
    
    // Lookup is faster on average
    if (unordered_nums.find(8) != unordered_nums.end()) {
        cout << "8 found in unordered set" << endl;
    }
    
    // Size and emptiness
    cout << "Set size: " << unordered_nums.size() << endl;
    cout << "Set empty: " << unordered_nums.empty() << endl;
    
    // Multiset - allows duplicates
    cout << "\n=== Multiset Demo ===" << endl;
    multiset<int> multi_nums = {5, 2, 8, 1, 9, 2, 5, 8};  // Duplicates allowed
    
    cout << "Multiset: ";
    for (const auto& num : multi_nums) {
        cout << num << " ";
    }
    cout << endl;
    
    // Count occurrences
    cout << "Number of 2's: " << multi_nums.count(2) << endl;
    cout << "Number of 7's: " << multi_nums.count(7) << endl;
    
    // Insert duplicates
    multi_nums.insert(2);
    cout << "After inserting another 2: ";
    for (const auto& num : multi_nums) {
        cout << num << " ";
    }
    cout << endl;
    
    return 0;
}
```

## Container Adaptors

Container adaptors provide different interfaces to underlying containers.

### Exercise 5: Container Adaptors

Complete this container adaptor example:

```cpp
#include <iostream>
#include <stack>
#include <queue>
#include <priority_queue>
#include <vector>
#include <list>
using namespace std;

int main() {
    cout << "=== Stack Demo ===" << endl;
    
    // Stack - LIFO (Last In, First Out)
    stack<int> st;
    
    // Push elements
    for (int i = 1; i <= 5; i++) {
        st.push(i * 10);
        cout << "Pushed " << i * 10 << ", size: " << st.size() << endl;
    }
    
    cout << "Stack contents (top to bottom): ";
    stack<int> temp_st = st;  // Copy for display purposes
    while (!temp_st.empty()) {
        cout << temp_st.top() << " ";
        temp_st.pop();
    }
    cout << endl;
    
    // Pop elements
    cout << "Popping elements: ";
    while (!st.empty()) {
        cout << st.top() << " ";
        st.pop();
    }
    cout << endl;
    
    cout << "\n=== Queue Demo ===" << endl;
    
    // Queue - FIFO (First In, First Out)
    queue<string> q;
    
    q.push("First");
    q.push("Second");
    q.push("Third");
    
    cout << "Queue size: " << q.size() << endl;
    cout << "Front element: " << q.front() << endl;
    cout << "Back element: " << q.back() << endl;
    
    cout << "Dequeuing elements: ";
    while (!q.empty()) {
        cout << q.front() << " ";
        q.pop();
    }
    cout << endl;
    
    cout << "\n=== Priority Queue Demo ===" << endl;
    
    // Priority queue - largest element has highest priority
    priority_queue<int> pq;
    
    vector<int> values = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    
    for (int val : values) {
        pq.push(val);
    }
    
    cout << "Priority queue (popping highest priority first): ";
    while (!pq.empty()) {
        cout << pq.top() << " ";
        pq.pop();
    }
    cout << endl;
    
    // Custom priority queue (smallest first)
    priority_queue<int, vector<int>, greater<int>> min_pq;
    
    for (int val : values) {
        min_pq.push(val);
    }
    
    cout << "Min priority queue (smallest first): ";
    while (!min_pq.empty()) {
        cout << min_pq.top() << " ";
        min_pq.pop();
    }
    cout << endl;
    
    // Using different underlying container
    cout << "\n=== Stack with Different Container ===" << endl;
    stack<int, list<int>> list_stack;  // Using list instead of deque
    
    for (int i = 1; i <= 3; i++) {
        list_stack.push(i * 100);
    }
    
    cout << "Stack with list container: ";
    while (!list_stack.empty()) {
        cout << list_stack.top() << " ";
        list_stack.pop();
    }
    cout << endl;
    
    return 0;
}
```

## Iterators

Iterators provide a uniform way to access container elements.

### Exercise 6: Iterator Types and Operations

Complete this iterator example:

```cpp
#include <iostream>
#include <vector>
#include <list>
#include <set>
#include <map>
#include <algorithm>
#include <iterator>
using namespace std;

int main() {
    cout << "=== Iterator Types Demo ===" << endl;
    
    // Different containers have different iterator capabilities
    vector<int> vec = {1, 2, 3, 4, 5};
    list<int> lst = {10, 20, 30, 40, 50};
    set<int> st = {100, 200, 300, 400, 500};
    
    cout << "Vector iteration (random access): ";
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    
    // Random access iterators support arithmetic
    auto vec_it = vec.begin();
    cout << "Element at index 2: " << *(vec_it + 2) << endl;
    cout << "Distance: " << (vec.end() - vec.begin()) << endl;
    
    cout << "List iteration (bidirectional): ";
    for (auto it = lst.begin(); it != lst.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    
    // Bidirectional iterators support decrement
    auto lst_it = lst.end();
    --lst_it;  // Move to last element
    cout << "Last element in list: " << *lst_it << endl;
    
    cout << "Set iteration (bidirectional): ";
    for (auto it = st.begin(); it != st.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    
    // Reverse iterators
    cout << "Vector in reverse: ";
    for (auto rit = vec.rbegin(); rit != vec.rend(); ++rit) {
        cout << *rit << " ";
    }
    cout << endl;
    
    cout << "List in reverse: ";
    for (auto rit = lst.rbegin(); rit != lst.rend(); ++rit) {
        cout << *rit << " ";
    }
    cout << endl;
    
    // Iterator arithmetic with vector
    cout << "\n=== Iterator Arithmetic ===" << endl;
    auto start = vec.begin();
    auto end = vec.end();
    cout << "Vector size calculated with iterators: " << (end - start) << endl;
    
    // Advance iterator
    auto middle = vec.begin();
    advance(middle, 2);  // Move 2 positions
    cout << "Element at middle: " << *middle << endl;
    
    // Next and prev
    auto second = next(vec.begin(), 1);  // Second element
    auto third_from_end = prev(vec.end(), 3);  // Third from end
    cout << "Second element: " << *second << endl;
    cout << "Third from end: " << *third_from_end << endl;
    
    // Insert iterators
    cout << "\n=== Insert Iterators ===" << endl;
    vector<int> source = {1, 2, 3};
    vector<int> destination;
    
    // Back inserter
    copy(source.begin(), source.end(), back_inserter(destination));
    cout << "After back_insert: ";
    for (const auto& elem : destination) cout << elem << " ";
    cout << endl;
    
    // Front inserter (for containers that support it)
    list<int> lst_dest;
    copy(source.begin(), source.end(), front_inserter(lst_dest));
    cout << "After front_insert: ";
    for (const auto& elem : lst_dest) cout << elem << " ";
    cout << endl;
    
    // Stream iterators
    cout << "\n=== Stream Iterators ===" << endl;
    vector<int> stream_source = {10, 20, 30, 40, 50};
    
    // Output stream iterator
    ostream_iterator<int> output_it(cout, " ");
    cout << "Using ostream_iterator: ";
    copy(stream_source.begin(), stream_source.end(), output_it);
    cout << endl;
    
    return 0;
}
```

## STL Algorithms

STL provides a rich set of algorithms that work with iterators.

### Exercise 7: Basic STL Algorithms

Complete this algorithm example with errors:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
using namespace std;

int main() {
    vector<int> numbers = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    
    cout << "Original vector: ";
    for (const auto& num : numbers) cout << num << " ";
    cout << endl;
    
    // Sorting algorithms
    sort(numbers.begin(), numbers.end());
    cout << "After sort: ";
    for (const auto& num : numbers) cout << num << " ";
    cout << endl;
    
    // Reverse sort
    sort(numbers.rbegin(), numbers.rend());
    cout << "After reverse sort: ";
    for (const auto& num : numbers) cout << num << " ";
    cout << endl;
    
    // Reset for more examples
    numbers = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    
    // Searching algorithms
    auto it = find(numbers.begin(), numbers.end(), 8);
    if (it != numbers.end()) {
        cout << "Found 8 at position: " << (it - numbers.begin()) << endl;
    }
    
    // Check if element exists
    if (find(numbers.begin(), numbers.end(), 15) == numbers.end()) {
        cout << "15 not found in vector" << endl;
    }
    
    // Count occurrences
    int count_5 = count(numbers.begin(), numbers.end(), 5);
    cout << "Number of 5's: " << count_5 << endl;
    
    // Count if (with predicate)
    int evens = count_if(numbers.begin(), numbers.end(), 
                         [](int n) { return n % 2 == 0; });
    cout << "Number of even numbers: " << evens << endl;
    
    // Minimum and maximum
    auto min_max = minmax_element(numbers.begin(), numbers.end());
    cout << "Min: " << *(min_max.first) << ", Max: " << *(min_max.second) << endl;
    
    // Numeric algorithms
    int sum = accumulate(numbers.begin(), numbers.end(), 0);
    cout << "Sum: " << sum << endl;
    
    // Product
    int product = accumulate(numbers.begin(), numbers.end(), 1, 
                             multiplies<int>());
    cout << "Product: " << product << endl;
    
    // Partial sum
    vector<int> partial_sums(numbers.size());
    partial_sum(numbers.begin(), numbers.end(), partial_sums.begin());
    cout << "Partial sums: ";
    for (const auto& sum : partial_sums) cout << sum << " ";
    cout << endl;
    
    // Transform
    vector<int> doubled(numbers.size());
    transform(numbers.begin(), numbers.end(), doubled.begin(),
              [](int n) { return n * 2; });
    cout << "Doubled: ";
    for (const auto& num : doubled) cout << num << " ";
    cout << endl;
    
    // For each
    cout << "Squared values: ";
    for_each(numbers.begin(), numbers.end(), 
             [](int n) { cout << n * n << " "; });
    cout << endl;
    
    // Remove algorithms
    vector<int> temp_nums = numbers;
    temp_nums.erase(remove(temp_nums.begin(), temp_nums.end(), 3), temp_nums.end());
    cout << "After removing 3: ";
    for (const auto& num : temp_nums) cout << num << " ";
    cout << endl;
    
    // Remove if
    temp_nums = numbers;
    temp_nums.erase(remove_if(temp_nums.begin(), temp_nums.end(),
                               [](int n) { return n > 5; }), temp_nums.end());
    cout << "After removing elements > 5: ";
    for (const auto& num : temp_nums) cout << num << " ";
    cout << endl;
    
    return 0;
}
```

### Exercise 8: Advanced STL Algorithms

Complete this advanced algorithms example:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>
using namespace std;

int main() {
    vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    cout << "Original vector: ";
    for (const auto& num : numbers) cout << num << " ";
    cout << endl;
    
    // Partition algorithms
    vector<int> nums_copy = numbers;
    partition(nums_copy.begin(), nums_copy.end(), 
              [](int n) { return n % 2 == 0; });  // Evens first, odds last
    cout << "After partition (evens first): ";
    for (const auto& num : nums_copy) cout << num << " ";
    cout << endl;
    
    // Stable partition (preserves relative order)
    nums_copy = numbers;
    stable_partition(nums_copy.begin(), nums_copy.end(),
                     [](int n) { return n % 3 == 0; });  // Multiples of 3 first
    cout << "After stable partition (mult of 3 first): ";
    for (const auto& num : nums_copy) cout << num << " ";
    cout << endl;
    
    // Unique (removes consecutive duplicates)
    vector<int> with_duplicates = {1, 1, 2, 3, 3, 3, 4, 5, 5, 6};
    cout << "Before unique: ";
    for (const auto& num : with_duplicates) cout << num << " ";
    cout << endl;
    
    auto new_end = unique(with_duplicates.begin(), with_duplicates.end());
    with_duplicates.erase(new_end, with_duplicates.end());
    cout << "After unique: ";
    for (const auto& num : with_duplicates) cout << num << " ";
    cout << endl;
    
    // Binary search (requires sorted range)
    sort(numbers.begin(), numbers.end());
    if (binary_search(numbers.begin(), numbers.end(), 7)) {
        cout << "7 found in sorted vector" << endl;
    }
    
    // Lower and upper bound
    auto lower = lower_bound(numbers.begin(), numbers.end(), 6);
    auto upper = upper_bound(numbers.begin(), numbers.end(), 6);
    cout << "Lower bound of 6 at index: " << (lower - numbers.begin()) << endl;
    cout << "Upper bound of 6 at index: " << (upper - numbers.begin()) << endl;
    
    // Merge (requires sorted ranges)
    vector<int> vec1 = {1, 3, 5, 7, 9};
    vector<int> vec2 = {2, 4, 6, 8, 10};
    vector<int> merged(vec1.size() + vec2.size());
    
    merge(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), merged.begin());
    cout << "Merged vectors: ";
    for (const auto& num : merged) cout << num << " ";
    cout << endl;
    
    // Set operations (require sorted ranges)
    vector<int> set1 = {1, 2, 3, 4, 5};
    vector<int> set2 = {4, 5, 6, 7, 8};
    vector<int> intersection(min(set1.size(), set2.size()));
    
    auto it_end = set_intersection(set1.begin(), set1.end(),
                                   set2.begin(), set2.end(),
                                   intersection.begin());
    intersection.erase(it_end, intersection.end());
    cout << "Intersection: ";
    for (const auto& num : intersection) cout << num << " ";
    cout << endl;
    
    // Shuffle
    random_device rd;
    mt19937 gen(rd());
    vector<int> shuffled = numbers;
    shuffle(shuffled.begin(), shuffled.end(), gen);
    cout << "Shuffled: ";
    for (const auto& num : shuffled) cout << num << " ";
    cout << endl;
    
    // Next permutation
    vector<int> perm = {1, 2, 3};
    cout << "Permutations of {1, 2, 3}:" << endl;
    do {
        for (const auto& num : perm) cout << num << " ";
        cout << endl;
    } while (next_permutation(perm.begin(), perm.end()));
    
    // Rotate
    vector<int> rotate_test = {1, 2, 3, 4, 5, 6, 7, 8};
    rotate(rotate_test.begin(), rotate_test.begin() + 3, rotate_test.end());
    cout << "After rotating left by 3: ";
    for (const auto& num : rotate_test) cout << num << " ";
    cout << endl;
    
    return 0;
}
```

## Function Objects and Lambdas

Function objects and lambdas work well with STL algorithms.

### Exercise 9: Function Objects and Lambdas

Complete this example with function objects and lambdas:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
using namespace std;

// Custom function object
struct GreaterThan {
    int threshold;
    
    GreaterThan(int t) : threshold(t) {}
    
    bool operator()(int value) const {
        return value > threshold;
    }
};

// Custom comparator
struct DescendingComparator {
    bool operator()(int a, int b) const {
        return a > b;  // For descending order
    }
};

int main() {
    vector<int> numbers = {15, 3, 9, 1, 12, 7, 20, 4, 18, 6};
    
    cout << "Original vector: ";
    for (const auto& num : numbers) cout << num << " ";
    cout << endl;
    
    // Using lambda with count_if
    int greater_than_10 = count_if(numbers.begin(), numbers.end(),
                                   [](int n) { return n > 10; });
    cout << "Numbers greater than 10: " << greater_than_10 << endl;
    
    // Using custom function object
    int greater_than_5 = count_if(numbers.begin(), numbers.end(),
                                  GreaterThan(5));
    cout << "Numbers greater than 5: " << greater_than_5 << endl;
    
    // Using lambda with remove_if
    vector<int> filtered = numbers;
    filtered.erase(remove_if(filtered.begin(), filtered.end(),
                            [](int n) { return n % 2 == 0; }),  // Remove even numbers
                   filtered.end());
    cout << "After removing even numbers: ";
    for (const auto& num : filtered) cout << num << " ";
    cout << endl;
    
    // Using custom comparator with sort
    vector<int> sorted_desc = numbers;
    sort(sorted_desc.begin(), sorted_desc.end(), DescendingComparator());
    cout << "Sorted in descending order: ";
    for (const auto& num : sorted_desc) cout << num << " ";
    cout << endl;
    
    // Using lambda with sort
    vector<int> sorted_asc = numbers;
    sort(sorted_asc.begin(), sorted_asc.end(), 
         [](int a, int b) { return a < b; });  // Ascending order
    cout << "Sorted in ascending order: ";
    for (const auto& num : sorted_asc) cout << num << " ";
    cout << endl;
    
    // Using std::function to store callable objects
    function<bool(int)> is_even = [](int n) { return n % 2 == 0; };
    function<bool(int)> is_positive = [](int n) { return n > 0; };
    
    int even_count = count_if(numbers.begin(), numbers.end(), is_even);
    int positive_count = count_if(numbers.begin(), numbers.end(), is_positive);
    
    cout << "Even numbers: " << even_count << endl;
    cout << "Positive numbers: " << positive_count << endl;
    
    // Transform with different operations
    vector<int> results(numbers.size());
    
    // Square each element
    transform(numbers.begin(), numbers.end(), results.begin(),
              [](int n) { return n * n; });
    cout << "Squares: ";
    for (const auto& num : results) cout << num << " ";
    cout << endl;
    
    // Cube each element using std::function
    function<int(int)> cube = [](int n) { return n * n * n; };
    transform(numbers.begin(), numbers.end(), results.begin(), cube);
    cout << "Cubes: ";
    for (const auto& num : results) cout << num << " ";
    cout << endl;
    
    // Binary transform (element-wise operation between two containers)
    vector<int> nums1 = {1, 2, 3, 4, 5};
    vector<int> nums2 = {10, 20, 30, 40, 50};
    vector<int> sums(nums1.size());
    
    transform(nums1.begin(), nums1.end(), nums2.begin(), sums.begin(),
              plus<int>());  // Using standard function object
    cout << "Element-wise sums: ";
    for (const auto& sum : sums) cout << sum << " ";
    cout << endl;
    
    // Custom binary operation
    transform(nums1.begin(), nums1.end(), nums2.begin(), sums.begin(),
              [](int a, int b) { return a * b + 1; });
    cout << "Custom operation (a*b+1): ";
    for (const auto& result : sums) cout << result << " ";
    cout << endl;
    
    return 0;
}
```

## Practical Examples

### Exercise 10: Student Management with STL

Create a comprehensive example using STL containers and algorithms:

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <numeric>
#include <functional>
using namespace std;

class Student {
public:
    string name;
    int id;
    vector<double> grades;
    
    Student(const string& n, int i) : name(n), id(i) {}
    
    void addGrade(double grade) {
        if (grade >= 0 && grade <= 100) {
            grades.push_back(grade);
        }
    }
    
    double getAverage() const {
        if (grades.empty()) return 0.0;
        double sum = accumulate(grades.begin(), grades.end(), 0.0);
        return sum / grades.size();
    }
    
    double getHighestGrade() const {
        if (grades.empty()) return 0.0;
        return *max_element(grades.begin(), grades.end());
    }
    
    double getLowestGrade() const {
        if (grades.empty()) return 0.0;
        return *min_element(grades.begin(), grades.end());
    }
    
    size_t getGradeCount() const {
        return grades.size();
    }
    
    void display() const {
        cout << "Name: " << name << ", ID: " << id 
             << ", Avg: " << getAverage() << ", Count: " << getGradeCount() << endl;
    }
};

int main() {
    cout << "=== Student Management System ===" << endl;
    
    // Create students
    vector<Student> students;
    students.emplace_back("Alice Johnson", 1001);
    students.emplace_back("Bob Smith", 1002);
    students.emplace_back("Carol Davis", 1003);
    students.emplace_back("David Wilson", 1004);
    students.emplace_back("Eve Brown", 1005);
    
    // Add grades
    students[0].addGrade(85.5); students[0].addGrade(92.0); students[0].addGrade(78.5);
    students[1].addGrade(76.0); students[1].addGrade(81.5); students[1].addGrade(89.0);
    students[2].addGrade(94.5); students[2].addGrade(91.0); students[2].addGrade(98.5);
    students[3].addGrade(68.0); students[3].addGrade(72.5); students[3].addGrade(75.0);
    students[4].addGrade(88.0); students[4].addGrade(91.5); students[4].addGrade(87.0);
    
    cout << "\nAll Students:" << endl;
    for (const auto& student : students) {
        student.display();
    }
    
    // Sort students by average grade (descending)
    sort(students.begin(), students.end(),
         [](const Student& a, const Student& b) {
             return a.getAverage() > b.getAverage();
         });
    
    cout << "\nStudents sorted by average grade (highest first):" << endl;
    for (const auto& student : students) {
        student.display();
    }
    
    // Find student with highest average
    auto best_student = max_element(students.begin(), students.end(),
                                   [](const Student& a, const Student& b) {
                                       return a.getAverage() < b.getAverage();
                                   });
    cout << "\nBest student: ";
    best_student->display();
    
    // Calculate class statistics
    vector<double> averages;
    transform(students.begin(), students.end(), back_inserter(averages),
              [](const Student& s) { return s.getAverage(); });
    
    double class_average = accumulate(averages.begin(), averages.end(), 0.0) / averages.size();
    double highest_avg = *max_element(averages.begin(), averages.end());
    double lowest_avg = *min_element(averages.begin(), averages.end());
    
    cout << "\nClass Statistics:" << endl;
    cout << "Class Average: " << class_average << endl;
    cout << "Highest Average: " << highest_avg << endl;
    cout << "Lowest Average: " << lowest_avg << endl;
    
    // Find students with averages above class average
    vector<Student> above_average;
    copy_if(students.begin(), students.end(), back_inserter(above_average),
            [class_average](const Student& s) {
                return s.getAverage() > class_average;
            });
    
    cout << "\nStudents above class average:" << endl;
    for (const auto& student : above_average) {
        student.display();
    }
    
    // Count students with specific criteria
    int excellent_students = count_if(students.begin(), students.end(),
                                      [](const Student& s) {
                                          return s.getAverage() >= 90.0;
                                      });
    cout << "\nNumber of excellent students (avg >= 90): " << excellent_students << endl;
    
    // Create a map of students by ID for quick lookup
    map<int, const Student*> student_map;
    for (const auto& student : students) {
        student_map[student.id] = &student;
    }
    
    // Look up specific student
    int lookup_id = 1003;
    auto it = student_map.find(lookup_id);
    if (it != student_map.end()) {
        cout << "\nStudent with ID " << lookup_id << ": ";
        it->second->display();
    }
    
    // Group students by grade range
    map<string, vector<const Student*>> grade_groups;
    
    for (const auto& student : students) {
        double avg = student.getAverage();
        string group;
        if (avg >= 90) group = "A (90-100)";
        else if (avg >= 80) group = "B (80-89)";
        else if (avg >= 70) group = "C (70-79)";
        else if (avg >= 60) group = "D (60-69)";
        else group = "F (below 60)";
        
        grade_groups[group].push_back(&student);
    }
    
    cout << "\nStudents grouped by grade range:" << endl;
    for (const auto& group : grade_groups) {
        cout << group.first << " group:" << endl;
        for (const auto* student : group.second) {
            cout << "  " << student->name << " (avg: " << student->getAverage() << ")" << endl;
        }
    }
    
    return 0;
}
```

## Best Practices for STL

### Exercise 11: STL Best Practices

Demonstrate best practices in STL usage:

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
using namespace std;

int main() {
    cout << "=== STL Best Practices Demo ===" << endl;
    
    // 1. Use const iterators when not modifying
    vector<int> numbers = {1, 2, 3, 4, 5};
    auto find_element = [](const vector<int>& vec, int target) -> bool {
        return find(vec.cbegin(), vec.cend(), target) != vec.cend();  // cbegin/cend for const
    };
    
    cout << "Element 3 found: " << find_element(numbers, 3) << endl;
    
    // 2. Use emplace_back instead of push_back when possible
    vector<pair<string, int>> pairs;
    pairs.emplace_back("first", 1);   // Construct in place
    pairs.emplace_back("second", 2);  // More efficient than push_back(pair<string,int>("second", 2))
    
    cout << "Pairs: ";
    for (const auto& p : pairs) {
        cout << "(" << p.first << ", " << p.second << ") ";
    }
    cout << endl;
    
    // 3. Use reserve to avoid reallocations
    vector<int> reserved_vec;
    reserved_vec.reserve(1000);  // Reserve space upfront
    for (int i = 0; i < 1000; ++i) {
        reserved_vec.push_back(i);
    }
    cout << "Reserved vector size: " << reserved_vec.size() << endl;
    cout << "Reserved vector capacity: " << reserved_vec.capacity() << endl;
    
    // 4. Use algorithms instead of manual loops
    vector<int> calc_nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // Instead of manual loop to sum even numbers
    int sum_evens = 0;
    for (int n : calc_nums) {
        if (n % 2 == 0) sum_evens += n;
    }
    
    // Use algorithm with lambda
    int sum_evens_algo = accumulate(calc_nums.begin(), calc_nums.end(), 0,
                                    [](int sum, int n) { return n % 2 == 0 ? sum + n : sum; });
    
    cout << "Sum of evens (manual): " << sum_evens << endl;
    cout << "Sum of evens (algorithm): " << sum_evens_algo << endl;
    
    // 5. Use appropriate container for the use case
    // For frequent insertions/deletions in middle: use list
    // For sorted unique elements: use set
    // For key-value pairs with fast lookup: use unordered_map
    
    // 6. Use range-based for loops when possible
    cout << "Range-based for loop: ";
    for (const auto& num : calc_nums) {
        cout << num << " ";
    }
    cout << endl;
    
    // 7. Use auto for complex iterator types
    auto it = find(numbers.begin(), numbers.end(), 3);
    if (it != numbers.end()) {
        cout << "Found element: " << *it << endl;
    }
    
    // 8. Use algorithms that return useful information
    auto min_max_pair = minmax_element(numbers.begin(), numbers.end());
    cout << "Min: " << *min_max_pair.first << ", Max: " << *min_max_pair.second << endl;
    
    // 9. Use function objects for reusable logic
    struct IsEven {
        bool operator()(int n) const { return n % 2 == 0; }
    };
    
    auto even_count = count_if(numbers.begin(), numbers.end(), IsEven{});
    cout << "Count of even numbers: " << even_count << endl;
    
    // 10. Use move semantics with containers when appropriate
    vector<string> source = {"hello", "world", "cpp", "stl"};
    vector<string> destination;
    
    // Move elements instead of copying
    for (auto& str : source) {
        destination.push_back(move(str));  // Move each string
    }
    
    cout << "Destination after move: ";
    for (const auto& str : destination) {
        cout << str << " ";
    }
    cout << endl;
    
    cout << "Source after move (should be empty or in valid state): ";
    for (const auto& str : source) {
        cout << "\"" << str << "\" ";  // Moved-from strings are in valid but unspecified state
    }
    cout << endl;
    
    return 0;
}
```

## Summary

In this chapter, you learned:
- Sequential containers: vector, list, deque, array, forward_list
- Associative containers: map, set, multimap, multiset
- Unordered associative containers: unordered_map, unordered_set
- Container adaptors: stack, queue, priority_queue
- Iterator types and operations
- STL algorithms for searching, sorting, and modifying
- Function objects and lambdas with STL
- Best practices for efficient STL usage

## Key Takeaways

- Choose the right container for your specific needs
- Use iterators for uniform access across containers
- Leverage STL algorithms instead of manual loops
- Use const iterators when not modifying elements
- Reserve space in containers to avoid reallocations
- Use emplace functions for efficient element construction
- Use move semantics when transferring ownership
- Algorithms with function objects/lambdas provide flexibility
- Container adaptors provide specialized interfaces

## Common Mistakes to Avoid

1. Using the wrong container for the job (e.g., vector for frequent insertions in middle)
2. Forgetting that associative containers keep elements sorted
3. Not reserving space when the approximate size is known
4. Using algorithms that invalidate iterators at inappropriate times
5. Not using const iterators when appropriate
6. Forgetting that unordered containers have different performance characteristics
7. Using manual loops when STL algorithms would be clearer
8. Not considering the complexity of different operations
9. Forgetting that some algorithms require sorted ranges
10. Not using move semantics when transferring container contents

## Next Steps

Now that you understand STL containers and algorithms, you're ready to learn about memory management and smart pointers in Chapter 11.