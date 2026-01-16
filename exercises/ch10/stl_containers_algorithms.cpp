/*
 * Chapter 10 Exercise: STL Containers and Algorithms
 * 
 * Complete the program that demonstrates various STL containers and algorithms.
 * The program should manage a contact list with search and sorting capabilities.
 */

#include <iostream>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <algorithm>
#include <string>
#include <functional>

struct Contact {
    std::string name;
    std::string phoneNumber;
    std::string email;
    
    Contact(const std::string& n, const std::string& phone, const std::string& mail)
        : name(n), phoneNumber(phone), email(mail) {}
    
    // TODO: Implement comparison operators for sorting
    bool operator<(const Contact& other) const {
        // TODO: Compare by name
    }
    
    bool operator==(const Contact& other) const {
        // TODO: Compare by all fields
    }
};

// TODO: Implement a function to print a contact
void printContact(const Contact& contact) {
    // TODO: Print contact information in a formatted way
}

int main() {
    // TODO: Create a vector of contacts
    std::vector<Contact> contacts;
    
    // TODO: Add some sample contacts to the vector
    contacts.emplace_back("John Doe", "555-1234", "john@example.com");
    contacts.emplace_back("Jane Smith", "555-5678", "jane@example.com");
    contacts.emplace_back("Bob Johnson", "555-9012", "bob@example.com");
    // Add more contacts as needed
    
    std::cout << "Original contact list:" << std::endl;
    // TODO: Print all contacts using std::for_each algorithm
    
    // TODO: Sort contacts by name using std::sort
    std::cout << "\nContacts sorted by name:" << std::endl;
    // TODO: Print sorted contacts
    
    // TODO: Find a specific contact using std::find_if
    std::string searchName = "Jane Smith";
    auto found = // TODO: Use std::find_if to find contact by name
    if (found != contacts.end()) {
        std::cout << "\nFound contact: ";
        printContact(*found);
    } else {
        std::cout << "\nContact not found!" << std::endl;
    }
    
    // TODO: Count contacts with specific domain in email using std::count_if
    auto gmailCount = // TODO: Count contacts with "@gmail.com" in email
    std::cout << "\nNumber of Gmail contacts: " << gmailCount << std::endl;
    
    // TODO: Create a set of contact names for fast lookup
    std::set<std::string> contactNames;
    // TODO: Populate the set with names from contacts
    
    // TODO: Check if a name exists in the set
    std::string checkName = "John Doe";
    if (contactNames.find(checkName) != contactNames.end()) {
        std::cout << checkName << " exists in contacts." << std::endl;
    } else {
        std::cout << checkName << " does not exist in contacts." << std::endl;
    }
    
    // TODO: Create a map of contacts indexed by phone number
    std::map<std::string, Contact> phoneToContact;
    // TODO: Populate the map
    
    // TODO: Look up a contact by phone number
    std::string searchPhone = "555-5678";
    auto phoneIt = phoneToContact.find(searchPhone);
    if (phoneIt != phoneToContact.end()) {
        std::cout << "\nFound by phone: ";
        printContact(phoneIt->second);
    }
    
    // TODO: Transform all emails to lowercase using std::transform
    // Create a function or lambda to convert a string to lowercase
    // Apply it to all email addresses
    
    // TODO: Remove contacts with empty phone numbers using std::remove_if and erase
    // contacts.erase(/* TODO: Use remove_if to remove contacts with empty phone numbers */, contacts.end());
    
    // TODO: Create a list of just names using std::transform
    std::list<std::string> namesList;
    // TODO: Resize the list and use transform to populate it with names
    
    std::cout << "\nJust the names:" << std::endl;
    for (const auto& name : namesList) {
        std::cout << name << std::endl;
    }
    
    // TODO: Use a custom comparator with std::sort to sort by email
    std::cout << "\nContacts sorted by email:" << std::endl;
    // TODO: Sort and print
    
    return 0;
}