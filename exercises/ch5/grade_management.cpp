/*
 * Chapter 5 Exercise: Arrays and Strings
 * 
 * Complete the program that implements a simple grade management system.
 * The program should calculate average, find highest and lowest grades.
 */

#include <iostream>
#include <string>
#include <iomanip>

int main() {
    const int MAX_STUDENTS = 100;
    double grades[MAX_STUDENTS];
    std::string studentNames[MAX_STUDENTS];
    int numStudents = 0;
    
    std::cout << "Grade Management System" << std::endl;
    std::cout << "How many students? ";
    std::cin >> numStudents;
    
    if (numStudents <= 0 || numStudents > MAX_STUDENTS) {
        std::cout << "Invalid number of students!" << std::endl;
        return 1;
    }
    
    // TODO: Input student names and grades
    // Use a loop to get names and grades for each student
    
    // TODO: Calculate average grade
    // Use a loop to sum all grades, then divide by number of students
    
    // TODO: Find highest and lowest grades
    // Use loops to find the maximum and minimum values
    
    // TODO: Print results
    // Print average, highest, and lowest grades with 2 decimal places
    
    // TODO: Print all student names with their grades
    // Use a loop to print each student's name and grade
    
    return 0;
}