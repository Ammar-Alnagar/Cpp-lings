/*
 * Chapter 5 Exercise: Arrays and Strings
 *
 * Complete the program that implements a simple grade management system.
 * The program should calculate average, find highest and lowest grades.
 */

#include <iomanip>
#include <iostream>
#include <print>
#include <string>

#include <array>

int main() {
  const int MAX_STUDENTS = 10;
  std::array<double, MAX_STUDENTS> grades{};
  std::array<std::string, MAX_STUDENTS> studentNames{};
  int numStudents = 0;
  double gradesTotal = 0.0;

  std::cout << "Grade Management System\n";
  std::cout << "How many students? ";
  std::cin >> numStudents;

  if (numStudents <= 0 || numStudents > MAX_STUDENTS) {
    std::cout << "Invalid number of students!\n";
    return 1;
  }

  for (int i = 0; i < numStudents; ++i) {
    std::println("please enter the name of student #", i + 1);
    std::cin >> studentNames[i];

    std::println("please enter the grade of student #", i + 1);
    std::cin >> grades[i];
  }

  for (int i = 0; i < numStudents; ++i) {
    gradesTotal += grades[i];
  }

  double avgGrades = gradesTotal / numStudents;

  auto maxIt = std::max_element(grades.begin(), grades.begin() + numStudents);
  auto minIt = std::min_element(grades.begin(), grades.begin() + numStudents);
  double maxGrade = (maxIt != grades.end()) ? *maxIt : 0.0;
  double minGrade = (minIt != grades.end()) ? *minIt : 0.0;

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Average grade: " << avgGrades << "\n";
  std::cout << "Highest grade: " << maxGrade << "\n";
  std::cout << "Lowest grade: " << minGrade << "\n";

  std::cout << "Student grades:\n";
  for (int i = 0; i < numStudents; ++i) {
    std::cout << studentNames[i] << ": " << grades[i] << "\n";
  }

  return 0;
}
