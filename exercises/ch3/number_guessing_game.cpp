/*
 * Chapter 3 Exercise: Control Structures and Loops
 *
 * Complete the program that implements a simple number guessing game.
 * The user should guess a number between 1 and 100.
 */

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <print>

int main() {
  // Seed random number generator
  std::srand(std::time(0));

  // Generate random number between 1 and 100
  int secretNumber = std::rand() % 100 + 1;
  int guess;
  int attempts = 0;

  std::cout << "Welcome to the Number Guessing Game! \n";
  ;

  // TODO: Implement the game loop using while or do-while
  // The loop should:
  // 1. Ask the user for a guess
  // 2. Compare the guess to the secret number
  // 3. Tell the user if their guess is too high, too low, or correct
  // 4. Count the number of attempts
  // 5. Exit when the user guesses correctly
  while (true) {
    std::cout << "Enter your guess (1-100): ";
    std::cin >> guess;
    attempts++;

    if (guess < secretNumber) {
      std::cout << "Too low! Try again." << std::endl;
    } else if (guess > secretNumber) {
      std::cout << "Too high! Try again." << std::endl;
    } else {
      std::cout << "Congratulations! You've guessed the number!" << std::endl;
      break;
    }
  }

  // TODO: After the user wins, print the number of attempts taken
  std::println("You took {} attempts to guess the number.", attempts);

  return 0;
}