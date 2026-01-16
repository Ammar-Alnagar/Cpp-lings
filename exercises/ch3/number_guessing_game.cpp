/*
 * Chapter 3 Exercise: Control Structures and Loops
 * 
 * Complete the program that implements a simple number guessing game.
 * The user should guess a number between 1 and 100.
 */

#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
    // Seed random number generator
    std::srand(std::time(0));
    
    // Generate random number between 1 and 100
    int secretNumber = std::rand() % 100 + 1;
    int guess;
    int attempts = 0;
    
    std::cout << "Welcome to the Number Guessing Game!" << std::endl;
    
    // TODO: Implement the game loop using while or do-while
    // The loop should:
    // 1. Ask the user for a guess
    // 2. Compare the guess to the secret number
    // 3. Tell the user if their guess is too high, too low, or correct
    // 4. Count the number of attempts
    // 5. Exit when the user guesses correctly
    
    // TODO: After the user wins, print the number of attempts taken
    
    return 0;
}