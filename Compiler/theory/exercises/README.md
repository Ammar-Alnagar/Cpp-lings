# Compiler Theory Exercises

This directory contains comprehensive exercises for learning compiler construction. Each exercise builds on previous concepts and gradually increases in difficulty.

## Exercise Structure

Each exercise includes:
- **Problem Description**: What you need to implement
- **Learning Objectives**: What you'll learn
- **Difficulty Level**: Beginner, Intermediate, or Advanced
- **Hints**: Guidance without giving away the solution
- **Test Cases**: Input/output examples to verify your solution
- **Solution**: Complete working implementation (in solutions/ directory)

## Prerequisites

Before starting these exercises, you should:
- Complete reading the corresponding theory chapters
- Understand basic C++ programming
- Be familiar with data structures (arrays, vectors, maps)
- Have studied the provided examples

## Exercise Categories

### 1. Lexical Analysis (Exercises 01-05)
- Tokenizing different input formats
- Handling edge cases and errors
- Building state machines
- Optimizing lexer performance

### 2. Parsing (Exercises 06-10)
- Building recursive descent parsers
- Constructing Abstract Syntax Trees
- Handling operator precedence
- Error recovery

### 3. Semantic Analysis (Exercises 11-15)
- Symbol tables and scoping
- Type checking
- Name resolution
- Semantic validation

### 4. Intermediate Representation (Exercises 16-20)
- Converting AST to IR
- Three-address code generation
- Basic optimizations

### 5. Integrated Projects (Exercises 21-25)
- Complete mini-compilers
- Language extensions
- Tool building

## How to Use These Exercises

### Step 1: Read the Exercise
Carefully read the problem description and understand what's required.

### Step 2: Plan Your Solution
Think about data structures and algorithms before coding.

### Step 3: Implement
Write your solution in the exercises directory.

### Step 4: Test
Run the provided test cases and verify correctness.

### Step 5: Review Solution
Compare with the reference solution, understand different approaches.

### Step 6: Extend
Try the bonus challenges for deeper understanding.

## Building Exercises

```bash
# From Compiler directory
cd theory/exercises

# Compile a specific exercise
g++ -std=c++17 -Wall -Wextra exercise01_basic_lexer.cpp -o ex01

# Run it
./ex01

# Or use the provided makefile
make exercise01
make test-exercise01
```

## Grading Criteria

Exercises are evaluated on:
- **Correctness** (40%): Does it produce correct output?
- **Code Quality** (30%): Is it well-structured and readable?
- **Error Handling** (15%): Does it handle edge cases?
- **Performance** (15%): Is it reasonably efficient?

## Getting Help

If you're stuck:
1. Review the relevant theory chapter
2. Look at the hints provided
3. Study the example implementations
4. Break the problem into smaller parts
5. Check the solution (last resort)

## Difficulty Levels

**Beginner**: Basic concepts, follows examples closely
**Intermediate**: Requires combining multiple concepts
**Advanced**: Significant design decisions, optimization required

## Recommended Order

Follow the exercises in numerical order for best learning progression. Each exercise assumes knowledge from previous ones.

## Time Estimates

- Beginner exercises: 30-60 minutes each
- Intermediate exercises: 1-2 hours each
- Advanced exercises: 2-4 hours each
- Integrated projects: 4-8 hours each

## Additional Resources

- Theory chapters in `../`
- Example implementations in `../examples/`
- Solutions in `solutions/`
- Test data in `test_data/`

## Contributing

Found an issue or have a suggestion? Feel free to improve these exercises!

## Exercise List

### Lexical Analysis
- **Exercise 01**: Basic Number Lexer
- **Exercise 02**: Identifier Recognition
- **Exercise 03**: String Literal Handling
- **Exercise 04**: Multi-line Comments
- **Exercise 05**: Complete Language Lexer

### Parsing
- **Exercise 06**: Expression Parser
- **Exercise 07**: Statement Parser
- **Exercise 08**: Function Parser
- **Exercise 09**: Control Flow Parser
- **Exercise 10**: Complete Language Parser

### Semantic Analysis
- **Exercise 11**: Symbol Table Implementation
- **Exercise 12**: Scope Management
- **Exercise 13**: Type Checker
- **Exercise 14**: Name Resolution
- **Exercise 15**: Semantic Validator

### Intermediate Representation
- **Exercise 16**: AST to Three-Address Code
- **Exercise 17**: Basic Block Construction
- **Exercise 18**: Control Flow Graph
- **Exercise 19**: SSA Transformation
- **Exercise 20**: Dead Code Elimination

### Integrated Projects
- **Exercise 21**: Calculator Language
- **Exercise 22**: Simple Imperative Language
- **Exercise 23**: Language with Functions
- **Exercise 24**: Type System Implementation
- **Exercise 25**: Optimizing Compiler

Let's start building compilers!
