# Exercise 02: Expression Parser

**Difficulty**: Intermediate  
**Estimated Time**: 1-2 hours  
**Topics**: Parsing, Recursive Descent, Operator Precedence, AST Construction

## Learning Objectives

After completing this exercise, you will:
- Implement a recursive descent parser
- Handle operator precedence correctly
- Build Abstract Syntax Trees
- Understand grammar design
- Handle parsing errors gracefully

## Problem Description

Build a parser that can parse mathematical expressions and construct an Abstract Syntax Tree (AST). The parser should handle proper operator precedence and associativity.

### Supported Features

1. **Operators**: `+`, `-`, `*`, `/`, `%` (modulo), `^` (power)
2. **Parentheses**: `(` and `)`
3. **Unary Operators**: `-` (negation), `+` (positive)
4. **Numbers**: Integers and floating-point
5. **Variables**: Identifiers like `x`, `y`, `variable1`

### Operator Precedence (Low to High)

1. Addition, Subtraction: `+`, `-` (left associative)
2. Multiplication, Division, Modulo: `*`, `/`, `%` (left associative)
3. Power: `^` (right associative)
4. Unary: `-x`, `+x` (right associative)
5. Parentheses: `(expression)`

### Grammar

```
Expression   → Assignment
Assignment   → Identifier '=' Assignment | LogicalOr
LogicalOr    → LogicalAnd ('||' LogicalAnd)*
LogicalAnd   → Equality ('&&' Equality)*
Equality     → Comparison (('==' | '!=') Comparison)*
Comparison   → Addition (('<' | '>' | '<=' | '>=') Addition)*
Addition     → Multiplication (('+' | '-') Multiplication)*
Multiplication → Power (('*' | '/' | '%') Power)*
Power        → Unary ('^' Unary)*
Unary        → ('-' | '+') Unary | Primary
Primary      → NUMBER | IDENTIFIER | '(' Expression ')'
```

## AST Node Structure

```cpp
// Base class for all AST nodes
struct ASTNode {
    virtual ~ASTNode() = default;
    virtual void print(int indent = 0) const = 0;
    virtual double evaluate() const = 0;  // For bonus: evaluation
};

// Binary operations: a + b, a * b, etc.
struct BinaryOp : public ASTNode {
    std::unique_ptr<ASTNode> left;
    std::string op;  // "+", "-", "*", "/", "%", "^"
    std::unique_ptr<ASTNode> right;
    
    BinaryOp(std::unique_ptr<ASTNode> l, std::string o, std::unique_ptr<ASTNode> r)
        : left(std::move(l)), op(o), right(std::move(r)) {}
};

// Unary operations: -x, +x
struct UnaryOp : public ASTNode {
    std::string op;  // "-", "+"
    std::unique_ptr<ASTNode> operand;
    
    UnaryOp(std::string o, std::unique_ptr<ASTNode> operand)
        : op(o), operand(std::move(operand)) {}
};

// Number literal
struct NumberLiteral : public ASTNode {
    double value;
    
    NumberLiteral(double v) : value(v) {}
};

// Variable reference
struct Variable : public ASTNode {
    std::string name;
    
    Variable(std::string n) : name(n) {}
};
```

## Requirements

1. **Parse** valid mathematical expressions
2. **Build AST** representing the expression structure
3. **Handle** operator precedence correctly
4. **Support** parentheses for grouping
5. **Detect** syntax errors with helpful messages
6. **Print** the AST in a readable format

## Test Cases

### Test 1: Simple Addition
```
Input:  "2 + 3"
AST:    BinaryOp(+)
          ├── NumberLiteral(2)
          └── NumberLiteral(3)
Expected: 5
```

### Test 2: Operator Precedence
```
Input:  "2 + 3 * 4"
AST:    BinaryOp(+)
          ├── NumberLiteral(2)
          └── BinaryOp(*)
                ├── NumberLiteral(3)
                └── NumberLiteral(4)
Expected: 14
```

### Test 3: Parentheses
```
Input:  "(2 + 3) * 4"
AST:    BinaryOp(*)
          ├── BinaryOp(+)
          │     ├── NumberLiteral(2)
          │     └── NumberLiteral(3)
          └── NumberLiteral(4)
Expected: 20
```

### Test 4: Right Associative Power
```
Input:  "2 ^ 3 ^ 2"
AST:    BinaryOp(^)
          ├── NumberLiteral(2)
          └── BinaryOp(^)
                ├── NumberLiteral(3)
                └── NumberLiteral(2)
Expected: 512 (not 64)
```

### Test 5: Unary Operators
```
Input:  "-3 + 5"
AST:    BinaryOp(+)
          ├── UnaryOp(-)
          │     └── NumberLiteral(3)
          └── NumberLiteral(5)
Expected: 2
```

### Test 6: Complex Expression
```
Input:  "2 * (3 + 4) - 5 / (6 - 1)"
AST:    BinaryOp(-)
          ├── BinaryOp(*)
          │     ├── NumberLiteral(2)
          │     └── BinaryOp(+)
          │           ├── NumberLiteral(3)
          │           └── NumberLiteral(4)
          └── BinaryOp(/)
                ├── NumberLiteral(5)
                └── BinaryOp(-)
                      ├── NumberLiteral(6)
                      └── NumberLiteral(1)
Expected: 13
```

### Test 7: Variables
```
Input:  "x + y * 2"
AST:    BinaryOp(+)
          ├── Variable(x)
          └── BinaryOp(*)
                ├── Variable(y)
                └── NumberLiteral(2)
```

### Test 8: Error Cases
```
Input:  "2 + + 3"      → Error: Unexpected operator
Input:  "2 + (3"       → Error: Missing closing parenthesis
Input:  "(2 + 3"       → Error: Missing closing parenthesis
Input:  "2 + * 3"      → Error: Unexpected operator
Input:  "2 3"          → Error: Expected operator
```

## Hints

### Hint 1: Recursive Descent Structure
Each precedence level gets its own function:
```cpp
std::unique_ptr<ASTNode> parseExpression();
std::unique_ptr<ASTNode> parseAddition();
std::unique_ptr<ASTNode> parseMultiplication();
std::unique_ptr<ASTNode> parsePower();
std::unique_ptr<ASTNode> parseUnary();
std::unique_ptr<ASTNode> parsePrimary();
```

### Hint 2: Left Associative Parsing
Use a loop for left-associative operators:
```cpp
std::unique_ptr<ASTNode> parseAddition() {
    auto left = parseMultiplication();
    
    while (match(PLUS) || match(MINUS)) {
        std::string op = previous().lexeme;
        auto right = parseMultiplication();
        left = std::make_unique<BinaryOp>(std::move(left), op, std::move(right));
    }
    
    return left;
}
```

### Hint 3: Right Associative Parsing
Use recursion for right-associative operators:
```cpp
std::unique_ptr<ASTNode> parsePower() {
    auto left = parseUnary();
    
    if (match(POWER)) {
        std::string op = previous().lexeme;
        auto right = parsePower();  // Recursive call for right associativity
        return std::make_unique<BinaryOp>(std::move(left), op, std::move(right));
    }
    
    return left;
}
```

### Hint 4: Error Handling
Check for expected tokens and throw meaningful errors:
```cpp
void consume(TokenType type, const std::string& message) {
    if (check(type)) {
        advance();
        return;
    }
    
    throw ParseError(message + " at position " + std::to_string(current));
}
```

### Hint 5: AST Printing
Implement indented tree printing:
```cpp
void print(int indent = 0) const override {
    std::string spacing(indent * 2, ' ');
    std::cout << spacing << "BinaryOp(" << op << ")\n";
    left->print(indent + 1);
    right->print(indent + 1);
}
```

## Starter Code

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

// Token types
enum class TokenType {
    NUMBER, IDENTIFIER,
    PLUS, MINUS, STAR, SLASH, PERCENT, POWER,
    LPAREN, RPAREN,
    END, INVALID
};

struct Token {
    TokenType type;
    std::string lexeme;
    double value;  // For numbers
    int position;
};

// AST Nodes (define as shown above)
struct ASTNode { /* ... */ };
struct BinaryOp : public ASTNode { /* ... */ };
struct UnaryOp : public ASTNode { /* ... */ };
struct NumberLiteral : public ASTNode { /* ... */ };
struct Variable : public ASTNode { /* ... */ };

class Parser {
private:
    std::vector<Token> tokens;
    size_t current;
    
    Token peek() const {
        return tokens[current];
    }
    
    Token previous() const {
        return tokens[current - 1];
    }
    
    bool isAtEnd() const {
        return peek().type == TokenType::END;
    }
    
    Token advance() {
        if (!isAtEnd()) current++;
        return previous();
    }
    
    bool check(TokenType type) const {
        if (isAtEnd()) return false;
        return peek().type == type;
    }
    
    bool match(TokenType type) {
        if (check(type)) {
            advance();
            return true;
        }
        return false;
    }
    
    // TODO: Implement parsing functions
    std::unique_ptr<ASTNode> parseExpression() {
        // TODO
        return nullptr;
    }
    
    std::unique_ptr<ASTNode> parseAddition() {
        // TODO
        return nullptr;
    }
    
    // ... more parsing functions ...
    
public:
    Parser(const std::vector<Token>& toks) : tokens(toks), current(0) {}
    
    std::unique_ptr<ASTNode> parse() {
        try {
            return parseExpression();
        } catch (const std::exception& e) {
            std::cerr << "Parse error: " << e.what() << "\n";
            return nullptr;
        }
    }
};
```

## Bonus Challenges

1. **Evaluation**: Implement `evaluate()` to compute the result
2. **Assignment**: Support variable assignment `x = 5 + 3`
3. **Functions**: Support function calls `sin(x)`, `max(a, b)`
4. **Comparison**: Add comparison operators `<`, `>`, `==`, etc.
5. **Error Recovery**: Continue parsing after errors to find multiple issues
6. **Optimization**: Implement constant folding in the AST

## Common Mistakes to Avoid

1. **Wrong precedence**: Not following the grammar hierarchy
2. **Wrong associativity**: Using loop instead of recursion for right-associative
3. **Missing error checks**: Not validating expected tokens
4. **Memory leaks**: Not using smart pointers properly
5. **Infinite recursion**: Incorrect base cases in parsing functions

## Evaluation Criteria

Your solution will be evaluated on:
- Correctness of operator precedence
- Proper AST structure
- Error handling quality
- Code organization and readability
- Edge case handling

## Next Exercise

After completing this exercise, move on to **Exercise 03: Statement Parser** to learn about parsing statements and control flow.
