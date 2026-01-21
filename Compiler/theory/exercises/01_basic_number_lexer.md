# Exercise 01: Basic Number Lexer

**Difficulty**: Beginner  
**Estimated Time**: 30-45 minutes  
**Topics**: Lexical Analysis, State Machines, Number Recognition

## Learning Objectives

After completing this exercise, you will:
- Understand how to recognize numeric literals
- Implement a simple finite state machine
- Handle different number formats
- Practice basic error handling

## Problem Description

Implement a lexer that can recognize and tokenize different types of numeric literals:

### Supported Number Formats

1. **Integers**: `123`, `0`, `42`
2. **Floating Point**: `3.14`, `0.5`, `123.456`
3. **Scientific Notation**: `1e10`, `2.5e-3`, `1.5E+2`
4. **Hexadecimal**: `0xFF`, `0x1A2B`, `0xDEADBEEF`
5. **Binary**: `0b1010`, `0b11111111`
6. **Octal**: `0o755`, `0o123`

### Token Structure

```cpp
enum class TokenType {
    INTEGER,
    FLOAT,
    INVALID,
    END
};

struct Token {
    TokenType type;
    std::string lexeme;
    union {
        long long intValue;
        double floatValue;
    };
    int position;
};
```

## Requirements

1. **Tokenize** the input string into a vector of tokens
2. **Convert** numeric strings to actual numeric values
3. **Detect** invalid number formats
4. **Report** position of each token
5. **Handle** edge cases (leading zeros, empty input, etc.)

## Function Signature

```cpp
std::vector<Token> tokenizeNumbers(const std::string& input);
```

## Test Cases

### Test 1: Basic Integers
```cpp
Input:  "123 456 0 999"
Output: [INTEGER(123), INTEGER(456), INTEGER(0), INTEGER(999)]
```

### Test 2: Floating Point
```cpp
Input:  "3.14 0.5 123.456"
Output: [FLOAT(3.14), FLOAT(0.5), FLOAT(123.456)]
```

### Test 3: Scientific Notation
```cpp
Input:  "1e10 2.5e-3 1.5E+2"
Output: [FLOAT(1e10), FLOAT(2.5e-3), FLOAT(1.5e2)]
```

### Test 4: Hexadecimal
```cpp
Input:  "0xFF 0x1A 0xDEAD"
Output: [INTEGER(255), INTEGER(26), INTEGER(57005)]
```

### Test 5: Binary
```cpp
Input:  "0b1010 0b1111"
Output: [INTEGER(10), INTEGER(15)]
```

### Test 6: Mixed Formats
```cpp
Input:  "123 3.14 0xFF 0b1010"
Output: [INTEGER(123), FLOAT(3.14), INTEGER(255), INTEGER(10)]
```

### Test 7: Invalid Numbers
```cpp
Input:  "123abc 3.14.15 0x 0b2"
Output: [INVALID, INVALID, INVALID, INVALID]
```

### Test 8: Edge Cases
```cpp
Input:  "0 0.0 00 001"
Output: [INTEGER(0), FLOAT(0.0), INVALID, INVALID]
```

## Hints

### Hint 1: State Machine Design
Think about the states you need:
- START state
- INTEGER state (seen digits)
- FLOAT state (seen decimal point)
- EXPONENT state (seen 'e' or 'E')
- HEX state (seen "0x")
- BINARY state (seen "0b")
- OCTAL state (seen "0o")

### Hint 2: Character Classification
Use helper functions:
```cpp
bool isDigit(char c);
bool isHexDigit(char c);
bool isBinaryDigit(char c);
bool isOctalDigit(char c);
```

### Hint 3: Conversion
Use standard library functions:
```cpp
std::stoll()  // String to long long
std::stod()   // String to double
std::stoll(str, nullptr, 16)  // Hex to long long
std::stoll(str, nullptr, 2)   // Binary to long long
```

### Hint 4: Error Detection
Watch out for:
- Multiple decimal points
- Invalid digits for base (e.g., '2' in binary)
- Exponent without digits
- Missing digits after prefix (0x, 0b, 0o)

### Hint 5: Whitespace Handling
Skip whitespace between tokens using:
```cpp
while (pos < input.length() && std::isspace(input[pos])) {
    pos++;
}
```

## Starter Code

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <cctype>

enum class TokenType {
    INTEGER,
    FLOAT,
    INVALID,
    END
};

struct Token {
    TokenType type;
    std::string lexeme;
    union {
        long long intValue;
        double floatValue;
    };
    int position;
    
    Token(TokenType t, const std::string& lex, int pos)
        : type(t), lexeme(lex), position(pos), intValue(0) {}
};

class NumberLexer {
private:
    std::string input;
    size_t position;
    
    void skipWhitespace() {
        // TODO: Implement
    }
    
    bool isAtEnd() {
        return position >= input.length();
    }
    
    char peek() {
        if (isAtEnd()) return '\0';
        return input[position];
    }
    
    char advance() {
        return input[position++];
    }
    
    Token scanNumber() {
        // TODO: Implement main logic here
        return Token(TokenType::INVALID, "", position);
    }
    
public:
    NumberLexer(const std::string& src) : input(src), position(0) {}
    
    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        
        while (!isAtEnd()) {
            skipWhitespace();
            if (isAtEnd()) break;
            
            tokens.push_back(scanNumber());
        }
        
        return tokens;
    }
};

std::vector<Token> tokenizeNumbers(const std::string& input) {
    NumberLexer lexer(input);
    return lexer.tokenize();
}

// Test function
void runTests() {
    // TODO: Implement test cases
}

int main() {
    runTests();
    return 0;
}
```

## Bonus Challenges

1. **Underscores in Numbers**: Support digit separators like `1_000_000`
2. **Negative Numbers**: Handle negative signs `-123`, `-3.14`
3. **Suffix Support**: Support suffixes like `123L`, `3.14f`, `42U`
4. **Different Bases**: Support base-N numbers like `16#FF#`, `2#1010#`
5. **Error Messages**: Provide detailed error messages with position

## Common Mistakes to Avoid

1. **Not handling EOF properly**: Always check if position is valid
2. **Integer overflow**: Handle numbers too large for data types
3. **Floating point precision**: Be aware of precision limits
4. **Leading zeros**: `001` might be treated as octal in some languages
5. **Forgetting to advance position**: Can cause infinite loops

## Expected Output Format

```
Token 0: INTEGER "123" = 123 at position 0
Token 1: FLOAT "3.14" = 3.14 at position 4
Token 2: INTEGER "0xFF" = 255 at position 9
Token 3: INVALID "123abc" at position 14
```

## Validation Checklist

- [ ] Recognizes decimal integers correctly
- [ ] Recognizes floating point numbers correctly
- [ ] Recognizes scientific notation
- [ ] Recognizes hexadecimal numbers
- [ ] Recognizes binary numbers
- [ ] Recognizes octal numbers
- [ ] Detects invalid number formats
- [ ] Handles whitespace correctly
- [ ] Converts strings to numeric values
- [ ] Reports correct token positions
- [ ] Handles edge cases (0, 0.0, etc.)
- [ ] Code is well-commented
- [ ] All test cases pass

## Solution Location

After attempting the exercise, see `solutions/01_basic_number_lexer.cpp` for a complete implementation with detailed comments.

## Next Exercise

After completing this exercise, move on to **Exercise 02: Identifier Recognition** to learn about tokenizing identifiers and keywords.
