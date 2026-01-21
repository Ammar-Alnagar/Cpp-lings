// Solution: Exercise 01 - Basic Number Lexer
// This is a complete, well-commented solution for the number lexer exercise

#include <iostream>
#include <string>
#include <vector>
#include <cctype>
#include <cstdlib>
#include <cmath>

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
    std::string errorMsg;
    
    Token(TokenType t, const std::string& lex, int pos)
        : type(t), lexeme(lex), position(pos), intValue(0), errorMsg("") {}
    
    std::string toString() const {
        std::string result = "Token: ";
        switch (type) {
            case TokenType::INTEGER:
                result += "INTEGER \"" + lexeme + "\" = " + std::to_string(intValue);
                break;
            case TokenType::FLOAT:
                result += "FLOAT \"" + lexeme + "\" = " + std::to_string(floatValue);
                break;
            case TokenType::INVALID:
                result += "INVALID \"" + lexeme + "\"";
                if (!errorMsg.empty()) result += " (" + errorMsg + ")";
                break;
            case TokenType::END:
                result += "END";
                break;
        }
        result += " at position " + std::to_string(position);
        return result;
    }
};

class NumberLexer {
private:
    std::string input;
    size_t position;
    int tokenStartPos;
    
    // Skip whitespace characters
    void skipWhitespace() {
        while (position < input.length() && std::isspace(input[position])) {
            position++;
        }
    }
    
    // Check if we're at the end of input
    bool isAtEnd() const {
        return position >= input.length();
    }
    
    // Peek at current character without consuming
    char peek() const {
        if (isAtEnd()) return '\0';
        return input[position];
    }
    
    // Peek ahead n characters
    char peekNext(int n = 1) const {
        if (position + n >= input.length()) return '\0';
        return input[position + n];
    }
    
    // Consume and return current character
    char advance() {
        if (isAtEnd()) return '\0';
        return input[position++];
    }
    
    // Check if character matches, consume if true
    bool match(char expected) {
        if (isAtEnd()) return false;
        if (input[position] != expected) return false;
        position++;
        return true;
    }
    
    // Helper: Check if character is a hex digit
    bool isHexDigit(char c) const {
        return std::isdigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    }
    
    // Helper: Check if character is binary digit
    bool isBinaryDigit(char c) const {
        return c == '0' || c == '1';
    }
    
    // Helper: Check if character is octal digit
    bool isOctalDigit(char c) const {
        return c >= '0' && c <= '7';
    }
    
    // Scan hexadecimal number (0xFF)
    Token scanHexNumber() {
        tokenStartPos = position - 2;  // Account for "0x"
        std::string lexeme = "0x";
        
        // Must have at least one hex digit
        if (!isHexDigit(peek())) {
            Token token(TokenType::INVALID, lexeme, tokenStartPos);
            token.errorMsg = "Missing hex digits after 0x";
            return token;
        }
        
        // Consume all hex digits
        while (isHexDigit(peek())) {
            lexeme += advance();
        }
        
        // Check for invalid suffix (like 0xFFG)
        if (std::isalnum(peek())) {
            while (std::isalnum(peek())) {
                lexeme += advance();
            }
            Token token(TokenType::INVALID, lexeme, tokenStartPos);
            token.errorMsg = "Invalid hex number format";
            return token;
        }
        
        // Convert to integer
        Token token(TokenType::INTEGER, lexeme, tokenStartPos);
        try {
            token.intValue = std::stoll(lexeme, nullptr, 16);
        } catch (...) {
            token.type = TokenType::INVALID;
            token.errorMsg = "Hex number too large";
        }
        
        return token;
    }
    
    // Scan binary number (0b1010)
    Token scanBinaryNumber() {
        tokenStartPos = position - 2;  // Account for "0b"
        std::string lexeme = "0b";
        
        // Must have at least one binary digit
        if (!isBinaryDigit(peek())) {
            Token token(TokenType::INVALID, lexeme, tokenStartPos);
            token.errorMsg = "Missing binary digits after 0b";
            return token;
        }
        
        // Consume all binary digits
        while (isBinaryDigit(peek())) {
            lexeme += advance();
        }
        
        // Check for invalid suffix
        if (std::isalnum(peek())) {
            while (std::isalnum(peek())) {
                lexeme += advance();
            }
            Token token(TokenType::INVALID, lexeme, tokenStartPos);
            token.errorMsg = "Invalid binary digit";
            return token;
        }
        
        // Convert to integer
        Token token(TokenType::INTEGER, lexeme, tokenStartPos);
        try {
            token.intValue = std::stoll(lexeme.substr(2), nullptr, 2);
        } catch (...) {
            token.type = TokenType::INVALID;
            token.errorMsg = "Binary number too large";
        }
        
        return token;
    }
    
    // Scan octal number (0o755)
    Token scanOctalNumber() {
        tokenStartPos = position - 2;  // Account for "0o"
        std::string lexeme = "0o";
        
        // Must have at least one octal digit
        if (!isOctalDigit(peek())) {
            Token token(TokenType::INVALID, lexeme, tokenStartPos);
            token.errorMsg = "Missing octal digits after 0o";
            return token;
        }
        
        // Consume all octal digits
        while (isOctalDigit(peek())) {
            lexeme += advance();
        }
        
        // Check for invalid suffix
        if (std::isalnum(peek())) {
            while (std::isalnum(peek())) {
                lexeme += advance();
            }
            Token token(TokenType::INVALID, lexeme, tokenStartPos);
            token.errorMsg = "Invalid octal digit";
            return token;
        }
        
        // Convert to integer
        Token token(TokenType::INTEGER, lexeme, tokenStartPos);
        try {
            token.intValue = std::stoll(lexeme.substr(2), nullptr, 8);
        } catch (...) {
            token.type = TokenType::INVALID;
            token.errorMsg = "Octal number too large";
        }
        
        return token;
    }
    
    // Scan decimal number (integer or float)
    Token scanDecimalNumber() {
        std::string lexeme;
        bool isFloat = false;
        
        // Consume integer part
        while (std::isdigit(peek())) {
            lexeme += advance();
        }
        
        // Check for decimal point
        if (peek() == '.' && std::isdigit(peekNext())) {
            isFloat = true;
            lexeme += advance();  // Consume '.'
            
            // Consume fractional part
            while (std::isdigit(peek())) {
                lexeme += advance();
            }
        }
        
        // Check for exponent
        if (peek() == 'e' || peek() == 'E') {
            isFloat = true;
            lexeme += advance();  // Consume 'e' or 'E'
            
            // Optional sign
            if (peek() == '+' || peek() == '-') {
                lexeme += advance();
            }
            
            // Must have digits after exponent
            if (!std::isdigit(peek())) {
                while (std::isalnum(peek())) {
                    lexeme += advance();
                }
                Token token(TokenType::INVALID, lexeme, tokenStartPos);
                token.errorMsg = "Missing digits in exponent";
                return token;
            }
            
            // Consume exponent digits
            while (std::isdigit(peek())) {
                lexeme += advance();
            }
        }
        
        // Check for invalid suffix
        if (std::isalpha(peek())) {
            while (std::isalnum(peek())) {
                lexeme += advance();
            }
            Token token(TokenType::INVALID, lexeme, tokenStartPos);
            token.errorMsg = "Invalid number format";
            return token;
        }
        
        // Create token
        Token token(isFloat ? TokenType::FLOAT : TokenType::INTEGER, 
                   lexeme, tokenStartPos);
        
        try {
            if (isFloat) {
                token.floatValue = std::stod(lexeme);
            } else {
                token.intValue = std::stoll(lexeme);
            }
        } catch (...) {
            token.type = TokenType::INVALID;
            token.errorMsg = "Number too large";
        }
        
        return token;
    }
    
    // Main number scanning function
    Token scanNumber() {
        tokenStartPos = position;
        
        // Check for prefixed numbers (0x, 0b, 0o)
        if (peek() == '0') {
            char next = peekNext();
            
            if (next == 'x' || next == 'X') {
                advance();  // Consume '0'
                advance();  // Consume 'x'
                return scanHexNumber();
            }
            
            if (next == 'b' || next == 'B') {
                advance();  // Consume '0'
                advance();  // Consume 'b'
                return scanBinaryNumber();
            }
            
            if (next == 'o' || next == 'O') {
                advance();  // Consume '0'
                advance();  // Consume 'o'
                return scanOctalNumber();
            }
            
            // Check for invalid leading zeros (like 001)
            if (std::isdigit(next)) {
                std::string lexeme;
                while (std::isdigit(peek())) {
                    lexeme += advance();
                }
                Token token(TokenType::INVALID, lexeme, tokenStartPos);
                token.errorMsg = "Invalid leading zeros";
                return token;
            }
        }
        
        // Decimal number (integer or float)
        return scanDecimalNumber();
    }
    
public:
    NumberLexer(const std::string& src) : input(src), position(0), tokenStartPos(0) {}
    
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

// Wrapper function
std::vector<Token> tokenizeNumbers(const std::string& input) {
    NumberLexer lexer(input);
    return lexer.tokenize();
}

// Test framework
void runTest(const std::string& name, const std::string& input, 
             const std::vector<Token>& expected) {
    std::cout << "\n=== Test: " << name << " ===\n";
    std::cout << "Input: \"" << input << "\"\n";
    
    auto tokens = tokenizeNumbers(input);
    
    std::cout << "Output:\n";
    for (const auto& token : tokens) {
        std::cout << "  " << token.toString() << "\n";
    }
    
    // Verify
    bool passed = true;
    if (tokens.size() != expected.size()) {
        passed = false;
        std::cout << "FAILED: Expected " << expected.size() 
                  << " tokens, got " << tokens.size() << "\n";
    } else {
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i].type != expected[i].type) {
                passed = false;
                std::cout << "FAILED: Token " << i << " type mismatch\n";
            }
        }
    }
    
    if (passed) {
        std::cout << "PASSED ✓\n";
    }
}

int main() {
    std::cout << "=== Number Lexer Exercise - Solution ===\n";
    
    // Test 1: Basic Integers
    runTest("Basic Integers", "123 456 0 999", {
        Token(TokenType::INTEGER, "123", 0),
        Token(TokenType::INTEGER, "456", 4),
        Token(TokenType::INTEGER, "0", 8),
        Token(TokenType::INTEGER, "999", 10)
    });
    
    // Test 2: Floating Point
    runTest("Floating Point", "3.14 0.5 123.456", {
        Token(TokenType::FLOAT, "3.14", 0),
        Token(TokenType::FLOAT, "0.5", 5),
        Token(TokenType::FLOAT, "123.456", 9)
    });
    
    // Test 3: Scientific Notation
    runTest("Scientific Notation", "1e10 2.5e-3 1.5E+2", {
        Token(TokenType::FLOAT, "1e10", 0),
        Token(TokenType::FLOAT, "2.5e-3", 5),
        Token(TokenType::FLOAT, "1.5E+2", 12)
    });
    
    // Test 4: Hexadecimal
    runTest("Hexadecimal", "0xFF 0x1A 0xDEAD", {
        Token(TokenType::INTEGER, "0xFF", 0),
        Token(TokenType::INTEGER, "0x1A", 5),
        Token(TokenType::INTEGER, "0xDEAD", 10)
    });
    
    // Test 5: Binary
    runTest("Binary", "0b1010 0b1111", {
        Token(TokenType::INTEGER, "0b1010", 0),
        Token(TokenType::INTEGER, "0b1111", 7)
    });
    
    // Test 6: Octal
    runTest("Octal", "0o755 0o123", {
        Token(TokenType::INTEGER, "0o755", 0),
        Token(TokenType::INTEGER, "0o123", 6)
    });
    
    // Test 7: Mixed Formats
    runTest("Mixed Formats", "123 3.14 0xFF 0b1010", {
        Token(TokenType::INTEGER, "123", 0),
        Token(TokenType::FLOAT, "3.14", 4),
        Token(TokenType::INTEGER, "0xFF", 9),
        Token(TokenType::INTEGER, "0b1010", 14)
    });
    
    // Test 8: Invalid Numbers
    runTest("Invalid Numbers", "123abc 0x 0b2", {
        Token(TokenType::INVALID, "123abc", 0),
        Token(TokenType::INVALID, "0x", 7),
        Token(TokenType::INVALID, "0b2", 10)
    });
    
    // Test 9: Edge Cases
    runTest("Edge Cases", "0 0.0", {
        Token(TokenType::INTEGER, "0", 0),
        Token(TokenType::FLOAT, "0.0", 2)
    });
    
    std::cout << "\n=== All Tests Complete ===\n";
    
    return 0;
}
