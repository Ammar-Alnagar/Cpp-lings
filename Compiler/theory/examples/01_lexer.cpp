// Compiler Theory: Complete Lexer Implementation
// This example demonstrates a full lexical analyzer for a simple C-like language

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cctype>
#include <sstream>

// Token types for our language
enum class TokenType {
    // Keywords
    IF, ELSE, WHILE, FOR, RETURN,
    INT, FLOAT, VOID, BOOL,
    TRUE, FALSE,
    
    // Operators
    PLUS, MINUS, STAR, SLASH, PERCENT,
    EQUALS_EQUALS, NOT_EQUALS,
    LESS, GREATER, LESS_EQUALS, GREATER_EQUALS,
    AND_AND, OR_OR, NOT,
    EQUALS, PLUS_EQUALS, MINUS_EQUALS, STAR_EQUALS, SLASH_EQUALS,
    
    // Delimiters
    LPAREN, RPAREN, LBRACE, RBRACE, LBRACKET, RBRACKET,
    SEMICOLON, COMMA, DOT,
    
    // Literals
    IDENTIFIER, INTEGER_LITERAL, FLOAT_LITERAL, STRING_LITERAL,
    
    // Special
    END_OF_FILE, ERROR
};

// Convert token type to string for display
std::string tokenTypeToString(TokenType type) {
    static const std::unordered_map<TokenType, std::string> names = {
        {TokenType::IF, "IF"}, {TokenType::ELSE, "ELSE"}, {TokenType::WHILE, "WHILE"},
        {TokenType::FOR, "FOR"}, {TokenType::RETURN, "RETURN"},
        {TokenType::INT, "INT"}, {TokenType::FLOAT, "FLOAT"}, {TokenType::VOID, "VOID"},
        {TokenType::BOOL, "BOOL"}, {TokenType::TRUE, "TRUE"}, {TokenType::FALSE, "FALSE"},
        {TokenType::PLUS, "PLUS"}, {TokenType::MINUS, "MINUS"}, {TokenType::STAR, "STAR"},
        {TokenType::SLASH, "SLASH"}, {TokenType::PERCENT, "PERCENT"},
        {TokenType::EQUALS_EQUALS, "EQUALS_EQUALS"}, {TokenType::NOT_EQUALS, "NOT_EQUALS"},
        {TokenType::LESS, "LESS"}, {TokenType::GREATER, "GREATER"},
        {TokenType::LESS_EQUALS, "LESS_EQUALS"}, {TokenType::GREATER_EQUALS, "GREATER_EQUALS"},
        {TokenType::AND_AND, "AND_AND"}, {TokenType::OR_OR, "OR_OR"}, {TokenType::NOT, "NOT"},
        {TokenType::EQUALS, "EQUALS"}, {TokenType::PLUS_EQUALS, "PLUS_EQUALS"},
        {TokenType::MINUS_EQUALS, "MINUS_EQUALS"}, {TokenType::STAR_EQUALS, "STAR_EQUALS"},
        {TokenType::SLASH_EQUALS, "SLASH_EQUALS"},
        {TokenType::LPAREN, "LPAREN"}, {TokenType::RPAREN, "RPAREN"},
        {TokenType::LBRACE, "LBRACE"}, {TokenType::RBRACE, "RBRACE"},
        {TokenType::LBRACKET, "LBRACKET"}, {TokenType::RBRACKET, "RBRACKET"},
        {TokenType::SEMICOLON, "SEMICOLON"}, {TokenType::COMMA, "COMMA"}, {TokenType::DOT, "DOT"},
        {TokenType::IDENTIFIER, "IDENTIFIER"}, {TokenType::INTEGER_LITERAL, "INTEGER_LITERAL"},
        {TokenType::FLOAT_LITERAL, "FLOAT_LITERAL"}, {TokenType::STRING_LITERAL, "STRING_LITERAL"},
        {TokenType::END_OF_FILE, "END_OF_FILE"}, {TokenType::ERROR, "ERROR"}
    };
    auto it = names.find(type);
    return it != names.end() ? it->second : "UNKNOWN";
}

// Token structure representing a lexical unit
struct Token {
    TokenType type;
    std::string lexeme;
    int line;
    int column;
    
    // Value storage for literals
    union {
        int intValue;
        double floatValue;
    };
    
    Token(TokenType t, const std::string& lex, int ln, int col)
        : type(t), lexeme(lex), line(ln), column(col), intValue(0) {}
    
    std::string toString() const {
        std::stringstream ss;
        ss << "[" << tokenTypeToString(type) << " '" << lexeme << "' "
           << "at " << line << ":" << column << "]";
        return ss.str();
    }
};

// Main Lexer class
class Lexer {
private:
    std::string source;      // Source code to tokenize
    size_t position;         // Current position in source
    size_t start;            // Start of current lexeme
    int line;                // Current line number
    int column;              // Current column number
    
    // Keyword table for fast lookup
    static const std::unordered_map<std::string, TokenType> keywords;
    
    // Check if we've reached the end of source
    bool isAtEnd() const {
        return position >= source.length();
    }
    
    // Peek at current character without consuming
    char peek() const {
        if (isAtEnd()) return '\0';
        return source[position];
    }
    
    // Peek ahead n characters
    char peekNext(int n = 1) const {
        if (position + n >= source.length()) return '\0';
        return source[position + n];
    }
    
    // Consume and return current character
    char advance() {
        if (isAtEnd()) return '\0';
        column++;
        return source[position++];
    }
    
    // Consume current character if it matches expected
    bool match(char expected) {
        if (isAtEnd()) return false;
        if (source[position] != expected) return false;
        advance();
        return true;
    }
    
    // Create a token with current lexeme
    Token makeToken(TokenType type) {
        std::string lexeme = source.substr(start, position - start);
        return Token(type, lexeme, line, column - lexeme.length());
    }
    
    // Create an error token
    Token errorToken(const std::string& message) {
        return Token(TokenType::ERROR, message, line, column);
    }
    
    // Skip whitespace characters
    void skipWhitespace() {
        while (!isAtEnd()) {
            char c = peek();
            switch (c) {
                case ' ':
                case '\r':
                case '\t':
                    advance();
                    break;
                case '\n':
                    line++;
                    column = 0;
                    advance();
                    break;
                case '/':
                    // Check for comments
                    if (peekNext() == '/') {
                        // Single-line comment: skip until end of line
                        while (peek() != '\n' && !isAtEnd()) {
                            advance();
                        }
                    } else if (peekNext() == '*') {
                        // Multi-line comment
                        advance(); // consume '/'
                        advance(); // consume '*'
                        
                        // Skip until we find '*/'
                        while (!isAtEnd()) {
                            if (peek() == '*' && peekNext() == '/') {
                                advance(); // consume '*'
                                advance(); // consume '/'
                                break;
                            }
                            if (peek() == '\n') {
                                line++;
                                column = 0;
                            }
                            advance();
                        }
                    } else {
                        return; // Not a comment, stop skipping
                    }
                    break;
                default:
                    return;
            }
        }
    }
    
    // Scan an identifier or keyword
    Token identifier() {
        // Identifiers: [a-zA-Z_][a-zA-Z0-9_]*
        while (std::isalnum(peek()) || peek() == '_') {
            advance();
        }
        
        // Check if it's a keyword
        std::string text = source.substr(start, position - start);
        auto it = keywords.find(text);
        
        TokenType type = (it != keywords.end()) ? it->second : TokenType::IDENTIFIER;
        return makeToken(type);
    }
    
    // Scan a number (integer or float)
    Token number() {
        // Consume digits
        while (std::isdigit(peek())) {
            advance();
        }
        
        // Check for decimal point
        if (peek() == '.' && std::isdigit(peekNext())) {
            advance(); // consume '.'
            
            // Consume fractional part
            while (std::isdigit(peek())) {
                advance();
            }
            
            // Check for exponent (e.g., 1.5e-3)
            if (peek() == 'e' || peek() == 'E') {
                advance();
                if (peek() == '+' || peek() == '-') {
                    advance();
                }
                while (std::isdigit(peek())) {
                    advance();
                }
            }
            
            Token token = makeToken(TokenType::FLOAT_LITERAL);
            token.floatValue = std::stod(token.lexeme);
            return token;
        }
        
        // Integer literal
        Token token = makeToken(TokenType::INTEGER_LITERAL);
        token.intValue = std::stoi(token.lexeme);
        return token;
    }
    
    // Scan a string literal
    Token string() {
        // String format: "..."
        // advance() already consumed opening quote
        
        std::string value;
        
        while (peek() != '"' && !isAtEnd()) {
            if (peek() == '\n') {
                return errorToken("Unterminated string");
            }
            
            // Handle escape sequences
            if (peek() == '\\') {
                advance(); // consume backslash
                if (isAtEnd()) {
                    return errorToken("Unterminated string");
                }
                
                char escaped = advance();
                switch (escaped) {
                    case 'n': value += '\n'; break;
                    case 't': value += '\t'; break;
                    case 'r': value += '\r'; break;
                    case '\\': value += '\\'; break;
                    case '"': value += '"'; break;
                    case '0': value += '\0'; break;
                    default:
                        value += escaped; // Unknown escape, keep as-is
                }
            } else {
                value += advance();
            }
        }
        
        if (isAtEnd()) {
            return errorToken("Unterminated string");
        }
        
        // Consume closing quote
        advance();
        
        return makeToken(TokenType::STRING_LITERAL);
    }

public:
    // Constructor
    explicit Lexer(const std::string& src) 
        : source(src), position(0), start(0), line(1), column(0) {}
    
    // Get the next token from source
    Token nextToken() {
        skipWhitespace();
        
        start = position;
        
        if (isAtEnd()) {
            return makeToken(TokenType::END_OF_FILE);
        }
        
        char c = advance();
        
        // Identifiers and keywords
        if (std::isalpha(c) || c == '_') {
            return identifier();
        }
        
        // Numbers
        if (std::isdigit(c)) {
            return number();
        }
        
        // Single-character tokens and operators
        switch (c) {
            // Delimiters
            case '(': return makeToken(TokenType::LPAREN);
            case ')': return makeToken(TokenType::RPAREN);
            case '{': return makeToken(TokenType::LBRACE);
            case '}': return makeToken(TokenType::RBRACE);
            case '[': return makeToken(TokenType::LBRACKET);
            case ']': return makeToken(TokenType::RBRACKET);
            case ';': return makeToken(TokenType::SEMICOLON);
            case ',': return makeToken(TokenType::COMMA);
            case '.': return makeToken(TokenType::DOT);
            
            // Operators (with lookahead for multi-character)
            case '+':
                return makeToken(match('=') ? TokenType::PLUS_EQUALS : TokenType::PLUS);
            case '-':
                return makeToken(match('=') ? TokenType::MINUS_EQUALS : TokenType::MINUS);
            case '*':
                return makeToken(match('=') ? TokenType::STAR_EQUALS : TokenType::STAR);
            case '/':
                return makeToken(match('=') ? TokenType::SLASH_EQUALS : TokenType::SLASH);
            case '%':
                return makeToken(TokenType::PERCENT);
            
            case '=':
                return makeToken(match('=') ? TokenType::EQUALS_EQUALS : TokenType::EQUALS);
            case '!':
                return makeToken(match('=') ? TokenType::NOT_EQUALS : TokenType::NOT);
            case '<':
                return makeToken(match('=') ? TokenType::LESS_EQUALS : TokenType::LESS);
            case '>':
                return makeToken(match('=') ? TokenType::GREATER_EQUALS : TokenType::GREATER);
            
            case '&':
                if (match('&')) return makeToken(TokenType::AND_AND);
                return errorToken("Expected '&&'");
            case '|':
                if (match('|')) return makeToken(TokenType::OR_OR);
                return errorToken("Expected '||'");
            
            // String literals
            case '"':
                return string();
            
            default:
                return errorToken(std::string("Unexpected character: ") + c);
        }
    }
    
    // Tokenize entire source and return vector of tokens
    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        
        while (true) {
            Token token = nextToken();
            tokens.push_back(token);
            
            if (token.type == TokenType::END_OF_FILE || token.type == TokenType::ERROR) {
                break;
            }
        }
        
        return tokens;
    }
};

// Keyword initialization (must be outside class)
const std::unordered_map<std::string, TokenType> Lexer::keywords = {
    {"if", TokenType::IF},
    {"else", TokenType::ELSE},
    {"while", TokenType::WHILE},
    {"for", TokenType::FOR},
    {"return", TokenType::RETURN},
    {"int", TokenType::INT},
    {"float", TokenType::FLOAT},
    {"void", TokenType::VOID},
    {"bool", TokenType::BOOL},
    {"true", TokenType::TRUE},
    {"false", TokenType::FALSE}
};

// Test function to demonstrate lexer usage
void testLexer(const std::string& code) {
    std::cout << "\n=== Testing Lexer ===\n";
    std::cout << "Source code:\n" << code << "\n\n";
    std::cout << "Tokens:\n";
    
    Lexer lexer(code);
    std::vector<Token> tokens = lexer.tokenize();
    
    for (const auto& token : tokens) {
        std::cout << token.toString() << "\n";
    }
}

int main() {
    std::cout << "=== Compiler Theory: Lexical Analysis ===\n";
    
    // Test 1: Simple variable declaration
    testLexer("int x = 42;");
    
    // Test 2: Function definition
    testLexer(R"(
int add(int a, int b) {
    return a + b;
}
)");
    
    // Test 3: Control flow
    testLexer(R"(
if (x > 10) {
    y = x * 2;
} else {
    y = x + 5;
}
)");
    
    // Test 4: Operators and expressions
    testLexer("x = (a + b) * c - d / e % f;");
    
    // Test 5: String literals and comments
    testLexer(R"(
// This is a comment
string msg = "Hello, World!\n";
/* Multi-line
   comment */
int z = 100;
)");
    
    // Test 6: Float literals
    testLexer("float pi = 3.14159; float e = 2.71828e0;");
    
    // Test 7: Boolean and logical operators
    testLexer("bool result = (x > 5) && (y <= 10) || !flag;");
    
    // Test 8: Error case - invalid character
    testLexer("int x = 42 @ 3;");
    
    std::cout << "\n=== All lexer tests completed ===\n";
    
    return 0;
}
