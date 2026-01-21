# Chapter 2: Lexical Analysis

## Introduction to Lexical Analysis

Lexical analysis (also called scanning or tokenization) is the first phase of compilation. It reads the source code as a stream of characters and groups them into meaningful sequences called tokens.

## Role of the Lexer

The lexer serves as an interface between the source code and the parser:

```
Character Stream → [LEXER] → Token Stream → [PARSER]
```

### Input
Raw source code as a sequence of characters:
```
int x = 42;
```

### Output
Sequence of tokens:
```
[INT, IDENTIFIER("x"), EQUALS, NUMBER(42), SEMICOLON]
```

## Tokens, Lexemes, and Patterns

### Token
A category or type of lexical unit:
- Keywords: `if`, `while`, `int`
- Identifiers: variable and function names
- Literals: numbers, strings
- Operators: `+`, `-`, `*`, `/`, `==`
- Delimiters: `;`, `{`, `}`, `(`, `)`

### Lexeme
The actual character sequence that matches a token:
- Token: IDENTIFIER, Lexeme: "count"
- Token: NUMBER, Lexeme: "42"
- Token: KEYWORD, Lexeme: "while"

### Pattern
The rule describing possible lexemes for a token:
- Identifier: `[a-zA-Z_][a-zA-Z0-9_]*`
- Integer: `[0-9]+`
- Float: `[0-9]+\.[0-9]+`

## Token Specification

### Regular Expressions

Regular expressions describe patterns for tokens:

```
letter    = [a-zA-Z]
digit     = [0-9]
id        = letter (letter | digit)*
number    = digit+
float     = digit+ . digit+
whitespace = [ \t\n\r]+
```

### Token Classes

Typical token categories:

1. **Keywords**: Reserved words (`if`, `else`, `while`, `for`, etc.)
2. **Identifiers**: User-defined names
3. **Literals**: 
   - Integer literals: `42`, `0xFF`, `0b1010`
   - Float literals: `3.14`, `2.5e-3`
   - String literals: `"hello world"`
   - Character literals: `'a'`, `'\n'`
4. **Operators**:
   - Arithmetic: `+`, `-`, `*`, `/`, `%`
   - Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
   - Logical: `&&`, `||`, `!`
   - Bitwise: `&`, `|`, `^`, `~`, `<<`, `>>`
   - Assignment: `=`, `+=`, `-=`, etc.
5. **Delimiters**: `(`, `)`, `{`, `}`, `[`, `]`, `;`, `,`, `.`
6. **Comments**: Single-line `//` and multi-line `/* */`
7. **Whitespace**: Spaces, tabs, newlines (usually skipped)

## Finite Automata

Regular expressions are implemented using finite automata.

### Deterministic Finite Automaton (DFA)

A DFA has:
- Finite set of states
- Input alphabet
- Transition function (deterministic)
- Start state
- Set of accepting states

Example DFA for identifiers:

```
Pattern: [a-zA-Z][a-zA-Z0-9]*

States: {START, ID, ERROR}
Transitions:
  START --[a-zA-Z]--> ID (accepting)
  ID    --[a-zA-Z0-9]--> ID (accepting)
  ID    --[other]--> ERROR
```

### Non-deterministic Finite Automaton (NFA)

An NFA allows:
- Multiple transitions on same input
- Epsilon (ε) transitions (no input consumed)

NFAs are easier to construct but must be converted to DFAs for efficient execution.

## Lexer Implementation Approaches

### Hand-Written Lexer

Advantages:
- Full control over behavior
- Can be optimized for specific language
- Easy to debug and maintain
- Better error messages

Disadvantages:
- More code to write
- Must handle all edge cases manually

### Lexer Generator (Lex/Flex)

Advantages:
- Concise specification
- Automatically handles automaton construction
- Well-tested implementation

Disadvantages:
- Less control over details
- Generated code may be harder to debug
- Learning curve for specification language

## Lexer Algorithm

### Basic Structure

```
function getNextToken():
    while true:
        skip whitespace and comments
        
        if end of input:
            return EOF token
        
        current_char = peek()
        
        if is_letter(current_char):
            return read_identifier_or_keyword()
        
        if is_digit(current_char):
            return read_number()
        
        if current_char == '"':
            return read_string()
        
        if is_operator(current_char):
            return read_operator()
        
        if is_delimiter(current_char):
            return read_delimiter()
        
        error("Invalid character")
```

### Maximal Munch Principle

The lexer should consume the longest possible token:

```
Input: "ifx = 5"
```

Should be tokenized as:
```
[IDENTIFIER("ifx"), EQUALS, NUMBER(5)]
```

Not:
```
[KEYWORD("if"), IDENTIFIER("x"), EQUALS, NUMBER(5)]
```

The lexer reads as many characters as possible that form a valid token.

## Handling Special Cases

### Keywords vs Identifiers

Most languages have reserved keywords that look like identifiers:

```
Strategy 1: Check after reading identifier
  - Read as identifier
  - Look up in keyword table
  - Return keyword token if found, else identifier

Strategy 2: Separate DFA for each keyword
  - More complex but potentially faster
```

### Lookahead

Some tokens require looking ahead:

```
Input: "=="
  Read '=' → could be EQUALS or EQUALS_EQUALS
  Peek next character
  If '=' → consume it, return EQUALS_EQUALS
  Else → return EQUALS
```

### Comments

Single-line comments:
```
// This is a comment
```
- Start with `//`
- Continue until newline
- Typically skipped by lexer

Multi-line comments:
```
/* This is a
   multi-line comment */
```
- Start with `/*`
- End with `*/`
- Can nest (language-dependent)
- Usually skipped by lexer

### String Literals

Challenges:
- Escape sequences: `\n`, `\t`, `\\`, `\"`
- Multi-line strings
- Unicode characters
- Raw strings (no escape processing)

### Number Literals

Different formats:
```
Decimal:      42, 123
Hexadecimal:  0x2A, 0xFF
Binary:       0b101010
Octal:        0o52
Float:        3.14, 2.5e-3, 1.0e+10
```

### Operators

Multi-character operators require lookahead:
```
+ → PLUS
++ → INCREMENT
+= → PLUS_EQUALS
```

Must check longest match first.

## Error Handling

### Lexical Errors

1. **Invalid Characters**
```
Input: "int x = 5 @ 3;"
       Error: Invalid character '@' at position 10
```

2. **Malformed Tokens**
```
Input: "float x = 3.14.5;"
       Error: Invalid number '3.14.5'
```

3. **Unterminated Strings**
```
Input: "string s = "hello
       Error: Unterminated string literal
```

### Error Recovery

Strategies:
1. **Panic Mode**: Skip characters until valid token found
2. **Delete Invalid Character**: Remove and continue
3. **Insert Missing Character**: Add expected character

## Position Tracking

Track source location for error messages:

```cpp
struct SourceLocation {
    string filename;
    int line;
    int column;
};

struct Token {
    TokenType type;
    string lexeme;
    SourceLocation location;
};
```

Enables helpful error messages:
```
error: file.c:10:5: invalid character '@'
    int x = 5 @ 3;
            ^
```

## Lexer Optimization

### Performance Considerations

1. **Character Classification**
   - Use lookup tables instead of ranges
   - Faster than repeated comparisons

2. **Buffering**
   - Read input in chunks
   - Reduce system calls

3. **State Table Compression**
   - DFA state tables can be large
   - Use compression techniques

4. **Inlining Hot Paths**
   - Inline common token recognition
   - Reduces function call overhead

### Memory Management

1. **String Interning**
   - Store unique strings once
   - Use pointers for comparisons
   - Reduces memory usage

2. **Token Pool**
   - Reuse token objects
   - Avoid repeated allocation

## Example Language Specification

Let's define a simple language:

### Tokens

```
Keywords:
  if, else, while, for, return, int, float, void

Operators:
  +  -  *  /  %           (arithmetic)
  == != <  >  <= >=       (comparison)
  && ||  !                (logical)
  =  += -= *= /=          (assignment)

Delimiters:
  (  )  {  }  [  ]  ;  ,

Identifiers:
  [a-zA-Z_][a-zA-Z0-9_]*

Numbers:
  [0-9]+                  (integer)
  [0-9]+\.[0-9]+          (float)

Strings:
  "([^"\\]|\\.)*"

Comments:
  //.*                    (single-line)
  /\*([^*]|\*[^/])*\*/   (multi-line)

Whitespace:
  [ \t\n\r]+              (ignored)
```

### Regular Expression Precedence

When multiple patterns match, use priority:
1. Longest match wins (maximal munch)
2. If same length, first pattern in specification wins
3. Keywords checked after identifier pattern

## Practical Implementation

### Token Structure

```cpp
enum class TokenType {
    // Keywords
    IF, ELSE, WHILE, FOR, RETURN,
    INT, FLOAT, VOID,
    
    // Operators
    PLUS, MINUS, STAR, SLASH, PERCENT,
    EQUALS_EQUALS, NOT_EQUALS,
    LESS, GREATER, LESS_EQUALS, GREATER_EQUALS,
    AND_AND, OR_OR, NOT,
    EQUALS, PLUS_EQUALS, MINUS_EQUALS,
    
    // Delimiters
    LPAREN, RPAREN, LBRACE, RBRACE,
    LBRACKET, RBRACKET,
    SEMICOLON, COMMA,
    
    // Literals
    IDENTIFIER, NUMBER, FLOAT_LITERAL, STRING,
    
    // Special
    END_OF_FILE, ERROR
};

struct Token {
    TokenType type;
    std::string lexeme;
    int line;
    int column;
    
    // For numeric literals
    union {
        int intValue;
        double floatValue;
    } value;
};
```

### Lexer Class Structure

```cpp
class Lexer {
private:
    std::string source;
    size_t position;
    int line;
    int column;
    
    char peek();
    char advance();
    bool match(char expected);
    
    Token makeToken(TokenType type);
    Token errorToken(const std::string& message);
    
    void skipWhitespace();
    void skipComment();
    
    Token identifier();
    Token number();
    Token string();
    
public:
    Lexer(const std::string& source);
    Token nextToken();
    std::vector<Token> tokenize();
};
```

## Advanced Topics

### Unicode Support

Challenges:
- Multi-byte character encodings (UTF-8, UTF-16)
- Character classification (what's a letter?)
- Normalization (different representations of same character)

### Macro Expansion

Some languages support macros:
```c
#define MAX(a, b) ((a) > (b) ? (a) : (b))
```
- Lexer may need to handle preprocessor directives
- Or separate preprocessing phase

### Conditional Compilation

```c
#ifdef DEBUG
    // debug code
#endif
```
- Lexer may skip disabled sections
- Or pass to preprocessor

## Testing the Lexer

### Test Categories

1. **Basic Tokens**
   - Each token type individually
   - Whitespace handling

2. **Edge Cases**
   - Empty input
   - Single character input
   - Very long tokens

3. **Error Cases**
   - Invalid characters
   - Malformed literals
   - Unterminated strings

4. **Complex Input**
   - Complete programs
   - All token types mixed
   - Comments and strings

### Example Test

```cpp
void testLexer() {
    Lexer lexer("int x = 42;");
    
    Token t1 = lexer.nextToken();
    assert(t1.type == TokenType::INT);
    
    Token t2 = lexer.nextToken();
    assert(t2.type == TokenType::IDENTIFIER);
    assert(t2.lexeme == "x");
    
    Token t3 = lexer.nextToken();
    assert(t3.type == TokenType::EQUALS);
    
    Token t4 = lexer.nextToken();
    assert(t4.type == TokenType::NUMBER);
    assert(t4.value.intValue == 42);
    
    Token t5 = lexer.nextToken();
    assert(t5.type == TokenType::SEMICOLON);
    
    Token t6 = lexer.nextToken();
    assert(t6.type == TokenType::END_OF_FILE);
}
```

## Summary

Key concepts covered:
- Lexical analysis converts characters to tokens
- Regular expressions specify token patterns
- Finite automata implement recognition
- Maximal munch principle for longest match
- Error handling and recovery strategies
- Position tracking for error messages
- Implementation approaches and optimizations

## Next Steps

In Chapter 3, we will study parsing techniques, learning how to build Abstract Syntax Trees from token streams using context-free grammars.

## Code Example

See `examples/01_lexer.cpp` for a complete implementation of a lexer for our simple language.
