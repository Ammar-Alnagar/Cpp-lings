# Chapter 3: Parsing and Syntax Analysis

## Introduction to Parsing

Parsing (or syntax analysis) is the second phase of compilation. It takes the token stream from the lexer and builds a hierarchical structure called an Abstract Syntax Tree (AST) that represents the grammatical structure of the program.

## Role of the Parser

```
Token Stream → [PARSER] → Abstract Syntax Tree (AST)
```

### Input
Token stream from lexer:
```
[INT, IDENTIFIER("x"), EQUALS, NUMBER(42), SEMICOLON]
```

### Output
Abstract Syntax Tree representing the structure:
```
VarDecl
  ├── Type: int
  ├── Name: x
  └── Init: 42
```

## Context-Free Grammars

Parsers are based on context-free grammars (CFG), which define the syntax of a language.

### Grammar Components

1. **Terminals**: Tokens from lexer (cannot be decomposed)
   - Keywords: `if`, `while`
   - Operators: `+`, `-`, `*`
   - Literals: numbers, strings

2. **Non-terminals**: Syntactic categories (can be decomposed)
   - Expression, Statement, Declaration

3. **Productions (Rules)**: Define how to construct non-terminals
   - `Expression → Term + Expression`
   - `Expression → Term`

4. **Start Symbol**: Top-level non-terminal
   - Usually `Program` or `CompilationUnit`

### Example Grammar

```
Program    → Statement*
Statement  → VarDecl | Assignment | IfStmt | WhileStmt
VarDecl    → Type IDENTIFIER EQUALS Expression SEMICOLON
Assignment → IDENTIFIER EQUALS Expression SEMICOLON
Expression → Term ((PLUS | MINUS) Term)*
Term       → Factor ((STAR | SLASH) Factor)*
Factor     → NUMBER | IDENTIFIER | LPAREN Expression RPAREN
```

### BNF Notation

Backus-Naur Form is a standard notation for grammars:

```
<program>    ::= <statement>*
<statement>  ::= <var-decl> | <assignment>
<var-decl>   ::= <type> <identifier> "=" <expression> ";"
<expression> ::= <term> ( ("+" | "-") <term> )*
<term>       ::= <factor> ( ("*" | "/") <factor> )*
<factor>     ::= <number> | <identifier> | "(" <expression> ")"
```

### Extended BNF (EBNF)

```
Expression  = Term { ("+" | "-") Term }
Term        = Factor { ("*" | "/") Factor }
Factor      = NUMBER | IDENTIFIER | "(" Expression ")"

// Notation:
// { } = zero or more
// [ ] = optional
// ( ) = grouping
// |   = alternative
```

## Abstract Syntax Tree (AST)

The AST is a tree representation of the source code structure.

### AST vs Parse Tree

**Parse Tree**: Shows every detail of how grammar rules were applied
```
Expression
  ├── Term
  │   └── Factor
  │       └── NUMBER(2)
  ├── PLUS
  └── Term
      └── Factor
          └── NUMBER(3)
```

**AST**: Simplified, semantic structure
```
BinaryOp(+)
  ├── Literal(2)
  └── Literal(3)
```

### AST Node Types

Common node types:

```cpp
// Base node
struct ASTNode {
    virtual ~ASTNode() = default;
};

// Expressions
struct Expression : ASTNode {};

struct BinaryOp : Expression {
    std::unique_ptr<Expression> left;
    TokenType op;
    std::unique_ptr<Expression> right;
};

struct Literal : Expression {
    int value;
};

struct Variable : Expression {
    std::string name;
};

// Statements
struct Statement : ASTNode {};

struct VarDecl : Statement {
    std::string type;
    std::string name;
    std::unique_ptr<Expression> initializer;
};

struct Assignment : Statement {
    std::string variable;
    std::unique_ptr<Expression> value;
};

struct IfStatement : Statement {
    std::unique_ptr<Expression> condition;
    std::unique_ptr<Statement> thenBranch;
    std::unique_ptr<Statement> elseBranch;
};
```

## Parsing Techniques

### Top-Down Parsing

Starts from start symbol and derives the input.

**Types**:
- Recursive Descent (hand-written)
- LL(k) (k tokens lookahead)
- Predictive parsing

**Advantages**:
- Easy to implement
- Good error messages
- Full control

**Disadvantages**:
- Can't handle left recursion
- Limited to LL(k) grammars

### Bottom-Up Parsing

Starts from input and reduces to start symbol.

**Types**:
- LR(0), SLR(1), LALR(1), LR(1)
- Operator precedence

**Advantages**:
- More powerful (handles more grammars)
- Can handle left recursion

**Disadvantages**:
- More complex to implement
- Usually need parser generator
- Error messages can be cryptic

## Recursive Descent Parsing

Hand-written parser using recursive functions.

### Basic Structure

Each non-terminal becomes a function:

```cpp
class Parser {
    std::vector<Token> tokens;
    size_t current = 0;
    
    // Expression → Term ((PLUS | MINUS) Term)*
    std::unique_ptr<Expression> parseExpression() {
        auto expr = parseTerm();
        
        while (match(PLUS) || match(MINUS)) {
            TokenType op = previous().type;
            auto right = parseTerm();
            expr = std::make_unique<BinaryOp>(std::move(expr), op, std::move(right));
        }
        
        return expr;
    }
    
    // Term → Factor ((STAR | SLASH) Factor)*
    std::unique_ptr<Expression> parseTerm() {
        auto expr = parseFactor();
        
        while (match(STAR) || match(SLASH)) {
            TokenType op = previous().type;
            auto right = parseFactor();
            expr = std::make_unique<BinaryOp>(std::move(expr), op, std::move(right));
        }
        
        return expr;
    }
    
    // Factor → NUMBER | IDENTIFIER | LPAREN Expression RPAREN
    std::unique_ptr<Expression> parseFactor() {
        if (match(NUMBER)) {
            return std::make_unique<Literal>(previous().intValue);
        }
        
        if (match(IDENTIFIER)) {
            return std::make_unique<Variable>(previous().lexeme);
        }
        
        if (match(LPAREN)) {
            auto expr = parseExpression();
            consume(RPAREN, "Expect ')' after expression");
            return expr;
        }
        
        throw ParserError("Expect expression");
    }
    
    // Helper functions
    bool match(TokenType type) {
        if (check(type)) {
            advance();
            return true;
        }
        return false;
    }
    
    bool check(TokenType type) {
        if (isAtEnd()) return false;
        return peek().type == type;
    }
    
    Token advance() {
        if (!isAtEnd()) current++;
        return previous();
    }
    
    Token peek() {
        return tokens[current];
    }
    
    Token previous() {
        return tokens[current - 1];
    }
    
    bool isAtEnd() {
        return peek().type == END_OF_FILE;
    }
    
    void consume(TokenType type, const std::string& message) {
        if (check(type)) {
            advance();
            return;
        }
        throw ParserError(message);
    }
};
```

### Handling Operator Precedence

Grammar encodes precedence:

```
// Lowest precedence
Expression  → Assignment
Assignment  → LogicalOr (EQUALS Assignment)?
LogicalOr   → LogicalAnd (OR LogicalAnd)*
LogicalAnd  → Equality (AND Equality)*
Equality    → Comparison ((EQUALS_EQUALS | NOT_EQUALS) Comparison)*
Comparison  → Addition ((LESS | GREATER | LESS_EQUALS | GREATER_EQUALS) Addition)*
Addition    → Multiplication ((PLUS | MINUS) Multiplication)*
Multiplication → Unary ((STAR | SLASH) Unary)*
Unary       → (NOT | MINUS) Unary | Primary
Primary     → NUMBER | IDENTIFIER | LPAREN Expression RPAREN
// Highest precedence
```

Each level handles operators of that precedence.

### Handling Associativity

**Left Associative** (a + b + c = (a + b) + c):
```cpp
// Use iteration
Expression → Term ((PLUS | MINUS) Term)*

std::unique_ptr<Expression> parseExpression() {
    auto left = parseTerm();
    
    while (match(PLUS) || match(MINUS)) {
        TokenType op = previous().type;
        auto right = parseTerm();
        left = std::make_unique<BinaryOp>(std::move(left), op, std::move(right));
    }
    
    return left;
}
```

**Right Associative** (a = b = c = (a = (b = c))):
```cpp
// Use recursion
Assignment → LogicalOr (EQUALS Assignment)?

std::unique_ptr<Expression> parseAssignment() {
    auto expr = parseLogicalOr();
    
    if (match(EQUALS)) {
        auto value = parseAssignment();  // Recursive call
        return std::make_unique<Assignment>(std::move(expr), std::move(value));
    }
    
    return expr;
}
```

## Error Handling

Good parsers provide clear error messages and recover gracefully.

### Panic Mode Recovery

Skip tokens until synchronization point:

```cpp
void synchronize() {
    advance();
    
    while (!isAtEnd()) {
        if (previous().type == SEMICOLON) return;
        
        switch (peek().type) {
            case IF:
            case WHILE:
            case FOR:
            case RETURN:
                return;
        }
        
        advance();
    }
}

std::unique_ptr<Statement> parseStatement() {
    try {
        if (match(IF)) return parseIfStatement();
        if (match(WHILE)) return parseWhileStatement();
        // ... other statements
        
        throw ParserError("Invalid statement");
    } catch (const ParserError& e) {
        reportError(e);
        synchronize();
        return nullptr;
    }
}
```

### Error Production Rules

Add rules for common mistakes:

```cpp
// Allow missing semicolons after certain statements
Statement → Expression (SEMICOLON | /* missing semicolon */)
```

### Error Messages

Provide context and suggestions:

```cpp
void consume(TokenType expected, const std::string& message) {
    if (check(expected)) {
        advance();
        return;
    }
    
    std::stringstream error;
    error << "Error at " << peek().line << ":" << peek().column << ": "
          << message << "\n"
          << "  Got: " << tokenToString(peek().type) << "\n"
          << "  Expected: " << tokenToString(expected);
    
    throw ParserError(error.str());
}
```

## LL(1) Parsing

Predictive parsing with 1 token lookahead.

### First and Follow Sets

**First(α)**: Set of terminals that can appear at start of α

**Follow(A)**: Set of terminals that can appear after A

### LL(1) Parsing Table

```
        |  id    +     *     (     )     $
--------|-----------------------------------
E       | E→TE' |     |     | E→TE'|     |
E'      |       | E'→+TE'|  |     | E'→ε | E'→ε
T       | T→FT' |     |     | T→FT'|     |
T'      |       | T'→ε| T'→*FT'|  | T'→ε | T'→ε
F       | F→id  |     |     | F→(E)|     |
```

## Operator Precedence Parsing

Special technique for expression parsing.

### Precedence Table

```cpp
struct OpInfo {
    int precedence;
    bool leftAssoc;
};

std::unordered_map<TokenType, OpInfo> operators = {
    {PLUS,  {1, true}},
    {MINUS, {1, true}},
    {STAR,  {2, true}},
    {SLASH, {2, true}},
    {POWER, {3, false}}  // Right associative
};
```

### Pratt Parsing (Top-Down Operator Precedence)

Elegant technique for expression parsing:

```cpp
std::unique_ptr<Expression> parseExpression(int precedence = 0) {
    auto left = parsePrimary();
    
    while (!isAtEnd() && getPrecedence(peek().type) > precedence) {
        TokenType op = peek().type;
        advance();
        
        int nextPrecedence = getNextPrecedence(op);
        auto right = parseExpression(nextPrecedence);
        
        left = std::make_unique<BinaryOp>(std::move(left), op, std::move(right));
    }
    
    return left;
}

int getNextPrecedence(TokenType op) {
    int prec = getPrecedence(op);
    if (isLeftAssociative(op)) {
        return prec + 1;  // Left assoc: higher precedence for right
    }
    return prec;  // Right assoc: same precedence
}
```

## LR Parsing

Bottom-up parsing using shift-reduce.

### LR Parsing Actions

1. **Shift**: Push token onto stack
2. **Reduce**: Replace symbols with non-terminal
3. **Accept**: Parsing complete
4. **Error**: Invalid input

### Example: Parsing "id + id"

```
Stack         Input        Action
$             id + id $    Shift
$ id          + id $       Reduce: F → id
$ F           + id $       Reduce: T → F
$ T           + id $       Reduce: E → T
$ E           + id $       Shift
$ E +         id $         Shift
$ E + id      $            Reduce: F → id
$ E + F       $            Reduce: T → F
$ E + T       $            Reduce: E → E + T
$ E           $            Accept
```

### LR Parser Generator (YACC/Bison)

```yacc
%token NUMBER PLUS MINUS STAR SLASH
%%
expr: expr PLUS term   { $$ = $1 + $3; }
    | expr MINUS term  { $$ = $1 - $3; }
    | term             { $$ = $1; }
    ;
    
term: term STAR factor { $$ = $1 * $3; }
    | term SLASH factor{ $$ = $1 / $3; }
    | factor           { $$ = $1; }
    ;
    
factor: NUMBER         { $$ = $1; }
      | LPAREN expr RPAREN { $$ = $2; }
      ;
```

## Ambiguous Grammars

Some grammars allow multiple parse trees for the same input.

### Dangling Else Problem

```
Statement → IF LPAREN Expression RPAREN Statement
          | IF LPAREN Expression RPAREN Statement ELSE Statement
```

Input: `if (a) if (b) x = 1; else x = 2;`

Two interpretations:
```
1. if (a) { if (b) x = 1; else x = 2; }
2. if (a) { if (b) x = 1; } else x = 2;
```

**Solution**: Longest match rule (else binds to nearest if)

### Expression Ambiguity

```
Expression → Expression + Expression
           | Expression * Expression
           | NUMBER
```

Input: `2 + 3 * 4` has multiple parse trees.

**Solution**: Use precedence and associativity rules.

## Advanced Topics

### Left Recursion Elimination

Convert left-recursive grammar to right-recursive:

```
// Left recursive (problematic for recursive descent)
E → E + T | T

// After elimination
E  → T E'
E' → + T E' | ε
```

### Left Factoring

Factor out common prefixes:

```
// Common prefix
Statement → IF LPAREN Expression RPAREN Statement ELSE Statement
          | IF LPAREN Expression RPAREN Statement

// After left factoring
Statement → IF LPAREN Expression RPAREN Statement IfRest
IfRest → ELSE Statement | ε
```

## Parser Performance

### Time Complexity

- Recursive Descent: O(n) for LL(k) grammars
- LR Parsing: O(n)
- Earley Parser: O(n³) worst case, O(n) for most grammars

### Memory Usage

- Recursive Descent: O(d) where d is max parse tree depth
- LR Parsing: O(n) for parse stack

## Testing Parsers

### Test Categories

1. **Valid Programs**: Should parse successfully
2. **Syntax Errors**: Should report errors
3. **Edge Cases**: Empty input, single tokens
4. **Complex Expressions**: Deep nesting, all operators
5. **Error Recovery**: Multiple errors in one file

### Example Tests

```cpp
void testParser() {
    // Valid expression
    auto ast1 = parse("2 + 3 * 4");
    assert(ast1 != nullptr);
    
    // Invalid: missing operand
    try {
        auto ast2 = parse("2 + * 4");
        assert(false);  // Should have thrown
    } catch (ParserError& e) {
        // Expected
    }
    
    // Complex nesting
    auto ast3 = parse("((1 + 2) * (3 - 4)) / 5");
    assert(ast3 != nullptr);
}
```

## Summary

Key concepts covered:
- Role of parsing in compilation
- Context-free grammars and BNF notation
- Abstract Syntax Trees
- Recursive descent parsing
- Operator precedence and associativity
- Error handling and recovery
- LL and LR parsing
- Ambiguous grammars and solutions

## Next Steps

In Chapter 4, we will explore semantic analysis, including type checking, symbol tables, and scope resolution.

## Code Example

See `examples/02_parser.cpp` for a complete recursive descent parser implementation that builds an AST from our lexer's tokens.
