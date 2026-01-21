# Final Project: Complete Simple Language Compiler

**Difficulty**: Advanced  
**Estimated Time**: 8-12 hours  
**Topics**: All compiler phases, Integration, LLVM Backend

## Project Overview

Build a complete compiler for "SimpleLang" - a simple imperative programming language with variables, functions, control flow, and basic arithmetic. The compiler will translate SimpleLang source code to executable machine code via LLVM IR.

## Learning Objectives

After completing this project, you will have:
- Built a complete end-to-end compiler
- Integrated lexer, parser, semantic analyzer, and code generator
- Generated executable programs from source code
- Debugged a complex software system
- Experience with real-world compiler architecture

## SimpleLang Language Specification

### Overview

SimpleLang is a statically-typed imperative language with C-like syntax.

### Features

1. **Data Types**: `int`, `float`, `bool`, `void`
2. **Variables**: Declaration and assignment
3. **Functions**: Definition and calls
4. **Control Flow**: `if-else`, `while`, `for`
5. **Operators**: Arithmetic, comparison, logical
6. **I/O**: `print()` function
7. **Comments**: Single-line `//` and multi-line `/* */`

### Grammar

```
Program        → Declaration* EOF

Declaration    → FunctionDecl | VarDecl

FunctionDecl   → Type IDENTIFIER '(' Parameters? ')' Block

VarDecl        → Type IDENTIFIER ('=' Expression)? ';'

Parameters     → Type IDENTIFIER (',' Type IDENTIFIER)*

Block          → '{' Statement* '}'

Statement      → VarDecl
               | ExprStmt
               | IfStmt
               | WhileStmt
               | ForStmt
               | ReturnStmt
               | Block

ExprStmt       → Expression ';'

IfStmt         → 'if' '(' Expression ')' Statement ('else' Statement)?

WhileStmt      → 'while' '(' Expression ')' Statement

ForStmt        → 'for' '(' (VarDecl | ExprStmt | ';') 
                           Expression? ';' 
                           Expression? ')' Statement

ReturnStmt     → 'return' Expression? ';'

Expression     → Assignment

Assignment     → IDENTIFIER '=' Assignment
               | LogicalOr

LogicalOr      → LogicalAnd ('||' LogicalAnd)*

LogicalAnd     → Equality ('&&' Equality)*

Equality       → Comparison (('==' | '!=') Comparison)*

Comparison     → Addition (('<' | '>' | '<=' | '>=') Addition)*

Addition       → Multiplication (('+' | '-') Multiplication)*

Multiplication → Unary (('*' | '/') Unary)*

Unary          → ('!' | '-') Unary | Call

Call           → Primary ('(' Arguments? ')')?

Primary        → NUMBER | IDENTIFIER | 'true' | 'false'
               | '(' Expression ')'

Arguments      → Expression (',' Expression)*

Type           → 'int' | 'float' | 'bool' | 'void'
```

### Example Programs

#### Example 1: Factorial
```c
// SimpleLang: Factorial

int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

int main() {
    int result = factorial(5);
    print(result);  // Should print 120
    return 0;
}
```

#### Example 2: Fibonacci
```c
// SimpleLang: Fibonacci

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    
    int prev = 0;
    int curr = 1;
    int i = 2;
    
    while (i <= n) {
        int next = prev + curr;
        prev = curr;
        curr = next;
        i = i + 1;
    }
    
    return curr;
}

int main() {
    int result = fibonacci(10);
    print(result);  // Should print 55
    return 0;
}
```

#### Example 3: Array Sum (Optional)
```c
// SimpleLang: Sum of numbers 1 to N

int sum_to_n(int n) {
    int sum = 0;
    for (int i = 1; i <= n; i = i + 1) {
        sum = sum + i;
    }
    return sum;
}

int main() {
    int result = sum_to_n(100);
    print(result);  // Should print 5050
    return 0;
}
```

## Project Structure

```
simple_compiler/
├── src/
│   ├── lexer.h              # Lexer interface
│   ├── lexer.cpp            # Lexer implementation
│   ├── parser.h             # Parser interface
│   ├── parser.cpp           # Parser implementation
│   ├── ast.h                # AST node definitions
│   ├── semantic.h           # Semantic analyzer
│   ├── semantic.cpp         # Type checking, symbol table
│   ├── codegen.h            # LLVM IR generator
│   ├── codegen.cpp          # Code generation
│   ├── main.cpp             # Compiler driver
│   └── utils.h              # Utility functions
├── tests/
│   ├── test_lexer.cpp
│   ├── test_parser.cpp
│   ├── test_codegen.cpp
│   └── examples/            # Test SimpleLang programs
├── CMakeLists.txt
├── Makefile
└── README.md
```

## Implementation Phases

### Phase 1: Lexer (2 hours)

Implement tokenization for SimpleLang.

**Token Types**:
```cpp
enum class TokenType {
    // Keywords
    INT, FLOAT, BOOL, VOID,
    IF, ELSE, WHILE, FOR, RETURN,
    TRUE, FALSE,
    
    // Identifiers and literals
    IDENTIFIER, INTEGER_LITERAL, FLOAT_LITERAL,
    
    // Operators
    PLUS, MINUS, STAR, SLASH,
    EQUALS, EQUALS_EQUALS, NOT_EQUALS,
    LESS, GREATER, LESS_EQUALS, GREATER_EQUALS,
    AND_AND, OR_OR, NOT,
    
    // Delimiters
    LPAREN, RPAREN, LBRACE, RBRACE,
    SEMICOLON, COMMA,
    
    // Special
    END_OF_FILE, ERROR
};
```

**Requirements**:
- Recognize all keywords
- Handle identifiers and literals
- Skip whitespace and comments
- Report line/column for errors

### Phase 2: Parser (3 hours)

Build AST from token stream.

**AST Node Types**:
```cpp
// Base
struct ASTNode { virtual ~ASTNode() = default; };

// Declarations
struct FunctionDecl : ASTNode {
    std::string returnType;
    std::string name;
    std::vector<std::pair<std::string, std::string>> parameters;
    std::unique_ptr<Block> body;
};

struct VarDecl : ASTNode {
    std::string type;
    std::string name;
    std::unique_ptr<Expression> initializer;
};

// Statements
struct Statement : ASTNode {};
struct Block : Statement {
    std::vector<std::unique_ptr<Statement>> statements;
};
struct IfStmt : Statement {
    std::unique_ptr<Expression> condition;
    std::unique_ptr<Statement> thenBranch;
    std::unique_ptr<Statement> elseBranch;
};
struct WhileStmt : Statement { /* ... */ };
struct ReturnStmt : Statement { /* ... */ };

// Expressions
struct Expression : ASTNode { 
    std::string type;  // Set during semantic analysis
};
struct BinaryExpr : Expression {
    std::unique_ptr<Expression> left;
    std::string op;
    std::unique_ptr<Expression> right;
};
struct CallExpr : Expression {
    std::string function;
    std::vector<std::unique_ptr<Expression>> arguments;
};
// ... more expression types
```

**Requirements**:
- Implement recursive descent parser
- Build complete AST
- Handle operator precedence
- Report parse errors with location

### Phase 3: Semantic Analysis (2 hours)

Type checking and semantic validation.

**Components**:

1. **Symbol Table**:
```cpp
class SymbolTable {
    struct Symbol {
        std::string name;
        std::string type;
        bool isFunction;
        // For functions: parameter types
        std::vector<std::string> paramTypes;
    };
    
    std::vector<std::map<std::string, Symbol>> scopes;
    
public:
    void enterScope();
    void exitScope();
    void declare(const std::string& name, const Symbol& sym);
    Symbol* lookup(const std::string& name);
};
```

2. **Type Checker**:
```cpp
class TypeChecker {
public:
    void checkProgram(Program* program);
    void checkFunction(FunctionDecl* func);
    void checkStatement(Statement* stmt);
    std::string checkExpression(Expression* expr);
    
private:
    SymbolTable symTable;
    std::string currentFunction;
    std::string currentReturnType;
};
```

**Requirements**:
- Build symbol table
- Check type compatibility
- Verify function signatures
- Ensure all variables declared
- Check return statements

### Phase 4: Code Generation (3 hours)

Generate LLVM IR from AST.

**Code Generator**:
```cpp
class CodeGenerator {
    LLVMContext context;
    IRBuilder<> builder;
    std::unique_ptr<Module> module;
    std::map<std::string, Value*> namedValues;
    std::map<std::string, Function*> functions;
    
public:
    CodeGenerator();
    
    void generateProgram(Program* program);
    Function* generateFunction(FunctionDecl* func);
    void generateStatement(Statement* stmt);
    Value* generateExpression(Expression* expr);
    
    Module* getModule() { return module.get(); }
};
```

**Requirements**:
- Generate LLVM IR for all constructs
- Handle function calls
- Implement control flow (if, while, for)
- Generate printf calls for print()
- Produce valid, verifiable IR

### Phase 5: Integration & Testing (2 hours)

Put it all together and test thoroughly.

**Compiler Driver**:
```cpp
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <source.sl>\n";
        return 1;
    }
    
    // Read source file
    std::string source = readFile(argv[1]);
    
    // Lexing
    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    
    // Parsing
    Parser parser(tokens);
    auto ast = parser.parse();
    if (!ast) return 1;
    
    // Semantic Analysis
    TypeChecker checker;
    if (!checker.check(ast.get())) return 1;
    
    // Code Generation
    CodeGenerator codegen;
    codegen.generate(ast.get());
    
    // Output IR
    codegen.getModule()->print(llvm::outs(), nullptr);
    
    return 0;
}
```

## Test Suite

### Test 1: Basic Arithmetic
```c
int main() {
    int x = 5 + 3 * 2;
    print(x);  // Should print 11
    return 0;
}
```

### Test 2: Conditionals
```c
int main() {
    int x = 10;
    if (x > 5) {
        print(1);
    } else {
        print(0);
    }
    return 0;
}
```

### Test 3: Loops
```c
int main() {
    int i = 0;
    while (i < 5) {
        print(i);
        i = i + 1;
    }
    return 0;
}
```

### Test 4: Functions
```c
int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(5, 3);
    print(result);  // Should print 8
    return 0;
}
```

### Test 5: Recursion
```c
int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

int main() {
    print(factorial(5));  // Should print 120
    return 0;
}
```

## Building and Running

```bash
# Build the compiler
mkdir build && cd build
cmake ..
make

# Compile a SimpleLang program
./simple_compiler ../tests/examples/factorial.sl > output.ll

# Verify the IR
llvm-as output.ll

# Execute with LLVM interpreter
lli output.ll

# Or compile to native executable
llc output.ll -o output.s
gcc output.s -o program
./program
```

## Milestones and Checkpoints

### Milestone 1: Lexer Complete
- [ ] All tokens recognized
- [ ] Comments handled
- [ ] Error reporting works
- [ ] Test: Tokenize factorial.sl successfully

### Milestone 2: Parser Complete
- [ ] All grammar rules implemented
- [ ] AST built correctly
- [ ] Parse errors reported
- [ ] Test: Parse factorial.sl without errors

### Milestone 3: Semantic Analysis Complete
- [ ] Symbol table working
- [ ] Type checking implemented
- [ ] Semantic errors detected
- [ ] Test: Detect type errors in test programs

### Milestone 4: Code Generation Complete
- [ ] IR generated for all constructs
- [ ] IR verifies without errors
- [ ] Test: Generate IR for factorial.sl

### Milestone 5: Integration Complete
- [ ] All phases work together
- [ ] Can compile and run factorial.sl
- [ ] All test programs work
- [ ] Test: Run complete test suite

## Debugging Tips

1. **Lexer**: Print all tokens to verify correctness
2. **Parser**: Print AST structure to check parsing
3. **Semantic**: Print symbol table at each scope
4. **Codegen**: Generate and verify IR incrementally
5. **Integration**: Test each phase independently first

## Bonus Features

If you finish early, add these features:

1. **Arrays**: `int arr[10];`
2. **Strings**: `string s = "hello";`
3. **Floating Point**: Full float support with type coercion
4. **Optimization**: Constant folding, dead code elimination
5. **Better Errors**: Colored output, error recovery
6. **Standard Library**: More built-in functions (sqrt, abs, etc.)
7. **Debugging Info**: Generate DWARF debug information

## Common Challenges

1. **Memory Management**: Use smart pointers throughout
2. **Error Handling**: Propagate errors gracefully
3. **Type System**: Keep track of types consistently
4. **Scope Management**: Handle nested scopes correctly
5. **LLVM API**: Read documentation, check examples

## Evaluation Criteria

Your compiler will be evaluated on:

### Functionality (50%)
- Compiles all test programs correctly
- Generated code produces correct output
- Handles errors gracefully

### Code Quality (25%)
- Well-organized and modular
- Clear separation of concerns
- Good naming and comments

### Error Handling (15%)
- Meaningful error messages
- Correct source locations
- Doesn't crash on invalid input

### Performance (10%)
- Reasonable compilation speed
- Generated code efficiency

## Submission

Submit:
1. Complete source code
2. CMakeLists.txt or Makefile
3. README with build instructions
4. Test examples demonstrating features
5. Brief report (1-2 pages) describing:
   - Architecture decisions
   - Challenges faced
   - What you learned

## Expected Time Breakdown

- Lexer: 2 hours
- Parser: 3 hours
- Semantic Analysis: 2 hours
- Code Generation: 3 hours
- Integration & Testing: 2 hours
- **Total: ~12 hours**

## Resources

- Chapters 1-3 (Theory)
- LLVM chapters
- Example compilers in theory/examples/
- LLVM documentation
- Kaleidoscope tutorial (LLVM)

## Conclusion

This project brings together everything you've learned. Take it step by step, test frequently, and don't hesitate to refer back to the theory chapters and examples.

Good luck building your compiler!
