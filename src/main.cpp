#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Token类型枚举
enum class TokenType {
    Identifier,
    Number,
    Keyword,
    Operator,
    Separator,
    EndOfFile,
    CharLiteral,
    BoolLiteral,
    StringLiteral,
};

struct Token {
    TokenType type;
    std::string lexeme;
    int line;
    int column;
};

// 值类型枚举
enum class ValueType { Int, Float, Bool, Char, String, Void };

// AST节点类型枚举
enum class ASTNodeType {
    Program,
    Declaration,
    Assignment,
    BinaryOp,
    UnaryOp,
    If,
    While,
    For,
    Block,
    Return,
    FunctionCall,
    Variable,
    Literal,
    Print,
    Switch,
    Case,
    Default,
    DoWhile,
    Break,
    Continue,
    ConstDeclaration,
    TernaryOp
};

struct ASTNode {
    ASTNodeType type;
    std::string value;
    ValueType dataType;
    std::vector<std::shared_ptr<ASTNode>> children;
    int line, column;

    ASTNode(ASTNodeType t, const std::string& v = "", ValueType dt = ValueType::Void, int l = 0, int c = 0)
        : type(t), value(v), dataType(dt), line(l), column(c) {}
};

// 三地址码结构
struct Tac {
    std::string op;
    std::string arg1;
    std::string arg2;
    std::string result;
};

// 词法分析器
class Lexer {
  public:
    explicit Lexer(std::string src) : source(std::move(src)) {}

    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        while (true) {
            skipWhitespaceAndComments();
            if (isAtEnd()) break;
            char c = peek();
            int startLine = line;
            int startCol = column + 1;

            if (std::isalpha(c) || c == '_') {
                tokens.push_back(identifier(startLine, startCol));
            } else if (std::isdigit(c)) {
                tokens.push_back(number(startLine, startCol));
            } else if (c == '\'') {
                tokens.push_back(characterLiteral(startLine, startCol));
            } else if (c == '"') {
                tokens.push_back(stringLiteral(startLine, startCol));
            } else if (c == ':') {
                // 冒号单独处理，作为Operator
                tokens.push_back(Token{TokenType::Operator, ":", startLine, startCol});
                advance();
            } else if (c == '?') {
                // 问号单独处理，作为Operator
                tokens.push_back(Token{TokenType::Operator, "?", startLine, startCol});
                advance();
            } else if (singleCharSeparators.count(c)) {
                tokens.push_back(singleCharToken(TokenType::Separator, std::string(1, c), startLine, startCol));
                advance();
            } else {
                tokens.push_back(operatorToken(startLine, startCol));
            }
        }
        tokens.push_back(Token{TokenType::EndOfFile, "<eof>", line, column});
        return tokens;
    }

  private:
    std::string source;
    size_t current = 0;
    int line = 1;
    int column = 0;
    const std::unordered_map<std::string, TokenType> keywords{
        {"int", TokenType::Keyword},   {"float", TokenType::Keyword}, {"bool", TokenType::Keyword},
        {"char", TokenType::Keyword},  {"const", TokenType::Keyword}, {"if", TokenType::Keyword},
        {"else", TokenType::Keyword},  {"while", TokenType::Keyword}, {"for", TokenType::Keyword},
        {"do", TokenType::Keyword},    {"switch", TokenType::Keyword}, {"case", TokenType::Keyword},
        {"default", TokenType::Keyword}, {"break", TokenType::Keyword}, {"continue", TokenType::Keyword},
        {"return", TokenType::Keyword},
        {"true", TokenType::BoolLiteral}, {"false", TokenType::BoolLiteral}};
    const std::unordered_map<std::string, std::string> twoCharOps{
        {"==", "=="}, {"!=", "!="}, {"<=", "<="}, {">=", ">="}, {"&&", "&&"}, {"||", "||"},
        {"+=", "+="}, {"-=", "-="}, {"*=", "*="}, {"/=", "/="}, {"%=", "%="}, {"++", "++"},
        {"--", "--"}};
    const std::unordered_set<char> singleCharSeparators{'(', ')', '{', '}', ';', ',', '[', ']'};

    bool isAtEnd() const { return current >= source.size(); }
    char peek() const { return source[current]; }
    char peekNext() const { return (current + 1 < source.size()) ? source[current + 1] : '\0'; }
    char advance() {
        column++;
        return source[current++];
    }

    void skipWhitespaceAndComments() {
        while (!isAtEnd()) {
            char c = peek();
            if (c == ' ' || c == '\r' || c == '\t') {
                advance();
            } else if (c == '\n') {
                advance();
                line++;
                column = 0;
            } else if (c == '/' && peekNext() == '/') {
                while (!isAtEnd() && peek() != '\n') advance();
            } else if (c == '/' && peekNext() == '*') {
                advance();
                advance();
                while (!isAtEnd()) {
                    if (peek() == '*' && peekNext() == '/') {
                        advance();
                        advance();
                        break;
                    }
                    if (peek() == '\n') {
                        line++;
                        column = 0;
                    }
                    advance();
                }
            } else {
                break;
            }
        }
    }

    Token identifier(int startLine, int startCol) {
        std::string lexeme;
        while (!isAtEnd() && (std::isalnum(peek()) || peek() == '_')) {
            lexeme += advance();
        }
        if (keywords.count(lexeme)) {
            TokenType type = keywords.at(lexeme);
            return Token{type, lexeme, startLine, startCol};
        }
        return Token{TokenType::Identifier, lexeme, startLine, startCol};
    }

    Token number(int startLine, int startCol) {
        std::string lexeme;
        bool hasDot = false;
        while (!isAtEnd() && (std::isdigit(peek()) || peek() == '.')) {
            if (peek() == '.') {
                if (hasDot) break;
                hasDot = true;
            }
            lexeme += advance();
        }
        return Token{TokenType::Number, lexeme, startLine, startCol};
    }

    Token characterLiteral(int startLine, int startCol) {
        std::string lexeme;
        advance();

        while (!isAtEnd() && peek() != '\'') {
            if (peek() == '\\') {
                advance();
                switch (peek()) {
                    case 'n': lexeme += '\n'; break;
                    case 't': lexeme += '\t'; break;
                    case '\\': lexeme += '\\'; break;
                    case '\'': lexeme += '\''; break;
                    case '"': lexeme += '"'; break;
                    default: lexeme += '\\'; lexeme += peek(); break;
                }
            } else {
                lexeme += peek();
            }
            advance();
        }

        if (peek() == '\'') {
            advance();
        } else {
            throw std::runtime_error("未闭合的字符常量，位置: line " +
                                   std::to_string(startLine) + " col " +
                                   std::to_string(startCol));
        }

        if (lexeme.length() == 1) {
            int ascii = static_cast<int>(lexeme[0]);
            return Token{TokenType::CharLiteral, std::to_string(ascii), startLine, startCol};
        } else if (lexeme.length() == 0) {
            throw std::runtime_error("空字符常量，位置: line " +
                                   std::to_string(startLine) + " col " +
                                   std::to_string(startCol));
        } else {
            return Token{TokenType::CharLiteral, std::to_string(static_cast<int>(lexeme[0])), startLine, startCol};
        }
    }

    Token stringLiteral(int startLine, int startCol) {
        std::string lexeme;
        advance();

        while (!isAtEnd() && peek() != '"') {
            if (peek() == '\\') {
                advance();
                switch (peek()) {
                    case 'n': lexeme += '\n'; break;
                    case 't': lexeme += '\t'; break;
                    case '\\': lexeme += '\\'; break;
                    case '\'': lexeme += '\''; break;
                    case '"': lexeme += '"'; break;
                    default: lexeme += '\\'; lexeme += peek(); break;
                }
                advance();
            } else {
                lexeme += advance();
            }
        }

        if (peek() == '"') {
            advance();
        } else {
            throw std::runtime_error("未闭合的字符串常量，位置: line " +
                                   std::to_string(startLine) + " col " +
                                   std::to_string(startCol));
        }

        return Token{TokenType::StringLiteral, lexeme, startLine, startCol};
    }

    Token singleCharToken(TokenType type, const std::string& lexeme, int startLine, int startCol) {
        return Token{type, lexeme, startLine, startCol};
    }

    Token operatorToken(int startLine, int startCol) {
        std::string lexeme;
        lexeme += advance();
        
        // 检查是否为双字符运算符
        if (!isAtEnd()) {
            std::string twoChars = lexeme + peek();
            if (twoCharOps.count(twoChars)) {
                lexeme += advance();
                return Token{TokenType::Operator, lexeme, startLine, startCol};
            }
        }
        
        // 如果是已经单独处理的字符，不应该走到这里
        if (lexeme == ":" || lexeme == "?") {
            return Token{TokenType::Operator, lexeme, startLine, startCol};
        }
        
        // 检查是否为有效的单字符运算符
        static const std::unordered_set<char> validOps = {'+', '-', '*', '/', '%', '=', '<', '>', '!', '&', '|'};
        if (validOps.count(lexeme[0])) {
            return Token{TokenType::Operator, lexeme, startLine, startCol};
        }
        
        throw std::runtime_error("Unexpected character '" + lexeme + "' at line " + 
                               std::to_string(startLine) + " column " + 
                               std::to_string(startCol));
    }
};

// 符号表
class SymbolTable {
  public:
    struct Entry {
        std::string name;
        ValueType type;
        int scopeDepth;
        int declLine;
        int declCol;
        bool isConst;
    };

    SymbolTable() { pushScope(); }

    void pushScope() { scopes.emplace_back(); }
    void popScope() {
        if (scopes.size() <= 1) return;
        scopes.pop_back();
    }

    int currentDepth() const { return static_cast<int>(scopes.size()) - 1; }

    void declare(const std::string& name, ValueType type, int line, int col, bool isConst = false) {
        auto& currentScope = scopes.back();
        if (currentScope.count(name)) {
            throw std::runtime_error("重复声明标识符 '" + name + "'，位置: line " + std::to_string(line) + " col " +
                                     std::to_string(col));
        }
        currentScope[name] = {type, isConst};
        history.push_back(Entry{name, type, currentDepth(), line, col, isConst});
    }

    bool isConst(const std::string& name, int line, int col) const {
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            auto found = it->find(name);
            if (found != it->end()) return found->second.isConst;
        }
        throw std::runtime_error("未声明的标识符 '" + name + "'，位置: line " + std::to_string(line) + " col " +
                                 std::to_string(col));
    }

    ValueType lookup(const std::string& name, int line, int col) const {
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            auto found = it->find(name);
            if (found != it->end()) return found->second.type;
        }
        throw std::runtime_error("未声明的标识符 '" + name + "'，位置: line " + std::to_string(line) + " col " +
                                 std::to_string(col));
    }

    const std::vector<Entry>& getHistory() const { return history; }

    void print() const {
        auto typeStr = [](ValueType t) {
            switch (t) {
            case ValueType::Int: return std::string("int");
            case ValueType::Float: return std::string("float");
            case ValueType::Bool: return std::string("bool");
            case ValueType::Char: return std::string("char");
            case ValueType::String: return std::string("string");
            case ValueType::Void: return std::string("void");
            }
            return std::string("unknown");
        };

        std::cout << "=== SYMTAB ===\n";
        if (history.empty()) {
            std::cout << "(空符号表)\n";
            return;
        }

        int maxDepth = 0;
        for (const auto& e : history) maxDepth = std::max(maxDepth, e.scopeDepth);

        for (int d = 0; d <= maxDepth; ++d) {
            bool any = false;
            for (const auto& e : history) {
                if (e.scopeDepth != d) continue;
                if (!any) {
                    std::cout << "[scope " << d << "]\n";
                    any = true;
                }
                std::cout << "  " << e.name << "\t" << typeStr(e.type)
                          << (e.isConst ? " (const)" : "")
                          << "\t@(" << e.declLine << ":" << e.declCol << ")\n";
            }
        }
    }

  private:
    struct SymbolInfo {
        ValueType type;
        bool isConst;
    };
    std::vector<std::unordered_map<std::string, SymbolInfo>> scopes;
    std::vector<Entry> history;
};

// 代码生成器
class CodeGenerator {
  public:
    std::string newTemp(ValueType type) {
        std::string prefix;
        switch(type) {
            case ValueType::Int: prefix = "i"; break;
            case ValueType::Float: prefix = "f"; break;
            case ValueType::Bool: prefix = "b"; break;
            case ValueType::Char: prefix = "c"; break;
            case ValueType::String: prefix = "s"; break;
            default: prefix = "t"; break;
        }
        std::string name = prefix + std::to_string(tempIndex++);
        tempType[name] = type;
        return name;
    }

    std::string newLabel() { return "L" + std::to_string(labelIndex++); }

    void emit(const std::string& op, const std::string& a1 = "", const std::string& a2 = "",
               const std::string& res = "") {
        code.push_back({op, a1, a2, res});
    }

    const std::vector<Tac>& getCode() const { return code; }
    std::unordered_map<std::string, std::string> stringTable;
    int stringIndex = 0;

  private:
    int tempIndex = 0;
    int labelIndex = 0;
    std::unordered_map<std::string, ValueType> tempType;
    std::vector<Tac> code;
};

// 表达式结果
struct ExprResult {
    std::string place;
    ValueType type;
    std::shared_ptr<ASTNode> expr;
};

// 语法分析器
class Parser {
  public:
    Parser(std::vector<Token> tokens, CodeGenerator& gen, SymbolTable& symbols)
        : toks(std::move(tokens)), code(gen), symbols(symbols) {
        astRoot = std::make_shared<ASTNode>(ASTNodeType::Program, "program");
        breakStack.push_back(std::vector<std::string>());
        continueStack.push_back(std::vector<std::string>());
    }

    void parseProgram() {
        while (!check(TokenType::EndOfFile)) {
            auto stmt = parseStatement();
            if (stmt) astRoot->children.push_back(stmt);
        }
    }

    std::shared_ptr<ASTNode> getASTRoot() const { return astRoot; }

  private:
    std::vector<Token> toks;
    size_t current = 0;
    CodeGenerator& code;
    SymbolTable& symbols;
    std::shared_ptr<ASTNode> astRoot;
    std::vector<std::vector<std::string>> breakStack;
    std::vector<std::vector<std::string>> continueStack;

    bool isAtEnd() const { return peek().type == TokenType::EndOfFile; }
    const Token& peek() const { return toks[current]; }
    const Token& previous() const { return toks[current - 1]; }

    bool check(TokenType type, const std::string& lexeme = "") const {
        if (isAtEnd()) return type == TokenType::EndOfFile;
        if (peek().type != type) return false;
        if (!lexeme.empty() && peek().lexeme != lexeme) return false;
        return true;
    }

    const Token& advance() {
        if (!isAtEnd()) current++;
        return previous();
    }

    bool match(TokenType type, const std::string& lexeme = "") {
        if (check(type, lexeme)) {
            advance();
            return true;
        }
        return false;
    }

    const Token& consume(TokenType type, const std::string& lexeme, const std::string& msg) {
        if (check(type, lexeme)) return advance();
        throw std::runtime_error(errorAt(peek(), msg));
    }

    const Token& consume(TokenType type, const std::string& msg) {
        if (check(type)) return advance();
        throw std::runtime_error(errorAt(peek(), msg));
    }

    std::string errorAt(const Token& tok, const std::string& msg) {
        std::ostringstream oss;
        oss << "语法错误 (line " << tok.line << ", col " << tok.column << "): " << msg << "，发现 '" << tok.lexeme
            << "'";
        return oss.str();
    }

    ValueType parseType() {
        if (match(TokenType::Keyword, "int")) {
            return ValueType::Int;
        } else if (match(TokenType::Keyword, "float")) {
            return ValueType::Float;
        } else if (match(TokenType::Keyword, "bool")) {
            return ValueType::Bool;
        } else if (match(TokenType::Keyword, "char")) {
            return ValueType::Char;
        } else {
            throw std::runtime_error("未知类型");
        }
    }

    std::shared_ptr<ASTNode> parseStatement() {
        if (match(TokenType::Keyword, "const")) {
            ValueType t = parseType();
            return parseConstDeclaration(t);
        } else if (check(TokenType::Keyword, "int") || check(TokenType::Keyword, "float") ||
                   check(TokenType::Keyword, "bool") || check(TokenType::Keyword, "char")) {
            ValueType t = parseType();
            return parseDeclaration(t);
        } else if (match(TokenType::Keyword, "if")) {
            return parseIf();
        } else if (match(TokenType::Keyword, "while")) {
            return parseWhile();
        } else if (match(TokenType::Keyword, "do")) {
            return parseDoWhile();
        } else if (match(TokenType::Keyword, "for")) {
            return parseFor();
        } else if (match(TokenType::Keyword, "switch")) {
            return parseSwitch();
        } else if (match(TokenType::Keyword, "break")) {
            return parseBreak();
        } else if (match(TokenType::Keyword, "continue")) {
            return parseContinue();
        } else if (match(TokenType::Separator, "{")) {
            symbols.pushScope();
            auto block = parseBlock();
            symbols.popScope();
            return block;
        } else if (match(TokenType::Keyword, "return")) {
            return parseReturn();
        } else {
            return parseExpressionStatement();
        }
    }

    std::shared_ptr<ASTNode> parseDeclaration(ValueType type) {
        const Token& idTok = consume(TokenType::Identifier, "缺少标识符");
        symbols.declare(idTok.lexeme, type, idTok.line, idTok.column, false);

        auto declNode = std::make_shared<ASTNode>(ASTNodeType::Declaration, idTok.lexeme, type, idTok.line, idTok.column);
        auto varNode = std::make_shared<ASTNode>(ASTNodeType::Variable, idTok.lexeme, type, idTok.line, idTok.column);
        declNode->children.push_back(varNode);

        if (match(TokenType::Operator, "=")) {
            ExprResult init = assignment();
            ExprResult rhs = cast(init, type);
            if (symbols.isConst(idTok.lexeme, idTok.line, idTok.column)) {
                throw std::runtime_error("不能给常量 '" + idTok.lexeme + "' 赋值");
            }
            code.emit("=", rhs.place, "", idTok.lexeme);
            auto assignNode = std::make_shared<ASTNode>(ASTNodeType::Assignment, "=", type, idTok.line, idTok.column);
            assignNode->children.push_back(varNode);
            assignNode->children.push_back(init.expr);
            declNode->children.push_back(assignNode);
        }
        consume(TokenType::Separator, ";", "缺少分号");
        return declNode;
    }

    std::shared_ptr<ASTNode> parseConstDeclaration(ValueType type) {
        const Token& idTok = consume(TokenType::Identifier, "缺少标识符");
        symbols.declare(idTok.lexeme, type, idTok.line, idTok.column, true);
        consume(TokenType::Operator, "=", "常量必须初始化");
        ExprResult init = assignment();
        ExprResult rhs = cast(init, type);
        code.emit("=", rhs.place, "", idTok.lexeme);
        auto constDeclNode = std::make_shared<ASTNode>(ASTNodeType::ConstDeclaration, idTok.lexeme, type, idTok.line, idTok.column);
        auto varNode = std::make_shared<ASTNode>(ASTNodeType::Variable, idTok.lexeme, type, idTok.line, idTok.column);
        constDeclNode->children.push_back(varNode);
        constDeclNode->children.push_back(init.expr);
        consume(TokenType::Separator, ";", "缺少分号");
        return constDeclNode;
    }

    std::shared_ptr<ASTNode> parseIf() {
        const Token& ifTok = previous();
        consume(TokenType::Separator, "(", "缺少 '('");
        ExprResult cond = assignment();
        consume(TokenType::Separator, ")", "缺少 ')'");
        std::string elseLabel = code.newLabel();
        std::string endLabel = code.newLabel();
        code.emit("iffalse", ensureBool(cond).place, "", elseLabel);
        auto ifNode = std::make_shared<ASTNode>(ASTNodeType::If, "if", ValueType::Void, ifTok.line, ifTok.column);
        ifNode->children.push_back(cond.expr);
        auto thenStmt = parseStatement();
        ifNode->children.push_back(thenStmt);
        code.emit("goto", "", "", endLabel);
        code.emit("label", "", "", elseLabel);
        if (match(TokenType::Keyword, "else")) {
            auto elseStmt = parseStatement();
            ifNode->children.push_back(elseStmt);
        }
        code.emit("label", "", "", endLabel);
        return ifNode;
    }

    std::shared_ptr<ASTNode> parseWhile() {
        const Token& whileTok = previous();
        std::string startLabel = code.newLabel();
        std::string endLabel = code.newLabel();
        std::string continueLabel = code.newLabel();
        breakStack.back().push_back(endLabel);
        continueStack.back().push_back(continueLabel);
        code.emit("label", "", "", startLabel);
        consume(TokenType::Separator, "(", "缺少 '('");
        ExprResult cond = assignment();
        consume(TokenType::Separator, ")", "缺少 ')'");
        code.emit("iffalse", ensureBool(cond).place, "", endLabel);
        auto whileNode = std::make_shared<ASTNode>(ASTNodeType::While, "while", ValueType::Void, whileTok.line, whileTok.column);
        whileNode->children.push_back(cond.expr);
        auto bodyStmt = parseStatement();
        whileNode->children.push_back(bodyStmt);
        code.emit("label", "", "", continueLabel);
        code.emit("goto", "", "", startLabel);
        code.emit("label", "", "", endLabel);
        breakStack.back().pop_back();
        continueStack.back().pop_back();
        return whileNode;
    }

    std::shared_ptr<ASTNode> parseDoWhile() {
        const Token& doTok = previous();
        std::string startLabel = code.newLabel();
        std::string endLabel = code.newLabel();
        std::string continueLabel = code.newLabel();
        breakStack.back().push_back(endLabel);
        continueStack.back().push_back(continueLabel);
        code.emit("label", "", "", startLabel);
        auto doWhileNode = std::make_shared<ASTNode>(ASTNodeType::DoWhile, "do-while", ValueType::Void, doTok.line, doTok.column);
        auto bodyStmt = parseStatement();
        doWhileNode->children.push_back(bodyStmt);
        code.emit("label", "", "", continueLabel);
        consume(TokenType::Keyword, "while", "缺少 'while'");
        consume(TokenType::Separator, "(", "缺少 '('");
        ExprResult cond = assignment();
        consume(TokenType::Separator, ")", "缺少 ')'");
        doWhileNode->children.push_back(cond.expr);
        consume(TokenType::Separator, ";", "缺少分号");
        code.emit("iftrue", ensureBool(cond).place, "", startLabel);
        code.emit("label", "", "", endLabel);
        breakStack.back().pop_back();
        continueStack.back().pop_back();
        return doWhileNode;
    }

    std::shared_ptr<ASTNode> parseSwitch() {
        const Token& switchTok = previous();
        consume(TokenType::Separator, "(", "缺少 '('");
        ExprResult expr = assignment();
        consume(TokenType::Separator, ")", "缺少 ')'");
        consume(TokenType::Separator, "{", "缺少 '{'");
        
        auto switchNode = std::make_shared<ASTNode>(
            ASTNodeType::Switch, "switch", ValueType::Void,
            switchTok.line, switchTok.column
        );
        switchNode->children.push_back(expr.expr);
        
        std::string endLabel = code.newLabel();
        
        // 为switch语句创建一个新的break作用域
        breakStack.push_back(std::vector<std::string>());
        breakStack.back().push_back(endLabel);
        
        std::string switchTemp = code.newTemp(expr.type);
        code.emit("=", expr.place, "", switchTemp);
        
        std::string defaultLabel = code.newLabel();
        bool hasDefault = false;
        
        // 存储case标签信息
        struct CaseInfo {
            std::string caseValue;
            std::string caseLabel;
            ExprResult exprResult;
        };
        std::vector<CaseInfo> caseInfos;
        
        // 第一次扫描：收集所有case和default信息
        size_t savePos = current;
        while (!check(TokenType::Separator, "}") && !isAtEnd()) {
            if (match(TokenType::Keyword, "case")) {
                ExprResult caseExpr = assignment();
                if (match(TokenType::Operator, ":")) {
                    std::string caseLabel = code.newLabel();
                    caseInfos.push_back({caseExpr.place, caseLabel, caseExpr});
                } else {
                    throw std::runtime_error(errorAt(peek(), "缺少 ':'"));
                }
                // 跳过case内的语句
                while (!check(TokenType::Keyword, "case") && 
                       !check(TokenType::Keyword, "default") && 
                       !check(TokenType::Separator, "}")) {
                    advance();
                }
            } else if (match(TokenType::Keyword, "default")) {
                if (match(TokenType::Operator, ":")) {
                    hasDefault = true;
                } else {
                    throw std::runtime_error(errorAt(peek(), "缺少 ':'"));
                }
                // 跳过default内的语句
                while (!check(TokenType::Separator, "}")) {
                    advance();
                }
            } else {
                advance();
            }
        }
        
        // 恢复位置
        current = savePos;
        
        // 生成比较和跳转代码
        for (const auto& caseInfo : caseInfos) {
            std::string cmpTemp = code.newTemp(ValueType::Bool);
            code.emit("==", switchTemp, caseInfo.caseValue, cmpTemp);
            code.emit("iftrue", cmpTemp, "", caseInfo.caseLabel);
        }
        
        // 跳转到default标签（如果没有匹配的case）
        code.emit("goto", "", "", defaultLabel);
        
        // 第二次扫描：生成case和default的代码
        while (!check(TokenType::Separator, "}") && !isAtEnd()) {
            if (match(TokenType::Keyword, "case")) {
                ExprResult caseExpr = assignment();
                consume(TokenType::Operator, ":", "缺少 ':'");
                
                // 找到对应的标签
                std::string currentCaseLabel = "";
                for (const auto& caseInfo : caseInfos) {
                    if (caseInfo.caseValue == caseExpr.place) {
                        currentCaseLabel = caseInfo.caseLabel;
                        break;
                    }
                }
                
                auto caseNode = std::make_shared<ASTNode>(
                    ASTNodeType::Case, "case", ValueType::Void,
                    previous().line, previous().column
                );
                caseNode->children.push_back(caseExpr.expr);
                
                code.emit("label", "", "", currentCaseLabel);
                
                std::vector<std::shared_ptr<ASTNode>> caseStmts;
                while (!check(TokenType::Keyword, "case") &&
                       !check(TokenType::Keyword, "default") &&
                       !check(TokenType::Separator, "}")) {
                    auto stmt = parseStatement();
                    if (stmt) {
                        caseStmts.push_back(stmt);
                        caseNode->children.push_back(stmt);
                    }
                }
                
                switchNode->children.push_back(caseNode);
                
            } else if (match(TokenType::Keyword, "default")) {
                consume(TokenType::Operator, ":", "缺少 ':'");
                
                auto defaultNode = std::make_shared<ASTNode>(
                    ASTNodeType::Default, "default", ValueType::Void,
                    previous().line, previous().column
                );
                
                code.emit("label", "", "", defaultLabel);
                
                std::vector<std::shared_ptr<ASTNode>> defaultStmts;
                while (!check(TokenType::Separator, "}")) {
                    auto stmt = parseStatement();
                    if (stmt) {
                        defaultStmts.push_back(stmt);
                        defaultNode->children.push_back(stmt);
                    }
                }
                
                switchNode->children.push_back(defaultNode);
                hasDefault = true;
                
            } else {
                // switch块内的其他语句（如break）
                auto stmt = parseStatement();
                if (stmt) {
                    switchNode->children.push_back(stmt);
                }
            }
        }
        
        consume(TokenType::Separator, "}", "缺少 '}'");
        
        // 如果没有default，仍然需要定义default标签
        if (!hasDefault) {
            code.emit("label", "", "", defaultLabel);
        }
        
        code.emit("label", "", "", endLabel);
        breakStack.pop_back();  // 移除switch的break作用域
        
        return switchNode;
    }

    std::shared_ptr<ASTNode> parseBreak() {
        const Token& breakTok = previous();
        consume(TokenType::Separator, ";", "缺少 ';'");
        if (breakStack.empty() || breakStack.back().empty()) {
            throw std::runtime_error(errorAt(breakTok, "break语句必须在循环或switch语句内"));
        }
        std::string targetLabel = breakStack.back().back();
        code.emit("goto", "", "", targetLabel);
        return std::make_shared<ASTNode>(
            ASTNodeType::Break, "break", ValueType::Void,
            breakTok.line, breakTok.column
        );
    }

    std::shared_ptr<ASTNode> parseContinue() {
        const Token& continueTok = previous();
        consume(TokenType::Separator, ";", "缺少 ';' ");
        if (continueStack.empty() || continueStack.back().empty()) {
            throw std::runtime_error(errorAt(continueTok, "continue语句必须在循环语句内"));
        }
        std::string targetLabel = continueStack.back().back();
        code.emit("goto", "", "", targetLabel);
        return std::make_shared<ASTNode>(
            ASTNodeType::Continue, "continue", ValueType::Void,
            continueTok.line, continueTok.column
        );
    }

    std::shared_ptr<ASTNode> parseFor() {
        const Token& forTok = previous();
        consume(TokenType::Separator, "(", "缺少 '('");
        auto forNode = std::make_shared<ASTNode>(ASTNodeType::For, "for", ValueType::Void, forTok.line, forTok.column);
        if (!match(TokenType::Separator, ";")) {
            if (check(TokenType::Keyword, "const") || check(TokenType::Keyword, "int") ||
                check(TokenType::Keyword, "float") || check(TokenType::Keyword, "bool") ||
                check(TokenType::Keyword, "char")) {
                bool isConst = match(TokenType::Keyword, "const");
                ValueType t = parseType();
                if (isConst) {
                    auto initDecl = parseConstDeclaration(t);
                    forNode->children.push_back(initDecl);
                } else {
                    auto initDecl = parseDeclaration(t);
                    forNode->children.push_back(initDecl);
                }
            } else {
                auto initExpr = parseExpressionStatement();
                forNode->children.push_back(initExpr);
            }
        }

        std::string condLabel = code.newLabel();
        std::string endLabel = code.newLabel();
        std::string bodyLabel = code.newLabel();
        std::string continueLabel = code.newLabel();
        breakStack.back().push_back(endLabel);
        continueStack.back().push_back(continueLabel);
        code.emit("label", "", "", condLabel);

        ExprResult cond = { "", ValueType::Bool, nullptr };
        if (!check(TokenType::Separator, ";")) {
            cond = assignment();
            forNode->children.push_back(cond.expr);
        } else {
            cond.place = "1";
            cond.type = ValueType::Bool;
        }
        consume(TokenType::Separator, ";", "缺少';'");
        code.emit("iffalse", ensureBool(cond).place, "", endLabel);

        std::vector<Token> incTokens;
        size_t savePos = current;
        while (!check(TokenType::Separator, ")")) {
            incTokens.push_back(peek());
            advance();
        }
        consume(TokenType::Separator, ")", "缺少')'");
        code.emit("goto", "", "", bodyLabel);
        code.emit("label", "", "", bodyLabel);

        auto bodyStmt = parseStatement();
        forNode->children.push_back(bodyStmt);

        code.emit("label", "", "", continueLabel);
        if (!incTokens.empty()) {
            incTokens.push_back(Token{TokenType::Separator, ";", peek().line, peek().column});
            incTokens.push_back(Token{TokenType::EndOfFile, "<eof>", peek().line, peek().column});
            Parser incParser(incTokens, code, symbols);
            auto incExpr = incParser.parseExpressionStatement();
            forNode->children.push_back(incExpr);
        }

        code.emit("goto", "", "", condLabel);
        code.emit("label", "", "", endLabel);
        breakStack.back().pop_back();
        continueStack.back().pop_back();
        return forNode;
    }

    std::shared_ptr<ASTNode> parseBlock() {
        auto blockNode = std::make_shared<ASTNode>(ASTNodeType::Block, "block", ValueType::Void, peek().line, peek().column);
        while (!check(TokenType::Separator, "}") && !isAtEnd()) {
            auto stmt = parseStatement();
            if (stmt) blockNode->children.push_back(stmt);
        }
        consume(TokenType::Separator, "}", "缺少 '}'");
        return blockNode;
    }

    std::shared_ptr<ASTNode> parseReturn() {
        const Token& retTok = previous();
        auto returnNode = std::make_shared<ASTNode>(ASTNodeType::Return, "return", ValueType::Void, retTok.line, retTok.column);
        if (match(TokenType::Separator, ";")) {
            code.emit("return", "", "", "");
            return returnNode;
        }
        ExprResult value = assignment();
        consume(TokenType::Separator, ";", "缺少分号");
        code.emit("return", value.place, "", "");
        returnNode->children.push_back(value.expr);
        return returnNode;
    }

    std::shared_ptr<ASTNode> parseExpressionStatement() {
        ExprResult value = assignment();
        consume(TokenType::Separator, ";", "缺少分号");
        return value.expr;
    }

    // 表达式优先级层次（从低到高）：
    // 1. assignment (包括复合赋值) - 最低优先级
    // 2. conditional (三元条件运算符)
    // 3. logical_or
    // 4. logical_and
    // 5. equality
    // 6. relational
    // 7. additive
    // 8. multiplicative
    // 9. unary (包括前缀自增自减)
    // 10. postfix (包括后缀自增自减)
    // 11. primary - 最高优先级

    ExprResult assignment() {
        // 先解析条件表达式
        ExprResult result = conditional();
        
        // 检查是否为赋值运算符
        if (match(TokenType::Operator, "=") || 
            match(TokenType::Operator, "+=") || match(TokenType::Operator, "-=") ||
            match(TokenType::Operator, "*=") || match(TokenType::Operator, "/=") ||
            match(TokenType::Operator, "%=")) {
            
            std::string op = previous().lexeme;
            const Token& opTok = previous();
            
            // 检查左值是否为变量
            if (result.expr->type != ASTNodeType::Variable) {
                throw std::runtime_error(errorAt(opTok, "赋值运算符左边必须是变量"));
            }
            
            std::string varName = result.expr->value;
            ValueType varType = result.type;
            
            // 检查是否为常量
            if (symbols.isConst(varName, opTok.line, opTok.column)) {
                throw std::runtime_error("不能给常量 '" + varName + "' 赋值");
            }
            
            // 解析右值
            ExprResult right = assignment();  // 右结合
            
            // 处理复合赋值
            if (op != "=") {
                // 提取基础运算符
                std::string baseOp = op.substr(0, 1);
                
                // 生成左值的临时副本
                std::string tempLeft = code.newTemp(varType);
                code.emit("=", result.place, "", tempLeft);
                
                // 进行运算
                ExprResult rhsCast = cast(right, varType);
                std::string tempResult = code.newTemp(varType);
                code.emit(baseOp, tempLeft, rhsCast.place, tempResult);
                
                // 赋值
                code.emit("=", tempResult, "", varName);
                
                // 构建AST节点
                auto compAssignNode = std::make_shared<ASTNode>(
                    ASTNodeType::Assignment, op, varType, 
                    opTok.line, opTok.column
                );
                compAssignNode->children.push_back(result.expr);
                
                // 创建复合运算的AST节点
                auto binopNode = std::make_shared<ASTNode>(
                    ASTNodeType::BinaryOp, baseOp, varType,
                    opTok.line, opTok.column
                );
                auto leftCopyNode = std::make_shared<ASTNode>(
                    ASTNodeType::Variable, varName, varType,
                    result.expr->line, result.expr->column
                );
                binopNode->children.push_back(leftCopyNode);
                binopNode->children.push_back(rhsCast.expr);
                
                compAssignNode->children.push_back(binopNode);
                return ExprResult{tempResult, varType, compAssignNode};
            } else {
                // 简单赋值
                ExprResult casted = cast(right, varType);
                code.emit("=", casted.place, "", varName);
                
                auto assignNode = std::make_shared<ASTNode>(
                    ASTNodeType::Assignment, "=", varType,
                    opTok.line, opTok.column
                );
                assignNode->children.push_back(result.expr);
                assignNode->children.push_back(casted.expr);
                
                return ExprResult{casted.place, varType, assignNode};
            }
        }
        
        return result;
    }

    ExprResult conditional() {
        ExprResult cond = logical_or();
        
        if (match(TokenType::Operator, "?")) {
            const Token& questionTok = previous();
            
            // 解析true表达式
            ExprResult trueExpr = assignment();
            
            consume(TokenType::Operator, ":", "缺少 ':' 在三元条件运算符中");
            
            // 解析false表达式
            ExprResult falseExpr = conditional();  // 右结合
            
            // 生成代码
            ValueType resultType = promote(trueExpr.type, falseExpr.type);
            std::string temp = code.newTemp(resultType);
            std::string falseLabel = code.newLabel();
            std::string endLabel = code.newLabel();
            
            // 条件判断
            code.emit("iffalse", ensureBool(cond).place, "", falseLabel);
            
            // true分支
            ExprResult castedTrue = cast(trueExpr, resultType);
            code.emit("=", castedTrue.place, "", temp);
            code.emit("goto", "", "", endLabel);
            
            // false分支
            code.emit("label", "", "", falseLabel);
            ExprResult castedFalse = cast(falseExpr, resultType);
            code.emit("=", castedFalse.place, "", temp);
            code.emit("label", "", "", endLabel);
            
            // 构建AST节点
            auto ternaryNode = std::make_shared<ASTNode>(
                ASTNodeType::TernaryOp, "?:", resultType, 
                questionTok.line, questionTok.column
            );
            ternaryNode->children.push_back(cond.expr);
            ternaryNode->children.push_back(castedTrue.expr);
            ternaryNode->children.push_back(castedFalse.expr);
            
            return ExprResult{temp, resultType, ternaryNode};
        }
        
        return cond;
    }

    ExprResult logical_or() {
        ExprResult left = logical_and();
        while (match(TokenType::Operator, "||")) {
            ExprResult right = logical_and();
            ExprResult result = {code.newTemp(ValueType::Bool), ValueType::Bool, nullptr};
            std::string evalRight = code.newLabel();
            std::string endLabel = code.newLabel();
            std::string setTrue = code.newLabel();
            code.emit("=", "0", "", result.place);
            code.emit("iffalse", ensureBool(left).place, "", evalRight);
            code.emit("goto", "", "", setTrue);
            code.emit("label", "", "", evalRight);
            code.emit("iffalse", ensureBool(right).place, "", endLabel);
            code.emit("label", "", "", setTrue);
            code.emit("=", "1", "", result.place);
            code.emit("label", "", "", endLabel);
            auto binopNode = std::make_shared<ASTNode>(ASTNodeType::BinaryOp, "||", ValueType::Bool, previous().line, previous().column);
            binopNode->children.push_back(left.expr);
            binopNode->children.push_back(right.expr);
            result.expr = binopNode;
            left = result;
        }
        return left;
    }

    ExprResult logical_and() {
        ExprResult left = equality();
        while (match(TokenType::Operator, "&&")) {
            ExprResult right = equality();
            ExprResult result = {code.newTemp(ValueType::Bool), ValueType::Bool, nullptr};
            std::string endLabel = code.newLabel();
            code.emit("=", "0", "", result.place);
            code.emit("iffalse", ensureBool(left).place, "", endLabel);
            code.emit("iffalse", ensureBool(right).place, "", endLabel);
            code.emit("=", "1", "", result.place);
            code.emit("label", "", "", endLabel);
            auto binopNode = std::make_shared<ASTNode>(ASTNodeType::BinaryOp, "&&", ValueType::Bool, previous().line, previous().column);
            binopNode->children.push_back(left.expr);
            binopNode->children.push_back(right.expr);
            result.expr = binopNode;
            left = result;
        }
        return left;
    }

    ExprResult equality() {
        ExprResult left = relational();
        while (match(TokenType::Operator, "==") || match(TokenType::Operator, "!=")) {
            std::string op = previous().lexeme;
            ExprResult right = relational();
            ExprResult result = {code.newTemp(ValueType::Bool), ValueType::Bool, nullptr};
            code.emit(op, left.place, right.place, result.place);
            auto binopNode = std::make_shared<ASTNode>(ASTNodeType::BinaryOp, op, ValueType::Bool, previous().line, previous().column);
            binopNode->children.push_back(left.expr);
            binopNode->children.push_back(right.expr);
            result.expr = binopNode;
            left = result;
        }
        return left;
    }

    ExprResult relational() {
        ExprResult left = additive();
        while (match(TokenType::Operator, "<") || match(TokenType::Operator, ">") ||
               match(TokenType::Operator, "<=") || match(TokenType::Operator, ">=")) {
            std::string op = previous().lexeme;
            ExprResult right = additive();
            ExprResult result = {code.newTemp(ValueType::Bool), ValueType::Bool, nullptr};
            code.emit(op, left.place, right.place, result.place);
            auto binopNode = std::make_shared<ASTNode>(ASTNodeType::BinaryOp, op, ValueType::Bool, previous().line, previous().column);
            binopNode->children.push_back(left.expr);
            binopNode->children.push_back(right.expr);
            result.expr = binopNode;
            left = result;
        }
        return left;
    }

    ExprResult additive() {
        ExprResult left = multiplicative();
        while (match(TokenType::Operator, "+") || match(TokenType::Operator, "-")) {
            std::string op = previous().lexeme;
            ExprResult right = multiplicative();
            ValueType resType = promote(left.type, right.type);
            ExprResult lCast = cast(left, resType);
            ExprResult rCast = cast(right, resType);
            ExprResult result = {code.newTemp(resType), resType};
            code.emit(op, lCast.place, rCast.place, result.place);
            auto binopNode = std::make_shared<ASTNode>(ASTNodeType::BinaryOp, op, resType, previous().line, previous().column);
            binopNode->children.push_back(lCast.expr);
            binopNode->children.push_back(rCast.expr);
            result.expr = binopNode;
            left = result;
        }
        return left;
    }

    ExprResult multiplicative() {
        ExprResult left = unary();
        while (match(TokenType::Operator, "*") || match(TokenType::Operator, "/") || match(TokenType::Operator, "%")) {
            std::string op = previous().lexeme;
            ExprResult right = unary();
            ValueType resType = promote(left.type, right.type);
            ExprResult lCast = cast(left, resType);
            ExprResult rCast = cast(right, resType);
            ExprResult result = {code.newTemp(resType), resType};
            code.emit(op, lCast.place, rCast.place, result.place);
            auto binopNode = std::make_shared<ASTNode>(ASTNodeType::BinaryOp, op, resType, previous().line, previous().column);
            binopNode->children.push_back(lCast.expr);
            binopNode->children.push_back(rCast.expr);
            result.expr = binopNode;
            left = result;
        }
        return left;
    }

    ExprResult unary() {
        // 处理前缀自增自减
        if (match(TokenType::Operator, "++") || match(TokenType::Operator, "--")) {
            std::string op = previous().lexeme;
            const Token& opTok = previous();
            
            // 解析右操作数
            ExprResult rhs = unary();
            
            // 检查是否为变量
            if (rhs.expr->type != ASTNodeType::Variable) {
                throw std::runtime_error(errorAt(opTok, "前缀自增/自减运算符只能用于变量"));
            }
            
            std::string varName = rhs.expr->value;
            ValueType varType = rhs.type;
            
            // 检查是否为常量
            if (symbols.isConst(varName, opTok.line, opTok.column)) {
                throw std::runtime_error("不能修改常量 '" + varName + "'");
            }
            
            // 前缀：先自增/自减，再使用新值
            std::string one = (varType == ValueType::Float) ? "1.0" : "1";
            std::string newTemp = code.newTemp(varType);
            if (op == "++") {
                code.emit("+", rhs.place, one, newTemp);
            } else {
                code.emit("-", rhs.place, one, newTemp);
            }
            
            // 赋值回变量
            code.emit("=", newTemp, "", varName);
            
            // 构建AST节点
            auto prefixNode = std::make_shared<ASTNode>(
                ASTNodeType::UnaryOp, op + "pre", varType,
                opTok.line, opTok.column
            );
            prefixNode->children.push_back(rhs.expr);
            
            return ExprResult{newTemp, varType, prefixNode};
        }
        
        // 处理其他一元运算符
        if (match(TokenType::Operator, "-")) {
            ExprResult rhs = unary();
            ExprResult zero = {rhs.type == ValueType::Float ? "0.0" :
                              (rhs.type == ValueType::Char ? "0" : "0"), 
                              rhs.type, nullptr};
            ExprResult result = {code.newTemp(rhs.type), rhs.type};
            code.emit("-", zero.place, rhs.place, result.place);
            auto unaryNode = std::make_shared<ASTNode>(
                ASTNodeType::UnaryOp, "-", rhs.type, 
                previous().line, previous().column
            );
            unaryNode->children.push_back(rhs.expr);
            result.expr = unaryNode;
            return result;
        }
        if (match(TokenType::Operator, "!")) {
            ExprResult rhs = unary();
            ExprResult result = {code.newTemp(ValueType::Bool), ValueType::Bool, nullptr};
            code.emit("!", rhs.place, "", result.place);
            auto unaryNode = std::make_shared<ASTNode>(
                ASTNodeType::UnaryOp, "!", ValueType::Bool, 
                previous().line, previous().column
            );
            unaryNode->children.push_back(rhs.expr);
            result.expr = unaryNode;
            return result;
        }
        
        // 不是一元运算符，解析后缀表达式
        return postfix();
    }

    ExprResult postfix() {
        // 先解析主表达式
        ExprResult expr = primary();
        
        // 处理后缀运算符
        while (true) {
            if (match(TokenType::Operator, "++") || match(TokenType::Operator, "--")) {
                std::string op = previous().lexeme;
                const Token& opTok = previous();
                
                // 检查表达式是否为变量
                if (expr.expr->type != ASTNodeType::Variable) {
                    throw std::runtime_error(errorAt(opTok, "自增/自减运算符只能用于变量"));
                }
                
                std::string varName = expr.expr->value;
                ValueType varType = expr.type;
                
                // 检查是否为常量
                if (symbols.isConst(varName, opTok.line, opTok.column)) {
                    throw std::runtime_error("不能修改常量 '" + varName + "'");
                }
                
                // 后缀：先使用原值，再自增/自减
                std::string resultTemp = code.newTemp(varType);
                code.emit("=", expr.place, "", resultTemp);  // 保存原值
                
                // 计算新值
                std::string one = (varType == ValueType::Float) ? "1.0" : "1";
                std::string newTemp = code.newTemp(varType);
                if (op == "++") {
                    code.emit("+", expr.place, one, newTemp);
                } else {
                    code.emit("-", expr.place, one, newTemp);
                }
                
                // 赋值回变量
                code.emit("=", newTemp, "", varName);
                
                // 构建AST节点
                auto postfixNode = std::make_shared<ASTNode>(
                    ASTNodeType::UnaryOp, op + "post", varType,
                    opTok.line, opTok.column
                );
                postfixNode->children.push_back(expr.expr);
                
                return ExprResult{resultTemp, varType, postfixNode};
            } else {
                break;
            }
        }
        
        return expr;
    }

    ExprResult primary() {
        if (match(TokenType::Number)) {
            const Token& numTok = previous();
            ValueType vt = (numTok.lexeme.find('.') != std::string::npos) ? ValueType::Float : ValueType::Int;
            auto node = std::make_shared<ASTNode>(ASTNodeType::Literal, numTok.lexeme, vt, numTok.line, numTok.column);
            return ExprResult{numTok.lexeme, vt, node};
        }
        else if (match(TokenType::CharLiteral)) {
            const Token& charTok = previous();
            auto node = std::make_shared<ASTNode>(ASTNodeType::Literal, charTok.lexeme, ValueType::Char, charTok.line, charTok.column);
            return ExprResult{charTok.lexeme, ValueType::Char, node};
        }
        else if (match(TokenType::BoolLiteral)) {
            const Token& boolTok = previous();
            std::string val = (boolTok.lexeme == "true") ? "1" : "0";
            auto node = std::make_shared<ASTNode>(ASTNodeType::Literal, val, ValueType::Bool, boolTok.line, boolTok.column);
            return ExprResult{val, ValueType::Bool, node};
        }
        else if (match(TokenType::StringLiteral)) {
            const Token& strTok = previous();
            std::string strId = "s" + std::to_string(code.stringIndex++);
            code.stringTable[strId] = strTok.lexeme;
            auto node = std::make_shared<ASTNode>(ASTNodeType::Literal, strId, ValueType::String, strTok.line, strTok.column);
            return ExprResult{strId, ValueType::String, node};
        }
        else if (match(TokenType::Identifier)) {
            const Token& idTok = previous();
            if (match(TokenType::Separator, "(")) {
                std::vector<ExprResult> args;
                if (!check(TokenType::Separator, ")")) {
                    args.push_back(assignment());
                    while (match(TokenType::Separator, ",")) {
                        args.push_back(assignment());
                    }
                }
                consume(TokenType::Separator, ")", "缺少 ')'");
                for (auto& a : args) {
                    code.emit("param", a.place, "", "");
                }
                if (idTok.lexeme == "print") {
                    if (args.empty()) {
                        throw std::runtime_error(errorAt(idTok, "print 至少需要 1 个参数"));
                    }
                    for (auto& a : args) code.emit("print", a.place, "", "");
                    auto printNode = std::make_shared<ASTNode>(ASTNodeType::Print, "print", ValueType::Void, idTok.line, idTok.column);
                    for (auto& a : args) {
                        printNode->children.push_back(a.expr);
                    }
                    return ExprResult{"0", ValueType::Int, printNode};
                }
                std::string ret = code.newTemp(ValueType::Int);
                code.emit("call", idTok.lexeme, std::to_string(args.size()), ret);
                auto callNode = std::make_shared<ASTNode>(ASTNodeType::FunctionCall, idTok.lexeme, ValueType::Int, idTok.line, idTok.column);
                for (auto& a : args) {
                    callNode->children.push_back(a.expr);
                }
                return ExprResult{ret, ValueType::Int, callNode};
            }

            ValueType t = symbols.lookup(idTok.lexeme, idTok.line, idTok.column);
            auto varNode = std::make_shared<ASTNode>(ASTNodeType::Variable, idTok.lexeme, t, idTok.line, idTok.column);
            return ExprResult{idTok.lexeme, t, varNode};
        }
        else if (match(TokenType::Separator, "(")) {
            ExprResult inner = assignment();
            consume(TokenType::Separator, ")", "缺少 ')'");
            return inner;
        }
        throw std::runtime_error(errorAt(peek(), "无法解析的表达式"));
    }

    ExprResult ensureBool(const ExprResult& expr) {
        if (expr.type == ValueType::Bool) return expr;
        if (expr.type == ValueType::Void) {
            throw std::runtime_error(errorAt(previous(), "void 类型不能作为条件表达式"));
        }
        ExprResult result = {code.newTemp(ValueType::Bool), ValueType::Bool, nullptr};
        std::string zero;
        switch (expr.type) {
            case ValueType::Int: zero = "0"; break;
            case ValueType::Float: zero = "0.0"; break;
            case ValueType::Char: zero = "0"; break;
            case ValueType::String: zero = ""; break;
            case ValueType::Bool: zero = "0"; break;
            default: zero = "0"; break;
        }
        code.emit("!=", expr.place, zero, result.place);
        auto boolNode = std::make_shared<ASTNode>(ASTNodeType::BinaryOp, "!=", ValueType::Bool,
                                                 expr.expr ? expr.expr->line : 0,
                                                 expr.expr ? expr.expr->column : 0);
        if (expr.expr) boolNode->children.push_back(expr.expr);
        auto zeroNode = std::make_shared<ASTNode>(ASTNodeType::Literal, zero, expr.type,
                                                 expr.expr ? expr.expr->line : 0,
                                                 expr.expr ? expr.expr->column : 0);
        boolNode->children.push_back(zeroNode);
        result.expr = boolNode;
        return result;
    }

    ValueType promote(ValueType a, ValueType b) {
        if (a == ValueType::String || b == ValueType::String) {
            throw std::runtime_error("字符串类型不支持算术运算");
        }
        if (a == ValueType::Float || b == ValueType::Float)
            return ValueType::Float;
        if (a == ValueType::Int || b == ValueType::Int)
            return ValueType::Int;
        if (a == ValueType::Char || b == ValueType::Char)
            return ValueType::Char;
        return ValueType::Bool;
    }

    ExprResult cast(const ExprResult& expr, ValueType target) {
        if (expr.type == target) return expr;
        if (expr.type == ValueType::String || target == ValueType::String) {
            throw std::runtime_error("不支持字符串类型的转换");
        }
        std::string temp = code.newTemp(target);
        code.emit("cast", expr.place, typeName(target), temp);
        auto castNode = std::make_shared<ASTNode>(ASTNodeType::BinaryOp, "cast", target,
                                                 expr.expr ? expr.expr->line : 0,
                                                 expr.expr ? expr.expr->column : 0);
        if (expr.expr) castNode->children.push_back(expr.expr);
        auto typeNode = std::make_shared<ASTNode>(ASTNodeType::Literal, typeName(target), target,
                                                 expr.expr ? expr.expr->line : 0,
                                                 expr.expr ? expr.expr->column : 0);
        castNode->children.push_back(typeNode);
        return {temp, target, castNode};
    }

    std::string typeName(ValueType t) const {
        switch (t) {
        case ValueType::Int:
            return "int";
        case ValueType::Float:
            return "float";
        case ValueType::Bool:
            return "bool";
        case ValueType::Char:
            return "char";
        case ValueType::String:
            return "string";
        case ValueType::Void:
        default:
            return "void";
        }
    }
};

// 读取文件
std::string readFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("无法打开文件: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string tokenTypeName(TokenType t);
void printTokens(const std::vector<Token>& tokens) {
    std::cout << "=== TOKENS ===\n";
    for (const auto& t : tokens) {
        if (t.type == TokenType::EndOfFile) break;
        std::cout << std::setw(4) << t.line << ":" << std::setw(3) << t.column << "  " << std::left << std::setw(12)
                  << tokenTypeName(t.type) << "  " << t.lexeme << "\n";
    }
}

std::string tokenTypeName(TokenType t) {
    switch (t) {
    case TokenType::Identifier:
        return "Identifier";
    case TokenType::Number:
        return "Number";
    case TokenType::Keyword:
        return "Keyword";
    case TokenType::Operator:
        return "Operator";
    case TokenType::Separator:
        return "Separator";
    case TokenType::EndOfFile:
        return "EOF";
    case TokenType::CharLiteral:
        return "CharLiteral";
    case TokenType::BoolLiteral:
        return "BoolLiteral";
    case TokenType::StringLiteral:
        return "StringLiteral";
    }
    return "Unknown";
}

std::string nodeTypeName(ASTNodeType t) {
    switch (t) {
    case ASTNodeType::Program: return "Program";
    case ASTNodeType::Declaration: return "Declaration";
    case ASTNodeType::Assignment: return "Assignment";
    case ASTNodeType::BinaryOp: return "BinaryOp";
    case ASTNodeType::UnaryOp: return "UnaryOp";
    case ASTNodeType::If: return "If";
    case ASTNodeType::While: return "While";
    case ASTNodeType::For: return "For";
    case ASTNodeType::Block: return "Block";
    case ASTNodeType::Return: return "Return";
    case ASTNodeType::FunctionCall: return "FunctionCall";
    case ASTNodeType::Variable: return "Variable";
    case ASTNodeType::Literal: return "Literal";
    case ASTNodeType::Print: return "Print";
    case ASTNodeType::Switch: return "Switch";
    case ASTNodeType::Case: return "Case";
    case ASTNodeType::Default: return "Default";
    case ASTNodeType::DoWhile: return "DoWhile";
    case ASTNodeType::Break: return "Break";
    case ASTNodeType::Continue: return "Continue";
    case ASTNodeType::ConstDeclaration: return "ConstDeclaration";
    case ASTNodeType::TernaryOp: return "TernaryOp";
    }
    return "Unknown";
}

std::string valueTypeName(ValueType t) {
    switch (t) {
    case ValueType::Int: return "int";
    case ValueType::Float: return "float";
    case ValueType::Bool: return "bool";
    case ValueType::Char: return "char";
    case ValueType::String: return "string";
    case ValueType::Void: return "void";
    }
    return "unknown";
}

void printAST(const std::shared_ptr<ASTNode>& node, int indent = 0) {
    if (!node) return;
    std::string indentStr(indent, ' ');
    std::cout << indentStr << nodeTypeName(node->type) << " '" << node->value << "' "
              << valueTypeName(node->dataType) << " (" << node->line << ":" << node->column << ")";
    if (!node->children.empty()) {
        std::cout << " {\n";
        for (const auto& child : node->children) {
            printAST(child, indent + 2);
        }
        std::cout << indentStr << "}";
    }
    std::cout << "\n";
}

// 生成 AST 的 Graphviz DOT
static void astDotDfs(const std::shared_ptr<ASTNode>& node,
                      std::ostream& os,
                      int& nextId,
                      std::unordered_map<const ASTNode*, int>& ids) {
    if (!node) return;
    int myId = nextId++;
    ids[node.get()] = myId;

    auto esc = [](const std::string& s) {
        std::string r;
        for (char c : s) {
            if (c == '"' || c == '\\') r.push_back('\\');
            r.push_back(c);
        }
        return r;
    };

    os << "  n" << myId << " [label=\""
       << esc(nodeTypeName(node->type)) << "\\n"
       << esc(node->value) << "\\n"
       << esc(valueTypeName(node->dataType)) << "\\n"
       << node->line << ":" << node->column
       << "\"];\n";

    for (const auto& ch : node->children) {
        if (!ch) continue;
        astDotDfs(ch, os, nextId, ids);
        int cid = ids[ch.get()];
        os << "  n" << myId << " -> n" << cid << ";\n";
    }
}

static std::string astToDot(const std::shared_ptr<ASTNode>& root) {
    std::ostringstream os;
    os << "digraph AST {\n";
    os << "  node [shape=box, fontname=\"Consolas\"];\n";
    int nextId = 0;
    std::unordered_map<const ASTNode*, int> ids;
    astDotDfs(root, os, nextId, ids);
    os << "}\n";
    return os.str();
}

// TAC虚拟机
class TacVM {
  public:
    explicit TacVM(const std::vector<Tac>& code, const std::unordered_map<std::string, std::string>& strTab = {})
        : code(code), stringTable(strTab) {
        for (size_t i = 0; i < code.size(); ++i) {
            if (code[i].op == "label") {
                labels[code[i].result] = static_cast<int>(i);
            }
        }
    }

    bool run(double& retVal) {
        int pc = 0;
        while (pc >= 0 && pc < static_cast<int>(code.size())) {
            const auto& ins = code[pc];
            if (ins.op == "label") { pc++; continue; }
            if (ins.op == "goto") { pc = jumpTo(ins.result); continue; }
            if (ins.op == "iffalse") {
                double c = valueOf(ins.arg1);
                if (c == 0.0) pc = jumpTo(ins.result);
                else pc++;
                continue;
            }
            if (ins.op == "iftrue") {
                double c = valueOf(ins.arg1);
                if (c != 0.0) pc = jumpTo(ins.result);
                else pc++;
                continue;
            }
            if (ins.op == "return") {
                retVal = ins.arg1.empty() ? 0.0 : valueOf(ins.arg1);
                return true;
            }
            if (ins.op == "print") {
                if (ins.arg1.size() > 0 && ins.arg1[0] == 's' && stringTable.count(ins.arg1)) {
                    std::cout << stringTable.at(ins.arg1) << "\n";
                } else {
                    std::cout << valueOf(ins.arg1) << "\n";
                }
                pc++;
                continue;
            }
            if (ins.op == "param" || ins.op == "call") {
                pc++;
                continue;
            }

            if (ins.op == "=") {
                setVar(ins.result, valueOf(ins.arg1));
                pc++;
                continue;
            }
            if (ins.op == "!") {
                setVar(ins.result, valueOf(ins.arg1) == 0.0 ? 1.0 : 0.0);
                pc++;
                continue;
            }
            if (ins.op == "cast") {
                double v = valueOf(ins.arg1);
                if (ins.arg2 == "int") v = static_cast<int>(v);
                else if (ins.arg2 == "float") v = static_cast<double>(v);
                else if (ins.arg2 == "bool") v = (v == 0.0 ? 0.0 : 1.0);
                else if (ins.arg2 == "char") v = static_cast<int>(v) & 0xFF;
                setVar(ins.result, v);
                pc++;
                continue;
            }

            double a = valueOf(ins.arg1);
            double b = valueOf(ins.arg2);
            if (ins.op == "+") setVar(ins.result, a + b);
            else if (ins.op == "-") setVar(ins.result, a - b);
            else if (ins.op == "*") setVar(ins.result, a * b);
            else if (ins.op == "/") setVar(ins.result, a / b);
            else if (ins.op == "%") setVar(ins.result, std::fmod(a, b));
            else if (ins.op == "==") setVar(ins.result, (a == b) ? 1.0 : 0.0);
            else if (ins.op == "!=") setVar(ins.result, (a != b) ? 1.0 : 0.0);
            else if (ins.op == "<") setVar(ins.result, (a < b) ? 1.0 : 0.0);
            else if (ins.op == "<=") setVar(ins.result, (a <= b) ? 1.0 : 0.0);
            else if (ins.op == ">") setVar(ins.result, (a > b) ? 1.0 : 0.0);
            else if (ins.op == ">=") setVar(ins.result, (a >= b) ? 1.0 : 0.0);
            else if (ins.op == "&&") setVar(ins.result, (a != 0.0 && b != 0.0) ? 1.0 : 0.0);
            else if (ins.op == "||") setVar(ins.result, (a != 0.0 || b != 0.0) ? 1.0 : 0.0);
            else {
                throw std::runtime_error("VM 不支持的 op: " + ins.op);
            }
            pc++;
        }
        return false;
    }

private:
    const std::vector<Tac>& code;
    std::unordered_map<std::string, int> labels;
    std::unordered_map<std::string, double> vars;
    std::unordered_map<std::string, std::string> stringTable;

    bool isNumber(const std::string& s) const {
        if (s.empty()) return false;
        char* end = nullptr;
        std::strtod(s.c_str(), &end);
        return end && *end == '\0';
    }

    double valueOf(const std::string& x) {
        if (x.empty()) return 0.0;
        if (x.size() > 0 && x[0] == 's' && stringTable.count(x)) {
            return 1.0;
        }
        if (isNumber(x)) return std::stod(x);
        auto it = vars.find(x);
        if (it == vars.end()) return 0.0;
        return it->second;
    }

    void setVar(const std::string& name, double value) {
        vars[name] = value;
    }

    int jumpTo(const std::string& label) {
        auto it = labels.find(label);
        if (it == labels.end()) {
            throw std::runtime_error("未定义的标签: " + label);
        }
        return it->second;
    }
};

// 汇编代码生成器
class AssemblyGenerator {
public:
    explicit AssemblyGenerator(const std::vector<Tac>& code) : tac(code) {}

    std::string generate() {
        std::ostringstream out;
        collectSymbols();
        out << "; === ASM (MIPS-like pseudo) ===\n";
        out << ".data\n";

        for (const auto& v : variables) out << v << ":\t.word 0\n";
        for (const auto& s : spills) out << s << ":\t.word 0\n";
        for (const auto& [strId, strVal] : stringTable) {
            out << strId << ":\t.asciiz \"" << escapeString(strVal) << "\"\n";
        }

        out << "\n.text\n";
        out << ".globl main\n";
        out << "main:\n";

        for (const auto& ins : tac) {
            if (ins.op == "label") {
                out << ins.result << ":\n";
                continue;
            }

            if (ins.op == "goto") {
                out << "\tj " << ins.result << "\n";
                continue;
            }

            if (ins.op == "iffalse") {
                std::string r = ensureReg(ins.arg1, out, "$t8");
                out << "\tbeq " << r << ", $zero, " << ins.result << "\n";
                continue;
            }

            if (ins.op == "iftrue") {
                std::string r = ensureReg(ins.arg1, out, "$t8");
                out << "\tbne " << r << ", $zero, " << ins.result << "\n";
                continue;
            }

            if (ins.op == "print") {
                if (ins.arg1.size() > 0 && ins.arg1[0] == 's' && stringTable.count(ins.arg1)) {
                    out << "\t# print string\n";
                    out << "\tli $v0, 4\n";
                    out << "\tla $a0, " << ins.arg1 << "\n";
                    out << "\tsyscall\n";
                } else {
                    std::string r = ensureReg(ins.arg1, out, "$t8");
                    out << "\t# print value\n";
                    out << "\tli $v0, 1\n";
                    out << "\tmove $a0, " << r << "\n";
                    out << "\tsyscall\n";
                }
                out << "\t# print newline\n";
                out << "\tli $v0, 11\n";
                out << "\tli $a0, 10\n";
                out << "\tsyscall\n";
                continue;
            }

            if (ins.op == "return") {
                if (!ins.arg1.empty()) {
                    std::string r = ensureReg(ins.arg1, out, "$t8");
                    out << "\tmove $v0, " << r << "\n";
                }
                out << "\tjr $ra\n";
                continue;
            }

            if (ins.op == "param" || ins.op == "call") {
                out << "\t# " << ins.op << " " << ins.arg1 << " " << ins.arg2 << " " << ins.result
                    << " (not lowered)\n";
                continue;
            }

            if (ins.op == "cast") {
                std::string src = ensureReg(ins.arg1, out, "$t8");
                std::string dst = ensureDest(ins.result);
                out << "\t# cast (" << ins.arg2 << ") " << ins.arg1 << "\n";
                emitMoveToDest(dst, src, out);
                continue;
            }

            if (ins.op == "!") {
                std::string src = ensureReg(ins.arg1, out, "$t8");
                std::string dst = ensureDest(ins.result);
                std::string dreg = ensureDestReg(dst, out, "$t9");
                out << "\t# logical not\n";
                out << "\tseq " << dreg << ", " << src << ", $zero\n";
                storeDest(dst, dreg, out);
                continue;
            }

            if (ins.op == "=") {
                std::string src = ensureReg(ins.arg1, out, "$t8");
                std::string dst = ensureDest(ins.result);
                emitMoveToDest(dst, src, out);
                continue;
            }

            if (ins.op == "+" || ins.op == "-" || ins.op == "*" || ins.op == "/" || ins.op == "%" ||
                ins.op == "==" || ins.op == "!=" || ins.op == "<" || ins.op == "<=" || ins.op == ">" || ins.op == ">=" ||
                ins.op == "&&" || ins.op == "||") {

                std::string a = ensureReg(ins.arg1, out, "$t8");
                std::string b = ensureReg(ins.arg2, out, "$t9");
                std::string dst = ensureDest(ins.result);
                std::string dreg = ensureDestReg(dst, out, "$t8");

                if (ins.op == "+") out << "\tadd " << dreg << ", " << a << ", " << b << "\n";
                else if (ins.op == "-") out << "\tsub " << dreg << ", " << a << ", " << b << "\n";
                else if (ins.op == "*") out << "\tmul " << dreg << ", " << a << ", " << b << "\n";
                else if (ins.op == "/") {
                    out << "\tdiv " << a << ", " << b << "\n";
                    out << "\tmflo " << dreg << "\n";
                } else if (ins.op == "%") {
                    out << "\tdiv " << a << ", " << b << "\n";
                    out << "\tmfhi " << dreg << "\n";
                } else if (ins.op == "==") out << "\tseq " << dreg << ", " << a << ", " << b << "\n";
                else if (ins.op == "!=") out << "\tsne " << dreg << ", " << a << ", " << b << "\n";
                else if (ins.op == "<") out << "\tslt " << dreg << ", " << a << ", " << b << "\n";
                else if (ins.op == ">") out << "\tslt " << dreg << ", " << b << ", " << a << "\n";
                else if (ins.op == "<=") {
                    out << "\tslt " << dreg << ", " << b << ", " << a << "\n";
                    out << "\txori " << dreg << ", " << dreg << ", 1\n";
                } else if (ins.op == ">=") {
                    out << "\tslt " << dreg << ", " << a << ", " << b << "\n";
                    out << "\txori " << dreg << ", " << dreg << ", 1\n";
                } else if (ins.op == "&&") {
                    out << "\tsne " << dreg << ", " << a << ", $zero\n";
                    out << "\tbeq " << dreg << ", $zero, skip_" << labelCounter << "\n";
                    out << "\tsne " << dreg << ", " << b << ", $zero\n";
                    out << "skip_" << labelCounter << ":\n";
                    labelCounter++;
                } else if (ins.op == "||") {
                    out << "\tsne " << dreg << ", " << a << ", $zero\n";
                    out << "\tbne " << dreg << ", $zero, skip_" << labelCounter << "\n";
                    out << "\tsne " << dreg << ", " << b << ", $zero\n";
                    out << "skip_" << labelCounter << ":\n";
                    labelCounter++;
                }

                storeDest(dst, dreg, out);
                continue;
            }

            out << "\t# Unsupported: " << ins.op << " " << ins.arg1 << " " << ins.arg2 << " " << ins.result << "\n";
        }

        out << "\t# Exit program\n";
        out << "\tli $v0, 10\n";
        out << "\tsyscall\n";

        return out.str();
    }

private:
    const std::vector<Tac>& tac;
    std::unordered_set<std::string> variables;
    std::unordered_set<std::string> spills;
    std::unordered_map<std::string, std::string> tempReg;
    std::unordered_map<std::string, std::string> stringTable;
    int nextTempReg = 0;
    int labelCounter = 0;

    std::string escapeString(const std::string& s) {
        std::string res;
        for (char c : s) {
            switch (c) {
                case '\n': res += "\\n"; break;
                case '\t': res += "\\t"; break;
                case '\\': res += "\\\\"; break;
                case '"': res += "\\\""; break;
                case '\'': res += "\\'"; break;
                default: res += c; break;
            }
        }
        return res;
    }

    static bool isNumber(const std::string& s) {
        if (s.empty()) return false;
        char* end = nullptr;
        std::strtod(s.c_str(), &end);
        return end && *end == '\0';
    }

    static bool isTemp(const std::string& s) { return !s.empty() && (s[0] == 't' || s[0] == 'i' || s[0] == 'f' || s[0] == 'b' || s[0] == 'c' || s[0] == 's'); }
    static bool isLabel(const std::string& s) { return !s.empty() && s[0] == 'L'; }
    static bool isStringId(const std::string& s) { return !s.empty() && s[0] == 's'; }

    static bool isIdentLike(const std::string& s) {
        if (s.empty()) return false;
        if (!(std::isalpha((unsigned char)s[0]) || s[0] == '_')) return false;
        for (char c : s) {
            if (!(std::isalnum((unsigned char)c) || c == '_')) return false;
        }
        return true;
    }

    void collectSymbols() {
        auto touch = [&](const std::string& x) {
            if (x.empty()) return;
            if (isLabel(x) || x == "<eof>") return;
            if (isNumber(x)) return;
            if (isStringId(x)) {
                stringTable[x] = "";
                return;
            }
            if (isTemp(x)) return;
            if (isIdentLike(x)) variables.insert(x);
        };

        for (const auto& ins : tac) {
            if (ins.op == "label" || ins.op == "goto") continue;
            touch(ins.arg1);
            touch(ins.arg2);
            touch(ins.result);
        }

        std::unordered_set<std::string> temps;
        for (const auto& ins : tac) {
            if (isTemp(ins.arg1)) temps.insert(ins.arg1);
            if (isTemp(ins.arg2)) temps.insert(ins.arg2);
            if (isTemp(ins.result)) temps.insert(ins.result);
        }

        int cnt = 0;
        for (const auto& t : temps) {
            if (cnt < 8) {
                tempReg[t] = "$t" + std::to_string(cnt);
            } else {
                spills.insert("_spill_" + t);
            }
            cnt++;
        }
    }

    std::string ensureDest(const std::string& x) const {
        if (isTemp(x) || isIdentLike(x)) return x;
        return x;
    }

    std::string ensureReg(const std::string& op, std::ostringstream& out, const std::string& scratchPref) {
        if (op.empty()) return "$zero";
        if (isNumber(op)) {
            out << "\tli " << scratchPref << ", " << op << "\n";
            return scratchPref;
        }
        if (isTemp(op)) {
            auto it = tempReg.find(op);
            if (it != tempReg.end()) return it->second;
            out << "\tlw " << scratchPref << ", _spill_" << op << "\n";
            return scratchPref;
        }
        if (isIdentLike(op)) {
            out << "\tlw " << scratchPref << ", " << op << "\n";
            return scratchPref;
        }
        return scratchPref;
    }

    std::string ensureDestReg(const std::string& dst, std::ostringstream& out, const std::string& scratchPref) {
        if (isTemp(dst)) {
            auto it = tempReg.find(dst);
            if (it != tempReg.end()) return it->second;
            return scratchPref;
        }
        return scratchPref;
    }

    void storeDest(const std::string& dst, const std::string& reg, std::ostringstream& out) {
        if (isTemp(dst)) {
            auto it = tempReg.find(dst);
            if (it != tempReg.end()) return;
            out << "\tsw " << reg << ", _spill_" << dst << "\n";
            return;
        }
        if (isIdentLike(dst)) {
            out << "\tsw " << reg << ", " << dst << "\n";
            return;
        }
    }

    void emitMoveToDest(const std::string& dst, const std::string& srcReg, std::ostringstream& out) {
        if (isTemp(dst)) {
            auto it = tempReg.find(dst);
            if (it != tempReg.end()) {
                out << "\tmove " << it->second << ", " << srcReg << "\n";
            } else {
                out << "\tsw " << srcReg << ", _spill_" << dst << "\n";
            }
            return;
        }
        if (isIdentLike(dst)) {
            out << "\tsw " << srcReg << ", " << dst << "\n";
            return;
        }
        out << "\t# move (ignored dest=" << dst << ")\n";
    }
};

// 主函数
int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "用法: compiler <源代码文件> [--run] [--ast-dot <dot文件路径>]\n";
            std::cerr << "选项:\n";
            std::cerr << "  --run        运行TAC虚拟机\n";
            std::cerr << "  --ast-dot    生成AST的Graphviz DOT文件\n";
            return 1;
        }

        bool runVm = false;
        std::string filePath;
        std::string astDotPath;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--run") {
                runVm = true;
            } else if (arg == "--ast-dot") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("参数 --ast-dot 需要指定输出文件路径");
                }
                astDotPath = argv[++i];
            } else if (arg.find("--") == 0) {
                throw std::runtime_error("未知参数: " + arg);
            } else {
                filePath = arg;
            }
        }

        if (filePath.empty()) {
            throw std::runtime_error("缺少源代码文件路径");
        }

        // 1. 读取源文件
        std::string source = readFile(filePath);

        // 2. 词法分析
        Lexer lexer(source);
        auto tokens = lexer.tokenize();
        std::cout << "=== TOKENS ===\n";
        for (const auto& t : tokens) {
            if (t.type == TokenType::EndOfFile) break;
            std::cout << std::setw(4) << t.line << ":" << std::setw(3) << t.column << "  " << std::left
                        << std::setw(12) << tokenTypeName(t.type) << "  " << t.lexeme << "\n";
        }

        // 3. 语法分析 + 语义分析 + IR生成
        CodeGenerator codeGen;
        SymbolTable symbols;
        Parser parser(tokens, codeGen, symbols);
        parser.parseProgram();
        auto astRoot = parser.getASTRoot();

        // 4. 输出AST
        std::cout << "\n=== AST ===\n";
        printAST(astRoot);

        // 生成AST的DOT文件
        if (!astDotPath.empty()) {
            std::ofstream dotFile(astDotPath);
            if (!dotFile) {
                throw std::runtime_error("无法创建DOT文件: " + astDotPath);
            }
            dotFile << astToDot(astRoot);
            dotFile.close();
            std::cout << "(AST DOT文件已保存到: " << astDotPath << ")\n";
        }

        // 5. 输出符号表
        std::cout << "\n=== SYMTAB ===\n";
        symbols.print();

        // 6. 输出IR（TAC）
        const auto& tacCode = codeGen.getCode();
        std::cout << "\n=== IR ===\n";
        int irIndex = 0;
        for (const auto& ins : tacCode) {
            std::cout << std::setw(3) << irIndex++ << ": ";
            if (ins.op == "label") {
                std::cout << ins.result << ":";
            } else if (ins.op == "goto" || ins.op == "iffalse" || ins.op == "iftrue") {
                std::cout << ins.op << " " << ins.arg1 << " " << ins.result;
            } else if (ins.op == "return") {
                std::cout << "return " << ins.arg1;
            } else if (ins.op == "print") {
                std::cout << "print " << ins.arg1;
            } else if (ins.op == "param") {
                std::cout << "param " << ins.arg1;
            } else if (ins.op == "call") {
                std::cout << ins.result << " = call " << ins.arg1 << ", " << ins.arg2;
            } else if (ins.op == "=" && ins.arg2.empty()) {
                std::cout << ins.result << " = " << ins.arg1;
            } else if (ins.op == "cast") {
                std::cout << ins.result << " = (" << ins.arg2 << ")" << ins.arg1;
            } else if (ins.op == "!" && ins.arg2.empty()) {
                std::cout << ins.result << " = !" << ins.arg1;
            } else {
                std::cout << ins.result << " = " << ins.arg1 << " " << ins.op << " " << ins.arg2;
            }
            std::cout << "\n";
        }

        // 7. 生成汇编代码
        std::cout << "\n=== ASM ===\n";
        AssemblyGenerator asmGen(tacCode);
        std::cout << asmGen.generate();

        std::cout << "\n编译成功！\n";

        // 8. 运行TAC虚拟机
        if (runVm) {
            std::cout << "\n=== 虚拟机运行结果 ===\n";
            TacVM vm(tacCode, codeGen.stringTable);
            double retVal = 0.0;
            bool hasReturn = vm.run(retVal);
            if (hasReturn) {
                std::cout << "\n程序返回值: " << retVal << "\n";
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "\n编译失败: " + std::string(e.what()) + "\n";
        return 1;
    }
    return 0;
}