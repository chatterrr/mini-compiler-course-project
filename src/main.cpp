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

enum class TokenType {
    Identifier,
    Number,
    Keyword,
    Operator,
    Separator,
    EndOfFile
};

struct Token {
    TokenType type;
    std::string lexeme;
    int line;
    int column;
};

enum class ValueType { Int, Float, Bool, Void };

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
    Print
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

struct Tac {
    std::string op;
    std::string arg1;
    std::string arg2;
    std::string result;
};

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
        {"int", TokenType::Keyword},   {"float", TokenType::Keyword}, {"if", TokenType::Keyword},
        {"else", TokenType::Keyword},  {"while", TokenType::Keyword}, {"for", TokenType::Keyword},
        {"return", TokenType::Keyword}};
    const std::unordered_map<std::string, std::string> twoCharOps{
        {"==", "=="}, {"!=", "!="}, {"<=", "<="}, {">=", ">="}, {"&&", "&&"}, {"||", "||"}};
    const std::unordered_map<char, std::string> oneCharOps{
        {'+', "+"}, {'-', "-"}, {'*', "*"}, {'/', "/"}, {'%', "%"}, {'=', "="},
        {'<', "<"}, {'>', ">"}, {'!', "!"}};
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
            return Token{TokenType::Keyword, lexeme, startLine, startCol};
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

    Token singleCharToken(TokenType type, const std::string &lexeme, int startLine, int startCol) {
        return Token{type, lexeme, startLine, startCol};
    }

    Token operatorToken(int startLine, int startCol) {
        std::string lexeme;
        lexeme += advance();
        std::string twoChars = lexeme + peek();
        if (!isAtEnd() && twoCharOps.count(twoChars)) {
            lexeme += advance();
            return Token{TokenType::Operator, lexeme, startLine, startCol};
        }
        if (oneCharOps.count(lexeme[0])) {
            return Token{TokenType::Operator, lexeme, startLine, startCol};
        }
        throw std::runtime_error("Unexpected character '" + lexeme + "' at line " + std::to_string(startLine) +
                                " column " + std::to_string(startCol));
    }
};

class SymbolTable {
  public:
    struct Entry {
        std::string name;
        ValueType type;
        int scopeDepth;
        int declLine;
        int declCol;
    };

    SymbolTable() { pushScope(); }

    void pushScope() { scopes.emplace_back(); }
    void popScope() {
        // 课程设计展示用：我们允许退出作用域，但会保留历史声明用于最终打印
        if (scopes.size() <= 1) return;
        scopes.pop_back();
    }

    int currentDepth() const { return static_cast<int>(scopes.size()) - 1; }

    void declare(const std::string &name, ValueType type, int line, int col) {
        auto &currentScope = scopes.back();
        if (currentScope.count(name)) {
            throw std::runtime_error("重复声明标识符 '" + name + "'，位置: line " + std::to_string(line) + " col " +
                                     std::to_string(col));
        }
        currentScope[name] = type;
        history.push_back(Entry{name, type, currentDepth(), line, col});
    }

    ValueType lookup(const std::string &name, int line, int col) const {
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            auto found = it->find(name);
            if (found != it->end()) return found->second;
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
        for (const auto &e : history) maxDepth = std::max(maxDepth, e.scopeDepth);

        for (int d = 0; d <= maxDepth; ++d) {
            bool any = false;
            for (const auto &e : history) {
                if (e.scopeDepth != d) continue;
                if (!any) {
                    std::cout << "[scope " << d << "]\n";
                    any = true;
                }
                std::cout << "  " << e.name << "\t" << typeStr(e.type)
                          << "\t@(" << e.declLine << ":" << e.declCol << ")\n";
            }
        }
    }

  private:
    std::vector<std::unordered_map<std::string, ValueType>> scopes;
    std::vector<Entry> history;
};


class CodeGenerator {
  public:
    std::string newTemp(ValueType type) {
        std::string name = "t" + std::to_string(tempIndex++);
        tempType[name] = type;
        return name;
    }

    std::string newLabel() { return "L" + std::to_string(labelIndex++); }

    void emit(const std::string &op, const std::string &a1 = "", const std::string &a2 = "",
               const std::string &res = "") {
        code.push_back({op, a1, a2, res});
    }

    const std::vector<Tac> &getCode() const { return code; }

  private:
    int tempIndex = 0;
    int labelIndex = 0;
    std::unordered_map<std::string, ValueType> tempType;
    std::vector<Tac> code;
};

struct ExprResult {
    std::string place;
    ValueType type;
    std::shared_ptr<ASTNode> expr;
};

class Parser {
  public:
    Parser(std::vector<Token> tokens, CodeGenerator &gen, SymbolTable &symbols)
        : toks(std::move(tokens)), code(gen), symbols(symbols) {
        astRoot = std::make_shared<ASTNode>(ASTNodeType::Program, "program");
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
    CodeGenerator &code;
    SymbolTable &symbols;
    std::shared_ptr<ASTNode> astRoot;

    bool isAtEnd() const { return peek().type == TokenType::EndOfFile; }
    const Token &peek() const { return toks[current]; }
    const Token &previous() const { return toks[current - 1]; }

    bool check(TokenType type, const std::string &lexeme = "") const {
        if (isAtEnd()) return type == TokenType::EndOfFile;
        if (peek().type != type) return false;
        if (!lexeme.empty() && peek().lexeme != lexeme) return false;
        return true;
    }

    const Token &advance() {
        if (!isAtEnd()) current++;
        return previous();
    }

    bool match(TokenType type, const std::string &lexeme = "") {
        if (check(type, lexeme)) {
            advance();
            return true;
        }
        return false;
    }

    const Token &consume(TokenType type, const std::string &lexeme, const std::string &msg) {
        if (check(type, lexeme)) return advance();
        throw std::runtime_error(errorAt(peek(), msg));
    }

    const Token &consume(TokenType type, const std::string &msg) {
        if (check(type)) return advance();
        throw std::runtime_error(errorAt(peek(), msg));
    }

    std::string errorAt(const Token &tok, const std::string &msg) {
        std::ostringstream oss;
        oss << "语法错误 (line " << tok.line << ", col " << tok.column << "): " << msg << "，发现 '" << tok.lexeme
            << "'";
        return oss.str();
    }

    std::shared_ptr<ASTNode> parseStatement() {
        if (match(TokenType::Keyword, "int") || match(TokenType::Keyword, "float")) {
            ValueType t = (previous().lexeme == "int") ? ValueType::Int : ValueType::Float;
            return parseDeclaration(t);
        } else if (match(TokenType::Keyword, "if")) {
            return parseIf();
        } else if (match(TokenType::Keyword, "while")) {
            return parseWhile();
        } else if (match(TokenType::Keyword, "for")) {
            return parseFor();
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
        const Token &idTok = consume(TokenType::Identifier, "缺少标识符");
        symbols.declare(idTok.lexeme, type, idTok.line, idTok.column);

        auto declNode = std::make_shared<ASTNode>(ASTNodeType::Declaration, idTok.lexeme, type, idTok.line, idTok.column);
        auto varNode = std::make_shared<ASTNode>(ASTNodeType::Variable, idTok.lexeme, type, idTok.line, idTok.column);
        declNode->children.push_back(varNode);

        if (match(TokenType::Operator, "=")) {
            ExprResult init = expression();
            ExprResult rhs = cast(init, type);
            code.emit("=", rhs.place, "", idTok.lexeme);

            auto assignNode = std::make_shared<ASTNode>(ASTNodeType::Assignment, "=", type, idTok.line, idTok.column);
            assignNode->children.push_back(varNode);
            assignNode->children.push_back(init.expr);
            declNode->children.push_back(assignNode);
        }
        consume(TokenType::Separator, ";", "缺少分号");
        return declNode;
    }

    std::shared_ptr<ASTNode> parseIf() {
        const Token &ifTok = previous();
        consume(TokenType::Separator, "(", "缺少 '('");
        ExprResult cond = expression();
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
        const Token &whileTok = previous();
        std::string startLabel = code.newLabel();
        std::string endLabel = code.newLabel();
        code.emit("label", "", "", startLabel);
        consume(TokenType::Separator, "(", "缺少 '('");
        ExprResult cond = expression();
        consume(TokenType::Separator, ")", "缺少 ')'");
        code.emit("iffalse", ensureBool(cond).place, "", endLabel);

        auto whileNode = std::make_shared<ASTNode>(ASTNodeType::While, "while", ValueType::Void, whileTok.line, whileTok.column);
        whileNode->children.push_back(cond.expr);

        auto bodyStmt = parseStatement();
        whileNode->children.push_back(bodyStmt);

        code.emit("goto", "", "", startLabel);
        code.emit("label", "", "", endLabel);
        return whileNode;
    }

    std::shared_ptr<ASTNode> parseFor() {
        const Token &forTok = previous();
        consume(TokenType::Separator, "(", "缺少 '('");

        auto forNode = std::make_shared<ASTNode>(ASTNodeType::For, "for", ValueType::Void, forTok.line, forTok.column);

        // init
        if (!match(TokenType::Separator, ";")) {
            if (check(TokenType::Keyword, "int") || check(TokenType::Keyword, "float")) {
                ValueType t = (peek().lexeme == "int") ? ValueType::Int : ValueType::Float;
                advance();
                auto initDecl = parseDeclaration(t);
                forNode->children.push_back(initDecl);
            } else {
                auto initExpr = parseExpressionStatement();
                forNode->children.push_back(initExpr);
            }
        }

        std::string condLabel = code.newLabel();
        std::string endLabel = code.newLabel();
        std::string bodyLabel = code.newLabel();
        code.emit("label", "", "", condLabel);

        ExprResult cond = { "", ValueType::Bool, nullptr };
        if (!check(TokenType::Separator, ";")) {
            cond = expression();
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

        // parse increment using a nested parser on the token slice
        if (!incTokens.empty()) {
            incTokens.push_back(Token{TokenType::Separator, ";", peek().line, peek().column});
            incTokens.push_back(Token{TokenType::EndOfFile, "<eof>", peek().line, peek().column});
            Parser incParser(incTokens, code, symbols);
            auto incExpr = incParser.parseExpressionStatement();
            forNode->children.push_back(incExpr);
        }

        code.emit("goto", "", "", condLabel);
        code.emit("label", "", "", endLabel);
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
        const Token &retTok = previous();
        auto returnNode = std::make_shared<ASTNode>(ASTNodeType::Return, "return", ValueType::Void, retTok.line, retTok.column);

        if (match(TokenType::Separator, ";")) {
            code.emit("return", "", "", "");
            return returnNode;
        }
        ExprResult value = expression();
        consume(TokenType::Separator, ";", "缺少分号");
        code.emit("return", value.place, "", "");

        returnNode->children.push_back(value.expr);
        return returnNode;
    }

    std::shared_ptr<ASTNode> parseExpressionStatement() {
        ExprResult value = expression();
        consume(TokenType::Separator, ";", "缺少分号");
        return value.expr;
    }

    ExprResult expression() { return logical_or(); }

ExprResult logical_or() {
    ExprResult left = logical_and();
    while (match(TokenType::Operator, "||")) {
        ExprResult right = logical_and();
        // short-circuit OR
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
        // short-circuit AND
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
        if (match(TokenType::Operator, "-")) {
            ExprResult rhs = unary();
            ExprResult zero = {rhs.type == ValueType::Float ? "0.0" : "0", rhs.type, nullptr};
            ExprResult result = {code.newTemp(rhs.type), rhs.type};
            code.emit("-", zero.place, rhs.place, result.place);

            auto unaryNode = std::make_shared<ASTNode>(ASTNodeType::UnaryOp, "-", rhs.type, previous().line, previous().column);
            unaryNode->children.push_back(rhs.expr);
            result.expr = unaryNode;
            return result;
        }
        if (match(TokenType::Operator, "!")) {
            ExprResult rhs = unary();
            ExprResult result = {code.newTemp(ValueType::Bool), ValueType::Bool, nullptr};
            code.emit("!", rhs.place, "", result.place);

            auto unaryNode = std::make_shared<ASTNode>(ASTNodeType::UnaryOp, "!", ValueType::Bool, previous().line, previous().column);
            unaryNode->children.push_back(rhs.expr);
            result.expr = unaryNode;
            return result;
        }
        return primary();
    }

    ExprResult primary() {
        if (match(TokenType::Number)) {
            const Token &numTok = previous();
            ValueType vt = (numTok.lexeme.find('.') != std::string::npos) ? ValueType::Float : ValueType::Int;
            auto node = std::make_shared<ASTNode>(ASTNodeType::Literal, numTok.lexeme, vt, numTok.line, numTok.column);
            return ExprResult{numTok.lexeme, vt, node};
        }
if (match(TokenType::Identifier)) {
    const Token &idTok = previous();
    // function / procedure call: Identifier '(' args? ')'
    if (match(TokenType::Separator, "(")) {
        std::vector<ExprResult> args;
        if (!check(TokenType::Separator, ")")) {
            args.push_back(expression());
            while (match(TokenType::Separator, ",")) {
                args.push_back(expression());
            }
        }
        consume(TokenType::Separator, ")", "缺少 ')'");

        // emit params
        for (auto &a : args) {
            code.emit("param", a.place, "", "");
        }

        // built-in: print(...)
        if (idTok.lexeme == "print") {
            if (args.empty()) {
                throw std::runtime_error(errorAt(idTok, "print 至少需要 1 个参数"));
            }
            for (auto &a : args) code.emit("print", a.place, "", "");

            auto printNode = std::make_shared<ASTNode>(ASTNodeType::Print, "print", ValueType::Void, idTok.line, idTok.column);
            for (auto &a : args) {
                printNode->children.push_back(a.expr);
            }

            // as an expression, return 0
            return ExprResult{"0", ValueType::Int, printNode};
        }

        // generic call: call name, nargs -> t
        std::string ret = code.newTemp(ValueType::Int);
        code.emit("call", idTok.lexeme, std::to_string(args.size()), ret);

        auto callNode = std::make_shared<ASTNode>(ASTNodeType::FunctionCall, idTok.lexeme, ValueType::Int, idTok.line, idTok.column);
        for (auto &a : args) {
            callNode->children.push_back(a.expr);
        }

        return ExprResult{ret, ValueType::Int, callNode};
    }

    // assignment or variable reference
    ValueType t = symbols.lookup(idTok.lexeme, idTok.line, idTok.column);
    auto varNode = std::make_shared<ASTNode>(ASTNodeType::Variable, idTok.lexeme, t, idTok.line, idTok.column);

    if (match(TokenType::Operator, "=")) {
        ExprResult rhs = expression();
        ExprResult casted = cast(rhs, t);
        code.emit("=", casted.place, "", idTok.lexeme);

        auto assignNode = std::make_shared<ASTNode>(ASTNodeType::Assignment, "=", t, idTok.line, idTok.column);
        assignNode->children.push_back(varNode);
        assignNode->children.push_back(casted.expr);

        return ExprResult{casted.place, t, assignNode};
    }
    return ExprResult{idTok.lexeme, t, varNode};
}
        if (match(TokenType::Separator, "(")) {
            ExprResult inner = expression();
            consume(TokenType::Separator, ")", "缺少 ')'");
            return inner;
        }
        throw std::runtime_error(errorAt(peek(), "无法解析的表达式"));
    }

    ExprResult ensureBool(const ExprResult &expr) {
        if (expr.type == ValueType::Bool) return expr;
        if (expr.type == ValueType::Void) {
            throw std::runtime_error(errorAt(previous(), "void 类型不能作为条件表达式"));
        }
        ExprResult result = {code.newTemp(ValueType::Bool), ValueType::Bool, nullptr};
        std::string zero = (expr.type == ValueType::Float) ? "0.0" : "0";
        code.emit("!=", expr.place, zero, result.place);

        auto boolNode = std::make_shared<ASTNode>(ASTNodeType::BinaryOp, "!=", ValueType::Bool, expr.expr ? expr.expr->line : 0, expr.expr ? expr.expr->column : 0);
        if (expr.expr) boolNode->children.push_back(expr.expr);
        auto zeroNode = std::make_shared<ASTNode>(ASTNodeType::Literal, zero, expr.type, expr.expr ? expr.expr->line : 0, expr.expr ? expr.expr->column : 0);
        boolNode->children.push_back(zeroNode);
        result.expr = boolNode;

        return result;
    }

    ValueType promote(ValueType a, ValueType b) {
        if (a == ValueType::Float || b == ValueType::Float) return ValueType::Float;
        if (a == ValueType::Int || b == ValueType::Int) return ValueType::Int;
        return ValueType::Bool;
    }

    ExprResult cast(const ExprResult &expr, ValueType target) {
        if (expr.type == target) return expr;
        std::string temp = code.newTemp(target);
        code.emit("cast", expr.place, typeName(target), temp);

        auto castNode = std::make_shared<ASTNode>(ASTNodeType::BinaryOp, "cast", target, expr.expr ? expr.expr->line : 0, expr.expr ? expr.expr->column : 0);
        if (expr.expr) castNode->children.push_back(expr.expr);
        auto typeNode = std::make_shared<ASTNode>(ASTNodeType::Literal, typeName(target), target, expr.expr ? expr.expr->line : 0, expr.expr ? expr.expr->column : 0);
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
        case ValueType::Void:
        default:
            return "void";
        }
    }
};

std::string readFile(const std::string &path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("无法打开文件: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string tokenTypeName(TokenType t);

void printTokens(const std::vector<Token> &tokens) {
    std::cout << "=== TOKENS ===\n";
    for (const auto &t : tokens) {
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
    }
    return "Unknown";
}

void printTac(const std::vector<Tac> &code) {
    std::cout << "\n=== IR ===\n";
    int index = 0;
    for (const auto &i : code) {
        std::cout << std::setw(3) << index++ << ": ";
        if (i.op == "label") {
            std::cout << i.result << ":";
        } else if (i.op == "goto" || i.op == "iffalse") {
            std::cout << i.op << " " << i.arg1 << " " << i.result;
        } else if (i.op == "return") {
            std::cout << "return " << i.arg1;
        } else if (i.op == "print") {
            std::cout << "print " << i.arg1;
        } else if (i.op == "param") {
            std::cout << "param " << i.arg1;
        } else if (i.op == "call") {
            std::cout << i.result << " = call " << i.arg1 << ", " << i.arg2;
        } else if (i.op == "=" && i.arg2.empty()) {
            std::cout << i.result << " = " << i.arg1;
        } else if (i.op == "cast") {
            std::cout << i.result << " = (" << i.arg2 << ")" << i.arg1;
        } else if (i.op == "!" && i.arg2.empty()) {
            std::cout << i.result << " = !" << i.arg1;
        } else {
            std::cout << i.result << " = " << i.arg1 << " " << i.op << " " << i.arg2;
        }
        std::cout << "\n";
    }
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
    }
    return "Unknown";
}

std::string valueTypeName(ValueType t) {
    switch (t) {
    case ValueType::Int: return "int";
    case ValueType::Float: return "float";
    case ValueType::Bool: return "bool";
    case ValueType::Void: return "void";
    }
    return "unknown";
}

void printAST(const std::shared_ptr<ASTNode> &node, int indent = 0) {
    if (!node) return;

    std::string indentStr(indent, ' ');
    std::cout << indentStr << nodeTypeName(node->type) << " '" << node->value << "' "
              << valueTypeName(node->dataType) << " (" << node->line << ":" << node->column << ")";

    if (!node->children.empty()) {
        std::cout << " {\n";
        for (const auto &child : node->children) {
            printAST(child, indent + 2);
        }
        std::cout << indentStr << "}";
    }
    std::cout << "\n";
}
// 生成 AST 的 Graphviz DOT（可选：用于图形化展示）
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



class TacVM {
  public:
    explicit TacVM(const std::vector<Tac> &code) : code(code) {
        for (size_t i = 0; i < code.size(); ++i) {
            if (code[i].op == "label") {
                labels[code[i].result] = static_cast<int>(i);
            }
        }
    }

    // execute TAC; return optional return value (if any)
    bool run(double &retVal) {
        int pc = 0;
        while (pc >= 0 && pc < static_cast<int>(code.size())) {
            const auto &ins = code[pc];
            if (ins.op == "label") { pc++; continue; }
            if (ins.op == "goto") { pc = jumpTo(ins.result); continue; }
            if (ins.op == "iffalse") {
                double c = valueOf(ins.arg1);
                if (c == 0.0) pc = jumpTo(ins.result);
                else pc++;
                continue;
            }
            if (ins.op == "return") {
                retVal = ins.arg1.empty() ? 0.0 : valueOf(ins.arg1);
                return true;
            }
            if (ins.op == "print") {
                std::cout << valueOf(ins.arg1) << "\n";
                pc++;
                continue;
            }
            if (ins.op == "param" || ins.op == "call") {
                // 教学版：除 print 外暂不执行通用函数调用
                pc++;
                continue;
            }

            // assignment / unary / binary / cast
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
                setVar(ins.result, v);
                pc++;
                continue;
            }

            // binary operations
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
            else {
                throw std::runtime_error("VM 不支持的 op: " + ins.op);
            }
            pc++;
        }
        return false;
    }

  private:
    const std::vector<Tac> &code;
    std::unordered_map<std::string, int> labels;
    std::unordered_map<std::string, double> vars;

    bool isNumber(const std::string &s) const {
        if (s.empty()) return false;
        char *end = nullptr;
        std::strtod(s.c_str(), &end);
        return end && *end == '\0';
    }

    double valueOf(const std::string &x) {
        if (x.empty()) return 0.0;
        if (isNumber(x)) return std::stod(x);
        auto it = vars.find(x);
        if (it == vars.end()) return 0.0; // 未初始化变量按 0 处理（教学版）
        return it->second;
    }

    void setVar(const std::string &name, double v) { vars[name] = v; }

    int jumpTo(const std::string &label) {
        auto it = labels.find(label);
        if (it == labels.end()) throw std::runtime_error("未知 label: " + label);
        return it->second;
    }
};

class AssemblyGenerator {
  public:
    explicit AssemblyGenerator(const std::vector<Tac> &code) : tac(code) {}

    // 说明：这里输出的是“教学用 MIPS-like 伪汇编”
    // - 变量/溢出临时：放在 .data 段（label + .word）
    // - 临时 t0,t1...：优先分配 $t0-$t7；超过 8 个则溢出到内存（label: _spill_tX）
    // - 立即数：用 li 装载到 $t8/$t9 作为 scratch
    std::string generate() {
        std::ostringstream out;

        collectSymbols();

        out << "; === ASM (MIPS-like pseudo) ===\n";
        out << ".data\n";

        for (const auto &v : variables) out << v << ":\t.word 0\n";
        for (const auto &s : spills) out << s << ":\t.word 0\n";

        out << "\n.text\n";
        out << ".globl main\n";
        out << "main:\n";

        for (const auto &ins : tac) {
            if (ins.op == "label") {
                out << ins.result << ":\n";
                continue;
            }

            if (ins.op == "goto") {
                out << "\tj " << ins.result << "\n";
                continue;
            }

            if (ins.op == "iffalse") {
                std::string r = ensureReg(ins.arg1, out, /*scratchPref=*/"$t8");
                out << "\tbeq " << r << ", $zero, " << ins.result << "\n";
                continue;
            }

            if (ins.op == "print") {
                std::string r = ensureReg(ins.arg1, out, "$t8");
                out << "\t# print\n";
                out << "\tli $v0, 1\n";
                out << "\tmove $a0, " << r << "\n";
                out << "\tsyscall\n";
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
                // 教学版：不区分浮点寄存器，cast 仅作为注释保留
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

            // 二元运算
            if (ins.op == "+" || ins.op == "-" || ins.op == "*" || ins.op == "/" || ins.op == "%" ||
                ins.op == "==" || ins.op == "!=" || ins.op == "<" || ins.op == "<=" || ins.op == ">" || ins.op == ">=") {

                std::string a = ensureReg(ins.arg1, out, "$t8");
                std::string b = ensureReg(ins.arg2, out, "$t9");
                std::string dst = ensureDest(ins.result);
                std::string dreg = ensureDestReg(dst, out, "$t8"); // 结果优先放 $t* 或 scratch

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
                    // a <= b  <=> !(b < a)
                    out << "\tslt " << dreg << ", " << b << ", " << a << "\n";
                    out << "\txori " << dreg << ", " << dreg << ", 1\n";
                } else if (ins.op == ">=") {
                    // a >= b  <=> !(a < b)
                    out << "\tslt " << dreg << ", " << a << ", " << b << "\n";
                    out << "\txori " << dreg << ", " << dreg << ", 1\n";
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
    const std::vector<Tac> &tac;

    std::unordered_set<std::string> variables;
    std::unordered_set<std::string> spills;
    std::unordered_map<std::string, std::string> tempReg; // tX -> $t0-$t7
    int nextTempReg = 0;

    static bool isNumber(const std::string &s) {
        if (s.empty()) return false;
        char *end = nullptr;
        std::strtod(s.c_str(), &end);
        return end && *end == '\0';
    }

    static bool isTemp(const std::string &s) { return !s.empty() && s[0] == 't'; }
    static bool isLabel(const std::string &s) { return !s.empty() && s[0] == 'L'; }

    static bool isIdentLike(const std::string &s) {
        if (s.empty()) return false;
        if (!(std::isalpha((unsigned char)s[0]) || s[0] == '_')) return false;
        for (char c : s) {
            if (!(std::isalnum((unsigned char)c) || c == '_')) return false;
        }
        return true;
    }

    void collectSymbols() {
        auto touch = [&](const std::string &x) {
            if (x.empty()) return;
            if (isLabel(x) || x == "<eof>") return;
            if (isNumber(x)) return;
            if (isTemp(x)) {
                // 可能会溢出，先不决定
                return;
            }
            if (isIdentLike(x)) variables.insert(x);
        };

        for (const auto &ins : tac) {
            if (ins.op == "label" || ins.op == "goto") {
                // label / goto 不触碰
            } else {
                touch(ins.arg1);
                touch(ins.arg2);
                touch(ins.result);
            }
        }

        // 预扫描临时变量数，如果超过 8 个就会 spill
        std::unordered_set<std::string> temps;
        for (const auto &ins : tac) {
            if (isTemp(ins.arg1)) temps.insert(ins.arg1);
            if (isTemp(ins.arg2)) temps.insert(ins.arg2);
            if (isTemp(ins.result)) temps.insert(ins.result);
        }

        int cnt = 0;
        for (const auto &t : temps) {
            if (cnt < 8) {
                tempReg[t] = "$t" + std::to_string(cnt); // $t0-$t7
            } else {
                spills.insert("_spill_" + t);
            }
            cnt++;
        }
    }

    std::string ensureDest(const std::string &x) const {
        // 目标只能是 temp 或变量
        if (isTemp(x) || isIdentLike(x)) return x;
        return x; // 容错：直接返回
    }

    // 确保 op 在寄存器中，必要时产生 li / lw
    std::string ensureReg(const std::string &op, std::ostringstream &out, const std::string &scratchPref) {
        if (op.empty()) return "$zero";
        if (isNumber(op)) {
            out << "\tli " << scratchPref << ", " << op << "\n";
            return scratchPref;
        }
        if (isTemp(op)) {
            auto it = tempReg.find(op);
            if (it != tempReg.end()) return it->second;
            // spill temp
            out << "\tlw " << scratchPref << ", _spill_" << op << "\n";
            return scratchPref;
        }
        if (isIdentLike(op)) {
            out << "\tlw " << scratchPref << ", " << op << "\n";
            return scratchPref;
        }
        // fallback
        return scratchPref;
    }

    // 目标的值应该最终落到寄存器 dreg（若目标是变量/溢出临时，需要 sw）
    std::string ensureDestReg(const std::string &dst, std::ostringstream &out, const std::string &scratchPref) {
        if (isTemp(dst)) {
            auto it = tempReg.find(dst);
            if (it != tempReg.end()) return it->second;
            // spill temp 用 scratch
            return scratchPref;
        }
        // 变量用 scratch
        return scratchPref;
    }

    void storeDest(const std::string &dst, const std::string &reg, std::ostringstream &out) {
        if (isTemp(dst)) {
            auto it = tempReg.find(dst);
            if (it != tempReg.end()) {
                // 已在寄存器，无需 store
                return;
            }
            out << "\tsw " << reg << ", _spill_" << dst << "\n";
            return;
        }
        if (isIdentLike(dst)) {
            out << "\tsw " << reg << ", " << dst << "\n";
            return;
        }
    }

    void emitMoveToDest(const std::string &dst, const std::string &srcReg, std::ostringstream &out) {
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


int main(int argc, char **argv) {
    try {
        if (argc < 2) {
            std::cerr << "用法: compiler <源代码文件> [--run] [--ast-dot <dot文件路径>]\n";
            return 1;
        }

        bool runVm = false;
        std::string filePath;
        std::string astDotPath;

        // 允许参数任意顺序
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--run") {
                runVm = true;
            } else if (a == "--ast-dot") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("参数 --ast-dot 需要一个输出路径");
                }
                astDotPath = argv[++i];
            } else if (!a.empty() && a[0] == '-') {
                throw std::runtime_error("未知参数: " + a);
            } else {
                filePath = a;
            }
        }

        if (filePath.empty()) {
            throw std::runtime_error("缺少源代码文件路径");
        }

        std::string src = readFile(filePath);

        // ===== 1) TOKENS =====
        Lexer lexer(src);
        auto tokens = lexer.tokenize();
        std::cout << "=== TOKENS ===\n";
        for (const auto &t : tokens) {
            if (t.type == TokenType::EndOfFile) break;
            std::cout << std::setw(4) << t.line << ":" << std::setw(3) << t.column << "  " << std::left
                      << std::setw(12) << tokenTypeName(t.type) << "  " << t.lexeme << "\n";
        }

        // ===== 2) PARSE + IR GEN =====
        CodeGenerator gen;
        SymbolTable symbols;
        Parser parser(tokens, gen, symbols);
        parser.parseProgram();

        // ===== 3) AST =====
        std::cout << "\n=== AST ===\n";
        printAST(parser.getASTRoot());

        // 可选：输出 DOT 文件
        if (!astDotPath.empty()) {
            std::ofstream out(astDotPath);
            if (!out) throw std::runtime_error("无法写入 AST DOT 文件: " + astDotPath);
            out << astToDot(parser.getASTRoot());
            out.close();
            std::cout << "(AST DOT 已输出到: " << astDotPath << ")\n";
        }

        // ===== 4) SYMTAB =====
        std::cout << "\n";
        symbols.print();

        // ===== 5) IR =====
        std::cout << "\n=== IR ===\n";
        int index = 0;
        for (const auto &i : gen.getCode()) {
            std::cout << std::setw(3) << index++ << ": ";
            if (i.op == "label") {
                std::cout << i.result << ":";
            } else if (i.op == "goto" || i.op == "iffalse") {
                std::cout << i.op << " " << i.arg1 << " " << i.result;
            } else if (i.op == "return") {
                std::cout << "return " << i.arg1;
            } else if (i.op == "print") {
                std::cout << "print " << i.arg1;
            } else if (i.op == "param") {
                std::cout << "param " << i.arg1;
            } else if (i.op == "call") {
                std::cout << i.result << " = call " << i.arg1 << ", " << i.arg2;
            } else if (i.op == "=" && i.arg2.empty()) {
                std::cout << i.result << " = " << i.arg1;
            } else if (i.op == "cast") {
                std::cout << i.result << " = (" << i.arg2 << ")" << i.arg1;
            } else if (i.op == "!" && i.arg2.empty()) {
                std::cout << i.result << " = !" << i.arg1;
            } else {
                std::cout << i.result << " = " << i.arg1 << " " << i.op << " " << i.arg2;
            }
            std::cout << "\n";
        }

        // ===== 6) ASM =====
        std::cout << "\n=== ASM ===\n";
        AssemblyGenerator asmGen(gen.getCode());
        std::cout << asmGen.generate();

        std::cout << "\n编译成功。\n";

        if (runVm) {
            std::cout << "\n=== RUN (TAC VM) ===\n";
            TacVM vm(gen.getCode());
            double ret = 0.0;
            bool hasRet = vm.run(ret);
            if (hasRet) {
                std::cout << "(return) " << ret << "\n";
            }
        }

    } catch (const std::exception &ex) {
        std::cerr << "编译失败: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
