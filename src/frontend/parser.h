#pragma once

#include <memory>
#include <string>
#include <vector>

#include "frontend/ast.h"
#include "frontend/string_interner.h"
#include "frontend/token.h"

// 语法分析器：只负责 Token -> AST，不进行符号检查、类型检查或 IR 生成
class Parser {
  public:
    explicit Parser(std::vector<Token> tokens);

    void parseProgram();

    std::shared_ptr<ASTNode> getASTRoot() const { return astRoot; }
    const StringInterner& getStrings() const { return *strings; }

  private:
    Parser(std::vector<Token> tokens, StringInterner& shared);

    // Token 辅助
    bool isAtEnd() const;
    const Token& peek() const;
    const Token& previous() const;
    bool check(TokenType type, const std::string& lexeme = "") const;
    const Token& advance();
    bool match(TokenType type, const std::string& lexeme = "");
    const Token& consume(TokenType type, const std::string& lexeme, const std::string& msg);
    const Token& consume(TokenType type, const std::string& msg);
    std::string errorAt(const Token& tok, const std::string& msg) const;

    // 语句
    ValueType parseType();
    std::shared_ptr<ASTNode> parseStatement();
    std::shared_ptr<ASTNode> parseDeclaration(ValueType type);
    std::shared_ptr<ASTNode> parseConstDeclaration(ValueType type);
    std::shared_ptr<ASTNode> parseIf();
    std::shared_ptr<ASTNode> parseWhile();
    std::shared_ptr<ASTNode> parseDoWhile();
    std::shared_ptr<ASTNode> parseSwitch();
    std::shared_ptr<ASTNode> parseBreak();
    std::shared_ptr<ASTNode> parseContinue();
    std::shared_ptr<ASTNode> parseFor();
    std::shared_ptr<ASTNode> parseBlock();
    std::shared_ptr<ASTNode> parseReturn();
    std::shared_ptr<ASTNode> parseExpressionStatement();

    // 表达式优先级层次（从低到高）：
    // assignment -> conditional -> logical_or -> logical_and -> equality
    // -> relational -> additive -> multiplicative -> unary -> postfix -> primary
    std::shared_ptr<ASTNode> assignment();
    std::shared_ptr<ASTNode> conditional();
    std::shared_ptr<ASTNode> logical_or();
    std::shared_ptr<ASTNode> logical_and();
    std::shared_ptr<ASTNode> equality();
    std::shared_ptr<ASTNode> relational();
    std::shared_ptr<ASTNode> additive();
    std::shared_ptr<ASTNode> multiplicative();
    std::shared_ptr<ASTNode> unary();
    std::shared_ptr<ASTNode> postfix();
    std::shared_ptr<ASTNode> primary();

    std::shared_ptr<ASTNode> makeVariable(const Token& tok, ValueType type = ValueType::Void);

    std::vector<Token> toks;
    size_t current = 0;
    std::shared_ptr<ASTNode> astRoot;

    // break/continue 的上下文检查
    int loopDepth = 0;
    int switchDepth = 0;

    StringInterner ownedStrings;
    StringInterner* strings;
};
