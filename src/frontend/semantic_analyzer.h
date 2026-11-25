#pragma once

#include <memory>
#include <string>

#include "common/types.h"
#include "frontend/ast.h"
#include "frontend/symbol_table.h"

// 语义分析：符号表维护、类型检查、隐式转换插入、AST 类型标注
class SemanticAnalyzer {
  public:
    explicit SemanticAnalyzer(SymbolTable& symbols);

    void analyze(const std::shared_ptr<ASTNode>& root);

  private:
    void analyzeNode(const std::shared_ptr<ASTNode>& node);
    void analyzeDeclaration(const std::shared_ptr<ASTNode>& node);
    void analyzeConstDeclaration(const std::shared_ptr<ASTNode>& node);
    void analyzeAssignment(const std::shared_ptr<ASTNode>& node);
    void analyzeBinaryOp(const std::shared_ptr<ASTNode>& node);
    void analyzeUnaryOp(const std::shared_ptr<ASTNode>& node);
    void analyzeTernaryOp(const std::shared_ptr<ASTNode>& node);
    void analyzeFor(const std::shared_ptr<ASTNode>& node);

    std::shared_ptr<ASTNode> castExpr(const std::shared_ptr<ASTNode>& expr, ValueType target);
    std::shared_ptr<ASTNode> ensureBool(const std::shared_ptr<ASTNode>& expr);

    bool isNumeric(ValueType t) const;
    bool isIntegral(ValueType t) const;
    ValueType promote(ValueType a, ValueType b);
    std::string typeName(ValueType t) const;
    std::string errorAt(const ASTNode& node, const std::string& msg) const;

    void requireNumericBinary(const ASTNode& node, const ASTNode& l, const ASTNode& r) const;
    void requireIntegralBinary(const ASTNode& node, const ASTNode& l, const ASTNode& r) const;

    SymbolTable& symbols;
};
