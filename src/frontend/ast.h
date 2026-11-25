#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/types.h"

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
