#include "output/ast_printer.h"

#include <ostream>

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

void printAST(const std::shared_ptr<ASTNode>& node, std::ostream& os, int indent) {
    if (!node) return;
    std::string indentStr(indent, ' ');
    os << indentStr << nodeTypeName(node->type) << " '" << node->value << "' "
       << valueTypeName(node->dataType) << " (" << node->line << ":" << node->column << ")";
    if (!node->children.empty()) {
        os << " {\n";
        for (const auto& child : node->children) {
            printAST(child, os, indent + 2);
        }
        os << indentStr << "}";
    }
    os << "\n";
}
