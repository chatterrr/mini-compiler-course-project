#pragma once

#include <iosfwd>
#include <memory>
#include <string>

#include "frontend/ast.h"

std::string nodeTypeName(ASTNodeType t);
std::string valueTypeName(ValueType t);

// 以缩进树形式输出 AST
void printAST(const std::shared_ptr<ASTNode>& node, std::ostream& os, int indent = 0);
