#pragma once

#include <memory>
#include <string>

#include "frontend/ast.h"

// 生成 AST 的 Graphviz DOT 描述
std::string astToDot(const std::shared_ptr<ASTNode>& root);
