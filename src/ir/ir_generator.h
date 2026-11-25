#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "frontend/ast.h"
#include "ir/ir_builder.h"
#include "ir/tac.h"

// IR 生成：遍历带类型信息的 AST，生成三地址码
class IrGenerator {
  public:
    explicit IrGenerator(const std::unordered_map<std::string, std::string>& stringTable);

    TacProgram generate(const std::shared_ptr<ASTNode>& root);

  private:
    std::string genNode(const std::shared_ptr<ASTNode>& node);
    std::string genAssignment(const std::shared_ptr<ASTNode>& node);
    std::string genBinaryOp(const std::shared_ptr<ASTNode>& node);
    std::string genUnaryOp(const std::shared_ptr<ASTNode>& node);
    std::string genTernaryOp(const std::shared_ptr<ASTNode>& node);
    std::string genCall(const std::shared_ptr<ASTNode>& node);
    std::string genSwitch(const std::shared_ptr<ASTNode>& node);

    // 生成表达式本体，但若表达式外层是 cast 节点，则先不发射 cast。
    std::string genRaw(const std::shared_ptr<ASTNode>& node);
    // 若节点是 cast，则在当前位置发射 cast 指令。
    std::string applyCast(const std::shared_ptr<ASTNode>& node, const std::string& rawPlace);

    std::string typeName(ValueType t) const;

    TacBuilder builder;
    std::unordered_map<std::string, std::string> stringTable;
    std::vector<std::string> breakStack;
    std::vector<std::string> continueStack;
};
