#include "frontend/semantic_analyzer.h"

#include <sstream>
#include <stdexcept>

SemanticAnalyzer::SemanticAnalyzer(SymbolTable& symbols) : symbols(symbols) {}

void SemanticAnalyzer::analyze(const std::shared_ptr<ASTNode>& root) { analyzeNode(root); }

void SemanticAnalyzer::analyzeNode(const std::shared_ptr<ASTNode>& node) {
    if (!node) return;

    switch (node->type) {
    case ASTNodeType::Program:
        for (const auto& child : node->children) analyzeNode(child);
        break;

    case ASTNodeType::Block:
        symbols.pushScope();
        for (const auto& child : node->children) analyzeNode(child);
        symbols.popScope();
        break;

    case ASTNodeType::Declaration:
        analyzeDeclaration(node);
        break;

    case ASTNodeType::ConstDeclaration:
        analyzeConstDeclaration(node);
        break;

    case ASTNodeType::Assignment:
        analyzeAssignment(node);
        break;

    case ASTNodeType::BinaryOp:
        analyzeBinaryOp(node);
        break;

    case ASTNodeType::UnaryOp:
        analyzeUnaryOp(node);
        break;

    case ASTNodeType::TernaryOp:
        analyzeTernaryOp(node);
        break;

    case ASTNodeType::For:
        analyzeFor(node);
        break;

    case ASTNodeType::If:
        analyzeNode(node->children[0]);
        ensureBool(node->children[0]);
        analyzeNode(node->children[1]);
        if (node->children.size() > 2) analyzeNode(node->children[2]);
        break;

    case ASTNodeType::While:
        analyzeNode(node->children[0]);
        ensureBool(node->children[0]);
        analyzeNode(node->children[1]);
        break;

    case ASTNodeType::DoWhile:
        analyzeNode(node->children[0]);
        analyzeNode(node->children[1]);
        ensureBool(node->children[1]);
        break;

    case ASTNodeType::Switch:
        analyzeNode(node->children[0]);
        for (size_t i = 1; i < node->children.size(); ++i) analyzeNode(node->children[i]);
        break;

    case ASTNodeType::Case:
    case ASTNodeType::Default:
        for (const auto& child : node->children) analyzeNode(child);
        break;

    case ASTNodeType::Return:
        if (!node->children.empty()) analyzeNode(node->children[0]);
        break;

    case ASTNodeType::Print:
        if (node->children.empty()) {
            throw std::runtime_error(errorAt(*node, "print 至少需要 1 个参数，发现 'print'"));
        }
        for (const auto& child : node->children) analyzeNode(child);
        node->dataType = ValueType::Void;
        break;

    case ASTNodeType::FunctionCall:
        for (const auto& child : node->children) analyzeNode(child);
        node->dataType = ValueType::Int;
        break;

    case ASTNodeType::Variable:
        node->dataType = symbols.lookup(node->value, node->line, node->column);
        break;

    case ASTNodeType::Literal:
    case ASTNodeType::Break:
    case ASTNodeType::Continue:
        break;
    }
}

void SemanticAnalyzer::analyzeDeclaration(const std::shared_ptr<ASTNode>& node) {
    ValueType type = node->dataType;
    symbols.declare(node->value, type, node->line, node->column, false);

    if (!node->children.empty()) node->children[0]->dataType = type;
    if (node->children.size() > 1 && node->children[1]) {
        auto init = node->children[1];
        if (init->type == ASTNodeType::Assignment && init->children.size() >= 2) {
            // 声明初始化在 AST 中保留原始表达式，转换指令由 IR 阶段生成。
            analyzeNode(init->children[0]);
            analyzeNode(init->children[1]);
            init->dataType = type;
            castExpr(init->children[1], type); // 仅校验转换是否合法
        } else {
            analyzeNode(init);
        }
    }
}

void SemanticAnalyzer::analyzeConstDeclaration(const std::shared_ptr<ASTNode>& node) {
    ValueType type = node->dataType;
    symbols.declare(node->value, type, node->line, node->column, true);

    if (!node->children.empty()) node->children[0]->dataType = type;
    if (node->children.size() > 1 && node->children[1]) {
        analyzeNode(node->children[1]);
        // 同样保留原始表达式，转换在 IR 阶段按声明类型生成。
        castExpr(node->children[1], type);
    }
}

void SemanticAnalyzer::analyzeAssignment(const std::shared_ptr<ASTNode>& node) {
    if (node->children.size() < 2) return;

    auto lhs = node->children[0];
    auto rhs = node->children[1];

    if (lhs->type != ASTNodeType::Variable) {
        throw std::runtime_error(errorAt(*node, "赋值运算符左边必须是变量"));
    }

    analyzeNode(lhs);
    ValueType varType = lhs->dataType;

    if (symbols.isConst(lhs->value, node->line, node->column)) {
        throw std::runtime_error("不能给常量 '" + lhs->value + "' 赋值");
    }

    if (node->value == "=") {
        analyzeNode(rhs);
        node->children[1] = castExpr(rhs, varType);
        node->dataType = varType;
        return;
    }

    // 复合赋值：AST 形状为 Assignment(op, [lhs, BinaryOp(baseOp, [lhsCopy, rhs])])
    if (rhs->type != ASTNodeType::BinaryOp || rhs->children.size() < 2) return;
    analyzeNode(rhs->children[0]);
    analyzeNode(rhs->children[1]);
    rhs->children[1] = castExpr(rhs->children[1], varType);
    rhs->dataType = varType;
    node->dataType = varType;
}

void SemanticAnalyzer::analyzeBinaryOp(const std::shared_ptr<ASTNode>& node) {
    const std::string& op = node->value;

    if (op == "cast") {
        if (!node->children.empty()) analyzeNode(node->children[0]);
        return;
    }

    auto left = node->children[0];
    auto right = node->children[1];

    if (op == "&&" || op == "||") {
        analyzeNode(left);
        analyzeNode(right);
        ensureBool(left);
        ensureBool(right);
        node->dataType = ValueType::Bool;
        return;
    }

    if (op == "+" || op == "-" || op == "*" || op == "/" || op == "%") {
        analyzeNode(left);
        analyzeNode(right);
        requireNumericBinary(*node, *left, *right);
        if (op == "%") requireIntegralBinary(*node, *left, *right);

        ValueType resType = promote(left->dataType, right->dataType);
        node->children[0] = castExpr(left, resType);
        node->children[1] = castExpr(right, resType);
        node->dataType = resType;
        return;
    }

    // 关系/相等运算
    analyzeNode(left);
    analyzeNode(right);
    node->dataType = ValueType::Bool;
}

void SemanticAnalyzer::analyzeUnaryOp(const std::shared_ptr<ASTNode>& node) {
    const std::string& op = node->value;
    auto operand = node->children[0];

    if (op == "++pre" || op == "--pre" || op == "++post" || op == "--post") {
        if (operand->type != ASTNodeType::Variable) {
            bool isPrefix = (op == "++pre" || op == "--pre");
            throw std::runtime_error(errorAt(
                *node, isPrefix ? "前缀自增/自减运算符只能用于变量" : "自增/自减运算符只能用于变量"));
        }

        analyzeNode(operand);
        ValueType type = operand->dataType;
        if (!isNumeric(type)) {
            throw std::runtime_error(
                errorAt(*node, "自增/自减运算符只能用于数值类型变量，当前为 " + typeName(type)));
        }
        if (symbols.isConst(operand->value, node->line, node->column)) {
            throw std::runtime_error("不能修改常量 '" + operand->value + "'");
        }
        node->dataType = type;
        return;
    }

    if (op == "-") {
        analyzeNode(operand);
        node->dataType = operand->dataType;
        return;
    }

    if (op == "!") {
        analyzeNode(operand);
        node->dataType = ValueType::Bool;
        return;
    }

    analyzeNode(operand);
}

void SemanticAnalyzer::analyzeTernaryOp(const std::shared_ptr<ASTNode>& node) {
    auto cond = node->children[0];
    auto trueExpr = node->children[1];
    auto falseExpr = node->children[2];

    analyzeNode(cond);
    ensureBool(cond);
    analyzeNode(trueExpr);
    analyzeNode(falseExpr);

    ValueType resultType = promote(trueExpr->dataType, falseExpr->dataType);
    node->children[1] = castExpr(trueExpr, resultType);
    node->children[2] = castExpr(falseExpr, resultType);
    node->dataType = resultType;
}

void SemanticAnalyzer::analyzeFor(const std::shared_ptr<ASTNode>& node) {
    // 固定槽位：[init, cond, body, inc]
    if (node->children.size() > 0 && node->children[0]) analyzeNode(node->children[0]);
    if (node->children.size() > 1 && node->children[1]) {
        analyzeNode(node->children[1]);
        ensureBool(node->children[1]);
    }
    if (node->children.size() > 2 && node->children[2]) analyzeNode(node->children[2]);
    if (node->children.size() > 3 && node->children[3]) analyzeNode(node->children[3]);
}

std::shared_ptr<ASTNode> SemanticAnalyzer::castExpr(const std::shared_ptr<ASTNode>& expr, ValueType target) {
    if (!expr) return expr;
    if (expr->dataType == target) return expr;
    if (expr->dataType == ValueType::String || target == ValueType::String) {
        throw std::runtime_error("不支持字符串类型的转换");
    }

    auto castNode = std::make_shared<ASTNode>(ASTNodeType::BinaryOp, "cast", target, expr->line, expr->column);
    castNode->children.push_back(expr);
    castNode->children.push_back(
        std::make_shared<ASTNode>(ASTNodeType::Literal, typeName(target), target, expr->line, expr->column));
    return castNode;
}

std::shared_ptr<ASTNode> SemanticAnalyzer::ensureBool(const std::shared_ptr<ASTNode>& expr) {
    if (expr->dataType == ValueType::Bool) return expr;
    if (expr->dataType == ValueType::Void) {
        throw std::runtime_error(errorAt(*expr, "void 类型不能作为条件表达式"));
    }
    throw std::runtime_error(
        errorAt(*expr, "条件表达式类型错误：必须为 bool，当前为 " + typeName(expr->dataType)));
}

bool SemanticAnalyzer::isNumeric(ValueType t) const {
    return t == ValueType::Int || t == ValueType::Float || t == ValueType::Char;
}

bool SemanticAnalyzer::isIntegral(ValueType t) const {
    return t == ValueType::Int || t == ValueType::Char;
}

ValueType SemanticAnalyzer::promote(ValueType a, ValueType b) {
    if (a == ValueType::String || b == ValueType::String) {
        throw std::runtime_error("字符串类型不支持算术运算");
    }
    if (a == ValueType::Float || b == ValueType::Float) return ValueType::Float;
    if (a == ValueType::Int || b == ValueType::Int) return ValueType::Int;
    if (a == ValueType::Char || b == ValueType::Char) return ValueType::Char;
    return ValueType::Bool;
}

std::string SemanticAnalyzer::typeName(ValueType t) const {
    switch (t) {
    case ValueType::Int: return "int";
    case ValueType::Float: return "float";
    case ValueType::Bool: return "bool";
    case ValueType::Char: return "char";
    case ValueType::String: return "string";
    case ValueType::Void:
    default: return "void";
    }
}

std::string SemanticAnalyzer::errorAt(const ASTNode& node, const std::string& msg) const {
    std::ostringstream oss;
    oss << "语义错误 (line " << node.line << ", col " << node.column << "): " << msg;
    return oss.str();
}

void SemanticAnalyzer::requireNumericBinary(const ASTNode& node, const ASTNode& l, const ASTNode& r) const {
    if (!isNumeric(l.dataType) || !isNumeric(r.dataType)) {
        throw std::runtime_error(errorAt(
            node, "算术运算符 '" + node.value + "' 的操作数类型非法: " + typeName(l.dataType) + " 与 " +
                      typeName(r.dataType)));
    }
}

void SemanticAnalyzer::requireIntegralBinary(const ASTNode& node, const ASTNode& l, const ASTNode& r) const {
    if (!isIntegral(l.dataType) || !isIntegral(r.dataType)) {
        throw std::runtime_error(errorAt(
            node, "取模运算符 '%' 仅支持整型/字符型操作数: " + typeName(l.dataType) + " 与 " +
                      typeName(r.dataType)));
    }
}
