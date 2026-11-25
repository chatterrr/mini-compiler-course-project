#include "ir/ir_generator.h"

#include <stdexcept>

IrGenerator::IrGenerator(const std::unordered_map<std::string, std::string>& stringTable)
    : stringTable(stringTable) {}

TacProgram IrGenerator::generate(const std::shared_ptr<ASTNode>& root) {
    genNode(root);

    TacProgram program;
    program.code = builder.getCode();
    program.stringTable = stringTable;
    return program;
}

std::string IrGenerator::typeName(ValueType t) const {
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

std::string IrGenerator::genRaw(const std::shared_ptr<ASTNode>& node) {
    if (node && node->type == ASTNodeType::BinaryOp && node->value == "cast" && !node->children.empty()) {
        return genNode(node->children[0]);
    }
    return genNode(node);
}

std::string IrGenerator::applyCast(const std::shared_ptr<ASTNode>& node, const std::string& rawPlace) {
    if (node && node->type == ASTNodeType::BinaryOp && node->value == "cast") {
        std::string temp = builder.newTemp(node->dataType);
        builder.emit("cast", rawPlace, typeName(node->dataType), temp);
        return temp;
    }
    return rawPlace;
}

std::string IrGenerator::genAssignment(const std::shared_ptr<ASTNode>& node) {
    auto lhs = node->children[0];
    auto rhs = node->children[1];

    if (node->value == "=") {
        std::string rhsPlace = genNode(rhs);
        if (rhs->dataType != lhs->dataType) {
            std::string casted = builder.newTemp(lhs->dataType);
            builder.emit("cast", rhsPlace, typeName(lhs->dataType), casted);
            rhsPlace = casted;
        }
        builder.emit("=", rhsPlace, "", lhs->value);
        return rhsPlace;
    }

    // 复合赋值：Assignment(op, [lhs, BinaryOp(baseOp, [lhsCopy, rhs])])
    auto binop = rhs;
    std::string rhsRaw = genRaw(binop->children[1]);
    std::string tempLeft = builder.newTemp(lhs->dataType);
    builder.emit("=", lhs->value, "", tempLeft);
    std::string rhsPlace = applyCast(binop->children[1], rhsRaw);
    std::string tempResult = builder.newTemp(lhs->dataType);
    builder.emit(binop->value, tempLeft, rhsPlace, tempResult);
    builder.emit("=", tempResult, "", lhs->value);
    return tempResult;
}

std::string IrGenerator::genBinaryOp(const std::shared_ptr<ASTNode>& node) {
    const std::string& op = node->value;

    if (op == "cast") {
        std::string src = genNode(node->children[0]);
        std::string temp = builder.newTemp(node->dataType);
        builder.emit("cast", src, typeName(node->dataType), temp);
        return temp;
    }

    if (op == "&&") {
        std::string left = genNode(node->children[0]);
        std::string right = genNode(node->children[1]);
        std::string result = builder.newTemp(ValueType::Bool);
        std::string endLabel = builder.newLabel();
        builder.emit("=", "0", "", result);
        builder.emit("iffalse", left, "", endLabel);
        builder.emit("iffalse", right, "", endLabel);
        builder.emit("=", "1", "", result);
        builder.emit("label", "", "", endLabel);
        return result;
    }

    if (op == "||") {
        std::string left = genNode(node->children[0]);
        std::string right = genNode(node->children[1]);
        std::string result = builder.newTemp(ValueType::Bool);
        std::string evalRight = builder.newLabel();
        std::string endLabel = builder.newLabel();
        std::string setTrue = builder.newLabel();
        builder.emit("=", "0", "", result);
        builder.emit("iffalse", left, "", evalRight);
        builder.emit("goto", "", "", setTrue);
        builder.emit("label", "", "", evalRight);
        builder.emit("iffalse", right, "", endLabel);
        builder.emit("label", "", "", setTrue);
        builder.emit("=", "1", "", result);
        builder.emit("label", "", "", endLabel);
        return result;
    }

    if (op == "+" || op == "-" || op == "*" || op == "/" || op == "%") {
        // 先解析左右操作数，再统一插入转换，匹配单遍解析时的指令顺序。
        std::string leftRaw = genRaw(node->children[0]);
        std::string rightRaw = genRaw(node->children[1]);
        std::string left = applyCast(node->children[0], leftRaw);
        std::string right = applyCast(node->children[1], rightRaw);
        std::string result = builder.newTemp(node->dataType);
        builder.emit(op, left, right, result);
        return result;
    }

    // 关系 / 相等
    std::string left = genNode(node->children[0]);
    std::string right = genNode(node->children[1]);
    std::string result = builder.newTemp(ValueType::Bool);
    builder.emit(op, left, right, result);
    return result;
}

std::string IrGenerator::genUnaryOp(const std::shared_ptr<ASTNode>& node) {
    const std::string& op = node->value;
    auto operand = node->children[0];

    if (op == "++pre" || op == "--pre") {
        std::string place = genNode(operand);
        std::string temp = builder.newTemp(node->dataType);
        std::string one = (node->dataType == ValueType::Float) ? "1.0" : "1";
        builder.emit(op == "++pre" ? "+" : "-", place, one, temp);
        builder.emit("=", temp, "", operand->value);
        return temp;
    }

    if (op == "++post" || op == "--post") {
        std::string place = genNode(operand);
        std::string resultTemp = builder.newTemp(node->dataType);
        builder.emit("=", place, "", resultTemp);
        std::string one = (node->dataType == ValueType::Float) ? "1.0" : "1";
        std::string newTemp = builder.newTemp(node->dataType);
        builder.emit(op == "++post" ? "+" : "-", place, one, newTemp);
        builder.emit("=", newTemp, "", operand->value);
        return resultTemp;
    }

    if (op == "-") {
        std::string place = genNode(operand);
        std::string zero = (node->dataType == ValueType::Float) ? "0.0" : "0";
        std::string temp = builder.newTemp(node->dataType);
        builder.emit("-", zero, place, temp);
        return temp;
    }

    if (op == "!") {
        std::string place = genNode(operand);
        std::string temp = builder.newTemp(ValueType::Bool);
        builder.emit("!", place, "", temp);
        return temp;
    }

    return genNode(operand);
}

std::string IrGenerator::genTernaryOp(const std::shared_ptr<ASTNode>& node) {
    std::string cond = genNode(node->children[0]);
    std::string trueRaw = genRaw(node->children[1]);
    std::string falseRaw = genRaw(node->children[2]);

    std::string temp = builder.newTemp(node->dataType);
    std::string falseLabel = builder.newLabel();
    std::string endLabel = builder.newLabel();

    builder.emit("iffalse", cond, "", falseLabel);
    std::string truePlace = applyCast(node->children[1], trueRaw);
    builder.emit("=", truePlace, "", temp);
    builder.emit("goto", "", "", endLabel);
    builder.emit("label", "", "", falseLabel);
    std::string falsePlace = applyCast(node->children[2], falseRaw);
    builder.emit("=", falsePlace, "", temp);
    builder.emit("label", "", "", endLabel);
    return temp;
}

std::string IrGenerator::genCall(const std::shared_ptr<ASTNode>& node) {
    std::vector<std::string> args;
    for (const auto& arg : node->children) {
        args.push_back(genNode(arg));
    }
    for (const auto& place : args) {
        builder.emit("param", place, "", "");
    }

    if (node->type == ASTNodeType::Print) {
        for (const auto& place : args) {
            builder.emit("print", place, "", "");
        }
        return "0";
    }

    std::string ret = builder.newTemp(ValueType::Int);
    builder.emit("call", node->value, std::to_string(args.size()), ret);
    return ret;
}

std::string IrGenerator::genSwitch(const std::shared_ptr<ASTNode>& node) {
    std::string exprPlace = genNode(node->children[0]);
    std::string endLabel = builder.newLabel();
    std::string switchTemp = builder.newTemp(node->children[0]->dataType);
    builder.emit("=", exprPlace, "", switchTemp);

    std::string defaultLabel = builder.newLabel();
    bool hasDefault = false;

    struct CaseInfo {
        const ASTNode* node;
        std::string value;
        std::string label;
    };
    std::vector<CaseInfo> cases;

    for (size_t i = 1; i < node->children.size(); ++i) {
        const auto& child = node->children[i];
        if (!child) continue;
        if (child->type == ASTNodeType::Case) {
            std::string value = genNode(child->children[0]);
            std::string label = builder.newLabel();
            cases.push_back({child.get(), value, label});
        } else if (child->type == ASTNodeType::Default) {
            hasDefault = true;
        }
    }

    for (const auto& ci : cases) {
        std::string cmpTemp = builder.newTemp(ValueType::Bool);
        builder.emit("==", switchTemp, ci.value, cmpTemp);
        builder.emit("iftrue", cmpTemp, "", ci.label);
    }

    builder.emit("goto", "", "", defaultLabel);
    breakStack.push_back(endLabel);

    for (size_t i = 1; i < node->children.size(); ++i) {
        const auto& child = node->children[i];
        if (!child) continue;
        if (child->type == ASTNodeType::Case) {
            std::string label;
            for (const auto& ci : cases) {
                if (ci.node == child.get()) {
                    label = ci.label;
                    break;
                }
            }
            builder.emit("label", "", "", label);
            for (size_t j = 1; j < child->children.size(); ++j) {
                genNode(child->children[j]);
            }
        } else if (child->type == ASTNodeType::Default) {
            builder.emit("label", "", "", defaultLabel);
            for (const auto& stmt : child->children) genNode(stmt);
        } else {
            genNode(child);
        }
    }

    breakStack.pop_back();

    if (!hasDefault) {
        builder.emit("label", "", "", defaultLabel);
    }
    builder.emit("label", "", "", endLabel);
    return "";
}

std::string IrGenerator::genNode(const std::shared_ptr<ASTNode>& node) {
    if (!node) return "";

    switch (node->type) {
    case ASTNodeType::Program:
    case ASTNodeType::Block:
        for (const auto& child : node->children) genNode(child);
        return "";

    case ASTNodeType::Declaration:
        if (node->children.size() > 1 && node->children[1]) {
            genNode(node->children[1]);
        }
        return "";

    case ASTNodeType::ConstDeclaration: {
        if (node->children.size() > 1 && node->children[1]) {
            auto init = node->children[1];
            std::string place = genNode(init);
            if (init->dataType != node->dataType) {
                std::string casted = builder.newTemp(node->dataType);
                builder.emit("cast", place, typeName(node->dataType), casted);
                place = casted;
            }
            builder.emit("=", place, "", node->value);
            return place;
        }
        return "";
    }

    case ASTNodeType::Assignment:
        return genAssignment(node);

    case ASTNodeType::BinaryOp:
        return genBinaryOp(node);

    case ASTNodeType::UnaryOp:
        return genUnaryOp(node);

    case ASTNodeType::TernaryOp:
        return genTernaryOp(node);

    case ASTNodeType::Variable:
    case ASTNodeType::Literal:
        return node->value;

    case ASTNodeType::FunctionCall:
    case ASTNodeType::Print:
        return genCall(node);

    case ASTNodeType::If: {
        std::string cond = genNode(node->children[0]);
        std::string elseLabel = builder.newLabel();
        std::string endLabel = builder.newLabel();
        builder.emit("iffalse", cond, "", elseLabel);
        genNode(node->children[1]);
        builder.emit("goto", "", "", endLabel);
        builder.emit("label", "", "", elseLabel);
        if (node->children.size() > 2) genNode(node->children[2]);
        builder.emit("label", "", "", endLabel);
        return "";
    }

    case ASTNodeType::While: {
        std::string startLabel = builder.newLabel();
        std::string endLabel = builder.newLabel();
        std::string continueLabel = builder.newLabel();
        breakStack.push_back(endLabel);
        continueStack.push_back(continueLabel);

        builder.emit("label", "", "", startLabel);
        std::string cond = genNode(node->children[0]);
        builder.emit("iffalse", cond, "", endLabel);
        genNode(node->children[1]);
        builder.emit("label", "", "", continueLabel);
        builder.emit("goto", "", "", startLabel);
        builder.emit("label", "", "", endLabel);

        breakStack.pop_back();
        continueStack.pop_back();
        return "";
    }

    case ASTNodeType::DoWhile: {
        std::string startLabel = builder.newLabel();
        std::string endLabel = builder.newLabel();
        std::string continueLabel = builder.newLabel();
        breakStack.push_back(endLabel);
        continueStack.push_back(continueLabel);

        builder.emit("label", "", "", startLabel);
        genNode(node->children[0]);
        builder.emit("label", "", "", continueLabel);
        std::string cond = genNode(node->children[1]);
        builder.emit("iftrue", cond, "", startLabel);
        builder.emit("label", "", "", endLabel);

        breakStack.pop_back();
        continueStack.pop_back();
        return "";
    }

    case ASTNodeType::For: {
        auto init = node->children.size() > 0 ? node->children[0] : nullptr;
        auto condNode = node->children.size() > 1 ? node->children[1] : nullptr;
        auto body = node->children.size() > 2 ? node->children[2] : nullptr;
        auto inc = node->children.size() > 3 ? node->children[3] : nullptr;

        if (init) genNode(init);

        std::string condLabel = builder.newLabel();
        std::string endLabel = builder.newLabel();
        std::string bodyLabel = builder.newLabel();
        std::string continueLabel = builder.newLabel();
        breakStack.push_back(endLabel);
        continueStack.push_back(continueLabel);

        builder.emit("label", "", "", condLabel);
        std::string cond = condNode ? genNode(condNode) : "1";
        builder.emit("iffalse", cond, "", endLabel);
        builder.emit("goto", "", "", bodyLabel);
        builder.emit("label", "", "", bodyLabel);
        if (body) genNode(body);
        builder.emit("label", "", "", continueLabel);
        if (inc) genNode(inc);
        builder.emit("goto", "", "", condLabel);
        builder.emit("label", "", "", endLabel);

        breakStack.pop_back();
        continueStack.pop_back();
        return "";
    }

    case ASTNodeType::Switch:
        return genSwitch(node);

    case ASTNodeType::Return:
        if (!node->children.empty()) {
            std::string place = genNode(node->children[0]);
            builder.emit("return", place, "", "");
        } else {
            builder.emit("return", "", "", "");
        }
        return "";

    case ASTNodeType::Break:
        if (!breakStack.empty()) builder.emit("goto", "", "", breakStack.back());
        return "";

    case ASTNodeType::Continue:
        if (!continueStack.empty()) builder.emit("goto", "", "", continueStack.back());
        return "";

    case ASTNodeType::Case:
    case ASTNodeType::Default:
        for (const auto& child : node->children) genNode(child);
        return "";
    }
    return "";
}
