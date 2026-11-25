#include "ir/ir_builder.h"

std::string TacBuilder::newTemp(ValueType type) {
    std::string prefix;
    switch (type) {
    case ValueType::Int: prefix = "i"; break;
    case ValueType::Float: prefix = "f"; break;
    case ValueType::Bool: prefix = "b"; break;
    case ValueType::Char: prefix = "c"; break;
    case ValueType::String: prefix = "s"; break;
    default: prefix = "t"; break;
    }
    std::string name = prefix + std::to_string(tempIndex++);
    tempType[name] = type;
    return name;
}

std::string TacBuilder::newLabel() { return "L" + std::to_string(labelIndex++); }

void TacBuilder::emit(const std::string& op, const std::string& a1, const std::string& a2,
                      const std::string& res) {
    code.push_back({op, a1, a2, res});
}
