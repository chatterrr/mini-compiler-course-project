#include "frontend/symbol_table.h"

#include <stdexcept>

SymbolTable::SymbolTable() { pushScope(); }

void SymbolTable::pushScope() { scopes.emplace_back(); }

void SymbolTable::popScope() {
    if (scopes.size() <= 1) return;
    scopes.pop_back();
}

int SymbolTable::currentDepth() const { return static_cast<int>(scopes.size()) - 1; }

void SymbolTable::declare(const std::string& name, ValueType type, int line, int col, bool isConst) {
    auto& currentScope = scopes.back();
    if (currentScope.count(name)) {
        throw std::runtime_error("重复声明标识符 '" + name + "'，位置: line " + std::to_string(line) + " col " +
                                 std::to_string(col));
    }
    currentScope[name] = {type, isConst};
    history.push_back(Entry{name, type, currentDepth(), line, col, isConst});
}

bool SymbolTable::isConst(const std::string& name, int line, int col) const {
    for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
        auto found = it->find(name);
        if (found != it->end()) return found->second.isConst;
    }
    throw std::runtime_error("未声明的标识符 '" + name + "'，位置: line " + std::to_string(line) + " col " +
                             std::to_string(col));
}

ValueType SymbolTable::lookup(const std::string& name, int line, int col) const {
    for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
        auto found = it->find(name);
        if (found != it->end()) return found->second.type;
    }
    throw std::runtime_error("未声明的标识符 '" + name + "'，位置: line " + std::to_string(line) + " col " +
                             std::to_string(col));
}
