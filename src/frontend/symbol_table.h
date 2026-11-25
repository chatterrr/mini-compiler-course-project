#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.h"

// 符号表：支持嵌套作用域、重复声明检查、常量标记
class SymbolTable {
  public:
    struct Entry {
        std::string name;
        ValueType type;
        int scopeDepth;
        int declLine;
        int declCol;
        bool isConst;
    };

    SymbolTable();

    void pushScope();
    void popScope();

    int currentDepth() const;

    void declare(const std::string& name, ValueType type, int line, int col, bool isConst = false);

    bool isConst(const std::string& name, int line, int col) const;

    ValueType lookup(const std::string& name, int line, int col) const;

    const std::vector<Entry>& getHistory() const { return history; }

  private:
    struct SymbolInfo {
        ValueType type;
        bool isConst;
    };
    std::vector<std::unordered_map<std::string, SymbolInfo>> scopes;
    std::vector<Entry> history;
};
