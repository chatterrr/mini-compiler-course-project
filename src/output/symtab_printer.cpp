#include "output/symtab_printer.h"

#include <algorithm>
#include <ostream>

void printSymbolTable(const SymbolTable& symbols, std::ostream& os) {
    auto typeStr = [](ValueType t) {
        switch (t) {
        case ValueType::Int: return std::string("int");
        case ValueType::Float: return std::string("float");
        case ValueType::Bool: return std::string("bool");
        case ValueType::Char: return std::string("char");
        case ValueType::String: return std::string("string");
        case ValueType::Void: return std::string("void");
        }
        return std::string("unknown");
    };

    os << "=== SYMTAB ===\n";
    const auto& history = symbols.getHistory();
    if (history.empty()) {
        os << "(空符号表)\n";
        return;
    }

    int maxDepth = 0;
    for (const auto& e : history) maxDepth = std::max(maxDepth, e.scopeDepth);

    for (int d = 0; d <= maxDepth; ++d) {
        bool any = false;
        for (const auto& e : history) {
            if (e.scopeDepth != d) continue;
            if (!any) {
                os << "[scope " << d << "]\n";
                any = true;
            }
            os << "  " << e.name << "\t" << typeStr(e.type)
               << (e.isConst ? " (const)" : "")
               << "\t@(" << e.declLine << ":" << e.declCol << ")\n";
        }
    }
}
