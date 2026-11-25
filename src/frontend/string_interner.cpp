#include "frontend/string_interner.h"

std::string StringInterner::intern(const std::string& value) {
    std::string id = "s" + std::to_string(index++);
    table[id] = value;
    return id;
}
