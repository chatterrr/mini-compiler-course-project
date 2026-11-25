#pragma once

#include <iosfwd>

#include "frontend/symbol_table.h"

// 输出符号表内容（含旧的 === SYMTAB === 标题，保持输出格式不变）
void printSymbolTable(const SymbolTable& symbols, std::ostream& os);
