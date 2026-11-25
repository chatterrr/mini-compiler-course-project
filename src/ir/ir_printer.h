#pragma once

#include <iosfwd>
#include <vector>

#include "ir/tac.h"

// 输出 TAC 指令序列（不输出 === IR === 标题，由调用方负责）
void printIR(const std::vector<Tac>& code, std::ostream& os);
