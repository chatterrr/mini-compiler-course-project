#pragma once

#include <string>
#include <unordered_map>
#include <vector>

// 三地址码结构
struct Tac {
    std::string op;
    std::string arg1;
    std::string arg2;
    std::string result;
};

// 一个完整的 TAC 程序：指令序列 + 字符串常量表
struct TacProgram {
    std::vector<Tac> code;
    std::unordered_map<std::string, std::string> stringTable;
};
