#pragma once

#include <string>
#include <unordered_map>

// 字符串常量池：为字符串字面量分配 s0、s1 ... 形式的编号
class StringInterner {
  public:
    std::string intern(const std::string& value);
    const std::unordered_map<std::string, std::string>& getTable() const { return table; }

  private:
    int index = 0;
    std::unordered_map<std::string, std::string> table;
};
