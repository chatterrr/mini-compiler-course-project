#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "ir/tac.h"

// TAC 发射器：负责临时量与标签的分配
class TacBuilder {
  public:
    std::string newTemp(ValueType type);
    std::string newLabel();
    void emit(const std::string& op, const std::string& a1 = "", const std::string& a2 = "",
              const std::string& res = "");

    const std::vector<Tac>& getCode() const { return code; }

  private:
    int tempIndex = 0;
    int labelIndex = 0;
    std::unordered_map<std::string, ValueType> tempType;
    std::vector<Tac> code;
};
