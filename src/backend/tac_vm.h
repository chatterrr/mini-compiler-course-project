#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "ir/tac.h"

// TAC 虚拟机：解释执行三地址码
class TacVM {
  public:
    explicit TacVM(const std::vector<Tac>& code,
                   const std::unordered_map<std::string, std::string>& strTab = {});

    bool run(double& retVal);

  private:
    bool isNumber(const std::string& s) const;
    double valueOf(const std::string& x);
    void setVar(const std::string& name, double value);
    int jumpTo(const std::string& label);

    const std::vector<Tac>& code;
    std::unordered_map<std::string, int> labels;
    std::unordered_map<std::string, double> vars;
    std::unordered_map<std::string, std::string> stringTable;
};
