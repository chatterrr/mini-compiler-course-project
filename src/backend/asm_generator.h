#pragma once

#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ir/tac.h"

// 汇编代码生成器：TAC -> 类 MIPS 伪汇编
class AssemblyGenerator {
  public:
    explicit AssemblyGenerator(const std::vector<Tac>& code);

    std::string generate();

  private:
    std::string escapeString(const std::string& s);

    static bool isNumber(const std::string& s);
    static bool isTemp(const std::string& s);
    static bool isLabel(const std::string& s);
    static bool isStringId(const std::string& s);
    static bool isIdentLike(const std::string& s);

    void collectSymbols();
    std::string ensureDest(const std::string& x) const;
    std::string ensureReg(const std::string& op, std::ostringstream& out, const std::string& scratchPref);
    std::string ensureDestReg(const std::string& dst, std::ostringstream& out, const std::string& scratchPref);
    void storeDest(const std::string& dst, const std::string& reg, std::ostringstream& out);
    void emitMoveToDest(const std::string& dst, const std::string& srcReg, std::ostringstream& out);

    const std::vector<Tac>& tac;
    std::unordered_set<std::string> variables;
    std::unordered_set<std::string> spills;
    std::unordered_map<std::string, std::string> tempReg;
    std::unordered_map<std::string, std::string> stringTable;
    int nextTempReg = 0;
    int labelCounter = 0;
};
