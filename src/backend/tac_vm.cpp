#include "backend/tac_vm.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

TacVM::TacVM(const std::vector<Tac>& code, const std::unordered_map<std::string, std::string>& strTab)
        : code(code), stringTable(strTab) {
        for (size_t i = 0; i < code.size(); ++i) {
            if (code[i].op == "label") {
                labels[code[i].result] = static_cast<int>(i);
            }
        }
    }

bool TacVM::run(double& retVal) {
        int pc = 0;
        while (pc >= 0 && pc < static_cast<int>(code.size())) {
            const auto& ins = code[pc];
            if (ins.op == "label") { pc++; continue; }
            if (ins.op == "goto") { pc = jumpTo(ins.result); continue; }
            if (ins.op == "iffalse") {
                double c = valueOf(ins.arg1);
                if (c == 0.0) pc = jumpTo(ins.result);
                else pc++;
                continue;
            }
            if (ins.op == "iftrue") {
                double c = valueOf(ins.arg1);
                if (c != 0.0) pc = jumpTo(ins.result);
                else pc++;
                continue;
            }
            if (ins.op == "return") {
                retVal = ins.arg1.empty() ? 0.0 : valueOf(ins.arg1);
                return true;
            }
            if (ins.op == "print") {
                if (ins.arg1.size() > 0 && ins.arg1[0] == 's' && stringTable.count(ins.arg1)) {
                    std::cout << stringTable.at(ins.arg1) << "\n";
                } else {
                    std::cout << valueOf(ins.arg1) << "\n";
                }
                pc++;
                continue;
            }
            if (ins.op == "param" || ins.op == "call") {
                pc++;
                continue;
            }

            if (ins.op == "=") {
                setVar(ins.result, valueOf(ins.arg1));
                pc++;
                continue;
            }
            if (ins.op == "!") {
                setVar(ins.result, valueOf(ins.arg1) == 0.0 ? 1.0 : 0.0);
                pc++;
                continue;
            }
            if (ins.op == "cast") {
                double v = valueOf(ins.arg1);
                if (ins.arg2 == "int") v = static_cast<int>(v);
                else if (ins.arg2 == "float") v = static_cast<double>(v);
                else if (ins.arg2 == "bool") v = (v == 0.0 ? 0.0 : 1.0);
                else if (ins.arg2 == "char") v = static_cast<int>(v) & 0xFF;
                setVar(ins.result, v);
                pc++;
                continue;
            }

            double a = valueOf(ins.arg1);
            double b = valueOf(ins.arg2);
            if (ins.op == "+") setVar(ins.result, a + b);
            else if (ins.op == "-") setVar(ins.result, a - b);
            else if (ins.op == "*") setVar(ins.result, a * b);
            else if (ins.op == "/") setVar(ins.result, a / b);
            else if (ins.op == "%") setVar(ins.result, std::fmod(a, b));
            else if (ins.op == "==") setVar(ins.result, (a == b) ? 1.0 : 0.0);
            else if (ins.op == "!=") setVar(ins.result, (a != b) ? 1.0 : 0.0);
            else if (ins.op == "<") setVar(ins.result, (a < b) ? 1.0 : 0.0);
            else if (ins.op == "<=") setVar(ins.result, (a <= b) ? 1.0 : 0.0);
            else if (ins.op == ">") setVar(ins.result, (a > b) ? 1.0 : 0.0);
            else if (ins.op == ">=") setVar(ins.result, (a >= b) ? 1.0 : 0.0);
            else if (ins.op == "&&") setVar(ins.result, (a != 0.0 && b != 0.0) ? 1.0 : 0.0);
            else if (ins.op == "||") setVar(ins.result, (a != 0.0 || b != 0.0) ? 1.0 : 0.0);
            else {
                throw std::runtime_error("VM 不支持的 op: " + ins.op);
            }
            pc++;
        }
        return false;
    }

bool TacVM::isNumber(const std::string& s) const {
        if (s.empty()) return false;
        char* end = nullptr;
        std::strtod(s.c_str(), &end);
        return end && *end == '\0';
    }

double TacVM::valueOf(const std::string& x) {
        if (x.empty()) return 0.0;
        if (x.size() > 0 && x[0] == 's' && stringTable.count(x)) {
            return 1.0;
        }
        if (isNumber(x)) return std::stod(x);
        auto it = vars.find(x);
        if (it == vars.end()) return 0.0;
        return it->second;
    }

void TacVM::setVar(const std::string& name, double value) {
        vars[name] = value;
    }

int TacVM::jumpTo(const std::string& label) {
        auto it = labels.find(label);
        if (it == labels.end()) {
            throw std::runtime_error("未定义的标签: " + label);
        }
        return it->second;
    }
