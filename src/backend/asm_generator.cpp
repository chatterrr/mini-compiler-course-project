#include "backend/asm_generator.h"
#include <cctype>
#include <cstdlib>
#include <sstream>

AssemblyGenerator::AssemblyGenerator(const std::vector<Tac>& code) : tac(code) {}

std::string AssemblyGenerator::generate() {
        std::ostringstream out;
        collectSymbols();
        out << "; === ASM (MIPS-like pseudo) ===\n";
        out << ".data\n";

        for (const auto& v : variables) out << v << ":\t.word 0\n";
        for (const auto& s : spills) out << s << ":\t.word 0\n";
        for (const auto& [strId, strVal] : stringTable) {
            out << strId << ":\t.asciiz \"" << escapeString(strVal) << "\"\n";
        }

        out << "\n.text\n";
        out << ".globl main\n";
        out << "main:\n";

        for (const auto& ins : tac) {
            if (ins.op == "label") {
                out << ins.result << ":\n";
                continue;
            }

            if (ins.op == "goto") {
                out << "\tj " << ins.result << "\n";
                continue;
            }

            if (ins.op == "iffalse") {
                std::string r = ensureReg(ins.arg1, out, "$t8");
                out << "\tbeq " << r << ", $zero, " << ins.result << "\n";
                continue;
            }

            if (ins.op == "iftrue") {
                std::string r = ensureReg(ins.arg1, out, "$t8");
                out << "\tbne " << r << ", $zero, " << ins.result << "\n";
                continue;
            }

            if (ins.op == "print") {
                if (ins.arg1.size() > 0 && ins.arg1[0] == 's' && stringTable.count(ins.arg1)) {
                    out << "\t# print string\n";
                    out << "\tli $v0, 4\n";
                    out << "\tla $a0, " << ins.arg1 << "\n";
                    out << "\tsyscall\n";
                } else {
                    std::string r = ensureReg(ins.arg1, out, "$t8");
                    out << "\t# print value\n";
                    out << "\tli $v0, 1\n";
                    out << "\tmove $a0, " << r << "\n";
                    out << "\tsyscall\n";
                }
                out << "\t# print newline\n";
                out << "\tli $v0, 11\n";
                out << "\tli $a0, 10\n";
                out << "\tsyscall\n";
                continue;
            }

            if (ins.op == "return") {
                if (!ins.arg1.empty()) {
                    std::string r = ensureReg(ins.arg1, out, "$t8");
                    out << "\tmove $v0, " << r << "\n";
                }
                out << "\tjr $ra\n";
                continue;
            }

            if (ins.op == "param" || ins.op == "call") {
                out << "\t# " << ins.op << " " << ins.arg1 << " " << ins.arg2 << " " << ins.result
                    << " (not lowered)\n";
                continue;
            }

            if (ins.op == "cast") {
                std::string src = ensureReg(ins.arg1, out, "$t8");
                std::string dst = ensureDest(ins.result);
                out << "\t# cast (" << ins.arg2 << ") " << ins.arg1 << "\n";
                emitMoveToDest(dst, src, out);
                continue;
            }

            if (ins.op == "!") {
                std::string src = ensureReg(ins.arg1, out, "$t8");
                std::string dst = ensureDest(ins.result);
                std::string dreg = ensureDestReg(dst, out, "$t9");
                out << "\t# logical not\n";
                out << "\tseq " << dreg << ", " << src << ", $zero\n";
                storeDest(dst, dreg, out);
                continue;
            }

            if (ins.op == "=") {
                std::string src = ensureReg(ins.arg1, out, "$t8");
                std::string dst = ensureDest(ins.result);
                emitMoveToDest(dst, src, out);
                continue;
            }

            if (ins.op == "+" || ins.op == "-" || ins.op == "*" || ins.op == "/" || ins.op == "%" ||
                ins.op == "==" || ins.op == "!=" || ins.op == "<" || ins.op == "<=" || ins.op == ">" || ins.op == ">=" ||
                ins.op == "&&" || ins.op == "||") {

                std::string a = ensureReg(ins.arg1, out, "$t8");
                std::string b = ensureReg(ins.arg2, out, "$t9");
                std::string dst = ensureDest(ins.result);
                std::string dreg = ensureDestReg(dst, out, "$t8");

                if (ins.op == "+") out << "\tadd " << dreg << ", " << a << ", " << b << "\n";
                else if (ins.op == "-") out << "\tsub " << dreg << ", " << a << ", " << b << "\n";
                else if (ins.op == "*") out << "\tmul " << dreg << ", " << a << ", " << b << "\n";
                else if (ins.op == "/") {
                    out << "\tdiv " << a << ", " << b << "\n";
                    out << "\tmflo " << dreg << "\n";
                } else if (ins.op == "%") {
                    out << "\tdiv " << a << ", " << b << "\n";
                    out << "\tmfhi " << dreg << "\n";
                } else if (ins.op == "==") out << "\tseq " << dreg << ", " << a << ", " << b << "\n";
                else if (ins.op == "!=") out << "\tsne " << dreg << ", " << a << ", " << b << "\n";
                else if (ins.op == "<") out << "\tslt " << dreg << ", " << a << ", " << b << "\n";
                else if (ins.op == ">") out << "\tslt " << dreg << ", " << b << ", " << a << "\n";
                else if (ins.op == "<=") {
                    out << "\tslt " << dreg << ", " << b << ", " << a << "\n";
                    out << "\txori " << dreg << ", " << dreg << ", 1\n";
                } else if (ins.op == ">=") {
                    out << "\tslt " << dreg << ", " << a << ", " << b << "\n";
                    out << "\txori " << dreg << ", " << dreg << ", 1\n";
                } else if (ins.op == "&&") {
                    out << "\tsne " << dreg << ", " << a << ", $zero\n";
                    out << "\tbeq " << dreg << ", $zero, skip_" << labelCounter << "\n";
                    out << "\tsne " << dreg << ", " << b << ", $zero\n";
                    out << "skip_" << labelCounter << ":\n";
                    labelCounter++;
                } else if (ins.op == "||") {
                    out << "\tsne " << dreg << ", " << a << ", $zero\n";
                    out << "\tbne " << dreg << ", $zero, skip_" << labelCounter << "\n";
                    out << "\tsne " << dreg << ", " << b << ", $zero\n";
                    out << "skip_" << labelCounter << ":\n";
                    labelCounter++;
                }

                storeDest(dst, dreg, out);
                continue;
            }

            out << "\t# Unsupported: " << ins.op << " " << ins.arg1 << " " << ins.arg2 << " " << ins.result << "\n";
        }

        out << "\t# Exit program\n";
        out << "\tli $v0, 10\n";
        out << "\tsyscall\n";

        return out.str();
    }

std::string AssemblyGenerator::escapeString(const std::string& s) {
        std::string res;
        for (char c : s) {
            switch (c) {
                case '\n': res += "\\n"; break;
                case '\t': res += "\\t"; break;
                case '\\': res += "\\\\"; break;
                case '"': res += "\\\""; break;
                case '\'': res += "\\'"; break;
                default: res += c; break;
            }
        }
        return res;
    }

bool AssemblyGenerator::isNumber(const std::string& s) {
        if (s.empty()) return false;
        char* end = nullptr;
        std::strtod(s.c_str(), &end);
        return end && *end == '\0';
    }

bool AssemblyGenerator::isTemp(const std::string& s) { return !s.empty() && (s[0] == 't' || s[0] == 'i' || s[0] == 'f' || s[0] == 'b' || s[0] == 'c' || s[0] == 's'); }

bool AssemblyGenerator::isLabel(const std::string& s) { return !s.empty() && s[0] == 'L'; }

bool AssemblyGenerator::isStringId(const std::string& s) { return !s.empty() && s[0] == 's'; }

bool AssemblyGenerator::isIdentLike(const std::string& s) {
        if (s.empty()) return false;
        if (!(std::isalpha((unsigned char)s[0]) || s[0] == '_')) return false;
        for (char c : s) {
            if (!(std::isalnum((unsigned char)c) || c == '_')) return false;
        }
        return true;
    }

void AssemblyGenerator::collectSymbols() {
        auto touch = [&](const std::string& x) {
            if (x.empty()) return;
            if (isLabel(x) || x == "<eof>") return;
            if (isNumber(x)) return;
            if (isStringId(x)) {
                stringTable[x] = "";
                return;
            }
            if (isTemp(x)) return;
            if (isIdentLike(x)) variables.insert(x);
        };

        for (const auto& ins : tac) {
            if (ins.op == "label" || ins.op == "goto") continue;
            touch(ins.arg1);
            touch(ins.arg2);
            touch(ins.result);
        }

        std::unordered_set<std::string> temps;
        for (const auto& ins : tac) {
            if (isTemp(ins.arg1)) temps.insert(ins.arg1);
            if (isTemp(ins.arg2)) temps.insert(ins.arg2);
            if (isTemp(ins.result)) temps.insert(ins.result);
        }

        int cnt = 0;
        for (const auto& t : temps) {
            if (cnt < 8) {
                tempReg[t] = "$t" + std::to_string(cnt);
            } else {
                spills.insert("_spill_" + t);
            }
            cnt++;
        }
    }

std::string AssemblyGenerator::ensureDest(const std::string& x) const {
        if (isTemp(x) || isIdentLike(x)) return x;
        return x;
    }

std::string AssemblyGenerator::ensureReg(const std::string& op, std::ostringstream& out, const std::string& scratchPref) {
        if (op.empty()) return "$zero";
        if (isNumber(op)) {
            out << "\tli " << scratchPref << ", " << op << "\n";
            return scratchPref;
        }
        if (isTemp(op)) {
            auto it = tempReg.find(op);
            if (it != tempReg.end()) return it->second;
            out << "\tlw " << scratchPref << ", _spill_" << op << "\n";
            return scratchPref;
        }
        if (isIdentLike(op)) {
            out << "\tlw " << scratchPref << ", " << op << "\n";
            return scratchPref;
        }
        return scratchPref;
    }

std::string AssemblyGenerator::ensureDestReg(const std::string& dst, std::ostringstream& out, const std::string& scratchPref) {
    (void)out;
        if (isTemp(dst)) {
            auto it = tempReg.find(dst);
            if (it != tempReg.end()) return it->second;
            return scratchPref;
        }
        return scratchPref;
    }

void AssemblyGenerator::storeDest(const std::string& dst, const std::string& reg, std::ostringstream& out) {
        if (isTemp(dst)) {
            auto it = tempReg.find(dst);
            if (it != tempReg.end()) return;
            out << "\tsw " << reg << ", _spill_" << dst << "\n";
            return;
        }
        if (isIdentLike(dst)) {
            out << "\tsw " << reg << ", " << dst << "\n";
            return;
        }
    }

void AssemblyGenerator::emitMoveToDest(const std::string& dst, const std::string& srcReg, std::ostringstream& out) {
        if (isTemp(dst)) {
            auto it = tempReg.find(dst);
            if (it != tempReg.end()) {
                out << "\tmove " << it->second << ", " << srcReg << "\n";
            } else {
                out << "\tsw " << srcReg << ", _spill_" << dst << "\n";
            }
            return;
        }
        if (isIdentLike(dst)) {
            out << "\tsw " << srcReg << ", " << dst << "\n";
            return;
        }
        out << "\t# move (ignored dest=" << dst << ")\n";
    }
