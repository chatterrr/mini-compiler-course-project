#include "ir/ir_printer.h"

#include <iomanip>
#include <ostream>

void printIR(const std::vector<Tac>& code, std::ostream& os) {
    int irIndex = 0;
    for (const auto& ins : code) {
        os << std::setw(3) << irIndex++ << ": ";
        if (ins.op == "label") {
            os << ins.result << ":";
        } else if (ins.op == "goto" || ins.op == "iffalse" || ins.op == "iftrue") {
            os << ins.op << " " << ins.arg1 << " " << ins.result;
        } else if (ins.op == "return") {
            os << "return " << ins.arg1;
        } else if (ins.op == "print") {
            os << "print " << ins.arg1;
        } else if (ins.op == "param") {
            os << "param " << ins.arg1;
        } else if (ins.op == "call") {
            os << ins.result << " = call " << ins.arg1 << ", " << ins.arg2;
        } else if (ins.op == "=" && ins.arg2.empty()) {
            os << ins.result << " = " << ins.arg1;
        } else if (ins.op == "cast") {
            os << ins.result << " = (" << ins.arg2 << ")" << ins.arg1;
        } else if (ins.op == "!" && ins.arg2.empty()) {
            os << ins.result << " = !" << ins.arg1;
        } else {
            os << ins.result << " = " << ins.arg1 << " " << ins.op << " " << ins.arg2;
        }
        os << "\n";
    }
}
