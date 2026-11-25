#include "output/token_printer.h"

#include <iomanip>
#include <ostream>

std::string tokenTypeName(TokenType t) {
    switch (t) {
    case TokenType::Identifier:
        return "Identifier";
    case TokenType::Number:
        return "Number";
    case TokenType::Keyword:
        return "Keyword";
    case TokenType::Operator:
        return "Operator";
    case TokenType::Separator:
        return "Separator";
    case TokenType::EndOfFile:
        return "EOF";
    case TokenType::CharLiteral:
        return "CharLiteral";
    case TokenType::BoolLiteral:
        return "BoolLiteral";
    case TokenType::StringLiteral:
        return "StringLiteral";
    }
    return "Unknown";
}

void printTokens(const std::vector<Token>& tokens, std::ostream& os) {
    os << "=== TOKENS ===\n";
    for (const auto& t : tokens) {
        if (t.type == TokenType::EndOfFile) break;
        os << std::setw(4) << t.line << ":" << std::setw(3) << t.column << "  " << std::left << std::setw(12)
           << tokenTypeName(t.type) << "  " << t.lexeme << "\n";
    }
}
