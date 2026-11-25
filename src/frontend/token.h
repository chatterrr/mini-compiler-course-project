#pragma once

#include <string>

// Token类型枚举
enum class TokenType {
    Identifier,
    Number,
    Keyword,
    Operator,
    Separator,
    EndOfFile,
    CharLiteral,
    BoolLiteral,
    StringLiteral,
};

struct Token {
    TokenType type;
    std::string lexeme;
    int line;
    int column;
};
