#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "frontend/token.h"

// 词法分析器：源码字符流 -> Token 序列
class Lexer {
  public:
    explicit Lexer(std::string src);

    std::vector<Token> tokenize();

  private:
    bool isAtEnd() const;
    char peek() const;
    char peekNext() const;
    char advance();
    void skipWhitespaceAndComments();

    Token identifier(int startLine, int startCol);
    Token number(int startLine, int startCol);
    Token characterLiteral(int startLine, int startCol);
    Token stringLiteral(int startLine, int startCol);
    Token singleCharToken(TokenType type, const std::string& lexeme, int startLine, int startCol);
    Token operatorToken(int startLine, int startCol);

    std::string source;
    size_t current = 0;
    int line = 1;
    int column = 0;

    const std::unordered_map<std::string, TokenType> keywords{
        {"int", TokenType::Keyword},     {"float", TokenType::Keyword},   {"bool", TokenType::Keyword},
        {"char", TokenType::Keyword},    {"const", TokenType::Keyword},   {"if", TokenType::Keyword},
        {"else", TokenType::Keyword},    {"while", TokenType::Keyword},   {"for", TokenType::Keyword},
        {"do", TokenType::Keyword},      {"switch", TokenType::Keyword},  {"case", TokenType::Keyword},
        {"default", TokenType::Keyword}, {"break", TokenType::Keyword},   {"continue", TokenType::Keyword},
        {"return", TokenType::Keyword},  {"true", TokenType::BoolLiteral}, {"false", TokenType::BoolLiteral}};

    const std::unordered_map<std::string, std::string> twoCharOps{
        {"==", "=="}, {"!=", "!="}, {"<=", "<="}, {">=", ">="}, {"&&", "&&"}, {"||", "||"},
        {"+=", "+="}, {"-=", "-="}, {"*=", "*="}, {"/=", "/="}, {"%=", "%="}, {"++", "++"},
        {"--", "--"}};

    const std::unordered_set<char> singleCharSeparators{'(', ')', '{', '}', ';', ',', '[', ']'};
};
