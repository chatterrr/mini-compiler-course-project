#include "frontend/lexer.h"
#include <cctype>
#include <stdexcept>
#include <utility>

Lexer::Lexer(std::string src) : source(std::move(src)) {}

std::vector<Token> Lexer::tokenize() {
        std::vector<Token> tokens;
        while (true) {
            skipWhitespaceAndComments();
            if (isAtEnd()) break;
            char c = peek();
            int startLine = line;
            int startCol = column + 1;

            if (std::isalpha(c) || c == '_') {
                tokens.push_back(identifier(startLine, startCol));
            } else if (std::isdigit(c)) {
                tokens.push_back(number(startLine, startCol));
            } else if (c == '\'') {
                tokens.push_back(characterLiteral(startLine, startCol));
            } else if (c == '"') {
                tokens.push_back(stringLiteral(startLine, startCol));
            } else if (c == ':') {
                // 冒号单独处理，作为Operator
                tokens.push_back(Token{TokenType::Operator, ":", startLine, startCol});
                advance();
            } else if (c == '?') {
                // 问号单独处理，作为Operator
                tokens.push_back(Token{TokenType::Operator, "?", startLine, startCol});
                advance();
            } else if (singleCharSeparators.count(c)) {
                tokens.push_back(singleCharToken(TokenType::Separator, std::string(1, c), startLine, startCol));
                advance();
            } else {
                tokens.push_back(operatorToken(startLine, startCol));
            }
        }
        tokens.push_back(Token{TokenType::EndOfFile, "<eof>", line, column});
        return tokens;
    }

bool Lexer::isAtEnd() const { return current >= source.size(); }

char Lexer::peek() const { return source[current]; }

char Lexer::peekNext() const { return (current + 1 < source.size()) ? source[current + 1] : '\0'; }

char Lexer::advance() {
        column++;
        return source[current++];
    }

void Lexer::skipWhitespaceAndComments() {
        while (!isAtEnd()) {
            char c = peek();
            if (c == ' ' || c == '\r' || c == '\t') {
                advance();
            } else if (c == '\n') {
                advance();
                line++;
                column = 0;
            } else if (c == '/' && peekNext() == '/') {
                while (!isAtEnd() && peek() != '\n') advance();
            } else if (c == '/' && peekNext() == '*') {
                advance();
                advance();
                while (!isAtEnd()) {
                    if (peek() == '*' && peekNext() == '/') {
                        advance();
                        advance();
                        break;
                    }
                    if (peek() == '\n') {
                        line++;
                        column = 0;
                    }
                    advance();
                }
            } else {
                break;
            }
        }
    }

Token Lexer::identifier(int startLine, int startCol) {
        std::string lexeme;
        while (!isAtEnd() && (std::isalnum(peek()) || peek() == '_')) {
            lexeme += advance();
        }
        if (keywords.count(lexeme)) {
            TokenType type = keywords.at(lexeme);
            return Token{type, lexeme, startLine, startCol};
        }
        return Token{TokenType::Identifier, lexeme, startLine, startCol};
    }

Token Lexer::number(int startLine, int startCol) {
        std::string lexeme;
        bool hasDot = false;
        while (!isAtEnd() && (std::isdigit(peek()) || peek() == '.')) {
            if (peek() == '.') {
                if (hasDot) break;
                hasDot = true;
            }
            lexeme += advance();
        }
        return Token{TokenType::Number, lexeme, startLine, startCol};
    }

Token Lexer::characterLiteral(int startLine, int startCol) {
        std::string lexeme;
        advance();

        while (!isAtEnd() && peek() != '\'') {
            if (peek() == '\\') {
                advance();
                switch (peek()) {
                    case 'n': lexeme += '\n'; break;
                    case 't': lexeme += '\t'; break;
                    case '\\': lexeme += '\\'; break;
                    case '\'': lexeme += '\''; break;
                    case '"': lexeme += '"'; break;
                    default: lexeme += '\\'; lexeme += peek(); break;
                }
            } else {
                lexeme += peek();
            }
            advance();
        }

        if (peek() == '\'') {
            advance();
        } else {
            throw std::runtime_error("未闭合的字符常量，位置: line " +
                                   std::to_string(startLine) + " col " +
                                   std::to_string(startCol));
        }

        if (lexeme.length() == 1) {
            int ascii = static_cast<int>(lexeme[0]);
            return Token{TokenType::CharLiteral, std::to_string(ascii), startLine, startCol};
        } else if (lexeme.length() == 0) {
            throw std::runtime_error("空字符常量，位置: line " +
                                   std::to_string(startLine) + " col " +
                                   std::to_string(startCol));
        } else {
            return Token{TokenType::CharLiteral, std::to_string(static_cast<int>(lexeme[0])), startLine, startCol};
        }
    }

Token Lexer::stringLiteral(int startLine, int startCol) {
        std::string lexeme;
        advance();

        while (!isAtEnd() && peek() != '"') {
            if (peek() == '\\') {
                advance();
                switch (peek()) {
                    case 'n': lexeme += '\n'; break;
                    case 't': lexeme += '\t'; break;
                    case '\\': lexeme += '\\'; break;
                    case '\'': lexeme += '\''; break;
                    case '"': lexeme += '"'; break;
                    default: lexeme += '\\'; lexeme += peek(); break;
                }
                advance();
            } else {
                lexeme += advance();
            }
        }

        if (peek() == '"') {
            advance();
        } else {
            throw std::runtime_error("未闭合的字符串常量，位置: line " +
                                   std::to_string(startLine) + " col " +
                                   std::to_string(startCol));
        }

        return Token{TokenType::StringLiteral, lexeme, startLine, startCol};
    }

Token Lexer::singleCharToken(TokenType type, const std::string& lexeme, int startLine, int startCol) {
        return Token{type, lexeme, startLine, startCol};
    }

Token Lexer::operatorToken(int startLine, int startCol) {
        std::string lexeme;
        lexeme += advance();
        
        // 检查是否为双字符运算符
        if (!isAtEnd()) {
            std::string twoChars = lexeme + peek();
            if (twoCharOps.count(twoChars)) {
                lexeme += advance();
                return Token{TokenType::Operator, lexeme, startLine, startCol};
            }
        }
        
        // 如果是已经单独处理的字符，不应该走到这里
        if (lexeme == ":" || lexeme == "?") {
            return Token{TokenType::Operator, lexeme, startLine, startCol};
        }
        
        // 检查是否为有效的单字符运算符
        static const std::unordered_set<char> validOps = {'+', '-', '*', '/', '%', '=', '<', '>', '!', '&', '|'};
        if (validOps.count(lexeme[0])) {
            return Token{TokenType::Operator, lexeme, startLine, startCol};
        }
        
        throw std::runtime_error("Unexpected character '" + lexeme + "' at line " + 
                               std::to_string(startLine) + " column " + 
                               std::to_string(startCol));
    }
