#include "frontend/parser.h"

#include <sstream>
#include <stdexcept>
#include <utility>

Parser::Parser(std::vector<Token> tokens) : toks(std::move(tokens)), strings(&ownedStrings) {
    astRoot = std::make_shared<ASTNode>(ASTNodeType::Program, "program");
}

Parser::Parser(std::vector<Token> tokens, StringInterner& shared)
    : toks(std::move(tokens)), strings(&shared) {
    astRoot = std::make_shared<ASTNode>(ASTNodeType::Program, "program");
}

void Parser::parseProgram() {
    while (!check(TokenType::EndOfFile)) {
        auto stmt = parseStatement();
        if (stmt) astRoot->children.push_back(stmt);
    }
}

bool Parser::isAtEnd() const { return peek().type == TokenType::EndOfFile; }

const Token& Parser::peek() const { return toks[current]; }

const Token& Parser::previous() const { return toks[current - 1]; }

bool Parser::check(TokenType type, const std::string& lexeme) const {
    if (isAtEnd()) return type == TokenType::EndOfFile;
    if (peek().type != type) return false;
    if (!lexeme.empty() && peek().lexeme != lexeme) return false;
    return true;
}

const Token& Parser::advance() {
    if (!isAtEnd()) current++;
    return previous();
}

bool Parser::match(TokenType type, const std::string& lexeme) {
    if (check(type, lexeme)) {
        advance();
        return true;
    }
    return false;
}

const Token& Parser::consume(TokenType type, const std::string& lexeme, const std::string& msg) {
    if (check(type, lexeme)) return advance();
    throw std::runtime_error(errorAt(peek(), msg));
}

const Token& Parser::consume(TokenType type, const std::string& msg) {
    if (check(type)) return advance();
    throw std::runtime_error(errorAt(peek(), msg));
}

std::string Parser::errorAt(const Token& tok, const std::string& msg) const {
    std::ostringstream oss;
    oss << "语法错误 (line " << tok.line << ", col " << tok.column << "): " << msg << "，发现 '" << tok.lexeme
        << "'";
    return oss.str();
}

ValueType Parser::parseType() {
    if (match(TokenType::Keyword, "int")) {
        return ValueType::Int;
    } else if (match(TokenType::Keyword, "float")) {
        return ValueType::Float;
    } else if (match(TokenType::Keyword, "bool")) {
        return ValueType::Bool;
    } else if (match(TokenType::Keyword, "char")) {
        return ValueType::Char;
    }
    throw std::runtime_error("未知类型");
}

std::shared_ptr<ASTNode> Parser::makeVariable(const Token& tok, ValueType type) {
    return std::make_shared<ASTNode>(ASTNodeType::Variable, tok.lexeme, type, tok.line, tok.column);
}

std::shared_ptr<ASTNode> Parser::parseStatement() {
    if (match(TokenType::Keyword, "const")) {
        ValueType t = parseType();
        return parseConstDeclaration(t);
    } else if (check(TokenType::Keyword, "int") || check(TokenType::Keyword, "float") ||
               check(TokenType::Keyword, "bool") || check(TokenType::Keyword, "char")) {
        ValueType t = parseType();
        return parseDeclaration(t);
    } else if (match(TokenType::Keyword, "if")) {
        return parseIf();
    } else if (match(TokenType::Keyword, "while")) {
        return parseWhile();
    } else if (match(TokenType::Keyword, "do")) {
        return parseDoWhile();
    } else if (match(TokenType::Keyword, "for")) {
        return parseFor();
    } else if (match(TokenType::Keyword, "switch")) {
        return parseSwitch();
    } else if (match(TokenType::Keyword, "break")) {
        return parseBreak();
    } else if (match(TokenType::Keyword, "continue")) {
        return parseContinue();
    } else if (match(TokenType::Separator, "{")) {
        return parseBlock();
    } else if (match(TokenType::Keyword, "return")) {
        return parseReturn();
    }
    return parseExpressionStatement();
}

std::shared_ptr<ASTNode> Parser::parseDeclaration(ValueType type) {
    const Token& idTok = consume(TokenType::Identifier, "缺少标识符");

    auto declNode =
        std::make_shared<ASTNode>(ASTNodeType::Declaration, idTok.lexeme, type, idTok.line, idTok.column);
    auto varNode = makeVariable(idTok, type);
    declNode->children.push_back(varNode);

    if (match(TokenType::Operator, "=")) {
        auto init = assignment();
        auto assignNode =
            std::make_shared<ASTNode>(ASTNodeType::Assignment, "=", type, idTok.line, idTok.column);
        assignNode->children.push_back(varNode);
        assignNode->children.push_back(init);
        declNode->children.push_back(assignNode);
    }
    consume(TokenType::Separator, ";", "缺少分号");
    return declNode;
}

std::shared_ptr<ASTNode> Parser::parseConstDeclaration(ValueType type) {
    const Token& idTok = consume(TokenType::Identifier, "缺少标识符");
    consume(TokenType::Operator, "=", "常量必须初始化");
    auto init = assignment();

    auto constDeclNode = std::make_shared<ASTNode>(ASTNodeType::ConstDeclaration, idTok.lexeme, type,
                                                  idTok.line, idTok.column);
    auto varNode = makeVariable(idTok, type);
    constDeclNode->children.push_back(varNode);
    constDeclNode->children.push_back(init);
    consume(TokenType::Separator, ";", "缺少分号");
    return constDeclNode;
}

std::shared_ptr<ASTNode> Parser::parseIf() {
    const Token& ifTok = previous();
    consume(TokenType::Separator, "(", "缺少 '('");
    auto cond = assignment();
    consume(TokenType::Separator, ")", "缺少 ')'");

    auto ifNode = std::make_shared<ASTNode>(ASTNodeType::If, "if", ValueType::Void, ifTok.line, ifTok.column);
    ifNode->children.push_back(cond);
    ifNode->children.push_back(parseStatement());
    if (match(TokenType::Keyword, "else")) {
        ifNode->children.push_back(parseStatement());
    }
    return ifNode;
}

std::shared_ptr<ASTNode> Parser::parseWhile() {
    const Token& whileTok = previous();
    consume(TokenType::Separator, "(", "缺少 '('");
    auto cond = assignment();
    consume(TokenType::Separator, ")", "缺少 ')'");

    ++loopDepth;
    auto body = parseStatement();
    --loopDepth;

    auto whileNode =
        std::make_shared<ASTNode>(ASTNodeType::While, "while", ValueType::Void, whileTok.line, whileTok.column);
    whileNode->children.push_back(cond);
    whileNode->children.push_back(body);
    return whileNode;
}

std::shared_ptr<ASTNode> Parser::parseDoWhile() {
    const Token& doTok = previous();

    ++loopDepth;
    auto body = parseStatement();
    --loopDepth;

    consume(TokenType::Keyword, "while", "缺少 'while'");
    consume(TokenType::Separator, "(", "缺少 '('");
    auto cond = assignment();
    consume(TokenType::Separator, ")", "缺少 ')'");
    consume(TokenType::Separator, ";", "缺少分号");

    auto doWhileNode = std::make_shared<ASTNode>(ASTNodeType::DoWhile, "do-while", ValueType::Void, doTok.line,
                                                doTok.column);
    doWhileNode->children.push_back(body);
    doWhileNode->children.push_back(cond);
    return doWhileNode;
}

std::shared_ptr<ASTNode> Parser::parseSwitch() {
    const Token& switchTok = previous();
    consume(TokenType::Separator, "(", "缺少 '('");
    auto expr = assignment();
    consume(TokenType::Separator, ")", "缺少 ')'");
    consume(TokenType::Separator, "{", "缺少 '{'");

    auto switchNode =
        std::make_shared<ASTNode>(ASTNodeType::Switch, "switch", ValueType::Void, switchTok.line, switchTok.column);
    switchNode->children.push_back(expr);

    ++switchDepth;
    while (!check(TokenType::Separator, "}") && !isAtEnd()) {
        if (match(TokenType::Keyword, "case")) {
            auto caseExpr = assignment();
            consume(TokenType::Operator, ":", "缺少 ':'");
            const Token& colonTok = previous();

            auto caseNode =
                std::make_shared<ASTNode>(ASTNodeType::Case, "case", ValueType::Void, colonTok.line, colonTok.column);
            caseNode->children.push_back(caseExpr);

            while (!check(TokenType::Keyword, "case") && !check(TokenType::Keyword, "default") &&
                   !check(TokenType::Separator, "}")) {
                auto stmt = parseStatement();
                if (stmt) caseNode->children.push_back(stmt);
            }
            switchNode->children.push_back(caseNode);
        } else if (match(TokenType::Keyword, "default")) {
            consume(TokenType::Operator, ":", "缺少 ':'");
            const Token& colonTok = previous();

            auto defaultNode = std::make_shared<ASTNode>(ASTNodeType::Default, "default", ValueType::Void,
                                                         colonTok.line, colonTok.column);
            while (!check(TokenType::Separator, "}")) {
                auto stmt = parseStatement();
                if (stmt) defaultNode->children.push_back(stmt);
            }
            switchNode->children.push_back(defaultNode);
        } else {
            auto stmt = parseStatement();
            if (stmt) switchNode->children.push_back(stmt);
        }
    }
    --switchDepth;
    consume(TokenType::Separator, "}", "缺少 '}'");
    return switchNode;
}

std::shared_ptr<ASTNode> Parser::parseBreak() {
    const Token& breakTok = previous();
    consume(TokenType::Separator, ";", "缺少 ';'");
    if (loopDepth == 0 && switchDepth == 0) {
        throw std::runtime_error(errorAt(breakTok, "break语句必须在循环或switch语句内"));
    }
    return std::make_shared<ASTNode>(ASTNodeType::Break, "break", ValueType::Void, breakTok.line, breakTok.column);
}

std::shared_ptr<ASTNode> Parser::parseContinue() {
    const Token& continueTok = previous();
    consume(TokenType::Separator, ";", "缺少 ';' ");
    if (loopDepth == 0) {
        throw std::runtime_error(errorAt(continueTok, "continue语句必须在循环语句内"));
    }
    return std::make_shared<ASTNode>(ASTNodeType::Continue, "continue", ValueType::Void, continueTok.line,
                                     continueTok.column);
}

std::shared_ptr<ASTNode> Parser::parseFor() {
    const Token& forTok = previous();
    consume(TokenType::Separator, "(", "缺少 '('");

    std::shared_ptr<ASTNode> init;
    if (!match(TokenType::Separator, ";")) {
        if (check(TokenType::Keyword, "const") || check(TokenType::Keyword, "int") ||
            check(TokenType::Keyword, "float") || check(TokenType::Keyword, "bool") ||
            check(TokenType::Keyword, "char")) {
            bool isConst = match(TokenType::Keyword, "const");
            ValueType t = parseType();
            init = isConst ? parseConstDeclaration(t) : parseDeclaration(t);
        } else {
            init = parseExpressionStatement();
        }
    }

    std::shared_ptr<ASTNode> cond;
    if (!check(TokenType::Separator, ";")) {
        cond = assignment();
    }
    consume(TokenType::Separator, ";", "缺少';'");

    std::vector<Token> incTokens;
    while (!check(TokenType::Separator, ")")) {
        incTokens.push_back(peek());
        advance();
    }
    consume(TokenType::Separator, ")", "缺少')'");

    ++loopDepth;
    auto body = parseStatement();
    --loopDepth;

    std::shared_ptr<ASTNode> inc;
    if (!incTokens.empty()) {
        incTokens.push_back(Token{TokenType::Separator, ";", peek().line, peek().column});
        incTokens.push_back(Token{TokenType::EndOfFile, "<eof>", peek().line, peek().column});
        Parser incParser(incTokens, *strings);
        inc = incParser.parseExpressionStatement();
    }

    auto forNode = std::make_shared<ASTNode>(ASTNodeType::For, "for", ValueType::Void, forTok.line, forTok.column);
    // 固定槽位：[init, cond, body, inc]，缺失项为 nullptr
    forNode->children.push_back(init);
    forNode->children.push_back(cond);
    forNode->children.push_back(body);
    forNode->children.push_back(inc);
    return forNode;
}

std::shared_ptr<ASTNode> Parser::parseBlock() {
    auto blockNode =
        std::make_shared<ASTNode>(ASTNodeType::Block, "block", ValueType::Void, peek().line, peek().column);
    while (!check(TokenType::Separator, "}") && !isAtEnd()) {
        auto stmt = parseStatement();
        if (stmt) blockNode->children.push_back(stmt);
    }
    consume(TokenType::Separator, "}", "缺少 '}'");
    return blockNode;
}

std::shared_ptr<ASTNode> Parser::parseReturn() {
    const Token& retTok = previous();
    auto returnNode =
        std::make_shared<ASTNode>(ASTNodeType::Return, "return", ValueType::Void, retTok.line, retTok.column);
    if (match(TokenType::Separator, ";")) {
        return returnNode;
    }
    auto value = assignment();
    consume(TokenType::Separator, ";", "缺少分号");
    returnNode->children.push_back(value);
    return returnNode;
}

std::shared_ptr<ASTNode> Parser::parseExpressionStatement() {
    auto value = assignment();
    consume(TokenType::Separator, ";", "缺少分号");
    return value;
}

std::shared_ptr<ASTNode> Parser::assignment() {
    auto lhs = conditional();

    if (match(TokenType::Operator, "=") || match(TokenType::Operator, "+=") ||
        match(TokenType::Operator, "-=") || match(TokenType::Operator, "*=") ||
        match(TokenType::Operator, "/=") || match(TokenType::Operator, "%=")) {
        std::string op = previous().lexeme;
        const Token& opTok = previous();
        auto rhs = assignment();

        if (op == "=") {
            auto node =
                std::make_shared<ASTNode>(ASTNodeType::Assignment, "=", lhs->dataType, opTok.line, opTok.column);
            node->children.push_back(lhs);
            node->children.push_back(rhs);
            return node;
        }

        std::string baseOp = op.substr(0, 1);
        auto binop =
            std::make_shared<ASTNode>(ASTNodeType::BinaryOp, baseOp, ValueType::Void, opTok.line, opTok.column);
        if (lhs->type == ASTNodeType::Variable) {
            binop->children.push_back(
                std::make_shared<ASTNode>(ASTNodeType::Variable, lhs->value, lhs->dataType, lhs->line, lhs->column));
        } else {
            binop->children.push_back(lhs);
        }
        binop->children.push_back(rhs);

        auto node =
            std::make_shared<ASTNode>(ASTNodeType::Assignment, op, ValueType::Void, opTok.line, opTok.column);
        node->children.push_back(lhs);
        node->children.push_back(binop);
        return node;
    }
    return lhs;
}

std::shared_ptr<ASTNode> Parser::conditional() {
    auto cond = logical_or();
    if (match(TokenType::Operator, "?")) {
        const Token& questionTok = previous();
        auto trueExpr = assignment();
        consume(TokenType::Operator, ":", "缺少 ':' 在三元条件运算符中");
        auto falseExpr = conditional();

        auto node = std::make_shared<ASTNode>(ASTNodeType::TernaryOp, "?:", ValueType::Void, questionTok.line,
                                              questionTok.column);
        node->children.push_back(cond);
        node->children.push_back(trueExpr);
        node->children.push_back(falseExpr);
        return node;
    }
    return cond;
}

std::shared_ptr<ASTNode> Parser::logical_or() {
    auto left = logical_and();
    while (match(TokenType::Operator, "||")) {
        auto right = logical_and();
        const Token& pos = previous();
        auto node =
            std::make_shared<ASTNode>(ASTNodeType::BinaryOp, "||", ValueType::Bool, pos.line, pos.column);
        node->children.push_back(left);
        node->children.push_back(right);
        left = node;
    }
    return left;
}

std::shared_ptr<ASTNode> Parser::logical_and() {
    auto left = equality();
    while (match(TokenType::Operator, "&&")) {
        auto right = equality();
        const Token& pos = previous();
        auto node =
            std::make_shared<ASTNode>(ASTNodeType::BinaryOp, "&&", ValueType::Bool, pos.line, pos.column);
        node->children.push_back(left);
        node->children.push_back(right);
        left = node;
    }
    return left;
}

std::shared_ptr<ASTNode> Parser::equality() {
    auto left = relational();
    while (match(TokenType::Operator, "==") || match(TokenType::Operator, "!=")) {
        std::string op = previous().lexeme;
        auto right = relational();
        const Token& pos = previous();
        auto node = std::make_shared<ASTNode>(ASTNodeType::BinaryOp, op, ValueType::Bool, pos.line, pos.column);
        node->children.push_back(left);
        node->children.push_back(right);
        left = node;
    }
    return left;
}

std::shared_ptr<ASTNode> Parser::relational() {
    auto left = additive();
    while (match(TokenType::Operator, "<") || match(TokenType::Operator, ">") ||
           match(TokenType::Operator, "<=") || match(TokenType::Operator, ">=")) {
        std::string op = previous().lexeme;
        auto right = additive();
        const Token& pos = previous();
        auto node = std::make_shared<ASTNode>(ASTNodeType::BinaryOp, op, ValueType::Bool, pos.line, pos.column);
        node->children.push_back(left);
        node->children.push_back(right);
        left = node;
    }
    return left;
}

std::shared_ptr<ASTNode> Parser::additive() {
    auto left = multiplicative();
    while (match(TokenType::Operator, "+") || match(TokenType::Operator, "-")) {
        std::string op = previous().lexeme;
        auto right = multiplicative();
        const Token& pos = previous();
        auto node =
            std::make_shared<ASTNode>(ASTNodeType::BinaryOp, op, ValueType::Void, pos.line, pos.column);
        node->children.push_back(left);
        node->children.push_back(right);
        left = node;
    }
    return left;
}

std::shared_ptr<ASTNode> Parser::multiplicative() {
    auto left = unary();
    while (match(TokenType::Operator, "*") || match(TokenType::Operator, "/") ||
           match(TokenType::Operator, "%")) {
        std::string op = previous().lexeme;
        auto right = unary();
        const Token& pos = previous();
        auto node =
            std::make_shared<ASTNode>(ASTNodeType::BinaryOp, op, ValueType::Void, pos.line, pos.column);
        node->children.push_back(left);
        node->children.push_back(right);
        left = node;
    }
    return left;
}

std::shared_ptr<ASTNode> Parser::unary() {
    if (match(TokenType::Operator, "++") || match(TokenType::Operator, "--")) {
        std::string op = previous().lexeme;
        const Token& opTok = previous();
        auto rhs = unary();
        auto node = std::make_shared<ASTNode>(ASTNodeType::UnaryOp, op + "pre", rhs->dataType, opTok.line,
                                              opTok.column);
        node->children.push_back(rhs);
        return node;
    }

    if (match(TokenType::Operator, "-")) {
        auto rhs = unary();
        const Token& pos = previous();
        auto node = std::make_shared<ASTNode>(ASTNodeType::UnaryOp, "-", rhs->dataType, pos.line, pos.column);
        node->children.push_back(rhs);
        return node;
    }

    if (match(TokenType::Operator, "!")) {
        auto rhs = unary();
        const Token& pos = previous();
        auto node =
            std::make_shared<ASTNode>(ASTNodeType::UnaryOp, "!", ValueType::Bool, pos.line, pos.column);
        node->children.push_back(rhs);
        return node;
    }

    return postfix();
}

std::shared_ptr<ASTNode> Parser::postfix() {
    auto expr = primary();
    while (match(TokenType::Operator, "++") || match(TokenType::Operator, "--")) {
        std::string op = previous().lexeme;
        const Token& opTok = previous();
        auto node = std::make_shared<ASTNode>(ASTNodeType::UnaryOp, op + "post", expr->dataType, opTok.line,
                                              opTok.column);
        node->children.push_back(expr);
        expr = node;
    }
    return expr;
}

std::shared_ptr<ASTNode> Parser::primary() {
    if (match(TokenType::Number)) {
        const Token& numTok = previous();
        ValueType vt = (numTok.lexeme.find('.') != std::string::npos) ? ValueType::Float : ValueType::Int;
        return std::make_shared<ASTNode>(ASTNodeType::Literal, numTok.lexeme, vt, numTok.line, numTok.column);
    }
    if (match(TokenType::CharLiteral)) {
        const Token& charTok = previous();
        return std::make_shared<ASTNode>(ASTNodeType::Literal, charTok.lexeme, ValueType::Char, charTok.line,
                                         charTok.column);
    }
    if (match(TokenType::BoolLiteral)) {
        const Token& boolTok = previous();
        std::string val = (boolTok.lexeme == "true") ? "1" : "0";
        return std::make_shared<ASTNode>(ASTNodeType::Literal, val, ValueType::Bool, boolTok.line, boolTok.column);
    }
    if (match(TokenType::StringLiteral)) {
        const Token& strTok = previous();
        std::string strId = strings->intern(strTok.lexeme);
        return std::make_shared<ASTNode>(ASTNodeType::Literal, strId, ValueType::String, strTok.line, strTok.column);
    }
    if (match(TokenType::Identifier)) {
        const Token& idTok = previous();
        if (match(TokenType::Separator, "(")) {
            std::vector<std::shared_ptr<ASTNode>> args;
            if (!check(TokenType::Separator, ")")) {
                args.push_back(assignment());
                while (match(TokenType::Separator, ",")) {
                    args.push_back(assignment());
                }
            }
            consume(TokenType::Separator, ")", "缺少 ')'");

            if (idTok.lexeme == "print") {
                auto printNode = std::make_shared<ASTNode>(ASTNodeType::Print, "print", ValueType::Void,
                                                           idTok.line, idTok.column);
                printNode->children = std::move(args);
                return printNode;
            }

            auto callNode = std::make_shared<ASTNode>(ASTNodeType::FunctionCall, idTok.lexeme, ValueType::Int,
                                                      idTok.line, idTok.column);
            callNode->children = std::move(args);
            return callNode;
        }
        return makeVariable(idTok);
    }
    if (match(TokenType::Separator, "(")) {
        auto inner = assignment();
        consume(TokenType::Separator, ")", "缺少 ')'");
        return inner;
    }
    throw std::runtime_error(errorAt(peek(), "无法解析的表达式"));
}
