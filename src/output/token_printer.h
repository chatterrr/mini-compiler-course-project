#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include "frontend/token.h"

std::string tokenTypeName(TokenType t);

// 输出 === TOKENS === 段
void printTokens(const std::vector<Token>& tokens, std::ostream& os);
