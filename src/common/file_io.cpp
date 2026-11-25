#include "common/file_io.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

std::string readFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("无法打开文件: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
