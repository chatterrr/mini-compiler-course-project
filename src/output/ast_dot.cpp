#include "output/ast_dot.h"

#include <ostream>
#include <sstream>
#include <unordered_map>

#include "output/ast_printer.h"

namespace {

void astDotDfs(const std::shared_ptr<ASTNode>& node, std::ostream& os, int& nextId,
               std::unordered_map<const ASTNode*, int>& ids) {
    if (!node) return;
    int myId = nextId++;
    ids[node.get()] = myId;

    auto esc = [](const std::string& s) {
        std::string r;
        for (char c : s) {
            if (c == '"' || c == '\\') r.push_back('\\');
            r.push_back(c);
        }
        return r;
    };

    os << "  n" << myId << " [label=\""
       << esc(nodeTypeName(node->type)) << "\\n"
       << esc(node->value) << "\\n"
       << esc(valueTypeName(node->dataType)) << "\\n"
       << node->line << ":" << node->column
       << "\"];\n";

    for (const auto& ch : node->children) {
        if (!ch) continue;
        astDotDfs(ch, os, nextId, ids);
        int cid = ids[ch.get()];
        os << "  n" << myId << " -> n" << cid << ";\n";
    }
}

} // namespace

std::string astToDot(const std::shared_ptr<ASTNode>& root) {
    std::ostringstream os;
    os << "digraph AST {\n";
    os << "  node [shape=box, fontname=\"Consolas\"];\n";
    int nextId = 0;
    std::unordered_map<const ASTNode*, int> ids;
    astDotDfs(root, os, nextId, ids);
    os << "}\n";
    return os.str();
}
