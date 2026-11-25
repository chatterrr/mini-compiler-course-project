#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/asm_generator.h"
#include "backend/tac_vm.h"
#include "common/file_io.h"
#include "frontend/lexer.h"
#include "frontend/parser.h"
#include "frontend/semantic_analyzer.h"
#include "frontend/symbol_table.h"
#include "ir/ir_generator.h"
#include "ir/ir_printer.h"
#include "output/ast_dot.h"
#include "output/ast_printer.h"
#include "output/symtab_printer.h"
#include "output/token_printer.h"

// 主函数
int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "用法: compiler <源代码文件> [--run] [--ast-dot <dot文件路径>]\n";
            std::cerr << "选项:\n";
            std::cerr << "  --run        运行TAC虚拟机\n";
            std::cerr << "  --ast-dot    生成AST的Graphviz DOT文件\n";
            return 1;
        }

        bool runVm = false;
        std::string filePath;
        std::string astDotPath;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--run") {
                runVm = true;
            } else if (arg == "--ast-dot") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("参数 --ast-dot 需要指定输出文件路径");
                }
                astDotPath = argv[++i];
            } else if (arg.find("--") == 0) {
                throw std::runtime_error("未知参数: " + arg);
            } else {
                filePath = arg;
            }
        }

        if (filePath.empty()) {
            throw std::runtime_error("缺少源代码文件路径");
        }

        // 1. 读取源文件
        std::string source = readFile(filePath);

        // 2. 词法分析
        Lexer lexer(source);
        auto tokens = lexer.tokenize();
        printTokens(tokens, std::cout);

        // 3. 语法分析：Token -> AST
        Parser parser(tokens);
        parser.parseProgram();
        auto astRoot = parser.getASTRoot();

        // 4. 语义分析：符号表、类型检查、隐式转换、AST 类型标注
        SymbolTable symbols;
        SemanticAnalyzer semantic(symbols);
        semantic.analyze(astRoot);

        // 5. 输出AST
        std::cout << "\n=== AST ===\n";
        printAST(astRoot, std::cout);

        // 生成AST的DOT文件
        if (!astDotPath.empty()) {
            std::ofstream dotFile(astDotPath);
            if (!dotFile) {
                throw std::runtime_error("无法创建DOT文件: " + astDotPath);
            }
            dotFile << astToDot(astRoot);
            dotFile.close();
            std::cout << "(AST DOT文件已保存到: " << astDotPath << ")\n";
        }

        // 6. 输出符号表
        std::cout << "\n=== SYMTAB ===\n";
        printSymbolTable(symbols, std::cout);

        // 7. 中间代码生成：带类型 AST -> TAC
        IrGenerator irGen(parser.getStrings().getTable());
        TacProgram program = irGen.generate(astRoot);

        // 8. 输出IR（TAC）
        const auto& tacCode = program.code;
        std::cout << "\n=== IR ===\n";
        printIR(tacCode, std::cout);

        // 9. 生成汇编代码
        std::cout << "\n=== ASM ===\n";
        AssemblyGenerator asmGen(tacCode);
        std::cout << asmGen.generate();

        std::cout << "\n编译成功！\n";

        // 10. 运行TAC虚拟机
        if (runVm) {
            std::cout << "\n=== 虚拟机运行结果 ===\n";
            TacVM vm(tacCode, program.stringTable);
            double retVal = 0.0;
            bool hasReturn = vm.run(retVal);
            if (hasReturn) {
                std::cout << "\n程序返回值: " << retVal << "\n";
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "\n编译失败: " + std::string(e.what()) + "\n";
        return 1;
    }
    return 0;
}
