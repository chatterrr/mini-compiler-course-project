# Mini Compiler Course Project

## 一、项目简介

本项目是一个用于教学目的的简化编译器课程设计，实现了一个简易类 C 语言的完整编译流程。  
编译器采用 C++ 实现核心功能，并配套提供基于 Python（Tkinter）的桌面图形界面，用于交互式编译、运行及结果展示。

项目完整覆盖了编译原理课程中的主要阶段，包括：

- 词法分析（Lexical Analysis）
- 语法分析（Syntax Analysis）
- 语义分析（Semantic Analysis）
- 中间代码生成（IR, Three-Address Code）
- 汇编代码生成（Pseudo Assembly）
- 中间代码执行（IR Virtual Machine）
- AST 图形化展示（Graphviz）

---

## 二、项目结构说明

```text
Compiler_co/
│
├── src/
│   ├── main.cpp          # 编译器核心实现（Lexer / Parser / IR / ASM）
│   └── compiler.exe      # 编译生成的可执行文件
│
├── examples/
│   └── demo.txt          # 示例源程序
│
├── ui_desktop/
│   └── ui.py             # 桌面图形界面（Tkinter）
│
├── ast.dot               # AST 的 Graphviz 描述文件（可选生成）
└── README.md

## 三、支持的语言特性
当前支持的语言特性包括：
 - 变量赋值语句，例：x = 1 + 2;
 - 变算术表达式，例：a = (1 + 2) * 3;
 - 条件与循环语句，例：if (a > 0) { ... }、while (i < 10) { ... }
 - 内建过程调用，例：print(x);
错误检测：
 - 变量未定义
 - 重复定义（若开启）
 - 语法错误提示（含行号）

## 四、编译与运行方式
 - 编译生成compiler.exe（若需重新编译），使用g++：g++ -std=c++17 -O2 -o .\src\compiler.exe .\src\main.cpp
 - 命令行运行编译器：.\src\compiler.exe .\examples\demo.txt
 - 若要执行中间代码：.\src\compiler.exe .\examples\demo.txt --run
 - 编译器支持到处AST的Graphviz描述文件：.\src\compiler.exe .\examples\demo.txt --ast-dot ast.dot
生成的ast.dot可通过在线工具渲染：https://edotor.net/

##五、桌面图形界面UI
项目提供基于Python Tkinter的桌面UI，支持：
 - 源代码编辑
 - 一键编译/运行
 - 分标签页展示（TOKENS、AST、SYMTAB、IR、ASM、ERRORS）
 - 一键导出ast.dot
启动UI，在项目根目录运行：python .\ui_desktop\ui.py
