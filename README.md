# Mini Compiler Course Project

## 一、项目简介
本项目是一个面向编译原理教学的迷你类C语言编译器，完整实现了从源代码到可执行中间代码的全流程，同时配套了基于Python Tkinter的图形化界面，支持交互式编译、运行及结果可视化。

项目覆盖编译原理核心阶段：

- 词法分析（生成Token序列）
- 语法分析（构建抽象语法树AST）
- 语义分析（符号表管理、类型检查）
- 中间代码生成（三地址码TAC）
- 汇编代码生成（类MIPS伪汇编）
- 虚拟机执行（TAC解释器）
- AST 可视化（Graphviz DOT文件生成）

## 二、项目结构说明
text
MiniCompiler/
│
├── src/                     # 编译器核心实现目录
│   ├── main.cpp             # 编译器源码（Lexer/Parser/SymbolTable/IR/ASM/VM）
│   └── compiler.exe         # 编译后生成的编译器可执行文件
│
├── examples/                # 示例代码目录
│   └── demo.txt             # 全功能测试用例（覆盖所有语言特性）
│
├── ui_desktop/              # 图形化界面目录
│   └── ui.py                # Tkinter界面（源代码编辑、编译结果分标签展示）
│
├── ast.dot                  # 可选生成：AST的Graphviz描述文件
└── README.md                # 项目说明文档

## 三、支持的语言特性
1. 数据类型与声明
基础类型：int/float/bool/char
常量声明：const（支持类型检查、禁止赋值）
块作用域：大括号内变量的局部作用域

2. 表达式与运算符
算术运算符：+/-/*///%
复合赋值：+=/-=/*=//=/%=
自增自减：++（前缀/后缀）、--（前缀/后缀）
关系运算符：>/</==/!=/>=/<=
逻辑运算符：&&/||/!（支持短路求值）
三元运算符：?:（条件分支表达式）

3. 控制流语句
分支：if-else（支持嵌套）
循环：while/do-while/for
跳转：break/continue（仅在循环/switch 内生效）
多分支：switch-case（支持default分支）

4. 内置函数与字面量
内置函数：print（支持多参数、字符串/数值/字符输出）
字面量：整数/浮点数/布尔值（true/false）/字符（'A'）/字符串（"Hello"）

5. 错误检测
词法错误：未闭合的字符/字符串常量、非法字符
语法错误：缺少分号/括号、语法结构不合法
语义错误：未声明标识符、重复声明、常量赋值、类型不匹配

## 四、编译与运行方式

- 编译生成compiler.exe（若需重新编译），使用g++：g++ -std=c++17 -O2 -o .\src\compiler.exe .\src\main.cpp
 - 命令行运行编译器：.\src\compiler.exe .\examples\demo.txt
 - 若要执行中间代码：.\src\compiler.exe .\examples\demo.txt --run
 - 编译器支持到处AST的Graphviz描述文件：.\src\compiler.exe .\examples\demo.txt --ast-dot ast.dot
生成的ast.dot可通过在线工具渲染：https://edotor.net/ 
将ast.dot内容粘贴后点击 “Render” 即可查看 AST 图形。

## 五、图形化界面 UI
1. 启动 UI
在项目根目录执行：python ./ui_desktop/ui.py
2. UI 功能
- 源代码编辑：支持语法高亮（类C关键字/注释/字符串等）、撤销/重做
- 一键操作：编译/运行、清空输出、打开/保存源代码文件
- 结果展示：分标签页展示编译各阶段结果（TOKENS/AST/SYMTAB/IR/ASM/RUN/ERRORS）
- 辅助功能：加载预设示例代码、一键导出ast.dot文件