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

编译器源码已按功能拆分为多个模块：

```text
MiniCompiler/
│
├── CMakeLists.txt              # CMake 构建配置（可选）
│
├── src/                        # 编译器核心实现
│   ├── main.cpp                # CLI 入口与编译流水线编排
│   │
│   ├── common/                 # 通用基础设施
│   │   ├── types.h             # ValueType 等基础类型
│   │   └── file_io.h/.cpp      # 源文件读取
│   │
│   ├── frontend/               # 前端
│   │   ├── token.h             # TokenType / Token
│   │   ├── lexer.h/.cpp        # 词法分析
│   │   ├── ast.h               # ASTNodeType / ASTNode
│   │   ├── parser.h/.cpp       # 语法分析：Token -> AST
│   │   ├── string_interner.h/.cpp   # 字符串常量编号
│   │   ├── semantic_analyzer.h/.cpp # 语义分析：符号/类型检查
│   │   └── symbol_table.h/.cpp # 符号表与作用域
│   │
│   ├── ir/                     # 中间表示
│   │   ├── tac.h               # Tac / TacProgram
│   │   ├── ir_builder.h/.cpp   # 临时量、标签分配
│   │   ├── ir_generator.h/.cpp # 带类型 AST -> TAC
│   │   └── ir_printer.h/.cpp   # IR 文本输出
│   │
│   ├── backend/                # 后端
│   │   ├── tac_vm.h/.cpp       # TAC 虚拟机
│   │   └── asm_generator.h/.cpp # TAC -> 类 MIPS 伪汇编
│   │
│   ├── output/                 # 结果展示
│   │   ├── token_printer.h/.cpp
│   │   ├── ast_printer.h/.cpp
│   │   ├── ast_dot.h/.cpp
│   │   └── symtab_printer.h/.cpp
│   │
│   └── compiler.exe            # 编译后生成的编译器可执行文件（构建产物）
│
├── examples/                   # 示例代码目录
│   └── demo.txt                # 全功能测试用例（覆盖所有语言特性）
│
├── ui_desktop/                 # 图形化界面目录
│   └── ui.py                   # Tkinter界面（源代码编辑、编译结果分标签展示）
│
├── ast.dot                     # 可选生成：AST的Graphviz描述文件
└── README.md                   # 项目说明文档
```

### 模块职责

| 模块 | 职责 | 主要输入 | 主要输出 |
|---|---|---|---|
| `frontend/lexer` | 源码字符流 -> Token 序列 | 源码字符串 | `std::vector<Token>` |
| `frontend/parser` | Token -> AST，只负责语法结构 | Token 序列 | AST |
| `frontend/string_interner` | 为字符串字面量分配 `s0`、`s1` ... 编号 | 字符串字面量 | 字符串常量表 |
| `frontend/semantic_analyzer` | 符号表、作用域、类型检查、隐式转换、AST 类型标注 | AST + 符号表 | 带类型 AST / 语义错误 |
| `frontend/symbol_table` | 作用域管理、声明/查询、重复声明检查 | 声明/查询请求 | 变量类型与 const 信息 |
| `ir/ir_builder` | 临时量与标签分配 | 类型 | TAC 指令 |
| `ir/ir_generator` | 遍历带类型 AST，生成 TAC 与字符串表 | 带类型 AST | `TacProgram` |
| `ir/ir_printer` | TAC 指令文本格式化 | `std::vector<Tac>` | IR 文本 |
| `backend/tac_vm` | 解释执行 TAC | TAC + 字符串表 | 程序输出/返回值 |
| `backend/asm_generator` | TAC -> 类 MIPS 伪汇编 | TAC | 汇编文本 |
| `output/*` | TOKENS / AST / DOT / SYMTAB 展示 | 对应数据结构 | 文本 |
| `main.cpp` | 参数解析与各阶段串联 | 命令行参数 | 进程输出/退出码 |

> 编译流水线：`Lexer -> Parser -> SemanticAnalyzer -> IrGenerator -> IR/ASM/VM`。
> Parser 只负责语法结构；符号检查、类型检查和隐式转换集中在 `SemanticAnalyzer`；TAC 生成集中在 `IrGenerator`。

## 三、支持的语言特性
1. 数据类型与声明
基础类型：int/float/bool/char
常量声明：const（支持类型检查、禁止赋值）
块作用域：大括号内变量的局部作用域

2. 表达式与运算符
算术运算符：+ - * / %
复合赋值：+= -= *= /= %=
自增自减：++（前缀/后缀）、--（前缀/后缀）
关系运算符：> < == != >= <=
逻辑运算符：&& / || / !（支持短路求值）
三元运算符：?:（条件分支表达式）

3. 控制流语句
分支：if-else（支持嵌套）
循环：while/do-while/for
跳转：break（循环/switch 内）、continue（循环内）
多分支：switch-case（支持default分支）

4. 内置函数与字面量
内置函数：print（支持多参数、字符串/数值/字符输出）
字面量：整数/浮点数/布尔值（true/false）/字符（'A'）/字符串（"Hello"）

5. 错误检测
词法错误：未闭合的字符/字符串常量、非法字符
语法错误：缺少分号/括号、语法结构不合法
语义错误：未声明标识符、重复声明、常量赋值、类型不匹配

## 四、编译与运行方式

### 方式一：直接使用 g++（MinGW / MSYS2）

在项目根目录执行：

```powershell
g++ -std=c++17 -O2 -static -Isrc -o .\src\compiler.exe (Get-ChildItem -Recurse .\src -Filter *.cpp | ForEach-Object { $_.FullName })
```

如果你使用 Git Bash，也可以写成：

```bash
g++ -std=c++17 -O2 -static -Isrc -o src/compiler.exe $(find src -name '*.cpp')
```

> `-static` 用于把 C++ 运行库静态链接进 `compiler.exe`。在 MinGW/MSYS2 环境中，如果 PATH 里存在多个版本的 `libstdc++-6.dll`，动态链接可能导致程序启动或退出时崩溃，静态链接可以避免该类问题。使用 MSVC 或 CMake 时无需手动添加该参数（CMake 已对 MinGW 自动处理）。

### 方式二：CMake（可选）

如果本机已安装 CMake 和可用的 C++ 工具链（MinGW、MSVC 或 Ninja），可以执行：

```powershell
cmake -S . -B build
cmake --build build --config Release
```

构建产物会输出到 `src/compiler.exe`，与图形界面使用的路径保持一致。

> MSVC 构建时 CMake 会自动添加 `/utf-8`，避免源码中的中文被按系统代码页（如 GBK）解析，从而出现 `std::runtime_error` 默认构造函数、引用未初始化等看似奇怪的编译错误。

### 运行编译器

- 命令行运行编译器：`.\src\compiler.exe .\examples\demo.txt`
- 执行中间代码：`.\src\compiler.exe .\examples\demo.txt --run`
- 导出 AST 的 Graphviz 描述文件：`.\src\compiler.exe .\examples\demo.txt --ast-dot ast.dot`

生成的 `ast.dot` 可通过在线工具渲染：https://edotor.net/
将 `ast.dot` 内容粘贴后点击 “Render” 即可查看 AST 图形。

## 五、图形化界面 UI

1. 启动 UI
在项目根目录执行：`python .\ui_desktop\ui.py`

2. UI 功能
- 源代码编辑：支持语法高亮（类C关键字/注释/字符串等）、撤销/重做
- 一键操作：编译/运行、清空输出、打开/保存源代码文件
- 结果展示：分标签页展示编译各阶段结果（TOKENS/AST/SYMTAB/IR/ASM/RUN/ERRORS）
- 辅助功能：加载预设示例代码、一键导出 ast.dot 文件

UI 通过 `src/compiler.exe` 调用编译器，并依赖以下标准输出分段标题：

```text
=== TOKENS ===
=== AST ===
=== SYMTAB ===
=== IR ===
=== ASM ===
=== 虚拟机运行结果 ===
```

拆分源码时已保持这些输出格式不变，因此 `ui_desktop/ui.py` 无需修改。

## 六、编译流水线与阶段划分

源码拆分后，编译过程按以下阶段依次执行：

1. **词法分析（`frontend/lexer`）**
   源码字符流 -> `std::vector<Token>`，字符串字面量由 `frontend/string_interner` 分配 `s0`、`s1` ... 编号。

2. **语法分析（`frontend/parser`）**
   Token 序列 -> AST。该阶段只检查语法结构，不查询符号表、不做类型检查、不生成 TAC。

3. **语义分析（`frontend/semantic_analyzer`）**
   维护作用域与符号表，检查未声明/重复声明/常量赋值等语义规则，完成类型推导和隐式转换，并把类型信息写回 AST。需要转换的表达式会插入 `cast` 节点，便于后续统一生成转换指令。

4. **中间代码生成（`ir/ir_generator`）**
   遍历带类型信息的 AST，使用 `ir/ir_builder` 分配临时量和标签，生成三地址码 `TacProgram`。

5. **后端与输出（`backend/*`、`output/*`）**
   `backend/asm_generator` 生成类 MIPS 伪汇编，`backend/tac_vm` 解释执行 TAC；`output/*` 负责 TOKENS、AST、DOT、SYMTAB 和 IR 的文本展示。

错误信息按阶段区分：

- 词法错误：未闭合字符/字符串、非法字符；
- 语法错误：以 `语法错误 (line ..., col ...)` 开头；
- 语义错误：以 `语义错误 (line ..., col ...)` 开头，例如未声明标识符、重复声明、常量赋值、类型不匹配。

对于现有示例和常规用例（字面量 `case`、类型转换、复合赋值、自增自减、三元表达式等），拆分前后的 TOKENS / AST / SYMTAB / IR / ASM / 虚拟机运行结果保持一致；`ui_desktop/ui.py` 仍可直接使用。

拆分后还修正了原实现对 `case` 表达式重复求值、并按临时量字符串匹配 case 标签的问题。此前 `case 1 + 1:` 这类写法会生成空标签并在虚拟机阶段报 `未定义的标签`，现在会正常匹配并执行对应分支。语义错误现在由 `SemanticAnalyzer` 报告，前缀统一为 `语义错误`，部分类型错误的行列定位比拆分前更精确。

> 不同编译器/标准库对 `std::unordered_set` 的迭代顺序可能不同，因此汇编输出 `.data` 段中变量的排列顺序在 MSVC 与 MinGW 构建之间可能略有差异；变量集合、指令和运行结果本身一致。
