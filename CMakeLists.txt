cmake_minimum_required(VERSION 3.16)

project(MiniCompiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(compiler
    src/main.cpp
    src/common/file_io.cpp
    src/frontend/lexer.cpp
    src/frontend/parser.cpp
    src/frontend/semantic_analyzer.cpp
    src/frontend/symbol_table.cpp
    src/frontend/string_interner.cpp
    src/ir/ir_builder.cpp
    src/ir/ir_generator.cpp
    src/ir/ir_printer.cpp
    src/backend/tac_vm.cpp
    src/backend/asm_generator.cpp
    src/output/token_printer.cpp
    src/output/ast_printer.cpp
    src/output/ast_dot.cpp
    src/output/symtab_printer.cpp
)

target_include_directories(compiler PRIVATE src)

if(MSVC)
    # 源码和字符串字面量均为 UTF-8，必须显式告诉 MSVC，否则中文会被按
    # 系统代码页（如 GBK）解析，导致字符串字面量吞掉后续代码。
    target_compile_options(compiler PRIVATE /W4 /utf-8)
else()
    target_compile_options(compiler PRIVATE -Wall -Wextra)
endif()

# MinGW: link the C++ runtime statically so the produced compiler.exe does not
# depend on whichever libstdc++-6.dll happens to be first in PATH.
if(MINGW)
    target_link_options(compiler PRIVATE -static)
endif()

# Keep the historical output location: ui_desktop/ui.py looks for src/compiler.exe.
set_target_properties(compiler PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src"
)
foreach(config ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER "${config}" CONFIG_UPPER)
    set_target_properties(compiler PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_SOURCE_DIR}/src"
    )
endforeach()
