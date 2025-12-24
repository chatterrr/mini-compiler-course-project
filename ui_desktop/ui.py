import sys
import os
import subprocess
import tempfile
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import re

COMPILER_PATH = os.path.abspath(r".\src\compiler.exe")
COMPILE_TIMEOUT = 10
SRC_SUFFIX = ".c"


def build_cmd(src_path: str, run: bool, ast_dot_path: str = None):
    cmd = [COMPILER_PATH, src_path]
    if run:
        cmd.append("--run")
    if ast_dot_path:
        cmd.extend(["--ast-dot", ast_dot_path])
    return cmd


SECTION_MARKERS = [
    "=== TOKENS ===",
    "=== AST ===",
    "=== SYMTAB ===",
    "=== IR ===",
    "=== ASM ===",
]


def split_sections(text: str):
    idx = [(m, text.find(m)) for m in SECTION_MARKERS]
    present = [(m, i) for (m, i) in idx if i != -1]
    present.sort(key=lambda x: x[1])

    if not present:
        return {}, text.strip()

    ranges = []
    sections = {}
    for k, (m, start) in enumerate(present):
        end = present[k + 1][1] if k + 1 < len(present) else len(text)
        ranges.append((start, end))
        sections[m] = text[start:end].strip()

    parts = []
    last = 0
    for (s, e) in ranges:
        if s > last:
            parts.append(text[last:s])
        last = e
    if last < len(text):
        parts.append(text[last:])

    remainder = "".join(parts).strip()
    return sections, remainder


def make_readonly(text_widget: tk.Text):
    text_widget.config(state=tk.DISABLED)
    text_widget.bind("<Button-1>", lambda e: text_widget.focus_set())
    text_widget.bind("<Control-c>", lambda e: text_widget.event_generate("<<Copy>>"))
    text_widget.bind("<Control-v>", lambda e: "break")


def add_scrollbars(parent, text_widget: tk.Text):
    v_scroll = tk.Scrollbar(parent, orient=tk.VERTICAL, command=text_widget.yview)
    h_scroll = tk.Scrollbar(parent, orient=tk.HORIZONTAL, command=text_widget.xview)
    text_widget.config(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
    
    text_widget.grid(row=0, column=0, sticky="nsew")
    v_scroll.grid(row=0, column=1, sticky="ns")
    h_scroll.grid(row=1, column=0, sticky="ew")
    
    parent.grid_rowconfigure(0, weight=1)
    parent.grid_columnconfigure(0, weight=1)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mini Compiler GUI")
        self.geometry("1400x900")
        
        self.compiling = False
        
        self._build_widgets()
        self._load_demo()

    def _build_widgets(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        
        self.run_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="编译后运行 (--run)", variable=self.run_var).pack(side=tk.LEFT)
        
        ttk.Button(top, text="编译/运行", command=self.on_compile).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="保存 ast.dot", command=self.on_save_ast_dot).pack(side=tk.LEFT)
        ttk.Button(top, text="清空输出", command=self.clear_output).pack(side=tk.LEFT)
        ttk.Button(top, text="打开文件", command=self.on_open_file).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="保存文件", command=self.on_save_file).pack(side=tk.LEFT)
        
        ttk.Label(top, text="示例:").pack(side=tk.LEFT, padx=(20, 5))
        self.example_var = tk.StringVar()
        example_combo = ttk.Combobox(
            top, 
            textvariable=self.example_var, 
            values=list(self._get_examples().keys()), 
            width=20, 
            state="readonly"
        )
        example_combo.pack(side=tk.LEFT)
        example_combo.bind("<<ComboboxSelected>>", self.on_example_selected)
        example_combo.current(0)
        
        self.status = ttk.Label(top, text="就绪")
        self.status.pack(side=tk.RIGHT)

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(body)
        body.add(left, weight=1)

        ttk.Label(left, text="源代码").pack(anchor="w")
        code_frame = ttk.Frame(left)
        code_frame.pack(fill=tk.BOTH, expand=True)
        
        self.code = tk.Text(
            code_frame, 
            wrap="none", 
            font=("Consolas", 11),
            undo=True,
            maxundo=100
        )
        self._init_code_editor()
        add_scrollbars(code_frame, self.code)

        right = ttk.Frame(body)
        body.add(right, weight=1)

        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill=tk.BOTH, expand=True)

        self.out_widgets = {}
        tab_names = [
            ("TOKENS", "tokens"),
            ("AST", "ast"),
            ("SYMTAB", "symtab"),
            ("IR", "ir"),
            ("ASM", "asm"),
            ("RUN", "run"),
            ("ALL", "all"),
            ("ERRORS", "errors")
        ]
        for tab_text, key in tab_names:
            frame = ttk.Frame(self.tabs)
            text_widget = tk.Text(frame, wrap="none", font=("Consolas", 11))
            add_scrollbars(frame, text_widget)
            make_readonly(text_widget)
            self.tabs.add(frame, text=tab_text)
            self.out_widgets[key] = text_widget

        self.out_tokens = self.out_widgets["tokens"]
        self.out_ast = self.out_widgets["ast"]
        self.out_symtab = self.out_widgets["symtab"]
        self.out_ir = self.out_widgets["ir"]
        self.out_asm = self.out_widgets["asm"]
        self.out_run = self.out_widgets["run"]
        self.out_all = self.out_widgets["all"]
        self.out_errors = self.out_widgets["errors"]

    def _init_code_editor(self):
        self.code.tag_config('keyword', foreground='#0000ff', font=('Consolas', 11))
        self.code.tag_config('comment', foreground='#008000', font=('Consolas', 11, 'italic'))
        self.code.tag_config('string', foreground='#a31515', font=('Consolas', 11))
        self.code.tag_config('number', foreground='#098658', font=('Consolas', 11))
        self.code.tag_config('operator', foreground='#795e26', font=('Consolas', 11))
        self.code.tag_config('function', foreground='#267f99', font=('Consolas', 11))
        
        self.keywords = {'int', 'float', 'bool', 'char', 'const', 'if', 'else', 'while',
                        'for', 'do', 'switch', 'case', 'default', 'break', 'continue',
                        'return', 'true', 'false', 'print'}
        
        self.operators = {'=', '+', '-', '*', '/', '%', '==', '!=', '<', '>', '<=', '>=', 
                         '&&', '||', '!', '+=', '-=', '*=', '/=', '%=', '++', '--', '?:'}
        
        self.code.bind('<KeyRelease>', self._highlight_code)
        self.code.bind('<ButtonRelease-1>', self._highlight_code)

    def _highlight_code(self, event=None):
        for tag in ['keyword', 'comment', 'string', 'number', 'operator', 'function']:
            self.code.tag_remove(tag, '1.0', tk.END)
        
        text = self.code.get('1.0', tk.END)
        
        for keyword in self.keywords:
            start = '1.0'
            while True:
                start = self.code.search(r'\b' + re.escape(keyword) + r'\b', start, stopindex=tk.END, regexp=True)
                if not start:
                    break
                end = f"{start}+{len(keyword)}c"
                self.code.tag_add('keyword', start, end)
                start = end
        
        start = '1.0'
        while True:
            start = self.code.search(r'//.*', start, stopindex=tk.END, regexp=True)
            if not start:
                break
            end = f"{start} lineend"
            self.code.tag_add('comment', start, end)
            start = end
        
        start = '1.0'
        while True:
            start = self.code.search(r'".*?"', start, stopindex=tk.END, regexp=True)
            if not start:
                break
            end = f"{start}+1c"
            while True:
                ch = self.code.get(end)
                if ch == '"' and self.code.get(f"{end}-1c") != '\\':
                    break
                end = f"{end}+1c"
            end = f"{end}+1c"
            self.code.tag_add('string', start, end)
            start = end
        
        start = '1.0'
        while True:
            start = self.code.search(r'\b\d+(\.\d+)?\b', start, stopindex=tk.END, regexp=True)
            if not start:
                break
            line_start = self.code.index(f"{start} linestart")
            line_end = self.code.index(f"{start} lineend")
            line_text = self.code.get(line_start, line_end)
            
            col = int(start.split('.')[1])
            line_pos = col - 1
            
            match = re.match(r'\d+(\.\d+)?', line_text[line_pos:])
            if match:
                num_len = len(match.group())
                end = f"{start}+{num_len}c"
                self.code.tag_add('number', start, end)
                start = end
            else:
                start = f"{start}+1c"
        
        operators_patterns = [
            r'\+\+', r'--', r'\+=', r'-=', r'\*=', r'/=', r'%=',
            r'==', r'!=', r'<=', r'>=', r'&&', r'\|\|',
            r'=', r'\+', r'-', r'\*', r'/', r'%', r'<', r'>', r'!', r':', r'\?'
        ]
        
        for op_pattern in operators_patterns:
            start = '1.0'
            while True:
                start = self.code.search(op_pattern, start, stopindex=tk.END, regexp=True)
                if not start:
                    break
                line_start = self.code.index(f"{start} linestart")
                line_end = self.code.index(f"{start} lineend")
                line_text = self.code.get(line_start, line_end)
                col = int(start.split('.')[1])
                line_pos = col - 1
                
                match = re.match(op_pattern, line_text[line_pos:])
                if match:
                    op_len = len(match.group())
                    end = f"{start}+{op_len}c"
                    self.code.tag_add('operator', start, end)
                    start = end
                else:
                    start = f"{start}+1c"
        
        start = '1.0'
        while True:
            start = self.code.search(r'\bprint\b', start, stopindex=tk.END, regexp=True)
            if not start:
                break
            end = f"{start}+5c"
            self.code.tag_add('function', start, end)
            start = end

    def _get_examples(self):
        return {
            "基础示例": """// 基础示例
int a = 5;
int b = 3;
int sum = a + b;
print(sum);

if (a > b) {
    print("a is greater");
} else {
    print("b is greater");
}

for (int i = 0; i < 5; i = i + 1) {
    print(i);
}
""",
            "复合赋值": """// 复合赋值运算符示例
int a = 10;
a += 5;      // a = a + 5
print(a);    // 15

a -= 3;      // a = a - 3
print(a);    // 12

a *= 2;      // a = a * 2
print(a);    // 24

a /= 4;      // a = a / 4
print(a);    // 6

a %= 5;      // a = a % 5
print(a);    // 1
""",
            "自增自减": """// 自增自减运算符示例
int x = 5;
int y = 0;

y = x++;     // 后缀自增
print(x);    // 6
print(y);    // 5

y = ++x;     // 前缀自增
print(x);    // 7
print(y);    // 7

y = x--;     // 后缀自减
print(x);    // 6
print(y);    // 7

y = --x;     // 前缀自减
print(x);    // 5
print(y);    // 5
""",
            "三元条件运算符": """// 三元条件运算符示例
int a = 10;
int b = 20;

int max = (a > b) ? a : b;
print(max);  // 20

int score = 85;
char grade = (score >= 90) ? 'A' : (score >= 80) ? 'B' : (score >= 70) ? 'C' : 'D';
print(grade);  // B

int c = 5;
int d = 8;
int e = (c > d) ? (c += 10) : (d += 10);
print(c);  // 5
print(d);  // 18
print(e);  // 18
""",
            "混合运算": """// 混合运算符示例
int a = 10;
int b = 20;

a += ++b;    // b先自增，然后a = a + b
print(a);    // 31
print(b);    // 21

int c = 5;
int d = (c > 0) ? (c *= 2) : 0;
print(c);    // 10
print(d);    // 10

int result = (a > b) ? (a += b) : (b += a);
result += result > 50 ? 100 : 50;
print(result);  // 173
""",
            "循环控制": """// 循环控制示例
int i = 0;
while (i < 5) {
    print(i);
    i++;
}

int j = 0;
do {
    print(j);
    j++;
} while (j < 3);

for (int k = 0; k < 5; k++) {
    if (k == 2) {
        continue;
    }
    if (k == 4) {
        break;
    }
    print(k);
}

int sum = 0;
for (int n = 1; n <= 10; n++) {
    sum += n;
}
print(sum);  // 55
""",
            "switch语句": """// switch语句示例
int day = 3;

switch (day) {
    case 1:
        print("Monday");
        break;
    case 2:
        print("Tuesday");
        break;
    case 3:
        print("Wednesday");
        break;
    case 4:
        print("Thursday");
        break;
    case 5:
        print("Friday");
        break;
    default:
        print("Weekend");
        break;
}

int x = 1;
switch (x++) {
    case 1:
        print("First");
        x *= 2;
        break;
    case 2:
        print("Second");
        x += 10;
        break;
}
print(x);  // 3
"""
        }

    def _load_demo(self):
        examples = self._get_examples()
        first_example = list(examples.values())[0]
        self.code.delete("1.0", tk.END)
        self.code.insert("1.0", first_example)
        self._highlight_code()

    def on_example_selected(self, event=None):
        examples = self._get_examples()
        example = self.example_var.get()
        if example in examples:
            self.code.delete("1.0", tk.END)
            self.code.insert("1.0", examples[example])
            self._highlight_code()
            self.status.config(text=f"已加载 {example}")

    def _clear_text(self, w: tk.Text):
        w.config(state=tk.NORMAL)
        w.delete("1.0", tk.END)
        w.config(state=tk.DISABLED)

    def clear_output(self):
        for w in self.out_widgets.values():
            self._clear_text(w)
        self.status.config(text="已清空输出")

    def _insert_text(self, w: tk.Text, text: str):
        if not text:
            return
        w.config(state=tk.NORMAL)
        w.insert(tk.END, text)
        w.config(state=tk.DISABLED)
        w.see(tk.END)

    def _run_compiler(self, cmd):
        try:
            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=COMPILE_TIMEOUT,
                cwd=os.path.dirname(COMPILER_PATH) or None,
            )
            return p
        except FileNotFoundError:
            raise Exception(f"编译器未找到: {COMPILER_PATH}")
        except subprocess.TimeoutExpired as e:
            raise Exception(f"编译超时 (>{COMPILE_TIMEOUT}秒)")
        except Exception as e:
            raise Exception(f"执行错误: {str(e)}")

    def _compile_worker(self, src, run, ast_dot_path=None):
        result = {}
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                src_path = os.path.join(temp_dir, f"source{SRC_SUFFIX}")
                with open(src_path, "w", encoding="utf-8") as f:
                    f.write(src)

                cmd = build_cmd(src_path, run, ast_dot_path)

                p = self._run_compiler(cmd)

            combined = (p.stdout or "") + ("\n" if (p.stdout and p.stderr) else "") + (p.stderr or "")
            sections, remainder = split_sections(combined)
            result = {
                "sections": sections,
                "remainder": remainder,
                "returncode": p.returncode,
                "stderr": p.stderr,
                "combined": combined,
                "ast_dot_path": ast_dot_path,
                "success": True
            }
        except subprocess.TimeoutExpired:
            result = {"error": f"超时: 编译/运行超时（>{COMPILE_TIMEOUT}s）", "success": False}
        except Exception as e:
            result = {"error": f"异常: {str(e)}", "success": False}

        self.after(0, self._update_compile_result, result)

    def _update_compile_result(self, result):
        self.clear_output()

        if not result["success"]:
            self._insert_text(self.out_errors, result["error"])
            self.status.config(text="异常", foreground="red")
            self.tabs.select(self.out_widgets["errors"])
            return

        sections = result["sections"]
        remainder = result["remainder"]
        returncode = result["returncode"]
        stderr = result["stderr"]
        combined = result["combined"]
        ast_dot_path = result.get("ast_dot_path")

        self._insert_text(self.out_tokens, sections.get("=== TOKENS ===", ""))
        self._insert_text(self.out_ast, sections.get("=== AST ===", ""))
        self._insert_text(self.out_symtab, sections.get("=== SYMTAB ===", ""))
        self._insert_text(self.out_ir, sections.get("=== IR ===", ""))
        self._insert_text(self.out_asm, sections.get("=== ASM ===", ""))

        run_text = ""
        lines = combined.split('\n')
        in_run = False
        for line in lines:
            if line.startswith("=== 虚拟机运行结果 ==="):
                in_run = True
                continue
            if in_run and line.strip():
                run_text += line + "\n"
        
        if run_text:
            self._insert_text(self.out_run, run_text.strip())
        
        other_text = ""
        if remainder:
            other_text += remainder
        self._insert_text(self.out_all, combined.strip())

        if returncode != 0 or (stderr and stderr.strip()):
            err = stderr.strip() if (stderr and stderr.strip()) else combined.strip()
            self._insert_text(self.out_errors, err)
            self.status.config(text=f"失败（code={returncode}）", foreground="red")
            self.tabs.select(self.out_widgets["errors"])
        else:
            if ast_dot_path:
                if os.path.exists(ast_dot_path) and os.path.getsize(ast_dot_path) > 0:
                    self.status.config(text="已保存 ast.dot", foreground="green")
                    messagebox.showinfo("成功", f"ast.dot 已保存到：\n{ast_dot_path}")
                    if sections.get("=== AST ==="):
                        self.tabs.select(self.out_widgets["ast"])
                    else:
                        self.tabs.select(self.out_widgets["all"])
                else:
                    self.status.config(text="保存失败", foreground="red")
                    messagebox.showerror(
                        "生成失败",
                        f"未能生成 ast.dot（返回码={returncode}）。\n\n输出：\n{combined.strip()}"
                    )
                    self.tabs.select(self.out_widgets["errors"])
            else:
                self.status.config(text="成功", foreground="green")
                if run_text:
                    self.tabs.select(self.out_widgets["run"])
                elif sections.get("=== IR ==="):
                    self.tabs.select(self.out_widgets["ir"])
                elif sections.get("=== ASM ==="):
                    self.tabs.select(self.out_widgets["asm"])
                else:
                    self.tabs.select(self.out_widgets["all"])

    def on_compile(self):
        if self.compiling:
            messagebox.showwarning("请等待", "编译正在进行中，请稍候...")
            return
            
        global COMPILER_PATH
        
        if not os.path.exists(COMPILER_PATH):
            potential_paths = [
                os.path.abspath("./compiler.exe"),
                os.path.abspath("../src/compiler.exe"),
                os.path.abspath("compiler.exe"),
                os.path.join(os.path.dirname(__file__), "compiler.exe")
            ]
            
            found = False
            for path in potential_paths:
                if os.path.exists(path):
                    COMPILER_PATH = path
                    found = True
                    messagebox.showinfo("提示", f"自动找到编译器: {path}")
                    break
            
            if not found:
                messagebox.showerror("找不到编译器", 
                    f"未找到编译器程序: {COMPILER_PATH}\n"
                    "请将编译器放在同一目录下。")
                return
        
        src = self.code.get("1.0", tk.END).strip()
        if not src:
            messagebox.showwarning("空输入", "请输入源代码")
            return
        
        self.compiling = True
        self.status.config(text="处理中...", foreground="blue")
        self.update_idletasks()
        
        thread = threading.Thread(
            target=self._compile_worker, 
            args=(src, self.run_var.get())
        )
        thread.daemon = True
        thread.start()
        
        self._check_thread(thread)

    def _check_thread(self, thread):
        if thread.is_alive():
            self.after(100, lambda: self._check_thread(thread))
        else:
            self.compiling = False

    def on_save_ast_dot(self):
        global COMPILER_PATH
        
        if not os.path.exists(COMPILER_PATH):
            messagebox.showerror("找不到编译器", f"未找到: {COMPILER_PATH}")
            return

        src = self.code.get("1.0", tk.END).strip()
        if not src:
            messagebox.showwarning("空输入", "请输入源代码")
            return

        save_path = filedialog.asksaveasfilename(
            title="保存 AST DOT 文件",
            defaultextension=".dot",
            filetypes=[("Graphviz DOT", "*.dot"), ("All Files", "*.*")],
            initialfile="ast.dot",
        )
        if not save_path:
            return

        self.status.config(text="生成 ast.dot...", foreground="blue")
        self.update_idletasks()
        thread = threading.Thread(target=self._compile_worker, 
                                  args=(src, False, save_path))
        thread.daemon = True
        thread.start()

    def on_open_file(self):
        file_path = filedialog.askopenfilename(
            title="打开源代码文件",
            filetypes=[("C Files", "*.c"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    src = f.read()
                self.code.delete("1.0", tk.END)
                self.code.insert("1.0", src)
                self._highlight_code()
                self.status.config(text=f"已打开：{os.path.basename(file_path)}", foreground="green")
            except Exception as e:
                messagebox.showerror("打开失败", f"无法打开文件：{str(e)}")
                self.status.config(text="打开失败", foreground="red")

    def on_save_file(self):
        src = self.code.get("1.0", tk.END).strip()
        if not src:
            messagebox.showwarning("空内容", "没有可保存的源代码")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存源代码文件",
            defaultextension=SRC_SUFFIX,
            filetypes=[("C Files", "*.c"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(src)
                self.status.config(text=f"已保存：{os.path.basename(file_path)}", foreground="green")
            except Exception as e:
                messagebox.showerror("保存失败", f"无法保存文件：{str(e)}")
                self.status.config(text="保存失败", foreground="red")

if __name__ == "__main__":
    app = App()
    app.mainloop()