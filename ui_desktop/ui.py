import os
import subprocess
import tempfile
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ====== 1) 修改这里：指向你的 compiler.exe ======
# 推荐用绝对路径最稳定
COMPILER_PATH = os.path.abspath(r".\src\compiler.exe")

# ====== 2) 修改这里：你们编译器的命令行参数 ======
# 默认: compiler.exe <sourcefile> --run
# 如果你们没有 --run，就改成 [COMPILER_PATH, src_path]
def build_cmd(src_path: str, run: bool):
    cmd = [COMPILER_PATH, src_path]
    if run:
        cmd.append("--run")
    return cmd


def build_cmd_with_ast_dot(src_path: str, run: bool, ast_dot_path: str):
    """在原有命令基础上追加 --ast-dot <path>（如果你的编译器支持该参数）。"""
    cmd = build_cmd(src_path, run)
    cmd.extend(["--ast-dot", ast_dot_path])
    return cmd

SECTION_MARKERS = [
    "=== TOKENS ===",
    "=== AST ===",
    "=== SYMTAB ===",
    "=== IR ===",
    "=== ASM ===",
    # 可选：如果你的编译器还有其它分段，可以继续加
    "=== RUN",
]

def split_sections(text: str):
    """
    按 SECTION_MARKERS 将输出切成多个段落。

    返回：
      sections: dict[marker -> content]（含 marker 行）
      remainder: 没有被任何 marker 覆盖到的内容
    """
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

    # 计算 remainder（把 ranges 之外的内容拼起来）
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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mini Compiler Desktop UI")
        self.geometry("1180x740")

        self._build_widgets()
        self._load_demo()

    def _build_widgets(self):
        # 顶部按钮区
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        self.run_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="编译后运行 (--run)", variable=self.run_var).pack(side=tk.LEFT)

        ttk.Button(top, text="编译/运行", command=self.on_compile).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="保存 ast.dot", command=self.on_save_ast_dot).pack(side=tk.LEFT)
        ttk.Button(top, text="清空输出", command=self.clear_output).pack(side=tk.LEFT)

        self.status = ttk.Label(top, text="就绪")
        self.status.pack(side=tk.RIGHT)

        # 主体左右分栏
        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 左：代码编辑区
        left = ttk.Frame(body)
        body.add(left, weight=1)

        ttk.Label(left, text="源代码").pack(anchor="w")
        self.code = tk.Text(left, wrap="none", font=("Consolas", 11))
        self.code.pack(fill=tk.BOTH, expand=True)

        # 右：输出标签页
        right = ttk.Frame(body)
        body.add(right, weight=1)

        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill=tk.BOTH, expand=True)

        # 分栏输出（课程设计验收友好）
        self.out_tokens = tk.Text(self.tabs, wrap="none", font=("Consolas", 11))
        self.out_ast = tk.Text(self.tabs, wrap="none", font=("Consolas", 11))
        self.out_symtab = tk.Text(self.tabs, wrap="none", font=("Consolas", 11))
        self.out_ir = tk.Text(self.tabs, wrap="none", font=("Consolas", 11))
        self.out_asm = tk.Text(self.tabs, wrap="none", font=("Consolas", 11))
        self.out_other = tk.Text(self.tabs, wrap="none", font=("Consolas", 11))
        self.out_all = tk.Text(self.tabs, wrap="none", font=("Consolas", 11))
        self.out_errors = tk.Text(self.tabs, wrap="none", font=("Consolas", 11))

        self.tabs.add(self.out_tokens, text="TOKENS")
        self.tabs.add(self.out_ast, text="AST")
        self.tabs.add(self.out_symtab, text="SYMTAB")
        self.tabs.add(self.out_ir, text="IR")
        self.tabs.add(self.out_asm, text="ASM")
        self.tabs.add(self.out_other, text="OTHER")
        self.tabs.add(self.out_all, text="ALL（合并）")
        self.tabs.add(self.out_errors, text="ERRORS")

    def _load_demo(self):
        demo = """// demo
x = 1 + 2;
print(x);
"""
        self.code.insert("1.0", demo)

    def _clear_text(self, w: tk.Text):
        w.delete("1.0", tk.END)

    def clear_output(self):
        for w in (
            self.out_tokens,
            self.out_ast,
            self.out_symtab,
            self.out_ir,
            self.out_asm,
            self.out_other,
            self.out_all,
            self.out_errors,
        ):
            self._clear_text(w)
        self.status.config(text="已清空输出")

    def _run_compiler(self, cmd):
        """
        先用 UTF-8 解码；必要时回退到 cp936（GBK）。
        """
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            cwd=os.path.dirname(COMPILER_PATH) or None,
        )

        # 如果成功但完全无输出，兜底用 cp936 再试一次（适配部分 Windows 输出）
        if (not p.stdout and not p.stderr) and p.returncode == 0:
            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="cp936",
                errors="replace",
                timeout=10,
                cwd=os.path.dirname(COMPILER_PATH) or None,
            )
        return p

    def on_compile(self):
        if not os.path.exists(COMPILER_PATH):
            messagebox.showerror("找不到编译器", f"未找到: {COMPILER_PATH}\n请修改 ui.py 里的 COMPILER_PATH")
            return

        src = self.code.get("1.0", tk.END).strip()
        if not src:
            messagebox.showwarning("空输入", "请输入源代码")
            return

        self.status.config(text="处理中...")
        self.update_idletasks()

        # 写入临时源文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as f:
            f.write(src)
            src_path = f.name

        try:
            cmd = build_cmd(src_path, self.run_var.get())
            p = self._run_compiler(cmd)

            # 合并输出（有的项目把错误也写到 stdout；合并能保证 UI 不丢信息）
            combined = (p.stdout or "") + ("\n" if (p.stdout and p.stderr) else "") + (p.stderr or "")

            sections, remainder = split_sections(combined)

            # 清空输出
            self.clear_output()

            # 分段写入
            self.out_tokens.insert(tk.END, sections.get("=== TOKENS ===", ""))
            self.out_ast.insert(tk.END, sections.get("=== AST ===", ""))
            self.out_symtab.insert(tk.END, sections.get("=== SYMTAB ===", ""))
            self.out_ir.insert(tk.END, sections.get("=== IR ===", ""))
            self.out_asm.insert(tk.END, sections.get("=== ASM ===", ""))

            # RUN/OTHER 段：如果存在 RUN，就放到 OTHER（你也可以改成单独 RUN Tab）
            run_block = ""
            for k in sections:
                if k.startswith("=== RUN"):
                    run_block = sections[k]
                    break

            other_text = ""
            if run_block:
                other_text += run_block.strip() + "\n"
            if remainder:
                other_text += ("\n" if other_text else "") + remainder

            self.out_other.insert(tk.END, other_text.strip())
            self.out_all.insert(tk.END, combined.strip())

            # ERRORS 页：只要 returncode!=0 或 stderr 非空，就显示
            if p.returncode != 0 or (p.stderr and p.stderr.strip()):
                err = ""
                if p.stderr and p.stderr.strip():
                    err += p.stderr.strip()
                # 如果 stderr 为空（你们把错误都打到 stdout），就把 combined 放进去
                if not err:
                    err = combined.strip()
                self.out_errors.insert(tk.END, err)

            # 选中合适的 Tab
            if p.returncode == 0:
                self.status.config(text="成功")
                # 优先跳 IR；没有就跳 ALL
                if sections.get("=== IR ==="):
                    self.tabs.select(self.out_ir)
                elif sections.get("=== ASM ==="):
                    self.tabs.select(self.out_asm)
                else:
                    self.tabs.select(self.out_all)
            else:
                self.status.config(text=f"失败（code={p.returncode}）")
                self.tabs.select(self.out_errors)

        except subprocess.TimeoutExpired:
            self.status.config(text="超时")
            self.out_errors.insert(tk.END, "Timeout: 编译/运行超时（>10s）\n")
            self.tabs.select(self.out_errors)
        except Exception as e:
            self.status.config(text="异常")
            self.out_errors.insert(tk.END, f"Exception: {e}\n")
            self.tabs.select(self.out_errors)
        finally:
            try:
                os.remove(src_path)
            except:
                pass


    def on_save_ast_dot(self):
        """一键生成并保存 ast.dot（依赖编译器支持 --ast-dot <path> 参数）。"""
        if not os.path.exists(COMPILER_PATH):
            messagebox.showerror("找不到编译器", f"未找到: {COMPILER_PATH}\n请修改 ui.py 里的 COMPILER_PATH")
            return

        src = self.code.get("1.0", tk.END).strip()
        if not src:
            messagebox.showwarning("空输入", "请输入源代码")
            return

        # 选择保存位置
        save_path = filedialog.asksaveasfilename(
            title="保存 AST DOT 文件",
            defaultextension=".dot",
            filetypes=[("Graphviz DOT", "*.dot"), ("All Files", "*.*")],
            initialfile="ast.dot",
        )
        if not save_path:
            return

        self.status.config(text="生成 ast.dot...")
        self.update_idletasks()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as f:
            f.write(src)
            src_path = f.name

        try:
            # 生成 DOT 不需要 --run；这里固定 run=False，避免副作用/加速
            cmd = build_cmd_with_ast_dot(src_path, False, save_path)
            p = self._run_compiler(cmd)

            combined = (p.stdout or "") + ("\n" if (p.stdout and p.stderr) else "") + (p.stderr or "")
            sections, remainder = split_sections(combined)

            # 仍然把输出同步到 UI（便于你一键保存时顺便看到 tokens/IR 等）
            self.clear_output()
            self.out_tokens.insert(tk.END, sections.get("=== TOKENS ===", ""))
            self.out_ast.insert(tk.END, sections.get("=== AST ===", ""))
            self.out_symtab.insert(tk.END, sections.get("=== SYMTAB ===", ""))
            self.out_ir.insert(tk.END, sections.get("=== IR ===", ""))
            self.out_asm.insert(tk.END, sections.get("=== ASM ===", ""))

            run_block = ""
            for k in sections:
                if k.startswith("=== RUN"):
                    run_block = sections[k]
                    break

            other_text = ""
            if run_block:
                other_text += run_block.strip() + "\n"
            if remainder:
                other_text += ("\n" if other_text else "") + remainder
            self.out_other.insert(tk.END, other_text.strip())
            self.out_all.insert(tk.END, combined.strip())

            if p.returncode != 0 or (p.stderr and p.stderr.strip()):
                err = p.stderr.strip() if (p.stderr and p.stderr.strip()) else combined.strip()
                self.out_errors.insert(tk.END, err)

            # 检查是否真的生成了文件
            if p.returncode == 0 and os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                self.status.config(text="已保存 ast.dot")
                messagebox.showinfo("成功", f"ast.dot 已保存到：\n{save_path}")
                # 优先切到 AST 页
                if sections.get("=== AST ==="):
                    self.tabs.select(self.out_ast)
                else:
                    self.tabs.select(self.out_all)
            else:
                self.status.config(text="保存失败")
                # 如果编译器不支持 --ast-dot，很可能会在输出里提示 unknown option
                hint = "\n\n提示：如果看到类似 'unknown option --ast-dot'，说明你的 compiler.exe 还没实现该参数。"
                messagebox.showerror(
                    "生成失败",
                    f"未能生成 ast.dot（返回码={p.returncode}）。\n\n输出：\n{combined.strip()}{hint}")
                self.tabs.select(self.out_errors)

        except subprocess.TimeoutExpired:
            self.status.config(text="超时")
            self.out_errors.insert(tk.END, "Timeout: 编译/生成超时（>10s）\n")
            self.tabs.select(self.out_errors)
        except Exception as e:
            self.status.config(text="异常")
            self.out_errors.insert(tk.END, f"Exception: {e}\n")
            self.tabs.select(self.out_errors)
        finally:
            try:
                os.remove(src_path)
            except:
                pass

if __name__ == "__main__":
    App().mainloop()
