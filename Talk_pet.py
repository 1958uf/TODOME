# Talk.py - TODOME 桌面宠物
# 功能：在桌面显示可拖拽的宠物图标，点击展开待办事项管理面板（含 AI 对话）
# 输入：鼠标交互 + 自然语言文字输入
# 输出：桌面宠物 GUI + 持久化 TODO 数据

import os
import re
import json
import math
import threading
import tkinter as tk
from datetime import datetime
from typing import Optional, Literal
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# API 客户端配置
# ============================================================
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# 颜色主题（深蓝暗色系）
C = {
    "bg_dark":   "#1a1a2e",
    "bg_mid":    "#16213e",
    "bg_light":  "#0f3460",
    "accent":    "#e94560",
    "accent2":   "#533483",
    "text":      "#eaeaea",
    "text_dim":  "#777777",
    "success":   "#4ecca3",
    "warn":      "#f7b731",
    "trans":     "#010101",   # 窗口透明色（近黑但不纯黑）
}

# ============================================================
# 数据层：TODO 模型 + 本地存储
# ============================================================

class TodoItem(BaseModel):
    """待办事项数据模型"""
    id: str
    title: str
    priority: Literal["high", "medium", "low"] = "medium"
    status: Literal["pending", "done"] = "pending"
    created_at: str = ""
    tags: list[str] = []


class TodoStore:
    """
    功能：TODO 本地 JSON 持久化存储
    输入：JSON 文件路径
    输出：CRUD 操作结果
    """

    def __init__(self, filepath: str = "todos.json"):
        self.filepath = filepath
        self.todos: dict[str, TodoItem] = {}
        self._load()

    def _load(self):
        """从 JSON 文件加载 TODO 列表"""
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.todos = {k: TodoItem(**v) for k, v in data.items()}

    def _save(self):
        """将 TODO 列表持久化到 JSON 文件"""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(
                {k: v.model_dump() for k, v in self.todos.items()},
                f, ensure_ascii=False, indent=2
            )

    def add(self, item: TodoItem) -> TodoItem:
        item.created_at = datetime.now().isoformat()
        self.todos[item.id] = item
        self._save()
        return item

    def list_all(self) -> list[TodoItem]:
        return list(self.todos.values())

    def list_pending(self) -> list[TodoItem]:
        return [t for t in self.todos.values() if t.status == "pending"]

    def complete(self, item_id: str) -> Optional[TodoItem]:
        if item_id in self.todos:
            self.todos[item_id].status = "done"
            self._save()
            return self.todos[item_id]
        return None

    def delete(self, item_id: str) -> bool:
        if item_id in self.todos:
            del self.todos[item_id]
            self._save()
            return True
        return False

    def update_title(self, item_id: str, new_title: str) -> Optional[TodoItem]:
        if item_id in self.todos:
            self.todos[item_id].title = new_title
            self._save()
            return self.todos[item_id]
        return None


# ============================================================
# AI 层：LLM Agent
# ============================================================

SYSTEM_PROMPT = """你是 TODOME，一个智能待办事项助手。

职责：理解用户的自然语言指令，操作待办事项。

必须以 JSON 格式回复，格式如下：
{
  "action": "add|list|complete|delete|update|chat",
  "todo": {
    "id": "todo_XXX（唯一ID，新增时生成）",
    "title": "任务标题",
    "priority": "high|medium|low",
    "tags": []
  },
  "target_id": "操作目标的 todo ID（complete/delete/update 时必填）",
  "message": "给用户的自然语言回复（简洁）"
}

action 说明：
- add: 添加新任务，todo 字段必填
- list: 列出任务，无需 todo 字段
- complete: 完成任务，target_id 必填
- delete: 删除任务，target_id 必填
- update: 修改标题，target_id 和 todo.title 必填
- chat: 普通聊天，只需 message 字段

优先级判断：今天/紧急/重要 → high，普通 → medium，以后/有空 → low
必须返回合法 JSON，不能有任何额外说明文字。
"""


class TodoAgent:
    """
    功能：AI 待办事项 Agent，通过 LLM 理解自然语言并执行 CRUD 操作
    输入：用户自然语言字符串
    输出：操作结果的自然语言描述
    """

    def __init__(self, store: TodoStore):
        self.store = store
        self.history: list[dict] = []

    def chat(self, user_input: str) -> str:
        """
        功能：处理用户输入，调用 LLM，执行对应操作
        输入：用户自然语言
        输出：操作结果描述字符串
        """
        self.history.append({"role": "user", "content": user_input})

        todos_ctx = json.dumps(
            [t.model_dump() for t in self.store.list_all()],
            ensure_ascii=False
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"当前 TODO 列表：{todos_ctx}"},
        ] + self.history[-10:]

        try:
            resp = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            raw = resp.choices[0].message.content
            self.history.append({"role": "assistant", "content": raw})
            return self._execute(raw)
        except Exception as e:
            return f"⚠️ API 错误：{e}"

    def _execute(self, llm_response: str) -> str:
        """
        功能：解析 LLM JSON，执行对应的 TODO 操作
        输入：LLM 返回的 JSON 字符串
        输出：操作结果描述
        """
        try:
            data = json.loads(llm_response)
        except json.JSONDecodeError:
            return f"⚠️ 解析失败：{llm_response[:60]}"

        action = data.get("action", "chat")
        message = data.get("message", "完成")

        def get_id():
            return data.get("target_id") or data.get("todo", {}).get("id")

        if action == "add" and "todo" in data:
            item = TodoItem(**data["todo"])
            self.store.add(item)
            return f"✅ {message}"

        elif action == "list":
            todos = self.store.list_all()
            if not todos:
                return "📭 暂无待办事项"
            pri_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            lines = ["📋 待办事项："]
            for t in todos:
                icon = "✅" if t.status == "done" else "⏳"
                lines.append(f"  {icon}{pri_icon.get(t.priority,'')} {t.title}  [{t.id}]")
            return "\n".join(lines)

        elif action == "complete":
            item = self.store.complete(get_id())
            return f"✅ {message}" if item else "❌ 未找到该任务"

        elif action == "delete":
            ok = self.store.delete(get_id())
            return f"🗑️ {message}" if ok else "❌ 未找到该任务"

        elif action == "update":
            tid = get_id()
            new_title = data.get("todo", {}).get("title", "")
            if tid and new_title:
                item = self.store.update_title(tid, new_title)
                return f"✏️ {message}" if item else "❌ 未找到该任务"
            return "❌ 缺少目标 ID 或新标题"

        return message


# ============================================================
# GUI 层：待办事项管理面板
# ============================================================

class TodoPanel:
    """
    功能：浮动的待办事项管理面板（无边框深色主题）
    包含：TODO 列表 + AI 对话输入 + 快捷按钮
    交互：双击完成，右键菜单删除/修改，回车发送
    """

    W, H = 340, 490

    def __init__(self, px: int, py: int, store: TodoStore,
                 agent: TodoAgent, on_close):
        """
        功能：创建面板窗口，定位在宠物旁边
        输入：px/py（宠物窗口位置），store，agent，on_close 回调
        """
        self.store = store
        self.agent = agent
        self.on_close = on_close
        self._drag_x = self._drag_y = 0

        self.win = tk.Toplevel()
        self.win.overrideredirect(True)
        self.win.attributes('-topmost', True)
        self.win.configure(bg=C["bg_dark"])

        # 定位在宠物左侧，防止超出屏幕边界
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        x = max(0, min(px - self.W - 12, sw - self.W))
        y = max(0, min(py - self.H // 2, sh - self.H))
        self.win.geometry(f"{self.W}x{self.H}+{x}+{y}")

        self._build_ui()
        self.refresh_list()

    def _build_ui(self):
        """构建面板 UI 布局"""

        # ── 标题栏 ──────────────────────────────
        title_bar = tk.Frame(self.win, bg=C["bg_light"], height=42)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)

        tk.Label(title_bar, text="🐱  TODOME", fg=C["text"],
                 bg=C["bg_light"], font=("微软雅黑", 13, "bold")
                 ).pack(side="left", padx=12)

        # 关闭按钮
        x_lbl = tk.Label(title_bar, text="✕", fg=C["text_dim"],
                         bg=C["bg_light"], font=("Arial", 14), cursor="hand2")
        x_lbl.pack(side="right", padx=10)
        x_lbl.bind("<Button-1>", lambda e: self._close())
        x_lbl.bind("<Enter>", lambda e: x_lbl.config(fg=C["accent"]))
        x_lbl.bind("<Leave>", lambda e: x_lbl.config(fg=C["text_dim"]))

        # 标题栏拖拽
        for w in (title_bar,):
            w.bind("<ButtonPress-1>", self._drag_start)
            w.bind("<B1-Motion>", self._drag_move)

        # ── 快捷按钮栏 ──────────────────────────
        btn_bar = tk.Frame(self.win, bg=C["bg_mid"], pady=6)
        btn_bar.pack(fill="x")

        for label, cmd in [
            ("📋 全部",   "列出所有任务"),
            ("⏳ 未完成", "列出所有未完成的任务"),
            ("✅ 已完成", "列出所有已完成的任务"),
        ]:
            b = tk.Label(btn_bar, text=label, fg=C["text"], bg=C["bg_light"],
                         font=("微软雅黑", 9), padx=8, pady=4, cursor="hand2")
            b.pack(side="left", padx=4)
            b.bind("<Button-1>", lambda e, c=cmd: self._quick(c))
            b.bind("<Enter>", lambda e, w=b: w.config(bg=C["accent2"]))
            b.bind("<Leave>", lambda e, w=b: w.config(bg=C["bg_light"]))

        # ── TODO 列表 ──────────────────────────
        list_frame = tk.Frame(self.win, bg=C["bg_dark"])
        list_frame.pack(fill="both", expand=True, padx=8, pady=(6, 0))

        sb = tk.Scrollbar(list_frame, bg=C["bg_mid"], troughcolor=C["bg_dark"],
                          relief="flat", bd=0)
        sb.pack(side="right", fill="y")

        self.listbox = tk.Listbox(
            list_frame,
            bg=C["bg_mid"], fg=C["text"],
            font=("微软雅黑", 10),
            selectbackground=C["bg_light"],
            selectforeground=C["success"],
            borderwidth=0, highlightthickness=0,
            activestyle="none",
            yscrollcommand=sb.set
        )
        self.listbox.pack(fill="both", expand=True)
        sb.config(command=self.listbox.yview)

        self.listbox.bind("<Double-Button-1>", self._on_double_click)
        self.listbox.bind("<Button-3>", self._on_right_click)

        # ── 对话记录 ──────────────────────────
        sep = tk.Frame(self.win, bg=C["bg_light"], height=1)
        sep.pack(fill="x", padx=8, pady=(4, 0))

        self.chat_box = tk.Text(
            self.win,
            bg=C["bg_mid"], fg=C["text"],
            font=("微软雅黑", 9),
            height=5, wrap="word",
            borderwidth=0, highlightthickness=0,
            state="disabled", padx=6, pady=4
        )
        self.chat_box.pack(fill="x", padx=8)
        self.chat_box.tag_config("user", foreground=C["warn"])
        self.chat_box.tag_config("bot",  foreground=C["success"])
        self.chat_box.tag_config("err",  foreground=C["accent"])

        # ── 输入栏 ────────────────────────────
        input_row = tk.Frame(self.win, bg=C["bg_dark"])
        input_row.pack(fill="x", padx=8, pady=8)

        self.input_var = tk.StringVar()
        entry = tk.Entry(
            input_row,
            textvariable=self.input_var,
            bg=C["bg_light"], fg=C["text"],
            insertbackground=C["text"],
            font=("微软雅黑", 10),
            borderwidth=0, highlightthickness=1,
            highlightcolor=C["accent"],
            highlightbackground=C["bg_mid"],
            relief="flat"
        )
        entry.pack(side="left", fill="x", expand=True, ipady=7, padx=(0, 6))
        entry.bind("<Return>", lambda e: self._send())
        entry.focus()

        self.send_btn = tk.Label(
            input_row, text="发送", fg=C["text"], bg=C["accent"],
            font=("微软雅黑", 9, "bold"), padx=12, pady=7, cursor="hand2"
        )
        self.send_btn.pack(side="right")
        self.send_btn.bind("<Button-1>", lambda e: self._send())

    # ── 拖拽 ────────────────────────────────
    def _drag_start(self, e):
        self._drag_x = e.x_root - self.win.winfo_x()
        self._drag_y = e.y_root - self.win.winfo_y()

    def _drag_move(self, e):
        self.win.geometry(f"+{e.x_root - self._drag_x}+{e.y_root - self._drag_y}")

    # ── 列表操作 ─────────────────────────────
    def refresh_list(self):
        """
        功能：从存储重新加载并渲染 TODO 列表
        输入：无
        输出：无
        """
        self.listbox.delete(0, tk.END)
        todos = self.store.list_all()
        if not todos:
            self.listbox.insert(tk.END, "  📭  暂无待办事项，试试输入添加吧~")
            return

        pri_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        for t in todos:
            icon = "✅" if t.status == "done" else "⏳"
            line = f"  {icon}{pri_icon.get(t.priority,'')}  {t.title}   [{t.id}]"
            self.listbox.insert(tk.END, line)
            if t.status == "done":
                self.listbox.itemconfig(tk.END, fg=C["text_dim"])

    def _get_id_from_selection(self) -> Optional[str]:
        """从列表选中行提取 todo ID"""
        sel = self.listbox.curselection()
        if not sel:
            return None
        text = self.listbox.get(sel[0])
        m = re.search(r'\[([^\]]+)\]', text)
        return m.group(1) if m else None

    def _on_double_click(self, e):
        """双击列表项 → 标记完成"""
        tid = self._get_id_from_selection()
        if tid:
            self._quick(f"完成任务 {tid}")

    def _on_right_click(self, e):
        """右键弹出操作菜单"""
        idx = self.listbox.nearest(e.y)
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        tid = self._get_id_from_selection()
        if not tid:
            return

        menu = tk.Menu(self.win, tearoff=0,
                       bg=C["bg_light"], fg=C["text"],
                       activebackground=C["accent"], activeforeground=C["text"],
                       bd=0, relief="flat")
        menu.add_command(label="✅  标记完成",
                         command=lambda: self._quick(f"完成任务 {tid}"))
        menu.add_command(label="✏️  修改标题",
                         command=lambda: self._edit_title_dialog(tid))
        menu.add_separator()
        menu.add_command(label="🗑️  删除",
                         command=lambda: self._quick(f"删除任务 {tid}"))
        menu.tk_popup(e.x_root, e.y_root)

    def _edit_title_dialog(self, tid: str):
        """弹出小对话框让用户输入新标题"""
        dlg = tk.Toplevel(self.win)
        dlg.overrideredirect(True)
        dlg.attributes('-topmost', True)
        dlg.configure(bg=C["bg_dark"])
        dlg.geometry(f"280x90+{self.win.winfo_x()+30}+{self.win.winfo_y()+200}")

        tk.Label(dlg, text="修改标题：", fg=C["text"], bg=C["bg_dark"],
                 font=("微软雅黑", 10)).pack(pady=(10, 4))

        var = tk.StringVar()
        e = tk.Entry(dlg, textvariable=var, bg=C["bg_light"], fg=C["text"],
                     insertbackground=C["text"], font=("微软雅黑", 10),
                     borderwidth=0, highlightthickness=0, width=28)
        e.pack(padx=12)
        e.focus()

        def confirm():
            new = var.get().strip()
            if new:
                self._quick(f"修改任务 {tid} 的标题为：{new}")
            dlg.destroy()

        e.bind("<Return>", lambda ev: confirm())
        e.bind("<Escape>", lambda ev: dlg.destroy())

        btn_row = tk.Frame(dlg, bg=C["bg_dark"])
        btn_row.pack(pady=6)
        tk.Button(btn_row, text="确定", command=confirm, bg=C["accent"],
                  fg=C["text"], relief="flat", padx=12).pack(side="left", padx=6)
        tk.Button(btn_row, text="取消", command=dlg.destroy, bg=C["bg_light"],
                  fg=C["text"], relief="flat", padx=12).pack(side="left")

    # ── 对话 ─────────────────────────────────
    def _append(self, text: str, tag: str = "bot"):
        """向对话框追加一行文字"""
        self.chat_box.config(state="normal")
        self.chat_box.insert(tk.END, text + "\n", tag)
        self.chat_box.see(tk.END)
        self.chat_box.config(state="disabled")

    def _quick(self, cmd: str):
        """填入指令并发送"""
        self.input_var.set(cmd)
        self._send()

    def _send(self):
        """
        功能：获取输入内容，在后台线程调用 AI，避免 UI 卡顿
        输入：输入框文字
        输出：无（响应通过 after() 回调更新 UI）
        """
        text = self.input_var.get().strip()
        if not text:
            return
        self.input_var.set("")
        self._append(f"你: {text}", "user")
        self.send_btn.config(text="⏳", bg=C["bg_light"])

        def _call():
            result = self.agent.chat(text)
            self.win.after(0, self._on_result, result)

        threading.Thread(target=_call, daemon=True).start()

    def _on_result(self, result: str):
        """API 响应回调（运行在主线程）"""
        self._append(f"TODOME: {result}", "bot")
        self.send_btn.config(text="发送", bg=C["accent"])
        self.refresh_list()

    def _close(self):
        self.win.destroy()
        self.on_close()


# ============================================================
# GUI 层：桌面宠物窗口
# ============================================================

class PetWindow:
    """
    功能：桌面宠物主窗口，透明背景、始终置顶、可拖拽
    交互：点击切换 TODO 面板；右键退出
    动画：呼吸缩放 + 任务数徽章
    """

    SIZE = 82  # 宠物窗口像素尺寸

    def __init__(self):
        """功能：初始化宠物窗口，定位到屏幕右下角"""
        self.store = TodoStore()
        self.agent = TodoAgent(self.store)
        self.panel: Optional[TodoPanel] = None
        self._panel_open = False
        self._anim_step = 0.0

        # 拖拽状态
        self._drag_start_x = self._drag_start_y = 0
        self._drag_win_x = self._drag_win_y = 0
        self._is_dragging = False

        # 主窗口
        self.root = tk.Tk()
        self.root.overrideredirect(True)                         # 无标题栏
        self.root.attributes('-topmost', True)                   # 始终置顶
        self.root.wm_attributes('-transparentcolor', C["trans"]) # 透明背景
        self.root.configure(bg=C["trans"])

        S = self.SIZE
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{S}x{S}+{sw - S - 20}+{sh - S - 60}")

        self.canvas = tk.Canvas(self.root, width=S, height=S,
                                bg=C["trans"], highlightthickness=0)
        self.canvas.pack()

        self._bind_events()
        self._draw(1.0)
        self._animate()

    # ── 绘制 ─────────────────────────────────
    def _draw(self, scale: float = 1.0):
        """
        功能：绘制宠物圆形图标 + 任务数徽章
        输入：scale（缩放比例，用于呼吸动画）
        输出：无
        """
        self.canvas.delete("all")
        S = self.SIZE
        cx = cy = S // 2
        r = int((S // 2 - 5) * scale)

        # 阴影
        self.canvas.create_oval(
            cx - r + 4, cy - r + 4, cx + r + 4, cy + r + 4,
            fill="#111111", outline=""
        )

        # 主圆颜色：绿（全完成）→ 橙（少量）→ 红（较多）
        pending = len(self.store.list_pending())
        color = C["success"] if pending == 0 else (C["warn"] if pending <= 3 else C["accent"])

        self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill=color, outline="#cccccc", width=2
        )

        # Emoji
        self.canvas.create_text(cx, cy - 2, text="🐱",
                                 font=("", int(22 * scale)), anchor="center")

        # 未完成任务数徽章
        if pending > 0:
            bx, by = cx + r - 9, cy - r + 9
            self.canvas.create_oval(bx - 10, by - 10, bx + 10, by + 10,
                                    fill=C["accent"], outline="#1a1a2e", width=2)
            self.canvas.create_text(bx, by, text=str(min(pending, 99)),
                                    fill="white", font=("Arial", 8, "bold"))

    # ── 动画 ─────────────────────────────────
    def _animate(self):
        """呼吸动画：每 50ms 刷新一帧"""
        self._anim_step += 0.06
        scale = 1.0 + 0.04 * math.sin(self._anim_step)
        if not self._panel_open:
            self._draw(scale)
        self.root.after(50, self._animate)

    # ── 事件绑定 ──────────────────────────────
    def _bind_events(self):
        self.canvas.bind("<ButtonPress-1>",   self._drag_start)
        self.canvas.bind("<B1-Motion>",       self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._drag_end)
        self.canvas.bind("<Enter>", lambda e: self._draw(1.12))
        self.canvas.bind("<Leave>", lambda e: self._draw(1.0))
        self.canvas.bind("<Button-3>", self._context_menu)

    def _drag_start(self, e):
        self._drag_start_x = e.x_root
        self._drag_start_y = e.y_root
        self._drag_win_x = self.root.winfo_x()
        self._drag_win_y = self.root.winfo_y()
        self._is_dragging = False

    def _drag_move(self, e):
        if abs(e.x_root - self._drag_start_x) > 3 or abs(e.y_root - self._drag_start_y) > 3:
            self._is_dragging = True
        if self._is_dragging:
            nx = self._drag_win_x + (e.x_root - self._drag_start_x)
            ny = self._drag_win_y + (e.y_root - self._drag_start_y)
            self.root.geometry(f"+{nx}+{ny}")

    def _drag_end(self, e):
        """拖拽结束后若未移动则视为点击"""
        if not self._is_dragging:
            self._toggle_panel()
        self._is_dragging = False

    def _toggle_panel(self):
        """切换 TODO 面板显示/隐藏"""
        if self._panel_open and self.panel:
            self.panel._close()
        else:
            x = self.root.winfo_x()
            y = self.root.winfo_y()
            self.panel = TodoPanel(
                px=x, py=y + self.SIZE // 2,
                store=self.store, agent=self.agent,
                on_close=self._on_panel_close
            )
            self._panel_open = True

    def _on_panel_close(self):
        """面板关闭后刷新徽章"""
        self._panel_open = False
        self.panel = None
        self._draw()

    def _context_menu(self, e):
        """右键弹出菜单"""
        menu = tk.Menu(self.root, tearoff=0,
                       bg=C["bg_light"], fg=C["text"],
                       activebackground=C["accent"], activeforeground=C["text"], bd=0)
        menu.add_command(label="📋  打开待办面板", command=self._toggle_panel)
        menu.add_separator()
        menu.add_command(label="❌  退出 TODOME",  command=self.root.quit)
        menu.tk_popup(e.x_root, e.y_root)

    def run(self):
        """启动事件循环"""
        print("🐱 TODOME 桌面宠物已启动！")
        print(f"  → 当前未完成任务：{len(self.store.list_pending())} 个")
        print("  → 点击宠物打开待办面板，右键退出")
        self.root.mainloop()


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    PetWindow().run()
