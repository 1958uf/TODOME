# AI Agent 长期进阶指南
## ——以 TODOME 项目为主轴，从零到世界顶尖的完整成长路径

> 本指南为你量身定制，基于你的 Python 实践基础与深度学习背景，以"TODOME"（一个持续演进的待办事项 Agent）为核心项目，
> 将理论学习、工程实战、论文跟进、简历构建串联成一条清晰可执行的成长路径。
> 当前 LLM API 环境：NVIDIA NIM API（兼容 OpenAI SDK），模型：`nvidia/nemotron-*` 系列。

---

## 目录

1. [AI Agent 热门方向全景图](#1-ai-agent-热门方向全景图)
2. [量身定制的职业转换路径](#2-量身定制的职业转换路径)
3. [TODOME 项目演进路线图](#3-todome-项目演进路线图)
   - [V1 — 基础 LLM 对话层](#v1--基础-llm-对话层)
   - [V2 — ReAct 工具调用 Agent](#v2--react-工具调用-agent)
   - [V3 — 记忆与持久化](#v3--记忆与持久化)
   - [V4 — Plan-and-Execute 规划 Agent](#v4--plan-and-execute-规划-agent)
   - [V5 — RAG 知识增强](#v5--rag-知识增强)
   - [V6 — 多 Agent 协作系统](#v6--多-agent-协作系统)
   - [V7 — 自主进化与评估体系](#v7--自主进化与评估体系)
4. [通往世界顶尖水平的进阶地图](#4-通往世界顶尖水平的进阶地图)
5. [代码仓库结构建议](#5-代码仓库结构建议)
6. [里程碑时间线](#6-里程碑时间线)
7. [面试竞争力策略](#7-面试竞争力策略)
8. [资源清单](#8-资源清单)

---

## 1. AI Agent 热门方向全景图

### 1.1 工具使用型 Agent（Tool-Use Agent）

| 维度 | 内容 |
|------|------|
| **核心特点** | LLM 通过 Function Calling / Tool Use 调用外部 API、数据库、代码执行环境，突破纯语言生成的局限 |
| **技术难点** | 工具选择准确性、参数幻觉、工具组合链路设计、错误恢复 |
| **代表框架** | LangChain Tools、LlamaIndex Tools、OpenAI Function Calling、NVIDIA NIM Tool Use |
| **岗位趋势** | 最基础、最广泛的 Agent 能力，几乎所有 Agent 岗位都要求 |

### 1.2 多智能体协作系统（Multi-Agent Systems）

| 维度 | 内容 |
|------|------|
| **核心特点** | 多个专职 Agent 分工协作（Planner/Executor/Critic/Researcher），通过消息传递完成复杂任务 |
| **技术难点** | 通信协议设计、任务分配策略、冲突解决、全局状态管理 |
| **代表框架** | AutoGen、CrewAI、LangGraph（多节点图）、MetaGPT、OpenAI Swarm |
| **岗位趋势** | 企业级复杂任务自动化的核心方向，2024-2026 增速最快 |

### 1.3 记忆与检索增强（Memory & RAG）

| 维度 | 内容 |
|------|------|
| **核心特点** | 短期（对话上下文）+ 长期（向量/图数据库）+ 情节记忆（时序事件）三层记忆体系 |
| **技术难点** | 记忆检索的相关性与时效性、遗忘策略、记忆污染防御 |
| **代表框架** | Mem0、LangGraph Memory、Zep、Weaviate、Chroma、Qdrant |
| **岗位趋势** | 知识型 Agent、个人助手方向的核心能力，持续高需求 |

### 1.4 代码生成 Agent（Code Agent）

| 维度 | 内容 |
|------|------|
| **核心特点** | 生成代码 → 执行 → 观察结果 → 修复的自主循环，可完成数据分析、软件开发任务 |
| **技术难点** | 代码安全沙箱、测试验证、多文件项目理解、依赖管理 |
| **代表框架** | OpenHands（原 OpenDevin）、SWE-agent、Aider、Devin、GitHub Copilot Agent |
| **岗位趋势** | 软件工程自动化浪潮下最热门方向之一，薪资溢价明显 |

### 1.5 GUI/浏览器自动化 Agent

| 维度 | 内容 |
|------|------|
| **核心特点** | 通过截图/DOM 理解界面状态，执行点击/输入等操作完成网页或桌面任务 |
| **技术难点** | 视觉-动作对齐、状态感知、操作原子化设计 |
| **代表框架** | Browser-Use、Playwright-Agent、UI-TARS（字节）、OmniParser（微软）|
| **岗位趋势** | RPA + AI 融合方向，企业数字化转型强需求 |

### 1.6 垂直行业 Agent

| 维度 | 内容 |
|------|------|
| **核心特点** | 深耕特定领域（医疗、法律、金融、科研），需领域知识与 Agent 能力深度融合 |
| **技术难点** | 领域知识注入、可解释性要求、合规与安全 |
| **代表框架** | 各大厂定制化框架，OpenAI GPTs、Dify、Coze |
| **岗位趋势** | 高专业壁垒，但薪资高、竞争相对少 |

### 1.7 Agent 评估与安全（Agent Eval & Safety）

| 维度 | 内容 |
|------|------|
| **核心特点** | 设计 Benchmark、自动化评估流水线，防止 Agent 越权操作、提示注入攻击 |
| **技术难点** | 评估指标设计、对抗测试、可控性保障 |
| **代表框架** | AgentBench、GAIA、τ-bench、LangSmith、Langfuse |
| **岗位趋势** | 随 Agent 商业化加速，专项安全/评估岗位快速增长 |

---

## 2. 量身定制的职业转换路径

### 2.1 你的核心优势盘点

```
优势一：Python 工程能力
  ├── 已有脚本开发和工具开发经验
  ├── 熟悉 API 调用、流程自动化
  └── 对"工具提效"有直觉 → 天然契合 Tool-Use Agent 方向

优势二：深度学习背景（CV）
  ├── 理解张量、模型推理、训练流程
  ├── 对 Transformer 架构不陌生（CV 用 ViT/CLIP 等）
  └── 能快速理解 LLM 内部机制，跳过纯新手门槛

优势三：测试思维
  ├── 习惯写 Test Case → 天然适合 Agent 评估体系设计
  ├── 边界条件意识 → Agent 鲁棒性设计
  └── 自动化流水线思维 → Agent Orchestration 设计
```

### 2.2 进阶阶梯设计

```
阶段 0：热身期（第 1-2 周）
  目标：打通 NVIDIA NIM API，理解 LLM 基础调用
  标志：能用 API 流式输出对话，理解 Token、Temperature、System Prompt
  
阶段 1：Agent 新手（第 3-8 周）
  目标：掌握 Tool Use、ReAct 循环
  标志：TODOME V1-V2 完成，Agent 能调用工具完成简单任务

阶段 2：Agent 中级（第 9-20 周）
  目标：掌握记忆系统、规划架构、RAG 集成
  标志：TODOME V3-V5 完成，Agent 有长期记忆和知识库

阶段 3：Agent 高级（第 21-40 周）
  目标：多 Agent 系统设计、评估体系、生产级部署
  标志：TODOME V6-V7 完成，能独立设计和交付 Agent 系统

阶段 4：Agent 专家（第 41-60 周）
  目标：前沿研究跟进、开源贡献、架构决策能力
  标志：有 GitHub 开源贡献、发表技术博客、能主导 Agent 系统设计

阶段 5：世界顶尖（60 周+）
  目标：系统性创新、社区影响力、前沿研究能力
  标志：论文发表/复现、被业界引用的架构设计
```

---

## 3. TODOME 项目演进路线图

### 项目总体愿景

```
TODOME v1: "你好，帮我记个事儿"         ← 基础 LLM 调用
TODOME v2: "帮我查今天天气，安排我的日程"  ← Tool Use + ReAct
TODOME v3: "你还记得我上次说要做什么吗？" ← 记忆系统
TODOME v4: "帮我规划这个月的学习计划"     ← 规划 Agent
TODOME v5: "我的知识库里有没有相关资料？" ← RAG 增强
TODOME v6: "让你的团队帮我完成这个项目"   ← 多 Agent 协作
TODOME v7: "你能自我评估并改进吗？"      ← 自主进化
```

---

### V1 — 基础 LLM 对话层

**版本目标**：打通 NVIDIA NIM API，构建可对话的基础 TODO 管理器。

**核心 Agent 概念**：
- LLM 作为"大脑"：理解自然语言指令
- System Prompt 工程：定义 Agent 角色和行为边界
- 结构化输出：JSON 模式提取任务信息

**技术栈**：
- `openai` SDK（兼容 NVIDIA NIM）
- `pydantic` 数据验证
- `json` 结构化存储

**具体实操任务**：

```python
# todome/v1/agent.py
# 功能：基础 LLM 对话 Agent，支持自然语言增删改查 TODO
# 输入：用户自然语言指令
# 输出：结构化 TODO 操作结果 + 自然语言确认回复

import os
import json
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime

# NVIDIA NIM API 客户端配置
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# TODO 数据模型
class TodoItem(BaseModel):
    """待办事项数据模型"""
    id: str
    title: str                              # 任务标题
    priority: Literal["high", "medium", "low"] = "medium"  # 优先级
    status: Literal["pending", "done"] = "pending"          # 状态
    created_at: str = ""                    # 创建时间
    tags: list[str] = []                    # 标签列表

class TodoStore:
    """TODO 本地存储管理"""
    
    def __init__(self, filepath: str = "todos.json"):
        # 功能：初始化存储，加载已有数据
        # 输入：文件路径
        # 输出：无
        self.filepath = filepath
        self.todos: dict[str, TodoItem] = {}
        self._load()
    
    def _load(self):
        """从文件加载 TODO 列表"""
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.todos = {k: TodoItem(**v) for k, v in data.items()}
    
    def _save(self):
        """持久化 TODO 列表到文件"""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(
                {k: v.model_dump() for k, v in self.todos.items()},
                f, ensure_ascii=False, indent=2
            )
    
    def add(self, item: TodoItem) -> TodoItem:
        """添加新 TODO"""
        item.created_at = datetime.now().isoformat()
        self.todos[item.id] = item
        self._save()
        return item
    
    def list_all(self) -> list[TodoItem]:
        """获取所有 TODO"""
        return list(self.todos.values())
    
    def complete(self, item_id: str) -> Optional[TodoItem]:
        """标记 TODO 完成"""
        if item_id in self.todos:
            self.todos[item_id].status = "done"
            self._save()
            return self.todos[item_id]
        return None
    
    def delete(self, item_id: str) -> bool:
        """删除 TODO"""
        if item_id in self.todos:
            del self.todos[item_id]
            self._save()
            return True
        return False

# System Prompt：定义 Agent 角色
SYSTEM_PROMPT = """你是 TODOME，一个智能待办事项助手。

你的职责：
1. 理解用户的自然语言指令，提取待办事项操作意图
2. 以 JSON 格式返回操作结果，格式如下：

{
  "action": "add|list|complete|delete",
  "todo": {
    "id": "生成唯一ID如 todo_001",
    "title": "任务标题",
    "priority": "high|medium|low",
    "tags": ["标签1", "标签2"]
  },
  "message": "给用户的自然语言确认信息"
}

注意：
- 如果只是查看列表，action 为 "list"，todo 字段可省略
- 优先级判断：今天/紧急/重要 → high，普通 → medium，有空/以后 → low
- 必须返回合法 JSON，不要有额外说明文字
"""

class TodoAgent:
    """V1 基础对话 Agent"""
    
    def __init__(self):
        # 功能：初始化 Agent 和存储
        # 输入：无
        # 输出：无
        self.store = TodoStore()
        self.conversation_history = []
    
    def chat(self, user_input: str) -> str:
        """
        功能：处理用户输入，调用 LLM 理解意图，执行 TODO 操作
        输入：用户自然语言字符串
        输出：Agent 响应字符串
        """
        # 构建消息历史（保留最近 10 轮对话上下文）
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # 将当前 TODO 列表注入上下文
        todos_context = json.dumps(
            [t.model_dump() for t in self.store.list_all()],
            ensure_ascii=False
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"当前 TODO 列表：{todos_context}"},
        ] + self.conversation_history[-10:]  # 滑动窗口上下文
        
        # 调用 NVIDIA NIM API
        response = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",  # 使用 Llama 系列做结构化输出
            messages=messages,
            temperature=0.3,  # 降低随机性，提升结构化输出稳定性
            max_tokens=1024,
            response_format={"type": "json_object"}  # 强制 JSON 输出
        )
        
        raw_response = response.choices[0].message.content
        
        # 解析并执行操作
        result_message = self._execute_action(raw_response)
        
        self.conversation_history.append({
            "role": "assistant",
            "content": raw_response
        })
        
        return result_message
    
    def _execute_action(self, llm_response: str) -> str:
        """
        功能：解析 LLM JSON 响应，执行对应 TODO 操作
        输入：LLM 返回的 JSON 字符串
        输出：操作结果的自然语言描述
        """
        try:
            data = json.loads(llm_response)
            action = data.get("action")
            message = data.get("message", "操作完成")
            
            if action == "add" and "todo" in data:
                item = TodoItem(**data["todo"])
                self.store.add(item)
                return f"✅ {message}"
            
            elif action == "list":
                todos = self.store.list_all()
                if not todos:
                    return "📭 当前没有待办事项"
                result = "📋 待办事项列表：\n"
                for t in todos:
                    status_icon = "✅" if t.status == "done" else "⏳"
                    result += f"  {status_icon} [{t.priority.upper()}] {t.title} (ID: {t.id})\n"
                return result
            
            elif action == "complete" and "todo" in data:
                item = self.store.complete(data["todo"]["id"])
                return f"✅ {message}" if item else "❌ 未找到该待办事项"
            
            elif action == "delete" and "todo" in data:
                success = self.store.delete(data["todo"]["id"])
                return f"🗑️ {message}" if success else "❌ 未找到该待办事项"
            
            return message
            
        except json.JSONDecodeError:
            return f"⚠️ 解析响应失败，请重试。原始响应：{llm_response[:100]}"

# 主程序入口
if __name__ == "__main__":
    agent = TodoAgent()
    print("🤖 TODOME V1 启动！输入 'quit' 退出")
    print("-" * 40)
    
    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ["quit", "exit", "退出"]:
            print("再见！")
            break
        if not user_input:
            continue
        response = agent.chat(user_input)
        print(f"TODOME: {response}\n")
```

**验收标准**：
- [ ] 能用自然语言添加 TODO（"帮我记一下明天要开会"）
- [ ] 能列出所有 TODO 并显示优先级
- [ ] 能标记完成和删除
- [ ] TODO 数据持久化到 JSON 文件，重启不丢失
- [ ] LLM 响应始终为合法 JSON，不出现解析错误

---

### V2 — ReAct 工具调用 Agent

**版本目标**：引入真正的 Agent 推理循环，让 TODOME 能调用外部工具（时间、天气、日历），并基于观察结果动态决策。

**核心 Agent 概念**：
- **ReAct 框架**：Reasoning（推理）+ Acting（行动）的交替循环
  - `Thought`：Agent 分析当前状态
  - `Action`：调用具体工具
  - `Observation`：获取工具返回结果
  - 重复直到任务完成
- **Function Calling**：通过 OpenAI 兼容的 tools 参数定义工具
- **工具设计原则**：原子化、幂等性、明确的输入输出规范

**技术栈**：
- OpenAI SDK `tools` + `tool_choice` 参数
- `httpx` 异步 HTTP 请求
- `python-dateutil` 日期处理

**具体实操任务**：

```python
# todome/v2/tools.py
# 功能：定义 TODOME 可用的工具集
# 输入：各工具参数
# 输出：工具执行结果字符串

import json
from datetime import datetime, timedelta
from typing import Callable

# ============================================================
# 工具函数定义
# ============================================================

def get_current_time() -> str:
    """
    功能：获取当前日期和时间
    输入：无
    输出：当前时间字符串（ISO 格式）
    """
    now = datetime.now()
    return json.dumps({
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": ["周一","周二","周三","周四","周五","周六","周日"][now.weekday()],
        "week_of_year": now.isocalendar()[1]
    }, ensure_ascii=False)

def search_todos(keyword: str, status: str = "all") -> str:
    """
    功能：按关键词和状态搜索 TODO
    输入：keyword（关键词）, status（"all"|"pending"|"done"）
    输出：匹配的 TODO 列表 JSON
    """
    # 此处从持久化存储读取（示意，实际应注入存储实例）
    import os
    todos_file = "todos.json"
    if not os.path.exists(todos_file):
        return json.dumps({"todos": [], "count": 0})
    
    with open(todos_file, "r", encoding="utf-8") as f:
        todos = json.load(f)
    
    results = []
    for todo in todos.values():
        if status != "all" and todo.get("status") != status:
            continue
        if keyword.lower() in todo.get("title", "").lower():
            results.append(todo)
    
    return json.dumps({"todos": results, "count": len(results)}, ensure_ascii=False)

def estimate_task_duration(task_description: str) -> str:
    """
    功能：估算任务所需时间（基于规则的简单版本）
    输入：task_description（任务描述文本）
    输出：估算时间（分钟）和建议
    """
    # 简单关键词匹配估时
    estimates = {
        "会议|开会|meeting": (60, "通常建议预留 60-90 分钟"),
        "报告|文档|写作": (120, "建议分两个番茄钟完成"),
        "代码|开发|编程": (180, "建议拆分成更小的子任务"),
        "邮件|回复": (15, "快速处理，15分钟内完成"),
        "学习|阅读": (45, "建议一个番茄钟专注学习"),
    }
    
    for keywords, (minutes, advice) in estimates.items():
        import re
        if re.search(keywords, task_description):
            return json.dumps({
                "estimated_minutes": minutes,
                "advice": advice,
                "pomodoros": max(1, minutes // 25)
            }, ensure_ascii=False)
    
    return json.dumps({
        "estimated_minutes": 30,
        "advice": "默认估算30分钟，请根据实际情况调整",
        "pomodoros": 1
    }, ensure_ascii=False)

def set_reminder(todo_id: str, remind_at: str, message: str = "") -> str:
    """
    功能：为 TODO 设置提醒（模拟）
    输入：todo_id（任务ID）, remind_at（提醒时间ISO字符串）, message（提醒消息）
    输出：设置结果
    """
    # V2 阶段为模拟实现，V6 阶段接入真实通知系统
    reminders_file = "reminders.json"
    reminders = {}
    
    import os
    if os.path.exists(reminders_file):
        with open(reminders_file, "r", encoding="utf-8") as f:
            reminders = json.load(f)
    
    reminders[todo_id] = {
        "todo_id": todo_id,
        "remind_at": remind_at,
        "message": message or f"提醒：待办事项 {todo_id} 即将到期",
        "created_at": datetime.now().isoformat()
    }
    
    with open(reminders_file, "w", encoding="utf-8") as f:
        json.dump(reminders, f, ensure_ascii=False, indent=2)
    
    return json.dumps({
        "success": True,
        "message": f"已设置提醒：{remind_at}",
        "todo_id": todo_id
    }, ensure_ascii=False)

# ============================================================
# OpenAI Function Calling 工具定义（Schema）
# ============================================================

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间，当用户提到'今天'、'明天'、'本周'等相对时间时必须先调用此工具",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_todos",
            "description": "搜索待办事项，支持按关键词和状态过滤",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词，为空字符串时返回所有"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["all", "pending", "done"],
                        "description": "过滤状态"
                    }
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_task_duration",
            "description": "估算任务所需时间，帮助用户做时间规划",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "任务描述文本"
                    }
                },
                "required": ["task_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "为待办事项设置定时提醒",
            "parameters": {
                "type": "object",
                "properties": {
                    "todo_id": {"type": "string"},
                    "remind_at": {
                        "type": "string",
                        "description": "提醒时间，ISO 8601 格式"
                    },
                    "message": {"type": "string", "description": "提醒消息内容"}
                },
                "required": ["todo_id", "remind_at"]
            }
        }
    }
]

# 工具函数注册表
TOOL_REGISTRY: dict[str, Callable] = {
    "get_current_time": get_current_time,
    "search_todos": lambda keyword, status="all": search_todos(keyword, status),
    "estimate_task_duration": estimate_task_duration,
    "set_reminder": set_reminder,
}
```

```python
# todome/v2/react_agent.py
# 功能：基于 ReAct 模式的工具调用 Agent
# 输入：用户自然语言指令
# 输出：完成任务后的最终回复

import os
import json
from openai import OpenAI
from .tools import TOOLS_SCHEMA, TOOL_REGISTRY

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

REACT_SYSTEM_PROMPT = """你是 TODOME V2，一个具备工具调用能力的智能待办助手。

工作方式（ReAct 模式）：
1. 分析用户请求，判断是否需要调用工具
2. 如果需要时间信息，先调用 get_current_time
3. 如果需要搜索，调用 search_todos
4. 综合工具返回结果，给出最终回答

重要原则：
- 优先使用工具获取准确信息，不要猜测时间或任务详情
- 每次只调用一个最必要的工具
- 工具调用结果要整合到最终回答中
"""

class ReActAgent:
    """V2 ReAct 工具调用 Agent"""
    
    MAX_ITERATIONS = 5  # 防止无限循环
    
    def __init__(self):
        # 功能：初始化 ReAct Agent
        # 输入：无
        # 输出：无
        self.conversation_history = []
    
    def run(self, user_input: str) -> str:
        """
        功能：执行完整的 ReAct 推理-行动循环
        输入：用户指令字符串
        输出：Agent 最终回复
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT}
        ] + self.conversation_history
        
        # ReAct 循环
        for iteration in range(self.MAX_ITERATIONS):
            response = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=2048
            )
            
            choice = response.choices[0]
            
            # 情况 1：直接返回文本（任务完成）
            if choice.finish_reason == "stop":
                final_answer = choice.message.content
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_answer
                })
                return final_answer
            
            # 情况 2：需要调用工具（继续 ReAct 循环）
            if choice.finish_reason == "tool_calls":
                tool_calls = choice.message.tool_calls
                messages.append(choice.message)  # 添加 assistant 消息（含 tool_calls）
                
                # 执行所有工具调用
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    print(f"  [🔧 调用工具] {tool_name}({tool_args})")
                    
                    # 查找并执行工具
                    if tool_name in TOOL_REGISTRY:
                        tool_result = TOOL_REGISTRY[tool_name](**tool_args)
                    else:
                        tool_result = json.dumps({"error": f"未知工具: {tool_name}"})
                    
                    print(f"  [📊 工具结果] {tool_result[:100]}...")
                    
                    # 将工具结果添加到消息链
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                
                # 继续下一轮推理
                continue
            
            # 意外情况：终止循环
            break
        
        return "⚠️ Agent 达到最大迭代次数，请简化请求后重试"
```

**验收标准**：
- [ ] "今天几号？帮我安排明天的任务"——Agent 先调用 `get_current_time`，再生成建议
- [ ] "帮我找所有未完成的工作相关任务"——Agent 调用 `search_todos` 并过滤
- [ ] "这个开发任务大概要多久？帮我设个提醒"——Agent 链式调用两个工具
- [ ] 工具调用过程在控制台可见（调试日志）
- [ ] 循环不超过 5 次，防止无限 loop

---

### V3 — 记忆与持久化

**版本目标**：让 TODOME 拥有"记忆"，能记住用户偏好、历史对话要点和跨会话上下文。

**核心 Agent 概念**：
- **短期记忆**：对话历史窗口（已在 V2 实现）
- **长期记忆**：用户画像、偏好、重要事件的持久化存储
- **情节记忆**：按时间线组织的历史事件记录
- **记忆检索**：基于语义相似度找回相关历史记忆
- **记忆压缩**：摘要长对话，防止 Token 爆炸

**技术栈**：
- `chromadb`：本地向量数据库（无需网络）
- `sentence-transformers`：生成文本向量
- `Mem0`（可选）：高级记忆管理框架

**具体实操任务**：

```python
# todome/v3/memory.py
# 功能：三层记忆系统实现
# 输入：记忆内容、查询文本
# 输出：存储确认、相关记忆列表

import json
import os
from datetime import datetime
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions

class MemorySystem:
    """
    三层记忆架构：
    1. 工作记忆（WorkingMemory）：当前对话上下文，存于内存
    2. 情节记忆（EpisodicMemory）：重要对话片段，存于向量数据库
    3. 语义记忆（SemanticMemory）：用户画像和知识，存于 JSON
    """
    
    def __init__(self, user_id: str = "default"):
        # 功能：初始化三层记忆系统
        # 输入：user_id（用户标识符）
        # 输出：无
        self.user_id = user_id
        
        # 工作记忆：最近 N 轮对话
        self.working_memory: list[dict] = []
        self.WORKING_MEMORY_SIZE = 20
        
        # 向量数据库（情节记忆）
        self.chroma_client = chromadb.PersistentClient(path=f"./memory_db/{user_id}")
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.episodic_collection = self.chroma_client.get_or_create_collection(
            name="episodic_memory",
            embedding_function=self.embedding_fn
        )
        
        # 语义记忆（用户画像）
        self.semantic_file = f"./memory_db/{user_id}/semantic.json"
        self.semantic_memory = self._load_semantic()
    
    def _load_semantic(self) -> dict:
        """加载用户语义记忆（画像）"""
        if os.path.exists(self.semantic_file):
            with open(self.semantic_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "user_preferences": {},
            "important_dates": {},
            "work_context": {},
            "communication_style": "friendly"
        }
    
    def _save_semantic(self):
        """持久化语义记忆"""
        os.makedirs(os.path.dirname(self.semantic_file), exist_ok=True)
        with open(self.semantic_file, "w", encoding="utf-8") as f:
            json.dump(self.semantic_memory, f, ensure_ascii=False, indent=2)
    
    def add_to_working(self, role: str, content: str):
        """
        功能：添加消息到工作记忆（滑动窗口）
        输入：role（"user"|"assistant"），content（消息内容）
        输出：无
        """
        self.working_memory.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # 超出窗口时，将旧记忆压缩存入情节记忆
        if len(self.working_memory) > self.WORKING_MEMORY_SIZE:
            self._compress_to_episodic()
    
    def _compress_to_episodic(self):
        """
        功能：将工作记忆压缩摘要后存入情节记忆
        输入：无（使用 self.working_memory）
        输出：无
        """
        # 取出最旧的一半对话
        to_compress = self.working_memory[:self.WORKING_MEMORY_SIZE // 2]
        self.working_memory = self.working_memory[self.WORKING_MEMORY_SIZE // 2:]
        
        # 生成摘要文本（简单拼接，V5 后用 LLM 摘要）
        summary = " | ".join([
            f"{m['role']}: {m['content'][:50]}" 
            for m in to_compress
        ])
        
        # 存入向量数据库
        doc_id = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.episodic_collection.add(
            documents=[summary],
            metadatas=[{
                "timestamp": datetime.now().isoformat(),
                "type": "conversation_summary"
            }],
            ids=[doc_id]
        )
    
    def recall_relevant(self, query: str, n_results: int = 3) -> list[str]:
        """
        功能：根据查询检索相关的情节记忆
        输入：query（查询文本），n_results（返回数量）
        输出：相关记忆文本列表
        """
        if self.episodic_collection.count() == 0:
            return []
        
        results = self.episodic_collection.query(
            query_texts=[query],
            n_results=min(n_results, self.episodic_collection.count())
        )
        
        return results["documents"][0] if results["documents"] else []
    
    def update_user_profile(self, key: str, value):
        """
        功能：更新用户画像（语义记忆）
        输入：key（属性名），value（属性值）
        输出：无
        """
        self.semantic_memory["user_preferences"][key] = value
        self._save_semantic()
    
    def get_context_for_prompt(self, current_query: str) -> str:
        """
        功能：为当前 LLM 调用生成记忆上下文摘要
        输入：current_query（当前用户问题）
        输出：注入 System Prompt 的记忆摘要字符串
        """
        # 检索相关情节记忆
        relevant_episodes = self.recall_relevant(current_query)
        
        context_parts = []
        
        # 用户画像
        if self.semantic_memory["user_preferences"]:
            prefs = json.dumps(self.semantic_memory["user_preferences"], ensure_ascii=False)
            context_parts.append(f"用户偏好：{prefs}")
        
        # 相关历史记忆
        if relevant_episodes:
            context_parts.append(f"相关历史记录：{'；'.join(relevant_episodes)}")
        
        return "\n".join(context_parts) if context_parts else ""
```

**验收标准**：
- [ ] 关闭程序重新打开后，Agent 仍然记得"用户喜欢高优先级任务排在前面"
- [ ] "上次我提到的那个项目"——Agent 能检索到相关历史对话
- [ ] 长对话不会导致 Token 超限（压缩机制正常工作）
- [ ] 向量数据库文件在 `memory_db/` 目录下持久化

---

### V4 — Plan-and-Execute 规划 Agent

**版本目标**：让 TODOME 能处理复杂的多步骤目标，自动分解为子任务并逐步执行。

**核心 Agent 概念**：
- **Plan-and-Execute**：先用 Planner LLM 生成执行计划，再用 Executor LLM 逐步执行
- **任务分解**（Task Decomposition）：将大目标拆分为 DAG（有向无环图）
- **进度跟踪**：实时更新计划执行状态
- **动态重规划**：执行失败时触发重新规划

**技术栈**：
- `langgraph`：状态机 + 图结构工作流
- `pydantic`：结构化计划数据模型

```python
# todome/v4/planner.py
# 功能：Plan-and-Execute 规划 Agent 核心模块
# 输入：用户高层目标
# 输出：结构化执行计划 + 执行结果

from pydantic import BaseModel
from typing import Literal, Optional
import json
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

class SubTask(BaseModel):
    """子任务数据结构"""
    id: str                          # 子任务唯一ID
    description: str                 # 任务描述
    depends_on: list[str] = []       # 依赖的前置子任务 ID
    status: Literal["pending", "in_progress", "done", "failed"] = "pending"
    result: Optional[str] = None     # 执行结果
    tool_needed: Optional[str] = None  # 需要调用的工具

class ExecutionPlan(BaseModel):
    """执行计划数据结构"""
    goal: str                        # 原始目标
    subtasks: list[SubTask]          # 子任务列表
    created_at: str = ""
    status: Literal["planning", "executing", "completed", "failed"] = "planning"

PLANNER_PROMPT = """你是一个任务规划专家。用户会给你一个复杂目标，你需要将其分解为具体可执行的子任务列表。

输出格式（严格 JSON）：
{
  "goal": "原始目标描述",
  "subtasks": [
    {
      "id": "task_1",
      "description": "具体操作描述",
      "depends_on": [],
      "tool_needed": "get_current_time|search_todos|null"
    }
  ]
}

规则：
1. 子任务不超过 6 个，每个任务独立可执行
2. 明确标注依赖关系（depends_on）
3. 越具体越好，避免"做 A 和 B"这样的合并任务
"""

class PlanAndExecuteAgent:
    """V4 Plan-and-Execute Agent"""
    
    def __init__(self, tools: dict):
        # 功能：初始化规划执行 Agent
        # 输入：tools（工具注册表）
        # 输出：无
        self.tools = tools
        self.current_plan: Optional[ExecutionPlan] = None
    
    def plan(self, goal: str) -> ExecutionPlan:
        """
        功能：将用户目标分解为执行计划
        输入：goal（用户的高层目标）
        输出：结构化 ExecutionPlan 对象
        """
        response = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": f"目标：{goal}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=2048
        )
        
        plan_data = json.loads(response.choices[0].message.content)
        plan = ExecutionPlan(**plan_data)
        from datetime import datetime
        plan.created_at = datetime.now().isoformat()
        self.current_plan = plan
        return plan
    
    def execute_step(self, subtask: SubTask) -> str:
        """
        功能：执行单个子任务
        输入：subtask（待执行的子任务）
        输出：执行结果字符串
        """
        subtask.status = "in_progress"
        
        # 如果需要工具，调用工具
        if subtask.tool_needed and subtask.tool_needed in self.tools:
            tool_result = self.tools[subtask.tool_needed]()
            result = f"工具调用结果：{tool_result}"
        else:
            # 用 LLM 执行子任务
            response = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[
                    {"role": "system", "content": "你是一个任务执行助手，完成分配给你的具体任务并简洁报告结果。"},
                    {"role": "user", "content": f"执行任务：{subtask.description}"}
                ],
                temperature=0.3,
                max_tokens=512
            )
            result = response.choices[0].message.content
        
        subtask.status = "done"
        subtask.result = result
        return result
    
    def run(self, goal: str) -> str:
        """
        功能：完整的规划-执行流程
        输入：goal（用户高层目标）
        输出：执行完成的总结报告
        """
        print(f"\n🎯 目标：{goal}")
        print("📋 正在制定执行计划...\n")
        
        plan = self.plan(goal)
        plan.status = "executing"
        
        print(f"📌 计划包含 {len(plan.subtasks)} 个子任务：")
        for t in plan.subtasks:
            print(f"  [{t.id}] {t.description}")
        print()
        
        results = []
        completed_ids = set()
        
        # 按依赖顺序执行（简单拓扑排序）
        max_iterations = len(plan.subtasks) * 2
        iteration = 0
        
        while len(completed_ids) < len(plan.subtasks) and iteration < max_iterations:
            iteration += 1
            for subtask in plan.subtasks:
                if subtask.id in completed_ids:
                    continue
                if subtask.status == "failed":
                    continue
                # 检查依赖是否完成
                if all(dep in completed_ids for dep in subtask.depends_on):
                    print(f"▶️  执行：{subtask.description}")
                    result = self.execute_step(subtask)
                    results.append(f"[{subtask.id}] {result}")
                    completed_ids.add(subtask.id)
                    print(f"  ✅ 完成：{result[:80]}...\n")
        
        plan.status = "completed"
        
        # 生成总结
        summary_prompt = f"""
目标：{goal}
执行结果：
{chr(10).join(results)}

请生成一份简洁的任务完成总结，告知用户完成了什么。
"""
        summary_response = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.5,
            max_tokens=512
        )
        
        return summary_response.choices[0].message.content
```

**验收标准**：
- [ ] "帮我规划下周的学习计划，并自动创建对应的 TODO 任务"——Agent 生成 5+ 子任务并逐一执行
- [ ] 子任务有依赖关系时，按正确顺序执行
- [ ] 执行过程实时打印进度
- [ ] 单个子任务失败时，跳过并继续执行其他任务

---

### V5 — RAG 知识增强

**版本目标**：为 TODOME 接入个人知识库（笔记、文档），让 Agent 能基于私有知识回答问题和做决策。

**核心 Agent 概念**：
- **RAG**（Retrieval-Augmented Generation）：检索 + 生成的协同
- **文档分块策略**：Chunk Size、Overlap、语义分块 vs 固定分块
- **向量检索**：相似度搜索、混合检索（向量 + 关键词）
- **上下文压缩**：从大量检索结果中提炼最相关信息

**技术栈**：
- `chromadb`：向量存储
- `pypdf` / `markdown-it`：文档解析
- `tiktoken`：Token 计数，控制上下文长度

**具体实操任务**：

```python
# todome/v5/knowledge_base.py
# 功能：个人知识库管理，支持文档导入、语义搜索
# 输入：文档文件路径、查询文本
# 输出：相关知识片段列表

import os
import json
import hashlib
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions

class KnowledgeBase:
    """
    个人知识库：支持 Markdown、TXT 文档导入和语义检索
    """
    
    CHUNK_SIZE = 500       # 每个知识块的字符数
    CHUNK_OVERLAP = 100    # 相邻块的重叠字符数
    
    def __init__(self, kb_path: str = "./knowledge_base"):
        # 功能：初始化知识库（向量数据库）
        # 输入：kb_path（知识库存储目录）
        # 输出：无
        self.kb_path = kb_path
        os.makedirs(kb_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=kb_path)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="knowledge",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}  # 余弦相似度
        )
        
        # 记录已导入的文件（避免重复导入）
        self.index_file = os.path.join(kb_path, "index.json")
        self.file_index = self._load_index()
    
    def _load_index(self) -> dict:
        """加载文件导入记录"""
        if os.path.exists(self.index_file):
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """保存文件导入记录"""
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.file_index, f, ensure_ascii=False, indent=2)
    
    def _chunk_text(self, text: str, source: str) -> list[dict]:
        """
        功能：将长文本切分为带重叠的知识块
        输入：text（原始文本），source（文件来源）
        输出：知识块列表 [{"text": ..., "metadata": {...}}]
        """
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = min(start + self.CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source": source,
                        "chunk_idx": chunk_idx,
                        "char_start": start,
                        "char_end": end
                    }
                })
                chunk_idx += 1
            
            start += self.CHUNK_SIZE - self.CHUNK_OVERLAP
        
        return chunks
    
    def ingest_file(self, filepath: str) -> int:
        """
        功能：导入文档到知识库
        输入：filepath（文件路径，支持 .txt/.md）
        输出：导入的知识块数量
        """
        filepath = str(Path(filepath).resolve())
        
        # 计算文件哈希，检查是否已导入
        with open(filepath, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        if filepath in self.file_index and self.file_index[filepath] == file_hash:
            print(f"文件已导入（未变更）：{filepath}")
            return 0
        
        # 读取文件内容
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 切分为知识块
        chunks = self._chunk_text(content, source=os.path.basename(filepath))
        
        if not chunks:
            return 0
        
        # 存入向量数据库
        self.collection.add(
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            ids=[f"{file_hash}_{i}" for i in range(len(chunks))]
        )
        
        # 更新索引
        self.file_index[filepath] = file_hash
        self._save_index()
        
        print(f"✅ 已导入：{os.path.basename(filepath)}（{len(chunks)} 个知识块）")
        return len(chunks)
    
    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """
        功能：语义搜索知识库
        输入：query（查询文本），n_results（返回数量）
        输出：相关知识块列表 [{"text": ..., "score": ..., "source": ...}]
        """
        if self.collection.count() == 0:
            return []
        
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            {
                "text": doc,
                "score": 1 - dist,  # 转换为相似度分数
                "source": meta.get("source", "unknown"),
                "chunk_idx": meta.get("chunk_idx", 0)
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
    
    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """
        功能：为 LLM 生成知识库上下文（控制 Token 长度）
        输入：query（查询），max_tokens（最大 Token 数）
        输出：格式化的知识库上下文字符串
        """
        results = self.search(query)
        
        if not results:
            return ""
        
        context_parts = ["以下是知识库中的相关信息：\n"]
        total_chars = 0
        char_limit = max_tokens * 3  # 粗略估算：1 token ≈ 3 字符
        
        for r in results:
            if total_chars > char_limit:
                break
            part = f"[来源: {r['source']}]\n{r['text']}\n---\n"
            context_parts.append(part)
            total_chars += len(part)
        
        return "\n".join(context_parts)
```

**验收标准**：
- [ ] 能导入 Markdown 笔记文件，搜索结果准确
- [ ] "根据我的笔记，这个技术点有什么要注意的？"——Agent 检索并引用知识库
- [ ] 重复导入同一文件时，系统识别并跳过
- [ ] 知识库搜索结果包含来源文件信息

---

### V6 — 多 Agent 协作系统

**版本目标**：将 TODOME 进化为多 Agent 系统，不同 Agent 各司其职，通过消息协作完成复杂任务。

**核心 Agent 概念**：
- **Agent 角色设计**：Orchestrator（协调者）、Researcher（研究员）、Planner（规划者）、Executor（执行者）、Critic（评审者）
- **消息传递协议**：标准化 Agent 间通信格式
- **共享状态管理**：全局任务状态的读写一致性
- **LangGraph**：基于有向图的 Agent 工作流框架

**技术栈**：
- `langgraph`：Agent 工作流编排（核心）
- `langchain`：工具和链式调用

```python
# todome/v6/multi_agent.py
# 功能：多 Agent 协作系统，基于 LangGraph 状态机
# 输入：用户任务请求
# 输出：多 Agent 协作完成的任务结果

from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator
import json
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# ============================================================
# 共享状态定义（所有 Agent 共享此状态）
# ============================================================

class AgentState(TypedDict):
    """多 Agent 系统的共享状态"""
    messages: Annotated[list, operator.add]   # 消息历史（append only）
    task: str                                  # 原始任务描述
    plan: list[str]                            # 当前执行计划
    results: dict                              # 各步骤执行结果
    next_agent: str                            # 下一个执行的 Agent
    iteration: int                             # 迭代次数（防止无限循环）
    final_answer: str                          # 最终回答

# ============================================================
# Agent 节点函数
# ============================================================

def orchestrator_agent(state: AgentState) -> AgentState:
    """
    协调者 Agent：分析任务，决定调用哪个专职 Agent
    功能：根据任务类型和当前状态，路由到最合适的下一个 Agent
    输入：AgentState
    输出：更新后的 AgentState（含 next_agent 决策）
    """
    prompt = f"""你是任务协调者。分析当前任务状态，决定下一步行动。

任务：{state["task"]}
已完成步骤：{json.dumps(state["results"], ensure_ascii=False)}
当前计划：{state["plan"]}

可用的专职 Agent：
- researcher: 需要搜索信息或查询知识库时
- planner: 需要制定执行计划时
- executor: 计划已就绪，需要执行具体操作时
- critic: 需要评审和验证结果时
- FINISH: 任务已完成

以 JSON 格式回复：{{"next": "agent名称", "reason": "选择原因"}}
"""
    
    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {"role": "system", "content": "你是一个任务协调者，专注于分配工作给合适的专职Agent。"},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=256
    )
    
    decision = json.loads(response.choices[0].message.content)
    
    return {
        **state,
        "next_agent": decision["next"],
        "messages": state["messages"] + [{
            "agent": "orchestrator",
            "content": f"路由决策：{decision['reason']}"
        }]
    }

def researcher_agent(state: AgentState) -> AgentState:
    """
    研究员 Agent：搜索知识库和 TODO 历史，收集相关信息
    功能：为当前任务收集相关背景信息
    输入：AgentState
    输出：更新后的 AgentState（含搜索结果）
    """
    # 此处接入 V5 的 KnowledgeBase
    research_result = f"已搜索任务相关信息：{state['task'][:50]}..."
    
    return {
        **state,
        "results": {**state["results"], "research": research_result},
        "messages": state["messages"] + [{
            "agent": "researcher",
            "content": research_result
        }]
    }

def planner_agent(state: AgentState) -> AgentState:
    """
    规划者 Agent：将任务分解为可执行的步骤序列
    功能：基于任务和研究结果，生成详细执行计划
    输入：AgentState（含 research 结果）
    输出：更新后的 AgentState（含 plan）
    """
    research = state["results"].get("research", "")
    
    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {"role": "system", "content": "你是任务规划专家，将复杂任务分解为3-5个清晰步骤。"},
            {"role": "user", "content": f"任务：{state['task']}\n参考信息：{research}\n\n生成执行计划（JSON列表）："}
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=512
    )
    
    plan_data = json.loads(response.choices[0].message.content)
    plan = plan_data.get("steps", plan_data.get("plan", [str(plan_data)]))
    
    return {
        **state,
        "plan": plan,
        "messages": state["messages"] + [{
            "agent": "planner",
            "content": f"制定计划：{plan}"
        }]
    }

def executor_agent(state: AgentState) -> AgentState:
    """
    执行者 Agent：按照计划执行具体操作
    功能：逐步执行规划中的操作，调用工具完成任务
    输入：AgentState（含执行计划）
    输出：更新后的 AgentState（含执行结果）
    """
    plan_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(state["plan"])])
    
    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {"role": "system", "content": "你是任务执行专家，按照计划逐步完成任务。"},
            {"role": "user", "content": f"任务：{state['task']}\n计划：\n{plan_text}\n\n执行并报告结果："}
        ],
        temperature=0.4,
        max_tokens=1024
    )
    
    execution_result = response.choices[0].message.content
    
    return {
        **state,
        "results": {**state["results"], "execution": execution_result},
        "messages": state["messages"] + [{
            "agent": "executor",
            "content": execution_result
        }]
    }

def critic_agent(state: AgentState) -> AgentState:
    """
    评审者 Agent：检查执行结果的质量和完整性
    功能：评估任务完成度，判断是否需要重新执行
    输入：AgentState（含执行结果）
    输出：更新后的 AgentState（含评审意见和 final_answer）
    """
    execution = state["results"].get("execution", "")
    
    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {"role": "system", "content": "你是质量评审专家。评估任务完成质量，给出最终总结。"},
            {"role": "user", "content": f"原始任务：{state['task']}\n执行结果：{execution}\n\n质量评估和最终总结："}
        ],
        temperature=0.3,
        max_tokens=512
    )
    
    final = response.choices[0].message.content
    
    return {
        **state,
        "final_answer": final,
        "next_agent": "FINISH",
        "messages": state["messages"] + [{
            "agent": "critic",
            "content": final
        }]
    }

def should_continue(state: AgentState) -> Literal["orchestrator", "researcher", "planner", "executor", "critic", "__end__"]:
    """
    路由函数：决定下一个节点
    功能：根据 next_agent 字段路由到对应 Agent
    输入：AgentState
    输出：下一个节点名称
    """
    next_agent = state.get("next_agent", "orchestrator")
    iteration = state.get("iteration", 0)
    
    # 防止无限循环
    if iteration > 10 or next_agent == "FINISH":
        return "__end__"
    
    return next_agent

# ============================================================
# 构建 LangGraph 工作流
# ============================================================

def build_todome_graph():
    """
    功能：构建多 Agent 协作图
    输入：无
    输出：编译后的 LangGraph 工作流
    """
    workflow = StateGraph(AgentState)
    
    # 添加 Agent 节点
    workflow.add_node("orchestrator", orchestrator_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("executor", executor_agent)
    workflow.add_node("critic", critic_agent)
    
    # 设置入口
    workflow.set_entry_point("orchestrator")
    
    # 添加条件路由边
    workflow.add_conditional_edges(
        "orchestrator",
        should_continue,
        {
            "researcher": "researcher",
            "planner": "planner",
            "executor": "executor",
            "critic": "critic",
            "__end__": END
        }
    )
    
    # 所有专职 Agent 完成后回到 Orchestrator
    for agent in ["researcher", "planner", "executor"]:
        workflow.add_edge(agent, "orchestrator")
    
    # Critic 完成后结束
    workflow.add_edge("critic", END)
    
    return workflow.compile()

# 使用示例
if __name__ == "__main__":
    graph = build_todome_graph()
    
    result = graph.invoke({
        "messages": [],
        "task": "帮我整理本周的工作计划，并生成TODO任务，预估每项任务时间",
        "plan": [],
        "results": {},
        "next_agent": "orchestrator",
        "iteration": 0,
        "final_answer": ""
    })
    
    print("\n🎯 最终结果：")
    print(result["final_answer"])
```

**验收标准**：
- [ ] 多 Agent 调用链路在日志中清晰可见
- [ ] 不同类型任务自动路由到正确的 Agent
- [ ] 超过 10 次迭代时自动终止，防止死循环
- [ ] LangGraph 工作流图可以可视化输出

---

### V7 — 自主进化与评估体系

**版本目标**：让 TODOME 具备自我评估能力，能量化衡量 Agent 性能，并通过反馈机制持续改进。

**核心 Agent 概念**：
- **Agent 评估框架**：设计多维度评估指标（准确率、响应时间、工具使用效率）
- **轨迹分析**（Trajectory Analysis）：记录并分析 Agent 完整决策轨迹
- **自反思**（Self-Reflection）：Agent 对自己的决策进行反思和改进
- **A/B 测试**：对比不同 Prompt / 模型配置的效果

**技术栈**：
- `langfuse` / `langsmith`：Agent 追踪和可观察性
- `pandas` + `matplotlib`：评估数据分析可视化
- 自定义评估基准（Benchmark）

```python
# todome/v7/evaluator.py
# 功能：TODOME Agent 评估框架
# 输入：测试用例集、Agent 实例
# 输出：评估报告（准确率、延迟、工具使用统计）

import time
import json
from dataclasses import dataclass, field
from typing import Callable, Optional
from datetime import datetime

@dataclass
class TestCase:
    """单个测试用例"""
    id: str
    user_input: str                          # 用户输入
    expected_action: str                     # 期望的操作类型（add/list/complete等）
    expected_keywords: list[str] = field(default_factory=list)  # 期望响应中包含的关键词
    should_use_tools: list[str] = field(default_factory=list)    # 应该调用的工具列表
    max_iterations: int = 5                  # 最大允许迭代次数

@dataclass  
class EvalResult:
    """单次评估结果"""
    test_id: str
    passed: bool
    latency_ms: float
    iterations: int
    tools_called: list[str]
    response: str
    error: Optional[str] = None

class AgentEvaluator:
    """Agent 自动化评估框架"""
    
    # 标准测试用例集（TODOME Benchmark v1.0）
    STANDARD_TEST_CASES = [
        TestCase(
            id="tc_001",
            user_input="帮我添加一个明天下午3点的产品评审会议",
            expected_action="add",
            expected_keywords=["产品评审", "会议"],
            should_use_tools=["get_current_time"]
        ),
        TestCase(
            id="tc_002",
            user_input="列出所有未完成的高优先级任务",
            expected_action="list",
            should_use_tools=["search_todos"]
        ),
        TestCase(
            id="tc_003",
            user_input="把 todo_001 标记为完成",
            expected_action="complete",
            expected_keywords=["完成", "已完成"]
        ),
        TestCase(
            id="tc_004",
            user_input="今天我需要做什么",
            expected_action="list",
            should_use_tools=["get_current_time", "search_todos"]
        ),
        TestCase(
            id="tc_005",
            user_input="帮我规划这周的学习时间，包括 LangGraph 和 RAG",
            expected_action="add",
            expected_keywords=["LangGraph", "RAG"],
            max_iterations=8
        )
    ]
    
    def __init__(self, agent_factory: Callable):
        # 功能：初始化评估器
        # 输入：agent_factory（创建 Agent 实例的工厂函数）
        # 输出：无
        self.agent_factory = agent_factory
        self.results: list[EvalResult] = []
    
    def evaluate_single(self, test_case: TestCase, agent) -> EvalResult:
        """
        功能：执行单个测试用例评估
        输入：test_case（测试用例），agent（Agent 实例）
        输出：EvalResult 评估结果
        """
        start_time = time.time()
        tools_called = []
        error = None
        response = ""
        
        try:
            # 注入工具调用追踪（monkey-patching 原始工具）
            response = agent.run(test_case.user_input)
        except Exception as e:
            error = str(e)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # 评估逻辑
        passed = self._check_pass(test_case, response, tools_called, error)
        
        return EvalResult(
            test_id=test_case.id,
            passed=passed,
            latency_ms=latency_ms,
            iterations=len(tools_called),
            tools_called=tools_called,
            response=response,
            error=error
        )
    
    def _check_pass(self, tc: TestCase, response: str, tools: list, error: Optional[str]) -> bool:
        """
        功能：判断测试用例是否通过
        输入：测试用例、响应、工具调用列表、错误信息
        输出：是否通过（布尔值）
        """
        if error:
            return False
        if not response:
            return False
        
        # 检查关键词
        for kw in tc.expected_keywords:
            if kw not in response:
                return False
        
        return True
    
    def run_benchmark(self) -> dict:
        """
        功能：运行完整评估基准，生成评估报告
        输入：无
        输出：评估报告字典
        """
        print(f"🧪 开始 TODOME Benchmark 评估 ({len(self.STANDARD_TEST_CASES)} 个测试用例)")
        print("=" * 60)
        
        for tc in self.STANDARD_TEST_CASES:
            agent = self.agent_factory()
            result = self.evaluate_single(tc, agent)
            self.results.append(result)
            
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status} [{tc.id}] {tc.user_input[:40]}... ({result.latency_ms:.0f}ms)")
            if result.error:
                print(f"       错误：{result.error}")
        
        return self._generate_report()
    
    def _generate_report(self) -> dict:
        """
        功能：生成评估报告摘要
        输入：无（使用 self.results）
        输出：报告字典
        """
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        avg_latency = sum(r.latency_ms for r in self.results) / total if total else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": f"{passed/total*100:.1f}%" if total else "0%",
                "avg_latency_ms": f"{avg_latency:.0f}"
            },
            "details": [
                {
                    "test_id": r.test_id,
                    "passed": r.passed,
                    "latency_ms": r.latency_ms,
                    "tools": r.tools_called
                }
                for r in self.results
            ]
        }
        
        print("\n" + "=" * 60)
        print(f"📊 评估完成：{passed}/{total} 通过 ({report['summary']['pass_rate']})")
        print(f"⏱️  平均延迟：{avg_latency:.0f}ms")
        
        # 保存报告
        report_file = f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"📁 报告已保存：{report_file}")
        
        return report
```

**验收标准**：
- [ ] Benchmark 自动运行 5 个测试用例
- [ ] 生成包含通过率、平均延迟的评估报告 JSON
- [ ] 评估结果可用于对比不同模型（nemotron vs llama）
- [ ] 失败用例有详细的失败原因记录

---

## 4. 通往世界顶尖水平的进阶地图

### 4.1 基础层（配合 V1-V2 学习）

```
必学主题                    推荐资源
─────────────────────────────────────────────────────────────
Transformer 架构复习        Andrej Karpathy: "Let's build GPT"
Prompt Engineering          DAIR.AI Prompt Engineering Guide
Function Calling 原理        OpenAI Function Calling 官方文档
Token 与上下文窗口管理        tiktoken 库实践
Temperature/Top-P 调参       动手实验对比效果
```

### 4.2 中级层（配合 V3-V4 学习）

```
必学主题                    推荐资源
─────────────────────────────────────────────────────────────
ReAct 论文精读              "ReAct: Synergizing Reasoning and Acting" (2023)
ReWOO 架构                  "ReWOO: Decoupling Reasoning from Observations" (2023)
LLMCompiler                 "An LLM Compiler for Parallel Function Calling" (2023)
Reflexion                   "Reflexion: Language Agents with Verbal Reinforcement" (2023)
LangGraph 核心概念           官方文档 + Harrison Chase 系列视频
Chain-of-Thought 变体        Few-shot, Zero-shot, Tree-of-Thought
```

### 4.3 高级层（配合 V5-V6 学习）

```
必学主题                    推荐资源
─────────────────────────────────────────────────────────────
多 Agent 通信协议            MCP (Model Context Protocol) 规范
Agent 记忆架构设计           Cognitive Architecture for LLM Agents 综述
RAG 进阶：HyDE/RAPTOR       相关论文 + LlamaIndex 文档
向量数据库选型               Qdrant vs Weaviate vs Chroma 对比
Agent 安全：提示注入防御      OWASP LLM Top 10
生产级 Agent 可观察性        Langfuse / LangSmith 实践
```

### 4.4 顶尖层（60 周+）

```
必学主题                    推荐资源
─────────────────────────────────────────────────────────────
自主 Agent 系统架构          CAMEL, AutoGPT 源码解读
强化学习 + Agent             RLHF, DPO, Constitutional AI 原理
前沿论文跟进渠道              arxiv cs.AI + Hugging Face Daily Papers
开源贡献策略                 从 bug fix 到 feature PR
Agent 评估基准               AgentBench, GAIA, τ-bench 深度研究
多模态 Agent                 GPT-4V / Llama-Vision 工具调用
```

### 4.5 必读论文清单（按学习阶段排序）

| 阶段 | 论文标题 | 核心贡献 |
|------|----------|----------|
| 基础 | Chain-of-Thought Prompting (2022) | CoT 推理范式 |
| 基础 | Toolformer (2023) | 工具学习基础 |
| 中级 | ReAct (2023) | Reason + Act 循环 |
| 中级 | Reflexion (2023) | 语言强化学习 |
| 中级 | AutoGPT 架构分析 | 自主 Agent 雏形 |
| 高级 | LangGraph 白皮书 | 状态机 Agent 设计 |
| 高级 | MemGPT (2023) | 分层记忆管理 |
| 高级 | AgentBench (2023) | Agent 评估标准 |
| 顶尖 | CAMEL (2023) | 多 Agent 角色扮演 |
| 顶尖 | MetaGPT (2023) | 软件公司 Agent |
| 顶尖 | SWE-bench (2024) | 代码 Agent 评估 |
| 顶尖 | GAIA Benchmark (2024) | 通用 Agent 评估 |

---

## 5. 代码仓库结构建议

```
TODOME/
├── README.md                   # 项目总览（中文）
├── docs/
│   ├── architecture.md         # 系统架构设计文档
│   ├── api_reference.md        # API 参考文档
│   └── changelog.md            # 版本变更日志
├── discuss/
│   ├── design_decisions.md     # 设计决策记录（ADR）
│   └── learning_notes.md       # 学习笔记
│
├── todome/
│   ├── __init__.py
│   ├── config.py               # 配置管理（API Key、模型参数）
│   │
│   ├── v1/                     # 基础 LLM 对话层
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── store.py
│   │
│   ├── v2/                     # ReAct 工具调用
│   │   ├── __init__.py
│   │   ├── react_agent.py
│   │   └── tools.py
│   │
│   ├── v3/                     # 记忆系统
│   │   ├── __init__.py
│   │   └── memory.py
│   │
│   ├── v4/                     # Plan-and-Execute
│   │   ├── __init__.py
│   │   └── planner.py
│   │
│   ├── v5/                     # RAG 知识增强
│   │   ├── __init__.py
│   │   └── knowledge_base.py
│   │
│   ├── v6/                     # 多 Agent 协作
│   │   ├── __init__.py
│   │   └── multi_agent.py
│   │
│   └── v7/                     # 评估体系
│       ├── __init__.py
│       └── evaluator.py
│
├── knowledge/                  # 知识库文档目录（Markdown 笔记）
│   └── .gitkeep
│
├── memory_db/                  # 向量记忆数据库（gitignore）
│   └── .gitkeep
│
├── tests/
│   ├── test_v1.py
│   ├── test_v2.py
│   └── test_tools.py
│
├── notebooks/                  # Jupyter 实验笔记本
│   ├── 01_llm_basics.ipynb
│   ├── 02_react_exploration.ipynb
│   └── 03_rag_experiments.ipynb
│
├── scripts/
│   ├── run_benchmark.py        # 运行评估基准
│   └── ingest_knowledge.py     # 批量导入知识库
│
├── .env.example                # 环境变量模板
├── .gitignore
├── pyproject.toml              # 项目配置（用 uv 或 poetry）
└── requirements.txt
```

### config.py 示例

```python
# todome/config.py
# 功能：统一管理所有配置项，避免硬编码
# 输入：环境变量
# 输出：配置对象

import os
from dataclasses import dataclass

@dataclass
class NvidiaConfig:
    """NVIDIA NIM API 配置"""
    api_key: str = os.getenv("NVIDIA_API_KEY", "")
    base_url: str = "https://integrate.api.nvidia.com/v1"
    
    # 可用模型列表（按能力排序）
    model_chat: str = "meta/llama-3.1-70b-instruct"       # 通用对话
    model_reasoning: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"  # 推理任务
    model_fast: str = "meta/llama-3.1-8b-instruct"        # 快速响应

@dataclass
class AgentConfig:
    """Agent 行为配置"""
    max_iterations: int = 10
    temperature_creative: float = 0.7    # 创意任务
    temperature_precise: float = 0.2     # 精确任务
    memory_window_size: int = 20
    chunk_size: int = 500

# 全局配置实例
NVIDIA = NvidiaConfig()
AGENT = AgentConfig()
```

---

## 6. 里程碑时间线

```
月份    里程碑                              核心产出
──────────────────────────────────────────────────────────────────────
第1月   热身 + V1 完成                     可运行的 LLM TODO 管理器
        Week 1: NVIDIA API 熟悉，Prompt 工程基础
        Week 2-3: V1 开发 + 测试
        Week 4: 阅读 ReAct 论文，准备 V2

第2月   V2 完成 + ReAct 深入理解           带工具调用的 ReAct Agent
        Week 5-6: V2 工具系统开发
        Week 7: 增加 3+ 实用工具（日历/天气/计算）
        Week 8: ReAct 变体实验，写学习博客 #1

第3月   V3 完成 + 记忆系统                 三层记忆 Agent
        Week 9-10: ChromaDB + 向量检索
        Week 11: 记忆压缩和检索优化
        Week 12: 读 MemGPT 论文，写技术文章

第4月   V4 完成 + 规划架构                 Plan-and-Execute Agent
        Week 13-14: LangGraph 入门 + V4 开发
        Week 15: 重规划（Replan）机制
        Week 16: 准备简历 v1.0

第5-6月 V5 完成 + RAG 系统                 知识增强 Agent
        Week 17-20: RAG 完整链路
        Week 21-22: 检索质量优化（重排序/HyDE）
        Week 23-24: 技术博客 #2 + 面试准备

第7-9月 V6 完成 + 多 Agent                 生产级多 Agent 系统
        Week 25-28: LangGraph 多节点图
        Week 29-32: Agent 通信协议设计
        Week 33-36: 集成测试 + 性能优化

第10-12月 V7 完成 + 评估体系              完整 Agent 平台 + Benchmark
        Week 37-40: 评估框架开发
        Week 41-44: 开源准备 + 文档完善
        Week 45-52: 前沿论文跟进 + 社区贡献

第13-18月 专家阶段                         行业认可的 Agent 专家
        持续：开源贡献、技术演讲、论文复现

第18月+ 顶尖阶段                           系统性创新 + 社区领袖
        原创架构设计 + 论文发表
```

---

## 7. 面试竞争力策略

### 7.1 简历构建策略

```
项目经验描述模板（STAR 格式）：

情境（Situation）：
  "设计并实现了一个从零演进的 AI Agent 系统（TODOME），
   历时 X 个月，涵盖 7 个迭代版本"

任务（Task）：
  "解决 [具体技术问题]，例如：Agent 在多轮对话中的记忆管理"

行动（Action）：
  "采用 ChromaDB 向量数据库实现三层记忆架构，
   通过滑动窗口 + 语义压缩解决 Token 超限问题"

结果（Result）：
  "将长对话场景下的上下文保留率从 30% 提升至 85%，
   评估基准通过率达到 X%"
```

### 7.2 技术亮点展示矩阵

| 掌握版本 | 可展示的技术深度 | 对应岗位级别 |
|----------|-----------------|-------------|
| V1-V2 | LLM API 调用、Prompt 设计、工具调用 | 初级 Agent 工程师 |
| V3-V4 | 记忆系统、规划架构、LangGraph | 中级 Agent 工程师 |
| V5-V6 | RAG 全栈、多 Agent 系统设计 | 高级 Agent 工程师 |
| V7+ | 评估体系、系统优化、架构决策 | Agent 架构师 / Tech Lead |

### 7.3 GitHub 作品集策略

```
必须做到：
1. README 专业且完整（英中双语）
2. 每个版本有对应的 Git Tag（v1.0, v2.0...）
3. 有可运行的 Demo（命令行 / Streamlit UI）
4. 有完整的测试用例
5. Commit 信息规范（记录学习决策）

加分项：
1. GitHub Actions CI/CD 配置
2. Benchmark 评估结果可视化
3. 技术博客系列（掘金/知乎/Medium）
4. HuggingFace Space 上的在线演示
```

### 7.4 面试高频考点准备

```
技术面（必答题）：
Q: ReAct 和 Chain-of-Thought 有什么区别？
A: CoT 是纯推理链，不与外部交互；ReAct 在推理步骤间插入真实动作和观察，
   能处理需要外部信息的任务，且具有可验证性。

Q: 你如何防止 Agent 陷入无限循环？
A: (1)最大迭代次数限制 (2)循环检测（相似状态检测）(3)置信度阈值
   在 TODOME V2 中我用了 MAX_ITERATIONS=5 + 工具调用去重检测。

Q: RAG 和 Fine-tuning 如何选择？
A: 知识频繁更新 → RAG；需要风格/行为调整 → Fine-tuning；
   追求最佳效果 → RAG + Fine-tuning 结合。

Q: 多 Agent 系统中如何处理 Agent 间的冲突？
A: 设计仲裁机制（Orchestrator）+ 明确任务边界 + 共享状态的乐观锁。

系统设计面：
"设计一个企业级 AI 客服 Agent 系统"
→ 从 TODOME 多 Agent 架构出发，展开：
  工具系统设计、记忆层设计、评估体系、安全机制、降级策略
```

### 7.5 技术博客选题建议

```
早期博客（V1-V2 阶段）：
  - "用 NVIDIA NIM API 10 分钟构建你的第一个 AI Agent"
  - "深入理解 ReAct：让 LLM 学会用工具"

中期博客（V3-V5 阶段）：
  - "AI Agent 的记忆：从对话历史到向量数据库"
  - "RAG 踩坑实录：chunk 策略如何影响检索质量"
  - "LangGraph 入门：用状态机思维设计 Agent"

后期博客（V6-V7 阶段）：
  - "多 Agent 系统设计：角色分工与通信协议"
  - "如何评估你的 AI Agent：构建自动化 Benchmark"
  - "TODOME 一周年：一个 Agent 项目的完整演进路径"
```

---

## 8. 资源清单

### 框架与工具

| 工具 | 用途 | 学习优先级 |
|------|------|-----------|
| OpenAI SDK | NVIDIA NIM API 调用 | ⭐⭐⭐⭐⭐ |
| LangGraph | Agent 工作流编排 | ⭐⭐⭐⭐⭐ |
| ChromaDB | 本地向量数据库 | ⭐⭐⭐⭐ |
| Pydantic | 结构化数据验证 | ⭐⭐⭐⭐ |
| LangChain | 工具链构建 | ⭐⭐⭐ |
| Mem0 | 高级记忆管理 | ⭐⭐⭐ |
| Langfuse | Agent 追踪可观察性 | ⭐⭐⭐ |
| Streamlit | 快速 Demo UI | ⭐⭐⭐ |
| AutoGen | 多 Agent 框架对比 | ⭐⭐ |

### NVIDIA NIM 可用模型选择指南

```python
# 不同任务的模型推荐（基于 NVIDIA NIM 生态）
MODEL_GUIDE = {
    "快速对话/工具调用": "meta/llama-3.1-8b-instruct",
    "高质量推理/规划":   "meta/llama-3.1-70b-instruct",
    "深度推理/复杂分析": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "代码生成":         "meta/codellama-70b-instruct",
    "中文优化":         "qwen/qwen2.5-72b-instruct",
}
```

### 推荐学习社区

```
中文社区：
  - AIAgent.cn（国内 Agent 专区）
  - 掘金 AI 专栏
  - B站：吴恩达系列 + LangGraph 官方中文视频

英文社区：
  - LangChain Blog（langchain.com/blog）
  - Hugging Face Blog
  - Discord: LangChain / AutoGen / CrewAI 官方频道
  - Twitter/X: @hwchase17 @karpathy @OpenAI

论文跟进：
  - arxiv.org/list/cs.AI（每日更新）
  - Hugging Face Daily Papers
  - paperswithcode.com（含代码实现）
```

---

## 结语

这份指南的核心理念：**不要学 Agent，要构建 Agent。**

每一行代码、每一次 Benchmark 结果、每一篇技术博客，
都是你通往世界顶尖水平路上不可替代的里程碑。

**TODOME 不只是一个 TODO 应用，
它是你 AI Agent 技术成长的活档案。**

当你有一天打开这份代码仓库，看到从 V1 的 200 行朴素脚本
到 V7 的多 Agent 评估平台，你会明白：
**世界顶尖水平从来不是终点，而是每天持续进化的状态。**

---

*最后更新：2026-06-01*  
*当前 API 环境：NVIDIA NIM（base_url: https://integrate.api.nvidia.com/v1）*  
*核心模型：nvidia/nemotron-3-nano-omni-30b-a3b-reasoning（推理），meta/llama-3.1-70b-instruct（通用）*
