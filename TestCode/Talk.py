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
    api_key=os.getenv("NVIDIA_API_KEY", "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")
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
        # nemotron 推理模型不支持 response_format，改用 llama 做结构化输出
        response = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=messages,
            temperature=0.3,  # 降低随机性，提升结构化输出稳定性
            top_p=0.95,
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