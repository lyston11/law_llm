"""智能领域聊天机器人 — Agent 核心模块

基于 LLM Function Calling 的 Agent 架构：
  - LLM（Qwen）作为中枢大脑，自主决策是否调用领域工具
  - 通用问题直接回答，领域问题（如法律）通过工具获取专业知识后回答
  - 支持多轮对话，对话历史作为上下文

Agent 循环（ReAct 模式）：
  用户输入 → LLM 推理 → 调用工具（可选） → 观察结果 → LLM 生成最终回答
"""
import json
import time
import logging
import requests
from datetime import datetime

from config import Config
from src.rag import RAGRetriever
from src.tools import get_all_tool_definitions, get_tool_executor
from src.tools import knowledge_search, article_lookup, knowledge_graph

config = Config()
logger = logging.getLogger(__name__)


class DomainAgent:
    """
    智能领域聊天机器人 — Agent 核心

    核心职责：
    1. 管理对话上下文（conversation history）
    2. 调用 Qwen LLM（带 Function Calling）
    3. 自主判断是否需要调用领域工具（法律知识库、知识图谱等）
    4. 通用问题直接回答，领域问题通过工具增强回答
    5. 返回最终回答及 Agent 行为记录
    """

    # 系统提示词：定义 Agent 角色和行为准则
    SYSTEM_PROMPT = "\n".join([
        "# 你的身份",
        "你的名字叫「智能领域聊天机器人」。你是一个通用型 AI 助手，能够回答用户的各类问题。",
        "你不是法律专用助手。你是一个什么都能聊、什么都能答的全能助手。",
        "",
        "# 重要：自我介绍",
        "当用户打招呼（如'你好'、'hi'）或问'你是谁'时，你必须这样介绍自己：",
        "1. 先说你是「智能领域聊天机器人」，可以回答各种问题",
        "2. 然后简单提一句你在法律方面有专业知识库支持",
        "3. 绝对不要把自己说成'法律助手'或'法律智能助手'",
        "4. 绝对不要在自我介绍中只列举法律相关能力",
        "",
        "# 你的工具",
        "你配备了法律领域的专业工具（知识库检索、法条查询、知识图谱、场景分析）。",
        "这些工具只是你能力的一部分，不代表你的全部。",
        "- 用户问法律问题 -> 调用工具获取专业知识再回答",
        "- 用户问其他问题（日常聊天、技术、常识、数学等）-> 直接回答，不调用工具",
        "",
        "# 回答规范",
        "- 使用中文，通俗易懂",
        "- 不要凭记忆编造法条内容，必须通过工具查询",
        "- 当工具返回的信息不足时，诚实告知用户",
        "- 仅在法律问题回答末尾附免责声明：以上内容仅供参考，不构成法律意见。如需专业帮助，请咨询执业律师。",
        "- 非法律问题不要加免责声明",
    ])

    def __init__(self):
        """初始化 Agent"""
        self.api_key = config.API_KEY
        self.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.model = config.AGENT_MODEL
        self.max_tool_rounds = config.AGENT_MAX_TOOL_ROUNDS
        self.temperature = config.AGENT_TEMPERATURE

        # 初始化 RAG 检索器
        self.rag_retriever = None
        try:
            self.rag_retriever = RAGRetriever()
        except Exception as e:
            logger.warning(f"RAG 初始化失败: {e}")

        # 将 RAG 检索器注入到工具中
        if self.rag_retriever:
            knowledge_search.set_rag_retriever(self.rag_retriever)
            article_lookup.set_rag_retriever(self.rag_retriever)

        # 获取工具定义
        self.tools = get_all_tool_definitions()

        logger.info(f"✅ DomainAgent 初始化完成 | 模型: {self.model} | 工具数: {len(self.tools)}")

    def chat(self, user_input: str, conversation_history: list = None) -> dict:
        """
        Agent 对话入口

        Args:
            user_input: 用户输入
            conversation_history: 对话历史（OpenAI messages 格式）

        Returns:
            dict: {
                "response": 最终回答文本,
                "agent_actions": 工具调用记录列表,
                "sources": 引用的法律来源,
                "conversation_history": 更新后的对话历史,
            }
        """
        start_time = time.time()
        agent_actions = []
        sources = []

        # 构建消息列表
        messages = self._build_messages(user_input, conversation_history)

        # ========== Agent 循环 ==========
        for round_idx in range(self.max_tool_rounds):
            logger.info(f"  Agent 循环 Round {round_idx + 1}")

            # 调用 LLM
            llm_response = self._call_llm(messages)

            if llm_response is None:
                # LLM 调用失败，返回错误
                return self._build_error_response(
                    user_input, conversation_history,
                    "抱歉，AI 服务暂时不可用，请稍后再试。"
                )

            # 检查是否有工具调用
            tool_calls = llm_response.get("tool_calls")

            if tool_calls:
                # LLM 决定调用工具
                # 将 assistant 的工具调用消息加入历史
                messages.append({
                    "role": "assistant",
                    "content": llm_response.get("content") or "",
                    "tool_calls": tool_calls,
                })

                # 执行每个工具调用
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args_str = tool_call["function"]["arguments"]
                    tool_call_id = tool_call["id"]

                    logger.info(f"    🔧 调用工具: {tool_name}({tool_args_str})")

                    # 解析参数
                    try:
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        tool_args = {}

                    # 执行工具
                    tool_result = self._execute_tool(tool_name, tool_args)

                    # 记录 Agent 行为
                    action = {
                        "tool": tool_name,
                        "input": tool_args,
                        "result_summary": self._summarize_tool_result(tool_name, tool_result),
                    }
                    agent_actions.append(action)

                    # 提取来源信息
                    self._extract_sources(tool_name, tool_result, sources)

                    # 将工具结果加入消息
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_result,
                    })

                # 继续循环，让 LLM 看到工具结果后再决定
                continue

            else:
                # LLM 决定直接回答（不调用工具）
                final_response = llm_response.get("content", "")

                elapsed = time.time() - start_time
                logger.info(f"  Agent 完成 | 耗时: {elapsed:.2f}s | 工具调用: {len(agent_actions)} 次")

                # 更新对话历史
                updated_history = self._update_history(
                    user_input, final_response, conversation_history
                )

                return {
                    "response": final_response,
                    "agent_actions": agent_actions,
                    "sources": sources,
                    "conversation_history": updated_history,
                }

        # 超过最大工具调用轮次，强制生成回答
        logger.warning("  Agent 达到最大工具调用轮次，强制生成回答")
        final_response = self._force_final_answer(messages)

        updated_history = self._update_history(
            user_input, final_response, conversation_history
        )

        return {
            "response": final_response,
            "agent_actions": agent_actions,
            "sources": sources,
            "conversation_history": updated_history,
        }

    def _build_messages(self, user_input: str, conversation_history: list = None) -> list:
        """构建完整的消息列表"""
        # 动态注入当前时间，让 LLM 知道"今天是哪天"
        now = datetime.now()
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        time_info = f"\n\n# 当前时间\n{now.strftime('%Y年%m月%d日')} {weekdays[now.weekday()]} {now.strftime('%H:%M')}"
        system_prompt = self.SYSTEM_PROMPT + time_info
        messages = [{"role": "system", "content": system_prompt}]

        # 添加对话历史
        if conversation_history:
            # 只保留最近 N 轮（避免 token 超限）
            max_history = config.MAX_DIALOG_HISTORY * 2  # 每轮 2 条消息
            recent_history = conversation_history[-max_history:]
            messages.extend(recent_history)

        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})

        return messages

    def _call_llm(self, messages: list) -> dict | None:
        """
        调用 Qwen LLM（OpenAI 兼容格式，带 Function Calling）

        Returns:
            dict: LLM 响应的 message 对象（包含 content 和/或 tool_calls）
            None: 调用失败
        """
        try:
            # 过滤掉 messages 中的 None content（tool_calls 消息可能没有 content）
            clean_messages = []
            for msg in messages:
                clean_msg = {k: v for k, v in msg.items() if v is not None}
                # 确保 role 存在
                if "role" not in clean_msg:
                    continue
                clean_messages.append(clean_msg)

            payload = {
                "model": self.model,
                "messages": clean_messages,
                "tools": self.tools,
                "tool_choice": "auto",
                "temperature": self.temperature,
                "max_tokens": 1024,
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60,
            )

            logger.debug(f"  LLM 响应状态: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"  LLM API 错误: {response.status_code} - {response.text}")
                return None

            result = response.json()

            if "choices" in result and result["choices"]:
                return result["choices"][0]["message"]

            logger.error(f"  LLM 响应格式异常: {result}")
            return None

        except requests.Timeout:
            logger.error("  LLM API 超时")
            return None
        except Exception as e:
            logger.error(f"  LLM API 调用失败: {e}")
            return None

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """执行工具并返回结果"""
        executor = get_tool_executor(tool_name)
        if executor is None:
            return json.dumps({
                "success": False,
                "error": f"未知工具: {tool_name}",
            }, ensure_ascii=False)

        try:
            result = executor(**tool_args)
            return result
        except TypeError as e:
            # 参数不匹配
            logger.error(f"  工具参数错误 {tool_name}: {e}")
            return json.dumps({
                "success": False,
                "error": f"工具参数错误: {str(e)}",
            }, ensure_ascii=False)
        except Exception as e:
            logger.error(f"  工具执行失败 {tool_name}: {e}")
            return json.dumps({
                "success": False,
                "error": f"工具执行失败: {str(e)}",
            }, ensure_ascii=False)

    def _summarize_tool_result(self, tool_name: str, result_str: str) -> str:
        """生成工具结果的简短摘要（用于前端展示）"""
        try:
            result = json.loads(result_str)
            if not result.get("success", False):
                return result.get("error", "执行失败")

            if tool_name == "search_legal_knowledge":
                count = result.get("result_count", 0)
                return f"找到 {count} 条相关法律资料"
            elif tool_name == "lookup_legal_article":
                exact = result.get("exact_match_count", 0)
                total = result.get("total_results", 0)
                return f"找到 {exact} 条精确匹配, 共 {total} 条结果"
            elif tool_name == "query_knowledge_graph":
                count = result.get("relation_count", 0)
                return f"找到 {count} 个相关概念关系"
            elif tool_name == "analyze_legal_situation":
                domains = result.get("detected_domains", [])
                if domains:
                    return f"识别为 {domains[0]['domain']} 领域"
                return "场景分析完成"
            else:
                return "执行完成"
        except Exception:
            return "执行完成"

    def _extract_sources(self, tool_name: str, result_str: str, sources: list):
        """从工具结果中提取来源信息"""
        try:
            result = json.loads(result_str)
            if not result.get("success", False):
                return

            if tool_name in ("search_legal_knowledge", "lookup_legal_article"):
                for item in result.get("results", [])[:3]:
                    source = {
                        "domain": item.get("domain", "未知"),
                        "score": item.get("relevance_score", 0),
                        "snippet": item.get("content", "")[:100],
                    }
                    # 避免重复
                    if source not in sources:
                        sources.append(source)
        except Exception:
            pass

    def _update_history(
        self, user_input: str, response: str, conversation_history: list = None
    ) -> list:
        """更新对话历史（只保留 user/assistant 消息，不保留工具调用细节）"""
        history = list(conversation_history) if conversation_history else []
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        # 限制历史长度
        max_messages = config.MAX_DIALOG_HISTORY * 2
        if len(history) > max_messages:
            history = history[-max_messages:]

        return history

    def _force_final_answer(self, messages: list) -> str:
        """强制 LLM 生成最终回答（不再允许调用工具）"""
        try:
            payload = {
                "model": self.model,
                "messages": messages + [
                    {"role": "user", "content": "请根据以上工具返回的信息，直接给出最终回答。"}
                ],
                "temperature": self.temperature,
                "max_tokens": 1024,
                # 不传 tools，强制文本回答
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and result["choices"]:
                    return result["choices"][0]["message"]["content"]

            return "抱歉，处理过程中出现问题，请重新提问。"
        except Exception as e:
            logger.error(f"  强制生成回答失败: {e}")
            return "抱歉，处理过程中出现问题，请重新提问。"

    def _build_error_response(
        self, user_input: str, conversation_history: list, error_msg: str
    ) -> dict:
        """构建错误响应"""
        return {
            "response": error_msg,
            "agent_actions": [],
            "sources": [],
            "conversation_history": self._update_history(
                user_input, error_msg, conversation_history
            ),
        }

    def get_status(self) -> dict:
        """获取 Agent 状态信息"""
        rag_status = (
            self.rag_retriever.get_status()
            if self.rag_retriever
            else {"status": "unavailable", "doc_count": 0}
        )

        return {
            "agent": {
                "model": self.model,
                "tools_count": len(self.tools),
                "tools": [t["function"]["name"] for t in self.tools],
                "max_tool_rounds": self.max_tool_rounds,
            },
            "rag": rag_status,
        }
