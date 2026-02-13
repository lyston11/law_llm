"""
智能问题推荐模块
基于对话上下文生成相关问题推荐
"""
import json
import logging
import re
from typing import List, Dict, Any
import httpx
from config import Config

logger = logging.getLogger(__name__)


class QuestionRecommender:
    """智能问题推荐器"""

    def __init__(self):
        """初始化推荐器"""
        self.model = getattr(Config, 'RECOMMEND_MODEL', 'qwen-turbo')
        self.timeout = getattr(Config, 'RECOMMEND_TIMEOUT', 5)
        self.count_range = getattr(Config, 'RECOMMEND_COUNT', (3, 5))
        logger.info(f"QuestionRecommender 初始化: model={self.model}, timeout={self.timeout}")

    def _should_skip(self, conversation_history: List[Dict[str, str]], response: str) -> bool:
        """
        判断是否应该跳过推荐生成

        Args:
            conversation_history: 对话历史
            response: AI 回复内容

        Returns:
            True 表示跳过，False 表示生成推荐
        """
        # 输入验证：处理 None 值
        if conversation_history is None:
            conversation_history = []
        if response is None:
            response = ""

        # 场景 1: 检查简短对话模式（使用词边界匹配避免误报）
        # 使用正则表达式进行精确匹配，避免子字符串误匹配
        # 例如："我很感谢你" 不应匹配到 "感谢"
        # 匹配整个字符串为问候语，或问候语后接标点符号
        short_patterns = [
            r"^你好[！。]?$",  # 匹配整个字符串为 "你好" 或 "你好！"
            r"^您好[！。]?$",
            r"^谢谢[！。]?$",
            r"^感谢[！。]?$",
            r"^再见[！。]?$",
            r"^拜拜[！。]?$"
        ]
        last_user_msg = ""
        if conversation_history:
            last_user_msg = conversation_history[-1].get("content", "")

        # 检查最后一条用户消息是否完全匹配简短模式
        user_msg_stripped = last_user_msg.strip()
        for pattern in short_patterns:
            if re.match(pattern, user_msg_stripped):
                logger.debug("跳过推荐: 简短对话模式")
                return True

        # 场景 2: 用户明确表示没有其他问题
        # 使用词边界匹配避免误报
        no_question_patterns = [
            r"没有其他问题",
            r"暂无其他",
            r"不需要了"
        ]
        for pattern in no_question_patterns:
            if re.search(pattern, response):
                logger.debug("跳过推荐: 用户表示无其他问题")
                return True

        # 场景 3: 回复过短
        # 阈值 30 字符：用于过滤简短回复，确保推荐质量
        # 过短的回复通常包含的信息量不足以生成有价值的推荐问题
        if len(response) < 30:
            logger.debug(f"跳过推荐: 回复过短 (长度={len(response)})")
            return True

        return False

    def _format_context(
        self,
        conversation_history: List[Dict[str, str]],
        agent_actions: List[Dict],
        response: str
    ) -> str:
        """
        格式化对话上下文为结构化文本

        Args:
            conversation_history: 对话历史
            agent_actions: Agent 工具调用记录
            response: AI 回复

        Returns:
            格式化后的上下文字符串
        """
        # 输入验证：处理 None 值（与 _should_skip 保持一致）
        if conversation_history is None:
            conversation_history = []
        if agent_actions is None:
            agent_actions = []
        if response is None:
            response = ""

        context_parts = []

        # 1. 对话历史（最近 3 轮）
        context_parts.append("# 对话历史")
        recent_history = conversation_history[-3:] if conversation_history else []
        for msg in recent_history:
            role = "用户" if msg.get("role") == "user" else "助手"
            context_parts.append(f"- {role}: {msg.get('content', '')}")

        # 2. 工具调用记录
        context_parts.append("")
        context_parts.append("# 工具调用记录")
        if agent_actions:
            for action in agent_actions:
                tool_name = action.get("tool", "unknown")
                result_summary = action.get("result_summary", "无结果")
                context_parts.append(f"- 调用 {tool_name}")
                context_parts.append(f"  结果: {result_summary}")
        else:
            context_parts.append("(无工具调用)")

        # 3. AI 回复
        context_parts.append("")
        context_parts.append("# AI 回复")
        context_parts.append(response)

        return "\n".join(context_parts)

    def _build_prompt(
        self,
        conversation_history: List[Dict[str, str]],
        agent_actions: List[Dict],
        response: str
    ) -> List[Dict]:
        """
        构建用于生成推荐问题的 Prompt

        Args:
            conversation_history: 对话历史
            agent_actions: Agent 工具调用记录
            response: AI 回复

        Returns:
            OpenAI 格式的消息列表
        """
        context = self._format_context(conversation_history, agent_actions, response)

        prompt = [
            {
                "role": "system",
                "content": "基于当前对话，生成3-5个用户可能想问的相关问题。\n\n"
                          "要求：\n"
                          "1. 问题要具体、有价值，避免重复已有问题\n"
                          "2. 考虑对话主题和工具调用结果（如检索到的知识、实体关系）\n"
                          "3. 问题可以是：追问细节、了解相关概念、延伸话题、实用建议\n"
                          "4. 简洁明了，每题不超过20字\n\n"
                          "输出JSON格式：\n"
                          '{"questions": ["问题1", "问题2", "问题3"]}'
            },
            {
                "role": "user",
                "content": context
            }
        ]

        return prompt

    def generate(
        self,
        conversation_history: List[Dict[str, str]],
        agent_actions: List[Dict],
        response: str
    ) -> List[str]:
        """
        生成推荐问题

        Args:
            conversation_history: 对话历史
            agent_actions: Agent 工具调用记录
            response: AI 回复

        Returns:
            推荐问题列表，失败或跳过时返回空列表
        """
        # 1. 判断是否跳过
        if self._should_skip(conversation_history, response):
            return []

        # 2. 构建 Prompt
        try:
            prompt = self._build_prompt(conversation_history, agent_actions, response)
        except Exception as e:
            logger.error(f"构建 Prompt 失败: {e}")
            return []

        # 3. 调用 LLM
        try:
            llm_result = self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"LLM 调用失败: {e}")
            return []

        # 4. 解析结果
        try:
            return self._parse_response(llm_result)
        except Exception as e:
            logger.warning(f"解析 LLM 结果失败: {e}, result={llm_result}")
            return []

    def _call_llm(self, messages: List[Dict]) -> str:
        """
        调用 Qwen API 生成推荐问题

        Args:
            messages: OpenAI 格式的消息列表

        Returns:
            LLM 返回的原始文本

        Raises:
            Exception: API 调用失败时抛出
        """
        api_url = getattr(Config, 'API_URL', 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation')
        api_key = getattr(Config, 'API_KEY', '')

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "result_format": "message"
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()

                # 提取 content
                content = result["output"]["choices"][0]["message"]["content"]
                return content

        except Exception as e:
            logger.error(f"Qwen API 调用失败: {e}")
            raise

    def _parse_response(self, llm_output: str) -> List[str]:
        """
        解析 LLM 返回的 JSON 结果

        Args:
            llm_output: LLM 返回的原始文本

        Returns:
            问题列表
        """
        try:
            data = json.loads(llm_output)
            questions = data.get("questions", [])

            # 验证返回格式
            if not isinstance(questions, list):
                logger.warning(f"questions 不是列表: {type(questions)}")
                return []

            # 过滤空字符串
            questions = [q.strip() for q in questions if q and q.strip()]

            # 限制数量
            min_count, max_count = self.count_range
            if len(questions) > max_count:
                questions = questions[:max_count]

            logger.info(f"生成 {len(questions)} 个推荐问题")
            return questions

        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}, output={llm_output}")
            # 尝试提取数组部分
            import re
            match = re.search(r'\[.*?\]', llm_output)
            if match:
                try:
                    questions = json.loads(match.group())
                    return questions if isinstance(questions, list) else []
                except:
                    pass
            return []
