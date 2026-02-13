# 智能问题推荐功能实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为智能领域聊天机器人添加智能问题推荐功能，基于对话上下文在每次 AI 回复后自动生成 3-5 个相关问题，提升用户交互效率。

**Architecture:** 在现有 Agent 架构基础上，新增 QuestionRecommender 模块，通过 LLM（qwen-turbo）生成式推荐问题。后端在 /dialog 接口集成推荐生成，前端 ChatMessage.vue 展示推荐卡片。采用失败降级策略，确保推荐功能不影响主对话流程。

**Tech Stack:**
- 后端: Python 3.10+, FastAPI, 通义千问 API (qwen-turbo)
- 前端: Vue 3, Element Plus, Composition API
- 测试: pytest

---

## Task 1: 创建推荐模块基础结构

**Files:**
- Create: `src/recommendation.py`
- Create: `tests/test_recommendation.py`

**Step 1: 写基础测试 - 类初始化**

```python
# tests/test_recommendation.py
import pytest
from src.recommendation import QuestionRecommender

def test_recommender_initialization():
    """测试 QuestionRecommender 正确初始化"""
    recommender = QuestionRecommender()
    assert recommender.model == "qwen-turbo"
    assert recommender.timeout == 5
    assert recommender.count_range == (3, 5)
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_recommendation.py::test_recommender_initialization -v`
Expected: FAIL with "cannot import name 'QuestionRecommender'"

**Step 3: 创建 QuestionRecommender 类骨架**

```python
# src/recommendation.py
"""
智能问题推荐模块
基于对话上下文生成相关问题推荐
"""
import logging
from typing import List, Dict, Any
from config import config

logger = logging.getLogger(__name__)

class QuestionRecommender:
    """智能问题推荐器"""

    def __init__(self):
        """初始化推荐器"""
        self.model = getattr(config, 'RECOMMEND_MODEL', 'qwen-turbo')
        self.timeout = getattr(config, 'RECOMMEND_TIMEOUT', 5)
        self.count_range = getattr(config, 'RECOMMEND_COUNT', (3, 5))
        logger.info(f"QuestionRecommender 初始化: model={self.model}, timeout={self.timeout}")
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_recommendation.py::test_recommender_initialization -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/recommendation.py tests/test_recommendation.py
git commit -m "feat: 创建 QuestionRecommender 类基础结构"
```

---

## Task 2: 实现智能跳过判断逻辑

**Files:**
- Modify: `src/recommendation.py`
- Modify: `tests/test_recommendation.py`

**Step 1: 写跳过判断测试**

```python
# tests/test_recommendation.py
def test_should_skip_greeting():
    """测试打招呼场景应该跳过推荐"""
    recommender = QuestionRecommender()
    history = [{"role": "user", "content": "你好"}]
    response = "您好！有什么我可以帮您的？"
    assert recommender._should_skip(history, response) == True

def test_should_skip_thanks():
    """测试感谢场景应该跳过推荐"""
    recommender = QuestionRecommender()
    history = [{"role": "user", "content": "谢谢你的回答"}]
    response = "不客气！"
    assert recommender._should_skip(history, response) == True

def test_should_skip_short_response():
    """测试过短回复应该跳过推荐"""
    recommender = QuestionRecommender()
    history = [{"role": "user", "content": "好的"}]
    response = "好的"
    assert recommender._should_skip(history, response) == True

def test_should_not_skip_legal_question():
    """测试法律咨询场景不应该跳过"""
    recommender = QuestionRecommender()
    history = [{"role": "user", "content": "被公司辞退了怎么办"}]
    response = "根据劳动合同法第四十七条规定..."
    assert recommender._should_skip(history, response) == False
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_recommendation.py::test_should_skip -v`
Expected: FAIL with "'QuestionRecommender' object has no attribute '_should_skip'"

**Step 3: 实现 _should_skip 方法**

```python
# src/recommendation.py (在 QuestionRecommender 类中添加)

def _should_skip(self, conversation_history: List[Dict], response: str) -> bool:
    """
    判断是否应该跳过推荐生成

    Args:
        conversation_history: 对话历史
        response: AI 回复内容

    Returns:
        True 表示跳过，False 表示生成推荐
    """
    # 场景 1: 检查简短对话模式
    short_patterns = ["你好", "您好", "谢谢", "感谢", "再见", "拜拜"]
    last_user_msg = ""
    if conversation_history:
        last_user_msg = conversation_history[-1].get("content", "")

    if any(pattern in last_user_msg for pattern in short_patterns):
        logger.debug("跳过推荐: 简短对话模式")
        return True

    # 场景 2: 用户明确表示没有其他问题
    if any(pattern in response for pattern in ["没有其他问题", "暂无其他", "不需要了"]):
        logger.debug("跳过推荐: 用户表示无其他问题")
        return True

    # 场景 3: 回复过短
    if len(response) < 30:
        logger.debug(f"跳过推荐: 回复过短 (长度={len(response)})")
        return True

    return False
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_recommendation.py::test_should_skip -v`
Expected: PASS (all 4 tests)

**Step 5: 提交**

```bash
git add src/recommendation.py tests/test_recommendation.py
git commit -m "feat: 实现智能跳过判断逻辑"
```

---

## Task 3: 实现上下文格式化

**Files:**
- Modify: `src/recommendation.py`
- Modify: `tests/test_recommendation.py`

**Step 1: 写上下文格式化测试**

```python
# tests/test_recommendation.py
def test_format_context():
    """测试上下文格式化"""
    recommender = QuestionRecommender()
    history = [
        {"role": "user", "content": "被辞退了"},
        {"role": "assistant", "content": "根据劳动合同法..."}
    ]
    actions = [
        {
            "tool": "search_legal_knowledge",
            "input": {"query": "辞退赔偿"},
            "result_summary": "找到 5 条相关法律资料"
        }
    ]
    response = "根据劳动合同法第四十七条..."

    context = recommender._format_context(history, actions, response)

    assert "# 对话历史" in context
    assert "# 工具调用记录" in context
    assert "# AI 回复" in context
    assert "search_legal_knowledge" in context
    assert "找到 5 条相关法律资料" in context
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_recommendation.py::test_format_context -v`
Expected: FAIL with no attribute '_format_context'

**Step 3: 实现 _format_context 方法**

```python
# src/recommendation.py (在 QuestionRecommender 类中添加)

def _format_context(
    self,
    conversation_history: List[Dict],
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
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_recommendation.py::test_format_context -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/recommendation.py tests/test_recommendation.py
git commit -m "feat: 实现上下文格式化方法"
```

---

## Task 4: 实现 Prompt 构建

**Files:**
- Modify: `src/recommendation.py`
- Modify: `tests/test_recommendation.py`

**Step 1: 写 Prompt 构建测试**

```python
# tests/test_recommendation.py
def test_build_prompt():
    """测试 Prompt 构建"""
    recommender = QuestionRecommender()
    history = [{"role": "user", "content": "测试"}]
    actions = []
    response = "测试回复"

    prompt = recommender._build_prompt(history, actions, response)

    assert isinstance(prompt, list)
    assert len(prompt) == 2
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"
    assert "生成3-5个用户可能想问的相关问题" in prompt[0]["content"]
    assert "# 对话历史" in prompt[1]["content"]
```

**Step 2: 运行测试验证失败**

Run: `pytest tests/test_recommendation.py::test_build_prompt -v`
Expected: FAIL with no attribute '_build_prompt'

**Step 3: 实现 _build_prompt 方法**

```python
# src/recommendation.py (在 QuestionRecommender 类中添加)

def _build_prompt(
    self,
    conversation_history: List[Dict],
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
```

**Step 4: 运行测试验证通过**

Run: `pytest tests/test_recommendation.py::test_build_prompt -v`
Expected: PASS

**Step 5: 提交**

```bash
git add src/recommendation.py tests/test_recommendation.py
git commit -m "feat: 实现 Prompt 构建方法"
```

---

## Task 5: 实现推荐生成主逻辑

**Files:**
- Modify: `src/recommendation.py`
- Modify: `config/config.py`
- Modify: `tests/test_recommendation.py`

**Step 1: 添加配置项**

```python
# config/config.py (在文件末尾添加)

# ========== 智能推荐配置 ==========
RECOMMEND_ENABLED = True           # 是否启用推荐
RECOMMEND_COUNT = (3, 5)          # 推荐 3-5 个问题
RECOMMEND_TIMEOUT = 5              # 超时时间（秒）
RECOMMEND_MODEL = "qwen-turbo"    # 推荐生成模型
```

**Step 2: 写推荐生成测试**

```python
# tests/test_recommendation.py
from unittest.mock import Mock, patch

def test_generate_returns_list():
    """测试 generate 方法返回列表"""
    recommender = QuestionRecommender()
    history = [{"role": "user", "content": "被公司辞退了怎么办"}]
    actions = []
    response = "根据劳动合同法..."

    # Mock LLM 调用
    with patch.object(recommender, '_call_llm') as mock_llm:
        mock_llm.return_value = '{"questions": ["如何计算赔偿金？", "需要准备什么材料？"]}'
        result = recommender.generate(history, actions, response)

        assert isinstance(result, list)
        assert len(result) == 2
        assert "如何计算赔偿金？" in result

def test_generate_skip_short_conversation():
    """测试简单对话跳过推荐"""
    recommender = QuestionRecommender()
    history = [{"role": "user", "content": "你好"}]
    actions = []
    response = "您好！"

    result = recommender.generate(history, actions, response)
    assert result == []

def test_generate_handles_gracefully():
    """测试 LLM 调用失败时优雅降级"""
    recommender = QuestionRecommender()
    history = [{"role": "user", "content": "测试"}]
    actions = []
    response = "测试回复"

    with patch.object(recommender, '_call_llm') as mock_llm:
        mock_llm.side_effect = Exception("API error")
        result = recommender.generate(history, actions, response)
        assert result == []  # 失败返回空列表
```

**Step 3: 运行测试验证失败**

Run: `pytest tests/test_recommendation.py::test_generate -v`
Expected: FAIL with no attribute 'generate'

**Step 4: 实现 generate 和 _call_llm 方法**

```python
# src/recommendation.py (在 QuestionRecommender 类中添加)

import json
import httpx
from config import config

def generate(
    self,
    conversation_history: List[Dict],
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
    api_url = getattr(config, 'API_URL', 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation')
    api_key = getattr(config, 'API_KEY', '')

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
```

**Step 5: 运行测试验证通过**

Run: `pytest tests/test_recommendation.py -v`
Expected: PASS (all tests)

**Step 6: 提交**

```bash
git add src/recommendation.py tests/test_recommendation.py config/config.py
git commit -m "feat: 实现推荐生成主逻辑和 LLM 调用"
```

---

## Task 6: 集成到 FastAPI 后端

**Files:**
- Modify: `api/main.py`

**Step 1: 添加推荐器初始化**

```python
# api/main.py (在文件顶部的 import 区域后添加)

from src.recommendation import QuestionRecommender

# 初始化推荐器
recommender = None
if config.RECOMMEND_ENABLED:
    try:
        recommender = QuestionRecommender()
        logger.info("QuestionRecommender 初始化成功")
    except Exception as e:
        logger.warning(f"QuestionRecommender 初始化失败: {e}")
```

**Step 2: 修改 /dialog 接口集成推荐**

```python
# api/main.py (找到 @app.post("/dialog") 端点，修改返回部分)

@app.post("/dialog")
async def dialog(request: DialogRequest):
    # ... 现有的对话逻辑 ...

    # Agent 生成回复
    result = agent.chat(user_input, conversation_history)

    # ⭐ 新增：生成推荐问题
    recommended_questions = []
    if recommender:  # 检查推荐器是否可用
        try:
            recommended_questions = recommender.generate(
                conversation_history=conversation_history,
                agent_actions=result.get("agent_actions", []),
                response=result["response"]
            )
        except Exception as e:
            logger.warning(f"推荐生成失败: {e}")
            # 失败不影响主流程，返回空列表

    # 返回响应（添加 recommended_questions 字段）
    return {
        "response": result["response"],
        "session_id": session_id,
        "status": "success",
        "agent_actions": result.get("agent_actions", []),
        "sources": result.get("sources", []),
        "conversation_history": result.get("conversation_history", []),
        "recommended_questions": recommended_questions  # ⭐ 新增字段
    }
```

**Step 3: 测试 API 响应**

```bash
# 启动后端
python -m uvicorn api.main:app --reload

# 在另一个终端测试
curl -X POST http://localhost:8000/dialog \
  -H "Content-Type: application/json" \
  -d '{"user_input": "被公司辞退了怎么办", "session_id": "test123"}'

# 检查响应中是否包含 "recommended_questions" 字段
```

**Step 4: 提交**

```bash
git add api/main.py
git commit -m "feat: 集成推荐功能到 /dialog 接口"
```

---

## Task 7: 前端 API 类型更新

**Files:**
- Modify: `frontend/src/api/index.js`

**Step 1: 更新 API 类型定义（可选，如果使用 TypeScript）**

```javascript
// frontend/src/api/index.js (不需要修改，Axios 自动处理响应)
// 响应数据会自动包含 recommended_questions 字段
```

**Step 2: 提交**

```bash
git add frontend/src/api/index.js
git commit -m "chore: 确认 API 支持 recommended_questions 字段"
```

---

## Task 8: 前端推荐卡片组件实现

**Files:**
- Modify: `frontend/src/components/ChatMessage.vue`

**Step 1: 添加推荐卡片模板**

```vue
<!-- frontend/src/components/ChatMessage.vue -->
<!-- 在 </div> 之前的消息内容后添加推荐卡片 -->

<template>
  <div class="message-wrapper" :class="{ 'is-user': message.isUser }">
    <!-- 现有的消息内容展示 -->

    <!-- ⭐ 推荐问题卡片 -->
    <div
      v-if="!message.isUser && message.recommendedQuestions && message.recommendedQuestions.length > 0"
      class="recommended-questions"
    >
      <div class="rq-header">💡 您可能还想问：</div>
      <div class="rq-list">
        <div
          v-for="(question, idx) in message.recommendedQuestions"
          :key="idx"
          class="rq-item"
          @click="handleQuestionClick(question)"
        >
          {{ question }}
        </div>
      </div>
    </div>
  </div>
</template>
```

**Step 2: 添加点击处理方法**

```javascript
// frontend/src/components/ChatMessage.vue (在 <script setup> 中添加)

const emit = defineEmits(['fill-input'])

const handleQuestionClick = (question) => {
  // 发射事件，将问题填入输入框
  emit('fill-input', question)
}
```

**Step 3: 添加样式**

```css
/* frontend/src/components/ChatMessage.vue (在 <style scoped> 中添加) */

.recommended-questions {
  margin-top: 12px;
  padding: 12px;
  background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
  border-radius: 8px;
  border-left: 3px solid #409EFF;
}

.rq-header {
  font-size: 13px;
  font-weight: 500;
  color: #606266;
  margin-bottom: 8px;
}

.rq-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 8px;
}

.rq-item {
  padding: 8px 12px;
  background: white;
  border-radius: 6px;
  font-size: 13px;
  color: #303133;
  cursor: pointer;
  transition: all 0.2s;
  border: 1px solid #e4e7ed;
}

.rq-item:hover {
  background: #409EFF;
  color: white;
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.3);
  border-color: #409EFF;
}
```

**Step 4: 提交**

```bash
git add frontend/src/components/ChatMessage.vue
git commit -m "feat: 添加推荐问题卡片 UI 组件"
```

---

## Task 9: 前端父组件集成事件处理

**Files:**
- Modify: `frontend/src/App.vue`

**Step 1: 修改 ChatMessage 组件调用**

```vue
<!-- frontend/src/App.vue -->
<!-- 找到 <ChatMessage> 组件，添加事件监听 -->

<ChatMessage
  v-for="msg in messages"
  :key="msg.id"
  :message="msg"
  @fill-input="handleFillInput"  <!-- ⭐ 添加事件监听 -->
/>
```

**Step 2: 实现事件处理方法**

```javascript
// frontend/src/App.vue (在 <script setup> 中添加)

const handleFillInput = (question) => {
  // 填入输入框
  userInput.value = question
  // 聚焦输入框
  nextTick(() => {
    const inputElement = document.querySelector('.message-input textarea')
    if (inputElement) {
      inputElement.focus()
    }
  })
}
```

**Step 3: 测试前端交互**

```bash
# 在前端目录启动开发服务器
cd frontend
npm run dev

# 打开浏览器 http://localhost:3000
# 1. 输入问题并发送
# 2. 等待 AI 回复
# 3. 检查是否显示推荐卡片
# 4. 点击推荐问题
# 5. 验证是否填入输入框
```

**Step 4: 提交**

```bash
git add frontend/src/App.vue
git commit -m "feat: 实现推荐问题点击填入功能"
```

---

## Task 10: 端到端测试和优化

**Files:**
- Modify: `tests/test_recommendation.py`
- Create: `tests/manual/test_recommendation_e2e.py`

**Step 1: 编写端到端测试脚本**

```python
# tests/manual/test_recommendation_e2e.py
"""
推荐功能端到端测试
手动运行: python tests/manual/test_recommendation_e2e.py
"""
import requests
import json

API_URL = "http://localhost:8000"

def test_recommendation_e2e():
    """测试完整的推荐流程"""
    session_id = "test_recommend_e2e"

    # 测试场景 1: 法律咨询应该生成推荐
    print("\n=== 测试场景 1: 法律咨询 ===")
    response = requests.post(
        f"{API_URL}/dialog",
        json={
            "user_input": "被公司辞退了怎么办",
            "session_id": session_id
        }
    ).json()

    assert "recommended_questions" in response
    questions = response["recommended_questions"]
    assert isinstance(questions, list)
    print(f"✓ 生成推荐问题: {questions}")

    # 测试场景 2: 简单对话应该跳过推荐
    print("\n=== 测试场景 2: 简单对话 ===")
    response = requests.post(
        f"{API_URL}/dialog",
        json={
            "user_input": "你好",
            "session_id": f"{session_id}_2"
        }
    ).json()

    assert "recommended_questions" in response
    questions = response["recommended_questions"]
    assert len(questions) == 0  # 应该为空
    print(f"✓ 简单对话跳过推荐: {questions}")

    # 测试场景 3: 连续对话
    print("\n=== 测试场景 3: 连续对话 ===")
    response = requests.post(
        f"{API_URL}/dialog",
        json={
            "user_input": "如何计算赔偿金？",
            "session_id": session_id
        }
    ).json()

    questions = response["recommended_questions"]
    print(f"✓ 连续对话推荐问题: {questions}")

    print("\n=== 所有测试通过 ===")

if __name__ == "__main__":
    test_recommendation_e2e()
```

**Step 2: 运行端到端测试**

```bash
# 确保后端正在运行
python -m uvicorn api.main:app --reload

# 在另一个终端运行测试
python tests/manual/test_recommendation_e2e.py
```

**Step 3: 性能测试**

```python
# tests/manual/test_recommendation_performance.py
import time
import requests

def test_recommendation_performance():
    """测试推荐生成性能"""
    API_URL = "http://localhost:8000"

    print("测试推荐生成性能...")
    start_time = time.time()

    response = requests.post(
        f"{API_URL}/dialog",
        json={
            "user_input": "劳动合同法有哪些规定？",
            "session_id": "perf_test"
        }
    ).json()

    end_time = time.time()
    elapsed = (end_time - start_time) * 1000  # 转换为毫秒

    print(f"总响应时间: {elapsed:.0f}ms")
    print(f"推荐问题数量: {len(response['recommended_questions'])}")

    # 推荐生成时间应该 < 1000ms
    assert elapsed < 2000, f"响应时间过长: {elapsed}ms"

if __name__ == "__main__":
    test_recommendation_performance()
```

**Step 4: 提交**

```bash
git add tests/
git commit -m "test: 添加推荐功能端到端测试"
```

---

## Task 11: 文档完善

**Files:**
- Modify: `README.md` (如果有更新需求)

**Step 1: 检查文档完整性**

```bash
# 查看技术文档
cat docs/technical_doc.md | grep -A 5 "智能推荐"

# 查看用户手册
cat docs/user_manual.md | grep -A 5 "推荐"

# 查看论文初稿
cat docs/thesis_draft.md | grep -A 5 "QuestionRecommender"
```

**Step 2: 验证所有文档已更新**

- ✅ 技术文档已添加推荐模块说明
- ✅ 用户手册已添加使用说明
- ✅ 论文初稿已添加相关章节

**Step 3: 最终提交**

```bash
git add docs/
git commit -m "docs: 完善智能推荐功能文档"
```

---

## 验收标准

完成所有任务后，系统应该满足：

1. ✅ **功能完整性**
   - 每次 AI 回复后自动生成 3-5 个推荐问题
   - 简单对话（打招呼、感谢）正确跳过推荐
   - 推荐失败不影响主对话流程

2. ✅ **用户体验**
   - 推荐卡片样式美观，交互流畅
   - 点击推荐问题自动填入输入框
   - 支持编辑后再发送

3. ✅ **性能要求**
   - 推荐生成时间 < 1 秒（使用 qwen-turbo）
   - 主对话响应增加延迟 < 500ms

4. ✅ **代码质量**
   - 单元测试覆盖率 > 80%
   - 通过所有 pytest 测试
   - 通过端到端测试

5. ✅ **文档完整性**
   - 技术文档已更新
   - 用户手册已更新
   - 论文初稿已更新

---

## 后续优化方向

1. **推荐缓存**: 对相似问题缓存推荐结果，减少 LLM 调用
2. **推荐分类**: 将推荐分为"追问"、"延伸"、"实用"等类别
3. **A/B 测试**: 对比不同推荐策略的效果
4. **用户反馈**: 收集用户对推荐的点击率数据，优化算法
