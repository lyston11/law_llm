"""API 主文件 — 智能领域聊天机器人

核心变化（v3.0）：
  - 使用 DomainAgent（LLM Function Calling）替代旧的 DialogManager（规则系统）
  - Agent 自主判断是否调用领域工具（法律知识库等），通用问题直接回答
  - /dialog 端点返回 agent_actions 和 sources 信息
  - 保留反馈、历史、总结等辅助接口
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import logging
import asyncio

from config import Config
from src.agent import DomainAgent
from src.history import DialogHistoryManager
from src.feedback import FeedbackManager
from src.summary import DialogSummaryManager
from src.recommendation import QuestionRecommender

config = Config()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== FastAPI 应用 ====================

app = FastAPI(
    title=config.PROJECT_NAME,
    version=config.PROJECT_VERSION,
    description="智能领域聊天机器人 — 基于 Agent 架构，精通法律领域（LLM + Function Calling + RAG）",
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "language": "zh-CN",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 初始化组件 ====================

# Agent（核心）
agent = DomainAgent()

# 辅助管理器
history_manager = DialogHistoryManager()
feedback_manager = FeedbackManager()
summary_manager = DialogSummaryManager()

# 推荐器（根据配置决定是否启用）
recommender = None
if config.RECOMMEND_ENABLED:
    try:
        recommender = QuestionRecommender()
        logger.info("QuestionRecommender 初始化成功")
    except Exception as e:
        logger.warning(f"QuestionRecommender 初始化失败: {e}")

# 会话对话历史存储（session_id → conversation_history）
sessions: dict[str, list] = {}

# ==================== 请求/响应模型 ====================


class DialogRequest(BaseModel):
    user_input: str
    session_id: str


class DialogResponse(BaseModel):
    response: str
    session_id: str
    status: str = "success"
    agent_actions: list = []
    sources: list = []
    recommended_questions: list = []


class FeedbackRequest(BaseModel):
    session_id: str
    turn_index: int
    rating: int
    comment: Optional[str] = None
    feedback_type: str = "general"


class FeedbackResponse(BaseModel):
    feedback_id: str
    status: str = "success"
    message: str = "反馈提交成功"


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    domain: Optional[str] = None


# ==================== 核心接口 ====================


@app.on_event("startup")
async def startup_event():
    """启动事件"""
    logger.info(f"🚀 {config.PROJECT_NAME} v{config.PROJECT_VERSION} 启动成功")
    logger.info(f"   架构: Agent (LLM Function Calling)")
    logger.info(f"   模型: {config.AGENT_MODEL}")


@app.get("/health")
async def health_check():
    """健康检查"""
    agent_status = agent.get_status()
    return {
        "status": "healthy",
        "project": config.PROJECT_NAME,
        "version": config.PROJECT_VERSION,
        "architecture": "agent",
        "model": config.AGENT_MODEL,
        "tools": agent_status["agent"]["tools"],
    }


@app.post("/dialog", response_model=DialogResponse)
async def dialog(request: DialogRequest):
    """
    对话接口（Agent 版）

    流程：
    1. 获取会话的对话历史
    2. 将用户输入和历史交给 Agent
    3. Agent 自主决策：推理 → 调用工具 → 生成回答
    4. 生成推荐问题（如果启用）
    5. 返回回答 + 工具调用记录 + 来源 + 推荐问题
    """
    try:
        session_id = request.session_id
        conversation_history = sessions.get(session_id)

        # Agent 处理
        result = agent.chat(
            user_input=request.user_input,
            conversation_history=conversation_history,
        )

        # 更新会话历史
        sessions[session_id] = result["conversation_history"]

        # 记录到历史管理器（用于导出/统计）
        # 从 agent_actions 中提取分析信息
        tools_used = [a["tool"] for a in result.get("agent_actions", [])]
        history_manager.add_turn(
            session_id=session_id,
            user_input=request.user_input,
            system_response=result["response"],
            intent=", ".join(tools_used) if tools_used else "direct_answer",
            legal_field=_extract_domain(result),
            sentiment=None,
            entities=None,
            is_legal_question=True,
        )

        # 生成推荐问题
        recommended_questions = []
        if recommender:  # 检查推荐器是否可用
            try:
                # 使用 asyncio.to_thread 让同步方法不阻塞事件循环
                recommended_questions = await asyncio.to_thread(
                    recommender.generate,
                    conversation_history=conversation_history,
                    agent_actions=result.get("agent_actions", []),
                    response=result["response"]
                )
            except Exception as e:
                logger.warning(f"推荐生成失败: {e}")
                # 失败不影响主流程，返回空列表

        return DialogResponse(
            response=result["response"],
            session_id=session_id,
            agent_actions=result.get("agent_actions", []),
            sources=result.get("sources", []),
            recommended_questions=recommended_questions,
        )

    except Exception as e:
        import traceback
        logger.error(f"对话处理错误: {e}\n{traceback.format_exc()}")
        return DialogResponse(
            response="抱歉，处理过程中出现问题，请重新提问。",
            session_id=request.session_id,
            status="error",
        )


def _extract_domain(result: dict) -> str:
    """从 Agent 结果中提取法律领域"""
    for action in result.get("agent_actions", []):
        if action.get("tool") == "analyze_legal_situation":
            summary = action.get("result_summary", "")
            if "领域" in summary:
                return summary
        inp = action.get("input", {})
        if "domain" in inp and inp["domain"]:
            return inp["domain"]
    return "通用对话"


@app.delete("/dialog/{session_id}")
async def reset_dialog(session_id: str):
    """重置对话"""
    try:
        if session_id in sessions:
            del sessions[session_id]
        history_manager.clear_history(session_id)
        return {"status": "success", "message": "对话已重置"}
    except Exception as e:
        logger.error(f"重置对话错误: {e}")
        raise HTTPException(status_code=500, detail="重置对话错误")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": f"欢迎使用{config.PROJECT_NAME} API",
        "version": config.PROJECT_VERSION,
        "architecture": "Agent (LLM Function Calling)",
        "docs": "/docs",
    }


# ==================== 知识库接口 ====================


@app.get("/knowledge/status")
async def get_knowledge_base_status():
    """获取知识库状态"""
    try:
        agent_status = agent.get_status()
        rag_status = agent_status.get("rag", {})

        return {
            "status": rag_status.get("status", "unavailable"),
            "message": rag_status.get("message", ""),
            "doc_count": rag_status.get("doc_count", 0),
            "agent_model": agent_status["agent"]["model"],
            "tools": agent_status["agent"]["tools"],
        }
    except Exception as e:
        logger.error(f"获取知识库状态错误: {e}")
        raise HTTPException(status_code=500, detail="获取知识库状态错误")


@app.post("/knowledge/search")
async def search_knowledge_base(request: RAGSearchRequest):
    """手动搜索知识库（调试/演示用）"""
    try:
        if not agent.rag_retriever or not agent.rag_retriever.is_ready:
            raise HTTPException(status_code=503, detail="知识库未就绪")

        results = agent.rag_retriever.retrieve(
            query=request.query,
            k=request.top_k,
            domain=request.domain,
        )

        return {
            "status": "success",
            "query": request.query,
            "result_count": len(results),
            "results": results,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"知识库搜索错误: {e}")
        raise HTTPException(status_code=500, detail="知识库搜索错误")


# ==================== Agent 状态接口 ====================


@app.get("/agent/status")
async def get_agent_status():
    """获取 Agent 详细状态"""
    try:
        return {
            "status": "success",
            **agent.get_status(),
        }
    except Exception as e:
        logger.error(f"获取 Agent 状态错误: {e}")
        raise HTTPException(status_code=500, detail="获取 Agent 状态错误")


# ==================== 历史/反馈/总结接口 ====================


@app.get("/dialog/{session_id}/history")
async def get_dialog_history(session_id: str, limit: int = None):
    """获取对话历史"""
    try:
        history = history_manager.get_history(session_id, limit)
        return {"status": "success", "history": history, "session_id": session_id}
    except Exception as e:
        logger.error(f"获取对话历史错误: {e}")
        raise HTTPException(status_code=500, detail="获取对话历史错误")


@app.get("/dialog/{session_id}/history/export")
async def export_dialog_history(session_id: str):
    """导出对话历史"""
    try:
        history_json = history_manager.export_history(session_id)
        if not history_json:
            raise HTTPException(status_code=404, detail="对话历史不存在")
        return {"status": "success", "history": history_json, "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导出对话历史错误: {e}")
        raise HTTPException(status_code=500, detail="导出对话历史错误")


@app.get("/sessions")
async def get_sessions():
    """获取会话列表"""
    try:
        all_sessions = history_manager.get_all_sessions()
        return {"status": "success", "sessions": all_sessions, "count": len(all_sessions)}
    except Exception as e:
        logger.error(f"获取会话列表错误: {e}")
        raise HTTPException(status_code=500, detail="获取会话列表错误")


@app.get("/sessions/count")
async def get_sessions_count():
    """获取会话数量"""
    try:
        count = history_manager.get_session_count()
        return {"status": "success", "count": count}
    except Exception as e:
        logger.error(f"获取会话数量错误: {e}")
        raise HTTPException(status_code=500, detail="获取会话数量错误")


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """提交反馈"""
    try:
        if feedback.rating < 1 or feedback.rating > 5:
            raise HTTPException(status_code=400, detail="评分必须在1-5之间")
        feedback_id = feedback_manager.submit_feedback(
            session_id=feedback.session_id,
            turn_index=feedback.turn_index,
            rating=feedback.rating,
            comment=feedback.comment,
            feedback_type=feedback.feedback_type,
        )
        return FeedbackResponse(feedback_id=feedback_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"提交反馈错误: {e}")
        raise HTTPException(status_code=500, detail="提交反馈错误")


@app.get("/feedback/{session_id}")
async def get_session_feedback(session_id: str):
    """获取会话反馈"""
    try:
        feedback = feedback_manager.get_feedback_by_session(session_id)
        return {"status": "success", "feedback": feedback, "session_id": session_id}
    except Exception as e:
        logger.error(f"获取会话反馈错误: {e}")
        raise HTTPException(status_code=500, detail="获取会话反馈错误")


@app.get("/feedback/stats")
async def get_feedback_stats():
    """获取反馈统计"""
    try:
        stats = feedback_manager.get_feedback_stats()
        return {"status": "success", "stats": stats}
    except Exception as e:
        logger.error(f"获取反馈统计错误: {e}")
        raise HTTPException(status_code=500, detail="获取反馈统计错误")


@app.get("/feedback/all")
async def get_all_feedback():
    """获取所有反馈"""
    try:
        feedback = feedback_manager.get_all_feedback()
        return {"status": "success", "feedback": feedback, "count": len(feedback)}
    except Exception as e:
        logger.error(f"获取所有反馈错误: {e}")
        raise HTTPException(status_code=500, detail="获取所有反馈错误")


@app.get("/feedback/export")
async def export_feedback(feedback_type: str = None):
    """导出反馈"""
    try:
        feedback_json = feedback_manager.export_feedback(feedback_type=feedback_type)
        return {"status": "success", "feedback": feedback_json}
    except Exception as e:
        logger.error(f"导出反馈错误: {e}")
        raise HTTPException(status_code=500, detail="导出反馈错误")


# 对话总结
@app.get("/dialog/{session_id}/summary")
async def get_dialog_summary(session_id: str):
    """获取对话总结"""
    try:
        summary = summary_manager.get_summary(session_id)
        if not summary:
            history = history_manager.get_history(session_id)
            if not history:
                return {"status": "success", "summary": "当前对话暂无内容", "session_id": session_id}
            summary = summary_manager.generate_summary(session_id, history)
        return {"status": "success", "summary": summary, "session_id": session_id}
    except Exception as e:
        logger.error(f"获取对话总结错误: {e}")
        raise HTTPException(status_code=500, detail="获取对话总结错误")


@app.post("/dialog/{session_id}/summary")
async def generate_dialog_summary(session_id: str):
    """生成对话总结"""
    try:
        history = history_manager.get_history(session_id)
        if not history:
            return {"status": "success", "summary": "当前对话暂无内容", "session_id": session_id}
        summary = summary_manager.generate_summary(session_id, history)
        return {"status": "success", "summary": summary, "session_id": session_id, "message": "对话总结生成成功"}
    except Exception as e:
        logger.error(f"生成对话总结错误: {e}")
        raise HTTPException(status_code=500, detail="生成对话总结错误")


@app.delete("/dialog/{session_id}/summary")
async def clear_dialog_summary(session_id: str):
    """删除对话总结"""
    try:
        summary_manager.clear_summary(session_id)
        return {"status": "success", "message": "对话总结已删除", "session_id": session_id}
    except Exception as e:
        logger.error(f"删除对话总结错误: {e}")
        raise HTTPException(status_code=500, detail="删除对话总结错误")


# ==================== 启动 ====================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=config.LOG_LEVEL.lower(),
    )
