/**
 * API 接口封装（v3.0 Agent 架构）
 */
import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 60000, // Agent 可能需要多轮工具调用，超时调大
  headers: { 'Content-Type': 'application/json' },
})

// 健康检查
export const healthCheck = () => api.get('/health')

// 对话（返回 response + agent_actions + sources）
export const sendMessage = (userInput, sessionId) =>
  api.post('/dialog', { user_input: userInput, session_id: sessionId })

// 重置对话
export const resetDialog = (sessionId) => api.delete(`/dialog/${sessionId}`)

// 对话历史
export const getHistory = (sessionId, limit = 50) =>
  api.get(`/dialog/${sessionId}/history`, { params: { limit } })

// 对话总结
export const getSummary = (sessionId) =>
  api.post(`/dialog/${sessionId}/summary`)

// 反馈
export const submitFeedback = (data) => api.post('/feedback', data)

// RAG 知识库
export const getKnowledgeStatus = () => api.get('/knowledge/status')
export const searchKnowledge = (query, topK = 5, domain = null) =>
  api.post('/knowledge/search', { query, top_k: topK, domain })

// Agent 状态
export const getAgentStatus = () => api.get('/agent/status')

export default api
