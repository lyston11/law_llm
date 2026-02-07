<template>
  <div :class="['chat-message', msg.role === 'user' ? 'user-msg' : 'bot-msg']">
    <div class="avatar">
      <el-avatar :size="36" :style="{ background: msg.role === 'user' ? '#409eff' : '#67c23a' }">
        <el-icon v-if="msg.role === 'user'"><User /></el-icon>
        <el-icon v-else><Service /></el-icon>
      </el-avatar>
    </div>
    <div class="bubble-wrap">
      <div class="meta">
        <span class="name">{{ msg.role === 'user' ? '我' : '智能助手' }}</span>
        <span class="time">{{ msg.time }}</span>
      </div>

      <!-- Agent 工具调用过程 -->
      <div v-if="msg.agentActions && msg.agentActions.length" class="agent-actions">
        <div class="action-header">
          <el-icon><Operation /></el-icon>
          <span>Agent 推理过程</span>
        </div>
        <div v-for="(action, i) in msg.agentActions" :key="i" class="action-item">
          <el-tag size="small" :type="toolTagType(action.tool)" effect="dark" class="tool-tag">
            {{ toolLabel(action.tool) }}
          </el-tag>
          <span class="action-summary">{{ action.result_summary }}</span>
        </div>
      </div>

      <!-- 消息内容 -->
      <div class="bubble" v-html="formatted"></div>

      <!-- RAG 来源标注 -->
      <div v-if="msg.sources && msg.sources.length" class="rag-sources">
        <span class="sources-label">参考来源:</span>
        <el-tag v-for="(s, i) in msg.sources" :key="i" size="small" type="info" effect="plain" class="source-tag">
          {{ s.domain }} ({{ (s.score * 100).toFixed(0) }}%)
        </el-tag>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  msg: { type: Object, required: true },
})

const formatted = computed(() => {
  if (!props.msg.content) return ''
  return props.msg.content
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>')
    .replace(/⚖️/g, '<span style="font-size:1.2em">⚖️</span>')
})

const toolLabels = {
  search_legal_knowledge: '知识库检索',
  lookup_legal_article: '法条查询',
  query_knowledge_graph: '知识图谱',
  analyze_legal_situation: '场景分析',
}

const toolTagTypes = {
  search_legal_knowledge: 'primary',
  lookup_legal_article: 'success',
  query_knowledge_graph: 'warning',
  analyze_legal_situation: 'info',
}

function toolLabel(name) {
  return toolLabels[name] || name
}

function toolTagType(name) {
  return toolTagTypes[name] || ''
}
</script>

<style scoped>
.chat-message {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
  max-width: 85%;
}
.user-msg {
  flex-direction: row-reverse;
  margin-left: auto;
}
.bot-msg {
  margin-right: auto;
}
.bubble-wrap {
  flex: 1;
  min-width: 0;
}
.meta {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
  font-size: 12px;
  color: #909399;
}
.user-msg .meta {
  flex-direction: row-reverse;
}
.name {
  font-weight: 500;
  color: #606266;
}

/* Agent 推理过程 */
.agent-actions {
  margin-bottom: 8px;
  padding: 8px 12px;
  background: #f0f9eb;
  border-radius: 8px;
  border: 1px solid #e1f3d8;
}
.action-header {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  font-weight: 600;
  color: #67c23a;
  margin-bottom: 6px;
}
.action-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 3px 0;
  font-size: 12px;
}
.tool-tag {
  flex-shrink: 0;
}
.action-summary {
  color: #606266;
  font-size: 12px;
}

/* 消息气泡 */
.bubble {
  padding: 12px 16px;
  border-radius: 12px;
  line-height: 1.6;
  font-size: 14px;
  word-break: break-word;
}
.user-msg .bubble {
  background: #409eff;
  color: #fff;
  border-top-right-radius: 4px;
}
.bot-msg .bubble {
  background: #f4f4f5;
  color: #303133;
  border-top-left-radius: 4px;
}

/* 来源标注 */
.rag-sources {
  margin-top: 6px;
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  align-items: center;
}
.sources-label {
  font-size: 11px;
  color: #909399;
}
.source-tag {
  font-size: 11px;
}
</style>
