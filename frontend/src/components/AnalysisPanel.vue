<template>
  <div class="analysis-panel">
    <!-- 系统状态 -->
    <div class="panel-section">
      <div class="section-title">系统状态</div>
      <div class="info-row">
        <span class="label">服务状态</span>
        <el-tag :type="status.healthy ? 'success' : 'danger'" size="small">
          {{ status.healthy ? '正常' : '离线' }}
        </el-tag>
      </div>
      <div class="info-row">
        <span class="label">法律知识库</span>
        <el-tag :type="status.ragReady ? 'success' : 'warning'" size="small">
          {{ status.ragReady ? `${status.docCount} 条文档` : '未加载' }}
        </el-tag>
      </div>
      <div class="info-row">
        <span class="label">Agent 模型</span>
        <span class="value">{{ status.model || '-' }}</span>
      </div>
    </div>

    <!-- Agent 工具 -->
    <div class="panel-section">
      <div class="section-title">Agent 工具</div>
      <div v-if="status.tools && status.tools.length" class="tools-list">
        <el-tag
          v-for="tool in status.tools"
          :key="tool"
          size="small"
          :type="toolTagType(tool)"
          effect="plain"
          class="tool-item"
        >
          {{ toolLabel(tool) }}
        </el-tag>
      </div>
      <span v-else class="value">-</span>
    </div>

    <!-- 本轮 Agent 行为 -->
    <div class="panel-section">
      <div class="section-title">本轮分析</div>
      <div v-if="analysis.actions && analysis.actions.length">
        <div v-for="(action, i) in analysis.actions" :key="i" class="action-detail">
          <div class="action-tool">
            <el-tag size="small" :type="toolTagType(action.tool)" effect="dark">
              {{ toolLabel(action.tool) }}
            </el-tag>
          </div>
          <div class="action-input" v-if="action.input">
            <span class="detail-label">输入:</span>
            <span class="detail-value">{{ formatInput(action.input) }}</span>
          </div>
          <div class="action-result">
            <span class="detail-label">结果:</span>
            <span class="detail-value">{{ action.result_summary }}</span>
          </div>
        </div>
      </div>
      <div v-else class="info-row">
        <span class="value" style="color: #909399">等待对话...</span>
      </div>
    </div>

    <!-- 反馈 -->
    <div class="panel-section">
      <div class="section-title">评价反馈</div>
      <div class="rating-row">
        <el-rate v-model="rating" :max="5" />
      </div>
      <el-input
        v-model="comment"
        type="textarea"
        :rows="2"
        placeholder="对本次回答的建议..."
        size="small"
      />
      <el-button type="primary" size="small" style="margin-top: 8px; width: 100%" @click="submit">
        提交反馈
      </el-button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  status: {
    type: Object,
    default: () => ({
      healthy: false, ragReady: false, docCount: 0, model: '', tools: [],
    }),
  },
  analysis: {
    type: Object,
    default: () => ({ actions: [], sources: [] }),
  },
})

const emit = defineEmits(['feedback'])

const rating = ref(0)
const comment = ref('')

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

function formatInput(input) {
  if (!input) return '-'
  const parts = []
  if (input.query) parts.push(input.query)
  if (input.law_name) parts.push(input.law_name + (input.article_number || ''))
  if (input.entity) parts.push(input.entity)
  if (input.domain) parts.push(`[${input.domain}]`)
  if (input.text) parts.push(input.text.substring(0, 30) + (input.text.length > 30 ? '...' : ''))
  return parts.join(' ') || JSON.stringify(input).substring(0, 50)
}

function submit() {
  if (!rating.value) return
  emit('feedback', { rating: rating.value, comment: comment.value })
  rating.value = 0
  comment.value = ''
}
</script>

<style scoped>
.panel-section {
  margin-bottom: 20px;
}
.section-title {
  font-size: 13px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 10px;
  padding-bottom: 6px;
  border-bottom: 1px solid #ebeef5;
}
.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 4px 0;
  font-size: 13px;
}
.label {
  color: #909399;
}
.value {
  color: #303133;
  text-align: right;
  max-width: 60%;
  font-size: 12px;
}

/* 工具列表 */
.tools-list {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.tool-item {
  font-size: 11px;
}

/* Agent 行为详情 */
.action-detail {
  padding: 8px;
  margin-bottom: 8px;
  background: #fafafa;
  border-radius: 6px;
  border: 1px solid #f0f0f0;
}
.action-tool {
  margin-bottom: 4px;
}
.detail-label {
  font-size: 11px;
  color: #909399;
  margin-right: 4px;
}
.detail-value {
  font-size: 11px;
  color: #606266;
  word-break: break-all;
}
.action-input,
.action-result {
  padding: 2px 0;
}

.rating-row {
  margin-bottom: 8px;
}
</style>
