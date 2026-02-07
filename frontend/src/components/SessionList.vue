<template>
  <div class="session-list">
    <div class="session-header">
      <span class="title">会话列表</span>
      <el-button type="primary" :icon="Plus" size="small" circle @click="$emit('create')" />
    </div>
    <el-scrollbar height="calc(100vh - 200px)">
      <div
        v-for="s in sessions"
        :key="s.id"
        :class="['session-item', { active: s.id === currentId }]"
        @click="$emit('switch', s.id)"
      >
        <div class="session-name">
          <el-icon><ChatDotRound /></el-icon>
          <span>{{ s.name }}</span>
        </div>
        <div class="session-meta">
          <span class="time">{{ formatTime(s.lastActive) }}</span>
          <el-button
            type="danger"
            :icon="Delete"
            size="small"
            link
            @click.stop="$emit('delete', s.id)"
          />
        </div>
      </div>
      <el-empty v-if="!sessions.length" description="暂无会话" :image-size="60" />
    </el-scrollbar>
  </div>
</template>

<script setup>
import { Plus, Delete, ChatDotRound } from '@element-plus/icons-vue'

defineProps({
  sessions: { type: Array, default: () => [] },
  currentId: { type: String, default: '' },
})

defineEmits(['create', 'switch', 'delete'])

function formatTime(ts) {
  if (!ts) return ''
  return new Date(ts).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}
</script>

<style scoped>
.session-list {
  height: 100%;
}
.session-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}
.title {
  font-weight: 600;
  font-size: 15px;
  color: #303133;
}
.session-item {
  padding: 10px 12px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  margin-bottom: 4px;
}
.session-item:hover {
  background: #ecf5ff;
}
.session-item.active {
  background: #409eff;
  color: #fff;
}
.session-item.active .time {
  color: rgba(255, 255, 255, 0.7);
}
.session-name {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  margin-bottom: 2px;
}
.session-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.time {
  font-size: 11px;
  color: #909399;
}
</style>
