#!/bin/bash
# ============================================================
#  智能领域聊天机器人 — 数据还原脚本
#
#  用途：将 Git 仓库中的 .tar.gz 压缩包还原为原始数据目录
#  使用：bash restore_data.sh
#
#  本脚本会自动完成以下操作：
#    1. 合并拆分的压缩包（.part_aa, .part_ab, ...）
#    2. 解压到正确的目录位置
#    3. 验证还原结果
#
#  还原后的目录结构：
#    models/bge-small-zh-v1.5/     ← BGE 中文嵌入模型（RAG 向量化用）
#    data/knowledge_base/vector_db/ ← ChromaDB 向量知识库（法律知识检索用）
#    data/knowledge_graph/          ← 法律知识图谱（概念关系查询用）
#    data/legal/datasets/           ← 法律数据集源文件（构建知识库用）
# ============================================================

set -e

# ---------- 颜色 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $1"; exit 1; }

# ---------- 定位项目根目录 ----------
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  智能领域聊天机器人 — 数据还原脚本${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
info "项目目录: $PROJECT_DIR"
echo ""

# ---------- 函数：合并分片并解压 ----------
restore_split() {
    local parts_pattern="$1"   # 分片文件模式，如 models/bge-small-zh-v1.5.tar.gz.part_
    local tar_file="$2"        # 合并后的 tar.gz 文件名
    local extract_dir="$3"     # 解压到哪个目录
    local description="$4"     # 描述

    echo ""
    info "========== 还原: $description =========="

    # 检查分片文件是否存在
    if ! ls ${parts_pattern}* 1>/dev/null 2>&1; then
        warn "分片文件不存在: ${parts_pattern}*，跳过"
        return
    fi

    # 检查目标目录是否已存在
    # 从 tar_file 推断目标目录名
    if [ -d "$extract_dir" ] && [ "$(ls -A "$extract_dir" 2>/dev/null)" ]; then
        ok "$description 已存在，跳过（如需重新还原，请先删除 $extract_dir）"
        return
    fi

    # 合并分片
    info "合并分片文件..."
    cat ${parts_pattern}* > "$tar_file"
    ok "合并完成: $tar_file ($(du -h "$tar_file" | cut -f1))"

    # 解压
    info "解压到 $(dirname "$extract_dir")/ ..."
    mkdir -p "$(dirname "$extract_dir")"
    tar -xzf "$tar_file" -C "$(dirname "$extract_dir")"
    ok "解压完成: $extract_dir"

    # 清理临时 tar.gz
    rm -f "$tar_file"

    # 验证
    if [ -d "$extract_dir" ]; then
        local count=$(find "$extract_dir" -type f 2>/dev/null | wc -l | tr -d ' ')
        ok "验证通过: $extract_dir ($count 个文件)"
    else
        fail "验证失败: $extract_dir 目录不存在"
    fi
}

# ---------- 函数：解压单个 tar.gz ----------
restore_single() {
    local tar_file="$1"
    local extract_dir="$2"
    local target_dir="$3"
    local description="$4"

    echo ""
    info "========== 还原: $description =========="

    if [ ! -f "$tar_file" ]; then
        warn "压缩包不存在: $tar_file，跳过"
        return
    fi

    if [ -d "$extract_dir" ] && [ "$(ls -A "$extract_dir" 2>/dev/null)" ]; then
        ok "$description 已存在，跳过"
        return
    fi

    info "解压 $tar_file ..."
    mkdir -p "$target_dir"
    tar -xzf "$tar_file" -C "$target_dir"
    ok "解压完成: $extract_dir"

    if [ -d "$extract_dir" ]; then
        local count=$(find "$extract_dir" -type f 2>/dev/null | wc -l | tr -d ' ')
        ok "验证通过: $extract_dir ($count 个文件)"
    else
        fail "验证失败: $extract_dir 目录不存在"
    fi
}

# ==================== 开始还原 ====================

# 1. BGE 嵌入模型（拆分的）
restore_split \
    "models/bge-small-zh-v1.5.tar.gz.part_" \
    "models/bge-small-zh-v1.5.tar.gz" \
    "models/bge-small-zh-v1.5" \
    "BGE 嵌入模型 (models/bge-small-zh-v1.5/)"

# 2. ChromaDB 向量知识库（拆分的）
restore_split \
    "data/knowledge_base.tar.gz.part_" \
    "data/knowledge_base.tar.gz" \
    "data/knowledge_base" \
    "RAG 向量知识库 (data/knowledge_base/)"

# 3. 法律知识图谱（拆分的）
restore_split \
    "data/knowledge_graph.tar.gz.part_" \
    "data/knowledge_graph.tar.gz" \
    "data/knowledge_graph" \
    "法律知识图谱 (data/knowledge_graph/)"

# 4. 法律数据集（单个文件，未拆分）
restore_single \
    "data/legal/datasets.tar.gz" \
    "data/legal/datasets" \
    "data/legal" \
    "法律数据集 (data/legal/datasets/)"

# ==================== 完成 ====================

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  数据还原完成！${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "  还原的目录："
echo -e "    ${BLUE}models/bge-small-zh-v1.5/${NC}      BGE 嵌入模型"
echo -e "    ${BLUE}data/knowledge_base/${NC}           RAG 向量知识库"
echo -e "    ${BLUE}data/knowledge_graph/${NC}          法律知识图谱"
echo -e "    ${BLUE}data/legal/datasets/${NC}           法律数据集"
echo ""
echo "  下一步："
echo "    1. 配置 API Key:  编辑 .env 文件，填入你的阿里云百炼 API Key"
echo "    2. 安装依赖:      pip install -r requirements.txt"
echo "    3. 安装前端依赖:  cd frontend && npm install && cd .."
echo "    4. 启动系统:      bash start.sh"
echo ""
