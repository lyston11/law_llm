#!/bin/bash
# ============================================================
#  智能领域聊天机器人 — 一键启动脚本
#  用法: bash start.sh
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
info "项目目录: $PROJECT_DIR"

# ---------- 1. 检查 .env ----------
echo ""
info "========== 第 1 步: 检查环境变量 =========="
if [ ! -f .env ]; then
    warn ".env 文件不存在，正在从模板创建..."
    cp .env.example .env
    warn "请编辑 .env 文件，填入你的 API_KEY 后重新运行此脚本"
    warn "  文件位置: $PROJECT_DIR/.env"
    warn "  获取 API Key: https://bailian.console.aliyun.com/"
    exit 1
fi

# 检查 API_KEY 是否已填写
API_KEY=$(grep -E "^API_KEY=" .env | cut -d'=' -f2)
if [ -z "$API_KEY" ] || [ "$API_KEY" = "sk-your-api-key-here" ]; then
    fail "请先在 .env 文件中填入真实的 API_KEY\n       获取地址: https://bailian.console.aliyun.com/"
fi
ok ".env 配置已就绪"

# ---------- 2. 检查 Python ----------
echo ""
info "========== 第 2 步: 检查 Python 环境 =========="
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    fail "未找到 Python，请安装 Python 3.10+"
fi

PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    fail "Python 版本需要 >= 3.10，当前: $PYTHON_VERSION"
fi
ok "Python $PYTHON_VERSION ($PYTHON_CMD)"

# ---------- 3. 检查 Node.js ----------
echo ""
info "========== 第 3 步: 检查 Node.js 环境 =========="
if ! command -v node &> /dev/null; then
    fail "未找到 Node.js，请安装 Node.js 18+\n       下载地址: https://nodejs.org/"
fi
if ! command -v npm &> /dev/null; then
    fail "未找到 npm，请安装 Node.js 18+"
fi

NODE_VERSION=$(node --version | sed 's/v//')
NODE_MAJOR=$(echo "$NODE_VERSION" | cut -d. -f1)
if [ "$NODE_MAJOR" -lt 18 ]; then
    warn "Node.js 版本建议 >= 18，当前: $NODE_VERSION"
fi
ok "Node.js v$NODE_VERSION + npm $(npm --version)"

# ---------- 4. 安装 Python 依赖 ----------
echo ""
info "========== 第 4 步: 安装 Python 依赖 =========="
if $PYTHON_CMD -c "import fastapi, uvicorn, requests, dotenv" 2>/dev/null; then
    ok "Python 核心依赖已安装"
else
    info "正在安装 Python 依赖（首次可能需要几分钟）..."
    $PYTHON_CMD -m pip install -r requirements.txt -q
    ok "Python 依赖安装完成"
fi

# ---------- 5. 安装前端依赖 ----------
echo ""
info "========== 第 5 步: 安装前端依赖 =========="
if [ -d frontend/node_modules ]; then
    ok "前端依赖已安装"
else
    info "正在安装前端依赖..."
    cd frontend && npm install --silent && cd ..
    ok "前端依赖安装完成"
fi

# ---------- 6. 检查模型文件 ----------
echo ""
info "========== 第 6 步: 检查模型文件 =========="
if [ -d models/bge-small-zh-v1.5 ]; then
    ok "BGE 嵌入模型已存在"
else
    warn "BGE 嵌入模型未下载（知识库检索功能将不可用）"
    warn "如需法律知识库功能，请运行:"
    warn "  git clone https://hf-mirror.com/BAAI/bge-small-zh-v1.5 models/bge-small-zh-v1.5"
    warn "系统仍可正常启动，LLM 通用问答功能可用"
fi

# ---------- 7. 检查知识库 ----------
echo ""
info "========== 第 7 步: 检查知识库 =========="
if [ -d data/knowledge_base/vector_db ] && [ "$(ls -A data/knowledge_base/vector_db 2>/dev/null)" ]; then
    ok "RAG 向量知识库已构建"
else
    warn "RAG 向量知识库未构建（法律检索功能将不可用）"
    warn "如需构建，请运行: $PYTHON_CMD scripts/build_rag_knowledge_base.py"
    warn "系统仍可正常启动，LLM 通用问答功能可用"
fi

if [ -f data/knowledge_graph/law_knowledge_graph.graphml ]; then
    ok "法律知识图谱已构建"
else
    warn "法律知识图谱未构建"
    warn "如需构建，请运行: $PYTHON_CMD scripts/build_knowledge_graph.py"
fi

# ---------- 8. 清理旧进程 ----------
echo ""
info "========== 第 8 步: 检查端口 =========="
kill_port() {
    local port=$1
    local pid=$(lsof -ti :$port 2>/dev/null)
    if [ -n "$pid" ]; then
        warn "端口 $port 被占用 (PID: $pid)，正在释放..."
        kill -9 $pid 2>/dev/null || true
        sleep 1
        ok "端口 $port 已释放"
    else
        ok "端口 $port 可用"
    fi
}
kill_port 8000
kill_port 3000

# ---------- 9. 启动后端 ----------
echo ""
info "========== 第 9 步: 启动后端服务 =========="
$PYTHON_CMD -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
info "后端启动中 (PID: $BACKEND_PID)..."

# 等待后端就绪
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        ok "后端服务已就绪: http://127.0.0.1:8000"
        break
    fi
    if [ $i -eq 30 ]; then
        fail "后端启动超时，请检查日志"
    fi
    sleep 1
done

# ---------- 10. 启动前端 ----------
echo ""
info "========== 第 10 步: 启动前端服务 =========="
cd frontend && npm run dev &
FRONTEND_PID=$!
cd "$PROJECT_DIR"
info "前端启动中 (PID: $FRONTEND_PID)..."
sleep 3

# ---------- 启动完成 ----------
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  智能领域聊天机器人 — 启动成功！${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "  前端界面:       ${BLUE}http://localhost:3000${NC}"
echo -e "  后端 API:       ${BLUE}http://localhost:8000${NC}"
echo -e "  API 文档:       ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo -e "  后端 PID: $BACKEND_PID"
echo -e "  前端 PID: $FRONTEND_PID"
echo ""
echo -e "  按 ${YELLOW}Ctrl+C${NC} 停止所有服务"
echo ""

# ---------- 优雅退出 ----------
cleanup() {
    echo ""
    info "正在停止服务..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    ok "所有服务已停止"
    exit 0
}
trap cleanup SIGINT SIGTERM

# 保持前台运行
wait
