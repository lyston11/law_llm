# 系统启动指南

## 快速启动

### 1. 启动后端服务

```bash
conda activate bishe
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### 2. 启动前端服务（另一个终端）

```bash
cd frontend
npm run dev
```

## 访问地址

| 服务 | 地址 |
|------|------|
| 前端界面 | http://localhost:3000 |
| 后端 API | http://localhost:8000 |
| API 文档 (Swagger) | http://localhost:8000/docs |
| API 文档 (ReDoc) | http://localhost:8000/redoc |

## 停止服务

在每个终端窗口中按 `Ctrl + C` 即可。

## 常见问题

### 端口被占用

```bash
# 查看占用端口的进程
lsof -i :8000
# 终止进程
kill -9 <PID>
```

### 知识库未加载

确保已运行知识库构建脚本：

```bash
python scripts/build_rag_knowledge_base.py
```
