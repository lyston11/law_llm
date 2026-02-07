"""RAG（检索增强生成）模块

基于向量数据库的法律知识检索，结合大模型 API 生成专业法律回答。
核心流程：用户提问 → BGE 向量编码 → ChromaDB 检索 Top-K → 构建增强 Prompt → 大模型生成回答
"""
import os
import time
from config import Config

config = Config()


class RAGRetriever:
    """
    RAG 检索器，负责从向量知识库中检索相关法律文档，
    并构建增强型 Prompt 供大模型生成高质量回答。
    """

    def __init__(self):
        self.embeddings = None
        self.vector_db = None
        self.is_ready = False
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """加载向量知识库和 Embedding 模型"""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_chroma import Chroma

            vector_db_dir = config.KNOWLEDGE_BASE_DIR
            if not os.path.exists(vector_db_dir):
                print(f"⚠️  向量知识库目录不存在: {vector_db_dir}")
                return

            print("正在加载 RAG 向量知识库...")
            start_time = time.time()

            # 加载 BGE Embedding 模型（必须与构建知识库时使用的模型一致）
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.RAG_EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            # 加载 ChromaDB 向量数据库
            self.vector_db = Chroma(
                embedding_function=self.embeddings,
                persist_directory=vector_db_dir,
            )

            doc_count = self.vector_db._collection.count()
            elapsed = time.time() - start_time
            print(f"✅ RAG 知识库加载完成: {doc_count} 条文档, 耗时 {elapsed:.2f}s")
            self.is_ready = True

        except ImportError as e:
            print(f"⚠️  RAG 依赖未安装: {e}")
            print("   请运行: pip install langchain-huggingface langchain-chroma chromadb")
        except Exception as e:
            print(f"⚠️  RAG 知识库加载失败: {e}")

    def retrieve(self, query, k=None, domain=None):
        """
        从向量知识库中检索与查询最相关的文档

        Args:
            query (str): 用户查询文本
            k (int): 返回的文档数量，默认使用配置值
            domain (str): 可选的法律领域过滤（如 "劳动纠纷"）

        Returns:
            list[dict]: 检索到的文档列表，每项包含 content, metadata, score
        """
        if not self.is_ready:
            return []

        if k is None:
            k = config.RAG_TOP_K

        try:
            start_time = time.time()

            # 带相似度分数的检索
            if domain:
                # 按法律领域过滤检索
                results = self.vector_db.similarity_search_with_relevance_scores(
                    query,
                    k=k,
                    filter={"domain": domain},
                )
            else:
                results = self.vector_db.similarity_search_with_relevance_scores(
                    query, k=k
                )

            elapsed = time.time() - start_time

            # 格式化结果
            retrieved_docs = []
            for doc, score in results:
                retrieved_docs.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": round(score, 4),
                    }
                )

            print(
                f"  RAG 检索完成: {len(retrieved_docs)} 条结果, 耗时 {elapsed:.3f}s"
            )
            return retrieved_docs

        except Exception as e:
            print(f"⚠️  RAG 检索失败: {e}")
            return []

    def build_rag_prompt(self, user_input, retrieved_docs, legal_type=None, dialog_history=None):
        """
        基于检索结果构建增强型 Prompt

        Args:
            user_input (str): 用户输入
            retrieved_docs (list): 检索到的法律文档
            legal_type (str): 识别到的法律类型
            dialog_history (list): 对话历史

        Returns:
            str: 构建好的增强 Prompt
        """
        # 1. 系统角色设定
        system_instruction = (
            "你是一位专业的法律智能助手。请严格根据【参考法律资料】中的内容回答用户问题。\n"
            "回答要求：\n"
            "1. 必须基于参考资料中的法律条文和案例进行回答，明确引用来源\n"
            "2. 使用通俗易懂的语言解释法律条文\n"
            "3. 如果参考资料不足以完整回答问题，请明确说明\n"
            "4. 回答结尾附上法律免责声明\n"
            "5. 使用中文回答，控制在 500 字以内\n"
        )

        # 2. 对话历史上下文
        history_context = ""
        if dialog_history and len(dialog_history) > 0:
            recent_history = dialog_history[-3:]  # 最近 3 轮
            history_parts = []
            for turn in recent_history:
                history_parts.append(f"用户: {turn.get('user_input', '')}")
                history_parts.append(f"助手: {turn.get('system_response', '')}")
            history_context = f"\n【对话历史】\n{''.join(h + chr(10) for h in history_parts)}\n"

        # 3. 检索到的法律参考资料
        if retrieved_docs:
            ref_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                domain = doc["metadata"].get("domain", "未知领域")
                score = doc["score"]
                content = doc["content"][:400]  # 限制每条长度
                ref_parts.append(
                    f"[资料{i}] (领域: {domain}, 相关度: {score})\n{content}"
                )
            reference_context = "\n【参考法律资料】\n" + "\n\n".join(ref_parts) + "\n"
        else:
            reference_context = "\n【参考法律资料】\n暂无相关参考资料。\n"

        # 4. 法律领域提示
        domain_hint = ""
        if legal_type:
            domain_hint = f"\n【识别到的法律领域】{legal_type}\n"

        # 5. 用户问题
        user_question = f"\n【用户问题】\n{user_input}\n"

        # 6. 输出格式要求
        format_instruction = (
            "\n请按以下格式回答：\n"
            "1. 先给出直接回答\n"
            "2. 引用相关法律条文（如有）\n"
            "3. 给出实用建议\n"
            "4. 最后附上：\n"
            "   ⚖️ 免责声明：以上内容基于公开法律资料生成，仅供参考，不构成法律意见。如需专业帮助，请咨询执业律师。\n"
        )

        # 组合完整 Prompt
        prompt = (
            system_instruction
            + history_context
            + reference_context
            + domain_hint
            + user_question
            + format_instruction
        )

        return prompt

    def get_status(self):
        """
        获取 RAG 知识库状态信息

        Returns:
            dict: 知识库状态
        """
        if not self.is_ready:
            return {
                "status": "unavailable",
                "message": "RAG 知识库未加载",
                "doc_count": 0,
            }

        try:
            doc_count = self.vector_db._collection.count()
            return {
                "status": "ready",
                "message": "RAG 知识库已就绪",
                "doc_count": doc_count,
                "embedding_model": config.RAG_EMBEDDING_MODEL,
                "top_k": config.RAG_TOP_K,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"获取状态失败: {e}",
                "doc_count": 0,
            }
