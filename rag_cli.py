#!/usr/bin/python3.10
"""RAG 向量检索 CLI —— 导入文档 & 语义查询 & API 服务（优化版）

优化特性:
  1. 内容哈希去重 —— 相同内容的文档块只保留一份，合并来源路径
  2. Markdown 感知分块 —— 以标题为边界，注入父标题上下文
  3. Reranker 重排序 —— BGE-Reranker-v2-M3 对召回结果精排
  4. 混合检索 (BM25 + 向量) —— RRF 倒数排名融合，互补稀疏/稠密检索
  5. HTTP API 服务 —— FastAPI 提供 /query 接口，支持 SSE 流式输出
     · 启动时预加载所有模型和索引，请求时零延迟复用

依赖安装:
  pip install langchain langchain-huggingface langchain-chroma \\
              langchain-text-splitters chromadb sentence-transformers \\
              rank_bm25 fastapi uvicorn sse-starlette

用法:
  python rag_cli.py import        <目录路径> [选项]
  python rag_cli.py query         "查询内容"  [选项]
  python rag_cli.py query-server  [选项]
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from tqdm import tqdm

# jieba 缓存目录（必须在 import jieba 之前设置）
os.environ.setdefault("JIEBA_CACHE_DIR", os.path.expanduser("~/.cache/jieba"))

# 抑制无关紧要的警告
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated.*HF Hub.*")
warnings.filterwarnings("ignore", message=".*sending unauthenticated.*")
try:
    from langchain_core._api import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass

# ─────────────────────────── 常量 ───────────────────────────

DEFAULT_DB_DIR = "./chroma_db"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 10
DEFAULT_RECALL_K = 30
DEFAULT_BM25_K = 30
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
BM25_INDEX_FILE = "bm25_index.json"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

SUPPORTED_EXTS = {
    ".txt", ".md", ".html", ".htm", ".csv", ".json", ".jsonl",
    ".pdf", ".docx", ".pptx", ".xlsx", ".xls",
    ".py", ".go", ".java", ".js", ".ts", ".c", ".cpp", ".h",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".xml",
    ".rst", ".tex", ".log", ".sh", ".bat", ".sql",
}


# ─────────────────── 依赖检查 ───────────────────

def check_dependencies(extra: list[str] | None = None):
    """检查关键依赖是否已安装，未安装则给出提示。"""
    deps = {
        "langchain": "langchain",
        "langchain_huggingface": "langchain-huggingface",
        "langchain_chroma": "langchain-chroma",
        "langchain_text_splitters": "langchain-text-splitters",
        "chromadb": "chromadb",
        "sentence_transformers": "sentence-transformers",
    }
    if extra:
        for mod, pkg in [
            ("rank_bm25", "rank_bm25"),
            ("fastapi", "fastapi"),
            ("uvicorn", "uvicorn"),
            ("sse_starlette", "sse-starlette"),
        ]:
            if mod in extra:
                deps[mod] = pkg

    missing = []
    for mod, pkg in deps.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"❌ 缺少依赖: {', '.join(missing)}")
        print(f"   请运行: pip install {' '.join(missing)}")
        sys.exit(1)


# ─────────────────── 文件发现 ───────────────────

def discover_files(directory: str) -> list[str]:
    """递归扫描目录，返回所有支持的文件路径列表。"""
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for fn in sorted(filenames):
            if Path(fn).suffix.lower() in SUPPORTED_EXTS:
                files.append(os.path.join(dirpath, fn))
    return sorted(files)


# ─────────────────── 文档加载 ───────────────────

def _build_loader_map() -> dict:
    """构建扩展名 → Loader 类的映射。"""
    from langchain_community.document_loaders import (
        TextLoader, PyPDFLoader, CSVLoader, JSONLoader, PythonLoader,
    )

    loader_map: dict = {
        ".txt": TextLoader, ".rst": TextLoader,
        ".csv": CSVLoader, ".json": JSONLoader, ".jsonl": JSONLoader,
        ".pdf": PyPDFLoader, ".py": PythonLoader,
        ".go": TextLoader, ".java": TextLoader, ".js": TextLoader,
        ".ts": TextLoader, ".c": TextLoader, ".cpp": TextLoader,
        ".h": TextLoader, ".yaml": TextLoader, ".yml": TextLoader,
        ".toml": TextLoader, ".ini": TextLoader, ".cfg": TextLoader,
        ".conf": TextLoader, ".xml": TextLoader, ".tex": TextLoader,
        ".log": TextLoader, ".sh": TextLoader, ".bat": TextLoader,
        ".sql": TextLoader,
    }

    try:
        import unstructured  # noqa: F401
        from langchain_community.document_loaders import (
            UnstructuredHTMLLoader, UnstructuredMarkdownLoader,
        )
        loader_map[".md"] = UnstructuredMarkdownLoader
        loader_map[".html"] = UnstructuredHTMLLoader
        loader_map[".htm"] = UnstructuredHTMLLoader
    except ImportError:
        pass

    for ext, cls_name in [
        (".docx", "Docx2txtLoader"),
        (".xlsx", "UnstructuredExcelLoader"),
        (".xls", "UnstructuredExcelLoader"),
        (".pptx", "UnstructuredPowerPointLoader"),
    ]:
        try:
            from langchain_community import document_loaders as dl
            loader_map[ext] = getattr(dl, cls_name)
        except (ImportError, AttributeError):
            pass

    return loader_map


def load_documents(files: list[str]):
    """根据文件扩展名选择对应 Loader 加载文档，返回 Document 列表。"""
    loader_map = _build_loader_map()
    TextLoader = loader_map[".txt"]

    all_docs = []
    failed = []

    for fp in tqdm(files, desc="📄 加载文档", unit="file", ncols=80):
        ext = Path(fp).suffix.lower()
        loader_cls = loader_map.get(ext, TextLoader)
        try:
            loader = loader_cls(fp)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = fp
                doc.metadata["filename"] = os.path.basename(fp)
            all_docs.extend(docs)
        except Exception as e:
            failed.append((fp, str(e)))

    if failed:
        print(f"\n⚠ {len(failed)} 个文件加载失败:")
        for fp, err in failed:
            print(f"    {os.path.basename(fp)}: {err}")

    return all_docs


# ═══════════════════════════════════════════════════════════════
# 优化 2: Markdown 感知分块 + 父标题注入
# ═══════════════════════════════════════════════════════════════

def _extract_md_sections(text: str) -> list[dict]:
    """将 Markdown 文本按标题拆分为语义段落。"""
    lines = text.split("\n")
    sections = []
    heading_stack: dict[int, str] = {}
    current_body: list[str] = []

    def _flush():
        nonlocal current_body
        body = "\n".join(current_body).strip()
        if body:
            chain = " > ".join(heading_stack[k] for k in sorted(heading_stack))
            sections.append({"title": chain, "body": body})
        current_body = []

    heading_re = re.compile(r"^(#{1,6})\s+(.+)$")

    for line in lines:
        m = heading_re.match(line.strip())
        if m:
            _flush()
            level = len(m.group(1))
            title_text = m.group(2).strip()
            heading_stack[level] = title_text
            for k in list(heading_stack):
                if k > level:
                    del heading_stack[k]
        else:
            current_body.append(line)

    _flush()
    return sections


def _content_hash(text: str) -> str:
    """计算文本内容的 MD5 哈希。"""
    normalized = re.sub(r"\s+", " ", text.strip())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def split_documents_markdown(docs, chunk_size: int, chunk_overlap: int):
    """Markdown 感知分块：以标题为边界，注入父标题上下文。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";", " ", ""],
    )

    chunks = []

    for doc in tqdm(docs, desc="✂️  切分文本", unit="doc", ncols=80):
        source = doc.metadata.get("source", "")
        ext = Path(source).suffix.lower() if source else ""
        filename = doc.metadata.get("filename", "")

        if ext == ".md":
            sections = _extract_md_sections(doc.page_content)
            if sections:
                for sec in sections:
                    if sec["title"]:
                        chunk_text = f"【{sec['title']}】\n{sec['body']}"
                    else:
                        chunk_text = sec["body"]

                    if len(chunk_text) > chunk_size * 2:
                        sub_docs = fallback_splitter.split_text(chunk_text)
                        for sub in sub_docs:
                            chunks.append(_make_chunk(sub, doc.metadata, filename, source))
                    else:
                        chunks.append(_make_chunk(chunk_text, doc.metadata, filename, source))
            else:
                for c in fallback_splitter.split_documents([doc]):
                    chunks.append(c)
        else:
            for c in fallback_splitter.split_documents([doc]):
                chunks.append(c)

    return chunks


def _make_chunk(text, base_meta, filename, source):
    """创建带元数据的 chunk。"""
    from langchain_core.documents import Document
    meta = dict(base_meta)
    meta["source"] = source
    meta["filename"] = filename
    return Document(page_content=text, metadata=meta)


# ═══════════════════════════════════════════════════════════════
# 优化 1: 内容哈希去重
# ═══════════════════════════════════════════════════════════════

def deduplicate_chunks(chunks) -> list:
    """对文本块做内容哈希去重。"""
    seen: dict[str, int] = {}
    result = []
    dup_count = 0

    for chunk in tqdm(chunks, desc="🔍 内容去重", unit="chunk", ncols=80):
        h = _content_hash(chunk.page_content)
        if h in seen:
            idx = seen[h]
            existing = result[idx]
            existing_sources = existing.metadata.get("all_sources", [])
            new_source = chunk.metadata.get("source", "")
            if new_source and new_source not in existing_sources:
                existing_sources.append(new_source)
            existing.metadata["all_sources"] = existing_sources
            dup_count += 1
        else:
            chunk.metadata["all_sources"] = [chunk.metadata.get("source", "")]
            seen[h] = len(result)
            result.append(chunk)

    if dup_count:
        print(f"   去除 {dup_count} 个重复块，保留 {len(result)} 个唯一块")

    return result


# ═══════════════════════════════════════════════════════════════
# 优化 4: BM25 索引
# ═══════════════════════════════════════════════════════════════

def _jieba_tokenize(text: str) -> list[str]:
    """中文分词：优先 jieba，回退到字符级。"""
    try:
        import jieba
        import tempfile
        # 设置 jieba 缓存目录，避免 /tmp 权限问题
        cache_dir = os.path.expanduser("~/.cache/jieba")
        os.makedirs(cache_dir, exist_ok=True)
        jieba.dt.tmp_dir = cache_dir
        jieba.dt.cache_file = os.path.join(cache_dir, "jieba.cache")
        # 确保词典已加载
        jieba.initialize()
        return list(jieba.cut_for_search(text))
    except ImportError:
        return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def build_bm25_index(chunks, db_dir: str):
    """构建 BM25 索引并持久化到磁盘。"""
    check_dependencies(extra=["rank_bm25"])
    from rank_bm25 import BM25Okapi

    corpus = []
    for chunk in chunks:
        tokens = _jieba_tokenize(chunk.page_content)
        corpus.append(tokens)

    print(f"📊 构建 BM25 索引 ({len(corpus)} 个文档)...")
    bm25 = BM25Okapi(corpus)

    index_path = os.path.join(db_dir, BM25_INDEX_FILE)
    index_data = {
        "corpus": corpus,
        "chunks_meta": [
            {"page_content": c.page_content, "metadata": c.metadata}
            for c in chunks
        ],
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False)

    print(f"   BM25 索引已保存: {index_path}")
    return bm25, corpus


def load_bm25_index(db_dir: str):
    """从磁盘加载 BM25 索引。"""
    check_dependencies(extra=["rank_bm25"])
    from rank_bm25 import BM25Okapi
    from langchain_core.documents import Document

    index_path = os.path.join(db_dir, BM25_INDEX_FILE)
    if not os.path.exists(index_path):
        return None, None, None

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    corpus = index_data["corpus"]
    bm25 = BM25Okapi(corpus)
    chunks = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in index_data["chunks_meta"]
    ]
    return bm25, corpus, chunks


def bm25_search(bm25, chunks, query: str, top_k: int) -> list[tuple]:
    """BM25 检索，返回 [(Document, score), ...]。"""
    tokens = _jieba_tokenize(query)
    scores = bm25.get_scores(tokens)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append((chunks[idx], float(scores[idx])))
    return results


# ═══════════════════════════════════════════════════════════════
# 优化 3: Reranker 重排序
# ═══════════════════════════════════════════════════════════════

_reranker_cache = {}


def get_reranker():
    """加载 BGE Reranker 模型（带缓存）。"""
    if "model" in _reranker_cache:
        return _reranker_cache["model"], _reranker_cache["tokenizer"]

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"📦 加载重排序模型: {RERANKER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL, trust_remote_code=True,
    )
    model.eval()
    _reranker_cache["model"] = model
    _reranker_cache["tokenizer"] = tokenizer
    return model, tokenizer


def rerank(query: str, candidates: list[tuple], top_k: int) -> list[tuple]:
    """使用 Reranker 对候选结果重排序。"""
    import torch

    model, tokenizer = get_reranker()

    pairs = [(query, doc.page_content) for doc, _ in candidates]
    if not pairs:
        return []

    batch_size = 32
    all_scores = []

    for i in tqdm(range(0, len(pairs), batch_size),
                  desc="🔄 重排序", unit="batch", ncols=80):
        batch = pairs[i:i + batch_size]
        with torch.no_grad():
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            )
            logits = model(**inputs).logits.squeeze(-1)
            scores = logits.tolist()
            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)

    reranked = list(zip([doc for doc, _ in candidates], all_scores))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]


# ═══════════════════════════════════════════════════════════════
# 优化 4: RRF 倒数排名融合
# ═══════════════════════════════════════════════════════════════

def rrf_fusion(
    vec_results: list[tuple],
    bm25_results: list[tuple],
    top_k: int,
    k: int = 60,
    vec_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[tuple]:
    """Reciprocal Rank Fusion 合并向量和 BM25 结果。"""
    fused: dict[str, tuple] = {}

    for rank, (doc, score) in enumerate(vec_results):
        key = _content_hash(doc.page_content)
        rrf_score = vec_weight / (k + rank + 1)
        if key in fused:
            old_doc, old_score = fused[key]
            fused[key] = (old_doc, old_score + rrf_score)
        else:
            fused[key] = (doc, rrf_score)

    for rank, (doc, score) in enumerate(bm25_results):
        key = _content_hash(doc.page_content)
        rrf_score = bm25_weight / (k + rank + 1)
        if key in fused:
            old_doc, old_score = fused[key]
            fused[key] = (old_doc, old_score + rrf_score)
        else:
            fused[key] = (doc, rrf_score)

    results = sorted(fused.values(), key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ═══════════════════════════════════════════════════════════════
# 查询时去重
# ═══════════════════════════════════════════════════════════════

def deduplicate_results(results: list[tuple], top_k: int) -> list[tuple]:
    """对检索结果按内容哈希去重，保留相似度最高的一条。"""
    seen = set()
    deduped = []
    for doc, score in results:
        h = _content_hash(doc.page_content)
        if h not in seen:
            seen.add(h)
            deduped.append((doc, score))
            if len(deduped) >= top_k:
                break
    return deduped


# ─────────────────── 嵌入模型 ───────────────────

def get_embeddings():
    """加载 BGE 中文嵌入模型。"""
    from langchain_huggingface import HuggingFaceEmbeddings

    print(f"📦 加载嵌入模型: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


# ═══════════════════════════════════════════════════════════════
# RAG 引擎：启动时预加载所有资源，请求时复用
# ═══════════════════════════════════════════════════════════════

class RAGEngine:
    """RAG 检索引擎 —— 启动时预加载模型和索引，请求时零延迟复用。"""

    def __init__(self, db_dir: str):
        self.db_dir = os.path.abspath(db_dir)
        if not os.path.isdir(self.db_dir):
            raise FileNotFoundError(f"向量库不存在: {self.db_dir}")

        self.embeddings = None
        self.vectorstore = None
        self.bm25 = None
        self.bm25_chunks = None
        self.reranker_model = None
        self.reranker_tokenizer = None

    def load(self, load_bm25: bool = True, load_reranker: bool = True):
        """预加载所有资源。"""
        from langchain_chroma import Chroma

        print(f"\n🚀 预加载 RAG 引擎资源...")
        t0 = time.time()

        # 1. 嵌入模型
        self.embeddings = get_embeddings()

        # 2. 向量库
        print(f"📂 加载向量库: {self.db_dir}")
        self.vectorstore = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings,
        )

        # 3. BM25 索引
        if load_bm25:
            self.bm25, _, self.bm25_chunks = load_bm25_index(self.db_dir)
            if self.bm25:
                print(f"📊 BM25 索引已加载 ({len(self.bm25_chunks)} 个文档)")
            else:
                print("⚠ BM25 索引不存在")

        # 4. Reranker 模型
        if load_reranker:
            self.reranker_model, self.reranker_tokenizer = get_reranker()

        elapsed = time.time() - t0
        print(f"✅ 资源预加载完成，耗时 {elapsed:.1f}s\n")

    def query(
        self,
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
        recall_k: int = DEFAULT_RECALL_K,
        use_reranker: bool = True,
        use_bm25: bool = True,
    ) -> dict:
        """执行查询 —— 使用预加载的资源，无需重复加载。"""
        import torch

        t0 = time.time()

        # 向量检索
        vec_results = self.vectorstore.similarity_search_with_score(query_text, k=recall_k)
        vec_results = [(doc, max(0.0, 1.0 - score / 2.0)) for doc, score in vec_results]

        # BM25 检索
        bm25_results = []
        if use_bm25 and self.bm25:
            bm25_results = bm25_search(self.bm25, self.bm25_chunks, query_text, recall_k)

        # 融合
        if use_bm25 and bm25_results:
            candidates = rrf_fusion(vec_results, bm25_results, top_k=recall_k)
            mode = "混合检索(BM25+向量)+RRF"
        else:
            candidates = vec_results
            mode = "向量检索"

        # Reranker 精排（使用预加载的模型，无 tqdm）
        if use_reranker and candidates and self.reranker_model:
            pairs = [(query_text, doc.page_content) for doc, _ in candidates]
            batch_size = 64  # 更大的 batch size 提升吞吐量
            all_scores = []

            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                with torch.no_grad():
                    inputs = self.reranker_tokenizer(
                        batch, padding=True, truncation=True,
                        max_length=512, return_tensors="pt",
                    )
                    logits = self.reranker_model(**inputs).logits.squeeze(-1)
                    scores = logits.tolist()
                    if isinstance(scores, float):
                        scores = [scores]
                    all_scores.extend(scores)

            reranked = list(zip([doc for doc, _ in candidates], all_scores))
            reranked.sort(key=lambda x: x[1], reverse=True)
            final_results = reranked[:top_k]
            mode += " → Reranker精排"
        else:
            final_results = candidates[:top_k]

        # 结果去重
        final_results = deduplicate_results(final_results, top_k)

        elapsed = time.time() - t0

        # 结构化输出
        results = []
        for i, (doc, score) in enumerate(final_results, 1):
            results.append({
                "rank": i,
                "score": round(score, 4),
                "content": doc.page_content.strip(),
                "source": doc.metadata.get("filename", "未知"),
                "path": doc.metadata.get("source", ""),
                "all_sources": doc.metadata.get("all_sources", []),
            })

        return {
            "query": query_text,
            "mode": mode,
            "elapsed": round(elapsed, 2),
            "total": len(results),
            "results": results,
        }


# ═══════════════════════════════════════════════════════════════
# CLI: 导入子命令
# ═══════════════════════════════════════════════════════════════

def cmd_import(args):
    """导入子命令：加载 → 切分 → 去重 → 嵌入 → 存入 Chroma + BM25。"""
    check_dependencies()
    from langchain_chroma import Chroma

    root = os.path.abspath(args.directory)
    if not os.path.isdir(root):
        sys.exit(f"❌ 目录不存在: {root}")

    db_dir = os.path.abspath(args.db)

    print(f"🔍 扫描目录: {root}")
    files = discover_files(root)
    if not files:
        sys.exit(f"❌ 未找到任何支持的文件 (支持: {', '.join(sorted(SUPPORTED_EXTS))})")
    print(f"   找到 {len(files)} 个文件")

    docs = load_documents(files)
    if not docs:
        sys.exit("❌ 未能加载任何文档内容")
    print(f"   加载了 {len(docs)} 个文档片段")

    print(f"✂️  切分文本 (chunk_size={args.chunk_size}, overlap={args.chunk_overlap})...")
    chunks = split_documents_markdown(docs, args.chunk_size, args.chunk_overlap)
    print(f"   切分为 {len(chunks)} 个块")

    chunks = deduplicate_chunks(chunks)

    embeddings = get_embeddings()
    print(f"💾 写入向量库: {db_dir}")
    t0 = time.time()

    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    vectorstore = None
    for i in tqdm(range(0, len(chunks), batch_size),
                  desc="🔢 嵌入向量化", unit="batch", total=total_batches, ncols=80):
        batch = chunks[i:i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=db_dir,
            )
        else:
            vectorstore.add_documents(batch)

    build_bm25_index(chunks, db_dir)

    elapsed = time.time() - t0
    print(f"\n✅ 完成! 共索引 {len(chunks)} 个文本块，耗时 {elapsed:.1f}s")
    print(f"   向量库路径: {db_dir}")


# ═══════════════════════════════════════════════════════════════
# CLI: 查询子命令
# ═══════════════════════════════════════════════════════════════

def cmd_query(args):
    """查询子命令：创建引擎 → 加载资源 → 查询 → 格式化输出。"""
    check_dependencies()

    query_text = args.query.strip()
    if not query_text:
        sys.exit("❌ 查询内容不能为空")

    db_dir = os.path.abspath(args.db)
    use_reranker = not args.no_reranker
    use_bm25 = not args.no_bm25

    print(f"🔎 查询: {query_text}")
    print(f"   返回 Top-{args.top_k}\n")

    try:
        engine = RAGEngine(db_dir)
        engine.load(load_bm25=use_bm25, load_reranker=use_reranker)
        data = engine.query(
            query_text=query_text,
            top_k=args.top_k,
            recall_k=args.recall_k,
            use_reranker=use_reranker,
            use_bm25=use_bm25,
        )
    except FileNotFoundError as e:
        sys.exit(f"❌ {e}\n   请先运行 import 子命令导入数据")

    print(f"\n{'='*70}")
    print(f"检索完成 [{data['mode']}]，共 {data['total']} 条结果，耗时 {data['elapsed']:.2f}s")
    print(f"{'='*70}\n")

    for item in data["results"]:
        print(f"─── [{item['rank']}] 得分: {item['score']:.4f} ───")
        print(f"    来源: {item['source']}")
        if item["path"]:
            print(f"    路径: {item['path']}")
        if len(item["all_sources"]) > 1:
            others = [os.path.basename(s) for s in item["all_sources"][1:]]
            print(f"    其他来源: {', '.join(others)}")
        print(f"    内容:")
        for line in item["content"].split("\n"):
            print(f"      {line}")
        print()


# ═══════════════════════════════════════════════════════════════
# CLI: query-server 子命令
# ═══════════════════════════════════════════════════════════════

def cmd_query_server(args):
    """启动 HTTP API 服务 —— 启动时预加载所有资源，请求时直接复用。"""
    check_dependencies(extra=["fastapi", "uvicorn", "sse_starlette"])

    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    from sse_starlette.sse import EventSourceResponse
    import uvicorn

    db_dir = os.path.abspath(args.db)
    if not os.path.isdir(db_dir):
        sys.exit(f"❌ 向量库不存在: {db_dir}\n   请先运行 import 子命令导入数据")

    # ═══════════════════════════════════════════════════════════
    # 启动时预加载所有资源（只做一次）
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("🚀 RAG Query API Server 启动中...")
    print(f"{'='*70}")

    engine = RAGEngine(db_dir)
    engine.load(load_bm25=True, load_reranker=True)

    # ═══════════════════════════════════════════════════════════
    # FastAPI 应用
    # ═══════════════════════════════════════════════════════════
    app = FastAPI(
        title="RAG Query API",
        description="向量检索 API 服务，支持混合检索 + Reranker 精排",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── 请求模型 ──
    class QueryRequest(BaseModel):
        query: str = Field(..., description="查询内容", min_length=1)
        top_k: int = Field(DEFAULT_TOP_K, ge=1, le=100, description="返回结果数量")
        recall_k: int = Field(DEFAULT_RECALL_K, ge=1, le=200, description="召回数量")
        use_reranker: bool = Field(True, description="是否使用 Reranker 精排")
        use_bm25: bool = Field(True, description="是否使用 BM25 检索")
        stream: bool = Field(False, description="是否使用 SSE 流式输出")

    class QueryResult(BaseModel):
        rank: int
        score: float
        content: str
        source: str
        path: str
        all_sources: list[str]

    class QueryResponse(BaseModel):
        query: str
        mode: str
        elapsed: float
        total: int
        results: list[QueryResult]

    # ── 健康检查 ──
    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "db_dir": db_dir,
            "bm25_loaded": engine.bm25 is not None,
            "reranker_loaded": engine.reranker_model is not None,
        }

    # ── 查询接口 ──
    @app.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest):
        """执行向量检索，返回 Top-K 结果。

        - 默认返回 JSON 格式结果
        - 设置 stream=true 返回 SSE 流式响应
        """
        try:
            if req.stream:
                # SSE 流式输出
                async def event_generator():
                    loop = asyncio.get_running_loop()
                    data = await loop.run_in_executor(
                        None,
                        lambda: engine.query(
                            query_text=req.query,
                            top_k=req.top_k,
                            recall_k=req.recall_k,
                            use_reranker=req.use_reranker,
                            use_bm25=req.use_bm25,
                        )
                    )

                    yield {
                        "event": "meta",
                        "data": json.dumps({
                            "query": data["query"],
                            "mode": data["mode"],
                            "elapsed": data["elapsed"],
                            "total": data["total"],
                        }, ensure_ascii=False)
                    }

                    for item in data["results"]:
                        yield {
                            "event": "result",
                            "data": json.dumps(item, ensure_ascii=False)
                        }

                    yield {
                        "event": "done",
                        "data": json.dumps({"status": "completed"})
                    }

                return EventSourceResponse(event_generator())
            else:
                # 普通 JSON 响应 —— 直接使用预加载的引擎，无需重新加载
                data = engine.query(
                    query_text=req.query,
                    top_k=req.top_k,
                    recall_k=req.recall_k,
                    use_reranker=req.use_reranker,
                    use_bm25=req.use_bm25,
                )
                return data

        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

    # ── 启动服务 ──
    host = args.host
    port = args.port
    print(f"\n🌐 服务地址: http://{host}:{port}")
    print(f"   API 文档: http://{host}:{port}/docs")
    print(f"   健康检查: GET http://{host}:{port}/health")
    print(f"\n示例请求:")
    print(f"  curl -X POST http://localhost:{port}/query \\")
    print(f'    -H "Content-Type: application/json" \\')
    print(f"    -d '{{\"query\": \"查询内容\", \"top_k\": 10}}'")
    print(f"\n流式请求:")
    print(f"  curl -N -X POST http://localhost:{port}/query \\")
    print(f'    -H "Content-Type: application/json" \\')
    print(f"    -d '{{\"query\": \"查询内容\", \"stream\": true}}'")
    print(f"\n{'='*70}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


# ─────────────────── 主入口 ───────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAG 向量检索 CLI —— 导入文档 & 语义查询（优化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ── import 子命令 ──
    p_import = subparsers.add_parser(
        "import",
        help="将目录下文件加载、切分、去重、嵌入后存入 Chroma + BM25",
    )
    p_import.add_argument("directory", help="要导入的目录路径")
    p_import.add_argument(
        "--db", default=DEFAULT_DB_DIR,
        help=f"向量库持久化路径 (默认: {DEFAULT_DB_DIR})",
    )
    p_import.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"文本块大小 (默认: {DEFAULT_CHUNK_SIZE})",
    )
    p_import.add_argument(
        "--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
        help=f"文本块重叠大小 (默认: {DEFAULT_CHUNK_OVERLAP})",
    )
    p_import.add_argument(
        "--no-rerank", action="store_true",
        help="导入时标记不使用 reranker（查询时仍可用）",
    )

    # ── query 子命令 ──
    p_query = subparsers.add_parser(
        "query",
        help="混合检索 + Reranker 精排，返回 Top-K 结果",
    )
    p_query.add_argument("query", help="查询内容")
    p_query.add_argument(
        "--db", default=DEFAULT_DB_DIR,
        help=f"向量库路径 (默认: {DEFAULT_DB_DIR})",
    )
    p_query.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"返回结果数量 (默认: {DEFAULT_TOP_K})",
    )
    p_query.add_argument(
        "--recall-k", type=int, default=DEFAULT_RECALL_K,
        help=f"召回数量，rerank 前的候选集大小 (默认: {DEFAULT_RECALL_K})",
    )
    p_query.add_argument(
        "--no-reranker", action="store_true",
        help="禁用 Reranker 重排序",
    )
    p_query.add_argument(
        "--no-bm25", action="store_true",
        help="禁用 BM25 检索，仅使用向量检索",
    )

    # ── query-server 子命令 ──
    p_server = subparsers.add_parser(
        "query-server",
        help="启动 HTTP API 服务，提供 /query 接口（支持 SSE 流式输出）",
    )
    p_server.add_argument(
        "--db", default=DEFAULT_DB_DIR,
        help=f"向量库路径 (默认: {DEFAULT_DB_DIR})",
    )
    p_server.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"监听地址 (默认: {DEFAULT_HOST})",
    )
    p_server.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"监听端口 (默认: {DEFAULT_PORT})",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "import":
        cmd_import(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "query-server":
        cmd_query_server(args)


if __name__ == "__main__":
    main()