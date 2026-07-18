# 安装与下载记录

## Conda 环境

- 环境名：`agenticRAG`
- 路径：`C:\Users\ACER\.conda\envs\agenticRAG`
- Python：3.12.13

## 直接依赖

- FastAPI 0.139.0、Uvicorn 0.51.0、HTTPX 0.28.1
- Pydantic 2.13.4、PyYAML 6.0.3、python-dotenv 1.2.2
- OpenAI Python SDK 2.45.0、json-repair 0.61.5
- Qdrant Client 1.18.0
- python-multipart 0.0.32、pypdf 6.14.2、python-docx 1.2.0
- RagaAI Catalyst 2.1.7.4
- pytest 8.4.2

## 兼容性锁定

RagaAI Catalyst 2.1.7.4 的部分依赖声明较宽。为保证 Python 3.12 下可导入和运行，锁定：

- LiteLLM 1.61.15
- LangChain 0.2.17、LangChain Core 0.2.43、LangSmith 0.1.147
- NumPy 1.26.4、Packaging 24.2、pytz 2026.2
- setuptools 80.9.0

初次依赖预演时，最新版 LiteLLM 的源码构建探测将 `litellm` 源码包（约 15.1 MB）和 `rustup-init`（约 12.8 MB）放入下载缓存，但未安装 Rust/Cargo。随后改用兼容的 LiteLLM wheel 完成安装。

## 明确未下载

- 未下载 BGE-M3 模型权重。
- 未下载任何对话大模型权重。
- 未安装 Rust、Cargo 或本地推理框架。

BGE-M3 与对话模型均通过 API 调用。

## 知识库工作台

前端使用原生 HTML、CSS 和 JavaScript 实现，没有安装 Node、Vue 或其他前端依赖。上传进度和 Trace 复用现有 FastAPI、HTTPX、Qdrant Client 与 python-multipart。
