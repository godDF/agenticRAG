# ChinaTravel Agentic RAG

这是一个与 `ChinaTravel-main` 解耦的 Agentic RAG 微服务，默认监听 `127.0.0.1:8100`。旅行规划、DFS、安全护栏和人工审核仍由原项目负责；本服务只处理六类旅行规则知识的上传、检索、生成、校验与 Trace。

## 运行环境

- Conda 环境：`agenticRAG`
- Python：3.12.13
- 不下载本地 Embedding 或大模型权重
- Embedding：调用 OpenAI-compatible BGE-M3 API
- 向量数据库：Qdrant
- 可观测性：本地 Trace 始终启用，RagaAI Catalyst 可选启用

选择 Python 3.12 是为了适配当前 RagaAI Catalyst，同时通过版本锁定避开新版 LangChain/LiteLLM 的不兼容组合。

## Agentic RAG 流程

```text
用户问题（已由 ChinaTravel 完成安全分类）
  -> Query Planner（最多 2 个子问题）
  -> BGE-M3 Embedding + Qdrant 检索
  -> RRF 融合
  -> Evidence Grader
      -> 证据不足：Query Rewriter 再检索 1 次
      -> 证据充分：Grounded Answer Generator
  -> Answer Verifier
      -> pass：返回答案和来源
      -> revise：最多修正 1 次
      -> insufficient：安全降级为“未查询到可靠信息”
```

循环次数、子查询数和回答修订次数都有硬上限，不会无限调用模型。上传文档中的文本一律视为不可信数据，Agent Prompt 明确禁止执行文档内指令。

## 配置

首次创建环境：

```powershell
conda env create -f environment.yml
conda activate agenticRAG
```

本机环境已经创建时，只需激活。若要重新同步依赖：

```powershell
python -m pip install -r requirements.txt -c constraints.txt
```

```powershell
conda activate agenticRAG
Copy-Item .env.example .env
```

编辑 `.env`，至少填写：

- `LLM_API_KEY`：OpenAI-compatible 对话模型密钥。
- `BGE_M3_API_URL`、`BGE_M3_API_KEY`：BGE-M3 Embedding 服务。
- `QDRANT_URL`：Qdrant 地址。

RagaAI Catalyst 是可选功能。取得 Catalyst 的 Access Key 和 Secret Key 后，将 `RAGAAI_ENABLED` 改为 `true` 并填写对应配置。未配置或上报失败时，本地 RAG 查询仍可继续，Trace 可通过本服务接口查看。

## 启动与检查

```powershell
conda activate agenticRAG
python run_server.py
```

打开：

- Swagger：<http://127.0.0.1:8100/docs>
- 知识库工作台：<http://127.0.0.1:8100/ui>
- 健康检查：<http://127.0.0.1:8100/health>
- Qdrant 就绪检查：<http://127.0.0.1:8100/ready>

## API

项目默认采用本地演示模式，接口不需要内部 Token。如果未来部署到公网，在 `.env` 中设置：

```text
REQUIRE_SERVICE_API_KEY=true
SERVICE_API_KEY=<strong-random-key>
Authorization: Bearer <SERVICE_API_KEY>
```

### 上传文件

`POST /api/v1/documents`，使用 `multipart/form-data`：

- `file`：`.md`、`.txt`、`.pdf`、`.docx`，默认最大 10 MB。
- `title`：可选。留空时依次尝试 YAML `title`、DOCX/PDF 元数据、Markdown 标题、正文首行和文件名。
- `category`：六类之一。
- `source_name`：可选，留空时记录为“用户上传文档”。
- `source_url`：可选。
- `updated_at`：必填，例如 `2026-07-16`。

六类 category：

```text
child_ticket
elderly_ticket
student_ticket
flight_safety
highspeed_rail_safety
attraction_notice
```

上传后，系统提取正文，转为带 YAML 元数据的 Markdown，按约 400 字、80 字重叠切分，调用 BGE-M3 API，并写入 Qdrant。原始上传文件不会保留。

### 可视化工作台与异步进度

访问 `/ui` 即可使用。页面支持：

- 浏览器到服务端的文件上传百分比。
- 文档解析、Markdown 规范化、分片、BGE-M3 Embedding、Qdrant 写入和 SQLite 登记进度。
- Agent、LLM 和 Tool 类型过滤，以及每次调用的状态、参数摘要、耗时和错误。
- 上传完成后的 Token、端到端耗时、估算成本、调用次数和错误次数。
- Agentic RAG 查询调试、来源展示和完整执行 Trace。
- 已索引文档查看与删除。

异步上传接口：

- `POST /api/v1/document-jobs`：上传文件并立即取得 `job_id`。
- `GET /api/v1/document-jobs/{job_id}`：轮询处理进度、事件和指标。

知识入库是确定性工作流，不需要 LLM，因此上传 Trace 中 LLM 调用和 LLM Token 正常为 0；Embedding Token 使用供应商响应中的 `usage` 统计。查询调试区才会产生多 Agent 与 LLM Token。

Agentic 查询成本按 DeepSeek 官方 `deepseek-v4-flash` 人民币价格估算：

- 输入缓存命中：¥0.02 / 百万 Token。
- 输入缓存未命中：¥1 / 百万 Token。
- 输出：¥2 / 百万 Token。

系统直接读取 API `usage.prompt_cache_hit_tokens` 与 `prompt_cache_miss_tokens`。兼容接口没有返回缓存细分时，为避免低估，会将全部输入按缓存未命中计算。价格配置为：

```text
LLM_MODEL=deepseek-v4-flash
DEEPSEEK_CACHE_HIT_COST_PER_1M_TOKENS_CNY=0.02
DEEPSEEK_CACHE_MISS_COST_PER_1M_TOKENS_CNY=1
DEEPSEEK_OUTPUT_COST_PER_1M_TOKENS_CNY=2
```

计费参考：<https://api-docs.deepseek.com/zh-cn/quick_start/pricing/>。BGE-M3 来自独立 Embedding 服务商，不套用 DeepSeek 价格。当前配置使用 SiliconFlow `BAAI/bge-m3`，其价格页标记为免费，因此上传成本会明确显示“¥0（官方免费）”。若以后切换到 `Pro/BAAI/bge-m3` 或其他收费模型，再修改 `EMBEDDING_COST_PER_1M_TOKENS_CNY`。

### 查询

`POST /api/v1/query`

```json
{
  "session_id": "chinatravel-session-id",
  "request_id": "optional-idempotent-trace-id",
  "query": "学生票一年可以买几次？",
  "category": "student_ticket"
}
```

响应包含 `answer`、`sources`、`trace_id`、耗时、Token 使用量和各 Agent 阶段 Trace。查询类别由 ChinaTravel 的入口分类器决定，本服务内的 Agent 不能改类别。

### 文档与 Trace 管理

- `GET /api/v1/documents`
- `DELETE /api/v1/documents/{document_id}`
- `POST /api/v1/documents/{document_id}/reindex`
- `GET /api/v1/traces/{trace_id}`

## 测试

```powershell
conda activate agenticRAG
python -m pytest -q
```

单元测试不调用外部大模型、Embedding API 或 Qdrant。
