# ChinaTravel Agentic RAG 离线评测

本目录用于比较 Traditional RAG、Agentic RAG 和当前生产知识库。正式
`/api/v1/query` 接口不会暴露评测排名或 Chunk 内容。

## 1. 构建语料快照和候选评测集

在 `agenticRAG` 环境中执行：

```powershell
conda activate agenticRAG
cd D:\agent_project\chinaTravel_agenticRAG
python -m evaluation.build_dataset --version v1
```

该命令会：

1. 读取 `D:\agent_project\ChinaTravel-main\kb` 中的 Markdown。
2. 生成稳定 `chunk_uid` 和 `evaluation/manifests/kb_v1.json`。
3. 重建独立 Collection `chinatravel_safety_eval_v1`，不修改生产 Collection。
4. 调用现有 DeepSeek 生成120条候选问题。
5. 输出 JSONL 和便于人工审核的 CSV。

只检查本地文档、不调用 Embedding 或 LLM：

```powershell
python -m evaluation.build_dataset --version v1 --no-index --no-generate
```

## 2. 人工审核

编辑：

```text
evaluation/datasets/review_sheet_v1.csv
```

确认问题、参考答案、事实、禁止事实和相关 Chunk 后，将 `review_status` 改成
`approved`，填写 `reviewer`。然后合并并校验：

```powershell
python -m evaluation.apply_reviews --version v1
python -m evaluation.validate_dataset --version v1
```

未全部通过人工审核时，正式运行器会拒绝执行，避免用未经验证的合成答案作为金标。

## 3. 运行确定性评测

先运行30条 Smoke 集：

```powershell
python -m evaluation.run --suite smoke --systems baseline,agentic
```

正式Test集和生产库健康度：

```powershell
python -m evaluation.run --suite full --systems baseline,agentic,production --resume
```

运行包含Development与Test在内的全部120题：

```powershell
python -m evaluation.run --suite all --systems baseline,agentic
```

稳定性子集会重复3次：

```powershell
python -m evaluation.run --suite stability --systems baseline,agentic
```

如果任务中断，使用原 Run ID 续跑：

```powershell
python -m evaluation.run --suite full --systems baseline,agentic,production `
  --run-id <run_id> --resume
```

运行器默认并发2、单次超时45秒，瞬时错误最多重试2次。每完成一项立即写入
JSONL，因此异常退出不会丢失已完成结果。

## 4. RAGAS评分（独立环境）

不要把 `ragas==0.4.3` 安装到现有 `agenticRAG` 环境。其依赖会升级当前固定的
LangChain、NumPy 和 LangSmith。经用户审核后，创建隔离环境：

```powershell
conda env create -f environment-eval.yml
conda activate agenticRAG-eval
cd D:\agent_project\chinaTravel_agenticRAG
$env:RAGAS_DO_NOT_TRACK="true"
python -m evaluation.ragas_score --run-id <run_id>
```

RAGAS会补充：

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall
- Answer Correctness（Factual Correctness F1）

RAGAS逐项保存检查点，重复运行会跳过已完成项。评分结束后会重新生成对比报告。
评测脚本仅对离线 DeepSeek Judge 关闭 thinking 模式，以保证结构化评分稳定；
该设置不会改变生产 Agentic RAG 的规划、检索、证据评估或回答流程。Windows 下
脚本会从隔离 Conda 环境预加载 Arrow DLL，避免影响主运行环境。

## 5. 输出

每个运行目录包含：

```text
evaluation/runs/<run_id>/
├── config.json
├── raw_results.jsonl
├── scores.jsonl
├── ragas_scores.jsonl
├── summary.json
├── summary.csv
├── comparison_overall.csv
├── comparison_by_category.csv
├── comparison_by_question_type.csv
├── comparison_agentic_rounds.csv
├── comparison.md
├── failed_cases.jsonl
└── report.md
```

对比表中的相对改善会区分指标方向：Recall、MRR和Faithfulness越高越好；
耗时、成本和错误率越低越好。Baseline为0时显示 `N/A`，并通过成对Bootstrap
给出95%置信区间。
