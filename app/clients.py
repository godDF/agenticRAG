from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from json_repair import loads as repair_loads
from openai import AsyncOpenAI

from app.config import Settings
from app.observability import trace_llm, trace_tool


@dataclass
class LlmResult:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_input_tokens: int = 0
    cache_miss_input_tokens: int = 0
    cache_usage_reported: bool = False


def parse_llm_usage(usage: Any) -> dict[str, int | bool]:
    usage_data = usage.model_dump() if hasattr(usage, "model_dump") else {}
    model_extra = getattr(usage, "model_extra", None) or {}
    usage_data.update(model_extra)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    cache_hit = int(usage_data.get("prompt_cache_hit_tokens", 0) or 0)
    cache_miss_value = usage_data.get("prompt_cache_miss_tokens")
    cache_usage_reported = (
        "prompt_cache_hit_tokens" in usage_data
        or "prompt_cache_miss_tokens" in usage_data
    )
    if not cache_usage_reported:
        prompt_details = usage_data.get("prompt_tokens_details") or {}
        cache_hit = int(prompt_details.get("cached_tokens", 0) or 0)
        cache_usage_reported = "cached_tokens" in prompt_details
    cache_hit = min(max(0, cache_hit), input_tokens)
    cache_miss = (
        int(cache_miss_value or 0)
        if cache_miss_value is not None
        else input_tokens - cache_hit
    )
    if cache_hit + cache_miss != input_tokens:
        cache_miss = input_tokens - cache_hit
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_hit_input_tokens": cache_hit,
        "cache_miss_input_tokens": max(0, cache_miss),
        "cache_usage_reported": cache_usage_reported,
    }


class LlmClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if not self.settings.llm_api_key:
            raise RuntimeError("未配置 LLM_API_KEY")
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_api_url,
                timeout=self.settings.llm_timeout_seconds,
            )
        return self._client

    @trace_llm(name="chinatravel_rag_llm", model="openai-compatible")
    async def complete(self, prompt: str, *, json_mode: bool = False) -> LlmResult:
        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = await self._get_client().chat.completions.create(**kwargs)
        content = (response.choices[0].message.content or "").strip()
        usage = response.usage
        token_usage = parse_llm_usage(usage)
        return LlmResult(
            content=content,
            **token_usage,
        )

    async def complete_json(self, prompt: str) -> tuple[dict[str, Any], LlmResult]:
        result = await self.complete(prompt, json_mode=True)
        try:
            data = repair_loads(result.content)
        except Exception as exc:
            raise RuntimeError("LLM 未返回有效 JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError("LLM JSON 结果不是对象")
        return data, result


class EmbeddingClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _endpoint(self) -> str:
        if not self.settings.embedding_api_url or not self.settings.embedding_api_key:
            raise RuntimeError("未配置 BGE_M3_API_URL 或 BGE_M3_API_KEY")
        url = self.settings.embedding_api_url
        return url if url.endswith("/embeddings") else f"{url}/v1/embeddings"

    @trace_tool(name="bge_m3_embedding", tool_type="embedding_api")
    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors, _usage = await self.embed_with_usage(texts)
        return vectors

    async def embed_with_usage(
        self, texts: list[str]
    ) -> tuple[list[list[float]], dict[str, int]]:
        if not texts:
            return [], {"prompt_tokens": 0, "total_tokens": 0}
        headers = {
            "Authorization": f"Bearer {self.settings.embedding_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.settings.embedding_model, "input": texts}
        async with httpx.AsyncClient(
            timeout=self.settings.embedding_timeout_seconds,
            trust_env=False,
        ) as client:
            response = await client.post(self._endpoint(), headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
        rows = sorted(body.get("data", []), key=lambda row: row.get("index", 0))
        vectors = [row.get("embedding") for row in rows]
        if len(vectors) != len(texts) or any(not isinstance(vector, list) for vector in vectors):
            raise RuntimeError("BGE-M3 API 返回的向量数量不匹配")
        if vectors and len(vectors[0]) != self.settings.embedding_vector_size:
            raise RuntimeError(
                f"BGE-M3 向量维度为 {len(vectors[0])}，配置期望 "
                f"{self.settings.embedding_vector_size}"
            )
        raw_usage = body.get("usage") or {}
        prompt_tokens = int(raw_usage.get("prompt_tokens", 0) or 0)
        total_tokens = int(raw_usage.get("total_tokens", prompt_tokens) or prompt_tokens)
        return vectors, {
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        }
