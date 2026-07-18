from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

from app.config import Settings

logger = logging.getLogger(__name__)

# Avoid a network request during every process import. Catalyst/LiteLLM can use
# its packaged price map; tracing remains optional and must not delay startup.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def _noop_decorator(*_args, **_kwargs):
    def decorate(func):
        return func
    return decorate


RagaAICatalyst = None
Tracer = None
init_tracing = None
trace_agent = trace_llm = trace_tool = _noop_decorator

# Loading Catalyst and its instrumentation tree is relatively expensive. Keep
# default startup light; when enabled, real decorators are loaded before the
# decorated application modules are imported.
if os.getenv("RAGAAI_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}:
    try:
        from ragaai_catalyst import (
            RagaAICatalyst,
            Tracer,
            init_tracing,
            trace_agent,
            trace_llm,
            trace_tool,
        )
    except Exception as exc:  # tracing must never prevent the RAG service from starting
        logger.warning("RagaAI Catalyst import failed; local tracing remains active: %s", exc)
        RagaAICatalyst = None
        Tracer = None
        init_tracing = None
        trace_agent = trace_llm = trace_tool = _noop_decorator


class CatalystTracing:
    def __init__(self) -> None:
        self.tracer = None
        self.enabled = False
        self.error: str | None = None

    def initialize(self, settings: Settings) -> None:
        if not settings.raga_enabled:
            return
        if not settings.raga_access_key or not settings.raga_secret_key:
            self.error = "RAGAAI_ENABLED=true 但未配置 Access Key/Secret Key"
            logger.warning(self.error)
            return
        if RagaAICatalyst is None or Tracer is None or init_tracing is None:
            self.error = "RagaAI Catalyst SDK 不可用"
            return
        try:
            catalyst = RagaAICatalyst(
                access_key=settings.raga_access_key,
                secret_key=settings.raga_secret_key,
                base_url=settings.raga_base_url,
            )
            self.tracer = Tracer(
                project_name=settings.raga_project_name,
                dataset_name=settings.raga_dataset_name,
                tracer_type="agentic",
            )
            init_tracing(catalyst=catalyst, tracer=self.tracer)
            self.enabled = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("RagaAI tracing initialization failed; continuing locally")

    @contextmanager
    def request_trace(self) -> Iterator[None]:
        if not self.enabled or self.tracer is None:
            yield
            return
        try:
            with self.tracer:
                yield
        except Exception:
            logger.exception("RagaAI trace export failed; request result is preserved")


catalyst_tracing = CatalystTracing()
