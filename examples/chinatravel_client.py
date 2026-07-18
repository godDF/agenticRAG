"""Minimal async client that ChinaTravel can call without importing this service."""

from __future__ import annotations

import httpx


async def query_agentic_rag(
    *,
    base_url: str,
    service_api_key: str,
    session_id: str,
    query: str,
    category: str,
) -> dict:
    headers = {"Authorization": f"Bearer {service_api_key}"}
    payload = {
        "session_id": session_id,
        "query": query,
        "category": category,
    }
    async with httpx.AsyncClient(timeout=25, trust_env=False) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/api/v1/query",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()

