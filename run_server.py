from __future__ import annotations

import uvicorn

from app.config import settings


if __name__ == "__main__":
    print("启动 ChinaTravel Agentic RAG 服务...")
    print(f"API 文档: http://{settings.host}:{settings.port}/docs")
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
