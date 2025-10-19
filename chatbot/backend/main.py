# main.py
import asyncio
import json
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 导入你的 agent 模块
from agent import agent

app = FastAPI(title="LangGraph Chat Agent API", version="1.0")


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"  # 用于区分不同会话


async def event_stream(user_input: str, thread_id: str) -> AsyncGenerator[str, None]:
    """
    异步生成器：模拟 agent.stream() 的输出并逐块发送 SSE
    """
    from langchain_core.messages import HumanMessage

    messages = [HumanMessage(content=user_input)]
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # 遍历 agent 的 stream 输出
        async for event in agent.astream_events({"messages": messages}, config=config, version="v2"):
            # 只处理 'on_chain_end' 或 'on_chat_model_stream' 类型事件
            if event["event"] in ["on_chat_model_stream"]:
                chunk = event["data"]["chunk"]
                content = getattr(chunk, "content", "") if hasattr(chunk, "content") else str(chunk)
                if content:
                    yield f"data: {json.dumps({'type': 'chunk', 'content': content}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0)  # 让出控制权，避免阻塞

        # 发送结束标记
        yield f"data: {json.dumps({'type': 'end', 'content': '[DONE]'}, ensure_ascii=False)}\n\n"

    except Exception as e:
        error_msg = f"Error during streaming: {str(e)}"
        yield f"data: {json.dumps({'type': 'error', 'content': error_msg}, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    接收用户消息，启动 agent 并流式返回结果
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    return StreamingResponse(
        event_stream(request.message, request.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        }
    )


@app.get("/")
def root():
    return {"message": "LangGraph Chat Agent is running. POST to /api/chat with {\"message\": \"...\"}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
