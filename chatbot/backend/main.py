# main.py
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse

app = FastAPI()

# 允许前端跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],  # Next.js 默认端口
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: list = []


@app.post("/chat")
async def chat(request: Request):
    from agent import agent

    body = await request.json()
    user_message = body["message"]
    history = body.get("history", [])

    async def event_stream():
        try:
            # 构造输入状态
            input_state = {"messages": history + [{"role": "user", "content": user_message}]}

            # 使用 astream_events 获取细粒度事件（LangGraph v2）
            async for event in agent.astream_events(input_state, version="v2"):
                kind = event["event"]

                # 1. 当 LLM 正在生成文本时（这才是真正的 token 流！）
                if kind == "on_chat_model_stream":
                    content = event["data"].get("chunk", {}).content
                    if content:
                        yield f"data: {json.dumps({'type': 'token', 'text': content})}\n\n"

                # 2. 工具调用开始
                elif kind == "on_tool_start":
                    tool_name = event["name"]
                    yield f"data: {json.dumps({'type': 'thinking', 'text': f'[🔧 调用工具 {tool_name}...]'})}\n\n"

                # 3. 工具调用结束
                elif kind == "on_tool_end":
                    result = str(event["data"]["output"])
                    yield f"data: {json.dumps({'type': 'tool', 'text': f'[✅ 工具结果: {result}]'})}\n\n"

                # 4. Agent 完成最终回复
                elif kind == "on_chain_end" and event["name"] == "Agent":
                    # 可选：发送完成信号
                    yield f"data: {json.dumps({'type': 'final', 'text': ''})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': f'❌ {str(e)}'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")