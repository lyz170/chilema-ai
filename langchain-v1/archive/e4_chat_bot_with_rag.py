import getpass
import os
from datetime import datetime
from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

try:
    load_dotenv()
except ImportError:
    pass

if "DEEPSEEK_API_KEY" not in os.environ:
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass(
        prompt="Enter your OpenAI API key (required if using OpenAI): "
    )
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass(
        prompt="Enter your Tavily API key (required if using Tavily): "
    )

# 初始化模型和工具
llm = init_chat_model("deepseek-chat", model_provider="deepseek")
memory = MemorySaver()

from langchain_tavily import TavilySearch

tool = TavilySearch(max_results=2)
tools = [tool]


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    is_time_sensitive: bool  # 新增字段


graph_builder = StateGraph(State)

# Tell the LLM which tools it can call
llm_with_tools = llm.bind_tools(tools)


def is_time_sensitive_node(state: State):
    """
    让LLM判断问题是否为时效性问题。
    """
    question = state["messages"][-1].content
    system_prompt = (
        "你是一个对话机器人。请判断用户的问题是否涉及时效性（即答案会随时间变化），"
        "如果是请回答 '是'，否则回答 '否'。只需输出'是'或'否'。"
    )
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    result = llm.invoke(prompt)
    answer = result.content.strip()
    # print the question and the answer
    print(f"Question: {question} | Is time-sensitive? {answer}")
    return {"messages": state["messages"], "is_time_sensitive": answer == "是"}


def chatbot_with_tool(state: State):
    print("Using LLM with tools...")
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def chatbot_no_tool(state: State):
    print("Using LLM without tools...")
    return {"messages": [llm.invoke(state["messages"])]}


# 修改流程：START -> is_time_sensitive -> chatbot/tools
def route_by_time_sensitive(state: State):
    if state.get("is_time_sensitive"):
        return "chatbot_with_tool"  # 走原有工具判断流程
    else:
        return "chatbot_no_tool"  # 新增无tool流程


graph_builder.add_node("is_time_sensitive", is_time_sensitive_node)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("chatbot_with_tool", chatbot_with_tool)
graph_builder.add_node("chatbot_no_tool", chatbot_no_tool)

graph_builder.add_edge(START, "is_time_sensitive")
graph_builder.add_conditional_edges(
    "is_time_sensitive",
    route_by_time_sensitive,
    {"chatbot_with_tool": "chatbot_with_tool", "chatbot_no_tool": "chatbot_no_tool"}
)
graph_builder.add_conditional_edges(
    "chatbot_with_tool",
    tools_condition
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot_with_tool")

# chatbot_no_tool直接结束
graph_builder.add_edge("chatbot_no_tool", END)

graph = graph_builder.compile()

# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

with open("../image/e4_chat_bot_with_rag.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())
img = mpimg.imread("../image/e4_chat_bot_with_rag.png")
plt.imshow(img)
plt.axis('off')
plt.show()


def get_system_message():
    """
    返回包含机器人身份和当前日期时间的 system_message。
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = (
        "你是一个对话机器人，用于测试各种LLM API，因为仅用于测试，回答问题时请简明扼要。"
        f"调用Agent或AI tool时，请把当前时间也加上传过去。当前日期时间是：{now}。"
    )
    return {"role": "system", "content": content}


def stream_graph_updates(user_input: str):
    # 插入 system_message 到 prompt
    messages = [get_system_message(), {"role": "user", "content": user_input}]
    state = {"messages": messages}
    for event in graph.stream(state):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
