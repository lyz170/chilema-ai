import getpass
import os
from datetime import datetime

from dotenv import load_dotenv

try:
    load_dotenv()
except ImportError:
    pass

if "DEEPSEEK_API_KEY" not in os.environ:
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass(
        prompt="Enter your OpenAI API key (required if using OpenAI): "
    )

##---------------------------------------------------
## (1) Define Model & Memory
##---------------------------------------------------
# from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

llm = init_chat_model(
    model="deepseek-chat",
    temperature=0.1,  # A higher number makes responses more creative; lower ones make them more deterministic.
    timeout=30,
    max_tokens=1000,
    max_retries=2
)

checkpointer = InMemorySaver()

##---------------------------------------------------
## (2) Define tools
##---------------------------------------------------
from langchain_tavily import TavilySearch

tavily_tool = TavilySearch(max_results=2)

# Augment the LLM with tools
tools = [tavily_tool]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

##---------------------------------------------------
## (3) Define state
##---------------------------------------------------
from langchain.messages import AnyMessage, SystemMessage
from typing_extensions import TypedDict, Annotated
import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    is_time_sensitive: bool


##---------------------------------------------------
## (4) Define model node
##---------------------------------------------------
def llm_call(state: dict):
    print("Using LLM directly...")
    # 添加系统消息
    system_message = SystemMessage(
        content="你是一个对话机器人，用于测试各种LLM API，因为仅用于测试，回答问题时请简明扼要"
    )
    # 将系统消息添加到messages列表的最前面
    messages_with_system = [system_message] + state["messages"]
    return {
        "messages": [llm.invoke(messages_with_system)],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


def llm_call_with_tools(state: dict):
    print("Using LLM with tools...")
    # 添加系统消息
    system_message = SystemMessage(
        content="你是一个对话机器人，用于测试各种LLM API，因为仅用于测试，回答问题时请简明扼要"
    )
    # 将系统消息添加到messages列表的最前面
    messages_with_system = [system_message] + state["messages"]
    return {
        "messages": [llm_with_tools.invoke(messages_with_system)],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


##---------------------------------------------------
## (5) Define tool node
##---------------------------------------------------
def is_time_sensitive_node(state: MessagesState):
    """
    让LLM判断问题是否为时效性问题。
    """
    question = state["messages"][-1].content
    system_prompt = """
      你是一个对话机器人。请判断用户的问题是否涉及时效性（即答案会随时间变化），
      如果是请回答 'YES'，否则回答 'NO'。只需输出'YES'或'NO'。
    """
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    result = llm.invoke(prompt)
    answer = result.content.strip()
    # print the question and the answer
    print(f"Question: {question} | Is time-sensitive? {answer}")
    return {
        "messages": state["messages"],
        "llm_calls": state.get('llm_calls', 0) + 1,
        "is_time_sensitive": answer == "YES"
    }


def get_current_datetime_node(state: MessagesState):
    """
    获取当前的日期和时间，并将其作为新消息添加到messages列表中。
    """
    # 获取当前日期时间
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # 打印获取到的时间
    print(f"Current datetime: {formatted_time}")

    # 创建包含日期时间信息的系统消息
    datetime_message = SystemMessage(
        content=f"当前日期和时间: {formatted_time}"
    )

    # 返回更新后的state，将日期时间消息添加到messages列表
    return {
        "messages": state["messages"] + [datetime_message],
        "llm_calls": state.get('llm_calls', 0),
        "is_time_sensitive": state.get('is_time_sensitive', False)
    }


from langgraph.prebuilt import ToolNode, tools_condition

tool_node = ToolNode(tools=tools)

##---------------------------------------------------
## (6) Build and compile the agent
##---------------------------------------------------
from langgraph.graph import StateGraph, END

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("is_time_sensitive_node", is_time_sensitive_node)
graph_builder.add_node("get_current_datetime_node", get_current_datetime_node)
graph_builder.add_node("llm_call_with_tools", llm_call_with_tools)
graph_builder.add_node("llm_call", llm_call)
graph_builder.add_node("tool_node", tool_node)


# 定义条件函数来决定下一步
def decide_time_sensitive_route(state: MessagesState):
    """
    根据问题是否有时效性决定路由
    """
    if state.get("is_time_sensitive", False):
        return "time_sensitive"
    else:
        return "normal"


# 设置工作流
graph_builder.set_entry_point("is_time_sensitive_node")

# 从is_time_sensitive_node分支
graph_builder.add_conditional_edges(
    "is_time_sensitive_node",
    decide_time_sensitive_route,
    {
        "time_sensitive": "get_current_datetime_node",
        "normal": "llm_call"
    }
)

# 时效性问题路径：获取时间 -> 使用带工具的LLM
graph_builder.add_edge("get_current_datetime_node", "llm_call_with_tools")

# 对于带工具的LLM，使用tools_condition来决定是否需要调用工具
graph_builder.add_conditional_edges(
    "llm_call_with_tools",
    tools_condition,
    {
        "tools": "tool_node",
        END: END,
    }
)

# 工具调用后回到LLM（形成循环）
graph_builder.add_edge("tool_node", "llm_call_with_tools")

# 普通问题路径：直接到END
graph_builder.add_edge("llm_call", END)

# 编译图
agent = graph_builder.compile(checkpointer=checkpointer)

##---------------------------------------------------
## (7) Show the graph
##---------------------------------------------------
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# with open("./image/agent.png", "wb") as f:
#     f.write(agent.get_graph().draw_mermaid_png())
# img = mpimg.imread("./image/agent.png")
# plt.imshow(img)
# plt.axis('off')


##---------------------------------------------------
## (8) TEST
##---------------------------------------------------
# 兼容打印：支持消息对象（有 .content）或 dict
# def stream_graph_updates(user_input: str):
#     messages = [HumanMessage(content=user_input)]
#     config = {"configurable": {"thread_id": "abc123"}}
#     for event in agent.stream({"messages": messages}, config=config):
#         for value in event.values():
#             last = value["messages"][-1]
#             if hasattr(last, "content"):
#                 content = last.content
#             elif isinstance(last, dict):
#                 content = last.get("content") or last.get("text") or str(last)
#             else:
#                 content = str(last)
#             print("Assistant:", content)


# while True:
#     user_input = input("User: ")
#     if user_input.lower() in ["quit", "exit", "q"]:
#         print("Goodbye!")
#         break
#     stream_graph_updates(user_input)
