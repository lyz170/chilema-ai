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
## (1) Define tools and model
##---------------------------------------------------
from langchain.tools import tool
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="deepseek-chat",
    temperature=0
)


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


@tool
def get_current_datetime() -> str:
    """获取当前的日期和时间。
    
    Returns:
        str: 格式化的当前日期和时间字符串
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Augment the LLM with tools
tools = [add, multiply, divide, get_current_datetime]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

##---------------------------------------------------
## (2) Define state - used to store the messages and the number of LLM calls
##---------------------------------------------------
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


##---------------------------------------------------
## (3) Define model node - used to call the LLM and decide whether to call a tool or not
##---------------------------------------------------
from langchain.messages import SystemMessage


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant. You can perform arithmetic operations on numbers and also get the current date and time."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


##---------------------------------------------------
## (4) Define tool node - is used to call the tools and return the results
##---------------------------------------------------
from langchain.messages import ToolMessage


def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


##---------------------------------------------------
## (5) Define end logic - used to route to the tool node or end based upon whether the LLM made a tool call
##---------------------------------------------------
from typing import Literal
from langgraph.graph import StateGraph, START, END


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"
    # Otherwise, we stop (reply to the user)
    return END


##---------------------------------------------------
## (6) Build and compile the agent
##---------------------------------------------------
# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()