## pip install --pre -U langchain
## pip install --pre -U langchain-deepseek
## pip install python-dotenv

import getpass
import os

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


# Augment the LLM with tools
tools = [add, multiply, divide]
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
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
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

# Show the agent
# from IPython.display import Image, display
# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
# pip install matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

with open("../image/e1_quick_start.png", "wb") as f:
    f.write(agent.get_graph().draw_mermaid_png())
img = mpimg.imread("../image/e1_quick_start.png")
plt.imshow(img)
plt.axis('off')
# plt.show()

# Invoke
from langchain.messages import HumanMessage

messages = [HumanMessage(content="Add 3 and 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
