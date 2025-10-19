import getpass
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

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
deepseek_llm = init_chat_model("deepseek-chat", model_provider="deepseek")
memory = MemorySaver()

# Tavily 搜索工具
search = TavilySearch(max_results=2)
tools = [search]
agent_executor = create_react_agent(deepseek_llm, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}

# --- 测试运行 ---
if __name__ == "__main__":

    system_message = """
    你是一个对话机器人，用于测试各种LLM API，因为仅用于测试，回答问题时请简明扼要。
    """
    user_prompt = "{question}"
    prompt_template = ChatPromptTemplate([
        ("system", system_message),
        ("user", user_prompt)
    ])

    while True:
        user_input = input("Question: ")
        if user_input.lower() == "quit":
            break

        prompt = prompt_template.invoke({"question": user_input})
        input_message = {
            "role": "user",
            "content": prompt.to_messages()[-1].content,
        }
        for step in agent_executor.stream(
                {"messages": [input_message]}, config, stream_mode="values"
        ):
            step["messages"][-1].pretty_print()
