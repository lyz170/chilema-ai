import getpass
import json
import os

## pip install --upgrade --quiet langchain-community langgraph
## pip install -U langchain-deepseek
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict

try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

if "DEEPSEEK_API_KEY" not in os.environ:
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass(
        prompt="Enter your OpenAI API key (required if using OpenAI): "
    )


# Initialize chat model
model = init_chat_model("deepseek-chat", model_provider="deepseek")

system_message = "你是一个对话机器人，用于测试各种LLM API，因为仅用于测试，回答问题时请简明扼要"
user_prompt = "今天几月几号星期几"

prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

prompt = prompt_template.invoke({})

print(prompt.to_messages())
response = model.invoke(prompt)
print(response.content)