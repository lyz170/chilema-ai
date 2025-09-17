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

if "MONGODB_PASSWORD" not in os.environ:
    os.environ["MONGODB_PASSWORD"] = getpass.getpass(
        prompt="Enter your MongoDB password (required if using MongoDB): "
    )

# Initialize chat model
model = init_chat_model("deepseek-chat", model_provider="deepseek")

system_message = """
你是一个专业的菜谱生成助手。请按照以下步骤处理用户请求：
1. 接收用户提供的"菜谱名称"(必需项)和"额外说明"(可选项)
2. 根据下面的从MongoDB数据库中取出的3个菜谱JSON格式的例子，根据用户提供的"菜谱名称"和"额外说明"生成一个新的菜谱
  - 示例1: {input_recipe_sample_1}
  - 示例2: {input_recipe_sample_2}
  - 示例3: {input_recipe_sample_3}
3. 确保生成的菜谱也为JSON，并且数据结构与示例一致(id字段去掉以便MongoDB自动生成)，以便存储在MongoDB数据库中
4. 调用相应Agent，把这个JSON格式的菜谱存入MongoDB数据库
注意事项：
- 如果用户提供的菜谱名称不明确，请询问澄清
- 对于可选项"额外说明"，如果有值，要合理融入到菜谱生成中
- 确保生成的JSON格式严格符合数据库要求
- 如果遇到技术问题，请明确告知用户
"""

## define user prompt
user_prompt = "菜谱名称: {input_recipe_name}  额外说明: {input_additional_desc}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)


class State(TypedDict):
    """State of the orchestration."""
    recipe_name: str
    additional_desc: str
    recipe_id: str
    recipe_json: str
    additional_info_need_to_clarify: str
    technical_issue: str
    answer: str


class QueryOutput(TypedDict):
    """Generated MongoDB query."""
    recipe_json: str  # JSON string
    additional_info_need_to_clarify: str  # string, if no need to clarify, return empty string
    technical_issue: str  # string, if no technical issue, return empty string


from pymongo import MongoClient

uri = "mongodb://chilema_dev_user:" + os.environ["MONGODB_PASSWORD"] + "@localhost:27017/?authSource=chilema_dev"
mongo_client = MongoClient(uri)
db = mongo_client["chilema_dev"]
cn = db["recipes"]


def get_recipe_samples():
    """Get a sample of the collection."""
    samples = cn.find().limit(3)
    if not samples:
        return ['{}', '{}', '{}']
    sample_list = []
    for sample in samples:
        sample_list.append(str(sample))
    while len(sample_list) < 3:
        sample_list.append('{}')
    print("======> get_recipe_samples" + str(sample_list))
    return sample_list


def lc_write_recipe_query(state: State):
    sample_list = get_recipe_samples()
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "input_recipe_sample_1": sample_list[0],
            "input_recipe_sample_2": sample_list[1],
            "input_recipe_sample_3": sample_list[2],
            "input_recipe_name": state["recipe_name"],
            "input_additional_desc": state.get("additional_desc", ""),
        }
    )
    structured_llm = model.with_structured_output(QueryOutput)
    res = structured_llm.invoke(prompt)
    print("======> lc_write_recipe_query: " + str(res))
    return {
        "recipe_json": res["recipe_json"],
        "additional_info_need_to_clarify": res["additional_info_need_to_clarify"],
        "technical_issue": res["technical_issue"]
    }


def lc_insert_recipe(state: State):
    """Insert a recipe into the collection."""
    if state["additional_info_need_to_clarify"]:
        return {"recipe_id": ""}
    if state["technical_issue"]:
        return {"recipe_id": ""}
    try:
        recipe_json = state["recipe_json"]
        if not recipe_json:
            return {"recipe_id": ""}
        result = cn.insert_one(json.loads(recipe_json))
        recipe_id = result.inserted_id
        return {"recipe_id": str(recipe_id)}
    except Exception as e:
        return {"recipe_id": "", "technical_issue": str(e)}


def lc_generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        '''
        如果完成了菜谱生成，并且已经把菜谱存入数据库，请回复:
        菜谱已成功生成并存入数据库
        ID: {recipe_id}
        内容: {recipe_json}
        '''
        '''
        如果需要澄清更多信息，请回复:
        需要澄清以下信息以生成菜谱: {additional_info_need_to_clarify}
        '''
        '''
        如果遇到技术问题，请回复:
        遇到技术问题: {technical_issue}
        '''
    ).format(
        recipe_id=state.get("recipe_id", ""),
        recipe_json=state.get("recipe_json", ""),
        additional_info_need_to_clarify=state.get("additional_info_need_to_clarify", ""),
        technical_issue=state.get("technical_issue", "")
    )
    response = model.invoke(prompt)
    return {"answer": response}


## Use LangGraph to orchestrate the steps
from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence(
    [lc_write_recipe_query, lc_insert_recipe, lc_generate_answer]
)
graph_builder.add_edge(START, "lc_write_recipe_query")
graph = graph_builder.compile()

# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

with open("../image/generate_recipe.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())
img = mpimg.imread("../image/generate_recipe.png")
plt.imshow(img)
plt.axis('off')
plt.show()

last_step = None
for step in graph.stream(
        {"recipe_name": "红烧鸡块",
         "additional_desc": "只使用糖，盐，生抽老抽，料酒，葱姜等常见调料，不使用任何复杂调料"},
        stream_mode="values"
):
    last_step = step

if last_step and 'answer' in last_step:
    print(last_step['answer'])
