import logging
import json
import json_repair
import os
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape
from langgraph.prebuilt.chat_agent_executor import AgentState

import os
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()

# 値を取得する
NAME = os.getenv('NAME')
MODEL_NAME = os.getenv('MODEL_NAME')
TEMPERATURE = os.getenv('TEMPERATURE', None)
if TEMPERATURE:
    TEMPERATURE = float(TEMPERATURE)
API_BASE = os.getenv('API_BASE')
API_KEY = os.getenv('API_KEY')
API_VERSION = os.getenv('API_VERSION')
os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')

logger = logging.getLogger(__name__)


# Team configuration
TEAM_MEMBER_CONFIGRATIONS = {
    "researcher": {
        "name": "researcher",
        "desc": (
            "Responsible for searching and collecting relevant information, understanding user needs and conducting research analysis"
        ),
        "is_optional": False,
    },
    "reporter": {
        "name": "reporter",
        "desc": (
            "Responsible for summarizing analysis results, generating reports and presenting final outcomes to users, File output is not available."
        ),
        "is_optional": False,
    },
    "file_manager": {
        "name": "file_manager",
        "desc": (
            "Responsible for saving results to markdown files. Formats content nicely with proper markdown syntax before saving."
        ),
        "is_optional": True,
    },
}

TEAM_MEMBERS = list(TEAM_MEMBER_CONFIGRATIONS.keys())

def repair_json_output(content: str) -> str:
    """
    jsonを修復する
    """
    content = content.strip()
    if content.startswith(("{", "[")) or "```json" in content:
        try:
            # 如果内容被包裹在```json代码块中，提取JSON部分
            if content.startswith("```json"):
                content = content.removeprefix("```json")

            if content.endswith("```"):
                content = content.removesuffix("```")

            # 尝试修复并解析JSON
            repaired_content = json_repair.loads(content)
            return json.dumps(repaired_content)
        except Exception as e:
            logger.warning(f"JSON repair failed: {e}")
    return content


# Initialize Jinja2 environment
env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__)),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def get_prompt_template(prompt_name: str) -> str:
    """
    Load and return a prompt template using Jinja2.

    Args:
        prompt_name: Name of the prompt template file (without .md extension)

    Returns:
        The template string with proper variable substitution syntax
    """
    try:
        template = env.get_template(f"prompt/{prompt_name}.md")
        return template.render()
    except Exception as e:
        raise ValueError(f"Error loading template {prompt_name}: {e}")


def apply_prompt_template(prompt_name: str, state: AgentState) -> list:
    """
    Apply template variables to a prompt template and return formatted messages.

    Args:
        prompt_name: Name of the prompt template to use
        state: Current agent state containing variables to substitute

    Returns:
        List of messages with the system prompt as the first message
    """
    # Convert state to dict for template rendering
    state_vars = {
        "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
        **state,
    }

    try:
        template = env.get_template(f"prompt/{prompt_name}.md")
        system_prompt = template.render(**state_vars)
        return [{"role": "system", "content": system_prompt}] + state["messages"]
    except Exception as e:
        raise ValueError(f"Error applying template {prompt_name}: {e}")

def initialize_llm(name: str, model_name: str, temperature: float):
    """
    指定されたパラメータを用いて LLM を初期化する関数。
    """
    llm = None

    # 例: Azure, Google, VertexAI, HuggingFace, OpenAI_Base, xAI, Ollama 等
    if name == "Azure":
        from langchain_openai import AzureChatOpenAI
        llm = AzureChatOpenAI(
            azure_deployment=model_name,
            temperature=temperature,
            azure_endpoint=API_BASE,
            api_version=API_VERSION,
            api_key=API_KEY,
        )

    elif name == "Google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=API_KEY,
            temperature=temperature
        )


    elif name == "OpenAI_Base":
        # DeepSeekやOpenAIのエンドポイント指定例。用途に応じて調整してください。
        api_key = API_KEY
        api_endpoint = "https://api.openai.com/v1"
        if "deepseek" in model_name:
            api_endpoint = "https://api.deepseek.com"

        if api_key == "":
            raise ValueError("OpenAI API Key が設定されていません。")

        from langchain.chat_models import ChatOpenAI  # ここだけ例としてインポートを追加
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=api_endpoint,
            max_tokens=2048,  # CONFIGなどに合わせて調整
            temperature=temperature
        )

    elif name == "xAI":
        from some_xai_module import ChatXAI  # xAI用の仮の例
        llm = ChatXAI(
            model=model_name,
            api_key=API_KEY,
            max_tokens=2048,  # CONFIGなどに合わせて調整
            temperature=temperature,
        )
    else:
        raise ValueError("サポートされていない LLM 名が指定されました。")

    return llm