import logging
import json
import json_repair
import logging
from typing import Literal, Annotated
from typing_extensions import TypedDict
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.tools import tool

import json_repair
from langgraph.types import Command

from utils import initialize_llm, TEAM_MEMBERS, apply_prompt_template, repair_json_output, NAME, MODEL_NAME, TEMPERATURE, API_BASE, API_KEY, API_VERSION

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch, TavilyExtract
from langchain_community.tools import WriteFileTool
import os


logger = logging.getLogger(__name__)
# Define routing options
OPTIONS = TEAM_MEMBERS + ["FINISH"]
RESPONSE_FORMAT = "Response from {}:\n\n<response>\n{}\n</response>\n\n*Please execute the next step.*"


#==============================================================================
# LangGraph Workflow
#==============================================================================
class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*OPTIONS]

class State(MessagesState):
    """State for the agent system, extends MessagesState with next field."""

    # Constants
    TEAM_MEMBERS: list[str]

    # Runtime Variables
    next: str
    full_plan: str

llm = initialize_llm(name=NAME, model_name=MODEL_NAME, temperature=TEMPERATURE)

# Initialize Tavily Search Tool
tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
)
tavily_extract_tool = TavilyExtract()

write_file_tool = WriteFileTool()

# Create agents using configured LLM types
research_agent = create_react_agent(
    llm,
    tools=[tavily_search_tool, tavily_extract_tool],
    prompt=lambda state: apply_prompt_template("researcher", state),
    debug=False,
)

file_manager_agent = create_react_agent(
    llm,
    tools=[write_file_tool],
    prompt=lambda state: apply_prompt_template("file_manager", state),
    debug=False,
)



def research_node(state: State) -> Command[Literal["supervisor"]]:
    """
    リサーチノード: research_agent を起動して、外部検索や補助的な情報取得を行う。
    """
    logger.info("\n=== [Research Node] リサーチノードが開始されました ===")
    logger.info("リサーチエージェントがタスクを実行します。")
    logger.info(f"--- 入力メッセージ一覧 (State) ---\n{state.get('messages')}\n---")

    # 実際のエージェント呼び出し
    result = research_agent.invoke(state)

    logger.info("リサーチエージェントがタスクを完了しました。")

    # エージェントの最終出力メッセージを取得
    response_content = result["messages"][-1].content
    response_content = repair_json_output(response_content)

    logger.info(f"--- リサーチエージェントの最終出力 ---\n{response_content}\n---")
    logger.info("=== [Research Node] リサーチノードが終了します ===\n")

    return Command(
        update={
            "messages": [
                AIMessage(
                    content=response_content,
                    name="researcher",
                )
            ]
        },
        goto="supervisor",
    )


def supervisor_node(state: State) -> Command[Literal[*TEAM_MEMBERS, "__end__"]]:
    """
    スーパーバイザーノード: 次に動くノード(誰にタスクを渡すか)を判断する。
    """
    logger.info("\n=== [Supervisor Node] スーパーバイザーが次のアクションを評価します ===")
    logger.info(f"--- Supervisorの入力メッセージ一覧 (State) ---\n{state.get('messages')}\n---")

    messages = apply_prompt_template("supervisor", state)
    for message in messages:
        if isinstance(message, BaseMessage) and message.name in TEAM_MEMBERS:
            message.content = RESPONSE_FORMAT.format(message.name, message.content)
    
    logger.info(f"\n--- Supervisor用のプロンプト {messages} をLLMに送信します ---\n")
    # LLMにSupervisorのプロンプトを渡し、次にどのノードへ行くかを取得
    response = (
        llm.with_structured_output(schema=Router, method="json_mode")
        .invoke(messages)
    )
    goto = response["next"]

    logger.info(f"--- Supervisorの判断結果: 次に動くノードは[{goto}]です ---")

    if goto == "FINISH":
        goto = "__end__"
        logger.info("=== ワークフローが完了しました（FINISH） ===")
    else:
        logger.info(f"=== [{goto}]に処理を移します ===")

    logger.info("=== [Supervisor Node] スーパーバイザーノードが終了します ===\n")

    return Command(goto=goto, update={"next": goto})


def planner_node(state: State) -> Command[Literal["supervisor", "__end__"]]:
    """
    プランナーノード: 今後の計画（フルプラン）を生成する。
    """
    logger.info("\n=== [Planner Node] プランナーノードが開始されました ===")
    logger.info("プランナーが今後の全体計画を生成します。検索を行うかどうかを確認します。")
    logger.info(f"--- Plannerの入力メッセージ一覧 (State) ---\n{state.get('messages')}\n---")

    messages = apply_prompt_template("planner", state)

    logger.info("\n--- Planner用の最終プロンプト (messages) をLLMに送信します ---\n")

    response = llm.invoke(messages)
    full_response = response.content
    logger.info("\n--- Plannerが返したフルプラン (JSON 形式を想定) ---\n" + full_response + "\n---")

    # JSON構造の修正を試みる
    if full_response.startswith("\njson"):
        full_response = full_response.removeprefix("\njson")
    if full_response.endswith("\n"):
        full_response = full_response.removesuffix("\n")

    goto = "supervisor"
    try:
        repaired_response = json_repair.loads(full_response)
        full_response = json.dumps(repaired_response, ensure_ascii=False)
        logger.info("Plannerの出力は有効なJSONと判定し、修正を行いました。")
    except json.JSONDecodeError:
        logger.warning("Plannerの出力がJSONとして不正です。ワークフローを終了します。")
        goto = "__end__"

    logger.info(f"--- 今後のグラフの動き(フルプラン) ---\n{full_response}\n---")
    logger.info("=== [Planner Node] プランナーノードが終了します ===\n")

    return Command(
        update={
            "messages": [AIMessage(content=full_response, name="planner")],
            "full_plan": full_response,
        },
        goto=goto,
    )

def file_manager_node(state: State) -> Command[Literal["supervisor"]]:
    """
    file_managerノード: 必要に応じて結果をファイルに保存する。
    """
    logger.info("\n=== [file_manager Node] file_managerノードが開始されました ===")
    logger.info("file_managerノードがファイルに結果を保存します。")
    logger.info(f"--- file_managerの入力メッセージ一覧 (State) ---\n{state.get('messages')}\n---")

    result = file_manager_agent.invoke(state)
    response_content = result["messages"][-1].content

    logger.info("LLMからの回答を取得しました。JSONとして修正を試みます。")

    response_content = repair_json_output(response_content)

    logger.info(f"--- file_managerの出力 (修正後) ---\n{response_content}\n---")
    logger.info("=== [file_manager Node] file_managerノードが終了します ===\n")

    return Command(
        update={
            "messages": [
                AIMessage(
                    content=response_content,
                    name="file_manager",
                )
            ]
        },
        goto="supervisor",
    )


def reporter_node(state: State) -> Command[Literal["supervisor"]]:
    """
    レポーターノード: 最終的なレポートやサマリーを作成する。
    """
    logger.info("\n=== [Reporter Node] レポーターノードが開始されました ===")
    logger.info("レポーターが最終的なレポートをまとめます。")
    logger.info(f"--- Reporterの入力メッセージ一覧 (State) ---\n{state.get('messages')}\n---")

    messages = apply_prompt_template("reporter", state)
    response = llm.invoke(messages)

    logger.info("LLMからの回答を取得しました。JSONとして修正を試みます。")

    response_content = response.content
    response_content = repair_json_output(response_content)

    logger.info(f"--- レポーターの出力 (修正後) ---\n{response_content}\n---")
    logger.info("=== [Reporter Node] レポーターノードが終了します ===\n")

    return Command(
        update={
            "messages": [
                AIMessage(
                    content=response_content,
                    name="reporter",
                )
            ]
        },
        goto="supervisor",
    )
    
def build_graph():
    """Build and return the agent workflow graph."""
    builder = StateGraph(State)
    builder.add_edge(START, "planner")
    builder.add_node("planner", planner_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", research_node)
    builder.add_node("file_manager", file_manager_node)
    builder.add_node("reporter", reporter_node)
    return builder.compile()