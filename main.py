from utils import TEAM_MEMBERS
from graph import build_graph
import logging
import os

logger = logging.getLogger(__name__)

def main():
    """Main function to run the agent."""

    graph = build_graph()

    user_query = input("Enter your query: ")

    result = graph.invoke(
        {
            # Constants
            "TEAM_MEMBERS": TEAM_MEMBERS,
            # Runtime Variables
            "messages": [{"role": "user", "content": user_query}],
        }
    )

    print("\n=== Conversation History ===")
    for message in result["messages"]:
        role = message.type
        print(f"\n[{role.upper()}]: {message.content}")
    print("\n=============================\n")


if __name__ == "__main__":
    OUTPUT_LOG = "logs/output.log"
    if os.path.exists(OUTPUT_LOG):
        os.remove(OUTPUT_LOG)
    os.makedirs(os.path.dirname(OUTPUT_LOG), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,  # Default level is INFO
        filename=OUTPUT_LOG,
        filemode="a",  # "w"で上書き, "a"で追記
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()