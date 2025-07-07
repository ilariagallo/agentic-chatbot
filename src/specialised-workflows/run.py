import uuid

from langchain_core.messages import HumanMessage, ToolMessage

from graph import CustomerSupportBot
from database import update_dates, local_file
from utils import _print_event

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

db = update_dates(local_file)

if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            # The passenger_id is used in our flight tools to
            # fetch the user's flight information
            "passenger_id": "3442 587242",
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }

    assistant = CustomerSupportBot()

    conversation_ongoing = True
    while conversation_ongoing:
        _printed = set()

        user_input = input("\n👤 User:\n")
        # messages = [HumanMessage(content=user_input)]
        events = assistant.graph.stream({"messages": ("user", user_input)}, config, stream_mode='values')

        for event in events:
            _print_event(event, _printed)

        snapshot = assistant.graph.get_state(config)
        while snapshot.next:
            # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
            # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
            # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
            try:
                user_input = input(
                    "Do you approve of the above actions? Type 'y' to continue;"
                    " otherwise, explain your requested changed.\n\n"
                )
            except:
                user_input = "y"
            if user_input.strip() == "y":
                # Just continue
                result = assistant.graph.invoke(None, config)
            else:
                # Satisfy the tool invocation by
                # providing instructions on the requested changes / change of mind
                result = assistant.graph.invoke(
                    {
                        "messages": [
                            ToolMessage(
                                tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                content=f"API call denied by user. Reasoning: '{user_input}'. "
                                        f"Continue assisting, accounting for the user's input.",
                            )
                        ]
                    },
                    config,
                )
            snapshot = assistant.graph.get_state(config)
