import uuid

from langchain_core.messages import HumanMessage

from assistant import AssistantGraph
from database import update_dates, local_file
from utils import _print_event

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

    assistant = AssistantGraph()

    conversation_ongoing = True
    while conversation_ongoing:
        _printed = set()

        user_input = input("\n👤 User:\n")
        messages = [HumanMessage(content=user_input)]
        events = assistant.graph.stream({"messages": messages}, config)

        for event in events:
            _print_event(event, _printed)

        # # Display output
        # response = assistant.graph.invoke({"messages": messages}, config)
        # ai_message = response['messages'][-1]
        # print("\n🤖 Assistant:\n", ai_message)
