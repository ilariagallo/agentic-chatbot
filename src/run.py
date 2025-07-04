import uuid

from langchain_core.messages import HumanMessage

from src.assistant import AssistantGraph
from src.database import update_dates, local_file

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
        user_input = input("\n👤 User:\n")
        messages = [HumanMessage(content=user_input)]
        response = assistant.graph.invoke({"messages": messages}, config)
        ai_message = response['messages'][-1]

        # Display output
        print("\n🤖 Assistant:\n", ai_message)
