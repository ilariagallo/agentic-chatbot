from typing import Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages

from config import LLM
from tools import *
from utils import create_tool_node_with_fallback
from vector_store import *

from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# The checkpointer lets the graph persist its state
memory = MemorySaver()

tools = [
    TavilySearch(max_results=1),
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    update_ticket_to_new_flight,
    cancel_ticket,
    search_car_rentals,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    search_hotels,
    book_hotel,
    update_hotel,
    cancel_hotel,
    search_trip_recommendations,
    book_excursion,
    update_excursion,
    cancel_excursion,
]

# Define the state of the graph
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class AssistantGraph:

    """
    The main assistant class that orchestrates the customer support interactions.
    It uses a primary prompt template and a set of tools to handle various user queries.
    """

    def __init__(self):
        builder = StateGraph(State)

        # Define nodes: these do the work
        builder.add_node("assistant", self.primary_assistant)
        builder.add_node("tools", create_tool_node_with_fallback(tools))
        # Define edges: these determine how the control flow moves
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        builder.add_edge("tools", "assistant")
        self.graph = builder.compile(checkpointer=memory)

    def primary_assistant(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}

            primary_assistant_prompt = self.build_prompt(passenger_id)
            assistant_runnable = primary_assistant_prompt | LLM.bind_tools(tools)
            result = assistant_runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                    not result.content
                    or isinstance(result.content, list)
                    and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

    @staticmethod
    def build_prompt(user_info: str = None) -> ChatPromptTemplate:
        """
        Builds the primary assistant prompt template with the current time and user information.
        This is used to initialize the assistant's context for each interaction.
        """
        primary_assistant_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful customer support assistant for Swiss Airlines. "
                    " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
                    " When searching, be persistent. Expand your query bounds if the first search returns no results. "
                    " If a search comes up empty, expand your search before giving up."
                    "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
                    "\nCurrent time: {time}.",
                ),
                ("placeholder", "{messages}"),
            ]
        ).partial(time=datetime.now, user_info=user_info)

        return primary_assistant_prompt
