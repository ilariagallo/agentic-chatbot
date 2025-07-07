from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.constants import START

from utils import create_entry_node, create_tool_node_with_fallback, pop_dialog_state
from assistant import Assistant
from primary_assistant import *
from flight_booking_assistant import *
from car_rental_assistant import *
from hotel_booking_assistant import *
from excursion_assistant import *

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# The checkpointer lets the graph persist its state
memory = MemorySaver()

class CustomerSupportBot:
    """
    The main customer support bot class that orchestrates the conversation.
    It uses a primary assistant and a set of specialised workflows to handle various user queries.
    """

    def __init__(self):
        builder = StateGraph(State)
        builder.add_node("fetch_user_info", self.user_info)
        builder.add_edge(START, "fetch_user_info")

        # Flight booking assistant
        ## Nodes
        builder.add_node(
            "enter_update_flight",
            create_entry_node("Flight Updates & Booking Assistant", "update_flight"),
        )
        builder.add_node("update_flight", Assistant(update_flight_runnable))
        builder.add_node(
            "update_flight_sensitive_tools",
            create_tool_node_with_fallback(update_flight_sensitive_tools),
        )
        builder.add_node(
            "update_flight_safe_tools",
            create_tool_node_with_fallback(update_flight_safe_tools),
        )
        ## Edges
        builder.add_edge("enter_update_flight", "update_flight")
        builder.add_conditional_edges(
            "update_flight",
            route_update_flight,
            ["update_flight_sensitive_tools", "update_flight_safe_tools", "leave_skill", END],
        )
        builder.add_edge("update_flight_sensitive_tools", "update_flight")
        builder.add_edge("update_flight_safe_tools", "update_flight")

        # Car rental assistant
        ## Nodes
        builder.add_node(
            "enter_book_car_rental",
            create_entry_node("Car Rental Assistant", "book_car_rental"),
        )
        builder.add_node("book_car_rental", Assistant(book_car_rental_runnable))
        builder.add_node(
            "book_car_rental_safe_tools",
            create_tool_node_with_fallback(book_car_rental_safe_tools),
        )
        builder.add_node(
            "book_car_rental_sensitive_tools",
            create_tool_node_with_fallback(book_car_rental_sensitive_tools),
        )

        ## Edges
        builder.add_edge("enter_book_car_rental", "book_car_rental")
        builder.add_conditional_edges(
            "book_car_rental",
            route_book_car_rental,
            [
                "book_car_rental_safe_tools",
                "book_car_rental_sensitive_tools",
                "leave_skill",
                END,
            ],
        )
        builder.add_edge("book_car_rental_sensitive_tools", "book_car_rental")
        builder.add_edge("book_car_rental_safe_tools", "book_car_rental")

        # Hotel booking assistant
        ## Nodes
        builder.add_node(
            "enter_book_hotel", create_entry_node("Hotel Booking Assistant", "book_hotel")
        )
        builder.add_node("book_hotel", Assistant(book_hotel_runnable))
        builder.add_node(
            "book_hotel_safe_tools",
            create_tool_node_with_fallback(book_hotel_safe_tools),
        )
        builder.add_node(
            "book_hotel_sensitive_tools",
            create_tool_node_with_fallback(book_hotel_sensitive_tools),
        )

        ## Edges
        builder.add_edge("enter_book_hotel", "book_hotel")
        builder.add_conditional_edges(
            "book_hotel",
            route_book_hotel,
            ["leave_skill", "book_hotel_safe_tools", "book_hotel_sensitive_tools", END],
        )
        builder.add_edge("book_hotel_sensitive_tools", "book_hotel")
        builder.add_edge("book_hotel_safe_tools", "book_hotel")

        # Excursion assistant
        ## Nodes
        builder.add_node(
            "enter_book_excursion",
            create_entry_node("Trip Recommendation Assistant", "book_excursion"),
        )
        builder.add_node("book_excursion", Assistant(book_excursion_runnable))
        builder.add_node(
            "book_excursion_safe_tools",
            create_tool_node_with_fallback(book_excursion_safe_tools),
        )
        builder.add_node(
            "book_excursion_sensitive_tools",
            create_tool_node_with_fallback(book_excursion_sensitive_tools),
        )

        ## Edges
        builder.add_edge("enter_book_excursion", "book_excursion")
        builder.add_conditional_edges(
            "book_excursion",
            route_book_excursion,
            ["book_excursion_safe_tools", "book_excursion_sensitive_tools", "leave_skill", END],
        )
        builder.add_edge("book_excursion_sensitive_tools", "book_excursion")
        builder.add_edge("book_excursion_safe_tools", "book_excursion")


        # This node will be shared for exiting all specialized assistants
        builder.add_node("leave_skill", pop_dialog_state)
        builder.add_edge("leave_skill", "primary_assistant")

        # Primary assistant
        ## Nodes
        builder.add_node("primary_assistant", Assistant(assistant_runnable))
        builder.add_node(
            "primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools)
        )

        ## Edges
        builder.add_conditional_edges("fetch_user_info", self.route_to_workflow)
        # The assistant can route to one of the delegated assistants,
        # directly use a tool, or directly respond to the user
        builder.add_conditional_edges(
            "primary_assistant",
            route_primary_assistant,
            [
                "enter_update_flight",
                "enter_book_car_rental",
                "enter_book_hotel",
                "enter_book_excursion",
                "primary_assistant_tools",
                END,
            ],
        )
        builder.add_edge("primary_assistant_tools", "primary_assistant")

        self.graph = builder.compile(
            checkpointer=memory,
            # Let the user approve or deny the use of sensitive tools
            interrupt_before=[
                "update_flight_sensitive_tools",
                "book_car_rental_sensitive_tools",
                "book_hotel_sensitive_tools",
                "book_excursion_sensitive_tools",
            ],
        )

    def user_info(self, state: State):
        return {"user_info": fetch_user_flight_information.invoke({})}
        
    # Each delegated workflow can directly respond to the user
    # When the user responds, we want to return to the currently active workflow
    def route_to_workflow(self, state: State) -> Literal[
        "primary_assistant",
        "update_flight",
        "book_car_rental",
        "book_hotel",
        "book_excursion",
    ]:
        """If we are in a delegated state, route directly to the appropriate assistant."""
        dialog_state = state.get("dialog_state")
        if not dialog_state:
            return "primary_assistant"
        return dialog_state[-1]

