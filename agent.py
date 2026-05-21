import os
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from tools import TOOL_KIT

load_dotenv()


# =============================================================================
# 1. Define the Graph State Schema
# =============================================================================
class AgentState(TypedDict):
    """State schema for the Energy Advisor agent graph.

    Attributes:
        messages: List of messages in the conversation
        question: The user's question
        context: Optional context/location info
        tool_calls: Track which tools have been called
    """
    messages: Annotated[List[BaseMessage], "List of conversation messages"]
    question: str
    context: str | None
    tool_calls: List[str]


# =============================================================================
# 2. Define Graph Nodes
# =============================================================================
def create_nodes(llm):
    """Create graph nodes with the LLM bound to tools."""

    # Node: router - decides whether to use tools or respond
    def router_node(state: AgentState) -> AgentState:
        """Route the request: either call tools or generate final response."""
        messages = state["messages"]

        # LLM decides: tool call or final response
        response = llm.bind_tools(TOOL_KIT).invoke(messages)

        # Append the response
        new_messages = messages + [response]

        # Check if the LLM wants to call tools
        if hasattr(response, "tool_calls") and response.tool_calls:
            return {
                **state,
                "messages": new_messages,
                "tool_calls": [tc["name"] for tc in response.tool_calls]
            }
        else:
            # No tool calls - this is the final response
            return {
                **state,
                "messages": new_messages,
                "tool_calls": []
            }

    # Node: tool_executor - executes tool calls
    tool_node = ToolNode(TOOL_KIT)

    # Node: respond_final - generates final response after tool results
    def respond_final_node(state: AgentState) -> AgentState:
        """Generate final response after tools have been executed."""
        messages = state["messages"]

        # Get only the last message (which should be a tool result summary)
        # The LLM will generate a natural response based on tool outputs
        response = llm.invoke(messages)

        return {
            **state,
            "messages": messages + [response]
        }

    return {
        "router": router_node,
        "tools": tool_node,
        "respond_final": respond_final_node
    }


# =============================================================================
# 3. Define Graph Edges (Flow Logic)
# =============================================================================
def should_continue(state: AgentState) -> str:
    """Decide whether to continue to tools or end.

    Returns:
        "tools" if there are tool calls to execute
        "respond_final" if no more tool calls (or after tools run)
    """
    if state.get("tool_calls"):
        return "tools"
    return "respond_final"


# =============================================================================
# 4. Build the Graph
# =============================================================================
def create_graph(llm):
    """Create the Energy Advisor graph with explicit nodes and edges."""
    nodes = create_nodes(llm)

    # Define the state graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", nodes["router"])
    graph.add_node("tools", nodes["tools"])
    graph.add_node("respond_final", nodes["respond_final"])

    # Set entry point
    graph.set_entry_point("router")

    # Add conditional edge: router -> (tools OR respond_final)
    graph.add_conditional_edges(
        "router",
        should_continue,
        {
            "tools": "tools",
            "respond_final": "respond_final"
        }
    )

    # After tools, loop back to router to check if more tools needed
    graph.add_edge("tools", "router")

    # End after final response
    graph.add_edge("respond_final", END)

    return graph.compile()


# =============================================================================
# 5. Agent Class (Public Interface)
# =============================================================================
class Agent:
    def __init__(self, instructions: str, model: str = "gpt-4o-mini"):
        """Initialize the Energy Advisor agent.

        Args:
            instructions (str): System prompt/instructions for the agent
            model (str): LLM model to use (default: gpt-4o-mini)
        """
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            base_url="https://openai.vocareum.com/v1",
            api_key=os.getenv("VOCAREUM_API_KEY")
        )

        # Create system message from instructions
        self.system_message = SystemMessage(content=instructions)

        # Build the explicit graph
        self.graph = create_graph(self.llm)

    def invoke(self, question: str, context: str = None) -> dict:
        """Ask the Energy Advisor a question about energy optimization.

        Args:
            question (str): The user's question about energy optimization
            context (str): Optional context (e.g., location, user preferences)

        Returns:
            dict: Response containing messages and metadata
        """
        # Build initial state
        messages = []

        # Add context if provided (as system message for context)
        if context:
            messages.append(SystemMessage(content=context))

        # Add user question
        messages.append(HumanMessage(content=question))

        initial_state = {
            "messages": messages,
            "question": question,
            "context": context,
            "tool_calls": []
        }

        # Execute the graph
        result = self.graph.invoke(initial_state)

        return result

    def get_agent_tools(self):
        """Get list of available tools for the Energy Advisor.

        Returns:
            list: Names of available tools
        """
        return [t.name for t in TOOL_KIT]

    def get_graph_structure(self):
        """Get the graph structure for visualization/debugging.

        Returns:
            str: Graph structure description
        """
        return """
        Graph Structure:
        ─────────────────────────────────────────────
        Entry ──► [router] ──► ┬──► [tools] ──► [router] (loop)
                          └──► [respond_final] ──► END
        ─────────────────────────────────────────────

        Nodes:
        1. router: Decides whether to call tools or respond
        2. tools: Executes tool calls (Weather, Prices, RAG, etc.)
        3. respond_final: Generates final natural language response

        Edges:
        - router → tools (if tools needed)
        - router → respond_final (if no tools needed)
        - tools → router (loop back for more tool calls)
        - respond_final → END (finish)
        """