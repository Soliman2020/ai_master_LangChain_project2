import os
from typing import TypedDict, Annotated, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# MemorySaver enables simple in-memory persistence for graph state.
# If this import is unavailable in the installed version, we fall back to None.
# The agent can still run without memory; it just won't retain conversation state
# across calls when compiled without a checkpointer.
try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    MemorySaver = None

# TOOL_KIT should be a list of valid LangChain tools or @tool-decorated functions.
# These are used in two places:
# 1) llm.bind_tools(...) so the model can decide when to call them.
# 2) ToolNode(...) so LangGraph can execute the tool calls automatically.
from tools import TOOL_KIT

# Load environment variables from .env, including VOCAREUM_API_KEY.
load_dotenv()


class AgentState(TypedDict):
    # The conversation history for the graph.
    # add_messages is critical here: it tells LangGraph to APPEND/MERGE new messages
    # into the existing history instead of overwriting the list on each node update.
    # This is the standard pattern for chat- and tool-based agents.
    messages: Annotated[list[BaseMessage], add_messages]

    # Original user question, stored separately for convenience/debugging.
    question: str

    # Optional extra runtime context, such as location or preferences.
    context: Optional[str]


class Agent:
    def __init__(
        self,
        instructions: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        enable_memory: bool = False,
    ):
        """
        Energy Advisor agent built with LangGraph.

        Args:
            instructions: System prompt for the agent.
            model: Chat model name.
            temperature: Model temperature.
            enable_memory: If True, compile with in-memory checkpointing.
        """
        # Save the system instructions so they can be injected into each new run.
        self.instructions = instructions

        # Read API key from environment.
        # Failing early here makes setup issues much easier to debug.
        api_key = os.getenv("VOCAREUM_API_KEY")
        if not api_key:
            raise ValueError("VOCAREUM_API_KEY is not set in environment variables.")

        # Main chat model used by the agent.
        # base_url points to the Vocareum-compatible OpenAI endpoint used in your environment.
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url="https://openai.vocareum.com/v1",
            api_key=api_key,
        )

        # Bind tools to the model once during initialization.
        # This allows the model to return tool_calls when it decides a tool is needed.
        self.llm_with_tools = self.llm.bind_tools(TOOL_KIT)

        # Build and compile the LangGraph workflow.
        self.graph = self._build_graph(enable_memory=enable_memory)

    def _chatbot_node(self, state: AgentState):
        """
        Core LLM node.

        This node receives the full current message history from the graph state,
        asks the model for the next step, and returns exactly one AI message.

        If the model wants to call a tool, that AI message will include tool_calls.
        If not, it will contain a normal final answer.

        Because AgentState.messages uses add_messages, returning:
            {"messages": [response]}
        will append this new AI message to the existing conversation history.
        """
        response = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def _build_graph(self, enable_memory: bool = False):
        """
        Build a standard LangGraph tool-calling loop:

            chatbot -> tools? -> chatbot -> ... -> END

        Flow:
        1) chatbot node runs the model.
        2) tools_condition checks whether the last AI message contains tool calls.
        3) If yes, route to ToolNode to execute them.
        4) After tools finish, return to chatbot with tool results in state.
        5) If no tool calls remain, end the graph.
        """
        graph = StateGraph(AgentState)

        # Main reasoning node: model decides whether to answer directly or call tools.
        graph.add_node("chatbot", self._chatbot_node)

        # Tool execution node: runs any tool calls found in the last AI message.
        graph.add_node("tools", ToolNode(TOOL_KIT))

        # Start every run at the chatbot node.
        graph.set_entry_point("chatbot")

        # Conditional routing:
        # - "tools" means the model requested one or more tool calls.
        # - "__end__" means the model returned a normal answer and the run can finish.
        graph.add_conditional_edges(
            "chatbot",
            tools_condition,
            {
                "tools": "tools",
                "__end__": END,
            },
        )

        # After executing tools, go back to the chatbot so the model can:
        # - inspect tool outputs,
        # - decide whether more tools are needed,
        # - or produce the final answer.
        graph.add_edge("tools", "chatbot")

        # Optional memory:
        # If enabled and available, compile the graph with an in-memory checkpointer.
        # This allows state persistence across invocations when using the same thread_id.
        if enable_memory and MemorySaver is not None:
            memory = MemorySaver()
            return graph.compile(checkpointer=memory)

        # Default: compile without persistence.
        return graph.compile()

    def invoke(
        self,
        question: str,
        context: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> dict:
        """
        Ask the agent a question.

        Args:
            question: User query.
            context: Optional runtime context such as location.
            thread_id: Optional thread id for memory when enable_memory=True.

        Returns:
            Final graph state.
        """
        # Start a fresh message list for this invocation.
        # We inject the agent instructions as the first system message so the model
        # always sees its role and behavior guidelines.
        messages: list[BaseMessage] = [SystemMessage(content=self.instructions)]

        # Add optional runtime context as another system message.
        # This is useful for things like location, house profile, or user preferences.
        if context:
            messages.append(
                SystemMessage(
                    content=f"Additional context for this request: {context}"
                )
            )

        # Add the user's actual question.
        messages.append(HumanMessage(content=question))

        # Initial graph state.
        # question/context are stored in case you want to inspect or extend the state later.
        initial_state: AgentState = {
            "messages": messages,
            "question": question,
            "context": context,
        }

        # thread_id is only useful when the graph was compiled with memory enabled.
        # LangGraph uses it to keep separate conversation threads/checkpoints.
        config = None
        if thread_id:
            config = {"configurable": {"thread_id": thread_id}}

        # Run the graph and return the full resulting state.
        return self.graph.invoke(initial_state, config=config)

    def invoke_text(
        self,
        question: str,
        context: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> str:
        """
        Convenience method that returns only the final assistant text.

        This is helpful when the caller does not need the full graph state and only
        wants the human-readable final response.
        """
        result = self.invoke(question=question, context=context, thread_id=thread_id)
        messages = result.get("messages", [])

        # Walk backward through the message history to find the most recent AI message
        # that contains usable content.
        for message in reversed(messages):
            if isinstance(message, AIMessage) and getattr(message, "content", None):
                # Most providers return plain string content.
                if isinstance(message.content, str):
                    return message.content

                # Some providers may return structured content blocks.
                # In that case, extract any text blocks and join them.
                if isinstance(message.content, list):
                    parts = []
                    for item in message.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
                    return "\n".join(p for p in parts if p).strip()

        # Fallback if no assistant text could be found.
        return ""

    def get_agent_tools(self) -> list[str]:
        """
        Return names of registered tools.
        """
        names = []
        for tool in TOOL_KIT:
            # Some tool objects expose .name directly; otherwise use the class name.
            name = getattr(tool, "name", None)
            names.append(name or tool.__class__.__name__)
        return names

    def get_graph_structure(self) -> str:
        # A simple text diagram for learning, debugging, or explaining the workflow.
        return """
Graph Structure:
─────────────────────────────────────────────
Entry ──► [chatbot] ──► ┬──► [tools] ──► [chatbot] (loop)
                        └──► END
─────────────────────────────────────────────

Nodes:
1. chatbot: LLM decides whether to answer or call tools
2. tools: Executes tool calls using ToolNode

Edges:
- chatbot → tools (when tool calls are present)
- chatbot → END (when no tool calls remain)
- tools → chatbot (loop back after tool execution)
"""

    def debug_last_messages(
        self,
        result: dict,
        limit: int = 6,
    ) -> None:
        """
        Optional helper for debugging message history.

        This prints the last few messages, their types, tool calls, and content.
        Very useful when debugging tool routing or malformed message sequences.
        """
        messages = result.get("messages", [])
        tail = messages[-limit:]

        print(f"\n[Debug] Showing last {len(tail)} messages")
        for i, msg in enumerate(tail, start=max(0, len(messages) - len(tail))):
            msg_type = msg.__class__.__name__
            tool_calls = getattr(msg, "tool_calls", None)

            print(f"[{i}] {msg_type}")

            # If the message triggered tools, print the tool call metadata.
            if tool_calls:
                print(f"     tool_calls={tool_calls}")

            content = getattr(msg, "content", "")

            # Truncate long string content for cleaner console output.
            if isinstance(content, str):
                print(f"     content={content[:300]}")
            else:
                # Non-string content may be a structured list of content blocks.
                print(f"     content={content}")