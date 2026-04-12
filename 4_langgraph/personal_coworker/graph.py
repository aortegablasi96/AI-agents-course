from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from typing import Any
from PIL import Image
import io
import uuid
import asyncio


from models.state import State
from models.questioner import QuestionerOutput
from models.evaluator import EvaluatorOutput
from tools import Tools
from nodes import Nodes


class Graph:
    def __init__(self) -> None:
        self.thread_id = str(uuid.uuid4())
        self.memory = MemorySaver()
        self.nodes = Nodes()
        self.tools = Tools()
        self.graph = None
        self.browser = None
        self.playwright = None
        self.playwright_tool  = None
        self.tools_list = None

    async def setup(self):
        self.playwright_tool, self.browser, self.playwright = await self.tools.playwright_tools()
        self.tools_list = self.playwright_tool + self.tools.file_tool() + [self.tools.search_tool(),self.tools.pushover_tool(),self.tools.wikipedia_tool(),self.tools.python_tool()]

        questioner_llm = ChatOpenAI(model=self.nodes.model)
        self.nodes.questioner_llm_with_output = questioner_llm.with_structured_output(QuestionerOutput)
        worker_llm = ChatOpenAI(model=self.nodes.model)
        self.nodes.llm_with_tools = worker_llm.bind_tools(self.tools_list)
        evaluator_llm = ChatOpenAI(model=self.nodes.model)
        self.nodes.evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)
        await self.build_graph()

    async def build_graph(self):
        # Set up Graph Builder with State
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("questioner", self.nodes.questioner_node)
        graph_builder.add_node("worker", self.nodes.worker_node)
        graph_builder.add_node("tools", ToolNode(tools=self.tools_list))
        graph_builder.add_node("evaluator", self.nodes.evaluator_node)

        # Add edges
        graph_builder.add_edge(START,"questioner")
        graph_builder.add_edge("questioner","worker")
        graph_builder.add_conditional_edges(
            "worker", self.worker_router, {"tools": "tools", "evaluator": "evaluator"}
        )
        graph_builder.add_edge("tools", "worker")
        graph_builder.add_conditional_edges(
            "evaluator", self.route_based_on_evaluation, {"worker": "worker", "END": END}
        )

        # Compile the graph
        self.graph = graph_builder.compile(checkpointer=self.memory)

    def worker_router(self, state: State) -> str:
        last_message = state.messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        else:
            return "evaluator"

    def route_based_on_evaluation(self, state: State) -> str:
        if state.success_criteria_met or state.user_input_needed:
            return "END"
        else:
            return "worker"

    async def run_superstep(self, message, success_criteria, history):
        """Run one conversation turn: user message -> agent response -> evaluator feedback."""
        config = {"configurable": {"thread_id": self.thread_id}}

        state = {
            "messages": message,
            "success_criteria": success_criteria or "The answer should be clear and accurate",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
        }

        result = await self.graph.ainvoke(state, config=config)

        user = {"role": "user", "content": message}
        reply = {"role": "assistant", "content": result["messages"][-2].content}
        feedback = {"role": "assistant", "content": result["messages"][-1].content}
        return history + [user, reply, feedback]

    def get_nodes_diagram(self):
        """ Create an image of the Graph diagram."""
        return Image.open(io.BytesIO(self.graph.get_graph().draw_mermaid_png()))

    def cleanup(self):
            if self.browser:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.browser.close())
                    if self.playwright:
                        loop.create_task(self.playwright.stop())
                except RuntimeError:
                    # If no loop is running, do a direct run
                    asyncio.run(self.browser.close())
                    if self.playwright:
                        asyncio.run(self.playwright.stop())