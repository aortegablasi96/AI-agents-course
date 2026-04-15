from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from trello import TrelloClient
from langgraph.checkpoint.memory import MemorySaver
from typing import Any
from PIL import Image
import io
import uuid
import asyncio
import os
from dotenv import load_dotenv
import sqlite3
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


from models.state import State
from models.questioner import QuestionerOutput
from models.planner import PlannerOutput
from models.evaluator import EvaluatorOutput
from tools.web_tools import WebTools
from tools.misc_tools import MiscTools
from nodes import Nodes

load_dotenv(override=True)

class Graph:
    def __init__(self) -> None:
        self.thread_id = 632
        self.memory = None     
        self.nodes = Nodes()
        self.webtools = WebTools()
        self.misctools = MiscTools()
        self.graph = None
        self.browser = None
        self.playwright = None
        self.playwright_tool  = None
        # self.tool_list = None
        self.web_tools_list = None
        self.misc_tools_list = None
    
    async def setup(self):
        self.playwright_tool, self.browser, self.playwright = await self.webtools.playwright_tools()
        # self.tool_list = self.playwright_tool + self.misctools.file_tool() + [self.webtools.search_tool(),self.webtools.wikipedia_tool(),self.webtools.youtube_tool,self.webtools.yahoofinance_tool,self.misctools.pushover_tool(),self.misctools.python_tool()]
        self.web_tools_list = self.playwright_tool + [self.webtools.search_tool(),self.webtools.wikipedia_tool(),self.webtools.youtube_tool,self.webtools.yahoofinance_tool]
        self.misc_tools_list =  self.misctools.file_tool() + [self.misctools.pushover_tool(),self.misctools.python_tool()]

        conn = await aiosqlite.connect("memory.db")
        self.memory = AsyncSqliteSaver(conn)

        api_key = os.getenv("TRELLO_API_KEY")
        token = os.getenv("TRELLO_TOKEN")

        client = TrelloClient(api_key=api_key, token=token)

        boards = client.get_board(board_id="69c25923c234761cc0711f1a")

        board_cards_text = ""
        for card in boards.get_cards():
            board_cards_text+= f'Name: {card.name}\n'
            board_cards_text+= f'Url: {card.url}\n'
            board_cards_text+= f'Category: {card.get_list().name}\n'
            board_cards_text+= f'Description: {card.description}\n'
            board_cards_text+= f'Due date: {card.due_date}\n'
            board_cards_text+= f'Labels: {[label.name for label in card.labels]}\n\n'
        
        self.nodes.trello_text = board_cards_text

        questioner_llm = ChatOpenAI(model=self.nodes.model)
        self.nodes.questioner_llm_with_output = questioner_llm.with_structured_output(QuestionerOutput)
        planner_llm = ChatOpenAI(model=self.nodes.model)
        self.nodes.planner_llm_with_output = planner_llm.with_structured_output(PlannerOutput)
        webtools_worker_llm = ChatOpenAI(model=self.nodes.model)
        self.nodes.webtools_worker_llm = webtools_worker_llm.bind_tools(self.web_tools_list)
        misctools_worker_llm = ChatOpenAI(model=self.nodes.model)
        self.nodes.misctools_worker_llm = misctools_worker_llm.bind_tools(self.misc_tools_list)
        trello_llm = ChatOpenAI(model=self.nodes.model)
        self.nodes.trello_llm = trello_llm
        evaluator_llm = ChatOpenAI(model=self.nodes.model)
        self.nodes.evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)
        await self.build_graph()

    async def build_graph(self):
        # Set up Graph Builder with State
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("questioner", self.nodes.questioner_node)
        graph_builder.add_node("planner",self.nodes.planner_node)
        graph_builder.add_node("webworker", self.nodes.web_worker_node)
        graph_builder.add_node("miscworker",self.nodes.misc_worker_node)
        graph_builder.add_node("trelloworker",self.nodes.trello_worker_node)
        graph_builder.add_node("webtools", ToolNode(tools=self.web_tools_list))
        graph_builder.add_node("misctools", ToolNode(tools=self.misc_tools_list))
        graph_builder.add_node("evaluator", self.nodes.evaluator_node)

        # Add edges
        graph_builder.add_edge(START,"questioner")
        graph_builder.add_edge("questioner","planner")
        graph_builder.add_conditional_edges("planner",self.workers_router,{"webworker":"webworker","miscworker":"miscworker","trelloworker":"trelloworker","evaluator":"evaluator"})
        graph_builder.add_conditional_edges("webworker", self.worker_router, {"tools": "webtools", "webworker":"webworker","miscworker":"miscworker","trelloworker":"trelloworker","evaluator":"evaluator"})
        graph_builder.add_conditional_edges("miscworker", self.worker_router, {"tools": "misctools", "webworker":"webworker","miscworker":"miscworker","trelloworker":"trelloworker","evaluator":"evaluator"})
        graph_builder.add_conditional_edges("webtools",self.workers_router,{"webworker":"webworker","miscworker":"miscworker","trelloworker":"trelloworker","evaluator":"evaluator"})
        graph_builder.add_conditional_edges("misctools",self.workers_router,{"webworker":"webworker","miscworker":"miscworker","trelloworker":"trelloworker","evaluator":"evaluator"})
        graph_builder.add_conditional_edges("trelloworker",self.workers_router,{"webworker":"webworker","miscworker":"miscworker","trelloworker":"trelloworker","evaluator":"evaluator"})
        graph_builder.add_conditional_edges("evaluator", self.route_based_on_evaluation, {"planner": "planner", "END": END})

        # Compile the graph
        self.graph = graph_builder.compile(checkpointer=self.memory)

    def workers_router(self, state:State) -> str:
        if state.worker_steps:
            return state.worker_steps[0].worker
        else:
            return "evaluator"

    def worker_router(self, state: State) -> str:
        last_message = state.messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        else:
            if state.worker_steps:
                return state.worker_steps[0].worker
            else:
                return "evaluator"

    def route_based_on_evaluation(self, state: State) -> str:
        if state.success_criteria_met or state.user_input_needed:
            return "END"
        else:
            return "planner"

    async def run_superstep(self, message, success_criteria, history):
        """Run one conversation turn: user message -> agent response -> evaluator feedback."""
        config = {"configurable": {"thread_id": self.thread_id}}

        state = {
            "messages": message,
            "worker_steps": None,
            "is_first_step": True,
            "first_step_quest_num": 0,
            "success_criteria": success_criteria or "The answer should be clear and accurate",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
        }

        result = await self.graph.ainvoke(state, config=config)

        user = {"role": "user", "content": message}
        steps = {"role": "assistant", "content": result["messages"][result["first_step_quest_num"]].content}
        reply = {"role": "assistant", "content": result["messages"][-2].content} 
        feedback = {"role": "assistant", "content": result["messages"][-1].content}
        return history + [user, steps, reply, feedback]

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