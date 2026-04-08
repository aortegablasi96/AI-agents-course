import os
from pydantic import BaseModel, Field
from agents import Agent, function_tool
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

model = OpenAIChatCompletionsModel(
    model="llama3.2:1b",
    openai_client=client
)

INSTRUCTIONS = """You are an evaluator of the answers given to a query's serch. 
                The answer will consist in a summary detailing the search results. For the research done,
                check if it is certain and accurate."""

class EvalItem(BaseModel):
    search: str = Field(description="The search result")
    eval_answer: bool = Field(description="The output of the evaluation")

class EvaluationPlan(BaseModel):
    searches_and_evals: list[EvalItem] = Field(description="A list of search results and their evaluations")

evaluator_agent = Agent(
    name="EvaluatorAgent",
    instructions=INSTRUCTIONS,
    model=model,
    output_type=EvalItem,
)
