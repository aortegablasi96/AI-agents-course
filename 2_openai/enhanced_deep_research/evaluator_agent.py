import os
from pydantic import BaseModel, Field
from agents import Agent, function_tool


INSTRUCTIONS = """You are an evaluator of the answers given to a query. For each search given, check if the answers are correct and accurate."""

class EvalItem(BaseModel):
    search: str = Field(description="The search result")
    eval_answer: bool = Field(description="The output of the evaluation")

class EvaluationPlan(BaseModel):
    searches_and_evals: list[EvalItem] = Field(description="A list of search results and their evaluations")

evaluator_agent = Agent(
    name="EvaluatorAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=EvaluationPlan,
)
