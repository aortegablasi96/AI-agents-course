import os
from pydantic import BaseModel, Field
from agents import Agent


INSTRUCTIONS = """You are an evaluator of the answers given to a query's serch. 
                The answer will consist in a summary detailing the search results. For the research done,
                check if it is certain and accurate."""


class EvalItem(BaseModel):
    eval_answer: bool = Field(description="The output of the evaluation")


evaluator_agent = Agent(
    name="EvaluatorAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=EvalItem,
)
