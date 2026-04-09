from pydantic import BaseModel, Field
from agents import Agent

HOW_MANY_SEARCHES = 2

INSTRUCTIONS = f"You are a helpful research assistant. You'll be given a query with a set of clarifying questions to expand \
                the query's context and to understand it better. Given the query, come up with a set of web searches \
                to perform to best answer the query. Refine them by using the clarifying questions and their reasoning. \
                Output {HOW_MANY_SEARCHES} terms to query for."


class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")


planner_agent = Agent(
    name="PlannerAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=WebSearchPlan,
)