from pydantic import BaseModel,Field
from agents import Agent

HOW_MANY_QUESTIONS = 3

INSTRUCTIONS = f"You are a helpful research assistant. Given a query come up with a set of {HOW_MANY_QUESTIONS} clarifying questions \
to understand better query. These questions will be added to the query to then search about them using another agent."

class QueryQuestionItem(BaseModel):
    question: str = Field(description="A clarifying question to understand better the query.")
    

class FullQuery(BaseModel):
    clarifying_questions: list[QueryQuestionItem] = Field(description="A list of clarifying questions to perform to best understand the query.")
    query: str = Field(description="The search term to use for the web search.")

    
questioning_agent = Agent(
    name="QuestioningAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=FullQuery,
)