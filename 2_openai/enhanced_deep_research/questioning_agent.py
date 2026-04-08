from pydantic import BaseModel,Field
from agents import Agent
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

HOW_MANY_QUESTIONS = 2

INSTRUCTIONS = f"You are a helpful research assistant. Given a query, come up with a set of {HOW_MANY_QUESTIONS} clarifying questions \
that try to expand the context of the query and give clarification. These questions will be added to the query to then search about the query using another agent."

class QueryQuestionItem(BaseModel):
    question: str = Field(description="A clarifying question to understand better the query.")
    reason: str = Field(description="Your reasoning for why this clarifying question is important to the query. Describe how clarifies the query")
    
class FullQuery(BaseModel):
    clarifying_questions: list[QueryQuestionItem] = Field(description="A list of clarifying questions to perform to best understand the query.")
    query: str = Field(description="The search term to use for the web search.")
    
questioning_agent = Agent(
    name="QuestioningAgent",
    instructions=INSTRUCTIONS,
    model=model,
    output_type=FullQuery,
)