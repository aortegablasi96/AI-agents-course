from pydantic import BaseModel, Field

class QuestionerItem(BaseModel):
    question: str = Field(description="Question to that give reasoning to the query")
    reason: str = Field(description="Reasoning of proposing that claryfing question")

class QuestionerOutput(BaseModel):
    question_output: list[QuestionerItem] = Field(description="List of questions to get clarification to the query") 