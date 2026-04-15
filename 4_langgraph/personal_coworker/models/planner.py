from pydantic import BaseModel, Field
from typing import Literal, Optional

class PlannerItem(BaseModel):
    step: str = Field(description="Single action step to achieve the query")
    worker: Literal["webworker","miscworker","trelloworker"] = Field(description="Worker who is in charge of executing the work")

class PlannerOutput(BaseModel):
    steps: Optional[list[PlannerItem]] = Field(description="List of action step to achieve the query")
    question_for_user: Optional[str] = Field(description="Question to be answered by the user")