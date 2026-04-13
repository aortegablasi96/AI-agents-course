from pydantic import BaseModel, Field

class PlanerItem(BaseModel):
    step: str = Field(description="Single action step to achieve the query")

class PlanerOutput(BaseModel):
    steps: list[PlanerItem] = Field(description="List of action step to achieve the query") 