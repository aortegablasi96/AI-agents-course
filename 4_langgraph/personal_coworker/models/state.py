"""
State of current messages.
"""

from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from typing import Annotated
from typing import List, Any, Optional

from models.planner import PlannerItem

class State(BaseModel):
    messages: Annotated[List[Any], add_messages] = Field(description = "Current answer to the query")
    worker_steps: Optional[list[PlannerItem]] = Field(description="List of action step to achieve the query")
    is_first_step: bool = Field(description="Wether the worker is on his first step or not")
    first_step_quest_num: Optional[int] = Field(description="To show the right steps message in Gradio")
    success_criteria: str = Field(description = "What a successful answer looks like")
    success_criteria_met: bool = Field(description = "Whether the criteria defined is met or not")
    feedback_on_work: Optional[str] = Field(description = "Feedback on answer to query (if needed)")
    user_input_needed: bool = Field(description = "Whether the agent needs further user feedback to answer the query")