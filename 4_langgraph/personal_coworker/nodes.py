from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import List, Any, Optional, Dict

from tools import Tools
from models.state import State
from prompts import get_worker_system_message,get_evaluator_message

class Nodes:
    def __init__(self):
        self.tools_object = Tools()
        self.tools = None
        self.llm_with_tools = None
        self.evaluator_llm_with_output = None
        self.model = "gpt-4o-mini"
        self.llm = ChatOpenAI(model=self.model)

    def worker_node(self, state:State) -> Dict[str, Any]:
        system_message = get_worker_system_message(state)

         # Add in the system message
        found_system_message = False
        messages = state.messages
        for message in messages:
            if isinstance(message, SystemMessage):
                message.content = system_message
                found_system_message = True

        if not found_system_message:
            messages = [SystemMessage(content=system_message)] + messages

        # Invoke the LLM with tools
        response = self.llm_with_tools.invoke(messages)

        # Return updated state
        return {
            "messages": [response],
        }

    def evaluator_node(self, state:State) -> State:
        evaluator_messages = get_evaluator_message(state)

        eval_result = self.evaluator_llm_with_output.invoke(evaluator_messages)
        new_state = {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Evaluator Feedback on this answer: {eval_result.feedback}",
                }
            ],
            "feedback_on_work": eval_result.feedback,
            "success_criteria_met": eval_result.success_criteria_met,
            "user_input_needed": eval_result.user_input_needed,
        }
        return new_state

            
    def questioner_node(self, state: State) -> State:
        query = state.messages[-1].content

        system_message = f"You are a helpful research assistant. Given a query, come up with a set of 2 \
    clarifying questions that try to expand the context of the query and give clarification. These questions will be added \
    to the query to then search about the query using another agent."

        user_message = f"The query given is the following:\n{query}"

        questioner_messages = [SystemMessage(content=system_message), HumanMessage(content=user_message)]

        response = self.llm.invoke(questioner_messages)
        new_state = {
            "messages": response
        }
        return new_state