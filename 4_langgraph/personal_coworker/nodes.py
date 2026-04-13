from langchain_core.messages import SystemMessage
from typing import Any, Dict

from models.state import State
from prompts import get_worker_system_message,get_evaluator_message,get_questioner_message

class Nodes:
    def __init__(self):
        self.evaluator_llm_with_output = None
        self.questioner_llm_with_output = None
        self.planer_llm_with_output = None
        self.llm_with_tools = None
        # self.webtools_worker_llm = None
        # self.misctools_worker_llm = None
        self.model = "gpt-4o-mini"

    def questioner_node(self, state: State) -> State:
        questioner_messages = get_questioner_message(state)
        response = self.questioner_llm_with_output.invoke(questioner_messages)
        
        message = ""
        for quest_out_item in response.question_output:
            message += f'Clarifying question provided:{quest_out_item.question}\n'
            message += f'Reasoning of the last question provided:{quest_out_item.reason}\n'

        new_state = {
            "messages": [
                {
                    "role": "assistant",
                    "content": message
                }
            ],
        }

        return new_state

    def planer_node(self, state:State) -> Dict[str, Any]:
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
        response = self.planer_llm_with_output.invoke(messages)

        message = ""
        for i,step in enumerate(response.steps):
            message += f'Step proposed #{i+1}: ' + step.step + '\n'

        # Return updated state
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": message
                }
            ],
        }
        

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

    def evaluator_node(self, state: State) -> State:
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