from langchain_core.messages import SystemMessage, AIMessage
from typing import Any, Dict

from models.state import State
from prompts import get_questioner_message, get_planner_system_message, get_webworker_system_message, get_miscworker_system_message, get_trello_worker_system_message, get_evaluator_message 

class Nodes:
    def __init__(self):
        self.evaluator_llm_with_output = None
        self.questioner_llm_with_output = None
        self.planner_llm_with_output = None
        self.trello_llm = None
        self.trello_board = None
        self.trello_text = None
        # self.llm_with_tools = None
        self.webtools_worker_llm = None
        self.misctools_worker_llm = None
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

    def planner_node(self, state:State) -> State:
        system_message = get_planner_system_message(state)

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
        response = self.planner_llm_with_output.invoke(messages)

        if response.steps:
            message = ""
            for i,step in enumerate(response.steps):
                message += f'Step proposed #{i+1}: ' + step.step + f'. Done by {step.worker} worker.\n'

            # Return updated state
            new_state = {
                "worker_steps": response.steps,
                "is_first_step": True,
                "first_step_quest_num": len(state.messages),
                "messages": [
                    {
                        "role": "assistant",
                        "content": message
                    }
                ],
            }
        else: 
            new_state = {
                "messages": [
                    {
                        "role": "assistant",
                        "content": response.question_for_user
                    }
                ],
            }

        return new_state


    def master_node(self, state: State) -> State:   
        return state
        
    def web_worker_node(self, state:State) -> Dict[str, Any]:
        worker_steps = state.worker_steps
        work = worker_steps.pop(0)
        system_message = get_webworker_system_message()

        # Add in the mesasges to query in the llm
        messages = [SystemMessage(content=system_message)]
        
        if not state.is_first_step:
            messages.append(state.messages[-2])
            messages.append(state.messages[-1])

        messages.append(AIMessage(content=work.step))
       
        # Invoke the LLM with tools
        response = self.webtools_worker_llm.invoke(messages)

        # Return updated state
        return {
            "is_first_step": False,
            "messages": [response],
            "worker_steps": worker_steps
        }

    def misc_worker_node(self, state:State) -> Dict[str, Any]:
        worker_steps = state.worker_steps
        work = worker_steps.pop(0)
        system_message = get_miscworker_system_message()

        # Add in the mesasges to query in the llm
        messages = [SystemMessage(content=system_message)]

        if not state.is_first_step:
            messages.append(state.messages[-2])
            messages.append(state.messages[-1])

        messages.append(AIMessage(content=work.step))
        
        # Invoke the LLM with tools
        response = self.misctools_worker_llm.invoke(messages)

        # Return updated state
        return {
            "is_first_step": False,
            "messages": [response],
            "worker_steps": worker_steps
        }

    def trello_worker_node(self, state:State) -> Dict[str, Any]:
        worker_steps = state.worker_steps
        work = worker_steps.pop(0)
        
        system_message = get_trello_worker_system_message(self.trello_text)

        # Add in the mesasges to query in the llm
        messages = [SystemMessage(content=system_message)]

        if not state.is_first_step:
            messages.append(state.messages[-1])
            messages.append(state.messages[-2])

        messages.append(AIMessage(content=work.step))
        
        # Invoke the LLM with tools
        response = self.trello_llm.invoke(messages)

        # Return updated state
        return {
            "is_first_step": False,
            "messages": [response],
            "worker_steps": worker_steps
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