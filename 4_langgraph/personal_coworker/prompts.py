from models.state import State
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import List, Any, Optional, Dict
from datetime import datetime

def get_worker_system_message(state:State) -> str:
    system_message = f"""You are a helpful assistant that can use tools to complete tasks.
You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
You have many tools to help you, including tools to browse the internet, navigating and retrieving web pages.
You have a tool to run python code, but note that you would need to include a print() statement if you wanted to receive output.
The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This is the success criteria:
{state.success_criteria}
You should reply either with a question for the user about this assignment, or with your final response.
If you have a question for the user, you need to reply by clearly stating your question. An example might be:

Question: please clarify whether you want a summary or a detailed answer

If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer.
"""

    if state.feedback_on_work:
        system_message += f"""Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
    Here is the feedback on why this was rejected:
    {state.feedback_on_work}
    With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user."""


    return system_message

def get_evaluator_system_message() -> str:
    system_message = """You are an evaluator that determines if a task has been completed successfully by an Assistant.
Assess the Assistant's last response based on the given criteria. Respond with your feedback, and with your decision on 
whether the success criteria has been met, and whether more input is needed from the user."""

    return system_message


def get_evaluator_user_message(state:State) -> str:
    last_response = state.messages[-1]

    user_message = user_message = f"""You are evaluating a conversation between the User and Assistant. You decide what action to take based on the last response from the Assistant.

    The entire conversation with the assistant, with the user's original request and all replies, is:
    {format_conversation(state.messages)}

    The success criteria for this assignment is:
    {state.success_criteria}

    And the final response from the Assistant that you are evaluating is:
    {last_response}

    Respond with your feedback, and decide if the success criteria is met by this response.
    Also, decide if more user input is required, either because the assistant has a question, needs clarification, or seems to be stuck and unable to answer without help.

    The Assistant has access to a tool to write files. If the Assistant says they have written a file, then you can assume they have done so.
    Overall you should give the Assistant the benefit of the doubt if they say they've done something. But you should reject if you feel that more work should go into this.

    """
    if state.feedback_on_work:
        user_message += f"Also, note that in a prior attempt from the Assistant, you provided this feedback: {state.feedback_on_work}\n"
        user_message += "If you're seeing the Assistant repeating the same mistakes, then consider responding that user input is required."

    return user_message


def get_evaluator_message(state:State) -> [SystemMessage,HumanMessage]:
    evaluator_messages = [
            SystemMessage(content=get_evaluator_system_message()),
            HumanMessage(content=get_evaluator_user_message(state)),
        ]
    return evaluator_messages



def format_conversation(messages: List[Any]) -> str:
    conversation = "Conversation history:\n\n"
    for message in messages:
        if isinstance(message, HumanMessage):
            conversation += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            text = message.content or "[Tools use]"
            conversation += f"Assistant: {text}\n"
    return conversation