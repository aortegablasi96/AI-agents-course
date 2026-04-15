from models.state import State
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from datetime import datetime

def get_questioner_system_message(state: State)-> str:
    system_message = f"You are a helpful research assistant. Given a query, come up with 3 clarifying \
            statements or questions that provide more context to the query. They will be used afterwards to support the development of the query.\n"

    return system_message

def get_questioner_user_message(state: State):
    return f"This is the query: {state.messages[0]}" 

def get_questioner_message(state: State):
    return [SystemMessage(content=get_questioner_system_message(state)), HumanMessage(content=get_questioner_user_message(state))]

def get_planner_system_message(state:State) -> str:

    system_message = f"""You are a helpful planner that is given a tasks to complete.
Given the task, you need to come up with a list of basic steps to achieve it. This list will be then taken by the next agent which will execute
each task with the use of tools. You have 3 agents, one named 'webworker' equiped with tools which require interaction with internet (by searching on it using several tools),
another named 'miscworker' equiped with tools to use python , interact with local files and send pushover notifications, and another agent, named 'trelloworker', which is in charge
of checking the user tasks. Please, for each step to fulfill the task, decide which worker will be used.

You also have the historic data of messages. The user may ask again information that you have and, in that case, you'll rather take it from the historic data and assume the step as done rather than 
having to use the agents to redo it. This is the histoy of messages:
{format_conversation(state)}

This is the success criteria:
{state.success_criteria}

You should reply either with a question for the user about this assignment, or with your final response.
If you have a question for the user, you need to reply by clearly stating your question. An example might be:

Question please clarify whether you want a summary or a detailed answer

If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer.
"""

    if state.feedback_on_work:
        system_message += f"""Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
    Here is the feedback on why this was rejected:
    {state.feedback_on_work}
    With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user."""


    return system_message

def get_webworker_system_message() -> str:
       
    system_message = f"""You are a helpful assistant that can use tools to complete tasks.
    The 'planner' agent already splitted the tasks into simple steps. Each time you're called, you'll be given one of these steps.
You have many tools to help you completing the step, including tools to browse the internet, navigating and retrieving web pages or other web sites.
When using the tool to extract the hyperlinks you need to give them the output of the search that has been done previously (will be added as an assistant message this llm call).
The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
Regardless of which task do you finish, answer with: "Successfully finished!"
"""

    return system_message

def get_miscworker_system_message() -> str:
       
    system_message = f"""You are a helpful assistant that can use tools to complete tasks.
    The 'planner' agent already splitted the tasks into simple steps. Each time you're called, you'll be given one of these steps.
You have many tools to help you completing the step, including tools to run python code, to manage local file system and to send push notifications.
Regarding the tool for using python, note that you would need to include a print() statement if you wanted to receive output.
As an alternative, you can also format text. without using any tool.
The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Regardless of which task do you finish, answer with: "Successfully finished!"
"""

    return system_message

def get_trello_worker_system_message(trello_board: str) -> str:
    system_message = f"""You are a helpful trello assistant that can use tools to complete tasks.
The 'planner' agent already splitted the tasks into simple steps. Each time you're called, you'll be given one of these steps.
If they call you is because they want to know information about their Trello board. Please answer their requsts using the board information.

This is the board you will be requested about:
{trello_board}
"""

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
    {format_conversation(state)}

    The success criteria for this assignment is:
    {state.success_criteria}

    Review all the steps done and respond with your feedback, and decide if the success criteria is met by this response.
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

def format_conversation(state) -> str:
    conversation = "Conversation history:\n\n"
    for message in state.messages:
        if isinstance(message, HumanMessage):
            conversation += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            text = message.content or "[Tools use]"
            conversation += f"Assistant: {text}\n"
    return conversation