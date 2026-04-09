from agents import Runner, ModelSettings
from search_agent import search_agent
from planner_agent import planner_agent
from writer_agent import writer_agent
from email_agent import email_agent
from questioning_agent import questioning_agent
from evaluator_agent import evaluator_agent
from email_manager_agent import email_manager_agent
from agents import Agent


INSTRUCTIONS = (
    "You are a research assistant manager. The user will provide you with a query and you are required to research about it "
    "and then provide an email summary with your research to the users. You have several agent-as-tools to accomplish your task.\n"
    "You must always follow the following steps:\n"
    "1. You must start by adding clarifying questions to understand better the query, using the 'questioning_tool'.\n"
    "2. You have to use the query and the clarifying questions, as a whole and single summary, to plan searches, using the 'planification_tool'.\n"
    "3. Using the several searches given, you need to research about each of them to answer the original query, using the 'search_tool'.\n"
    "4. Then, you need to evaluate each search's response, using the 'evalution_tool'.\n"
    "4.1. Follow this path if the evaluation goes wrong. Restart from step 3 to search again better.\n"
    "4.2. Follow this path if the evaluation is correct. Handoff for Email Sending: Pass ONLY the original query and the search results obtained in step 3 "
    "to the 'Email Manager' agent.\n"
    "Always delegate the work to the different tools you have been provided, never try to do any of the steps by yourself."
    "You must hand off EXACTLY the original query along the search results obtained in step 3."
)


questioning_tool = questioning_agent.as_tool(tool_name="questioning_tool", tool_description="To generate claryfing questions about the query to understand it better")
planification_tool = planner_agent.as_tool(tool_name="planification_tool", tool_description="To generate search statements to answer the given query")
search_tool = search_agent.as_tool(tool_name="search_tool", tool_description="To serch about the given search statements")
evaluation_tool = evaluator_agent.as_tool(tool_name="evaluation_tool", tool_description="To evaluate the results of the searches")

tools = [questioning_tool,planification_tool,search_tool,evaluation_tool]

    
research_manager_agent = Agent(
    name="Research Manager agent",
    instructions=INSTRUCTIONS,
    tools=tools,
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
    handoffs=[email_manager_agent],
)
