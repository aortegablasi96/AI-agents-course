from agents import ModelSettings
from writer_agent import writer_agent, ReportData
from email_agent import email_agent
from agents import Agent, function_tool
import asyncio


INSTRUCTIONS = (
    "You are an email assistant manager. You will be given with a query and the research results and you are required "
    "to write a report using these results and the query and send it. You have several agent-as-tools to accomplish your task.\n"
    "You must always follow the following steps:\n"
    "1. With the different researches given, then you have to write down a report summarizing all of them, using the 'reporting_tool'.\n"
    "2. Finally, when the report is created, you need to send it by mail using the 'send_email_tool'.\n"
    "Always delegate the work to the different tools you have been provided, never try to do any of the steps by yourself."
    "When you're done, save as final output the report written, with no other remarks."
)


reporting_tool = writer_agent.as_tool(tool_name="reporting_tool", tool_description="To write the summary report")
send_email_tool = email_agent.as_tool(tool_name="send_email_tool", tool_description="To send the email to the user")

tools = [reporting_tool,send_email_tool]

    
email_manager_agent = Agent(
    name="Email Manager agent",
    instructions=INSTRUCTIONS,
    tools=tools,
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)
