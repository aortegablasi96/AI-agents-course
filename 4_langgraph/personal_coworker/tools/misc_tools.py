"""
Miscellaneous tools for our Agents to use.
"""

from dotenv import load_dotenv
import os
import requests
from langchain.agents import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools import YouTubeSearchTool
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

load_dotenv(override=True)

class MiscTools:

    def __init__(self):
        self.pushover_token = os.getenv("PUSHOVER_TOKEN")
        self.pushover_user = os.getenv("PUSHOVER_USER")
        self.pushover_url = "https://api.pushover.net/1/messages.json"
        self.serper = GoogleSerperAPIWrapper()
        self.wiki = WikipediaAPIWrapper()
        self.tool_names = []
    
    def tools(self):
        """ Main entry point for retrieving and defining available tools."""

        return [
            self.pushover_tool, 
            self.python_tool, 
            self.file_tool
        ]

    def file_tool(self, root_dir:str = "sandbox"):
        """ File tool - interacting with file explorer. """

        toolkit = FileManagementToolkit(root_dir=root_dir)
        return toolkit.get_tools()

    def push(self, text: str):
        """ Send a push notification to the user"""

        requests.post(self.pushover_url, data = {"token": self.pushover_token, "user": self.pushover_user, "message": text})
        return "success"

    def pushover_tool(self):
        """ Pushover tool - Send Push notifications to the user. """

        return Tool(name="send_push_notification", func=self.push, description="Use this tool when you want to send a push notification")

    def python_tool(self):
        """ Python repl tool - writing/running Python code. """

        return PythonREPLTool()