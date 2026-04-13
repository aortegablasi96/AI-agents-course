"""
Tools which require internet interaction for our Agents to use.
"""

from typing import Literal
from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from dotenv import load_dotenv
import os
import requests
from langchain.agents import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

load_dotenv(override=True)

class WebTools:

    def __init__(self):
        self.serper = GoogleSerperAPIWrapper()
        self.wiki = WikipediaAPIWrapper()
        self.tool_names = []
    
    def tools(self):
        """ Main entry point for retrieving and defining available tools."""

        return [
            self.search_tool,
            self.wikipedia_tool,
            self.youtube_tool,
            self.yahoofinance_tool,
        ]
    
    async def playwright_tools(self):
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=False)
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
        return toolkit.get_tools(), browser, playwright


    def wikipedia_tool(self):
        """ Wikipedia tool - performing wiki searches. """

        return WikipediaQueryRun(api_wrapper=self.wiki)

    def youtube_tool(self):
        """ Tool to query on youtube """

        return YouTubeSearchTool()
    
    def yahoofinance_tool(self):
        """ Tool to query about financial news in yahoo finance. """
        return YahooFinanceNewsTool()

    def search_tool(self):
        """ Search tool - perform web searches using Serper. """

        return Tool(
            name="search",
            func=self.serper.run,
            description="Use this tool when you want to get the results of an online web search"
        )