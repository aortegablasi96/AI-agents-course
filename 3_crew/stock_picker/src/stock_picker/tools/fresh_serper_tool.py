import os
import requests
from crewai.tools import BaseTool

class FreshSerperTool(BaseTool):
    name: str = "Fresh Serper Search"
    description: str = "Search Google via Serper with freshness filtering."

    def _run(self, query: str) -> str:
        url = "https://google.serper.dev/search"
        payload = {
            "q": query,
            "tbs": "qdr:y"  # last year (change to qdr:w, qdr:d, qdr:y)
        }

        headers = {
            "X-API-KEY": os.getenv("SERPER_API_KEY"),
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.text