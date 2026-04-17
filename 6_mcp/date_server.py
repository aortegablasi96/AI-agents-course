from mcp.server.fastmcp import FastMCP
from date import Date

mcp = FastMCP("date_server")

@mcp.tool()
async def get_date() -> str:
    """Get the current date and time in string format.

    """
    return Date.get().date

if __name__ == "__main__":
    mcp.run(transport='stdio')