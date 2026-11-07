"""Publish our tools once over MCP; any MCP client (Claude Desktop, IDEs) can use them.

Run:  pip install mcp   then   python mcp_server.py
"""
try:                                   # mcp 1.x
    from mcp.server.fastmcp import FastMCP as Server
except ImportError:                    # mcp 2.x renamed FastMCP -> MCPServer
    from mcp.server.mcpserver import MCPServer as Server

import memory
import tools

mcp = Server("scholar-tools")


@mcp.tool()
def search_arxiv(query: str, max_results: int = 5, year_from: int | None = None):
    """Search arXiv for papers. Use 3-6 keywords."""
    return tools.search_arxiv(query, max_results, year_from)


@mcp.tool()
def search_openalex(query: str, max_results: int = 5, year_from: int | None = None):
    """Search OpenAlex (all disciplines, with citation counts). Use 3-6 keywords."""
    return tools.search_openalex(query, max_results, year_from)


@mcp.tool()
def get_citations(title: str) -> dict:
    """Citation count, venue and year for ONE exact paper title."""
    return tools.get_citations(title)


@mcp.tool()
def recall_notes(query: str, k: int = 3) -> list[str]:
    """Recall paper notes saved by earlier research runs."""
    return memory.recall_notes(query, k)


if __name__ == "__main__":
    mcp.run()
