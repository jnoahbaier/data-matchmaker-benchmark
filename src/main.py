"""
Main entry point for the Green Agent server.

Combines:
- A2A server for agent-to-agent communication
- MCP server for benchmark file access via SSE
- Agent card endpoint for discovery
"""

import argparse
import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.routing import Mount

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor
from mcp_server import mcp

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GreenAgent")


def create_agent_card(base_url: str) -> AgentCard:
    """Create the agent card for discovery."""
    skill = AgentSkill(
        id="data_integration_benchmark",
        name="Data Integration Benchmark",
        description=(
            "Evaluates agents on TPC-DI style data integration tasks. "
            "Provides benchmark files via MCP resources and scores results "
            "against ground truth."
        ),
        tags=["data-integration", "benchmark", "evaluation", "etl", "tpc-di"],
        examples=[
            "Evaluate my data integration solution",
            "Get benchmark files for the TPC-DI task",
            "Submit my merged customer data for scoring"
        ]
    )

    return AgentCard(
        name="Data Integration Benchmark (Green Agent)",
        description=(
            "A Green Agent that benchmarks Purple Agents on TPC-DI style data integration tasks. "
            "Serves source files (CSV, Excel, XML) via MCP resources and evaluates "
            "merged/aggregated results against ground truth."
        ),
        url=base_url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )


def create_app(host: str = "0.0.0.0", port: int = 9009, card_url: str = None) -> FastAPI:
    """
    Create the combined FastAPI application.
    
    Args:
        host: Host to bind the server
        port: Port to bind the server
        card_url: Optional URL to advertise in the agent card
    
    Returns:
        FastAPI application with MCP and A2A servers mounted
    """
    base_url = card_url or f"http://{host}:{port}/"
    
    # Create FastAPI app
    app = FastAPI(
        title="Data Integration Benchmark - Green Agent",
        description="Green Agent for TPC-DI data integration benchmarking",
        version="1.0.0"
    )
    
    # Create agent card
    agent_card = create_agent_card(base_url)
    
    # Explicit agent card endpoint (for compatibility)
    @app.get("/.well-known/agent-card.json")
    async def get_agent_card():
        """Return the agent card for discovery."""
        return JSONResponse(content=agent_card.model_dump(mode="json", exclude_none=True))
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "green-agent"}
    
    # Info endpoint with MCP URL
    @app.get("/info")
    async def get_info():
        """Get server information including MCP URL."""
        mcp_url = os.getenv("MCP_SERVER_URL", f"{base_url.rstrip('/')}/mcp")
        return {
            "name": "Data Integration Benchmark - Green Agent",
            "version": "1.0.0",
            "mcp_url": mcp_url,
            "a2a_url": base_url,
            "description": "Serves TPC-DI benchmark files via MCP and evaluates data integration results"
        }
    
    # Mount MCP server at /mcp
    mcp_app = mcp.http_app()
    app.mount("/mcp", mcp_app)
    logger.info(f"MCP server mounted at /mcp")
    
    # Create A2A request handler
    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    
    # Create A2A application
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    # Build A2A routes and add them to the main app
    a2a_starlette = a2a_app.build()
    
    # Mount A2A at root, but after other routes
    # We need to add A2A routes manually to avoid conflicts with explicit routes
    for route in a2a_starlette.routes:
        # Skip the agent card route since we defined it explicitly
        if hasattr(route, 'path') and route.path == "/.well-known/agent-card.json":
            continue
        app.routes.append(route)
    
    logger.info(f"A2A server configured at {base_url}")
    
    return app


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run the Green Agent server.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9009,
        help="Port to bind the server (default: 9009)"
    )
    parser.add_argument(
        "--card-url",
        type=str,
        help="URL to advertise in the agent card (for public deployments)"
    )
    args = parser.parse_args()
    
    # Set MCP_SERVER_URL if not already set
    if not os.getenv("MCP_SERVER_URL"):
        base_url = args.card_url or f"http://{args.host}:{args.port}"
        mcp_url = f"{base_url.rstrip('/')}/mcp"
        os.environ["MCP_SERVER_URL"] = mcp_url
        logger.info(f"MCP_SERVER_URL set to: {mcp_url}")
    
    # Create and run the app
    app = create_app(args.host, args.port, args.card_url)
    
    logger.info(f"Starting Green Agent server at http://{args.host}:{args.port}")
    logger.info(f"MCP server available at http://{args.host}:{args.port}/mcp")
    logger.info(f"Agent card at http://{args.host}:{args.port}/.well-known/agent-card.json")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
