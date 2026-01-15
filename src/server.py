import argparse
import json
from pathlib import Path

import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor

# Path to benchmark data files
TASKS_DIR = Path(__file__).parent.parent / "jan15_tasks"


# ============================================================================
# File Serving Endpoints
# ============================================================================

async def list_files(request: Request) -> JSONResponse:
    """List all available benchmark data files."""
    files = []
    if TASKS_DIR.exists():
        for file_path in sorted(TASKS_DIR.iterdir()):
            if file_path.is_file() and not file_path.name.startswith("."):
                # Don't expose ground truth file
                if "ground_truth" in file_path.name.lower():
                    continue
                files.append({
                    "name": file_path.name,
                    "size_bytes": file_path.stat().st_size,
                    "url": f"/files/{file_path.name}",
                })
    return JSONResponse({"files": files})


async def get_file(request: Request) -> Response:
    """Download a specific benchmark data file."""
    filename = request.path_params.get("filename", "")
    
    # Security: prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        return PlainTextResponse("Invalid filename", status_code=400)
    
    file_path = TASKS_DIR / filename
    
    # Don't serve ground truth file
    if "ground_truth" in filename.lower():
        return PlainTextResponse("File not available", status_code=403)
    
    if not file_path.exists() or not file_path.is_file():
        return PlainTextResponse(f"File not found: {filename}", status_code=404)
    
    # Determine content type based on extension
    extension = file_path.suffix.lower()
    content_types = {
        ".csv": "text/csv",
        ".xml": "application/xml",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".json": "application/json",
    }
    content_type = content_types.get(extension, "application/octet-stream")
    
    # Read and return file content
    if extension in [".csv", ".xml", ".json"]:
        content = file_path.read_text(encoding="utf-8")
        return PlainTextResponse(content, media_type=content_type)
    else:
        content = file_path.read_bytes()
        return Response(content, media_type=content_type)


# File serving routes
file_routes = [
    Route("/files/", endpoint=list_files, methods=["GET"]),
    Route("/files/{filename}", endpoint=get_file, methods=["GET"]),
]


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    # Fill in your agent card
    # See: https://a2a-protocol.org/latest/tutorials/python/3-agent-skills-and-card/
    
    skill = AgentSkill(
        id="data_integration_evaluation",
        name="TPC-DI Data Integration Evaluation",
        description="Evaluates agents on their ability to integrate data from multiple sources (customers, accounts, trades), join tables correctly, and compute accurate per-customer aggregations.",
        tags=["data-integration", "benchmark", "evaluation", "tpc-di", "etl"],
        examples=[
            """
{
  "participants": {
    "data_integrator": "http://127.0.0.1:9010"
  },
  "config": {
    "timeout": 300
  }
}
""".strip()
        ]
    )

    agent_card = AgentCard(
        name="TPC-DI Data Integration Evaluator",
        description="A Green Agent that benchmarks Purple Agents on TPC-DI data integration tasks. Evaluates their ability to merge customer, account, and trade data into accurate per-customer aggregations.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    # Build the app and add file-serving routes
    app = server.build()
    app.routes.extend(file_routes)
    
    print(f"Starting TPC-DI Data Integration Evaluator on http://{args.host}:{args.port}")
    print(f"  A2A endpoint: http://{args.host}:{args.port}/")
    print(f"  File listing: http://{args.host}:{args.port}/files/")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
