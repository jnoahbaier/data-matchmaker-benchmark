"""
MCP Server for Data Integration Benchmark.

Exposes benchmark files as MCP Resources for Purple Agents to fetch and process.
"""

import base64
import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCPServer")

# Determine the tasks directory
TASKS_DIR = Path(__file__).parent.parent / "jan15_tasks"

# Create the MCP server
mcp = FastMCP(
    name="DataIntegrationBenchmark",
    instructions="""
    This MCP server provides access to TPC-DI style benchmark data for data integration tasks.
    
    Available resources:
    - benchmark://tasks/list - Get list of all available benchmark files
    - benchmark://tasks/{filename} - Get content of a specific file
    
    Available tools:
    - get_task_info() - Get task metadata and expected output schema
    - get_file_content(filename) - Get file content (alternative to resource)
    
    The benchmark task is to merge/join the source files (accounts, customers, trades, etc.)
    and produce an aggregated result matching the ground truth schema.
    """
)


def get_file_list() -> list[dict]:
    """Get list of all benchmark files with metadata."""
    files = []
    if not TASKS_DIR.exists():
        logger.warning(f"Tasks directory not found: {TASKS_DIR}")
        return files
    
    for file_path in TASKS_DIR.iterdir():
        if file_path.is_file():
            file_info = {
                "name": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "extension": file_path.suffix.lower(),
            }
            
            # Categorize files
            if "ground_truth" in file_path.name.lower():
                file_info["type"] = "ground_truth"
                file_info["description"] = "Expected output - do not use as input"
            elif file_path.suffix.lower() == ".csv":
                file_info["type"] = "source_csv"
                file_info["description"] = "CSV source file for merging"
            elif file_path.suffix.lower() == ".xlsx":
                file_info["type"] = "source_excel"
                file_info["description"] = "Excel source file for merging"
            elif file_path.suffix.lower() == ".xml":
                file_info["type"] = "source_xml"
                file_info["description"] = "XML source file for merging"
            else:
                file_info["type"] = "other"
                file_info["description"] = "Other file"
            
            files.append(file_info)
    
    return sorted(files, key=lambda x: x["name"])


def read_file_content(filename: str) -> tuple[str, str]:
    """
    Read file content and return (content, mime_type).
    
    For text files (CSV), returns raw text.
    For binary files (Excel, XML), returns base64-encoded content.
    """
    file_path = TASKS_DIR / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    
    if not file_path.is_file():
        raise ValueError(f"Not a file: {filename}")
    
    # Security check - prevent directory traversal
    if not file_path.resolve().is_relative_to(TASKS_DIR.resolve()):
        raise ValueError(f"Invalid file path: {filename}")
    
    extension = file_path.suffix.lower()
    
    if extension == ".csv":
        # Return CSV as plain text
        content = file_path.read_text(encoding="utf-8")
        return content, "text/csv"
    
    elif extension == ".xlsx":
        # Return Excel as base64
        content = base64.b64encode(file_path.read_bytes()).decode("utf-8")
        return content, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64"
    
    elif extension == ".xml":
        # Return XML as text (it's usually text-based)
        try:
            content = file_path.read_text(encoding="utf-8")
            return content, "application/xml"
        except UnicodeDecodeError:
            # Fallback to base64 if not valid UTF-8
            content = base64.b64encode(file_path.read_bytes()).decode("utf-8")
            return content, "application/xml;base64"
    
    else:
        # Default: try text, fallback to base64
        try:
            content = file_path.read_text(encoding="utf-8")
            return content, "text/plain"
        except UnicodeDecodeError:
            content = base64.b64encode(file_path.read_bytes()).decode("utf-8")
            return content, "application/octet-stream;base64"


# ============================================================================
# MCP Resources
# ============================================================================

@mcp.resource("benchmark://tasks/list")
def list_tasks_resource() -> str:
    """
    List all available benchmark files.
    
    Returns JSON array of file metadata including name, size, type, and description.
    """
    files = get_file_list()
    return json.dumps(files, indent=2)


@mcp.resource("benchmark://tasks/{filename}")
def get_file_resource(filename: str) -> str:
    """
    Get the content of a specific benchmark file.
    
    For CSV files, returns raw text content.
    For Excel/binary files, returns base64-encoded content with appropriate MIME type indicator.
    
    Args:
        filename: Name of the file to retrieve (e.g., 'customers_tpcdi_lite_v3.csv')
    
    Returns:
        File content as string (text or base64-encoded)
    """
    try:
        content, mime_type = read_file_content(filename)
        
        # For resources, we return a JSON envelope with metadata
        return json.dumps({
            "filename": filename,
            "mime_type": mime_type,
            "content": content,
            "is_base64": mime_type.endswith(";base64")
        })
    except FileNotFoundError as e:
        return json.dumps({"error": str(e), "filename": filename})
    except ValueError as e:
        return json.dumps({"error": str(e), "filename": filename})


# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool()
def get_task_info() -> dict:
    """
    Get information about the benchmark task.
    
    Returns task description, available files, and expected output schema.
    """
    files = get_file_list()
    
    # Separate source files from ground truth
    source_files = [f for f in files if f["type"] != "ground_truth"]
    ground_truth_files = [f for f in files if f["type"] == "ground_truth"]
    
    return {
        "task_name": "TPC-DI Data Integration Benchmark",
        "task_description": """
        Merge and aggregate data from multiple source files (accounts, customers, trades, etc.)
        to produce a customer-centric summary with trading statistics.
        
        The goal is to join the source tables on appropriate keys and compute aggregations
        per customer including: number of accounts, total balance, number of trades,
        total trade volume, total trade value, and symbols traded.
        """,
        "source_files": source_files,
        "ground_truth_files": ground_truth_files,
        "expected_output_schema": {
            "columns": [
                {"name": "customer_id", "type": "integer", "description": "Unique customer identifier"},
                {"name": "customer_name", "type": "string", "description": "Full name (first + last)"},
                {"name": "country", "type": "string", "description": "Customer's country"},
                {"name": "num_accounts", "type": "integer", "description": "Number of accounts owned"},
                {"name": "total_balance", "type": "float", "description": "Sum of all account balances"},
                {"name": "num_trades", "type": "integer", "description": "Total number of completed trades"},
                {"name": "total_trade_volume", "type": "integer", "description": "Sum of trade quantities"},
                {"name": "total_trade_value", "type": "float", "description": "Sum of (quantity * trade_price)"},
                {"name": "symbols_traded", "type": "string", "description": "Comma-separated unique symbols"}
            ],
            "notes": [
                "Only include trades with status 'CMPT' (completed)",
                "Join accounts to customers via customer_id",
                "Join trades to accounts via account_id",
                "Aggregate per customer"
            ]
        },
        "mcp_resources": [
            "benchmark://tasks/list",
            "benchmark://tasks/{filename}"
        ]
    }


@mcp.tool()
def get_file_content(filename: str) -> dict:
    """
    Get the content of a specific benchmark file.
    
    This is an alternative to using the resource URI directly.
    
    Args:
        filename: Name of the file to retrieve (e.g., 'customers_tpcdi_lite_v3.csv')
    
    Returns:
        Dictionary with filename, mime_type, content, and is_base64 flag
    """
    try:
        content, mime_type = read_file_content(filename)
        return {
            "filename": filename,
            "mime_type": mime_type,
            "content": content,
            "is_base64": mime_type.endswith(";base64")
        }
    except FileNotFoundError as e:
        return {"error": str(e), "filename": filename}
    except ValueError as e:
        return {"error": str(e), "filename": filename}


@mcp.tool()
def list_files() -> list[dict]:
    """
    List all available benchmark files.
    
    Returns list of file metadata including name, size, type, and description.
    """
    return get_file_list()


# For testing/debugging
if __name__ == "__main__":
    print("Tasks directory:", TASKS_DIR)
    print("Files available:", get_file_list())
    print("\nTask info:", json.dumps(get_task_info(), indent=2))
