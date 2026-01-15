#!/usr/bin/env python3
"""
Local end-to-end test script for the TPC-DI Data Integration Benchmark.

This script mirrors the official AgentBeats `agentbeats-run` flow:
1. Starts the green agent server
2. Starts the mock purple agent server
3. Waits for both to be ready
4. Sends the assessment request via A2A client
5. Prints the results
6. Shuts down cleanly

Usage:
    uv run scripts/run_local_test.py
    
Or with options:
    uv run scripts/run_local_test.py --show-logs --timeout 600
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from messenger import send_message, A2ANetworkError


# Configuration
GREEN_HOST = "127.0.0.1"
GREEN_PORT = 9009
PURPLE_HOST = "127.0.0.1"
PURPLE_PORT = 9010


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


async def check_agent_ready(url: str, timeout: float = 2.0) -> bool:
    """Check if an agent is responding by fetching the agent card."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{url}/.well-known/agent-card.json")
            return response.status_code == 200
    except Exception:
        return False


async def wait_for_agents(
    green_url: str,
    purple_url: str,
    timeout: int = 30
) -> bool:
    """Wait for all agents to be healthy and responding."""
    endpoints = [green_url, purple_url]
    print(f"Waiting for {len(endpoints)} agent(s) to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        ready_count = sum([
            await check_agent_ready(ep) for ep in endpoints
        ])
        
        if ready_count == len(endpoints):
            print("All agents ready!")
            return True
        
        print(f"  {ready_count}/{len(endpoints)} agents ready, waiting...")
        await asyncio.sleep(1)
    
    print(f"Timeout: Not all agents became ready after {timeout}s")
    return False


def print_result(result: dict) -> None:
    """Pretty print the assessment result."""
    print("\n" + "=" * 60)
    print("ASSESSMENT RESULT")
    print("=" * 60)
    
    if "response" in result:
        # Parse the response
        response = result["response"]
        try:
            # Try to extract JSON data from response
            if isinstance(response, str):
                # Look for JSON in the response
                lines = response.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            data = json.loads(line)
                            print(json.dumps(data, indent=2))
                        except json.JSONDecodeError:
                            print(line)
                    elif line:
                        print(line)
            else:
                print(json.dumps(response, indent=2))
        except Exception as e:
            print(f"Raw response: {response}")
    else:
        print(json.dumps(result, indent=2))
    
    print("=" * 60)


async def run_assessment(
    green_url: str,
    purple_url: str,
    timeout: int = 300
) -> dict:
    """Send the assessment request to the green agent."""
    request = {
        "participants": {
            "data_integrator": purple_url
        },
        "config": {
            "timeout": timeout,
            "file_server_url": green_url
        }
    }
    
    print(f"\nSending assessment request to green agent...")
    print(f"  Green Agent: {green_url}")
    print(f"  Purple Agent: {purple_url}")
    print(f"  Timeout: {timeout}s")
    
    result = await send_message(
        message=json.dumps(request),
        base_url=green_url,
        streaming=False,
        timeout=timeout + 60,  # Extra buffer for the full flow
    )
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run local end-to-end test for TPC-DI Data Integration Benchmark"
    )
    parser.add_argument(
        "--show-logs",
        action="store_true",
        help="Show agent stdout/stderr"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Assessment timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=30,
        help="Agent startup timeout in seconds (default: 30)"
    )
    args = parser.parse_args()
    
    project_root = get_project_root()
    green_url = f"http://{GREEN_HOST}:{GREEN_PORT}"
    purple_url = f"http://{PURPLE_HOST}:{PURPLE_PORT}"
    
    # Configure output
    sink = None if args.show_logs else subprocess.DEVNULL
    
    # Setup environment
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    procs = []
    
    try:
        # Start green agent
        print(f"Starting green agent at {green_url}...")
        green_proc = subprocess.Popen(
            [
                sys.executable, "src/server.py",
                "--host", GREEN_HOST,
                "--port", str(GREEN_PORT)
            ],
            cwd=project_root,
            env=env,
            stdout=sink,
            stderr=sink,
            start_new_session=True,
        )
        procs.append(green_proc)
        
        # Start purple agent
        print(f"Starting purple agent at {purple_url}...")
        purple_proc = subprocess.Popen(
            [
                sys.executable, "src/mock_purple.py",
                "--host", PURPLE_HOST,
                "--port", str(PURPLE_PORT)
            ],
            cwd=project_root,
            env=env,
            stdout=sink,
            stderr=sink,
            start_new_session=True,
        )
        procs.append(purple_proc)
        
        # Wait for agents to be ready
        ready = asyncio.run(wait_for_agents(
            green_url, purple_url, timeout=args.startup_timeout
        ))
        
        if not ready:
            print("Error: Agents failed to start. Exiting.")
            sys.exit(1)
        
        # Run the assessment
        print("\n" + "-" * 60)
        print("Running assessment...")
        print("-" * 60)
        
        result = asyncio.run(run_assessment(
            green_url, purple_url, timeout=args.timeout
        ))
        
        # Print result
        print_result(result)
        
        # Check status
        status = result.get("status", "completed")
        if status != "completed":
            print(f"\nAssessment ended with status: {status}")
            sys.exit(1)
        
        print("\nAssessment completed successfully!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except A2ANetworkError as e:
        print(f"\nNetwork error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
        print("\nShutting down agents...")
        for proc in procs:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass
        
        # Give them time to clean up
        time.sleep(1)
        
        # Force kill if still running
        for proc in procs:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
        
        print("Done.")


if __name__ == "__main__":
    main()
