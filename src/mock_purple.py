"""
Mock Purple Agent for TPC-DI Data Integration Benchmark.

This agent fetches files from the MCP server, merges them using pandas,
and returns the aggregated customer data.

Run with: python src/mock_purple.py --port 9010
"""

import argparse
import json
import os
from io import StringIO

import pandas as pd
import uvicorn
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Task,
    TaskState,
    Part,
    TextPart,
    DataPart,
)
from a2a.utils import get_message_text, new_agent_text_message, new_task

load_dotenv()

TERMINAL_STATES = {TaskState.completed, TaskState.canceled, TaskState.failed, TaskState.rejected}


class DataIntegrationAgent:
    """
    Purple Agent that performs TPC-DI data integration.
    
    Fetches source files, joins them, and computes customer aggregations.
    """

    def __init__(self):
        self.mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:9009/mcp")

    async def run(self, message_text: str, updater: TaskUpdater) -> None:
        """Process the data integration task."""
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Starting data integration task...")
        )

        try:
            request = json.loads(message_text)
            mcp_url = request.get("mcp_server_url", self.mcp_url)
        except json.JSONDecodeError:
            mcp_url = self.mcp_url

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Fetching files from MCP server...")
        )

        try:
            # For this mock, we'll read files directly from disk
            # In a real scenario, you'd use the MCP client to fetch files
            from pathlib import Path
            tasks_dir = Path(__file__).parent.parent / "jan15_tasks"

            # Load source files
            customers_df = pd.read_csv(tasks_dir / "customers_tpcdi_lite_v3.csv")
            accounts_df = pd.read_csv(tasks_dir / "accounts_tpcdi_lite_v3.csv")
            trades_df = pd.read_csv(tasks_dir / "trades_tpcdi_lite_v3.csv")

            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Merging and aggregating data...")
            )

            # Perform the data integration
            result_df = self.integrate_data(customers_df, accounts_df, trades_df)

            # Convert to CSV
            csv_output = result_df.to_csv(index=False)

            await updater.add_artifact(
                parts=[Part(root=DataPart(data={
                    "status": "success",
                    "rows": len(result_df),
                    "columns": list(result_df.columns),
                    "data": csv_output
                }))],
                name="Integration Result",
            )

        except Exception as e:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Error during integration: {str(e)}"))],
                name="Error",
            )

    def integrate_data(
        self,
        customers: pd.DataFrame,
        accounts: pd.DataFrame,
        trades: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Perform the TPC-DI data integration task.
        
        Join tables and compute per-customer aggregations.
        """
        # Join accounts to customers
        customer_accounts = pd.merge(
            customers,
            accounts,
            on="customer_id",
            how="left"
        )

        # Filter completed trades only
        completed_trades = trades[trades["trade_status"] == "CMPT"].copy()

        # Join trades to accounts
        account_trades = pd.merge(
            customer_accounts,
            completed_trades,
            on="account_id",
            how="left"
        )

        # Compute trade value
        account_trades["trade_value"] = (
            account_trades["quantity"].fillna(0) * 
            account_trades["trade_price"].fillna(0)
        )

        # Aggregate per customer
        result = account_trades.groupby("customer_id").agg(
            customer_name=("first_name", lambda x: f"{x.iloc[0]} {account_trades.loc[x.index[0], 'last_name']}" if len(x) > 0 else ""),
            country=("country", "first"),
            num_accounts=("account_id", "nunique"),
            total_balance=("balance", "sum"),
            num_trades=("trade_id", lambda x: x.notna().sum()),
            total_trade_volume=("quantity", lambda x: x.fillna(0).sum()),
            total_trade_value=("trade_value", "sum"),
            symbols_traded=("symbol", lambda x: ",".join(sorted(set(x.dropna().astype(str)))) if x.notna().any() else "")
        ).reset_index()

        # Fix customer_name (the lambda above doesn't work well with groupby)
        name_map = customers.set_index("customer_id").apply(
            lambda row: f"{row['first_name']} {row['last_name']}", axis=1
        ).to_dict()
        result["customer_name"] = result["customer_id"].map(name_map)

        # Reorder columns
        result = result[[
            "customer_id", "customer_name", "country", "num_accounts",
            "total_balance", "num_trades", "total_trade_volume",
            "total_trade_value", "symbols_traded"
        ]]

        return result


class MockPurpleExecutor(AgentExecutor):
    """Executor for the mock Purple Agent."""

    def __init__(self):
        self.agent = DataIntegrationAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            return

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            return

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        try:
            message_text = get_message_text(msg)
            await self.agent.run(message_text, updater)
            await updater.complete()
        except Exception as e:
            await updater.failed(new_agent_text_message(f"Error: {e}"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


def main():
    parser = argparse.ArgumentParser(description="Run mock Purple Agent for TPC-DI benchmark.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9010)
    args = parser.parse_args()

    skill = AgentSkill(
        id="data_integrator",
        name="Data Integrator",
        description="Integrates and aggregates data from multiple sources",
        tags=["data-integration", "etl", "pandas"],
        examples=["Merge customer, account, and trade data"]
    )

    agent_card = AgentCard(
        name="Mock Data Integrator (Purple Agent)",
        description="A mock Purple Agent that performs TPC-DI data integration for benchmarking",
        url=f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=MockPurpleExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print(f"🟣 Starting Mock Purple Agent on http://{args.host}:{args.port}")
    print(f"   This agent performs TPC-DI data integration")
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
