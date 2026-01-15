"""
Mock Purple Agent for TPC-DI Data Integration Benchmark.

This agent fetches files via HTTP from the green agent's file server,
merges them using pandas, and returns the aggregated customer data.

Run with: python src/mock_purple.py --port 9010
"""

import argparse
import json
import os
import re
from io import StringIO

import httpx
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
    
    Fetches source files via HTTP, joins them, and computes customer aggregations.
    """

    def __init__(self):
        self.default_file_server = os.getenv("FILE_SERVER_URL", "http://localhost:9009")

    def extract_file_urls(self, message_text: str) -> dict[str, str]:
        """Extract file URLs from the task message."""
        urls = {}
        
        # Look for URLs in the message
        url_pattern = r'(https?://[^\s]+\.csv)'
        matches = re.findall(url_pattern, message_text)
        
        for url in matches:
            if "customers" in url.lower():
                urls["customers"] = url
            elif "accounts" in url.lower():
                urls["accounts"] = url
            elif "trades" in url.lower():
                urls["trades"] = url
        
        return urls

    async def fetch_csv(self, url: str, client: httpx.AsyncClient) -> pd.DataFrame:
        """Fetch a CSV file from a URL and return as DataFrame."""
        response = await client.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))

    async def run(self, message_text: str, updater: TaskUpdater) -> None:
        """Process the data integration task."""
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Starting data integration task...")
        )

        # Extract file URLs from task message
        file_urls = self.extract_file_urls(message_text)
        
        if not all(key in file_urls for key in ["customers", "accounts", "trades"]):
            # Fall back to default file server
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Using default file server: {self.default_file_server}")
            )
            file_urls = {
                "customers": f"{self.default_file_server}/files/customers_tpcdi_lite_v3.csv",
                "accounts": f"{self.default_file_server}/files/accounts_tpcdi_lite_v3.csv",
                "trades": f"{self.default_file_server}/files/trades_tpcdi_lite_v3.csv",
            }

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Fetching files via HTTP...")
        )

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                # Fetch all CSV files
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Downloading customers file...")
                )
                customers_df = await self.fetch_csv(file_urls["customers"], client)
                
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Downloading accounts file...")
                )
                accounts_df = await self.fetch_csv(file_urls["accounts"], client)
                
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Downloading trades file...")
                )
                trades_df = await self.fetch_csv(file_urls["trades"], client)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Merging and aggregating data...")
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

        except httpx.HTTPError as e:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Error fetching files: {str(e)}"))],
                name="Error",
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
