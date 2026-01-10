"""
Mock Purple Agent for testing the Schema Merging Evaluator.
Uses Google Gemini to analyze tables and return schema merging results.

Run with: uv run src/mock_purple.py --port 9010
"""
import argparse
import json
import os

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
)
from a2a.utils import get_message_text, new_agent_text_message, new_task
from a2a.utils.errors import ServerError

load_dotenv()

TERMINAL_STATES = {TaskState.completed, TaskState.canceled, TaskState.failed, TaskState.rejected}


class MockPurpleAgent:
    """A simple purple agent that uses Gemini to merge schemas."""

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash")
        else:
            self.model = None

    async def run(self, message_text: str, updater: TaskUpdater) -> None:
        await updater.update_status(
            TaskState.working, new_agent_text_message("Analyzing tables...")
        )

        try:
            request = json.loads(message_text)
            tables = request.get("tables", [])
            task = request.get("task", "")
        except json.JSONDecodeError:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text='{"error": "Invalid JSON input"}'))],
                name="Error",
            )
            return

        if self.model:
            # Use Gemini for analysis
            prompt = f"""You are a data engineer analyzing database tables.

Given these tables:
{json.dumps(tables, indent=2)}

Task: {task}

Return ONLY valid JSON (no markdown, no explanation) with this exact structure:
{{
    "primary_keys": {{"table_name": "column_name", ...}},
    "join_columns": [["table1.col", "table2.col"], ...],
    "inconsistencies": ["description of inconsistency 1", ...],
    "merged_schema": {{"unified_table_name": ["col1", "col2", ...]}}
}}"""

            try:
                response = self.model.generate_content(prompt)
                result = response.text.strip()
                # Clean up markdown if present
                if result.startswith("```"):
                    result = result.split("```")[1]
                    if result.startswith("json"):
                        result = result[4:]
                    result = result.strip()
            except Exception as e:
                result = json.dumps({"error": str(e)})
        else:
            # Fallback: return a reasonable static response
            result = json.dumps({
                "primary_keys": {t["name"]: t["columns"][0] for t in tables},
                "join_columns": [],
                "inconsistencies": ["Unable to analyze without API key"],
                "merged_schema": {"merged": [col for t in tables for col in t["columns"]]}
            })

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=result))],
            name="Schema Analysis",
        )


class MockPurpleExecutor(AgentExecutor):
    def __init__(self):
        self.agent = MockPurpleAgent()

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
    parser = argparse.ArgumentParser(description="Run mock purple agent for testing.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9010)
    args = parser.parse_args()

    skill = AgentSkill(
        id="schema_merger",
        name="Schema Merger",
        description="Analyzes and merges database schemas",
        tags=["schema", "database"],
        examples=[]
    )

    agent_card = AgentCard(
        name="Mock Schema Merger",
        description="A mock purple agent for testing schema merging evaluation",
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

    print(f"Starting mock purple agent on http://{args.host}:{args.port}")
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
