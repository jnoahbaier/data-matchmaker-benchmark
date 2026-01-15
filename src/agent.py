"""
Green Agent for TPC-DI Data Integration Benchmark.

Evaluates Purple Agents on their ability to merge/join source data files
and produce correct aggregated customer-centric results.

Uses a hybrid evaluation approach:
1. Deterministic numerical scoring for accuracy metrics
2. LLM-powered qualitative feedback for insights and explanations
"""

import json
import logging
import os
import re
from io import StringIO
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
import pandas as pd
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataIntegrationEvaluator")

# Get Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    logger.info("Gemini API key loaded successfully")
else:
    logger.warning("GEMINI_API_KEY not set - LLM feedback will be disabled")

# Paths
TASKS_DIR = Path(__file__).parent.parent / "jan15_tasks"
GROUND_TRUTH_FILE = TASKS_DIR / "gold_ground_truth_tpcdi_lite_v3.csv"


class Agent:
    """Green Agent that evaluates Purple Agents on TPC-DI data integration tasks.
    
    Uses hybrid evaluation:
    - Deterministic numerical scoring for accuracy
    - LLM-powered feedback for qualitative insights
    """

    def __init__(self):
        self.messenger = Messenger()
        self._ground_truth_df: Optional[pd.DataFrame] = None
        
        # Initialize Gemini client for LLM feedback
        self._genai_client = None
        if GEMINI_API_KEY:
            try:
                self._genai_client = genai.Client(api_key=GEMINI_API_KEY)
                logger.info("Gemini client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")

    @property
    def ground_truth(self) -> pd.DataFrame:
        """Load and cache ground truth data."""
        if self._ground_truth_df is None:
            if not GROUND_TRUTH_FILE.exists():
                raise FileNotFoundError(f"Ground truth file not found: {GROUND_TRUTH_FILE}")
            self._ground_truth_df = pd.read_csv(GROUND_TRUTH_FILE)
            logger.info(f"Loaded ground truth: {len(self._ground_truth_df)} rows")
        return self._ground_truth_df

    async def generate_llm_feedback(
        self,
        submitted_df: pd.DataFrame,
        score: int,
        details: dict,
    ) -> dict:
        """
        Generate qualitative LLM feedback on the submission.
        
        This adds an intelligent analysis layer on top of the numerical scoring,
        explaining what the Purple Agent did well or poorly.
        
        Returns a dict with 'summary', 'strengths', 'weaknesses', and 'recommendations'.
        """
        if not self._genai_client:
            return {
                "enabled": False,
                "message": "LLM feedback disabled - GEMINI_API_KEY not configured"
            }
        
        try:
            # Prepare evaluation summary for the LLM
            ground_truth = self.ground_truth
            
            # Sample data for context (don't send entire dataset)
            gt_sample = ground_truth.head(3).to_string()
            sub_sample = submitted_df.head(3).to_string() if len(submitted_df) > 0 else "No data submitted"
            
            # Build prompt
            prompt = f"""You are an expert evaluator for a data integration benchmark competition.

## Task Context
Purple Agents are evaluated on their ability to:
1. Fetch data from multiple source files (customers, accounts, trades)
2. Join the tables correctly on customer_id and account_id
3. Filter to completed trades only (trade_status = 'CMPT')
4. Compute per-customer aggregations accurately

## Numerical Evaluation Results
- **Overall Score**: {score}/100
- **Column Check** ({details.get('columns', {}).get('score', 'N/A')}/20): 
  - Missing columns: {details.get('columns', {}).get('missing', [])}
  - Extra columns: {details.get('columns', {}).get('extra', [])}
- **Row Count** ({details.get('row_count', {}).get('score', 'N/A')}/10):
  - Expected: {details.get('row_count', {}).get('expected', 'N/A')} rows
  - Submitted: {details.get('row_count', {}).get('submitted', 'N/A')} rows
- **Customer Coverage** ({details.get('customer_coverage', {}).get('score', 'N/A')}/15):
  - Coverage: {details.get('customer_coverage', {}).get('coverage_pct', 'N/A')}%
- **Numeric Accuracy** ({details.get('numeric_accuracy', {}).get('score', 'N/A')}/40):
  {json.dumps(details.get('numeric_accuracy', {}).get('columns', {}), indent=2)}
- **String Accuracy** ({details.get('string_accuracy', {}).get('score', 'N/A')}/15):
  {json.dumps(details.get('string_accuracy', {}).get('fields', {}), indent=2)}

## Sample Ground Truth Data
{gt_sample}

## Sample Submitted Data
{sub_sample}

## Your Task
Provide a concise qualitative evaluation with:
1. **Summary** (2-3 sentences): Overall assessment of the submission quality
2. **Strengths** (bullet points): What the Purple Agent did well
3. **Weaknesses** (bullet points): Key issues or errors
4. **Recommendations** (bullet points): Specific improvements for the Purple Agent

Be constructive and specific. Focus on data integration quality, not generic feedback.
Respond in JSON format with keys: summary, strengths, weaknesses, recommendations."""

            # Call Gemini API using new google.genai SDK
            logger.info("Generating LLM feedback via Gemini...")
            response = await self._genai_client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                ),
            )
            
            # Parse response
            response_text = response.text.strip()
            
            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
                response_text = re.sub(r'\s*```$', '', response_text)
            
            feedback = json.loads(response_text)
            feedback["enabled"] = True
            feedback["model"] = "gemini-2.0-flash"
            
            logger.info("LLM feedback generated successfully")
            return feedback
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Return raw text if JSON parsing fails
            return {
                "enabled": True,
                "model": "gemini-2.0-flash",
                "summary": response_text if 'response_text' in dir() else "Failed to generate feedback",
                "parse_error": str(e),
            }
        except Exception as e:
            logger.error(f"LLM feedback generation failed: {e}")
            return {
                "enabled": False,
                "error": str(e),
                "message": "LLM feedback generation failed - falling back to numerical evaluation only"
            }

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Main entry point for assessment requests."""
        input_text = get_message_text(message)
        logger.info("Received assessment request")

        try:
            request = json.loads(input_text)
        except json.JSONDecodeError:
            # Check if this is a plain text submission (CSV data)
            if self._looks_like_csv(input_text):
                await self._evaluate_csv_submission(input_text, updater)
                return
            
            logger.error("Failed to parse request as JSON")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=(
                    "Error: Expected JSON request or CSV data submission.\n\n"
                    "For task request, send JSON with:\n"
                    "  {'participants': {'data_integrator': '<purple_agent_url>'}, 'config': {}}\n\n"
                    "Or submit CSV data directly for evaluation."
                )))],
                name="Error",
            )
            return

        # Handle different request types
        action = request.get("action", "evaluate")
        
        if action == "get_task":
            await self._handle_get_task(request, updater)
        elif action == "submit_result":
            await self._handle_submit_result(request, updater)
        elif action == "evaluate":
            await self._handle_evaluate_agent(request, updater)
        else:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Unknown action: {action}"))],
                name="Error",
            )

    async def _handle_get_task(self, request: dict, updater: TaskUpdater) -> None:
        """Handle request to get the benchmark task information."""
        mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:9009/mcp")
        
        task_info = {
            "task_name": "TPC-DI Data Integration Benchmark",
            "mcp_server_url": mcp_url,
            "instructions": """
Merge and aggregate data from the following source files to produce a customer-centric summary.

## Source Files (fetch from MCP server):
1. customers_tpcdi_lite_v3.csv - Customer demographics
2. accounts_tpcdi_lite_v3.csv - Account information (links to customers via customer_id)
3. trades_tpcdi_lite_v3.csv - Trade transactions (links to accounts via account_id)
4. prospect_tpcdi_lite_v3.xlsx - Prospect data (optional)
5. finwire_tpcdi_lite_v3.xml - Financial wire data (optional)

## Task:
Join the tables and compute per-customer aggregations:
- customer_id: The customer identifier
- customer_name: First name + " " + Last name
- country: Customer's country
- num_accounts: Count of accounts per customer
- total_balance: Sum of account balances
- num_trades: Count of COMPLETED trades (trade_status = 'CMPT')
- total_trade_volume: Sum of trade quantities for completed trades
- total_trade_value: Sum of (quantity * trade_price) for completed trades
- symbols_traded: Comma-separated unique stock symbols traded

## MCP Resources:
- Use 'get_task_info' tool for detailed schema
- Use 'list_files' tool to see available files
- Use 'get_file_content' tool to fetch file contents

## Submission:
Send your result as JSON with action='submit_result' and 'data' containing CSV string or list of records.
""",
            "expected_columns": [
                "customer_id", "customer_name", "country", "num_accounts",
                "total_balance", "num_trades", "total_trade_volume",
                "total_trade_value", "symbols_traded"
            ]
        }
        
        await updater.add_artifact(
            parts=[Part(root=DataPart(data=task_info))],
            name="Task Information",
        )

    async def _handle_submit_result(self, request: dict, updater: TaskUpdater) -> None:
        """Handle submission of integration results for evaluation."""
        data = request.get("data")
        
        if not data:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: No 'data' field in submission"))],
                name="Error",
            )
            return
        
        try:
            # Parse submitted data
            if isinstance(data, str):
                # CSV string
                submitted_df = pd.read_csv(StringIO(data))
            elif isinstance(data, list):
                # List of records
                submitted_df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Single record or column-oriented dict
                if all(isinstance(v, list) for v in data.values()):
                    submitted_df = pd.DataFrame(data)
                else:
                    submitted_df = pd.DataFrame([data])
            else:
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text=f"Error: Invalid data format: {type(data)}"))],
                    name="Error",
                )
                return
            
            # Step 1: Deterministic numerical evaluation
            score, details = self.evaluate_submission(submitted_df)
            
            # Step 2: LLM-powered qualitative feedback
            llm_feedback = await self.generate_llm_feedback(submitted_df, score, details)
            
            result = {
                "score": score,
                "max_score": 100,
                "details": details,
                "llm_feedback": llm_feedback,
                "submitted_rows": len(submitted_df),
                "expected_rows": len(self.ground_truth),
            }
            
            await updater.add_artifact(
                parts=[Part(root=DataPart(data=result))],
                name="Evaluation Result",
            )
            
        except Exception as e:
            logger.error(f"Error evaluating submission: {e}")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Error evaluating submission: {e}"))],
                name="Error",
            )

    async def _handle_evaluate_agent(self, request: dict, updater: TaskUpdater) -> None:
        """Handle full evaluation flow with a Purple Agent."""
        participants = request.get("participants", {})
        config = request.get("config", {})
        
        purple_url = (
            participants.get("data_integrator") or 
            participants.get("purple") or
            participants.get("agent")
        )
        
        if not purple_url:
            # No purple agent specified, return task info
            await self._handle_get_task(request, updater)
            return
        
        logger.info(f"Evaluating Purple Agent at: {purple_url}")
        
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Sending task to Purple Agent...")
        )
        
        # Get MCP URL
        mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:9009/mcp")
        
        # Send task to Purple Agent
        task_message = json.dumps({
            "action": "data_integration",
            "mcp_server_url": mcp_url,
            "task": """
Perform TPC-DI data integration:
1. Connect to the MCP server at the provided URL
2. Fetch source files: customers_tpcdi_lite_v3.csv, accounts_tpcdi_lite_v3.csv, trades_tpcdi_lite_v3.csv
3. Join tables: accounts.customer_id -> customers.customer_id, trades.account_id -> accounts.account_id
4. Filter to completed trades only (trade_status = 'CMPT')
5. Aggregate per customer:
   - customer_name = first_name + " " + last_name
   - country
   - num_accounts = count of accounts
   - total_balance = sum of account balances
   - num_trades = count of completed trades
   - total_trade_volume = sum of trade quantities
   - total_trade_value = sum of (quantity * trade_price)
   - symbols_traded = comma-separated unique symbols
6. Return result as CSV string
""",
            "expected_columns": [
                "customer_id", "customer_name", "country", "num_accounts",
                "total_balance", "num_trades", "total_trade_volume",
                "total_trade_value", "symbols_traded"
            ]
        })
        
        try:
            timeout = config.get("timeout", 300)
            response_text = await self.messenger.talk_to_agent(
                message=task_message,
                url=purple_url,
                new_conversation=True,
                timeout=timeout
            )
        except Exception as e:
            logger.error(f"Error communicating with Purple Agent: {e}")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Error communicating with Purple Agent: {e}"))],
                name="Error",
            )
            return
        
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Evaluating response...")
        )
        
        # Parse and evaluate response
        try:
            response = json.loads(response_text)
            data = response.get("data") or response.get("result") or response.get("csv")
        except json.JSONDecodeError:
            # Try to extract CSV directly
            data = self._extract_csv(response_text)
        
        if not data:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: Could not extract data from Purple Agent response"))],
                name="Error",
            )
            return
        
        # Evaluate
        try:
            if isinstance(data, str):
                submitted_df = pd.read_csv(StringIO(data))
            elif isinstance(data, list):
                submitted_df = pd.DataFrame(data)
            else:
                submitted_df = pd.DataFrame(data)
            
            # Step 1: Deterministic numerical evaluation
            score, details = self.evaluate_submission(submitted_df)
            
            # Step 2: LLM-powered qualitative feedback
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Generating AI-powered feedback...")
            )
            llm_feedback = await self.generate_llm_feedback(submitted_df, score, details)
            
            result = {
                "score": score,
                "max_score": 100,
                "details": details,
                "llm_feedback": llm_feedback,
                "purple_agent_url": purple_url,
                "submitted_rows": len(submitted_df),
                "expected_rows": len(self.ground_truth),
            }
            
            await updater.add_artifact(
                parts=[Part(root=DataPart(data=result))],
                name="Evaluation Result",
            )
            
        except Exception as e:
            logger.error(f"Error evaluating Purple Agent response: {e}")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Error evaluating response: {e}"))],
                name="Error",
            )

    async def _evaluate_csv_submission(self, csv_text: str, updater: TaskUpdater) -> None:
        """Evaluate a direct CSV submission."""
        try:
            submitted_df = pd.read_csv(StringIO(csv_text))
            
            # Step 1: Deterministic numerical evaluation
            score, details = self.evaluate_submission(submitted_df)
            
            # Step 2: LLM-powered qualitative feedback
            llm_feedback = await self.generate_llm_feedback(submitted_df, score, details)
            
            result = {
                "score": score,
                "max_score": 100,
                "details": details,
                "llm_feedback": llm_feedback,
                "submitted_rows": len(submitted_df),
                "expected_rows": len(self.ground_truth),
            }
            
            await updater.add_artifact(
                parts=[Part(root=DataPart(data=result))],
                name="Evaluation Result",
            )
        except Exception as e:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Error parsing CSV: {e}"))],
                name="Error",
            )

    def evaluate_submission(self, submitted: pd.DataFrame) -> tuple[int, dict]:
        """
        Evaluate submitted data against ground truth.
        
        Returns (score, details) where score is 0-100.
        """
        ground_truth = self.ground_truth
        details = {}
        total_score = 0
        
        # 1. Column check (20 points)
        expected_cols = set(ground_truth.columns)
        submitted_cols = set(submitted.columns)
        
        missing_cols = expected_cols - submitted_cols
        extra_cols = submitted_cols - expected_cols
        matching_cols = expected_cols & submitted_cols
        
        col_score = int(20 * len(matching_cols) / len(expected_cols)) if expected_cols else 0
        total_score += col_score
        
        details["columns"] = {
            "score": col_score,
            "max": 20,
            "expected": list(expected_cols),
            "submitted": list(submitted_cols),
            "missing": list(missing_cols),
            "extra": list(extra_cols),
        }
        
        if not matching_cols or "customer_id" not in matching_cols:
            details["error"] = "Missing required columns including customer_id"
            return total_score, details
        
        # 2. Row count check (10 points)
        row_diff = abs(len(submitted) - len(ground_truth))
        if row_diff == 0:
            row_score = 10
        elif row_diff <= 5:
            row_score = 8
        elif row_diff <= 10:
            row_score = 5
        elif row_diff <= 20:
            row_score = 2
        else:
            row_score = 0
        
        total_score += row_score
        details["row_count"] = {
            "score": row_score,
            "max": 10,
            "expected": len(ground_truth),
            "submitted": len(submitted),
            "difference": row_diff,
        }
        
        # 3. Customer coverage (15 points)
        gt_customers = set(ground_truth["customer_id"])
        sub_customers = set(submitted["customer_id"])
        
        matching_customers = gt_customers & sub_customers
        coverage = len(matching_customers) / len(gt_customers) if gt_customers else 0
        coverage_score = int(15 * coverage)
        total_score += coverage_score
        
        details["customer_coverage"] = {
            "score": coverage_score,
            "max": 15,
            "expected_customers": len(gt_customers),
            "submitted_customers": len(sub_customers),
            "matching_customers": len(matching_customers),
            "coverage_pct": round(coverage * 100, 2),
        }
        
        if not matching_customers:
            details["error"] = "No matching customers found"
            return total_score, details
        
        # Merge for detailed comparison
        try:
            merged = pd.merge(
                ground_truth,
                submitted,
                on="customer_id",
                how="inner",
                suffixes=("_expected", "_submitted")
            )
        except Exception as e:
            details["merge_error"] = str(e)
            return total_score, details
        
        # 4. Numeric accuracy (40 points total)
        numeric_cols = ["num_accounts", "total_balance", "num_trades", "total_trade_volume", "total_trade_value"]
        numeric_score = 0
        numeric_details = {}
        
        for col in numeric_cols:
            col_exp = f"{col}_expected"
            col_sub = f"{col}_submitted"
            
            if col_exp not in merged.columns or col_sub not in merged.columns:
                numeric_details[col] = {"error": "Column missing from one side"}
                continue
            
            try:
                exp_vals = pd.to_numeric(merged[col_exp], errors="coerce")
                sub_vals = pd.to_numeric(merged[col_sub], errors="coerce")
                
                # Calculate accuracy
                if col in ["num_accounts", "num_trades", "total_trade_volume"]:
                    # Integer columns - exact match percentage
                    exact_matches = (exp_vals == sub_vals).sum()
                    accuracy = exact_matches / len(merged) if len(merged) > 0 else 0
                else:
                    # Float columns - tolerance-based (within 1%)
                    tolerance = 0.01
                    close_matches = (abs(exp_vals - sub_vals) <= abs(exp_vals) * tolerance + 0.01).sum()
                    accuracy = close_matches / len(merged) if len(merged) > 0 else 0
                
                col_score = int(8 * accuracy)  # 8 points per column
                numeric_score += col_score
                
                numeric_details[col] = {
                    "accuracy_pct": round(accuracy * 100, 2),
                    "score": col_score,
                    "max": 8,
                }
            except Exception as e:
                numeric_details[col] = {"error": str(e)}
        
        total_score += numeric_score
        details["numeric_accuracy"] = {
            "score": numeric_score,
            "max": 40,
            "columns": numeric_details,
        }
        
        # 5. String fields (15 points)
        string_score = 0
        string_details = {}
        
        # Customer name check
        if "customer_name_expected" in merged.columns and "customer_name_submitted" in merged.columns:
            name_matches = (
                merged["customer_name_expected"].str.lower().str.strip() == 
                merged["customer_name_submitted"].str.lower().str.strip()
            ).sum()
            name_accuracy = name_matches / len(merged) if len(merged) > 0 else 0
            name_score = int(5 * name_accuracy)
            string_score += name_score
            string_details["customer_name"] = {
                "accuracy_pct": round(name_accuracy * 100, 2),
                "score": name_score,
                "max": 5,
            }
        
        # Country check
        if "country_expected" in merged.columns and "country_submitted" in merged.columns:
            country_matches = (
                merged["country_expected"].str.lower().str.strip() == 
                merged["country_submitted"].str.lower().str.strip()
            ).sum()
            country_accuracy = country_matches / len(merged) if len(merged) > 0 else 0
            country_score = int(5 * country_accuracy)
            string_score += country_score
            string_details["country"] = {
                "accuracy_pct": round(country_accuracy * 100, 2),
                "score": country_score,
                "max": 5,
            }
        
        # Symbols traded check (partial match)
        if "symbols_traded_expected" in merged.columns and "symbols_traded_submitted" in merged.columns:
            symbol_scores = []
            for _, row in merged.iterrows():
                exp_symbols = set(str(row.get("symbols_traded_expected", "")).split(","))
                sub_symbols = set(str(row.get("symbols_traded_submitted", "")).split(","))
                exp_symbols = {s.strip().upper() for s in exp_symbols if s.strip()}
                sub_symbols = {s.strip().upper() for s in sub_symbols if s.strip()}
                
                if exp_symbols:
                    overlap = len(exp_symbols & sub_symbols) / len(exp_symbols)
                    symbol_scores.append(overlap)
            
            symbol_accuracy = sum(symbol_scores) / len(symbol_scores) if symbol_scores else 0
            symbol_score = int(5 * symbol_accuracy)
            string_score += symbol_score
            string_details["symbols_traded"] = {
                "accuracy_pct": round(symbol_accuracy * 100, 2),
                "score": symbol_score,
                "max": 5,
            }
        
        total_score += string_score
        details["string_accuracy"] = {
            "score": string_score,
            "max": 15,
            "fields": string_details,
        }
        
        return min(100, total_score), details

    def _looks_like_csv(self, text: str) -> bool:
        """Check if text looks like CSV data."""
        lines = text.strip().split("\n")
        if len(lines) < 2:
            return False
        
        # Check if first line looks like a header with commas
        header = lines[0]
        if "," not in header:
            return False
        
        # Check if subsequent lines have similar comma count
        header_commas = header.count(",")
        for line in lines[1:min(5, len(lines))]:
            if abs(line.count(",") - header_commas) > 1:
                return False
        
        return True

    def _extract_csv(self, text: str) -> Optional[str]:
        """Try to extract CSV from text that might contain other content."""
        # Look for CSV in code blocks
        csv_match = re.search(r'```(?:csv)?\s*\n(.*?)\n```', text, re.DOTALL)
        if csv_match:
            return csv_match.group(1)
        
        # Look for lines that look like CSV
        lines = text.strip().split("\n")
        csv_lines = []
        in_csv = False
        
        for line in lines:
            if "," in line and (in_csv or self._looks_like_header(line)):
                in_csv = True
                csv_lines.append(line)
            elif in_csv and "," not in line:
                break
        
        if csv_lines:
            return "\n".join(csv_lines)
        
        return None

    def _looks_like_header(self, line: str) -> bool:
        """Check if a line looks like a CSV header."""
        parts = line.split(",")
        if len(parts) < 3:
            return False
        
        # Headers typically have lowercase/underscore names
        return all(
            re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', p.strip())
            for p in parts
        )
