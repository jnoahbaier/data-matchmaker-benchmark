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
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
import pandas as pd
from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger, A2ANetworkError, A2ATimeoutError, A2AConnectionError


class EvalRequest(BaseModel):
    """A2A assessment request format for AgentBeats platform."""
    participants: dict[str, HttpUrl]
    config: dict[str, Any]

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

    # Required participant roles for assessment
    required_roles: list[str] = ["data_integrator"]
    # Required config keys (optional - add if you need config params like "timeout")
    required_config_keys: list[str] = []

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

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate the assessment request has required roles and config keys."""
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing required participant roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing required config keys: {missing_config_keys}"

        return True, "ok"

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
        """Main entry point for A2A assessment requests.
        
        Expects JSON in the format:
        {
            "participants": {"data_integrator": "<purple_agent_url>"},
            "config": {}
        }
        """
        input_text = get_message_text(message)
        logger.info("Received assessment request")

        # Parse and validate the A2A assessment request using Pydantic
        try:
            request = EvalRequest.model_validate_json(input_text)
            ok, validation_msg = self.validate_request(request)
            if not ok:
                logger.error(f"Request validation failed: {validation_msg}")
                await updater.reject(new_agent_text_message(validation_msg))
                return
        except ValidationError as e:
            logger.error(f"Invalid request format: {e}")
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting assessment.\n{request.model_dump_json()}")
        )

        # Run the assessment with proper cleanup
        try:
            await self._run_assessment(request, updater)
        finally:
            self.messenger.reset()

    async def _run_assessment(self, request: EvalRequest, updater: TaskUpdater) -> None:
        """Run the full assessment flow with a Purple Agent.
        
        Args:
            request: Validated EvalRequest with participants and config
            updater: TaskUpdater for sending status updates and artifacts
        """
        purple_url = str(request.participants["data_integrator"])
        config = request.config
        
        logger.info(f"Evaluating Purple Agent at: {purple_url}")
        
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Sending task to Purple Agent at {purple_url}...")
        )
        
        # Get the file server URL (the green agent's own URL)
        # This can be configured via config, environment, or defaults to localhost
        file_server_url = (
            config.get("file_server_url") or 
            os.getenv("FILE_SERVER_URL") or 
            "http://localhost:9009"
        )
        
        # Build the task message for the Purple Agent with HTTP file download URLs
        task_message = f"""Perform TPC-DI data integration:

## Task Overview
Download source data files, join them, and compute per-customer aggregations.

## Step 1: Download Source Files
Fetch the following CSV files via HTTP GET:
- {file_server_url}/files/customers_tpcdi_lite_v3.csv
- {file_server_url}/files/accounts_tpcdi_lite_v3.csv
- {file_server_url}/files/trades_tpcdi_lite_v3.csv

You can also list all available files at: {file_server_url}/files/

## Step 2: Join Tables
- Join accounts to customers on: accounts.customer_id = customers.customer_id
- Join trades to accounts on: trades.account_id = accounts.account_id

## Step 3: Filter Data
- Only include trades with trade_status = 'CMPT' (completed trades)

## Step 4: Aggregate Per Customer
Compute the following for each customer:
- customer_id: The customer identifier
- customer_name: first_name + " " + last_name
- country: Customer's country
- num_accounts: Count of accounts owned by customer
- total_balance: Sum of all account balances
- num_trades: Count of completed trades
- total_trade_volume: Sum of trade quantities
- total_trade_value: Sum of (quantity * trade_price)
- symbols_traded: Comma-separated list of unique stock symbols traded

## Step 5: Return Result
Return your result as CSV data with these columns:
customer_id, customer_name, country, num_accounts, total_balance, num_trades, total_trade_volume, total_trade_value, symbols_traded"""

        try:
            timeout = int(config.get("timeout", 300))
            response_text = await self.messenger.talk_to_agent(
                message=task_message,
                url=purple_url,
                new_conversation=True,
                timeout=timeout
            )
        except A2ATimeoutError as e:
            logger.error(f"Purple Agent timed out: {e}")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Purple Agent timed out after {timeout}s. The agent may be overloaded or unresponsive."))],
                name="Error",
            )
            return
        except A2AConnectionError as e:
            logger.error(f"Failed to connect to Purple Agent: {e}")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Failed to connect to Purple Agent at {purple_url}. Please verify the agent is running and accessible."))],
                name="Error",
            )
            return
        except A2ANetworkError as e:
            logger.error(f"Network error with Purple Agent: {e}")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Network error communicating with Purple Agent: {e}"))],
                name="Error",
            )
            return
        except Exception as e:
            logger.error(f"Unexpected error communicating with Purple Agent: {e}")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Error communicating with Purple Agent: {e}"))],
                name="Error",
            )
            return
        
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Purple Agent responded. Evaluating response...")
        )
        
        # Parse the response - try JSON first, then CSV extraction
        data = None
        try:
            response = json.loads(response_text)
            data = response.get("data") or response.get("result") or response.get("csv")
        except json.JSONDecodeError:
            # Try to extract CSV directly from the response text
            data = self._extract_csv(response_text)
        
        if not data:
            # If no structured data, treat the entire response as potential CSV
            if self._looks_like_csv(response_text):
                data = response_text
            else:
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text="Error: Could not extract data from Purple Agent response"))],
                    name="Error",
                )
                return
        
        # Evaluate the submission
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
                parts=[
                    Part(root=TextPart(text=f"Assessment complete. Score: {score}/100")),
                    Part(root=DataPart(data=result)),
                ],
                name="Evaluation Result",
            )
            
        except Exception as e:
            logger.error(f"Error evaluating Purple Agent response: {e}")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Error evaluating response: {e}"))],
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
