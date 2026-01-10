from typing import Any
import pytest
import httpx
from uuid import uuid4

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


# A2A validation helpers - adapted from https://github.com/a2aproject/a2a-inspector/blob/main/backend/validators.py

def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    """Validate the structure and fields of an agent card."""
    errors: list[str] = []

    # Use a frozenset for efficient checking and to indicate immutability.
    required_fields = frozenset(
        [
            'name',
            'description',
            'url',
            'version',
            'capabilities',
            'defaultInputModes',
            'defaultOutputModes',
            'skills',
        ]
    )

    # Check for the presence of all required fields
    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    # Check if 'url' is an absolute URL (basic check)
    if 'url' in card_data and not (
        card_data['url'].startswith('http://')
        or card_data['url'].startswith('https://')
    ):
        errors.append(
            "Field 'url' must be an absolute URL starting with http:// or https://."
        )

    # Check if capabilities is a dictionary
    if 'capabilities' in card_data and not isinstance(
        card_data['capabilities'], dict
    ):
        errors.append("Field 'capabilities' must be an object.")

    # Check if defaultInputModes and defaultOutputModes are arrays of strings
    for field in ['defaultInputModes', 'defaultOutputModes']:
        if field in card_data:
            if not isinstance(card_data[field], list):
                errors.append(f"Field '{field}' must be an array of strings.")
            elif not all(isinstance(item, str) for item in card_data[field]):
                errors.append(f"All items in '{field}' must be strings.")

    # Check skills array
    if 'skills' in card_data:
        if not isinstance(card_data['skills'], list):
            errors.append(
                "Field 'skills' must be an array of AgentSkill objects."
            )
        elif not card_data['skills']:
            errors.append(
                "Field 'skills' array is empty. Agent must have at least one skill if it performs actions."
            )

    return errors


def _validate_task(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'id' not in data:
        errors.append("Task object missing required field: 'id'.")
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append("Task object missing required field: 'status.state'.")
    return errors


def _validate_status_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append(
            "StatusUpdate object missing required field: 'status.state'."
        )
    return errors


def _validate_artifact_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'artifact' not in data:
        errors.append(
            "ArtifactUpdate object missing required field: 'artifact'."
        )
    elif (
        'parts' not in data.get('artifact', {})
        or not isinstance(data.get('artifact', {}).get('parts'), list)
        or not data.get('artifact', {}).get('parts')
    ):
        errors.append("Artifact object must have a non-empty 'parts' array.")
    return errors


def _validate_message(data: dict[str, Any]) -> list[str]:
    errors = []
    if (
        'parts' not in data
        or not isinstance(data.get('parts'), list)
        or not data.get('parts')
    ):
        errors.append("Message object must have a non-empty 'parts' array.")
    if 'role' not in data or data.get('role') != 'agent':
        errors.append("Message from agent must have 'role' set to 'agent'.")
    return errors


def validate_event(data: dict[str, Any]) -> list[str]:
    """Validate an incoming event from the agent based on its kind."""
    if 'kind' not in data:
        return ["Response from agent is missing required 'kind' field."]

    kind = data.get('kind')
    validators = {
        'task': _validate_task,
        'status-update': _validate_status_update,
        'artifact-update': _validate_artifact_update,
        'message': _validate_message,
    }

    validator = validators.get(str(kind))
    if validator:
        return validator(data)

    return [f"Unknown message kind received: '{kind}'."]


# A2A messaging helpers

async def send_text_message(text: str, url: str, context_id: str | None = None, streaming: bool = False):
    async with httpx.AsyncClient(timeout=10) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=context_id,
        )

        events = [event async for event in client.send_message(msg)]

    return events


# A2A conformance tests

def test_agent_card(agent):
    """Validate agent card structure and required fields."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    assert response.status_code == 200, "Agent card endpoint must return 200"

    card_data = response.json()
    errors = validate_agent_card(card_data)

    assert not errors, f"Agent card validation failed:\n" + "\n".join(errors)

@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_message(agent, streaming):
    """Test that agent returns valid A2A message format."""
    events = await send_text_message("Hello", agent, streaming=streaming)

    all_errors = []
    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)

            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
                if update:
                    errors = validate_event(update.model_dump())
                    all_errors.extend(errors)

            case _:
                pytest.fail(f"Unexpected event type: {type(event)}")

    assert events, "Agent should respond with at least one event"
    assert not all_errors, f"Message validation failed:\n" + "\n".join(all_errors)

# Custom tests for Schema Merging Benchmark

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent import Agent


class TestTestCaseGeneration:
    """Test that test cases are generated correctly."""

    def setup_method(self):
        self.agent = Agent()

    def test_easy_case_structure(self):
        """Easy case should have 2 tables with proper structure."""
        case = self.agent.generate_test_case("easy")
        
        assert "tables" in case
        assert "ground_truth" in case
        assert len(case["tables"]) == 2
        
        for table in case["tables"]:
            assert "name" in table
            assert "columns" in table
            assert "sample_data" in table

    def test_medium_case_structure(self):
        """Medium case should have 3 tables."""
        case = self.agent.generate_test_case("medium")
        assert len(case["tables"]) == 3

    def test_hard_case_structure(self):
        """Hard case should have 5 tables."""
        case = self.agent.generate_test_case("hard")
        assert len(case["tables"]) == 5

    def test_ground_truth_has_all_fields(self):
        """Ground truth should have all required fields."""
        for difficulty in ["easy", "medium", "hard"]:
            case = self.agent.generate_test_case(difficulty)
            gt = case["ground_truth"]
            
            assert "primary_keys" in gt
            assert "join_columns" in gt
            assert "inconsistencies" in gt
            assert "merged_schema" in gt


class TestPrimaryKeyScoring:
    """Test primary key identification scoring."""

    def setup_method(self):
        self.agent = Agent()

    def test_perfect_score(self):
        """All keys correct should score 25."""
        expected = {"customers": "cust_id", "orders": "order_id"}
        response = {"customers": "cust_id", "orders": "order_id"}
        
        score, detail = self.agent._score_primary_keys(response, expected)
        assert score == 25
        assert "2/2" in detail

    def test_partial_score(self):
        """One correct out of two should score ~12."""
        expected = {"customers": "cust_id", "orders": "order_id"}
        response = {"customers": "cust_id", "orders": "wrong_id"}
        
        score, detail = self.agent._score_primary_keys(response, expected)
        assert score == 12
        assert "1/2" in detail

    def test_zero_score_empty(self):
        """Empty response should score 0."""
        expected = {"customers": "cust_id"}
        response = {}
        
        score, detail = self.agent._score_primary_keys(response, expected)
        assert score == 0

    def test_case_insensitive(self):
        """Scoring should be case-insensitive."""
        expected = {"customers": "cust_id"}
        response = {"customers": "CUST_ID"}
        
        score, _ = self.agent._score_primary_keys(response, expected)
        assert score == 25


class TestJoinColumnScoring:
    """Test join column identification scoring."""

    def setup_method(self):
        self.agent = Agent()

    def test_perfect_score(self):
        """Correct join columns should score 25."""
        expected = [["customers.cust_id", "orders.customer_ID"]]
        response = [["customers.cust_id", "orders.customer_ID"]]
        
        score, detail = self.agent._score_join_columns(response, expected)
        assert score == 25

    def test_reversed_order_still_matches(self):
        """Order within pair shouldn't matter."""
        expected = [["customers.cust_id", "orders.customer_ID"]]
        response = [["orders.customer_ID", "customers.cust_id"]]
        
        score, _ = self.agent._score_join_columns(response, expected)
        assert score == 25

    def test_zero_score_empty(self):
        """Empty response should score 0."""
        expected = [["a.col", "b.col"]]
        response = []
        
        score, detail = self.agent._score_join_columns(response, expected)
        assert score == 0


class TestMergedSchemaScoring:
    """Test merged schema scoring."""

    def setup_method(self):
        self.agent = Agent()

    def test_perfect_coverage(self):
        """All columns present should score 25."""
        expected = {"merged": ["col1", "col2", "col3"]}
        response = {"output": ["col1", "col2", "col3"]}
        
        score, _ = self.agent._score_merged_schema(response, expected)
        assert score == 25

    def test_partial_coverage(self):
        """Partial column coverage should score proportionally."""
        expected = {"merged": ["col1", "col2", "col3", "col4"]}
        response = {"output": ["col1", "col2"]}
        
        score, detail = self.agent._score_merged_schema(response, expected)
        assert score == 12  # 2/4 = 50% of 25
        assert "2/4" in detail

    def test_empty_response(self):
        """Empty response should score 0."""
        expected = {"merged": ["col1"]}
        response = {}
        
        score, _ = self.agent._score_merged_schema(response, expected)
        assert score == 0


class TestInconsistencyScoring:
    """Test inconsistency detection scoring."""

    def setup_method(self):
        self.agent = Agent()

    def test_keywords_detected(self):
        """Response with relevant keywords should score points."""
        expected = ["cust_id vs customer_ID (case)"]
        response = ["The column naming uses different case conventions: cust_id and customer_ID"]
        
        score, _ = self.agent._score_inconsistencies(response, expected)
        assert score > 0

    def test_empty_response(self):
        """Empty response should score 0."""
        expected = ["some inconsistency"]
        response = []
        
        score, _ = self.agent._score_inconsistencies(response, expected)
        assert score == 0


class TestJsonExtraction:
    """Test JSON extraction from markdown responses."""

    def setup_method(self):
        self.agent = Agent()

    def test_extract_from_markdown(self):
        """Should extract JSON from markdown code block."""
        text = '```json\n{"key": "value"}\n```'
        result = self.agent._extract_json(text)
        assert result == {"key": "value"}

    def test_extract_raw_json(self):
        """Should extract raw JSON object."""
        text = 'Here is the result: {"key": "value"} and more text'
        result = self.agent._extract_json(text)
        assert result == {"key": "value"}

    def test_returns_none_for_invalid(self):
        """Should return None for text without JSON."""
        text = "No JSON here"
        result = self.agent._extract_json(text)
        assert result is None
