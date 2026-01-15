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

# Custom tests for TPC-DI Data Integration Benchmark

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent import Agent, EvalRequest
from pydantic import HttpUrl


class TestEvalRequestValidation:
    """Test A2A request validation."""

    def setup_method(self):
        self.agent = Agent()

    def test_valid_request_passes(self):
        """Valid request with required role should pass validation."""
        request = EvalRequest(
            participants={"data_integrator": HttpUrl("http://127.0.0.1:9010")},
            config={"timeout": 300}
        )
        ok, msg = self.agent.validate_request(request)
        assert ok is True
        assert msg == "ok"

    def test_missing_role_fails(self):
        """Request missing required role should fail validation."""
        request = EvalRequest(
            participants={"wrong_role": HttpUrl("http://127.0.0.1:9010")},
            config={}
        )
        ok, msg = self.agent.validate_request(request)
        assert ok is False
        assert "data_integrator" in msg

    def test_required_roles_attribute(self):
        """Agent should have required_roles defined."""
        assert hasattr(self.agent, 'required_roles')
        assert "data_integrator" in self.agent.required_roles


class TestEvaluationScoring:
    """Test submission evaluation scoring."""

    def setup_method(self):
        self.agent = Agent()

    def test_evaluate_perfect_submission(self):
        """Perfect submission should score 100."""
        import pandas as pd
        # Use ground truth as submission for perfect score
        ground_truth = self.agent.ground_truth
        score, details = self.agent.evaluate_submission(ground_truth)
        assert score == 100
        assert details["columns"]["score"] == 20
        assert details["row_count"]["score"] == 10
        assert details["customer_coverage"]["score"] == 15

    def test_evaluate_empty_submission(self):
        """Empty submission should score low."""
        import pandas as pd
        empty_df = pd.DataFrame({"customer_id": []})
        score, details = self.agent.evaluate_submission(empty_df)
        assert score < 50

    def test_evaluate_missing_columns(self):
        """Submission with missing columns should lose points."""
        import pandas as pd
        partial_df = pd.DataFrame({
            "customer_id": [1, 2],
            "customer_name": ["Test 1", "Test 2"]
        })
        score, details = self.agent.evaluate_submission(partial_df)
        assert details["columns"]["missing"]
        assert details["columns"]["score"] < 20


class TestAgentInitialization:
    """Test Agent initialization."""

    def test_messenger_initialized(self):
        """Agent should have messenger attribute."""
        agent = Agent()
        assert hasattr(agent, 'messenger')
        assert agent.messenger is not None

    def test_ground_truth_loading(self):
        """Agent should load ground truth file."""
        agent = Agent()
        gt = agent.ground_truth
        assert len(gt) > 0
        assert "customer_id" in gt.columns
