import asyncio
import json
import logging
from uuid import uuid4

import httpx
from a2a.client import (
    A2ACardResolver,
    ClientConfig,
    ClientFactory,
    Consumer,
)
from a2a.types import (
    Message,
    Part,
    Role,
    TextPart,
    DataPart,
)


logger = logging.getLogger("messenger")

DEFAULT_TIMEOUT = 300
DEFAULT_MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.0


class A2ANetworkError(Exception):
    """Base exception for A2A network errors."""
    pass


class A2ATimeoutError(A2ANetworkError):
    """Exception for operation timeouts."""
    pass


class A2AConnectionError(A2ANetworkError):
    """Exception for connection failures."""
    pass


def create_message(
    *, role: Role = Role.user, text: str, context_id: str | None = None
) -> Message:
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def merge_parts(parts: list[Part]) -> str:
    chunks = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(json.dumps(part.root.data, indent=2))
    return "\n".join(chunks)


async def send_message(
    message: str,
    base_url: str,
    context_id: str | None = None,
    streaming: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    consumer: Consumer | None = None,
):
    """Returns dict with context_id, response and status (if exists).
    
    Raises:
        A2ATimeoutError: If the request times out
        A2AConnectionError: If connection to the agent fails
        A2ANetworkError: For other network-related errors
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
            agent_card = await resolver.get_agent_card()
            config = ClientConfig(
                httpx_client=httpx_client,
                streaming=streaming,
            )
            factory = ClientFactory(config)
            client = factory.create(agent_card)
            if consumer:
                await client.add_event_consumer(consumer)

            outbound_msg = create_message(text=message, context_id=context_id)
            last_event = None
            outputs = {"response": "", "context_id": None}

            # if streaming == False, only one event is generated
            async for event in client.send_message(outbound_msg):
                last_event = event

            match last_event:
                case Message() as msg:
                    outputs["context_id"] = msg.context_id
                    outputs["response"] += merge_parts(msg.parts)

                case (task, update):
                    outputs["context_id"] = task.context_id
                    outputs["status"] = task.status.state.value
                    msg = task.status.message
                    if msg:
                        outputs["response"] += merge_parts(msg.parts)
                    if task.artifacts:
                        for artifact in task.artifacts:
                            outputs["response"] += merge_parts(artifact.parts)

                case _:
                    pass

            return outputs
            
    except httpx.TimeoutException as e:
        raise A2ATimeoutError(f"Timeout connecting to {base_url}: {e}") from e
    except httpx.ConnectError as e:
        raise A2AConnectionError(f"Failed to connect to {base_url}: {e}") from e
    except httpx.HTTPStatusError as e:
        raise A2ANetworkError(f"HTTP error from {base_url}: {e.response.status_code}") from e
    except httpx.RequestError as e:
        raise A2ANetworkError(f"Network error communicating with {base_url}: {e}") from e


class Messenger:
    def __init__(self):
        self._context_ids = {}

    async def talk_to_agent(
        self,
        message: str,
        url: str,
        new_conversation: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> str:
        """
        Communicate with another agent by sending a message and receiving their response.

        Args:
            message: The message to send to the agent
            url: The agent's URL endpoint
            new_conversation: If True, start fresh conversation; if False, continue existing conversation
            timeout: Timeout in seconds for the request (default: 300)
            max_retries: Maximum number of retries for transient failures (default: 2)

        Returns:
            str: The agent's response message
            
        Raises:
            A2ATimeoutError: If all retry attempts timeout
            A2AConnectionError: If connection fails after all retries
            A2ANetworkError: For other network errors
            RuntimeError: If the agent returns a non-completed status
        """
        last_error: Exception | None = None
        
        for attempt in range(max_retries + 1):
            try:
                outputs = await send_message(
                    message=message,
                    base_url=url,
                    context_id=None if new_conversation else self._context_ids.get(url, None),
                    timeout=timeout,
                )
                
                if outputs.get("status", "completed") != "completed":
                    raise RuntimeError(f"{url} responded with: {outputs}")
                    
                self._context_ids[url] = outputs.get("context_id", None)
                return outputs["response"]
                
            except (A2ATimeoutError, A2AConnectionError) as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {url}: {e}. Retrying..."
                    )
                    await asyncio.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
                    continue
                raise
                
            except A2ANetworkError:
                # Non-retryable network errors
                raise
        
        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise RuntimeError("Unexpected error in talk_to_agent")

    def reset(self):
        self._context_ids = {}
