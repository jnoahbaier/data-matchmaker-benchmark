# Schema Merging Benchmark - Green Agent

A Green Agent for the AgentBeats competition that evaluates Purple Agents on their ability to merge database schemas, identify primary/foreign keys, and resolve naming inconsistencies.

## What This Benchmark Evaluates

This benchmark tests an agent's data engineering capabilities:

| Skill | Description | Points |
|-------|-------------|--------|
| **Primary Key Identification** | Correctly identify the primary key for each table | 25 |
| **Join Column Detection** | Find columns that can be used to join tables | 25 |
| **Naming Inconsistency Detection** | Identify naming inconsistencies (case, conventions) | 25 |
| **Schema Merging** | Produce a unified schema with normalized column names | 25 |

## Difficulty Levels

| Level | Tables | Challenges |
|-------|--------|------------|
| **Easy** | 2 | Obvious keys, single case inconsistency (`cust_id` vs `customer_ID`) |
| **Medium** | 3 | Mixed conventions (snake_case, camelCase, UPPER_CASE) |
| **Hard** | 5 | Complex relationships, ambiguous columns, self-referential joins |

## Quick Start

### Prerequisites

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <your-repo-url>
cd data-matchmaker-benchmark
uv sync
```

### Running the Benchmark

**Terminal 1 - Start the Green Agent (evaluator):**
```bash
uv run src/server.py --port 9009
```

**Terminal 2 - Start the Mock Purple Agent (for testing):**
```bash
# Copy and configure your Gemini API key
cp sample.env .env
# Edit .env and add your GOOGLE_API_KEY

uv run src/mock_purple.py --port 9010
```

**Terminal 3 - Run an evaluation:**
```bash
# Easy difficulty
curl -s -X POST http://localhost:9009/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "message/send", "id": "1", "params": {"message": {"kind": "message", "role": "user", "parts": [{"kind": "text", "text": "{\"participants\": {\"schema_merger\": \"http://localhost:9010\"}, \"config\": {\"difficulty\": \"easy\"}}"}], "messageId": "test"}}}' | python3 -m json.tool

# Medium difficulty
curl -s -X POST http://localhost:9009/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "message/send", "id": "1", "params": {"message": {"kind": "message", "role": "user", "parts": [{"kind": "text", "text": "{\"participants\": {\"schema_merger\": \"http://localhost:9010\"}, \"config\": {\"difficulty\": \"medium\"}}"}], "messageId": "test"}}}' | python3 -m json.tool

# Hard difficulty
curl -s -X POST http://localhost:9009/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "message/send", "id": "1", "params": {"message": {"kind": "message", "role": "user", "parts": [{"kind": "text", "text": "{\"participants\": {\"schema_merger\": \"http://localhost:9010\"}, \"config\": {\"difficulty\": \"hard\"}}"}], "messageId": "test"}}}' | python3 -m json.tool
```

## Project Structure

```
src/
├── server.py        # A2A server setup and agent card
├── agent.py         # Green agent: test generation + scoring
├── executor.py      # A2A request handling
├── messenger.py     # A2A messaging utilities
└── mock_purple.py   # Baseline purple agent for testing
```

## Input Format

The Green Agent expects a JSON message with:

```json
{
  "participants": {
    "schema_merger": "http://purple-agent-url:port"
  },
  "config": {
    "difficulty": "easy|medium|hard"
  }
}
```

## Output Format

The Green Agent returns an evaluation result:

```json
{
  "score": 87,
  "max_score": 100,
  "difficulty": "easy",
  "details": {
    "primary_keys": {"score": 25, "max": 25, "detail": "2/2 tables correct"},
    "join_columns": {"score": 25, "max": 25, "detail": "1/1 join relationships found"},
    "inconsistencies": {"score": 12, "max": 25, "detail": "Found 1 inconsistencies"},
    "merged_schema": {"score": 25, "max": 25, "detail": "6/6 expected columns present"}
  },
  "test_case": { ... },
  "purple_response": { ... }
}
```

## Purple Agent Requirements

Purple agents must return JSON with this structure:

```json
{
  "primary_keys": {"table_name": "column_name", ...},
  "join_columns": [["table1.col", "table2.col"], ...],
  "inconsistencies": ["description of inconsistency", ...],
  "merged_schema": {"unified_table_name": ["col1", "col2", ...]}
}
```

## Running with Docker

```bash
# Build the green agent
docker build -t schema-evaluator .

# Run
docker run -p 9009:9009 schema-evaluator
```

## Running Tests

```bash
uv sync --extra test
uv run pytest tests/
```

## Competition Info

This is a submission for the [AgentBeats Competition](https://rdi.berkeley.edu/agentx) Phase 1 (Green Agent).

- **Track**: Data Engineering / Other Agent
- **Benchmark Type**: New benchmark
- **Skills Tested**: Schema analysis, key identification, naming convention detection

## License

MIT
