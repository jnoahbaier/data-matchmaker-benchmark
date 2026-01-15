#!/usr/bin/env python3
"""
End-to-End Test Script for Data Integration Benchmark.

This script demonstrates how to:
1. Send a task to the Green Agent
2. Have it evaluate a Purple Agent
3. Get the evaluation score

Usage:
    # First, start both agents in separate terminals:
    # Terminal 1: python src/main.py --port 9009
    # Terminal 2: python src/mock_purple.py --port 9010
    
    # Then run this test:
    # python test_e2e.py
"""

import asyncio
import json
import sys
import uuid

import httpx


GREEN_AGENT_URL = "http://127.0.0.1:9009"
PURPLE_AGENT_URL = "http://127.0.0.1:9010"


async def check_agent_health(url: str, name: str) -> bool:
    """Check if an agent is running."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{url}/health", timeout=5.0)
            if resp.status_code == 200:
                print(f"✅ {name} is healthy at {url}")
                return True
    except Exception:
        pass
    print(f"❌ {name} not responding at {url}")
    return False


async def send_a2a_message(url: str, message_text: str) -> dict:
    """Send an A2A message and get the response."""
    async with httpx.AsyncClient() as client:
        # A2A uses JSON-RPC style messaging with proper message format
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "id": str(uuid.uuid4()),
            "params": {
                "message": {
                    "messageId": str(uuid.uuid4()),
                    "role": "user",
                    "parts": [{"kind": "text", "text": message_text}]
                }
            }
        }
        
        resp = await client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120.0  # Long timeout for processing
        )
        return resp.json()


async def test_direct_submission():
    """Test by directly submitting data to the Green Agent."""
    print("\n" + "="*60)
    print("TEST 1: Direct Data Submission")
    print("="*60)
    
    # Load and process data locally to simulate a Purple Agent result
    from pathlib import Path
    import pandas as pd
    
    tasks_dir = Path(__file__).parent / "jan15_tasks"
    
    # Load source files
    customers = pd.read_csv(tasks_dir / "customers_tpcdi_lite_v3.csv")
    accounts = pd.read_csv(tasks_dir / "accounts_tpcdi_lite_v3.csv")
    trades = pd.read_csv(tasks_dir / "trades_tpcdi_lite_v3.csv")
    
    # Simple integration (intentionally imperfect to show scoring)
    customer_accounts = pd.merge(customers, accounts, on="customer_id", how="left")
    completed_trades = trades[trades["trade_status"] == "CMPT"]
    
    merged = pd.merge(customer_accounts, completed_trades, on="account_id", how="left")
    merged["trade_value"] = merged["quantity"].fillna(0) * merged["trade_price"].fillna(0)
    
    # Aggregate
    result = merged.groupby("customer_id").agg(
        customer_name=("first_name", lambda x: f"{x.iloc[0]} {merged.loc[x.index[0], 'last_name']}" if len(x) > 0 else ""),
        country=("country", "first"),
        num_accounts=("account_id", "nunique"),
        total_balance=("balance", "sum"),
        num_trades=("trade_id", lambda x: x.notna().sum()),
        total_trade_volume=("quantity", lambda x: x.fillna(0).sum()),
        total_trade_value=("trade_value", "sum"),
        symbols_traded=("symbol", lambda x: ",".join(sorted(set(x.dropna().astype(str)))))
    ).reset_index()
    
    # Fix customer names
    name_map = customers.set_index("customer_id").apply(
        lambda row: f"{row['first_name']} {row['last_name']}", axis=1
    ).to_dict()
    result["customer_name"] = result["customer_id"].map(name_map)
    
    # Reorder
    result = result[[
        "customer_id", "customer_name", "country", "num_accounts",
        "total_balance", "num_trades", "total_trade_volume",
        "total_trade_value", "symbols_traded"
    ]]
    
    print(f"\n📊 Generated {len(result)} rows of integrated data")
    print(f"   Columns: {list(result.columns)}")
    
    # Submit to Green Agent for evaluation
    csv_data = result.to_csv(index=False)
    
    submission = json.dumps({
        "action": "submit_result",
        "data": csv_data
    })
    
    print(f"\n📤 Submitting to Green Agent at {GREEN_AGENT_URL}...")
    
    response = await send_a2a_message(GREEN_AGENT_URL, submission)
    
    print(f"\n📥 Response from Green Agent:")
    print(json.dumps(response, indent=2))
    
    # Extract and display score
    if "result" in response:
        try:
            result_data = response.get("result", {})
            artifacts = result_data.get("artifacts", [])
            for artifact in artifacts:
                parts = artifact.get("parts", [])
                for part in parts:
                    if "data" in part:
                        data = part["data"]
                        if isinstance(data, dict) and "score" in data:
                            print(f"\n🏆 SCORE: {data['score']}/{data.get('max_score', 100)}")
                            print(f"   Details: {json.dumps(data.get('details', {}), indent=2)}")
        except Exception as e:
            print(f"Could not parse score: {e}")
    
    return response


async def test_get_task_info():
    """Test getting task information from the Green Agent."""
    print("\n" + "="*60)
    print("TEST 2: Get Task Information")
    print("="*60)
    
    request = json.dumps({"action": "get_task"})
    
    print(f"\n📤 Requesting task info from Green Agent...")
    response = await send_a2a_message(GREEN_AGENT_URL, request)
    
    print(f"\n📥 Task Information:")
    print(json.dumps(response, indent=2))
    
    return response


async def test_full_evaluation():
    """Test full evaluation flow with Purple Agent."""
    print("\n" + "="*60)
    print("TEST 3: Full Evaluation (Green → Purple → Score)")
    print("="*60)
    
    # Check if Purple Agent is running
    purple_ok = await check_agent_health(PURPLE_AGENT_URL, "Purple Agent")
    if not purple_ok:
        print("\n⚠️  Purple Agent not running. Start it with:")
        print("    python src/mock_purple.py --port 9010")
        return None
    
    # Send evaluation request to Green Agent
    request = json.dumps({
        "action": "evaluate",
        "participants": {
            "data_integrator": PURPLE_AGENT_URL
        },
        "config": {
            "timeout": 60
        }
    })
    
    print(f"\n📤 Sending evaluation request to Green Agent...")
    print(f"   Green Agent: {GREEN_AGENT_URL}")
    print(f"   Purple Agent: {PURPLE_AGENT_URL}")
    
    response = await send_a2a_message(GREEN_AGENT_URL, request)
    
    print(f"\n📥 Evaluation Result:")
    print(json.dumps(response, indent=2))
    
    # Extract and display score
    if "result" in response:
        try:
            result_data = response.get("result", {})
            artifacts = result_data.get("artifacts", [])
            for artifact in artifacts:
                parts = artifact.get("parts", [])
                for part in parts:
                    if "data" in part:
                        data = part["data"]
                        if isinstance(data, dict) and "score" in data:
                            print(f"\n🏆 FINAL SCORE: {data['score']}/{data.get('max_score', 100)}")
        except Exception as e:
            print(f"Could not parse score: {e}")
    
    return response


async def main():
    """Run all tests."""
    print("🧪 Data Integration Benchmark - End-to-End Test")
    print("="*60)
    
    # Check Green Agent
    green_ok = await check_agent_health(GREEN_AGENT_URL, "Green Agent")
    if not green_ok:
        print("\n❌ Green Agent not running. Start it with:")
        print("    python src/main.py --port 9009")
        sys.exit(1)
    
    # Run tests
    print("\n" + "="*60)
    print("Running Tests...")
    print("="*60)
    
    # Test 1: Get task info
    await test_get_task_info()
    
    # Test 2: Direct submission
    await test_direct_submission()
    
    # Test 3: Full evaluation (only if Purple Agent is running)
    await test_full_evaluation()
    
    print("\n" + "="*60)
    print("✅ Tests Complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
