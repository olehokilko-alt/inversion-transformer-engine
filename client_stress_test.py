import asyncio
import httpx
import time
import json
import logging
import random
import numpy as np

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("StressTest")

BASE_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{BASE_URL}/predict"

def generate_valid_payload(batch_size=1, seq_len=50):
    """Generates a valid payload with random data."""
    data = np.random.randn(batch_size, seq_len, 1).tolist()
    return {"inputs": data, "return_repr": False}

async def test_health():
    """Checks if the server is up."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{BASE_URL}/health")
            if resp.status_code == 200:
                logger.info(f"✅ Health Check Passed: {resp.json()}")
                return True
            else:
                logger.error(f"❌ Health Check Failed: {resp.status_code}")
                return False
        except Exception as e:
            logger.critical(f"❌ Server Unreachable: {e}")
            return False

async def test_load(concurrency=50):
    """Sends concurrent requests to test stability."""
    logger.info(f"--- ⚡ LOAD TEST: {concurrency} Concurrent Requests ---")
    
    async def send_request(client):
        try:
            payload = generate_valid_payload()
            start = time.perf_counter()
            resp = await client.post(PREDICT_ENDPOINT, json=payload)
            duration = (time.perf_counter() - start) * 1000
            return resp.status_code, duration
        except:
            return 0, 0

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency)) as client:
        tasks = [send_request(client) for _ in range(concurrency)]
        results = await asyncio.gather(*tasks)
    
    successes = [r for r in results if r[0] == 200]
    avg_latency = sum(r[1] for r in successes) / len(successes) if successes else 0
    
    logger.info(f"📊 Results: {len(successes)}/{concurrency} Successful")
    logger.info(f"⏱️ Avg Latency: {avg_latency:.2f}ms")
    
    if len(successes) == concurrency:
        logger.info("✅ Load Test PASSED")
    else:
        logger.warning("⚠️ Load Test showed dropped requests")

async def test_security_injection():
    """Tests resilience against malformed inputs."""
    logger.info("--- 🛡️ SECURITY TEST: Malformed Inputs ---")
    
    malformed_payloads = [
        {"inputs": "DROP TABLE users;"}, # SQL Injection attempt
        {"inputs": 12345}, # Wrong type
        {"inputs": [[["string"]]]}, # Wrong data type inside array
        {"inputs": np.random.randn(1, 10000, 1).tolist()}, # Buffer Overflow attempt (huge sequence)
    ]
    
    async with httpx.AsyncClient() as client:
        for i, payload in enumerate(malformed_payloads):
            try:
                resp = await client.post(PREDICT_ENDPOINT, json=payload)
                if resp.status_code == 422:
                    logger.info(f"✅ Blocked Malformed Payload #{i+1} (422 Unprocessable Entity)")
                elif resp.status_code == 500:
                    logger.warning(f"⚠️ Payload #{i+1} caused 500 Error (Server Crash?)")
                else:
                    logger.info(f"ℹ️ Payload #{i+1} response: {resp.status_code}")
            except Exception as e:
                logger.error(f"❌ Connection Error on Payload #{i+1}: {e}")

async def main():
    print("\n🚀 STARTING CLIENT STRESS TEST SUITE")
    print("====================================")
    
    if await test_health():
        await test_load()
        await test_security_injection()
    else:
        print("❌ Skipping tests because server is down.")

if __name__ == "__main__":
    asyncio.run(main())
