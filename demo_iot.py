import asyncio
import requests
import logging
from adapters.stream_adapter import StreamAdapter

# Configuration
API_URL = "http://localhost:8000/predict"
BROKER = "mqtt://iot.factory.local"
TOPIC = "sensors/vibration/motor_1"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IoT_Demo")

# Buffer for sliding window
buffer = []
WINDOW_SIZE = 50

async def process_sensor_data(value: float):
    """Callback triggered when new data arrives from MQTT."""
    global buffer
    buffer.append([value])
    
    # Keep buffer at window size
    if len(buffer) > WINDOW_SIZE:
        buffer.pop(0)
        
    if len(buffer) == WINDOW_SIZE:
        # Send to AI Engine
        try:
            payload = {"inputs": [buffer], "return_repr": False}
            # Note: In production, use aiohttp for async requests
            # Here we use synchronous requests for simplicity
            resp = requests.post(API_URL, json=payload)
            
            if resp.status_code == 200:
                anomaly_score = resp.json()["predictions"][0][0]
                status = "🟢 NORMAL" if anomaly_score < 0.8 else "🔴 ANOMALY"
                print(f"Sensor Val: {value:.2f} | AI Score: {anomaly_score:.2f} | Status: {status}")
        except Exception as e:
            logger.error(f"AI Engine Error: {e}")

async def main():
    print(f"🚀 Starting Industrial IoT Monitor...")
    adapter = StreamAdapter(BROKER, TOPIC)
    
    # Run for 10 seconds then stop
    stream_task = asyncio.create_task(adapter.connect(process_sensor_data))
    await asyncio.sleep(10)
    adapter.disconnect()
    await stream_task

if __name__ == "__main__":
    asyncio.run(main())
