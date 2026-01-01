import asyncio
import logging
import json
from typing import Callable, Any

class StreamAdapter:
    """
    IoT Adapter: Simulates MQTT/Kafka streaming connection.
    """
    def __init__(self, source_url: str, topic: str):
        self.source_url = source_url
        self.topic = topic
        self.is_running = False
        self.logger = logging.getLogger("StreamAdapter")

    async def connect(self, on_message: Callable[[float], Any]):
        """
        Connects to the stream and triggers callback on new data.
        """
        self.is_running = True
        self.logger.info(f"🔌 Connected to {self.source_url} / {self.topic}")
        
        # Simulation Loop
        import random
        import math
        t = 0
        while self.is_running:
            # Simulate sensor data (Sine wave + Noise)
            val = math.sin(t * 0.1) + random.gauss(0, 0.1)
            
            # Async Callback
            await on_message(val)
            
            t += 1
            await asyncio.sleep(0.05) # 20Hz sample rate

    def disconnect(self):
        self.is_running = False
        self.logger.info("🔌 Disconnected")
