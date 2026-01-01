import csv
import json
import logging
from typing import List, Generator, Dict

class CSVAdapter:
    """
    FinTech Adapter: Converts CSV historical data into model-ready batches.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.logger = logging.getLogger("CSVAdapter")

    def load_batches(self, batch_size=50, value_column="close") -> Generator[List[float], None, None]:
        """
        Reads CSV and yields batches of data.
        Assumes standard OHLCV format or similar time-series CSV.
        """
        buffer = []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        val = float(row.get(value_column, 0.0))
                        buffer.append(val)
                        
                        if len(buffer) >= batch_size:
                            yield buffer
                            # Sliding window: remove first element (step=1) or clear (step=batch_size)
                            # Here we implement sliding window with step 1 for maximum resolution
                            buffer.pop(0) 
                    except ValueError:
                        continue
        except FileNotFoundError:
            self.logger.error(f"File not found: {self.file_path}")
