"""Pull Danish electricity data from Energi Data Service (Energinet) and run quick checks.

Run:   python fetch_eds.py            (skips files already in ./data/)
       python fetch_eds.py --force    (re-download everything)
Needs: pip install requests pandas matplotlib
Output: CSVs and overview.png in ./data/
The script waits between requests and retries on HTTP 429 (rate limit), so it can take a few minutes.
"""
import json
import os
import sys
import time

import pandas as pd
import requests

BASE = "https://api.energidataservice.dk/dataset"
OUT = "data"
MIN_GAP = 2.0  # seconds between any two requests
_last = [0.0]


