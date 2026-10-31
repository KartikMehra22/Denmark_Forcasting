"""Daily price-curve models with local-calendar lags and chronological validation.
Run .venv/bin/python benchmark.py. Last 180 days are exploratory (previously inspected).
"""
from datetime import timedelta
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
import holidays
from day_ahead_study import load_area

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'benchmark_v2'
OUT.mkdir(parents=True,exist_ok=True)
HOLIDAYS=holidays.Denmark(years=range(2021,2028))

