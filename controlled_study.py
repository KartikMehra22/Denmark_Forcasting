"""Matched-input, fixed-origin comparison and native quarter-hour study."""
from pathlib import Path
from datetime import timedelta
import os,json,hashlib
ROOT=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(ROOT/'.cache'/'huggingface');os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import numpy as np,pandas as pd,torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from transformers import PatchTSTConfig,PatchTSTForPrediction
from chronos import ChronosBoltPipeline
from day_ahead_study import load_area
OUT=ROOT/'results'/'controlled';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)

