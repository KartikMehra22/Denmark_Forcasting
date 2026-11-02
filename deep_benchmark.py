"""Fixed-origin PatchTST and Chronos-Bolt-tiny comparisons. No test tuning."""
from pathlib import Path
from datetime import timedelta
import os,json
ROOT=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(ROOT/'.cache'/'huggingface')
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import numpy as np
import pandas as pd
import torch
from transformers import PatchTSTConfig,PatchTSTForPrediction
from chronos import ChronosBoltPipeline
from day_ahead_study import load_area

OUT=ROOT/'results'/'benchmark_v2'
torch.set_num_threads(2)
torch.manual_seed(42);np.random.seed(42)
CONFIG=dict(context_length=168,prediction_length=25,patch_length=16,patch_stride=8,
 d_model=32,num_hidden_layers=2,num_attention_heads=4,ffn_dim=64,num_input_channels=1,
 scaling='std',loss='mse',head_dropout=0.1)

