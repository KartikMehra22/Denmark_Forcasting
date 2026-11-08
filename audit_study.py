"""Independent numerical, provenance and information-cutoff audit."""
from pathlib import Path
from datetime import timedelta
import hashlib,json
import numpy as np,pandas as pd
from controlled_study import source,samples,OUT,ROOT
findings=[]
