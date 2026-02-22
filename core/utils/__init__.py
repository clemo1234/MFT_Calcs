import sympy as sp
import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution
from joblib import Parallel, delayed
from tqdm import tqdm
import json
from pathlib import Path
#from .expansion import h, k, beta, gamma


dimension = 4

BASE_DIR = Path(__file__).resolve().parent


json_path = BASE_DIR.parent / "group_files" / "groups.json"
temp_path = BASE_DIR.parent / "temp.json"

print(BASE_DIR.parent / "group_files" / "groups.json")

CONFIG = "BT"



try:
    with open(temp_path, "r+") as file:
        data = json.load(file)
        NUM_RUNS = data["NUM_RUNS"] + 1
        data["NUM_RUNS"] = NUM_RUNS
        file.seek(0)
        #data.update({"NUM_RUNS" : NUM_RUNS})
        json.dump(data, file)
    
    
except FileNotFoundError:
    print("Error: The file 'temp.json' was not found.")
    
try:
    with open(json_path, "r") as file:
        data = json.load(file)
    FUNDAMENTAL_REP = np.array(data[CONFIG]["FUNDAMENTAL_REP"])
    ADJOINT_REP = np.array(data[CONFIG]["ADJOINT_REP"])
    SIZE = np.array(data[CONFIG]["SIZE"])
    
except FileNotFoundError:
    print("Error: The file 'groups.json' was not found.")    

    
print(FUNDAMENTAL_REP, SIZE)