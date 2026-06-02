import sympy as sp
import numpy as np
import Analysis_Tools
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution
from joblib import Parallel, delayed
from tqdm import tqdm
import json
from pathlib import Path
#from .expansion import h, k, beta, gamma


dimension = 4

mult_adjoint_idx = 0

BASE_DIR = Path(__file__).resolve().parent


json_path = BASE_DIR.parent / "group_files" / "groups.json"
temp_path = BASE_DIR.parent / "temp.json"

print(BASE_DIR.parent / "group_files" / "groups.json")

CONFIG = "A5"



try:
    """
    Opens files that keeps track of runs for organization purposes.
    Where the file name then becomes "ANALYSIS_for_{Group Name}_TrialNum{Number of Runs (lifetime)}...
    """
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
    SIZE = np.array(data[CONFIG]["SIZE"])
    if data[CONFIG]["MULT_ADJOINT"] == True:
        print((np.shape(data[CONFIG]["ADJOINT_REP"])))
        if mult_adjoint_idx in np.arange(0,len(data[CONFIG]["ADJOINT_REP"])) and len(np.shape(data[CONFIG]["ADJOINT_REP"])) > 1:
            ADJOINT_REP = np.array(data[CONFIG]["ADJOINT_REP"][mult_adjoint_idx])
            adjoint_dim = data[CONFIG]["ADJOINT_DIMS"][mult_adjoint_idx]
        else:
            raise ValueError(f"Adjoint index: {mult_adjoint_idx} is not in range.")
    else:
        ADJOINT_REP = np.array(data[CONFIG]["ADJOINT_REP"])
        try:
            adjoint_dim = data[CONFIG]["ADJOINT_DIMS"]
        except:
            print("!!!Warning: Adjoint Dimension Not Specified. Setting to 1!!!")
            adjoint_dim = 1
    try:
        adjoint_rep_label = data[CONFIG]["ADJOINT_LABEL"][mult_adjoint_idx]
    except:
        adjoint_rep_label = ADJOINT_REP.tolist()


    
    
except FileNotFoundError:
    print("Error: The file 'groups.json' was not found.")    


print(FUNDAMENTAL_REP, SIZE)