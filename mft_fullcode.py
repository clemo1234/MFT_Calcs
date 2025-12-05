import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, exp, cos, log, trigsimp, nsimplify, pi, lambdify, diff, solve, N
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution


class Gauge_Group:
    def __init__(self, N, group_elements):
        self.N = N
        self.d = 4
        self.group_elements = group_elements
    def dim_change(self, d):
        self.d = d
        
        
test = Gauge_Group(3)
