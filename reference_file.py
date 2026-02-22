import sympy as sp
import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution
from joblib import Parallel, delayed
from tqdm import tqdm

h, k, beta, gamma = sp.symbols(("h","\kappa",r"\beta", "\gamma"))

d = 4

N = 6

def funcs(group_elements, group_n):
    expr = 0
    for expr_i, size in zip(group_elements, group_n):
        expr += 1 * size * sp.exp((h*sp.re(expr_i) + k*sp.re(expr_i**2))/3)
        

    
    c_out = sp.ln(sp.simplify(expr))
    
    t_out = sp.diff(c_out, h)
    
    u_out = sp.diff(c_out, k)
    
    return c_out, t_out, u_out

ns = np.arange(0,N)

# g_test = np.exp(2*np.pi*1j*ns/N)
# g_n = np.ones(N)

def w(n):
    return (sp.exp(2*sp.pi*sp.I/3))**n

def p(n):
    return (sp.exp(2*sp.pi*sp.I/9))**n

def s(n):
    return (p(n)*(1+2*w(n)))**n

u1 =(1-sp.sqrt(5))/2

u2 = (1+sp.sqrt(5))/2

Sigma_F_108  = [3,3*w(1),3*w(2),0,0,-1,-w(1),-w(2),1,w(1),w(2),1,w(1),w(2)]
S_Sigma_F_108 = [1,1,1,12,12,9,9,9,9,9,9,9,9,9]

Sigma_F_216  = [3,3*w(1),3*w(2),0,-1,-w(1),-w(2),1,w(1),w(2),1,w(1),w(2),w(1),w(2),1]
S_Sigma_F_216 = [1,1,1,24,9,9,9,18,18,18,18,18,18,18,18,18]

Sigma_F_648  = [3,3*w(1),3*w(2),0,-1,-w(1),-w(2),1,w(1),w(2),0,0,w(2)*sp.conjugate(s(1)),sp.conjugate(s(1)),
            w(1)*sp.conjugate(s(1)),s(1),w(1)*s(1),w(2)*s(1),-p(2),-sp.conjugate(p(4)),-sp.conjugate(p(1)),-sp.conjugate(p(2)),-p(1),-p(4)]
S_Sigma_F_648 = [1,1,1,24,9,9,9,54,54,54,72,72,12,12,12,12,12,12,36,36,36,36,36,36]

Sigma_F_1080  = [3,u2,1,-sp.conjugate(w(1)),-w(1),u1*sp.conjugate(w(1)),u1*w(1),0,0,w(1),sp.conjugate(w(1)),u1,u2*w(1),u2*sp.conjugate(w(1)),-1,3*w(1),3*sp.conjugate(w(1))]
S_Sigma_F_1080 = [1,72,90,45,45,72,72,120,120,90,90,72,72,72,45,1,1]

g_test = Sigma_F_108
g_n = S_Sigma_F_108

def F_func2(h,k,beta,gamma):
    return (-d*funcs(g_test, g_n)[0] + d*funcs(g_test, g_n)[1]*(h - (1/2)*(d-1)*beta*funcs(g_test, g_n)[1]**3) + d*funcs(g_test, g_n)[2]*(k - (1/2)*(d-1)*gamma*funcs(g_test, g_n)[2]**3))    
    

F = F_func2(h,k,beta,gamma)

U = funcs(g_test, g_n)[2]    
T = funcs(g_test, g_n)[1]  
F_func = sp.lambdify((h, k, beta, gamma), F)
u = sp.lambdify((h, k), U)
t = sp.lambdify((h, k), T)
def calc(beta_val):
        #beta_val = 1.2
        def critical(gamma_val):
            # 3. Fix z and t

            
            #gamma_val = 1.2
            def heat_cap(h, k):
                return u(h,k)
            def mu(h,k):
                #return (1+u(h,k) - 2*t(h,k)**2)
                return (1 - 2*u(h,k) + t(h,k)**2)
                #return  u(h,k) - t(h,k)**2
            
            def out(h, k):
                val = [h,k]
                return val

            # Define function for minimizer
            def f_to_minimize(vars):
                return F_func(*vars,beta_val,gamma_val)

            # Initial guess
            initial_guess = [1, 1]
            bounds = [(0, None),  # x ≥ 0
                    (0, None)]  

            result = differential_evolution(
                f_to_minimize,
                bounds=[(0, 50), (0, 50)],
                strategy='best1bin',
                maxiter=7000,
                popsize=250,
                tol=1e-14,
                polish=True,   # optional local refinement (uses L-BFGS-B)
            )

            # Output result
            #print("Minimum at:", result.x)
            #print("Function value:", (mu(*result.x)))
            h, k = result.x
            return [mu(h,k), heat_cap(h,k), [h,k]]


        Num_points = 195
        
        gammas = np.linspace(-0.5,0.5,Num_points)
        mus = np.zeros(Num_points)
        crits_hk = [np.array([]) for _ in range(Num_points)]
        hc_list = np.zeros(Num_points)
        for i,j in enumerate(gammas):
            mu_temp, hc_temp, crit_hk_temp = critical(j)
            mus[i] = mu_temp
            hc_list[i] = hc_temp
            crits_hk[i] = crit_hk_temp


        return mus, hc_list, crits_hk
        #plt.scatter(gammas, mus)

beta_list = np.linspace(0, 18, int(14*16))
mus_list = []
heat_caps = []
crits = []

    #gammas_list = np.linspace(-1.1,2.0,130)

    #for i in tqdm(beta_list):
    #    mus_list.append(calc(i))
        
        
results  = Parallel(n_jobs=14)(
    delayed(calc)(beta) for beta in tqdm(beta_list)
)
        
mus_list, heat_caps, crits = zip(*results)
    
