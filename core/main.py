from utils import minimize
from utils import sp, np, Parallel, tqdm, delayed
from utils.expansion import mft_expansion
from utils import FUNDAMENTAL_REP, ADJOINT_REP, SIZE
from utils import dimension
#from utils import h, k, beta, gamma
from utils import differential_evolution, json
from utils import CONFIG, NUM_RUNS
h, k, beta, gamma = sp.symbols(("h","\kappa",r"\beta", "\gamma"))

c, t, u = mft_expansion(FUNDAMENTAL_REP, ADJOINT_REP, SIZE)


F_Energy = (-dimension*c + dimension*t*(h - (1/2)*(dimension-1)*beta*t**3)
            + dimension*u*(k - (1/2)*(dimension-1)*gamma*u**3))


F_func = sp.lambdify((h, k, beta, gamma), F_Energy, 'numpy')
u_func = sp.lambdify((h, k), u, 'numpy')
t_func = sp.lambdify((h, k), t, 'numpy')

def calc(beta_val):
        #beta_val = 1.2
        def critical(gamma_val):
            # 3. Fix z and t

            
            #gamma_val = 1.2
            def heat_cap(h, k):
                return u_func(h,k)
            def mu(h,k):
                #return (1+u(h,k) - 2*t(h,k)**2)
                return (1 - 2*u_func(h,k) + t_func(h,k)**2)
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


        Num_points = 300
        
        gammas = np.linspace(-1.5,1.5,Num_points)
        mus = np.zeros(Num_points)
        crits_hk = [np.array([]) for _ in range(Num_points)]
        hc_list = np.zeros(Num_points)
        for i,j in enumerate(gammas):
            mu_temp, hc_temp, crit_hk_temp = critical(j)
            mus[i] = mu_temp
            hc_list[i] = hc_temp
            crits_hk[i] = crit_hk_temp


        return mus, hc_list, crits_hk, gammas
        #plt.scatter(gammas, mus)

beta_list = np.linspace(0, 6, int(14*20))
gammas = []
mus_list = []
heat_caps = []
crits = []

    #gammas_list = np.linspace(-1.1,2.0,130)

    #for i in tqdm(beta_list):
    #    mus_list.append(calc(i))
        
        
results  = Parallel(n_jobs=14)(
    delayed(calc)(beta) for beta in tqdm(beta_list)
)
        
mus_list, heat_caps, crits, gammas = zip(*results)

data_dictionary = {
    "group_name" : CONFIG,
    "beta_range" : list(beta_list),
    "gamma_range" : np.array(gammas).tolist(),
    "order_parameter" : np.array(mus_list).tolist(),
    "u_critical" : np.array(heat_caps).tolist(),
    "crits_hk" : list(crits)
}



with open(f'data_ANALYSIS_for_{CONFIG}_trialNum{NUM_RUNS}.json', 'w') as fp:
    json.dump(data_dictionary, fp, indent=4)