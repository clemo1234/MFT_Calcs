from utils import sp, adjoint_dim


h, k, beta, gamma = sp.symbols(("h","\kappa",r"\beta", r"\gamma"))

def mft_expansion(group_elements_f, group_elements_a, group_n):
    print(f"Adjoint thing: {adjoint_dim}")
    expr = 0
    for expr_f, expr_a, size in zip(group_elements_f, group_elements_a, group_n):
        expr += size * sp.exp((h*sp.re(expr_f)/4 + k*sp.re(expr_a)/adjoint_dim))
        

    
    c_out = sp.ln(sp.simplify(expr))
    
    t_out = sp.diff(c_out, h)
    
    u_out = sp.diff(c_out, k)
    
    return c_out, t_out, u_out