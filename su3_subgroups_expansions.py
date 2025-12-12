import numpy as np
import sympy as sp

al = sp.Symbol(r'\alpha')
b = sp.Symbol(r"\beta")
Ed = sp.Symbol("E'")
n_p = sp.Symbol("N_P")
n_l = sp.Symbol("N_l")

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


def strong_expansion(Sigma_F, S_Sigma_F):

    expr = 0
    
    for u,size in zip(Sigma_F,S_Sigma_F):
            expr = expr + size*sp.exp(b*sp.re((u))/3)

    expr = sp.ln(sp.simplify(expr))        
            
    f = sp.diff(sp.simplify(expr), b)


    #sp.N(f)
    g = sp.expand(3*sum(f.taylor_term(n, b) for n in range(4)))
    final = 0 
    #print(g)
    for i in g.as_ordered_terms():
        final += sp.nsimplify(sp.N(i), rational=True)
    expr2 = sp.simplify(final)    
    threshold = 1e-18
    expr_clean = expr2.xreplace({c: 0 for c in expr2.atoms(sp.Number) if abs(c) < threshold})
    
    return expr_clean

print("S648")
print(strong_expansion(Sigma_F_648, S_Sigma_F_648))
# print("S1080")
# print(strong_expansion(Sigma_F_1080, S_Sigma_F_1080))