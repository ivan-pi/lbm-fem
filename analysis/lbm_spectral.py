# Linearised von Neumann analysis of the full collide-then-stream scheme
# (D2Q9 BGK about the rest state, Q1 consistent/lumped Taylor-Galerkin streaming):
#   g^{n+1}(theta) = S(theta) [I - omega (I - J)] g^n(theta),
# S = diag(G_alpha(theta)) the streaming amplification of direction alpha,
# J_{ab} = w_a (1 + 3 e_a.e_b) the Jacobian of f^eq at rho = 1, u = 0.
import numpy as np, sys, os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tg_spectral.py')).read().split('th = np.linspace')[0])
E = [(0,0),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
W = [4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]
J = np.array([[W[a]*(1+3*(E[a][0]*E[b][0]+E[a][1]*E[b][1])) for b in range(9)] for a in range(9)])

def G_lumped(tx,ty,ex,ey,C):
    mx,cx,kx=sym1d(tx); my,cy,ky=sym1d(ty)
    return 1-(C*(ex*cx*my+ey*mx*cy)+0.5*C**2*(ex**2*kx*my+ey**2*mx*ky-2*ex*ey*cx*cy))

def rho(C, omega, scheme, n=97):
    t = np.linspace(-np.pi, np.pi, n); r = 0.
    K = np.eye(9) - omega*(np.eye(9) - J)
    for tx in t:
        for ty in t:
            if scheme == 'lumped': S = [G_lumped(tx,ty,ex,ey,C) for ex,ey in E]
            else: S = [G_2d(tx,ty,ex,ey,C, scheme=='tg3') if (ex,ey)!=(0,0) else 1. for ex,ey in E]
            r = max(r, np.abs(np.linalg.eigvals(np.diag(S) @ K)).max())
    return r

def critical(omega, scheme):
    Cs = np.arange(0.05, 1.21, 0.05); last = 0
    for C in Cs:
        if rho(C, omega, scheme) > 1 + 1e-6: break
        last = C
    return last

print('critical per-axis Courant number of the coupled scheme (rho(G) <= 1), D2Q9 BGK + Q1:')
print('  omega    dt/lambda   TG2 consistent   TG3 consistent   TG2 lumped')
for dl in (1, 2.26, 4.5, 9, 36, 200):
    om = dl/(1+dl/2)
    print(f'  {om:5.3f}   {dl:7.2f}      {critical(om,"tg2"):5.2f}            {critical(om,"tg3"):5.2f}           {critical(om,"lumped"):5.2f}')
print('pure advection (omega = 0):', f'TG2 {critical(0,"tg2"):.2f}  TG3 {critical(0,"tg3"):.2f}  lumped {critical(0,"lumped"):.2f}')
