# TG2 with the consistent mass matrix replaced by k Richardson passes
# preconditioned by the lumped mass:  x_0 = M_L^{-1} r,  x_{j+1} = x_j + M_L^{-1}(r - M x_j)
# => M^{-1} ~ M_L^{-1} sum_{j<=k} (I - M M_L^{-1})^j, symbol sum (1-m)^j (h=1, M_L = 1).
import numpy as np, os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lbm_spectral.py')).read().split("print('critical")[0])
def mass_inv_symbol(m, k):        # k passes (k = 0: lumped, inf: consistent)
    return sum((1 - m)**j for j in range(k + 1)) if k >= 0 else 1 / m
def G_2d_rich(tx, ty, ex, ey, C, k):
    mx,cx,kx=sym1d(tx); my,cy,ky=sym1d(ty)
    Ce = ex*cx*my+ey*mx*cy; Ke = ex**2*kx*my+ey**2*mx*ky-2*ex*ey*cx*cy
    return 1 - (C*Ce + 0.5*C**2*Ke) * mass_inv_symbol(mx*my, k)
def rho_r(C, omega, k, n=73):
    t = np.linspace(-np.pi, np.pi, n); r = 0.; K = np.eye(9) - omega*(np.eye(9)-J)
    for tx in t:
        for ty in t:
            S = [G_2d_rich(tx,ty,ex,ey,C,k) if (ex,ey)!=(0,0) else 1. for ex,ey in E]
            r = max(r, np.abs(np.linalg.eigvals(np.diag(S) @ K)).max())
    return r
def crit_r(omega, k):
    last = 0
    for C in np.arange(0.05, 1.21, 0.05):
        if rho_r(C, omega, k) > 1 + 1e-6: break
        last = C
    return last
th = np.linspace(1e-3, np.pi, 400)
print('1-D symbol of the approximate inverse at theta = pi (consistent = 3, lumped = 1):',
      ' '.join(f'k={k}: {mass_inv_symbol(1/3, k):.3f}' for k in (0,1,2,3)))
print('1-D phase error of M^{-1}C at theta = pi/4 (relative), per number of passes:')
for k in (0,1,2,3,-1):
    m,c,_ = sym1d(np.pi/4); print(f'  k={k:2d}: {abs((c*mass_inv_symbol(m,k)).imag/np.sin(np.pi/4)/ (np.pi/4)*np.sin(np.pi/4) - 1):.2e}'.replace('k=-1','consistent'))
print('coupled scheme (TG2, D2Q9 BGK): critical per-axis C')
print('  dt/lambda  omega |  lumped(k=0)  k=1   k=2   k=3   consistent')
for dl in (2.26, 9, 36):
    om = dl/(1+dl/2)
    print(f'  {dl:7.2f}   {om:5.3f} |   {crit_r(om,0):4.2f}      {crit_r(om,1):4.2f}  {crit_r(om,2):4.2f}  {crit_r(om,3):4.2f}   {crit_r(om,-1):4.2f}')
