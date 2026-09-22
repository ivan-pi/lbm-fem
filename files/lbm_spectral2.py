# Extensions of lbm_spectral.py: (a) dimensionally split streaming (x sweep, then
# y sweep; exact for the diagonal shift at C = 1), (b) regularised collision.
import numpy as np
exec(open('lbm_spectral.py').read().split("print('critical")[0])
def S_split(tx, ty, ex, ey, C, tg3):
    gx = G_1d(tx, C*abs(ex), tg3) if ex else 1.
    gy = G_1d(ty, C*abs(ey), tg3) if ey else 1.
    return gx * gy   # 1-D symbol is even in theta under e -> -e with conj: handle sign
def S_split_signed(tx, ty, ex, ey, C, tg3):
    gx = G_1d(ex*tx, C, tg3) if ex else 1.
    gy = G_1d(ey*ty, C, tg3) if ey else 1.
    return gx * gy
# regularisation matrix: projects the non-equilibrium onto the 2nd-order Hermite space
Q = [np.array([[e[0]*e[0]-1/3, e[0]*e[1]],[e[0]*e[1], e[1]*e[1]-1/3]]) for e in E]
R = np.array([[W[a]*4.5*np.sum(Q[a]*Q[b]) for b in range(9)] for a in range(9)])
def rho2(C, omega, scheme, split, reg, n=73):
    t = np.linspace(-np.pi, np.pi, n); r = 0.
    K = J + (1-omega) * (R if reg else np.eye(9)) @ (np.eye(9) - J)
    for tx in t:
        for ty in t:
            if split: S = [S_split_signed(tx,ty,ex,ey,C,scheme=='tg3') for ex,ey in E]
            elif scheme == 'lumped': S = [G_lumped(tx,ty,ex,ey,C) for ex,ey in E]
            else: S = [G_2d(tx,ty,ex,ey,C,scheme=='tg3') if (ex,ey)!=(0,0) else 1. for ex,ey in E]
            r = max(r, np.abs(np.linalg.eigvals(np.diag(S) @ K)).max())
    return r
def crit2(omega, scheme, split=False, reg=False):
    last = 0
    for C in np.arange(0.05, 1.21, 0.05):
        if rho2(C, omega, scheme, split, reg) > 1 + 1e-6: break
        last = C
    return last
print('critical per-axis C:   dt/lambda  omega | TG3 unsplit | TG3 split | TG2 split | TG2 regularised | TG3 regularised')
for dl in (2.26, 9, 36, 200):
    om = dl/(1+dl/2)
    print(f'                        {dl:6.2f}  {om:5.3f} |    {crit2(om,"tg3"):4.2f}     |   {crit2(om,"tg3",True):4.2f}    |   {crit2(om,"tg2",True):4.2f}    |      {crit2(om,"tg2",False,True):4.2f}       |      {crit2(om,"tg3",False,True):4.2f}')
print('pure advection split: TG3', crit2(0,'tg3',True), ' TG2', crit2(0,'tg2',True))
# exactness of the split scheme at C = 1 along the diagonal
tx, ty = 0.7, -1.9
print('split TG3 at C=1, e=(1,1): |G - exp(-i(tx+ty))| =', abs(S_split_signed(tx,ty,1,1,1.0,True) - np.exp(-1j*(tx+ty))))
