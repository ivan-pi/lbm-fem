# Von Neumann analysis of the consistent-mass Taylor-Galerkin streaming step,
# linear (Q1) elements, uniform grid h = 1, mode exp(i k.x), theta = k h.
#   TG2: M (G-1) = -(dt C_e + dt^2/2 K_e)
#   TG3: (M + dt^2/6 K_e)(G-1) = -(dt C_e + dt^2/2 K_e)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sym1d(t):                      # symbols of the 1-D element matrices (h = 1)
    return (2 + np.cos(t)) / 3, 1j * np.sin(t), 2 * (1 - np.cos(t))   # m, c, k

def G_1d(theta, C, tg3):
    m, c, k = sym1d(theta)
    rhs = C * c + 0.5 * C**2 * k
    lhs = m + (C**2 / 6 * k if tg3 else 0)
    return 1 - rhs / lhs

def G_2d(tx, ty, ex, ey, C, tg3):  # tensor-product symbols, per-axis Courant C = dt/h
    mx, cx, kx = sym1d(tx); my, cy, ky = sym1d(ty)
    M = mx * my
    Ce = ex * cx * my + ey * mx * cy
    Ke = ex**2 * kx * my + ey**2 * mx * ky - 2 * ex * ey * cx * cy   # cross term: (B^T)_x (x) B_y + B_x (x) (B^T)_y = 2 sin tx sin ty
    rhs = C * Ce + 0.5 * C**2 * Ke
    lhs = M + (C**2 / 6 * Ke if tg3 else 0)
    return 1 - rhs / lhs

if __name__ == '__main__':
    th = np.linspace(0, np.pi, 721)
    # ---- 1-D amplification factors
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for tg3, a, name in ((False, ax[0], 'TG2 (Lax-Wendroff)'), (True, ax[1], 'TG3')):
        for C in (0.25, 0.5, 0.577, 0.7, 0.9, 1.0):
            a.plot(th / np.pi, np.abs(G_1d(th, C, tg3)), label=f'C = {C}')
        a.axhline(1, color='k', lw=0.6); a.set_ylim(0, 1.6); a.set_xlabel(r'$\theta/\pi$'); a.set_ylabel('|G|')
        a.set_title(f'{name}, 1-D, consistent mass'); a.legend(fontsize=8)
    # ---- symbols of the left-hand side
    for C, ls in ((0.25, ':'), (0.577, '--'), (1.0, '-')):
        m, c, k = sym1d(th)
        ax[2].plot(th / np.pi, m + C**2 / 6 * k, 'C1', ls=ls, label=f'TG3: $M + C^2K/6$, C = {C}')
    ax[2].plot(th / np.pi, sym1d(th)[0], 'C0', lw=2, label='TG2: $M$ (consistent)')
    ax[2].axhline(1, color='C2', lw=2, label='lumped $M_L$')
    ax[2].set_xlabel(r'$\theta/\pi$'); ax[2].set_ylabel('symbol'); ax[2].set_title('left-hand side symbol'); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig('tg_spectral_1d.png', dpi=110)

    # ---- exact limits in 1-D at theta = pi
    print('1-D, theta = pi:  TG2: G = 1 - 6 C^2  -> stable for C <= 1/sqrt(3) = 0.577')
    print('                  TG3: G = (1 - 4C^2)/(1 + 2C^2) -> stable for C <= 1')
    for C in (0.5, 0.577, 0.6, 0.9, 1.0):
        print(f'  C = {C}: max|G| TG2 = {np.abs(G_1d(th, C, False)).max():.4f}, TG3 = {np.abs(G_1d(th, C, True)).max():.4f}',
              f'  TG3 phase error at C=1: max |G - exp(-i theta)| = {np.abs(G_1d(th, C, True) - np.exp(-1j*th)).max():.2e}' if C == 1.0 else '')

    # ---- 2-D: max |G| over the 8 directions as a function of the per-axis Courant number
    TX, TY = np.meshgrid(np.linspace(-np.pi, np.pi, 181), np.linspace(-np.pi, np.pi, 181))
    dirs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    Cs = np.linspace(0.05, 1.2, 116)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for tg3, name in ((False, 'TG2'), (True, 'TG3')):
        for e, ls in (((1, 0), '--'), ((1, 1), '-')):
            g = [np.abs(G_2d(TX, TY, *e, C, tg3)).max() for C in Cs]
            ax.plot(Cs, g, ls=ls, label=f'{name}, e = {e}')
            crit = Cs[np.array(g) <= 1 + 1e-9].max()
            print(f'2-D {name} e={e}: stable up to per-axis C = {crit:.3f}')
    ax.axhline(1, color='k', lw=0.6); ax.set_ylim(0.9, 1.8); ax.set_xlabel(r'per-axis Courant number $C = \Delta t / h$')
    ax.set_ylabel(r'$\max_\theta |G|$'); ax.set_title('2-D Q1, consistent mass, axis and diagonal directions'); ax.legend()
    plt.tight_layout(); plt.savefig('tg_spectral_2d.png', dpi=110)
