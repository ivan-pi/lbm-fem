# Von Neumann analysis of the 1-D active flux scheme (Eymann & Roe) for linear
# advection u_t + c u_x = 0, c > 0, h = 1, Courant number C = c dt.
# DoFs per cell: the average ubar_i and the point value u_{i+1/2} at the right
# interface. Reconstruction in cell i: the quadratic with u_{i-1/2}, u_{i+1/2}
# at the ends and average ubar_i:
#   R_i(s) = u_L (1 - 4s + 3s^2) + u_R (-2s + 3s^2) + ubar (6s - 6s^2),  s in [0,1].
# Point values are evolved exactly along the characteristic (upwind cell i for
# the interface i+1/2), the average by the conservative flux update with Simpson
# in time: F = c (u^n + 4 u^{n+1/2} + u^{n+1})/6.
import numpy as np
def basis(s):   # weights of (u_L, u_R, ubar) in R(s)
    return np.array([1 - 4*s + 3*s**2, -2*s + 3*s**2, 6*s - 6*s**2])
def amplification(theta, C):
    E = np.exp(1j*theta)          # shift by one cell to the right
    # unknowns per cell i: (ubar_i, u_{i+1/2}); u_{i-1/2} = u_{i+1/2} / E
    def point(s):                 # value of R_i at s, as a row on (ubar_i, u_{i+1/2})
        wL, wR, wA = basis(s)
        return np.array([wA, wR + wL / E])
    p_new  = point(1 - C)         # u_{i+1/2}^{n+1}
    p_half = point(1 - C/2)       # u_{i+1/2}^{n+1/2}
    p_old  = np.array([0., 1.])   # u_{i+1/2}^n
    F_right = (p_old + 4*p_half + p_new) / 6      # flux at i+1/2 (per c)
    F_left  = F_right / E                         # flux at i-1/2
    avg_new = np.array([1., 0.]) - C * (F_right - F_left)
    return np.array([avg_new, p_new])
th = np.linspace(1e-3, np.pi, 1200)
print('active flux, 1-D: max |eigenvalue| over theta, and phase error of the physical mode at theta = pi/4 (per cell)')
for C in (0.1, 0.25, 0.5, 0.7, 0.9, 1.0, 1.05):
    rho = 0.; err = None
    for t in th:
        ev = np.linalg.eigvals(amplification(t, C))
        rho = max(rho, np.abs(ev).max())
        if err is None and t >= np.pi/4:
            exact = np.exp(-1j*C*t)
            err = np.abs(ev - exact).min()
    print(f'  C = {C:4.2f}: max|G| = {rho:.6f}   |G_phys - exp(-iC theta)| at pi/4 = {err:.2e}')
# compare with Q1 TG3 (2 DoF per cell would be Q2; TG3-Q1 has 1 DoF per cell) at the same theta = pi/4
m = (2+np.cos(np.pi/4))/3; c = 1j*np.sin(np.pi/4); k = 2*(1-np.cos(np.pi/4))
for C in (0.25, 0.5, 1.0):
    G = 1 - (C*c + 0.5*C**2*k)/(m + C**2/6*k)
    print(f'  Q1 TG3 C = {C}: |G - exp(-iC theta)| at pi/4 = {abs(G - np.exp(-1j*C*np.pi/4)):.2e}')
