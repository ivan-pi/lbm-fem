# Element matrices of the characteristic Galerkin schemes for Q_p Lagrange
# elements on a square h x h grid, assembled into nodal stencils, with the
# equivalent differential operator from a Taylor expansion.
import sympy as sp
from sympy import Rational as R
from itertools import product

x, y, h = sp.symbols('x y h', positive=True)

def lagrange_1d(p):
    nodes = [R(i, p) for i in range(p + 1)]           # equidistant on [0,1]
    t = sp.symbols('t')
    basis = []
    for i, xi in enumerate(nodes):
        L = sp.Integer(1)
        for j, xj in enumerate(nodes):
            if j != i:
                L *= (t - xj) / (xi - xj)
        basis.append(sp.expand(L))
    return t, nodes, basis

def element(p):
    """Shape functions N_a(x,y) on [0,h]^2, node coordinates (multiples of h/p)."""
    t, nodes, basis = lagrange_1d(p)
    N, X = [], []
    for j, i in product(range(p + 1), repeat=2):      # y outer, x inner
        N.append(basis[i].subs(t, x / h) * basis[j].subs(t, y / h))
        X.append((nodes[i], nodes[j]))
    return N, X

def integrate(expr):
    return sp.integrate(sp.integrate(sp.expand(expr), (x, 0, h)), (y, 0, h))

def element_matrices(p, e):
    """M, C_e, K_e = int (e.grad N)(e.grad N^T)  (Bardow D_e = K_e/2)."""
    N, X = element(p)
    n = len(N)
    edot = lambda f: e[0] * sp.diff(f, x) + e[1] * sp.diff(f, y)
    M = sp.Matrix(n, n, lambda a, b: integrate(N[a] * N[b]))
    C = sp.Matrix(n, n, lambda a, b: integrate(N[a] * edot(N[b])))
    K = sp.Matrix(n, n, lambda a, b: integrate(edot(N[a]) * edot(N[b])))
    return M, C, K, X

def stencils(p, A, X):
    """Assemble the element matrix A on an infinite equidistant grid with spacing
    h/p and return, for every node type (offset within the element), the stencil
    as dict {(dx,dy) in grid units: value}."""
    types = sorted({(sp.Integer(xi * p) % p, sp.Integer(yi * p) % p) for xi, yi in X})
    out = {}
    for tx, ty in types:
        st = {}
        # elements containing a node of this type at the origin: element corner at
        # (-tx - p*kx, -ty - p*ky)
        kxs = [0] if tx else [0, -1]
        kys = [0] if ty else [0, -1]
        for kx in kxs:
            for ky in kys:
                cx, cy = -tx + p * kx, -ty + p * ky            # element corner (grid units)
                idx = {tuple(int(v * p) for v in xy): i for i, xy in enumerate(X)}
                a = idx[(int(-cx), int(-cy))]                   # local index of origin node
                for b, (xb, yb) in enumerate(X):
                    key = (cx + int(xb * p), cy + int(yb * p))
                    st[key] = st.get(key, 0) + A[a, b]
        out[(tx, ty)] = {k: sp.nsimplify(v) for k, v in st.items() if v != 0}
    return out

def taylor(st, p, order=4):
    """Apply stencil to a smooth f and expand: returns dict derivative -> coeff."""
    f = sp.Function('f')
    hh = h / p
    dx, dy = sp.symbols('dx dy')
    expr = 0
    for (i, j), v in st.items():
        # Taylor expansion of f(i hh, j hh) to given order
        terms = 0
        for m in range(order + 1):
            for a in range(m + 1):
                b = m - a
                terms += (i * hh) ** a * (j * hh) ** b / (sp.factorial(a) * sp.factorial(b)) * sp.Symbol(f'f_{a}{b}')
        expr += v * terms
    expr = sp.expand(expr)
    res = {}
    for a in range(order + 1):
        for b in range(order + 1 - a):
            c = expr.coeff(sp.Symbol(f'f_{a}{b}'))
            if c != 0:
                res[(a, b)] = sp.simplify(c)
    return res

def fmt_taylor(res):
    d = []
    for (a, b), c in sorted(res.items(), key=lambda kv: (sum(kv[0]), kv[0])):
        name = 'f' + ('_{' + 'x' * a + 'y' * b + '}' if a + b else '')
        d.append(f'({c}) {name}')
    return ' + '.join(d)

if __name__ == '__main__':
    for p in (1, 2):
        for name, e in (('(1,0)', (1, 0)), ('(1,1)', (1, 1))):
            M, C, K, X = element_matrices(p, e)
            print(f'\n=== Q{p}, e = {name} ===')
            print('M/h^2 =', (M / h**2).applyfunc(sp.nsimplify).tolist() if p == 1 else '(see stencils)')
            for label, A in (('M', M), (f'C_{name}', C), (f'K_{name}', K)):
                for node, st in stencils(p, A, X).items():
                    print(f'{label}, node type {node}:')
                    keys = sorted(st)
                    xs = sorted({k[0] for k in keys}); ys = sorted({k[1] for k in keys}, reverse=True)
                    for j in ys:
                        print('   ', ' '.join(f'{str(st.get((i, j), 0)):>8}' for i in xs))
                    print('    sum =', sp.simplify(sum(st.values())), ';  Taylor:', fmt_taylor(taylor(st, p)))
