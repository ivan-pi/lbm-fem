# Generates fem_stencils.tex: element matrices -> nodal stencils -> equivalent
# differential operators for the characteristic Galerkin schemes (Q1, Q2).
import sympy as sp
from stencils import element_matrices, stencils, taylor, h

def tex(v):
    v = sp.nsimplify(v)
    if v == 0:
        return '0'
    s = sp.latex(v)
    return s

def tex_coeff(c):
    """coefficient without the h power: return (coefficient, power)"""
    c = sp.nsimplify(c)
    pw = sp.degree(sp.Poly(c, h)) if c.has(h) else 0
    return sp.latex(sp.simplify(c / h**pw)), pw

def operator(res):
    """Group the Taylor expansion by power of h: h^2(...) + h^4(...), x before y."""
    groups = {}
    for (a, b), c in res.items():
        coeff, pw = tex_coeff(c)
        name = 'f' if a + b == 0 else 'f_{' + 'x' * a + 'y' * b + '}'
        groups.setdefault(pw, []).append((coeff, name, (-a, b)))
    parts = []
    for pw in sorted(groups):
        terms = []
        for coeff, name, _ in sorted(groups[pw], key=lambda t: t[2]):
            terms.append(name if coeff == '1' else '-' + name if coeff == '-1' else coeff + ' ' + name)
        hs = f'h^{{{pw}}}' if pw else ''
        if len(terms) == 1:
            t = terms[0]
            s = ('-' + hs + ' ' + t[1:]) if t.startswith('-') else hs + ' ' + t
        else:
            s = hs + r' \left(' + ' + '.join(terms).replace('+ -', '- ') + r'\right)'
        parts.append(s.strip())
    return ' + '.join(parts).replace('+ -', '- ')

def stencil_tikz(st, cell=0.75, sign_colour=True):
    keys = sorted(st)
    xs = sorted({k[0] for k in keys}); ys = sorted({k[1] for k in keys})
    xs = list(range(min(xs), max(xs) + 1)); ys = list(range(min(ys), max(ys) + 1))
    out = []
    x0, x1 = (min(xs) - .5) * cell, (max(xs) + .5) * cell
    y0, y1 = (min(ys) - .5) * cell, (max(ys) + .5) * cell
    out.append(f'\\draw[gray!60] ({x0},{y0}) grid[step={cell}] ({x1},{y1});')
    out.append(f'\\draw[thick] ({x0},{y0}) rectangle ({x1},{y1});')
    for i in xs:
        for j in ys:
            v = st.get((i, j), 0)
            style = 'stencil' + (',central' if (i, j) == (0, 0) else '')
            out.append(f'\\node[{style}] at ({i*cell},{j*cell}) {{${tex(v)}$}};')
    w = (x1 - x0, y1 - y0)
    return '\n'.join(out), w

def patch_tikz(A, X, p=1, cell=1.4):
    """Q1 only: the 4 elements around the central node; A[a,b] written next to
    node b inside each element, a = local index of the central node."""
    out = []
    idx = {tuple(int(v) for v in xy): i for i, xy in enumerate(X)}
    for cx in (-1, 0):
        for cy in (-1, 0):
            a = idx[(-cx, -cy)]
            out.append(f'\\draw ({cx*cell},{cy*cell}) rectangle ({(cx+1)*cell},{(cy+1)*cell});')
            for b, (xb, yb) in enumerate(X):
                px, py = (cx + float(xb)) * cell, (cy + float(yb)) * cell
                mx, my = (cx + .5) * cell, (cy + .5) * cell
                qx, qy = px + .32 * (mx - px), py + .32 * (my - py)
                out.append(f'\\node[patch] at ({qx},{qy}) {{${tex(A[a, b])}$}};')
    for cx in (-1, 0, 1):
        for cy in (-1, 0, 1):
            out.append(f'\\fill ({cx*cell},{cy*cell}) circle (1.3pt);')
    out.append(f'\\fill[red] (0,0) circle (2pt);')
    return '\n'.join(out), (2 * cell, 2 * cell)

CCW = [0, 1, 3, 2]  # sympy/deal.II lexicographic -> counter-clockwise (reference-element) ordering

def pmatrix(A, perm=CCW):
    rows = [' & '.join(tex(A[perm[a], perm[b]]) for b in range(A.cols)) for a in range(A.rows)]
    return r'\begin{pmatrix}' + r'\\ '.join(rows) + r'\end{pmatrix}'

header = r'''\documentclass[tikz,border=6pt]{standalone}
\usepackage{amsmath}
\usetikzlibrary{positioning,matrix}
\tikzset{
  stencil/.style={minimum size=0.55cm,inner sep=1pt,font=\scriptsize,fill=white},
  central/.style={fill=red!12},
  patch/.style={inner sep=1pt,font=\tiny},
  lab/.style={font=\small,anchor=east},
  head/.style={font=\small\itshape,anchor=south},
}
\begin{document}
'''

def figure_q1():
    out = [header, r'\begin{tikzpicture}']
    rows = []
    for name, e in (('(1,0)', (1, 0)), ('(1,1)', (1, 1))):
        M, C, K, X = element_matrices(1, e)
        if name == '(1,0)':
            rows.append((r'M', M, X))
        rows.append((r'C_{' + name + '}', C, X))
        rows.append((r'K_{' + name + '}', K, X))
    y = 0
    out.append(r'\node[head] at (2.2,2.0) {element matrix $A$};')
    out.append(r'\node[head] at (7.5,2.0) {contributions to the central node};')
    out.append(r'\node[head] at (11.6,2.0) {assembled stencil};')
    out.append(r'\node[head] at (16.5,2.0) {equivalent operator, $h \to 0$};')
    for label, A, X in rows:
        st = stencils(1, A, X)[(0, 0)]
        # scale out h powers for the display
        pw = 2 if label == 'M' else (1 if label.startswith('C') else 0)
        Ah = (A / h**pw).applyfunc(sp.nsimplify)
        sth = {k: sp.nsimplify(v / h**pw) for k, v in st.items()}
        stx, _ = stencil_tikz(sth)
        ptx, _ = patch_tikz(Ah, X)
        hs = {2: 'h^2', 1: 'h', 0: ''}[pw]
        out.append(f'\\node[lab] at (-0.2,{y}) {{${label} = {hs}$}};')
        out.append(f'\\node[anchor=west,font=\\scriptsize] at (0,{y}) {{${pmatrix(Ah)}$}};')
        out.append(f'\\node at (4.5,{y}) {{$=$}};')
        out.append(f'\\begin{{scope}}[shift={{(7.5,{y})}}]\n{ptx}\n\\end{{scope}}')
        out.append(f'\\node at (10.0,{y}) {{$=$}};')
        out.append(f'\\begin{{scope}}[shift={{(11.6,{y})}}]\n{stx}\n\\end{{scope}}')
        op = operator(taylor(st, 1))
        out.append(f'\\node[anchor=west,font=\\small] at (13.4,{y}) {{$\\approx {op}$}};')
        out.append(f'\\node[anchor=west,font=\\scriptsize,gray] at (13.4,{y-0.55}) {{row sum ${tex(sp.simplify(sum(st.values())))}$}};')
        y -= 3.6
    out.append(r'''\node[anchor=north west,align=left,font=\footnotesize] at (-1.6,''' + f'{y+1.8}' + r''') {%
$Q_1$ elements, square $h \times h$ grid, counter-clockwise node ordering $0{:}(0,0)$, $1{:}(h,0)$, $2{:}(h,h)$, $3{:}(0,h)$ (deal.II uses $0,1,3,2$).\\
$M = \int N N^T$, $C_e = \int N\,(e\cdot\nabla N^T)$, $K_e = \int (e\cdot\nabla N)(e\cdot\nabla N^T) = 2 D_e$ (Lee \& Lin Eq.~(23) volume term, Bardow et al. Eq.~(13)).\\
The stencil is the row of the assembled matrix at the red node; $A$ acting on nodal values of a smooth $f$ gives the operator on the right.\\
Streaming step of the collide-then-stream scheme on this grid: $M(g^{n+1} - g^*) = (-\Delta t\, C_e - \tfrac12 \Delta t^2 K_e)\, g^*$; with $M$ lumped, $M_L = h^2 I$.};''')
    out.append(r'\end{tikzpicture}')
    out.append(r'\end{document}')
    return '\n'.join(out)

def figure_q2():
    out = [header, r'\begin{tikzpicture}']
    M, C, K, X = element_matrices(2, (1, 0))
    types = [((0, 0), 'vertex'), ((1, 0), '$x$-edge midpoint'), ((0, 1), '$y$-edge midpoint'), ((1, 1), 'cell centre')]
    x = 0
    colw = {(0, 0): 5.2, (1, 0): 4.4, (0, 1): 5.2, (1, 1): 4.4}
    for t, tname in types:
        out.append(f'\\node[head] at ({x},2.3) {{node type: {tname}}};')
        y = 0
        for label, A, pw, order in (('M', M, 2, 2), ('C_{(1,0)}', C, 1, 3), ('K_{(1,0)}', K, 0, 4)):
            st = stencils(2, A, X)[t]
            sth = {k: sp.nsimplify(v / h**pw) for k, v in st.items()}
            stx, (w, hh) = stencil_tikz(sth, cell=0.72)
            hs = {2: 'h^2', 1: 'h', 0: ''}[pw]
            if x == 0:
                out.append(f'\\node[lab] at (-2.4,{y}) {{${label} = {hs}{chr(92)+chr(44)+chr(92)+"times" if hs else ""}$}};')
            out.append(f'\\begin{{scope}}[shift={{({x},{y})}}]\n{stx}\n\\end{{scope}}')
            op = operator(taylor(st, 2, order)).replace(' + h^{4}', r'$\\\\$+ h^{4}')
            out.append(f'\\node[anchor=north,font=\\scriptsize,align=center] at ({x},{y - hh/2 - 0.1}) {{$\\approx {op}$}};')
            y -= 5.4
        x += colw[t] + 1.0
    out.append(r'''\node[anchor=north west,align=left,font=\footnotesize] at (-3.4,''' + f'{y+2.0}' + r''') {%
$Q_2$ elements on a square grid of element size $h$ (node spacing $h/2$), direction $e = (1,0)$. There is no single stencil: the four
node types of a $Q_2$ element (vertex, two edge midpoints, centre) have stencils of $5\times5$, $5\times3$, $3\times5$ and $3\times3$ nodes.\\
Row sums of $M$: $h^2/9,\ 2h^2/9,\ 2h^2/9,\ 4h^2/9$ (the 2-D Simpson weights); the leading operators differ between node types.\\
Element matrices: \texttt{stencils.py} (sympy) and \texttt{elem.cc} (deal.II \texttt{FEValues}) agree.};''')
    out.append(r'\end{tikzpicture}')
    out.append(r'\end{document}')
    return '\n'.join(out)

open('fem_stencils_q1.tex', 'w').write(figure_q1())
open('fem_stencils_q2.tex', 'w').write(figure_q2())


# ---------------------------------------------------------------------------
# Q2: element matrices and their contributions to each node type
# ---------------------------------------------------------------------------
from sympy import Rational as R

# display ordering: corners counter-clockwise, then edge midpoints
# counter-clockwise starting at the bottom edge, then the centre
Q2_ORDER = [(0, 0), (1, 0), (1, 1), (0, 1), (R(1, 2), 0), (1, R(1, 2)), (R(1, 2), 1), (0, R(1, 2)), (R(1, 2), R(1, 2))]

def perm_for(X, order):
    return [X.index(xy) for xy in order]

def patch_tikz_p(A, X, p, node_type, cell=2.1, fontsize=r'\tiny'):
    """Elements containing a node of the given type (offset within the element,
    in units of h/p) at the origin; A[a,b] written next to node b inside each
    element, a = local index of the central node."""
    tx, ty = node_type
    out = []
    kxs = [0] if tx else [0, -1]
    kys = [0] if ty else [0, -1]
    idx = {tuple(int(v * p) for v in xy): i for i, xy in enumerate(X)}
    u = cell / p  # grid unit
    for kx in kxs:
        for ky in kys:
            cx, cy = -tx + p * kx, -ty + p * ky      # element corner in grid units
            a = idx[(int(-cx), int(-cy))]
            out.append(f'\\draw ({cx*u},{cy*u}) rectangle ({(cx+p)*u},{(cy+p)*u});')
            mx, my = (cx + p / 2) * u, (cy + p / 2) * u
            for b, (xb, yb) in enumerate(X):
                px, py = (cx + float(xb) * p) * u, (cy + float(yb) * p) * u
                qx, qy = px + .28 * (mx - px), py + .28 * (my - py)
                out.append(f'\\node[inner sep=0.5pt,font={fontsize}] at ({qx},{qy}) {{${tex(A[a, b])}$}};')
                out.append(f'\\fill ({px},{py}) circle (1pt);')
    out.append(r'\fill[red] (0,0) circle (2pt);')
    return '\n'.join(out)

def reference_sketch(order, size=2.0):
    out = [f'\\draw[thick] (0,0) rectangle ({size},{size});']
    for k, (xi, yi) in enumerate(order):
        px, py = float(xi) * size, float(yi) * size
        out.append(f'\\fill ({px},{py}) circle (1.5pt);')
        dx = 0.28 if xi < 1 else -0.28
        dy = 0.28 if yi < 1 else -0.28
        out.append(f'\\node[font=\\scriptsize] at ({px + dx},{py + dy}) {{{k}}};')
    out.append(f'\\node[font=\\scriptsize,anchor=north] at ({size/2},-0.15) {{$h$}};')
    return '\n'.join(out)

def figure_q2_elements():
    M, C, K, X = element_matrices(2, (1, 0))
    perm = perm_for(X, Q2_ORDER)
    types = [((0, 0), 'vertex'), ((1, 0), '$x$-edge midpoint'), ((0, 1), '$y$-edge midpoint'), ((1, 1), 'cell centre')]
    out = [header, r'\begin{tikzpicture}']
    out.append(r'\node[head] at (3.2,3.4) {element matrix $A$ (node ordering as in the sketch)};')
    out.append(r'\begin{scope}[shift={(11.0,0.9)}]' + '\n' + reference_sketch(Q2_ORDER) + '\n' + r'\end{scope}')
    out.append(r'\node[head,align=center] at (12.0,3.4) {reference element,\\ node numbering};')
    xs = [15.0, 21.0, 25.8, 31.8]
    for (t, tname), x in zip(types, xs):
        out.append(f'\\node[head,align=center] at ({x},3.4) {{contributions to a\\\\ {tname} node}};')
    y = 0
    for label, A, pw in (('M', M, 2), ('C_{(1,0)}', C, 1), ('K_{(1,0)}', K, 0)):
        Ah = (A / h**pw).applyfunc(sp.nsimplify)
        hs = {2: 'h^2', 1: 'h', 0: ''}[pw]
        out.append(f'\\node[lab] at (-0.2,{y}) {{${label} = {hs}$}};')
        out.append(f'\\node[anchor=west,font=\\tiny] at (0,{y}) {{${pmatrix(Ah, perm)}$}};')
        for (t, tname), x in zip(types, xs):
            out.append(f'\\begin{{scope}}[shift={{({x},{y})}}]\n{patch_tikz_p(Ah, X, 2, t)}\n\\end{{scope}}')
            st = stencils(2, A, X)[t]
            out.append(f'\\node[font=\\scriptsize,gray,anchor=north] at ({x},{y - 2.35}) {{row sum ${tex(sp.simplify(sum(st.values()) / h**pw))}$}};')
        y -= 6.4
    out.append(r'''\node[anchor=north west,align=left,font=\footnotesize] at (-1.6,''' + f'{y+3.2}' + r''') {%
$Q_2$ element on a square $h \times h$ grid: $M = \int N N^T$, $C_e = \int N\,(e\cdot\nabla N^T)$, $K_e = \int (e\cdot\nabla N)(e\cdot\nabla N^T) = 2 D_e$, $e = (1,0)$.\\
In each patch the number written next to node $b$ inside an element is $A_{ab}$ of that element, $a$ being the local index of the red node; summing over the
elements sharing the red node gives its stencil (see \texttt{fem\_stencils\_q2}).\\
A vertex is shared by 4 elements ($5\times5$ stencil), an edge midpoint by 2 ($5\times3$ or $3\times5$), the centre belongs to one element ($3\times3$).};''')
    out.append(r'\end{tikzpicture}')
    out.append(r'\end{document}')
    return '\n'.join(out)

open('fem_stencils_q2_elements.tex', 'w').write(figure_q2_elements())
