import math
import random
import numpy as np
import sympy

# ─────────────────────────────────────────────────────────
# 1) Exemple de fonction f(x,y) = exp(-(x^2 + y^2)) (sympy)
#    On pourrait aussi faire un f Python direct, sans sympy
# ─────────────────────────────────────────────────────────

x_sym, y_sym = sympy.symbols('x y', real=True)
f_expr_2d = sympy.exp(-(x_sym**2 + y_sym**2))
f_2d = sympy.lambdify((x_sym, y_sym), f_expr_2d, 'numpy')  # f_2d(x,y)
# ─────────────────────────────────────────────────────────
# 2) Méthodes "classiques" 2D + Estimation d'erreur par maillage doublé
# ─────────────────────────────────────────────────────────

def rectangle_2d_midpoint(f, a, b, c, d, nx, ny):
    """
    Méthode des rectangles en 2D, sans estimation d'erreur.
    f(x,y) : fonction
    [a,b] x [c,d] : domaine
    nx, ny : subdivisions
    """
    hx = (b - a) / nx
    hy = (d - c) / ny
    total = 0.0
    for i in range(nx):
        for j in range(ny):
            x_mid = a + (i + 0.5)*hx
            y_mid = c + (j + 0.5)*hy
            total += f(x_mid, y_mid)
    return hx*hy*total


def rectangle_2d_midpoint_with_error(f, a, b, c, d, nx, ny):
    """
    Rectangle (milieu) 2D + estimation d'erreur par maillage doublé :
      - I1 : approximation avec (nx, ny)
      - I2 : approximation avec (2nx, 2ny)
      - err ~ |I2 - I1|
    Retourne (valeur, erreur_estimee).
    """
    I1 = rectangle_2d_midpoint(f, a, b, c, d, nx, ny)
    I2 = rectangle_2d_midpoint(f, a, b, c, d, 2*nx, 2*ny)
    return I1, abs(I2 - I1)


def trapeze_2d(f, a, b, c, d, nx, ny):
    """
    Trapèzes 2D, sans estimation d'erreur.
    """
    hx = (b - a)/nx
    hy = (d - c)/ny
    total = 0.0
    for i in range(nx+1):
        for j in range(ny+1):
            # Poids en x
            wx = 0.5 if (i == 0 or i == nx) else 1.0
            # Poids en y
            wy = 0.5 if (j == 0 or j == ny) else 1.0
            x_ij = a + i*hx
            y_ij = c + j*hy
            total += wx*wy*f(x_ij, y_ij)
    return hx*hy*total


def trapeze_2d_with_error(f, a, b, c, d, nx, ny):
    """
    Trapèzes 2D + estimation d'erreur par maillage doublé.
    """
    I1 = trapeze_2d(f, a, b, c, d, nx, ny)
    I2 = trapeze_2d(f, a, b, c, d, 2*nx, 2*ny)
    return I1, abs(I2 - I1)


def simpson_2d(f, a, b, c, d, nx, ny):
    """
    Simpson 2D, nx et ny doivent être pairs. Pas d'estimation d'erreur ici.
    """
    if nx % 2 != 0 or ny % 2 != 0:
        raise ValueError("nx et ny doivent être pairs pour Simpson 2D.")
    hx = (b - a)/nx
    hy = (d - c)/ny
    total = 0.0
    for i in range(nx+1):
        for j in range(ny+1):
            # Poids Simpson en x
            if i == 0 or i == nx:
                wx = 1
            elif i % 2 == 1:
                wx = 4
            else:
                wx = 2
            # Poids Simpson en y
            if j == 0 or j == ny:
                wy = 1
            elif j % 2 == 1:
                wy = 4
            else:
                wy = 2
            total += wx*wy*f(a + i*hx, c + j*hy)
    return (hx*hy/9.0)*total


def simpson_2d_with_error(f, a, b, c, d, nx, ny):
    """
    Simpson 2D + estimation d'erreur par maillage doublé.
    On impose nx, ny pairs. On double -> 2nx, 2ny (toujours pairs).
    """
    if nx % 2 != 0:
        nx += 1
    if ny % 2 != 0:
        ny += 1
    I1 = simpson_2d(f, a, b, c, d, nx, ny)

    # Double
    I2 = simpson_2d(f, a, b, c, d, 2*nx, 2*ny)
    return I1, abs(I2 - I1)


# ─────────────────────────────────────────────────────────
# 3) Romberg en 2D (approche “nestée”)
#    On fait : integrale en x par Romberg pour chaque y, puis Romberg en y
# ─────────────────────────────────────────────────────────

def romberg_1d(f, a, b, max_iter=5):
    """
    Romberg 1D déjà vu. Retourne (val, err_est).
    """
    R = np.zeros((max_iter, max_iter), dtype=float)
    h = (b - a)
    # Trapèze de base
    R[0,0] = 0.5 * h * (f(a) + f(b))

    for k in range(1, max_iter):
        n_sub = 2**k
        h = (b - a)/n_sub
        subtotal = 0.0
        for i in range(1, 2**(k-1)+1):
            x_mid = a + (2*i - 1)*h
            subtotal += f(x_mid)
        R[k, 0] = R[k-1, 0]/2.0 + subtotal*h

        for j in range(1, k+1):
            R[k, j] = ((4**j)*R[k, j-1] - R[k-1, j-1])/(4**j - 1)

    val = R[max_iter-1, max_iter-1]
    err = abs(val - R[max_iter-2, max_iter-2]) if max_iter>1 else 0.0
    return val, err


def romberg_2d(f, ax, bx, ay, by, max_iter=5):
    """
    Romberg 2D = on fait un Romberg (en x) pour chaque y (points de quadrature en y),
    puis on refait un Romberg (en y) sur le résultat.
    Approche : 
      g(y) = ∫[x=ax..bx] f(x,y) dx  (via Romberg 1D)
    puis on applique Romberg 1D à g(y) sur [ay, by].
    On prend l'estimation d'erreur du dernier Romberg en y comme 'err_est'.
    """
    def g(y):
        valx, _ = romberg_1d(lambda xx: f(xx, y), ax, bx, max_iter)
        return valx

    def G_romberg_1d(yy):
        return g(yy)

    # On applique Romberg 1D sur [ay,by]
    R = np.zeros((max_iter, max_iter), dtype=float)
    hy = (by - ay)
    # T0,0
    R[0,0] = 0.5 * hy * (G_romberg_1d(ay) + G_romberg_1d(by))

    for k in range(1, max_iter):
        n_sub = 2**k
        hy_ = (by - ay)/n_sub
        subtotal = 0.0
        for i in range(1, 2**(k-1)+1):
            y_mid = ay + (2*i - 1)*hy_
            subtotal += G_romberg_1d(y_mid)
        R[k,0] = R[k-1,0]/2.0 + subtotal*hy_
        for j in range(1, k+1):
            R[k,j] = ((4**j)*R[k,j-1] - R[k-1,j-1])/(4**j - 1)

    val = R[max_iter-1, max_iter-1]
    err = abs(val - R[max_iter-2, max_iter-2]) if max_iter>1 else 0.0
    return val, err


# ─────────────────────────────────────────────────────────
# 4) Simpson adaptatif en 2D (approche nestée)
#    On utilise un Simpson adaptatif 1D pour x, puis un Simpson adaptatif 1D pour y
# ─────────────────────────────────────────────────────────

def adaptive_simpson_1d(f, a, b, tol=1e-6, max_depth=15):
    """
    Simpson adaptatif 1D.
    Retourne (valeur, err_est).
    """
    def simpson_segment(left, right):
        m = 0.5*(left + right)
        return (right - left)/6.0 * (f(left) + 4*f(m) + f(right))

    def recurse(left, right, tol, depth, whole):
        m = 0.5*(left + right)
        left_simp  = simpson_segment(left, m)
        right_simp = simpson_segment(m, right)
        delta = left_simp + right_simp - whole
        if depth <= 0 or abs(delta) < 15*tol:
            return (left_simp + right_simp) + delta/15.0, abs(delta/15.0)
        val_left,  err_left  = recurse(left,  m, tol/2.0, depth-1, left_simp)
        val_right, err_right = recurse(m, right, tol/2.0, depth-1, right_simp)
        return val_left + val_right, err_left + err_right

    whole = simpson_segment(a, b)
    val, err = recurse(a, b, tol, max_depth, whole)
    return val, err


def adaptive_simpson_2d(f, ax, bx, ay, by, tol=1e-6, max_depth=10):
    """
    Simpson adaptatif 2D (version nestée) :
     - On définit G(y) = ∫[x=ax..bx] f(x,y) dx (par simpson adaptatif 1D)
     - Puis on réapplique simpson adaptatif 1D en y sur [ay, by].
    L'erreur finale est la somme (ou un majorant) des erreurs internes x + y.
    """
    def integrate_in_x(y, tol_x):
        valx, errx = adaptive_simpson_1d(lambda xx: f(xx,y), ax, bx, tol=tol_x, max_depth=max_depth)
        return valx, errx

    def g(y):
        valx, _ = integrate_in_x(y, tol_x=tol/10)  # un peu arbitraire
        return valx

    def g_simp(y):
        valx, _ = integrate_in_x(y, tol_x=tol/10)
        return valx

    valY, errY = adaptive_simpson_1d(g_simp, ay, by, tol=tol, max_depth=max_depth)
    return valY, errY


# ─────────────────────────────────────────────────────────
# 5) Exemple d’utilisation sur [0,1]x[0,1]
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    a, b = 0.0, 1.0
    c, d = 0.0, 1.0
    print(f"On intègre f(x,y) = exp(-(x^2 + y^2)) sur [{a},{b}] x [{c},{d}].\n")

    val_rect,  err_rect  = rectangle_2d_midpoint_with_error(f_2d, a, b, c, d, 10, 10)
    val_trap,  err_trap  = trapeze_2d_with_error(f_2d, a, b, c, d, 10, 10)
    val_simp,  err_simp  = simpson_2d_with_error(f_2d, a, b, c, d, 10, 10)
    val_rom2d, err_rom2d = romberg_2d(f_2d, a, b, c, d, max_iter=5)
    val_asimp2d, err_asimp2d = adaptive_simpson_2d(f_2d, a, b, c, d, tol=1e-4, max_depth=10)

    print(f"[Rectangle 2D] :  Intégrale={val_rect:.6f},  Erreur estimée={err_rect:.2e}")
    print(f"[Trapèzes 2D]  :  Intégrale={val_trap:.6f},  Erreur estimée={err_trap:.2e}")
    print(f"[Simpson 2D]   :  Intégrale={val_simp:.6f},  Erreur estimée={err_simp:.2e}")
    print(f"[Romberg 2D]   :  Intégrale={val_rom2d:.6f}, Erreur estimée={err_rom2d:.2e}")
    print(f"[Adapt. Simpson 2D] Intégrale={val_asimp2d:.6f}, Erreur estimée={err_asimp2d:.2e}")
    # ─────────────────────────────────────────────────────
    # Monte Carlo 2D
    # ─────────────────────────────────────────────────────
    def monte_carlo_2d(f, a, b, c, d, N=100_000):
        area = (b - a)*(d - c)
        s, s2 = 0.0, 0.0
        for _ in range(N):
            xx = a + (b - a)*random.random()
            yy = c + (d - c)*random.random()
            val = f(xx, yy)
            s  += val
            s2 += val*val
        mean_f  = s / N
        mean_f2 = s2 / N
        var_f   = mean_f2 - mean_f**2
        integral = area*mean_f
        err = area*math.sqrt(var_f / N)
        return integral, err

    val_mc2d, err_mc2d = monte_carlo_2d(f_2d, a, b, c, d, N=50_000)
    print(f"[Monte Carlo 2D] : Intégrale={val_mc2d:.6f}, Erreur estimée={err_mc2d:.2e}\n")

def best_method_2d(f, a, b, c, d):
    """
    Compare plusieurs méthodes d'intégration 2D, récupère l'erreur estimée,
    et retourne la méthode 'gagnante' (c.-à-d. celle avec la plus petite err_est).
    f        : fonction f(x,y)
    a,b,c,d  : bornes d'intégration (rectangle [a,b] x [c,d])
    Retourne (best_name, best_value, best_error).
    """

    results = []

    val_rect, err_rect = rectangle_2d_midpoint_with_error(f, a, b, c, d, 10, 10)
    results.append(("Rectangle_2D", val_rect, err_rect))

    val_trap, err_trap = trapeze_2d_with_error(f, a, b, c, d, 10, 10)
    results.append(("Trapezes_2D", val_trap, err_trap))

    val_simp, err_simp = simpson_2d_with_error(f, a, b, c, d, 10, 10)
    results.append(("Simpson_2D", val_simp, err_simp))

    val_rom, err_rom = romberg_2d(f, a, b, c, d, max_iter=5)
    results.append(("Romberg_2D", val_rom, err_rom))

    val_asimp, err_asimp = adaptive_simpson_2d(f, a, b, c, d, tol=1e-4, max_depth=10)
    results.append(("AdaptSimpson_2D", val_asimp, err_asimp))

    val_mc, err_mc = monte_carlo_2d(f, a, b, c, d, N=50_000)
    results.append(("MonteCarlo_2D", val_mc, err_mc))


    best = min(results, key=lambda x: x[2])  # x = (nom, valeur, err)
    best_name, best_val, best_err = best

    return best_name, best_val, best_err


if __name__ == "__main__":
    
    a, b = 0.0, 1.0
    c, d = 0.0, 1.0
    best_name, best_val, best_err = best_method_2d(f_2d, a, b, c, d)
    print(f"Méthode gagnante : {best_name}")
    print(f"Valeur approx    : {best_val:.6f}")
    print(f"Erreur estimée   : {best_err:.2e}")

