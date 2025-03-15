import sympy
import math
import random
import numpy as np

# ─────────────────────────────────────────────────────────
# 1) DÉFINITION SYMBOLIQUE DE f(x) = exp(-x^2)
#    et calcul de ses dérivées 2e & 4e via Sympy
# ─────────────────────────────────────────────────────────

x = sympy.Symbol('x', real=True)
f_expr = sympy.exp(-x**2)            # f(x) = e^{-x^2}

# On calcule les dérivées symboliques d'ordre 2 et 4
f_second_expr  = sympy.diff(f_expr, (x, 2))  # f''(x)
f_fourth_expr  = sympy.diff(f_expr, (x, 4))  # f^(4)(x)

# On convertit en fonctions Python (utilisables par numpy)
f_num      = sympy.lambdify(x, f_expr,         'numpy')  # f(x)
f_second   = sympy.lambdify(x, f_second_expr,  'numpy')  # f''(x)
f_fourth   = sympy.lambdify(x, f_fourth_expr,  'numpy')  # f^(4)(x)

# ─────────────────────────────────────────────────────────
# 2) MÉTHODES D'INTÉGRATION NUMÉRIQUE (1D) + ERREUR
# ─────────────────────────────────────────────────────────

# 2.1) Méthode du rectangle (point milieu)
#     Erreur ~ -((b-a)^3 / [24 n^2]) * f''(xi).
def midpoint_rectangle(f, f_second, a, b, n):
    h = (b - a) / n
    total = 0.0
    for i in range(n):
        x_mid = a + (i + 0.5)*h
        total += f(x_mid)
    integral = h * total

    # On évalue la 2e dérivée au milieu global (a+b)/2 comme approximation.
    x_mid_global = 0.5*(a + b)
    f2_mid = f_second(x_mid_global)
    error_est = abs(((b - a)**3 / (24*(n**2))) * f2_mid)

    return integral, error_est


# 2.2) Méthode des trapèzes
#     Erreur ~ -((b-a)^3 / [12 n^2]) * f''(xi).
def trapeze(f, f_second, a, b, n):
    h = (b - a) / n
    s = 0.5*(f(a) + f(b))
    for i in range(1, n):
        xi = a + i*h
        s += f(xi)
    integral = h * s

    x_mid_global = 0.5*(a + b)
    f2_mid = f_second(x_mid_global)
    error_est = abs(((b - a)**3 / (12*(n**2))) * f2_mid)

    return integral, error_est


# 2.3) Méthode de Simpson
#     Erreur ~ -((b-a)^5 / [180 n^4]) * f^(4)(xi).
def simpson(f, f_fourth, a, b, n):
    """
    f, f_fourth : fonction et sa 4e dérivée
    a, b        : bornes
    n           : nombre de subdivisions (doit être pair)
    """
    if n % 2 != 0:
        raise ValueError("n doit être pair pour Simpson.")

    h = (b - a) / n
    x_points = [a + i*h for i in range(n+1)]
    y_points = [f(xi) for xi in x_points]

    integral = y_points[0] + y_points[-1]
    for i in range(1, n, 2):
        integral += 4 * y_points[i]
    for i in range(2, n, 2):
        integral += 2 * y_points[i]
    integral *= (h / 3)

    x_mid_global = 0.5*(a + b)
    f4_mid = f_fourth(x_mid_global)
    error_est = abs(((b - a)**5 / (180*(n**4))) * f4_mid)

    return integral, error_est


# 2.4) Romberg
# Estimation d'erreur = |R[-1,-1] - R[-2,-2]|.
def romberg_integration(f, a, b, max_iter=5):
    R = np.zeros((max_iter, max_iter), dtype=float)
    # Trapèze de base
    h = (b - a)
    R[0,0] = 0.5 * h * (f(a) + f(b))

    for k in range(1, max_iter):
        # Nombre de subdivisions = 2^k
        n_sub = 2**k
        h = (b - a)/n_sub

        # Calcul T_{k,0}
        subtotal = 0.0
        for i in range(1, 2**(k-1) + 1):
            x_mid = a + (2*i - 1)*h
            subtotal += f(x_mid)
        R[k, 0] = R[k-1, 0]/2.0 + subtotal*h

        # Extrapolation de Richardson
        for j in range(1, k+1):
            R[k, j] = ((4**j)*R[k, j-1] - R[k-1, j-1]) / (4**j - 1)

    val = R[max_iter-1, max_iter-1]
    if max_iter > 1:
        err_est = abs(val - R[max_iter-2, max_iter-2])
    else:
        err_est = 0.0
    return val, err_est


# 2.5) Monte Carlo
# Erreur ~ (b-a)*sqrt(Var(f)/N).
def monte_carlo_integration(f, a, b, N=10000):
    s = 0.0
    s2 = 0.0
    largeur = (b - a)

    for _ in range(N):
        x_rand = a + largeur*random.random()
        fx = f(x_rand)
        s  += fx
        s2 += fx*fx

    mean_f = s / N
    mean_f2 = s2 / N
    var_f = mean_f2 - mean_f**2

    val = largeur * mean_f
    err = largeur * math.sqrt(var_f / N)
    return val, err


# 2.6) Gauss-Legendre
# On compare n et n+2 pour obtenir une estimation d'erreur.
def gauss_legendre_integration(f, a, b, n=5):
    xg, wg = np.polynomial.legendre.leggauss(n)
    mid  = 0.5*(a + b)
    half = 0.5*(b - a)

    val_n = 0.0
    for i in range(n):
        xi = mid + half*xg[i]
        val_n += wg[i]*f(xi)
    val_n *= half

    # On compare à n+2
    n2 = n+2
    xg2, wg2 = np.polynomial.legendre.leggauss(n2)
    val_n2 = 0.0
    for i in range(n2):
        xi = mid + half*xg2[i]
        val_n2 += wg2[i]*f(xi)
    val_n2 *= half

    err_est = abs(val_n2 - val_n)
    return val_n, err_est


# 2.7) Simpson adaptatif
# Erreur estimée via delta/15 par subdivision.
def adaptive_simpson_integration(f, a, b, tol=1e-6, max_depth=15):
    def simpson_segment(f, left, right):
        m = 0.5*(left + right)
        return (right - left)/6.0 * (f(left) + 4.0*f(m) + f(right))

    def recurse(left, right, tol, depth, whole):
        m = 0.5*(left + right)
        left_simp  = simpson_segment(f, left, m)
        right_simp = simpson_segment(f, m, right)
        delta = left_simp + right_simp - whole
        if depth <= 0 or abs(delta) < 15*tol:
            return (left_simp + right_simp) + delta/15.0, abs(delta/15.0)
        val_left, err_left   = recurse(left,  m, tol/2.0, depth-1, left_simp)
        val_right, err_right = recurse(m,     right, tol/2.0, depth-1, right_simp)
        return val_left + val_right, err_left + err_right

    whole = simpson_segment(f, a, b)
    val, err = recurse(a, b, tol, max_depth, whole)
    return val, err

# ─────────────────────────────────────────────────────────
# 3) POINT D'ENTRÉE : on fait tous les calculs sur la MÊME fonction
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    a, b = 0, 1
    print(f"Fonction test : f(x) = exp(-x^2) sur [{a}, {b}]")

    val_rom, err_rom = romberg_integration(f_num, a, b, max_iter=5)
    print(f"[Romberg]         Intégrale ={val_rom:.6f}, Erreur estimée={err_rom:.2e}")

    val_mc, err_mc = monte_carlo_integration(f_num, a, b, N=10000)
    print(f"[Monte Carlo]     Intégrale ={val_mc:.6f}, Erreur estimée={err_mc:.2e}")

    val_gauss, err_gauss = gauss_legendre_integration(f_num, a, b, n=5)
    print(f"[Gauss-Legendre]  Intégrale ={val_gauss:.6f}, Erreur estimée={err_gauss:.2e}")

    val_asimp, err_asimp = adaptive_simpson_integration(f_num, a, b, tol=1e-7)
    print(f"[Adapt. Simpson]  Intégrale ={val_asimp:.6f}, Erreur estimée={err_asimp:.2e}")

    n_sub = 10
    val_rect, err_rect = midpoint_rectangle(f_num, f_second, a, b, n_sub)
    print(f"[Rectangle milieu]  Intégrale ={val_rect:.6f}, Erreur estimée={err_rect:.2e}")

    val_trap, err_trap = trapeze(f_num, f_second, a, b, n_sub)
    print(f"[Trapèzes]         Intégrale ={val_trap:.6f}, Erreur estimée={err_trap:.2e}")

    #    (il faut n_sub pair)
    if n_sub % 2 == 1:
        n_sub += 1
    val_simp, err_simp = simpson(f_num, f_fourth, a, b, n_sub)
    print(f"[Simpson]          Intégrale ={val_simp:.6f}, Erreur estimée={err_simp:.2e}\n")

def best_method(f_num, f_second, f_fourth, a, b):
    """
    Compare les méthodes:
      - Romberg
      - Monte Carlo
      - Gauss-Legendre
      - Simpson adaptatif
      - Rectangle (point milieu)
      - Trapèzes
      - Simpson classique (non adapt.)
    et renvoie (nom_method, valeur_integration, erreur_estimee)
    correspondant à celle qui a la plus petite erreur estimée.
    
    Paramètres:
      f_num     : la fonction f(x)
      f_second  : la 2e dérivée (si nécessaire)
      f_fourth  : la 4e dérivée (si nécessaire)
      a, b      : bornes d'intégration
    """
    results = []

    val_rom, err_rom = romberg_integration(f_num, a, b, max_iter=5)
    results.append(("Romberg", val_rom, err_rom))

    val_mc, err_mc = monte_carlo_integration(f_num, a, b, N=10000)
    results.append(("MonteCarlo", val_mc, err_mc))

    val_gauss, err_gauss = gauss_legendre_integration(f_num, a, b, n=5)
    results.append(("GaussLegendre", val_gauss, err_gauss))

    val_asimp, err_asimp = adaptive_simpson_integration(f_num, a, b, tol=1e-7)
    results.append(("SimpsonAdapt", val_asimp, err_asimp))

    n_sub = 10
    val_rect, err_rect = midpoint_rectangle(f_num, f_second, a, b, n_sub)
    results.append(("RectangleMilieu", val_rect, err_rect))

    val_trap, err_trap = trapeze(f_num, f_second, a, b, n_sub)
    results.append(("Trapezes", val_trap, err_trap))

    #    il faut n_sub pair, donc on l’ajuste si besoin
    if n_sub % 2 == 1:
        n_sub += 1
    val_simp, err_simp = simpson(f_num, f_fourth, a, b, n_sub)
    results.append(("SimpsonClassique", val_simp, err_simp))

    # On sélectionne la méthode avec la plus petite erreur estimée
    best = min(results, key=lambda x: x[2])  # x = (nom, val, err)
    return best  # (nom_method, val_integration, err_estimee)


# Exemple d'utilisation:
if __name__ == "__main__":
    a, b = 0, 1
    
    best_name, best_val, best_err = best_method(f_num, f_second, f_fourth, a, b)
    print(f"Meilleure méthode: {best_name}")
    print(f"Valeur intégrée  : {best_val:.6f}")
    print(f"Erreur estimée   : {best_err:.2e}")

