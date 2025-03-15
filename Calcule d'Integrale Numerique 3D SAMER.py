import math
import random
import numpy as np

# ----------------------------------------------------------------------
# 1) Méthodes "classiques" 3D (Rectangle / Trapèzes / Simpson)
#    => On utilise ici le "double maillage" pour estimer l'erreur.
# ----------------------------------------------------------------------

def rectangle_3d(f, a, b, c, d, e, fz, nx, ny, nz):
    """
    Méthode du rectangle (point milieu) en 3D, sans estimation d'erreur.
    f(x,y,z) sur [a,b] x [c,d] x [e,fz].
    """
    hx = (b - a) / nx
    hy = (d - c) / ny
    hz = (fz - e) / nz
    total = 0.0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x_mid = a + (i + 0.5)*hx
                y_mid = c + (j + 0.5)*hy
                z_mid = e + (k + 0.5)*hz
                total += f(x_mid, y_mid, z_mid)
    val = total * hx * hy * hz
    return val

def rectangle_3d_with_error(f, a, b, c, d, e, fz, nx, ny, nz):
    """
    Rectangle 3D + estimation d'erreur par double maillage (nx,ny,nz) et (2nx,2ny,2nz).
    Retourne (valeur, erreur_est).
    """
    val1 = rectangle_3d(f, a, b, c, d, e, fz, nx, ny, nz)
    val2 = rectangle_3d(f, a, b, c, d, e, fz, 2*nx, 2*ny, 2*nz)
    return val1, abs(val2 - val1)


def trapezes_3d(f, a, b, c, d, e, fz, nx, ny, nz):
    """
    Trapèzes 3D, sans estimation d'erreur.
    """
    hx = (b - a) / nx
    hy = (d - c) / ny
    hz = (fz - e) / nz
    total = 0.0
    for i in range(nx+1):
        for j in range(ny+1):
            for k in range(nz+1):
                x_ij = a + i*hx
                y_ij = c + j*hy
                z_ij = e + k*hz

                # Poids 0.5 sur les bords en x
                wx = 0.5 if (i == 0 or i == nx) else 1.0
                # Poids 0.5 sur les bords en y
                wy = 0.5 if (j == 0 or j == ny) else 1.0
                # Poids 0.5 sur les bords en z
                wz = 0.5 if (k == 0 or k == nz) else 1.0

                total += wx * wy * wz * f(x_ij, y_ij, z_ij)
    val = hx * hy * hz * total
    return val

def trapezes_3d_with_error(f, a, b, c, d, e, fz, nx, ny, nz):
    val1 = trapezes_3d(f, a, b, c, d, e, fz, nx, ny, nz)
    val2 = trapezes_3d(f, a, b, c, d, e, fz, 2*nx, 2*ny, 2*nz)
    return val1, abs(val2 - val1)


def simpson_3d(f, a, b, c, d, e, fz, nx, ny, nz):
    """
    Simpson 3D classique (nx,ny,nz doivent être pairs).
    """
    if nx % 2 != 0 or ny % 2 != 0 or nz % 2 != 0:
        raise ValueError("nx, ny, nz doivent être pairs pour Simpson 3D.")

    hx = (b - a) / nx
    hy = (d - c) / ny
    hz = (fz - e) / nz
    total = 0.0
    for i in range(nx+1):
        for j in range(ny+1):
            for k in range(nz+1):
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

                # Poids Simpson en z
                if k == 0 or k == nz:
                    wz = 1
                elif k % 2 == 1:
                    wz = 4
                else:
                    wz = 2

                x_ijk = a + i*hx
                y_ijk = c + j*hy
                z_ijk = e + k*hz
                total += wx * wy * wz * f(x_ijk, y_ijk, z_ijk)

    val = (hx * hy * hz / 27.0) * total  # car (1/3)^3 = 1/27, et on a 4/2/1 etc.
    return val

def simpson_3d_with_error(f, a, b, c, d, e, fz, nx, ny, nz):
    """
    Simpson 3D + estimation d'erreur par double maillage.
    """
    # S'assurer que nx,ny,nz sont pairs
    if nx % 2 != 0: nx += 1
    if ny % 2 != 0: ny += 1
    if nz % 2 != 0: nz += 1

    val1 = simpson_3d(f, a, b, c, d, e, fz, nx, ny, nz)
    val2 = simpson_3d(f, a, b, c, d, e, fz, 2*nx, 2*ny, 2*nz)
    return val1, abs(val2 - val1)

# ----------------------------------------------------------------------
# 2) Monte Carlo 3D (direct, erreur = variance / sqrt(N))
# ----------------------------------------------------------------------

def monte_carlo_3d(f, a, b, c, d, e, fz, N=100000):
    volume = (b - a)*(d - c)*(fz - e)
    s  = 0.0
    s2 = 0.0
    for _ in range(N):
        xr = a + (b-a)*random.random()
        yr = c + (d-c)*random.random()
        zr = e + (fz-e)*random.random()
        val = f(xr, yr, zr)
        s  += val
        s2 += val*val
    mean_f  = s / N
    mean_f2 = s2 / N
    var_f   = mean_f2 - mean_f**2
    integral = volume * mean_f
    err = volume * math.sqrt(var_f / N)
    return integral, err

# ----------------------------------------------------------------------
# 3) Gauss-Legendre 3D
#    => On compare n et n+2 pour une estimation d'erreur
# ----------------------------------------------------------------------

def gauss_legendre_3d(f, a, b, c, d, e, fz, n=5):
    """
    Quadrature Gauss-Legendre 3D : produit tensoriel des points/poids en x,y,z.
    Estimation d'erreur : compare n et (n+2).
    """
    # On calcule d'abord l'intégrale pour n
    xg, wg = np.polynomial.legendre.leggauss(n)
    yg, wg_y = np.polynomial.legendre.leggauss(n)
    zg, wg_z = np.polynomial.legendre.leggauss(n)

    half_x = 0.5*(b - a)
    mid_x  = 0.5*(a + b)
    half_y = 0.5*(d - c)
    mid_y  = 0.5*(c + d)
    half_z = 0.5*(fz - e)
    mid_z  = 0.5*(fz + e)

    total_n = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                xx = mid_x + half_x*xg[i]
                yy = mid_y + half_y*yg[j]
                zz = mid_z + half_z*zg[k]
                total_n += wg[i] * wg_y[j] * wg_z[k] * f(xx, yy, zz)

    val_n = half_x*half_y*half_z * total_n

    # On recalcule pour n+2
    n2 = n+2
    xg2, wg2 = np.polynomial.legendre.leggauss(n2)
    yg2, wg2_y = np.polynomial.legendre.leggauss(n2)
    zg2, wg2_z = np.polynomial.legendre.leggauss(n2)

    total_n2 = 0.0
    for i in range(n2):
        for j in range(n2):
            for k in range(n2):
                xx = mid_x + half_x*xg2[i]
                yy = mid_y + half_y*yg2[j]
                zz = mid_z + half_z*zg2[k]
                total_n2 += wg2[i] * wg2_y[j] * wg2_z[k] * f(xx, yy, zz)

    val_n2 = half_x*half_y*half_z * total_n2

    err_est = abs(val_n2 - val_n)
    return val_n, err_est

# ----------------------------------------------------------------------
# 4) Romberg 3D (approche nestée : Romberg 1D sur x, puis y, puis z)
# ----------------------------------------------------------------------

def romberg_1d(f, a, b, max_iter=5):
    """
    Romberg 1D : retourne (val, err_est).
    """
    R = np.zeros((max_iter, max_iter), dtype=float)
    h = (b-a)
    R[0,0] = 0.5*h*(f(a) + f(b))
    for k in range(1, max_iter):
        n_sub = 2**k
        h_ = (b-a)/n_sub
        subtotal = 0.0
        for i in range(1, 2**(k-1)+1):
            x_mid = a + (2*i - 1)*h_
            subtotal += f(x_mid)
        R[k,0] = R[k-1,0]/2.0 + subtotal*h_
        for j in range(1, k+1):
            R[k,j] = ( (4**j)*R[k,j-1] - R[k-1,j-1] )/(4**j - 1)
    val = R[max_iter-1,max_iter-1]
    err = abs(val - R[max_iter-2,max_iter-2]) if max_iter>1 else 0.0
    return val, err

def romberg_3d(f, a, b, c, d, e, fz, max_iter=5):
    """
    Romberg 3D : on fait un Romberg 1D en x pour chaque (y,z),
    puis un Romberg 1D en y pour intégrer g(y,z), puis en z.
    Approche "nestée" (attention au coût).
    Retourne (val, err_est).
    """

    # On veut G(z) = ∫[y=c..d] [ ∫[x=a..b] f(x,y,z) dx ] dy
    # puis integrer G(z) de e..fz
    # => On va procéder en 2 étapes :

    # 1) Intégration en x "romberg" pour chaque (y,z)
    #    => on définit h(y,z) = ∫[x=a..b] f(x,y,z) dx
    def h(y, z):
        valx, _ = romberg_1d(lambda xx: f(xx, y, z), a, b, max_iter=max_iter)
        return valx

    # 2) Intégration en y "romberg" pour chaque z
    #    => on définit H(z) = ∫[y=c..d] h(y,z) dy
    def H(z):
        valy, _ = romberg_1d(lambda yy: h(yy, z), c, d, max_iter=max_iter)
        return valy

    # 3) Romberg final en z
    #    => ∫[z=e..fz] H(z) dz
    val_z, err_z = romberg_1d(H, e, fz, max_iter=max_iter)

    return val_z, err_z

# ----------------------------------------------------------------------
# 5) Simpson adaptatif 3D (approche nestée)
# ----------------------------------------------------------------------

def adaptive_simpson_1d(f, a, b, tol=1e-6, max_depth=15):
    """
    Simpson adaptatif 1D. Retourne (val, err_est).
    """
    def simpson_segment(left, right):
        m = 0.5*(left + right)
        return (right-left)/6.0*(f(left)+4.0*f(m)+f(right))

    def recurse(left, right, tol, depth, whole):
        m = 0.5*(left+right)
        left_simp  = simpson_segment(left, m)
        right_simp = simpson_segment(m, right)
        delta = left_simp + right_simp - whole
        if (depth<=0) or (abs(delta)<15*tol):
            return (left_simp+right_simp)+delta/15.0, abs(delta/15.0)
        val_l, err_l = recurse(left, m, tol/2.0, depth-1, left_simp)
        val_r, err_r = recurse(m, right, tol/2.0, depth-1, right_simp)
        return val_l+val_r, err_l+err_r

    whole = simpson_segment(a, b)
    val, err = recurse(a, b, tol, max_depth, whole)
    return val, err

def adaptive_simpson_3d(f, a, b, c, d, e, fz, tol=1e-6, max_depth=10):
    """
    Simpson adaptatif 3D, "nesté" :
      - On fait adapt. Simpson 1D en x pour chaque (y,z)
      - Puis adapt. Simpson 1D en y pour la fonction "intégrée sur x"
      - Puis adapt. Simpson 1D en z pour la fonction "intégrée sur x,y".
    Retourne (val, err_est).
    """

    # 1) Intégration en x "adaptative simpson" pour chaque (y,z)
    def integrate_x(y, z, tol_x):
        valx, _ = adaptive_simpson_1d(lambda xx: f(xx,y,z), a, b, tol=tol_x, max_depth=max_depth)
        return valx

    # 2) Intégration en y "adaptative simpson" pour chaque z
    def Gz(z, tol_y):
        def g(y):
            return integrate_x(y, z, tol_x=tol_y/10)  # on subdivise la tol
        valy, _ = adaptive_simpson_1d(g, c, d, tol=tol_y, max_depth=max_depth)
        return valy

    # 3) Intégration en z "adaptative simpson"
    def H(z):
        # On appelle Gz(z) avec tol dans la 2e étape
        return Gz(z, tol_y=tol/10)

    valz, errz = adaptive_simpson_1d(H, e, fz, tol=tol, max_depth=max_depth)

    return valz, errz

# ----------------------------------------------------------------------
# 6) Exemples d'utilisation
# ----------------------------------------------------------------------

if __name__ == "__main__":

    # Exemple : f(x,y,z) = exp(-(x^2 + y^2 + z^2)), sur [0,1]^3
    def f_test_3d(x, y, z):
        return math.exp(-(x**2 + y**2 + z**2))

    a, b = 0.0, 1.0
    c, d = 0.0, 1.0
    e, fz= 0.0, 1.0

    print("=== Rectangle 3D ===")
    val_rect, err_rect = rectangle_3d_with_error(f_test_3d, a, b, c, d, e, fz, 8, 8, 8)
    print(f"val={val_rect:.6f}, err_est={err_rect:.2e}")

    print("\n=== Trapèzes 3D ===")
    val_trap, err_trap = trapezes_3d_with_error(f_test_3d, a, b, c, d, e, fz, 8, 8, 8)
    print(f"val={val_trap:.6f}, err_est={err_trap:.2e}")

    print("\n=== Simpson 3D ===")
    val_simp, err_simp = simpson_3d_with_error(f_test_3d, a, b, c, d, e, fz, 8, 8, 8)
    print(f"val={val_simp:.6f}, err_est={err_simp:.2e}")

    print("\n=== Romberg 3D ===")
    val_rom, err_rom = romberg_3d(f_test_3d, a, b, c, d, e, fz, max_iter=4)
    print(f"val={val_rom:.6f}, err_est={err_rom:.2e}")

    print("\n=== Monte Carlo 3D ===")
    val_mc, err_mc = monte_carlo_3d(f_test_3d, a, b, c, d, e, fz, N=50000)
    print(f"val={val_mc:.6f}, err_est={err_mc:.2e}")

    print("\n=== Gauss-Legendre 3D ===")
    val_gauss, err_gauss = gauss_legendre_3d(f_test_3d, a, b, c, d, e, fz, n=5)
    print(f"val={val_gauss:.6f}, err_est={err_gauss:.2e}")

    print("\n=== Simpson adaptatif 3D ===")
    val_asimp, err_asimp = adaptive_simpson_3d(f_test_3d, a, b, c, d, e, fz, tol=1e-3, max_depth=6)
    print(f"val={val_asimp:.6f}, err_est={err_asimp:.2e}\n")

def best_method_3d(f_3d, a, b, c, d, e, fz):
    """
    Compare plusieurs méthodes 3D:
      - Rectangle
      - Trapèzes
      - Simpson (classique)
      - Romberg
      - Monte Carlo
      - Gauss-Legendre
      - Simpson adaptatif
    Chacune doit retourner (valeur, erreur).
    
    f_3d : fonction f(x,y,z) -> float
    [a,b] x [c,d] x [e,fz] : domaine d'intégration
    Retourne (best_name, best_val, best_err), où best_err est la plus petite erreur estimée.
    """

    results = []

    nx = ny = nz = 10
    val_rect, err_rect = rectangle_3d_with_error(f_3d, a, b, c, d, e, fz, nx, ny, nz)
    results.append(("Rectangle_3D", val_rect, err_rect))

    val_trap, err_trap = trapezes_3d_with_error(f_3d, a, b, c, d, e, fz, nx, ny, nz)
    results.append(("Trapezes_3D", val_trap, err_trap))

    #    (nx, ny, nz) doivent être pairs
    if nx % 2 != 0: nx += 1
    if ny % 2 != 0: ny += 1
    if nz % 2 != 0: nz += 1
    val_simp, err_simp = simpson_3d_with_error(f_3d, a, b, c, d, e, fz, nx, ny, nz)
    results.append(("Simpson_3D", val_simp, err_simp))

    val_rom, err_rom = romberg_3d(f_3d, a, b, c, d, e, fz)
    results.append(("Romberg_3D", val_rom, err_rom))


    val_mc, err_mc = monte_carlo_3d(f_3d, a, b, c, d, e, fz, N=50000)
    results.append(("MonteCarlo_3D", val_mc, err_mc))

    val_gauss, err_gauss = gauss_legendre_3d(f_3d, a, b, c, d, e, fz, n=5)
    results.append(("Gauss_3D", val_gauss, err_gauss))

    val_asimp, err_asimp = adaptive_simpson_3d(f_3d, a, b, c, d, e, fz, tol=1e-6, max_depth=10)
    results.append(("AdaptSimpson_3D", val_asimp, err_asimp))

    # On choisit la méthode avec la plus petite erreur estimée
    best = min(results, key=lambda x: x[2])  # x = (nom, valeur, erreur)
    best_name, best_val, best_err = best
    return best_name, best_val, best_err

if __name__ == "__main__":
    # Exemple de fonction 3D
    def f_test_3d(x, y, z):
        return np.exp(-(x**2 + y**2 + z**2))

    # Bornes
    a, b = 0, 1
    c, d = 0, 1
    e, fz = 0, 1

    best_name, best_val, best_err = best_method_3d(f_test_3d, a, b, c, d, e, fz)
    print(f"Méthode gagnante : {best_name}")
    print(f"Valeur approx    : {best_val:.6f}")
    print(f"Erreur estimée   : {best_err:.2e}")
