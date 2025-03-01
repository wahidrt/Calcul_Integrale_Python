#----------------------------------
#-----------les bibiotheque--------------
#----------------------------------
import numpy as np # type: ignore
import random
import math
import matplotlib.pyplot as plt # type: ignore

#----------------------------------
#-----------les Fonction--------------
#----------------------------------

def f(x): return x - 1

def g(x): return np.exp(np.sin(x)) + np.log(np.abs(np.cos(x)) + 1e-6)  # Évite log(0) ou log(valeur négative)

def f2(x, y): return x * y

def g2(x, y): return np.sin(xy) * np.cos(x)

def f3(x, y, z): return x * y * z

def g3(x, y, z): return np.sin(xy) * np.cos(x) * z
#----------------------------------
#-----------adaptive_simpson--------------
#----------------------------------

# Méthode de Simpson adaptative 1D avec estimation d'erreur
def adaptive_simpson(f, a, b, tol=1e-6, depth=10):
    """ Approximation adaptative de l'intégrale de f sur [a, b] avec tolérance tol. """
    
    def simpson_rule(f, a, b):
        """ Méthode de Simpson basique sur [a,b] """
        c = (a + b) / 2
        return (b - a) / 6 * (f(a) + 4 * f(c) + f(b))

    def recursive_simpson(f, a, b, tol, depth, whole):
        """ Méthode récursive pour améliorer la précision """
        c = (a + b) / 2
        left = simpson_rule(f, a, c)
        right = simpson_rule(f, c, b)
        if depth <= 0 or abs(left + right - whole) < 15 * tol:
            return left + right + (left + right - whole) / 15  # Correction d'erreur
        return recursive_simpson(f, a, c, tol / 2, depth - 1, left) + recursive_simpson(f, c, b, tol / 2, depth - 1, right)

    initial = simpson_rule(f, a, b)
    result = recursive_simpson(f, a, b, tol, depth, initial)
    error = abs(result - initial) / 15  # Estimation de l'erreur
    
    return result, error

# Méthode de Simpson adaptative 2D
def adaptive_simpson_2d(f, a, b, c, d, tol=1e-6):
    def inner_integral(y):
        return adaptive_simpson(lambda x: f(x, y), a, b, tol)[0]
    
    result, error = adaptive_simpson(inner_integral, c, d, tol)
    return result, error

# Méthode de Simpson adaptative 3D
def adaptive_simpson_3d(f, a, b, c, d, e, fz, tol=1e-6):
    def inner_integral_yz(y, z):
        return adaptive_simpson(lambda x: f(x, y, z), a, b, tol)[0]

    def inner_integral_z(z):
        return adaptive_simpson(lambda y: inner_integral_yz(y, z), c, d, tol)[0]

    result, error = adaptive_simpson(inner_integral_z, e, fz, tol)
    return result, error

#----------------------------------
#-----------Simpson--------------
#----------------------------------
def simpson(f, a, b, n):
    if n % 2 != 0: raise ValueError("Le nombre n'est pas pair.")
    
    h = (b - a) / n
    x = [a + i * h for i in range(n+1)]
    y = [f(x[i]) for i in range(n+1)]
    
    integral = y[0] + y[-1] 
    
    for i in range(1, n, 2):
        integral += 4 * y[i]
    
    for i in range(2, n-1, 2):
        integral += 2 * y[i]
    
    integral *= h / 3
    error = -((b - a)**5 / (180 * n**4)) * f_fourth_derivative((a + b) / 2)
    return integral, abs(error)

# Méthode de Simpson 2D avec estimation d'erreur
def simpson_2d(f, a, b, c, d, nx, ny):
    if nx % 2 == 1: nx += 1
    if ny % 2 == 1: ny += 1
    hx = (b - a) / nx
    hy = (d - c) / ny
    integral = 0

    for i in range(nx + 1):
        for j in range(ny + 1):
            x = a + i * hx
            y = c + j * hy
            weight = 1
            if i == 0 or i == nx: weight *= 1
            elif i % 2 == 1: weight *= 4
            else: weight *= 2
            if j == 0 or j == ny: weight *= 1
            elif j % 2 == 1: weight *= 4
            else: weight *= 2
            integral += weight * f(x, y)

    integral *= (hx * hy) / 9
    error = -((b - a) * (d - c) / (180 * nx**4 * ny**4)) * f2_fourth_derivative((a + b) / 2, (c + d) / 2)
    return integral, abs(error)

# Méthode de Simpson 3D avec estimation d'erreur
def simpson_3d(f, a, b, c, d, e, fz, nx, ny, nz):
    if nx % 2 == 1: nx += 1
    if ny % 2 == 1: ny += 1
    if nz % 2 == 1: nz += 1
    hx = (b - a) / nx
    hy = (d - c) / ny
    hz = (fz - e) / nz
    integral = 0

    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                x = a + i * hx
                y = c + j * hy
                z = e + k * hz
                weight = 1
                if i == 0 or i == nx: weight *= 1
                elif i % 2 == 1: weight *= 4
                else: weight *= 2
                if j == 0 or j == ny: weight *= 1
                elif j % 2 == 1: weight *= 4
                else: weight *= 2
                if k == 0 or k == nz: weight *= 1
                elif k % 2 == 1: weight *= 4
                else: weight *= 2
                integral += weight * f(x, y, z)

    integral *= (hx * hy * hz) / 27
    error = -((b - a) * (d - c) * (fz - e) / (180 * nx**4 * ny**4 * nz**4)) * f3_fourth_derivative((a + b) / 2, (c + d) / 2, (e + fz) / 2)
    return integral, abs(error)

#----------------------------------
#-----------Romberg--------------
#----------------------------------

def romberg(f, a, b, n=5):
    R = np.zeros((n, n))
    for k in range(n):
        R[k, 0] = trapezes(f, a, b, 2**k)
        for j in range(1, k + 1):
            R[k, j] = (4**j * R[k, j-1] - R[k-1, j-1]) / (4**j - 1)
    error = abs(R[-1, -1] - R[-2, -2]) if n > 1 else 0  # Estimation de l'erreur par la dernière correction
    return R[-1, -1], error

def romberg_2d(f, a, b, c, d, n=5):
    def inner_integral(y):
        return romberg(lambda x: f(x, y), a, b, n)

    result, error = romberg(inner_integral, c, d, n)
    return result, abs(error)

def romberg_3d(f, a, b, c, d, e, fz, n=5):
    def inner_integral_yz(y, z):
        return romberg(lambda x: f(x, y, z), a, b, n)

    def inner_integral_z(z):
        return romberg(lambda y: inner_integral_yz(y, z), c, d, n)

    result, error = romberg(inner_integral_z, e, fz, n)
    return result, abs(error)


#----------------------------------
#-----------Monte_Carlo--------------
#----------------------------------

def monte_carlo(f, a, b, n):
    somme_f = 0  # Somme des valeurs f(x_i)
    somme_f2 = 0  # Somme des carrés des valeurs f(x_i)
    
    for i in range(n):
        x_i = a + (b - a) * random.random()  # Génère un x_i aléatoire dans [a, b]
        f_xi = f(x_i)  # Évalue f(x_i)
        
        somme_f += f_xi
        somme_f2 += f_xi ** 2  # Pour le calcul de la variance
    
    moyenne_f = somme_f / n  # Moyenne de f(x)
    variance_f = (somme_f2 / n) - (moyenne_f ** 2)  # Variance de f(x)
    
    I = (b - a) * moyenne_f  # Approximation de l'intégrale
    erreur = (b - a) * math.sqrt(variance_f / n)  # Erreur statistique (écart-type)

    return I, erreur

def monte_carlo_2d(f, a, b, c, d, n):
    somme_f = 0
    somme_f2 = 0  # Pour calculer la variance

    for _ in range(n):
        x_i = a + (b - a) * random.random()
        y_i = c + (d - c) * random.random()
        f_xi_yi = f(x_i, y_i)
        
        somme_f += f_xi_yi
        somme_f2 += f_xi_yi ** 2

    moyenne_f = somme_f / n
    variance_f = (somme_f2 / n) - (moyenne_f ** 2)
    
    I = (b - a) * (d - c) * moyenne_f  # Estimation de l'intégrale
    erreur = (b - a) * (d - c) * math.sqrt(variance_f / n)  # Erreur statistique

    return I, erreur

def monte_carlo_3d(f, a, b, c, d, e, fz, n):
    somme_f = 0
    somme_f2 = 0  # Pour la variance

    for _ in range(n):
        x_i = a + (b - a) * random.random()
        y_i = c + (d - c) * random.random()
        z_i = e + (fz - e) * random.random()
        f_xi_yi_zi = f(x_i, y_i, z_i)
        
        somme_f += f_xi_yi_zi
        somme_f2 += f_xi_yi_zi ** 2

    moyenne_f = somme_f / n
    variance_f = (somme_f2 / n) - (moyenne_f ** 2)
    
    I = (b - a) * (d - c) * (fz - e) * moyenne_f  # Estimation de l'intégrale
    erreur = (b - a) * (d - c) * (fz - e) * math.sqrt(variance_f / n)  # Erreur statistique

    return I, erreur


#----------------------------------
#-----------rectangle--------------
#----------------------------------

def rectangle(f, a, b, n):
    h = (b - a) / n
    integral = 0
    for i in range(n):
        x = a + i * h
        integral += f(x)
    integral *= h
    error = -((b - a)**2 / (2 * n)) * f_second_derivative((a + b) / 2)
    return integral, abs(error)

def rectangle_2d(f, a, b, c, d, nx, ny):
    hx = (b - a) / nx
    hy = (d - c) / ny
    integral = 0

    for i in range(nx):
        for j in range(ny):
            x = a + (i + 0.5) * hx  # Point milieu en x
            y = c + (j + 0.5) * hy  # Point milieu en y
            integral += f(x, y)
    integral *= hx * hy
    error = -((b - a) * (d - c) / (2 * nx * ny)) * f2_second_derivative((a + b) / 2, (c + d) / 2)
    return integral, abs(error)


def rectangle_3d(f, a, b, c, d, e, fz, nx, ny, nz):
    hx = (b - a) / nx
    hy = (d - c) / ny
    hz = (fz - e) / nz
    integral = 0

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = a + (i + 0.5) * hx  # Point milieu en x
                y = c + (j + 0.5) * hy  # Point milieu en y
                z = e + (k + 0.5) * hz  # Point milieu en z
                integral += f(x, y, z)

    integral *= hx * hy * hz
    error = -((b - a) * (d - c) * (fz - e) / (2 * nx * ny * nz)) * f3_second_derivative((a + b) / 2, (c + d) / 2, (e + fz) / 2)
    return integral, abs(error)



#----------------------------------
#-----------trapezes---------------
#----------------------------------

def trapezes(f, a, b, n):
    h = (b - a) / n
    integral = (f(a) + f(b)) / 2
    for i in range(1, n):
        integral += f(a + i * h)
    integral *= h
    error = -((b - a)**3 / (12 * n**2)) * f_second_derivative((a + b) / 2)
    return integral, abs(error)

def trapezes_2d(f, a, b, c, d, nx, ny):
    hx = (b - a) / nx
    hy = (d - c) / ny
    integral = 0

    for i in range(nx + 1):
        for j in range(ny + 1):
            x = a + i * hx
            y = c + j * hy
            weight = 1

            if i == 0 or i == nx:
                weight *= 0.5
            if j == 0 or j == ny:
                weight *= 0.5

            integral += weight * f(x, y)

    integral *= hx * hy
    error = -((b - a) * (d - c) / (12 * nx * ny)) * f2_second_derivative((a + b) / 2, (c + d) / 2)
    return integral, abs(error)

def trapezes_3d(f, a, b, c, d, e, fz, nx, ny, nz):
    hx = (b - a) / nx
    hy = (d - c) / ny
    hz = (fz - e) / nz
    integral = 0
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                x = a + i * hx
                y = c + j * hy
                z = e + k * hz
                weight = 1
                if i == 0 or i == nx: weight *= 0.5
                if j == 0 or j == ny: weight *= 0.5
                if k == 0 or k == nz: weight *= 0.5
                integral += weight * f(x, y, z)
    integral *= hx * hy * hz
    error = -((b - a) * (d - c) * (fz - e) / (12 * nx * ny * nz)) * f3_second_derivative((a + b) / 2, (c + d) / 2, (e + fz) / 2)
    return integral, abs(error)


#----------------------------------
#-----------gauss--------------
#----------------------------------
def gauss(f, a, b, n):
    x, y = np.polynomial.legendre.leggauss(n)
    x_s = 0.5 * (b - a) * x + 0.5 * (b + a)
    integral = np.sum(y * f(x_s)) * 0.5 * (b - a)
    return integral

def gauss_2d(f, a, b, c, d, n=5):
    # Points et poids de Gauss-Legendre
    x_points, x_weights = np.polynomial.legendre.leggauss(n)
    y_points, y_weights = np.polynomial.legendre.leggauss(n)

    # Transformation d'intervalle [-1,1] vers [a,b] et [c,d]
    x_scaled = 0.5 * (b - a) * x_points + 0.5 * (b + a)
    y_scaled = 0.5 * (d - c) * y_points + 0.5 * (d + c)

    integral = 0
    for i in range(n):
        for j in range(n):
            integral += x_weights[i] * y_weights[j] * f(x_scaled[i], y_scaled[j])

    return 0.25 * (b - a) * (d - c) * integral

def gauss_3d(f, a, b, c, d, e, fz, n=5):
    # Points et poids de Gauss-Legendre
    x_points, x_weights = np.polynomial.legendre.leggauss(n)
    y_points, y_weights = np.polynomial.legendre.leggauss(n)
    z_points, z_weights = np.polynomial.legendre.leggauss(n)

    # Transformation d'intervalle [-1,1] vers [a,b], [c,d], [e,f]
    x_scaled = 0.5 * (b - a) * x_points + 0.5 * (b + a)
    y_scaled = 0.5 * (d - c) * y_points + 0.5 * (d + c)
    z_scaled = 0.5 * (fz - e) * z_points + 0.5 * (fz + e)

    integral = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                integral += x_weights[i] * y_weights[j] * z_weights[k] * f(x_scaled[i], y_scaled[j], z_scaled[k])

    return 0.125 * (b - a) * (d - c) * (fz - e) * integral



# Définition d'une fonction pour éviter la répétition de code
def afficher_approximation(f, nom_f):
    print(f"\nApproximations pour {nom_f}(x) :")
    print(f" - Méthode de Gauss       : {gauss(f, 0, 1, 1000):.6f}")
    print(f" - Méthode de Simpson     : {simpson(f, 0, 1, 1000):.6f}")
    
    monte_carlo_result = monte_carlo(f, 0, 1, 1000)
    print(f" - Méthode de Monte Carlo : {monte_carlo_result[0]:.6f} (erreur estimée: {monte_carlo_result[1]:.6f})")
    
    print(f" - Méthode des Trapèzes   : {trapezes(f, 0, 1, 1000):.6f}")
    print(f" - Méthode des Rectangles : {rectangle(f, 0, 1, 1000):.6f}")

# Affichage des approximations pour f(x)
afficher_approximation(f, "f")

# Affichage des approximations pour g(x)
afficher_approximation(g, "g")

#----------------------------------
#-----------methods-----------------
#----------------------------------
def compare_methods(f, a, b, n=1000):
    results = {
        "Simpson": simpson(f, a, b, n),
        "Monte Carlo": monte_carlo(f, a, b, n),
        "Trapèzes": trapezes(f, a, b, n),
        "Rectangle": rectangle(f, a, b, n),
        "Gauss": gauss(f, a, b, n),
        "Romberg": romberg(f, a, b, 5),
        "Adaptive Simpson": adaptive_simpson(f, a, b)
    }
    
    print("\nComparaison des méthodes :")
    for method, value in results.items():
        if isinstance(value, tuple):
            print(f"{method}: {value[0]:.6f} (erreur estimée: {value[1]:.6f})")
        else:
            print(f"{method}: {value:.6f}")

#----------------------------------
#--------------------------------------
#----------------------------------        
def plot_methods(f, a, b, n=100):
    x = np.linspace(a, b, 1000)
    y = [f(xi) for xi in x]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label="f(x)", color='blue')
    plt.fill_between(x, y, alpha=0.2, color='blue', label="Aire sous la courbe")
    plt.legend()
    plt.grid()
    plt.title("Intégration numérique d'une fonction")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

# Appel de la fonction pour visualiser les fonctions intégrées
plot_methods(f, 0, 1)
plot_methods(g, 0, 1)


# Comparaison des méthodes d'intégration pour f(x)
print("\nComparaison des méthodes pour f(x) :")
compare_methods(f, 0, 1)

# Comparaison des méthodes d'intégration pour g(x)
print("\nComparaison des méthodes pour g(x) :")
compare_methods(g, 0, 1)



