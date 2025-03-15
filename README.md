# 🧮 Calcul d'Intégrales Numériques en Python

## 📌 Description
Ce projet implémente plusieurs méthodes d'intégration numérique pour le calcul d'intégrales :
- **1D** : Intégration de fonctions f(x)
- **2D** : Intégration sur un domaine [a,b] x [c,d]
- **3D** : Intégration sur un volume [a,b] x [c,d] x [e,f]

Les méthodes utilisées incluent :
- **Méthode des rectangles**
- **Méthode des trapèzes**
- **Méthode de Simpson**
- **Méthode de Romberg**
- **Monte Carlo**
- **Quadrature de Gauss-Legendre**
- **Simpson adaptatif**

L'objectif est de comparer ces méthodes en termes de précision et d'erreur estimée.

---

## 🏗️ Structure du projet
📂 **Fichiers principaux** :
- `Calcule_d_Integrale_Numerique_SAMER.py` → Intégration 1D
- `Calcule_d_Integrale_Numerique_2D_SAMER.py` → Intégration 2D
- `Calcule_d_Integrale_Numerique_3D_SAMER.py` → Intégration 3D

📂 **Méthodes implémentées** :
- **1D** : `midpoint_rectangle`, `trapeze`, `simpson`, `romberg_integration`, `monte_carlo_integration`, `gauss_legendre_integration`, `adaptive_simpson_integration`
- **2D** : `rectangle_2d_midpoint`, `trapeze_2d`, `simpson_2d`, `romberg_2d`, `monte_carlo_2d`, `adaptive_simpson_2d`
- **3D** : `rectangle_3d`, `trapezes_3d`, `simpson_3d`, `romberg_3d`, `monte_carlo_3d`, `gauss_legendre_3d`, `adaptive_simpson_3d`

---

## 🚀 Installation et Exécution
### 1️⃣ **Cloner le dépôt**
```bash
git clone https://github.com/wahidrt/Calcul_Integrale_Python.git
cd Calcul_Integrale_Python
