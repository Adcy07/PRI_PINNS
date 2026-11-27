import numpy as np

def H(x):
    H0 = 1
    return H0 + (x**2)/2

def system_creation(x, H0):
    x = np.asarray(x)
    n = len(x)
    h = x[1] - x[0]

    A = np.zeros((n, n))
    b = np.zeros(n)

   
    A[0, 0] = 1
    A[n-1, n-1] = 1

    
    for i in range(1, n-1):
        a_minus = H(0.5 * (x[i-1] + x[i]))**3   # a_{i-1/2}
        a_plus  = H(0.5 * (x[i] + x[i+1]))**3   # a_{i+1/2}

        A[i, i-1] = a_minus / h**2
        A[i, i]   = -(a_minus + a_plus) / h**2
        A[i, i+1] = a_plus / h**2
        b[i] = x[i]                              # dH/dX = X

    return A, b

def solve_iterative(x, H0, eps=1e-9, maxit=1_000_00):
    """
    Schéma itératif avec correction de complémentarité :
        r^(k)      = b - A p^(k)
        p^(k+1)    = p^(k) + r^(k) / a_ii
        si p^(k+1) <= 0 : p^(k+1) = 0
    Arrêt : ||r_eff|| <= eps, où r_eff tient compte de la complémentarité.
    """
    A, b = system_creation(x, H0)
    d = np.diag(A).copy()
    m = len(b)
    p = np.zeros(m)                        # init : p = 0

    for k in range(maxit):
        r = b - A @ p

        # Résidu de complémentarité : p>0 => r doit être nul,
        # p=0 => r doit être >= 0 (seule la partie négative compte)
        r_eff = np.where(p > 0, r, np.minimum(r, 0.0))
        if np.linalg.norm(r_eff) <= eps:
            break

        p = p + r / d                      # mise à jour
        p = np.maximum(p, 0.0)             # correction de complémentarité

    
    
    return p


def solve_P_inverse_main(x, H0):

    A, b = system_creation(x, H0)
    P_int = np.linalg.solve(A, b)

    P = np.zeros(len(x))
    return P_int

def residual(x, P, H0):
    """Résidu de l'équation discrète aux nœuds intérieurs."""
    h = x[1] - x[0]
    H = H0 + (x**2) / 2.0
    a = (0.5 * (H[:-1] + H[1:])) ** 3
    flux = a * (P[1:] - P[:-1]) / h
    lhs = (flux[1:] - flux[:-1]) / h
    return np.max(np.abs(lhs - x[1:-1]))

x_diff = np.linspace(-5, 1, 100)
H0 = 1

P_main = solve_iterative(x_diff, H0) 