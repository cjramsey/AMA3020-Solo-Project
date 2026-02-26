import math
import warnings
import functools

import numpy as np


# Helper functions

def compensated_summation(x):
    '''
    Kahan compensated summation over a numpy array.
    '''
    s = 0.0
    e = 0.0
    for xi in x:
        temp = s
        y = xi + e
        s = temp + y
        e = (temp - s) + y
    return s


# Taylor Series 

@functools.lru_cache(maxsize=128)
def erf_taylor_coeff(k):
    '''
    Return the k-th coefficient of the Taylor (Maclaurin) series of erf(x).
    '''
    return (2 / math.sqrt(math.pi)) * ((-1)**k) / (math.factorial(k) * (2*k + 1))

@functools.lru_cache(maxsize=128)
def erf_taylor_coefficients(max_degree):
    '''
    Generate Taylor coefficients a_k for erf(x) up to max_degree.

    Even powers are zero since erf is odd.
    '''
    a = np.zeros(max_degree + 1)

    for k in range((max_degree // 2) + 1):
        power = 2*k + 1
        if power <= max_degree:
            a[power] = erf_taylor_coeff(k)

    return a


def erf_taylor(x, tol=1e-15, max_iter=50):
    '''
    Approximate erf(x) using the Taylor (Maclaurin) series.

    Adds terms until |term| < tol (checked before adding the final term)
    or max_iter terms have been summed. Issues a warning if max_iter is
    exhausted without convergence, since the series diverges for large x.

    Parameters:
    x (float)
    tol (float): Absolute convergence threshold (default 1e-15).
    max_iter (int): Maximum number of terms to sum (default 50).
                    Pass tol=0 and vary max_iter to evaluate exactly N terms.
    '''
    if x < 0:
        return -erf_taylor(-x, tol=tol, max_iter=max_iter)
    
    s = 0.0
    for k in range(max_iter):
        term = erf_taylor_coeff(k) * x**(2*k + 1)
        if abs(term) < tol:
            return s
        
        s += term

        
    warnings.warn(
        f"erf_taylor did not converge for x={x} within max_iter={max_iter} "
        "terms. The series diverges for large x; the returned value is unreliable.",
        RuntimeWarning,
        stacklevel=2,
    )
    return s


def erfc_asymptotic(x):
    '''
    Asymptotic expansion for erfc(x).

    Truncates after N = min(floor(x^2 + 1/2), 100) terms, which minimises
    the truncation error for a given x. Valid (and accurate) for x >= 4.
    '''
    if x <= 0:
        raise ValueError("Asymptotic expansion valid for x > 0.")

    N = min(int(math.floor(x**2 + 0.5)), 100)

    term = 1.0
    s = 1.0

    for n in range(1, N):
        term *= (2*n - 1) / (2*x**2)
        s += (-1)**n * term

    prefactor = math.exp(-x**2) / (x * math.sqrt(math.pi))
    return prefactor * s


# Pade Approximation

def pade_approximation(m, n, a):
    '''
   Construct the [m/n] Padé approximation from Taylor coefficients.

    The approximation r(x) = P(x)/Q(x) matches the Taylor series of f(x)
    to order m + n at x = 0.

    Parameters:
        m (int): Degree of the numerator polynomial P.
        n (int): Degree of denominator polynomial Q.
        a (np.array): Taylor coefficients a[0], a[1], ..., a[m+n] of f.

    Returns
        p (np.array) Numerator coefficients p[0] + p[1]*x + ...
        q (ndarray): enominator coefficients q[0]=1, q[1]*x + ... (normalised)
    '''
    if len(a) < m + n + 1:
        raise ValueError("Must have at least m + n + 1 Taylor coefficients.")
    
    a = np.array(a)

    # Solve linear system for denominator coefficients
    A = np.zeros((n, n))
    b = np.zeros(n)

    for i in range(n):
        for j in range(n):
            A[i, j] = a[m + i - j]
        b[i] = -a[m + i + 1]

    q_tail = np.linalg.solve(A, b)
    q = np.concatenate(([1.0], q_tail))

    # Compute numerator coefficients
    p = np.zeros(m + 1)
    for k in range(m + 1):
        s = 0.0
        for j in range(min(k, n) + 1):
            s += q[j] * a[k - j]
        p[k] = s

    return p, q


def evaluate_rational(x, p, q):
    '''
    Evaluate rational function r(x) = P(x) / Q(x),
    where p is the coefficients of P(x) and q is the coefficients of Q(x).

    Coefficients are in ascending order.
    '''
    numerator = np.polyval(p[::-1], x)
    denominator = np.polyval(q[::-1], x)

    return numerator / denominator


@functools.lru_cache(maxsize=128)
def _erf_pade_coeffs(m, n):
    '''
    Cached computation of the [m/n] Padé coefficients for erf.

    Separating coefficient computation from evaluation and caching on (m, n)
    means that repeated calls to erf_pade with the same approximant order
    (e.g. inside np.vectorize) do not recompute the linear system each time.
    '''
    a = erf_taylor_coefficients(m + n)
    return pade_approximation(m, n, a)


def erf_pade(x, m, n):
    '''
    Evaluate erf(x) using the [m/n] Padé approximation.

    Padé coefficients are computed once per (m, n) pair and cached.
    '''
    if x < 0:
        return -erf_pade(-x, m, n)

    p, q = _erf_pade_coeffs(m, n)
    return evaluate_rational(x, p, q)


# Quadrature Methods


def gauss_legendre_quadrature(f, a, b, n=20):
    '''Gauss-Legendre quadrature of f over [a, b].'''
    pivots, weights = np.polynomial.legendre.leggauss(n)

    t = ((b - a) * pivots + (b + a)) / 2.0
    fvals = f(t)

    integral = compensated_summation(weights * fvals)
    integral *= (b - a) / 2.0

    return integral


def erf_gauss_legendre(x, n=20):
    '''
    Compute erf(x) using n-point Gauss-Legendre quadrature.

    Suitable for small and moderate x.
    '''
    if x < 0:
        return -erf_gauss_legendre(-x, n)
    
    return (2 / math.sqrt(math.pi)) * gauss_legendre_quadrature(lambda x: np.exp(-x*x), 0, x, n=n)


def erf_gauss_legendre_error_bound(x, n):
    '''Estimate upper bound for the truncation error of Gauss-Legendre quadrature.'''
    C = 1.086435
    error_bound = C
    error_bound *= x**(2*n + 1) / (2*n + 1)
    error_bound *= math.factorial(n)**4 / math.factorial(2*n)**(5/2)
    error_bound *= 2**n
    return error_bound


def adaptive_gauss_legendre(f, a, b, tol=1e-12, n=8, max_depth=10):
    '''
    Adaptive Gauss-Legendre quadrature of f on [a, b].

    Recursively halves sub-intervals where the difference between the
    one-panel and two half-panel estimates exceeds the local tolerance.

    Parameters
    ----------
    f (callable): Vectorised integrand f(t).
    a, b (float): Integration limits.
    tol (float): Absolute error tolerance (default 1e-12).
    n (int): Number of GL points per panel (default 8).
    max_depth (int): Maximum recursion depth (default 10).
    '''
    pivots, weights = np.polynomial.legendre.leggauss(n)

    def quadrature_interval(lo, hi):
        t = 0.5*(hi - lo)*pivots + 0.5*(lo + hi)
        return 0.5*(hi - lo) * compensated_summation(weights * f(t))
    
    def recurse(lo, hi, depth, tol):
        mid = 0.5*(lo + hi)

        Q = quadrature_interval(lo, hi)
        QL = quadrature_interval(lo, mid)
        QR = quadrature_interval(mid, hi)

        if depth >= max_depth:
            return QL + QR
        
        # Simple heuristic for error estimate
        if abs(Q - (QL + QR)) < tol:
            return QL + QR
        
        return recurse(lo, mid, depth + 1, tol/2) + recurse(mid, hi, depth + 1, tol/2)

    return recurse(a, b, 0, tol)


def erf_adaptive_gauss_legendre(x, n=8, tol=1e-12):
    '''
    Compute erf(x) using adaptive Gauss-Legendre quadrature.
    '''
    if x < 0:
        return -erf_adaptive_gauss_legendre(-x, n=n, tol=tol)
    
    integral = adaptive_gauss_legendre(
        lambda t: np.exp(-t*t),
        0.0, x, tol=tol, n=n
    )

    return 2/math.sqrt(math.pi) * integral


# Gauss-Kronrod GK7-15 Quadrature

# Positive Kronrod nodes (including 0)
_xk = np.array([
    0.9914553711208126,
    0.9491079123427585,
    0.8648644233597691,
    0.7415311855993945,
    0.5860872354676911,
    0.4058451513773972,
    0.2077849550078985,
    0.0
])

_wk = np.array([
    0.0229353220105292,
    0.0630920926299785,
    0.1047900103222502,
    0.1406532597155259,
    0.1690047266392679,
    0.1903505780647854,
    0.2044329400752989,
    0.2094821410847278
])

# Indices within _xk/_wk that correspond to the embedded 7-point Gauss rule.
_gauss_indices = [1, 3, 5, 7]

_wg = np.array([
    0.1294849661688697,
    0.2797053914892766,
    0.3818300505051189,
    0.4179591836734694
])


def gk15_interval(f, a, b):
    '''
    Apply the GK7-15 rule to f on [a, b].

    Returns (G7, K15): the embedded 7-point Gauss estimate and the
    15-point Kronrod estimate. The difference |K15 - G7| is an inexpensive
    error indicator.
    '''
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)

    Qk = 0.0
    Qg = 0.0
    fx = 0.0
    fx1 = 0.0
    fx2 = 0.0

    gi = 0

    for i in range(len(_xk)):
        xi = _xk[i]
        wi = _wk[i]

        if xi == 0.0:
            fx = f(mid)
            Qk += wi * fx
        else:
            fx1 = f(mid + half * xi)
            fx2 = f(mid - half * xi)
            Qk += wi * (fx1 + fx2)

        if i in _gauss_indices:
            Qg += _wg[gi] * (fx if xi == 0.0 else fx1 + fx2)
            gi += 1

    Qk *= half
    Qg *= half

    return Qg, Qk


def adaptive_gk15(f, a, b, tol=1e-12, depth=0, max_depth=10):
    '''
    Adaptive GK7-15 quadrature of f on [a, b].

    Uses |K15 - G7| as the error estimate on each sub-interval and
    recursively bisects until the estimate is below the local tolerance.
    '''
    Qg, Qk = gk15_interval(f, a, b)
    err = abs(Qk - Qg)

    if err < tol or depth >= max_depth:
        return Qk

    mid = 0.5 * (a + b)

    left = adaptive_gk15(f, a, mid, tol/2, depth+1, max_depth)
    right = adaptive_gk15(f, mid, b, tol/2, depth+1, max_depth)

    return left + right


def erf_adaptive_gk15(x, tol=1e-14, max_depth=10):
    '''
    Compute erf(x) using adaptive GK7-15 quadrature.
    '''
    if x < 0:
        return -erf_adaptive_gk15(-x, tol=tol, max_depth=max_depth)
    
    I = adaptive_gk15(
        lambda x: np.exp(-x*x),
        0, x,
        tol=tol,
        max_depth=max_depth
    )

    return (2 / np.sqrt(np.pi)) * I


# Hybrid Methods

def series_hybrid(x, b=3.5253):
    '''
    Hybrid series approximation for erf(x).

    Uses the Taylor series for |x| < b and the asymptotic erfc expansion
    for |x| >= b. The default boundary b = 2.6162 is near the point where
    the two empirical errors cross.

    Maximum absolute error is approximately 1e-7 near the boundary;
    use hybrid_v2 for near-machine-precision accuracy.
    '''
    if x < 0:
        return -series_hybrid(-x, b=b)
    
    if x < b:
        return erf_taylor(x, tol=1e-15)
    
    return 1 - erfc_asymptotic(x)



def hybrid_v2(x, boundary1=1, boundary2=5):
    '''
    Improved hybrid method for approximating erf(x).

    Small x (|x| < boundary1): Taylor series
    Medium x (boundary1 <= |x| < boundary2): Adaptive GK7-15 (tol=1e-14)
    Large x (|x| >= boundary2): Asymptotic series for erfc

    Achieves near-machine-precision accuracy across [0, 10] with roughly
    half the function evaluations of a pure adaptive quadrature approach,
    since the cheap Taylor and asymptotic branches handle the easy regions.
    '''
    if x < 0:
        return -hybrid_v2(-x, boundary1=boundary1, boundary2=boundary2)
    
    if x < boundary1:
        return erf_taylor(x)

    if x < boundary2:
        return erf_adaptive_gk15(x, tol=1e-14)
    
    return 1 - erfc_asymptotic(x)


if __name__ == "__main__":
    pass