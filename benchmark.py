'''
This code implements the same methods from error.py except
that each method uses a evaluation counter to keep track of
the number of times the exponential function is evaluated.
'''

import math
import numpy as np


def erf_taylor_coeff(k):
	return (2 / math.sqrt(math.pi)) * ((-1) ** k) / (math.factorial(k) * (2 * k + 1))


class EvalCounter:
	'''Wraps exp(-t^2) and counts every evaluation of it.'''
	def __init__(self):
		self.count = 0

	def f(self, t):
		'''Evaluate exp(-t^2), incrementing the counter by the number of points.'''
		t = np.asarray(t)
		self.count += t.size     
		return np.exp(-t * t)

	def reset(self):
		self.count = 0


def _compensated_sum(x):
	s = e = 0.0
	for xi in x:
		temp = s
		y = xi + e
		s = temp + y
		e = (temp - s) + y
	return s


def series_hybrid_counted(x, b=3.5253):
	'''Returns (value, n_exp_evals).'''
	if x < 0:
		val, n = series_hybrid_counted(-x, b)
		return -val, n

	if x < b:
		tol = 1e-15
		max_iter = 50
		s = 0.0
		for k in range(max_iter):
			term = erf_taylor_coeff(k) * x ** (2 * k + 1)
			if abs(term) < tol:
				return s, 0
			s += term
		return s, 0
	else:
		N = min(int(math.floor(x ** 2 + 0.5)), 100)
		term = 1.0
		s = 1.0
		for n in range(1, N):
			term *= (2 * n - 1) / (2 * x ** 2)
			s += (-1) ** n * term
		prefactor = math.exp(-x ** 2) / (x * math.sqrt(math.pi))
		return 1.0 - prefactor * s, 1 


def adaptive_gl_counted(x, n=8, tol=1e-14):
	'''Returns (value, n_evals).'''
	if x < 0:
		val, c = adaptive_gl_counted(-x, n, tol)
		return -val, c

	counter = EvalCounter()

	pivots, weights = np.polynomial.legendre.leggauss(n)

	def quadrature_interval(a, b):
		t = 0.5 * (b - a) * pivots + 0.5 * (a + b)
		return 0.5 * (b - a) * _compensated_sum(weights * counter.f(t))

	def recurse(a, b, depth, tol_r):
		mid = 0.5 * (b + a)
		Q_whole = quadrature_interval(a, b)
		Q_left  = quadrature_interval(a, mid)
		Q_right = quadrature_interval(mid, b)
		if depth >= 10:
			return Q_left + Q_right
		if abs(Q_whole - (Q_left + Q_right)) < tol_r:
			return Q_left + Q_right
		return (recurse(a, mid, depth + 1, tol_r / 2) +
				recurse(mid, b, depth + 1, tol_r / 2))

	integral = recurse(0.0, x, 0, tol)
	return (2 / math.sqrt(math.pi)) * integral, counter.count


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


def gk15_interval_counted(f_counter, a, b):
	mid  = 0.5 * (a + b)
	half = 0.5 * (b - a)
	Qk = Qg = 0.0
	fx = fx1 = fx2 = 0.0
	gi = 0
	for i, (xi, wi) in enumerate(zip(_xk, _wk)):
		if xi == 0.0:
			fx = f_counter.f(np.array([mid]))[0]
			Qk += wi * fx
		else:
			fx1 = f_counter.f(np.array([mid + half * xi]))[0]
			fx2 = f_counter.f(np.array([mid - half * xi]))[0]
			Qk += wi * (fx1 + fx2)
		if i in _gauss_indices:
			# Reuse already-computed values — no extra exp(-t^2) calls.
			Qg += _wg[gi] * (fx if xi == 0.0 else fx1 + fx2)
			gi += 1
	return Qg * half, Qk * half


def adaptive_gk15_counted(f_counter, a, b, tol=1e-14, depth=0, max_depth=15):
	Qg, Qk = gk15_interval_counted(f_counter, a, b)
	if abs(Qk - Qg) < tol or depth >= max_depth:
		return Qk
	mid = 0.5 * (a + b)
	return (adaptive_gk15_counted(f_counter, a, mid, tol / 2, depth + 1, max_depth) +
			adaptive_gk15_counted(f_counter, mid, b, tol / 2, depth + 1, max_depth))


def erf_gk15_counted(x, tol=1e-14):
	if x < 0:
		val, c = erf_gk15_counted(-x, tol)
		return -val, c
	counter = EvalCounter()
	integral = adaptive_gk15_counted(counter, 0.0, x, tol=tol)
	return (2 / math.sqrt(math.pi)) * integral, counter.count


def final_hybrid_counted(x, boundary1=1.0, boundary2=5.0, tol=1e-15):
	'''Returns (value, n_exp_evals).'''
	if x < 0:
		val, c = final_hybrid_counted(-x, boundary1, boundary2, tol)
		return -val, c

	if x < boundary1:
		tol_t = 1e-15
		s = 0.0
		for k in range(50):
			term = erf_taylor_coeff(k) * x ** (2 * k + 1)
			if abs(term) < tol_t:
				return s, 0
			s += term
		return s, 0

	if x < boundary2:
		return erf_gk15_counted(x, tol=tol)

	N = min(int(math.floor(x ** 2 + 0.5)), 100)
	term = 1.0
	s    = 1.0
	for n in range(1, N):
		term *= (2 * n - 1) / (2 * x ** 2)
		s    += (-1) ** n * term
	prefactor = math.exp(-x ** 2) / (x * math.sqrt(math.pi))
	return 1.0 - prefactor * s, 1


if __name__ == "__main__":
	pass
