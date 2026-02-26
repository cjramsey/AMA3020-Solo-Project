# The Error Function

This is a repository containing supplementary material, figures and code relating to the solo project for AMA3020 Investigations (Queen's University Belfast). 

The error function is defined as

$$
  \text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \ dt.
$$

The project investigates different techniques for approximating the error function with the aim of achieving global maximum absolute error
less than $10 \varepsilon_{\text{mach}}$, where $\varepsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$.
To achieve uniform precision across the entire real line, we decompose the domain into 3 regimes: small $x$, moderate $x$ and large $x$.


The methods tested are:
- Taylor series for erf
- Asmyptotic series for erfc
- Adaptive Gauss-Legendre Quadrature
- Adaptive Gauss-Kronrod Quadrature
- Padé Approximation


Ultimately, we arrive at a hybrid method which utilizes both series representations and adaptive GK7-15.
