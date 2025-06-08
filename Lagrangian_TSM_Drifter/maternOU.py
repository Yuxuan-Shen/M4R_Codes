import numpy as np
from scipy.special import beta, gamma, kv  # kv is BesselK
from scipy.optimize import minimize
from scipy.linalg import toeplitz, cholesky
import time

def complexouacvs(Q, N, Delta):
    """
    Computes the autocovariance for a complex OU process of length N,
    with spacing Delta, amplitude Q[0], frequency Q[1], and damping Q[2].
    
    Parameters:
    Q : array-like
        Parameters [amplitude, frequency, damping].
    N : int
        Length of the autocovariance sequence.
    Delta : float
        Spacing between observations. Time lag.
    
    Returns:
    acv : numpy.ndarray
        Autocovariance sequence from lag 0 to N-1.
    """
    lags = np.arange(N)  # Lags from 0 to N-1
    acv = (Q[0]**2 / abs(2 * Q[2])) * \
          np.exp(1j * Q[1] * Delta * lags) * \
          np.exp(-abs(Q[2]) * Delta * lags)
    return acv


def maternacvs(Q, N, Delta):
    """
    Computes the autocovariance for a Matern process of length N,
    with spacing Delta, amplitude Q[0], slope Q[1], and damping Q[2].

    Parameters:
    Q : array-like
        Parameters [amplitude, slope, damping].
    N : int
        Length of the autocovariance sequence.
    Delta : float
        Spacing between observations.

    Returns:
    acv : numpy.ndarray
        Autocovariance sequence from lag 0 to N-1.
    """
    acv = np.zeros(N)  # Initialize autocovariance sequence

    # Variance at lag 0
    acv[0] = (Q[0]**2 * beta(0.5, Q[1] - 0.5) / (2 * np.pi * abs(Q[2])**(2 * Q[1] - 1)+1e-8))
    
    # Autocovariance for lags 1 to N-1
    lags = np.arange(1, N)
    factor = (abs(Q[2]) * Delta * lags)**(Q[1] - 0.5)
    bessel_term = kv(Q[1] - 0.5, abs(Q[2]) * Delta * lags)
    denominator = (np.sqrt(np.pi) * 2**(Q[1] - 0.5) *
                   abs(Q[2])**(2 * Q[1] - 1) * gamma(Q[1]))
    acv[1:] = Q[0]**2 * factor * bessel_term / (denominator + 1e-8)

    return acv


def fminsearchbnd(fun, x0, LB=None, UB=None, options=None, *args):
    """
    Minimize a function with bound constraints using transformations.
    
    Parameters:
    fun : callable
        The objective function to minimize.
    x0 : array-like
        Initial guess.
    LB : array-like, optional
        Lower bounds (default: no lower bounds).
    UB : array-like, optional
        Upper bounds (default: no upper bounds).
    options : dict, optional
        Options for the optimizer.
    *args : tuple
        Additional arguments for the objective function.
    
    Returns:
    result : OptimizeResult
        Result object with optimized values, function value, and exit status.
    """
    # Ensure bounds are valid
    x0 = np.asarray(x0)
    n = len(x0)
    LB = np.full(n, -np.inf) if LB is None else np.asarray(LB)
    UB = np.full(n, np.inf) if UB is None else np.asarray(UB)

    if len(LB) != n or len(UB) != n:
        raise ValueError("x0, LB, and UB must have the same length.")
    if not np.all(LB <= UB):
        raise ValueError("Each lower bound must be less than or equal to its corresponding upper bound.")
    
    # Determine fixed variables
    fixed_vars = LB == UB

    # Helper to transform variables into unconstrained space
    def transform_to_unconstrained(x):
        xu = []
        for i in range(n):
            if fixed_vars[i]:
                continue  # Skip fixed variables
            elif np.isfinite(LB[i]) and np.isfinite(UB[i]):
                # Dual bounds
                xu.append(np.arcsin(2 * (x[i] - LB[i]) / (UB[i] - LB[i]) - 1))
            elif np.isfinite(LB[i]):
                # Lower bound only
                xu.append(np.sqrt(max(0, x[i] - LB[i])))
            elif np.isfinite(UB[i]):
                # Upper bound only
                xu.append(np.sqrt(max(0, UB[i] - x[i])))
            else:
                # Unconstrained
                xu.append(x[i])
        return np.array(xu)
    
    # Transform back to constrained space
    def transform_to_constrained(xu):
        x = np.zeros(n)
        k = 0
        for i in range(n):
            if fixed_vars[i]:
                x[i] = LB[i]  # Fixed variable
            elif np.isfinite(LB[i]) and np.isfinite(UB[i]):
                # Dual bounds
                x[i] = LB[i] + (np.sin(xu[k]) + 1) * (UB[i] - LB[i]) / 2
                x[i] = np.clip(x[i], LB[i], UB[i])  # Avoid numerical issues
                k += 1
            elif np.isfinite(LB[i]):
                # Lower bound only
                x[i] = LB[i] + xu[k]**2
                k += 1
            elif np.isfinite(UB[i]):
                # Upper bound only
                x[i] = UB[i] - xu[k]**2
                k += 1
            else:
                # Unconstrained
                x[i] = xu[k]
                k += 1
        return x
    
    # Objective function wrapper for transformed variables
    def transformed_fun(xu):
        x = transform_to_constrained(xu)
        return fun(x, *args)
    
    # Initial guess in unconstrained space
    x0u = transform_to_unconstrained(x0)
    
    # Minimize using scipy's minimize
    result = minimize(transformed_fun, x0u, method="Nelder-Mead", options=options)
    
    # Transform result back to original space
    result.x = transform_to_constrained(result.x)
    result.success = result.success  # MATLAB-style exit flag
    return result



def maternOUmodel(x, xb, SX, N, LB, UB, MF, ZEROF):
    """
    Computes the value of the Whittle likelihood for value x * xb (rescaled
    back to normal units) for periodogram SX, data length N.
    
    Parameters:
    x : array-like
        Input parameters for the model.
    xb : array-like
        Scaling factors for parameters.
    SX : array-like
        Periodogram values.
    N : int
        Length of the data.
    LB : int
        Lowest frequency index fitted.
    UB : int
        Highest frequency index fitted.
    MF : int
        Index of the zero frequency.
    ZEROF : int
        Determines if the zero-frequency is included in the likelihood.
    
    Returns:
    out : float
        Whittle likelihood value.
    """
    # Autocovariance sequence
    acv = (maternacvs(x[3:6] * xb[3:6], N, 1) + 
           complexouacvs(x[0:3] * xb[0:3], N, 1))
    
    # Blurred spectrum
    ESF2 = 2 * np.fft.fft(acv * (1 - np.arange(N) / N)) - acv[0]
    ESF3 = np.abs(np.real(np.fft.fftshift(ESF2)))
    
    # Whittle likelihood calculation
    if ZEROF == 0:
        indices = np.concatenate((np.arange(LB, MF), np.arange(MF + 1, UB + 1)))
        out = np.sum(np.log(ESF3[indices])) + np.sum(SX[indices] / ESF3[indices])
    else:
        indices = np.arange(LB, UB + 1)
        out = np.sum(np.log(ESF3[indices])) + np.sum(SX[indices] / ESF3[indices])
    
    return out


def MaternOUFit(X, CF, Delta, LPCNT, UPCNT, ZEROF):
    """
    Fits a Matern-OU model to the given data using blurred Whittle likelihood.

    Parameters:
    X : array-like
        Input data.
    CF : float
        Coriolis frequency in radians per unit cycle.
    Delta : float
        Time interval in desired units.
    LPCNT : float
        Fraction of negative rotary frequencies included (1=ALL, 0=NONE).
    UPCNT : float
        Fraction of positive rotary frequencies included (1=ALL, 0=NONE).
    ZEROF : int
        Include zero frequency in estimation (1=yes, 0=no).

    Returns:
    None
        Prints the fitted parameters and optimization details.
    """
    # Calculations
    N = len(X)
    omega = np.arange(0, 2 * np.pi * (1 - 1/N)+1e-10, 2 * np.pi / N)
    omega = np.fft.fftshift(omega)
    omega[:N//2] -= 2 * np.pi

    MF = N // 2  + 1 # as python start indexing at 0
    LB = int(round((MF) * (1 - LPCNT) + 1)) - 1  # as python start indexing at 0
    UB = int(round(MF + UPCNT * (N - MF))) - 1 # as python start indexing at 0
    SZ = (Delta / N) * np.abs(np.fft.fft(X))**2
    SZ = np.fft.fftshift(SZ)

    # Initial parameters
    xb = np.zeros(6)
    xb[1] = CF
    IObnd = int(round(MF * (1 + 0.5 * CF / np.pi)))
    
    if CF > 0:
        IOmax = np.max(SZ[IObnd-1:])
        IOloc = np.argmax(SZ[IObnd-1:]) + IObnd
    else:
        IOmax = np.max(SZ[:IObnd])
        IOloc = np.argmax(SZ[:IObnd]) + 1

    NF = int(abs(MF - IOloc) / 3)
    # xb[2] = np.sqrt(np.median(SZ[np.r_[max(IOloc - NF, 0):IOloc, 
    #                                     IOloc + 1:min(IOloc + NF + 1, N)]] *
    #                           (omega[np.r_[max(IOloc - NF, 0):IOloc, 
    #                                        IOloc + 1:min(IOloc + NF + 1, N)]] - xb[1])**2 /
    #                           (IOmax - SZ[np.r_[max(IOloc - NF, 0):IOloc, 
    #                                             IOloc + 1:min(IOloc + NF + 1, N)]])))
    xb[2] = np.sqrt(np.median(SZ[np.r_[IOloc - NF - 1:IOloc - 1, IOloc:IOloc + NF]] * 
                         (omega[np.r_[IOloc - NF - 1:IOloc - 1, IOloc:IOloc + NF]] - xb[1])**2 / 
                         (IOmax - SZ[np.r_[IOloc - NF - 1:IOloc - 1, IOloc:IOloc + NF]])))

    xb[0] = np.sqrt(IOmax) * xb[2]

    xb[4] = 1
    valMAX = np.max(SZ)
    # xb[5] = np.sqrt(np.median(SZ[np.r_[max(MF - NF, 0):MF, 
    #                                     MF + 1:min(MF + NF + 1, N)]] *
    #                           omega[np.r_[max(MF - NF, 0):MF, 
    #                                        MF + 1:min(MF + NF + 1, N)]]**2 /
    #                           (valMAX - SZ[np.r_[max(MF - NF, 0):MF, 
    #                                              MF + 1:min(MF + NF + 1, N)]])))
    xb[5] = np.sqrt(np.median(SZ[np.r_[MF - NF - 1:MF - 1, MF:MF + NF]] * 
                         (omega[np.r_[MF - NF - 1:MF - 1, MF:MF + NF]])**2 / 
                         (valMAX - SZ[np.r_[MF - NF - 1:MF - 1, MF:MF + NF]])))
    xb[3] = np.sqrt(valMAX) * xb[5]

    def objective(x):
        return maternOUmodel(x, xb, np.conj(SZ.T), N, LB, UB, MF, ZEROF)

    start_time = time.time()
    result = fminsearchbnd(objective, [1, 1, 1, 1, 1, 1], 
                           LB=np.array([0, -np.pi, 0, 0, 0.5, 0]), UB=np.array([np.inf, np.pi, np.inf, np.inf, np.inf, np.inf]), 
                           options={'maxiter': 100000, 'tol': 1e-7})
    x1 = result.x * xb
    elapsed_time = time.time() - start_time

    # Output
    # print(f"OU Amplitude = {x1[0]}")
    # print(f"OU Frequency = {x1[1]}")
    # print(f"OU Damping = {x1[2]}")
    # print(f"Matern Amplitude = {x1[3]}")
    # print(f"Matern Slope = {x1[4]}")
    # print(f"Matern Damping = {x1[5]}")
    if result.success:
        return x1[0], x1[1], x1[2], x1[3], x1[4], x1[5]
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def MaternOUData(Q, N):
    """
    Generates complex-valued stochastic data from the Matern + complex OU model
    using an O(N^3) Cholesky method.
    
    Parameters:
    Q : array-like
        Parameters of the model.
        Q[0:3] are the inertial amplitude, frequency, and damping parameters.
        Q[3:6] are the Matern amplitude, slope, and damping parameters.
    N : int
        Length of the generated data.
    
    Returns:
    X : numpy.ndarray
        Complex-valued time series data.
    """
    # Generate Matern process component
    tpz = maternacvs(Q[3:6], N, 1)
    T = cholesky(toeplitz(tpz)).T
    Z = np.random.randn(N) + 1j * np.random.randn(N)
    XX = T @ Z / np.sqrt(2)
    
    # Generate complex OU process component
    tpz2 = complexouacvs(Q[0:3], N, 1)
    T2 = cholesky(toeplitz(tpz2)).T
    Z2 = np.random.randn(N) + 1j * np.random.randn(N)
    XX2 = T2 @ Z2 / np.sqrt(2)
    
    # Combine both components
    X = XX + XX2
    return X

# Example usage:
# X = MaternOUData([10, -0.6, 0.1, 10, 1.25, 0.1], 1000)
# MaternOUFit(X, -0.5, 1, 1, 1, 1)
