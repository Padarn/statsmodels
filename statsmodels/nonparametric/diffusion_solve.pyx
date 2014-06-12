#cython profile=True
"""
cython -a heat_equation.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include/ -o heat_equation.so heat_equation.c
"""

cimport cython
cimport numpy as np
import numpy as np

ctypedef np.float64_t DOUBLE
ctypedef np.int_t INT

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def explicit_difference_neuman(np.ndarray[DOUBLE] rhs, np.ndarray[DOUBLE] PDF_PILOT,
                                double sigma_explicit):
    """
    explicit differencing for right hand side
    """
    cdef:
        Py_ssize_t i
        int Nx = len(rhs)
        np.ndarray[DOUBLE] tmp = rhs.copy()
        np.ndarray[DOUBLE] sol = rhs.copy()

    tmp = tmp/PDF_PILOT
    # LEFT HAND SIDE
    sol[0] = sigma_explicit*(-tmp[0]+tmp[1])
    # INTERIOR
    for i in range(1,Nx-1):
        sol[i]= sigma_explicit*(tmp[i-1]-2*tmp[i]+tmp[i+1])
    # RHS
    sol[Nx-1] = sigma_explicit*(-tmp[Nx-1]+tmp[Nx-2])

    sol = sol+rhs

    return sol

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def tridiagonal_solve(np.ndarray[DOUBLE] upper, np.ndarray[DOUBLE] diag,
                      np.ndarray[DOUBLE] lower, np.ndarray[DOUBLE] rhs):
    """
    Tribonal Solver
    """
    
    cdef:
        Py_ssize_t i
        int Nx = len(rhs)
        np.ndarray[DOUBLE] c = upper.copy()
        np.ndarray[DOUBLE] b = diag.copy()
        np.ndarray[DOUBLE] a = lower.copy()

    c[0] /= b[0]
    rhs[0] /= b[0]

    for i in range(1,Nx-1):
        c[i] /= b[i] - a[i-1] * c[i-1]
        rhs[i] = (rhs[i] - a[i-1] * rhs[i-1]) / ( b[i] - a[i-1]*c[i-1])

    rhs[Nx-1] = (rhs[Nx-1] - a[Nx-2] * rhs[Nx-2]) / ( b[Nx-1] - a[Nx-2]*c[Nx-2]) 

    for i in range(Nx-2,-1,-1):
        rhs[i] = rhs[i] - c[i] * rhs[i+1]

    return rhs

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def heat_equation_pilot(np.ndarray[DOUBLE] X_GRID,
                  np.ndarray[DOUBLE] Y_INITIAL,
                  np.ndarray[DOUBLE] PDF_PILOT,
                  double time, double dt, double implicit_ratio,
                  double D):
    """
    Solve the heat equation with pilot density:

                      df/dt = D d^2(f/pilot)/dx^2

    using a mixed implicit/explicit one
    step method. Ratio between implicit and explicit controlled 
    by 'implicit_ratio' = r. 

    df/dt = D(r d^2(f^Pn+1}/pilot)/dx^2 + (1-r) d^2(f^n/pilot)/dx^2)

                      r = 1 -> Backwards Euler
                       = 0.5 -> Crank Nicholson
                       = 0 -> Forward Euler
    No checking on time step done. Use backwards Euler if large
    time step desired.

    Uses Neuman d(f/pilot)/dx = 0 boundary conditions
    """

    cdef:
        Py_ssize_t i
        int Nx = X_GRID.shape[0]
        int Nt = np.ceil(time/dt)
        double dx = X_GRID[1]-X_GRID[0]
        double sigma = D*dt/(1.0*dx*dx)  # Check that 2 is correct
        double sigma_implicit = implicit_ratio * sigma
        double sigma_explicit = (1-implicit_ratio) * sigma
        np.ndarray[DOUBLE] upperdiag = np.ones(Nx-1, np.float64)
        np.ndarray[DOUBLE] diag =  np.ones(Nx, np.float64)
        np.ndarray[DOUBLE] lowerdiag = np.ones(Nx-1, np.float64)
        np.ndarray[DOUBLE] rhs = np.ones(Nx, np.float64)
        np.ndarray[DOUBLE] sol0 = np.ones(Nx, np.float64)
        np.ndarray[DOUBLE] sol1 = np.ones(Nx, np.float64)

    # intialize rhs
    rhs = explicit_difference_neuman(Y_INITIAL, PDF_PILOT, sigma_explicit)

    # intiailize diagonals
    upperdiag = -upperdiag*sigma_implicit/PDF_PILOT[1:]
    lowerdiag = -lowerdiag*sigma_implicit/PDF_PILOT[0:Nx-1]
    diag = 2*sigma_implicit*diag
    diag[0]=1*sigma_implicit;
    diag[Nx-1]=1*sigma_implicit;
    diag = diag/PDF_PILOT
    diag = diag + np.ones(Nx, np.float64)

    # do last two steps manually to keep non differenced results
    for i in range(Nt-2):
        rhs = tridiagonal_solve(upperdiag,diag,lowerdiag,rhs)
        rhs = explicit_difference_neuman(rhs, PDF_PILOT, sigma_explicit)

    # final steps with no differencing at end
    sol0 = tridiagonal_solve(upperdiag,diag,lowerdiag,rhs)
    rhs = explicit_difference_neuman(sol0, PDF_PILOT, sigma_explicit)
    sol1 = tridiagonal_solve(upperdiag,diag,lowerdiag,rhs)
    return sol0, sol1;