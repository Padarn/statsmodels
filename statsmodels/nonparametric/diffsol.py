import numpy
from matplotlib import pyplot
import numpy as np
numpy.set_printoptions(precision=3)


def diffsol(x_grid, yinitial,pdf,dpdf,ddpdf):
    L = x_grid[-1]-x_grid[0]
    J = len(x_grid)
    dx = x_grid[1]-x_grid[0]

    T = 0.005
    N = 500
    dt = float(T)/float(N-1)
    t_grid = numpy.array([n*dt for n in range(N)])

    D_u = 0.5

    f = (dpdf**2)/(pdf**3) - ddpdf/(pdf**2)
    f_vec = lambda U: f*U

    v = np.diag(-2*dpdf/(pdf**2))
    d = np.diag(1./pdf)
     
    sigma_u = float(D_u*dt)/float(2.*dx*dx)
    rho = float(dt)/float(4*dx)


    U = yinitial

    A1 =  numpy.diagflat([-sigma_u for i in range(J-1)], -1) +\
          numpy.diagflat([sigma_u]+[2.*sigma_u for i in range(J-2)]+[sigma_u]) +\
          numpy.diagflat([-(sigma_u) for i in range(J-1)], 1)
    A2 =  numpy.diagflat([rho for i in range(J-1)], -1) +\
          numpy.diagflat([rho]+[0 for i in range(J-2)]+[-rho]) +\
          numpy.diagflat([-(rho) for i in range(J-1)], 1)
    A3 =  numpy.eye(A1.shape[0])

    A_u = d.dot(A1)+v.dot(A2)+A3

    B1 =  numpy.diagflat([-sigma_u for i in range(J-1)], -1) +\
          numpy.diagflat([sigma_u]+[2.*sigma_u for i in range(J-2)]+[sigma_u]) +\
          numpy.diagflat([-(sigma_u) for i in range(J-1)], 1)
    B2 =  numpy.diagflat([rho for i in range(J-1)], -1) +\
          numpy.diagflat([rho]+[0 for i in range(J-2)]+[-rho]) +\
          numpy.diagflat([-(rho) for i in range(J-1)], 1)
    B3 =  numpy.eye(A1.shape[0])

    B_u = -d.dot(B1)-v.dot(B2)+B3

    U_record = []

    U_record.append(U)

    for ti in range(1,N):
        f=f*dt
        U_new = numpy.linalg.solve(1*A_u, 1*B_u.dot(U) + f_vec(U))
        
        U = U_new
        
        U_record.append(U)


    U_record = numpy.array(U_record)

    fig, ax = pyplot.subplots()
    pyplot.xlabel('x'); pyplot.ylabel('t')
    heatmap = ax.pcolor(x_grid, t_grid, U_record, vmin=0., vmax=1.2)
    colorbar = pyplot.colorbar(heatmap)
    colorbar.set_label('concentration U')
    pyplot.show()

    fig, ax = pyplot.subplots()
    
    ax.plot(x_grid,yinitial, alpha=0.5)
    ax.plot(x_grid,U, linewidth=2)
    
    pyplot.show()