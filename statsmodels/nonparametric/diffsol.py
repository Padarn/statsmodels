import numpy
from matplotlib import pyplot
import numpy as np
from profilehooks import timecall
import scipy.linalg as linalg
numpy.set_printoptions(precision=3)


def diffsol(x_grid, yinitial,pdf,dpdf,ddpdf):
    L = x_grid[-1]-x_grid[0]
    J = len(x_grid)
    dx = x_grid[1]-x_grid[0]

    T = 10
    N = 1000
    dt = float(T)/float(N-1)
    t_grid = numpy.array([n*dt for n in range(N)])

    D_u = 0.5
    epsilon = 0

    f = dt*(2*(dpdf**2)/(pdf**3+epsilon) - ddpdf/(pdf**2+epsilon))
    v = np.diag(-2*dpdf/(pdf**2+epsilon))
    d = np.diag(1./(pdf+epsilon))
    

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
        U_new = numpy.linalg.solve(A_u, B_u.dot(U) + f*U)    
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
    
    ax.plot(x_grid,U, linewidth=2)
    ax.plot(x_grid,pdf)
    
    print np.sum(dx*U)
    print np.sum(dx*pdf)

    return ax

@timecall
def diffsol_noadvec(x_grid, yinitial,pdf,t, samp, siginv):
    
    # step 1 ------------------

    L = x_grid[-1]-x_grid[0]
    J = len(x_grid)
    dx = x_grid[1]-x_grid[0]

    T = t
    N = 100
    dt = float(T)/float(N-1)
    t_grid = numpy.array([n*dt for n in range(N)])

    D_u = 0.5
    epsilon = 0
    
    sigma_u = float(D_u*dt)/float(2.*dx*dx)
    print 'dx',dx


    alpha = 1
    sigma_b = float(D_u*dt)/float(1.*dx*dx) * (1-alpha)
    sigma_a = float(D_u*dt)/float(1.*dx*dx) * alpha

    print 'sigma',sigma_a
    U = yinitial

    A1 =  numpy.diagflat([-1 for i in range(J-1)], -1) +\
          numpy.diagflat([1]+[2.*1 for i in range(J-2)]+[1]) +\
          numpy.diagflat([-(1) for i in range(J-1)], 1)

    A3 =  numpy.eye(A1.shape[0])

    A_u = sigma_a*A1*1.0/pdf+A3

    B1 =  numpy.diagflat([-1 for i in range(J-1)], -1) +\
          numpy.diagflat([1]+[2.*1 for i in range(J-2)]+[1]) +\
          numpy.diagflat([-(1) for i in range(J-1)], 1)

    B3 =  numpy.eye(A1.shape[0])

    B_u =  -sigma_b*B1*1.0/pdf+B3

    U_record = []

    U_record.append(U)


    LUfact = linalg.lu_factor(A_u)

    for ti in range(1,N):
        #U_new = numpy.linalg.solve(A_u, B_u.dot(U)) 
        U_new = linalg.lu_solve(LUfact,B_u.dot(U))
        U = U_new
        U_record.append(U)


    U_record = numpy.array(U_record)

    U_first = U
    U_first_record = U_record

    #fig, ax = pyplot.subplots()
    #pyplot.xlabel('x'); pyplot.ylabel('t')
    #heatmap = ax.pcolor(x_grid, t_grid, U_record, vmin=0., vmax=1.2)
    #colorbar = pyplot.colorbar(heatmap)
    #colorbar.set_label('concentration U')
    #pyplot.show()

    #fig, ax = pyplot.subplots()
    
    #ax.plot(x_grid,U, linewidth=2)
    #ax.plot(x_grid,pdf)
    
    # step 2 -------------------

    lf2 = np.sum(dx*(B1.dot(U_record[-1]/dt-U_record[-2]/dt))**2)

    sign1 = siginv

    tstar =  (sign1/(2*len(samp)*np.sqrt(np.pi)*lf2))**(2.0/5.0)

    # step 3 -------------------

    L = x_grid[-1]-x_grid[0]
    J = len(x_grid)
    dx = x_grid[1]-x_grid[0]

    T = tstar
    N = 100
    dt = float(T)/float(N-1)
    t_grid = numpy.array([n*dt for n in range(N)])

    D_u = 0.5
    epsilon = 0
    
    sigma_b = float(D_u*dt)/float(1.*dx*dx) * (1-alpha)
    sigma_a = float(D_u*dt)/float(1.*dx*dx) * alpha

    U = yinitial

    A1 =  numpy.diagflat([-1 for i in range(J-1)], -1) +\
          numpy.diagflat([1]+[2.*1 for i in range(J-2)]+[1]) +\
          numpy.diagflat([-(1) for i in range(J-1)], 1)

    A3 =  numpy.eye(A1.shape[0])

    A_u = sigma_a*A1*1.0/pdf+A3

    B1 =  numpy.diagflat([-1 for i in range(J-1)], -1) +\
          numpy.diagflat([1]+[2.*1 for i in range(J-2)]+[1]) +\
          numpy.diagflat([-(1) for i in range(J-1)], 1)

    B3 =  numpy.eye(A1.shape[0])

    B_u =  -sigma_b*B1*1.0/pdf+B3

    U_record = []

    U_record.append(U)

    LUfact = linalg.lu_factor(A_u)

    for ti in range(1,N):
        #U_new = numpy.linalg.solve(A_u, B_u.dot(U))    
        U_new = linalg.lu_solve(LUfact,B_u.dot(U))
        U = U_new
        U_record.append(U)


    fig, ax = pyplot.subplots()
    
    ax.plot(x_grid,U, linewidth =2, label='diffusion kde')

    print 't',t,'tstar',tstar

    return ax