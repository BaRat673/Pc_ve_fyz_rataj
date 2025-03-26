import numpy as np
import matplotlib.pyplot as plt
import random as rd

r = 3.5
N = 100
x0 = rd.random()
rmax = 3.56994567
rmin = 3
rnum = 300

def trajectory_calcul(r, N, x0):
    trajectory = []
    trajectory.append(x0)
    x = x0
    for i in range(N):
        trajectory.append(x)
        x = r*x*(1-x)
    return trajectory


def bifurkace(rmax, rmin, rnum, N, M, x0, graf):
    bif=1
    bif_points=[]
    for r in np.linspace(rmin,rmax,rnum):
        t=trajectory_calcul(r,N,x0)[M+1:]
        unique=np.unique(np.round(t,decimals=2))
        if graf == 1: plt.plot([r]*(N-M),t, 'b,')
        bif_count=len(unique)

        if bif_count!=bif:
            bif_points.append(np.round(r,decimals=2))
            bif=bif_count
    if graf == 1:
        plt.title('Bifurkace')
        plt.ylabel('x')
        plt.xlabel('r')
        plt.show()

    bifurcation_points=[]
    for i in bif_points:
        if i not in bifurcation_points:
            bifurcation_points.append(i)
    return bifurcation_points


def lyapun_exp(N, x0, rmax, rmin, rnum):
    r_values = np.linspace(rmin, rmax, rnum)
    lambd_values = np.zeros(rnum)

    for idx, r in enumerate(r_values):
        trajectory = trajectory_calcul(r, N, x0)
        sum_log_derivative = 0

        for x in trajectory:
            derivative = abs(r * (1 - 2 * x))
            if derivative > 0:
                sum_log_derivative += np.log(derivative)

        lambd_values[idx] = sum_log_derivative / N

    plt.plot(r_values, lambd_values)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Lyapunov Exponent')
    plt.xlabel('r')
    plt.ylabel('Lyapunov Exponent')
    plt.show()

def feigenbaum(abj):
    return (abj[1]-abj[0])/(abj[2]-abj[1])

#lyapun_exp(N, x0, rmax, rmin, rnum)
print(bifurkace(rmax,rmin,rnum,2000,500,x0,0))
print(f'aproximace feigenbaumovy konstanty:{feigenbaum(bifurkace(rmax,rmin,rnum,2000,500,x0,0))}')