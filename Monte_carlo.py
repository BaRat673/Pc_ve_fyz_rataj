import numpy as np
import matplotlib.pyplot as plt

def integration_mc(fnc, a, b, n=100000):
    x_values = np.zeros(n)
    x = np.random.uniform(a,b,size=n)
    for i in range(n):
        x_values[i] = fnc(x[i])
    sum = np.sum(x_values)
    koef = (b - a) / (n)
    result = koef*sum
    return result

def function1(x):
    return np.exp(-x)*np.sin(x)

def function2(x):
    return 1 / (1 + 1/( 1 + x ** 3) ** (1 / 2)) ** (1 / 2)

print('Úloha 1')
print(integration_mc(function1, 0, np.sqrt(10)))
print(integration_mc(function2, 0, 1))
print()


def volume_of_d_dim_sphere(d,n=10000):
    hit = 0
    for i in range(n):
        coordinates=np.random.uniform(-1,1,d)
        distance=np.sum(coordinates ** 2)
        if distance < 1:
            hit += 1
    v_whole=2 ** d
    hm_ratio=hit / n
    result = v_whole * hm_ratio
    error = v_whole * np.sqrt((hm_ratio*(1-hm_ratio))/n)
    return result,error

def sphere_graph(d,n=100000):
    for i in range(d):
        volume=volume_of_d_dim_sphere(i,n)[0]
        error=volume_of_d_dim_sphere(i,n)[1]
        plt.scatter(i,volume)
        plt.errorbar(i,volume,xerr=None,yerr=error,capsize=2)
    plt.xlabel('dimension')
    plt.ylabel('volume')
    plt.title('Sphere Graph')
    plt.grid(True)
    plt.show()

print('Úloha 2')
print(f'{volume_of_d_dim_sphere(10)[0]}+-{volume_of_d_dim_sphere(10)[1]}')
print()
sphere_graph(11)


def n_dim_integral(fnc, con,n=10000,dim=1):
    x_values = []
    result = 0
    for i in range(n):
        x = np.random.uniform(-1.5,1.5,dim)
        if con(x):
            x_values.append(x)
    hit = len(x_values)
    for i in range(hit):
        result += fnc(x_values[i])
    volume = hit / n
    result = result / hit * volume
    return result

def condition1(x):
    return np.sum(x ** 2)<1

def function3(x):
    return np.cos(np.sum(x))

print('Úloha 3')
print(n_dim_integral(function3, n=100000, dim=2,con=condition1))

def function4(x):
    return 1 / 2 * np.pi * np.exp(-0.5 * (x[0] ** 2 * x[1] ** 2))

def condition2(val):
    x=val[0]
    y=val[1]
    return (x ** 2 + y ** 2 - 1) ** 3 - x ** 2 * y ** 3 <= 0

print('Úloha 4')
print(n_dim_integral(function4, n=100000, dim=2,con=condition2))

def heart(n,dim,con):
    '''
    Moje vlastni funkce kde jsem si chtel vykreslit srdicko :) <3
    '''
    x_values = []
    for i in range(n):
        x = np.random.uniform(-2, 2, dim)
        if con(x):
            x_values.append(x)
    for i in range(len(x_values)): plt.scatter(x_values[i][0],x_values[i][1],color='r')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title('MC_Heart')
    plt.show()
#heart(15000,2,condition2)