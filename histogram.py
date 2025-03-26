 import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp



def overeni_nprandom(n):
    x=[]
    for i in range(n):
        x.append(np.random.rand())
    return x

def strednih_rozptyl(data,n):
    x=0
    vysledek=[0,0]
    #vypocet stredni hodnoty
    for i in range(n):
        x+=data[i]
    vysledek[0]=x/n
    #vypocet rozptylu
    sigma=0
    for i in range(n):
        sigma+=(data[i]-vysledek[0])**2
    sigma_fin=sigma/(n-1)
    vysledek[1]=sigma_fin
    return vysledek

def soucet_k_rand_cisel(n,k):
    data=[]
    for i in range(n):
        x=0
        for j in range(k):
            x+=np.random.rand()
        data.append(x)
    return data

def hist_k_rand_cisel(n,k,bins):
    data=soucet_k_rand_cisel(n, k)
    normalni_rozdeleni(6, 1,data,n,bins)
    plt.hist(data, bins=bins)
    plt.title('Histogram of soucet k rand cisel')
    plt.show()
    return

def normalni_rozdeleni(mu,sigma,data,n,bins):
    datmin=min(data)
    datmax=max(data)
    x = np.linspace(datmin-1, datmax+1, bins)
    sirka_binu=(datmax-datmin)/bins
    norm_rozdeleni=sp.norm.pdf(x,mu,sigma)
    norm_rozdeleni*=sirka_binu*n
    plt.plot(x,norm_rozdeleni,'r-')
    return

def hist_logist_rovnice(r,x0,n,bins):
    x=np.zeros(n)
    x[0]=x0
    for i in range(1,n):
        x[i]=r*x[i-1]*(1-x[i-1])
    plt.hist(x,bins=bins)

    mu=np.mean(x)
    sigma=np.std(x)

    normalni_rozdeleni(mu,sigma,x,n,bins)
    plt.title('Histogram of Logistic Distribution')
    plt.show()

def wigneruv_pulkruh(a,R):
    return (2 / (np.pi * R ** 2)) * np.sqrt(np.maximum(R ** 2 - a ** 2, 0))

def gen_sim_matice(n,sigma):
    A=np.random.normal(0,sigma,(n,n))
    M=1/2*(A+A.T)
    return M

def histogram_vlastnich_cisel(n,sigma,bins):
    M=gen_sim_matice(n,sigma)
    hodnoty=np.linalg.eigvals(M)
    plt.hist(hodnoty,bins=bins,density=True)

    R=2*sigma*np.sqrt(n)

    x=np.linspace(-R,R,bins)
    plt.plot(x, wigneruv_pulkruh(x, R), 'r-')

    plt.title('Histogram of Wigner UV')
    plt.show()


#konstanty
n=10000
bins=100
r=4
sigma=1
x0=np.random.rand()

#volani jednotlivich fci
x1=overeni_nprandom(n) #uloha 1
plt.hist(x1,density=True)
plt.title('Vyber z <0,1>')
plt.show()
print(strednih_rozptyl(x1,n))

x2=soucet_k_rand_cisel(n,2) #uloha2
plt.hist(x2,bins=25,density=True)
print(strednih_rozptyl(x2,n))

hist_k_rand_cisel(n,12,bins) #uloha3

hist_logist_rovnice(r,x0,n,bins) #uloha4

histogram_vlastnich_cisel(n,sigma,bins) #uloha5




