import numpy as np
import sympy as sp
from matplotlib import pyplot as plt,cm
from numba import njit, prange
import math
#funzione di Richards per wall model

def Richard(y,utau,ni):

    k = 0.395
    Ck = 7.8
    yp = y*utau/ni
    Rich = (1/k)*np.log(1+k*yp+1e-14) + Ck*(1 - np.exp(-yp/11) - (yp/11)*np.exp(-0.33*yp))

    return Rich

@njit
def newton_raphson(y,u,utau,ni):

    #parametri modello
    k = 0.41
    Ck = 7.8
    tol_max = 1e-14
    iter_max = 500
    tol = 1
    num_iter = 0
    tol_list = []
    iter_list = []
    while (tol > tol_max) and (num_iter < iter_max):
        yp = y*utau/ni
       
        #evitare overflow dovuto ad esponente e troppo grande
        val1 = -yp/11
        val2 = -0.33*yp
        val1 = min(max(val1,-700.0),700.0)
        val2 = min(max(val2,-700.0),700.0)
        exp1 = math.exp(val1)
        exp2 = math.exp(val2) 
        f0 = -u/(utau + 1e-14) + (1/k)*np.log(np.maximum(1+k*yp,1e-14)) + Ck*(1 - exp1 - (yp/11)*exp2)
        f1 = (u/(utau + 1e-14)**2) + (yp/(utau + 1e-14))*(1/(1+k*yp+1e-14) + Ck/11 * (exp1 - (1-0.33*yp)*exp2))
        #f1i = (u/(utau+1e-14)**2) + 1/(ni+k*y*utau+1e-14) + (Ck*y/ni)*np.exp(-y*utau/ni) + (y/ni)*np.exp(-0.33*y*utau/ni)*(-1+0.33*y*utau/ni) 
        
        utau_new = utau - f0/(f1+1e-14)
        tol = np.abs(utau_new - utau)
        tol_list.append(tol)
        iter_list.append(num_iter)
        utau = utau_new
        num_iter = num_iter + 1
    
    return utau_new, tol_list, iter_list
    
k = 0.41
Ck = 7.8

u, utau, y, ni  = sp.symbols('u utau y ni')
richard = -u/(utau + 1e-14) + (1/k)*sp.log(1+k*y*utau/ni+1e-14) + Ck*(1 - sp.exp(y*utau/ni/11) - (y*utau/ni/11)*sp.exp(-0.33*y*utau/ni))

richard_dev = richard.diff(utau)

def newton_raphson_2(y,u,utau,ni):

    #parametri modello
    k = 0.41
    Ck = 7.8
    tol_max = 1e-10
    iter_max = 300
    tol = 1
    num_iter = 0
    tol_array = np.array([])
    num_array = np.array([])
    while (tol > tol_max) and (num_iter < iter_max):

        f0 = -u/(utau + 1e-14) + (1/k)*np.log(1+k*y*utau/ni+1e-14) + Ck*(1 - np.exp(-y*utau/ni/11) - (y*utau/ni/11)*np.exp(-0.33*y*utau/ni))
        f1 = u/(utau + 1.0e-14)**2 - 0.709090909090909*y*np.exp(utau*y/(11*ni))/ni - 0.709090909090909*y*np.exp(-0.33*utau*y/ni)/ni + 1.0*y/(ni*(1.00000000000001 + 0.41*utau*y/ni)) + 0.234*utau*y**2*np.exp(-0.33*utau*y/ni)/ni**2
        utau_new = utau - f0/(f1+1e-14)
        tol = np.abs(utau_new - utau)
        utau_new = utau
        tol_array = np.append(tol_array, tol)
        num_array = np.append(num_array, num_iter)
        num_iter += 1
 

    return utau_new, tol_array, num_array











