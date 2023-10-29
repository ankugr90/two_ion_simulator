# refer to Jarvey, Henrique, Gupta J. Electrochem Society 169 (9), 093506 2022 
# for details on how to couple EDLs to bulk solution

# contact ankur.gupta at Colorado dot Edu for questions
# updated 10/29/2023

# if you use this code in your classroom, please send me an email note. It would be really appreciated

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

phi0=-3 # cathode potential 
phi0eq=-1 # equilibrium potential; if phi0 is less negative than phi0eq, oxidation will take over. if phi0=phi0eq, no reaction should occur and only EDLs form
delta=1 # ratio of stren layer to Debye length 
lamD=0.02 # ratio of Debye length to cell length 
kF=0.2 # dimenionless reaction constant; Frumkin-Butler-Volmer kinetics assumed. If kF=0, no reaction should occur and only EDLs form
alpha=0.5 # transfer coefficient 

# change paramters and observe the plot below; the example without EDLs is commonly taught in undergraduate/graduate classrooms
# parameters should be chosen such that the concentration in bulk shouldn't become negative 

def func(x, phi0, phi0eq, delta, lamD, kF, alpha):
    A = x[0]
    E = x[1]
    phiS = x[2]
    phi_neg = math.log(-A + 1) + E
    phi_pos = math.log(A + 1) + E
    c_neg = -A + 1
    c_pos = A + 1
    j = kF*math.exp(-(1-alpha)*(phiS-phi_neg))*(-c_neg*math.exp(-alpha*phi0) + math.exp((1-alpha)*phi0 - phi0eq))
    return [A + 0.5*j,
            math.sinh(0.5*(phi_neg - phiS)) + math.sinh(0.5*(phiS + phi_pos)),
            phiS - phi0 - 2*delta*math.sinh(0.5*(phi_neg - phiS))]


def funcval(phi0, phi0eq, delta, lamD, kF, alpha):

    args = (phi0, phi0eq, delta, lamD, kF, alpha)
    [A, E, phiS] = fsolve(func, [0.3, 0.2, 0], args)

    n_b = 10000
    x_b = np.linspace(-1 + delta*lamD,1-delta*lamD,n_b)
    c_b = A*x_b + 1
    phi_b = np.log(A*x_b + 1) + E

    phi_n_DL = phi_b[0] + 4*np.arctanh( math.tanh(0.25*(phiS-phi_b[0]))*np.exp(-(x_b+1-delta*lamD)/lamD) )
    c_n_DL_pos = c_b[0]*np.exp(-(phi_n_DL-phi_b[0]))
    c_n_DL_neg = c_b[0]*np.exp((phi_n_DL-phi_b[0]))

    x_p_DL = np.linspace(0,10)
    phi_p_DL = phi_b[n_b-1] + 4*np.arctanh( math.tanh(0.25*(-phiS-phi_b[n_b-1]))*np.exp((x_b-1+delta*lamD)/lamD ) )
    c_p_DL_pos = c_b[n_b-1]*np.exp(-(phi_p_DL-phi_b[n_b-1]))
    c_p_DL_neg = c_b[n_b-1]*np.exp((phi_p_DL-phi_b[n_b-1]))

    x_n_Stern = np.linspace(0,1)
    phi_n_Stern = np.linspace(phi0,phiS)

    x_p_Stern = np.linspace(0,1)
    phi_p_Stern = np.linspace(-phi0,-phiS)

    x_total = np.concatenate( (-1 + lamD*delta*x_n_Stern, x_b, 1 - lamD*delta*np.flip(x_p_Stern)), axis=None )
    phi_total = np.concatenate( (phi_n_Stern, phi_n_DL - phi_b[0] + phi_b - phi_b[n_b-1] + phi_p_DL, np.flip(phi_p_Stern) ), axis=None)
    c_pos = c_n_DL_pos - c_b[0] + c_b - c_b[n_b-1] + c_p_DL_pos 
    c_neg = c_n_DL_neg - c_b[0] + c_b - c_b[n_b-1] + c_p_DL_neg

    return [x_total, 
            phi_total,
            c_pos,
            c_neg,
            x_b,
            phi_b,
            c_b]


[x_total, phi_total, c_pos, c_neg, x_b, phi_b, c_b] =  funcval(phi0, phi0eq, delta, lamD, kF, alpha)   
c_low=min(np.min(c_pos),np.min(c_neg))
c_high=max(np.max(c_pos),np.max(c_neg))
fig, axs = plt.subplots(nrows=1, ncols=4,figsize=(25, 5))

val=12
font = {'family' : 'normal',
        'size'   : val}
plt.rc('font', **font)

axs[0].set_title('phi(x) full region', fontsize=val)
l1, = axs[0].plot(x_total, phi_total, 'k-',linewidth=1.5)
axs[1].set_title('c(x) full region',fontsize=val)
l2, = axs[1].plot(x_b,c_pos,label=c_pos,linewidth=1.5)
l3, = axs[1].plot(x_b,c_neg,label=c_neg,linewidth=1.5)
axs[1].legend(('positive', 'negative'), loc='upper center', frameon=False)
axs[2].set_title('phi(x) bulk region',fontsize=val)
l4, = axs[2].plot(x_b, phi_b,'k-',linewidth=1.5)
axs[3].set_title('c(x) bulk region (equal for both ions)',fontsize=val)
l5, = axs[3].plot(x_b,c_b,'k-',label=c_pos,linewidth=1.5)

plt.show()