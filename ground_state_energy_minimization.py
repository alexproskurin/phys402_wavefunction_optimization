from scipy import constants, power
from scipy.optimize import minimize
from scipy.integrate import quad
#from sympy.functions.elementary.exponential import exp
#import sympy
import numpy as np
## One dimentional problem of a particle of mass m moving in a potential:
# V(x) = -1/2*m*w^2*x^2 + (lambda)*(x/2)^4
# where, lambda = m^2*w^3/(h_bar)

# Constants
# h_bar = constants.hbar
# m = constants.electron_mass
h_bar = 1
m = 1
omega = 1
lambd = m**2*omega**3/h_bar


## Define wavefunction
# Numeric wavefunction
def psi(x,param):
    return (np.exp(-param[0]*(x+param[1])*(x+param[1])))*(np.exp(-param[2]*(x+param[3])*(x+param[3])))+(np.exp(-param[0]*(-x+param[1])*(-x+param[1])))*(np.exp(-param[2]*(-x+param[3])*(-x+param[3])))
    #return param[0]*np.exp((-param[1]*x*x)/2)
def psi_sqr(x,param):
    return psi(x,param)*psi(x,param)
def psi_inner_sqr(x,param):
    return psi_sqr(x,param)*(x*x)
def psi_inner_4(x,param):
    return psi_sqr(x,param)*(x*x)*(x*x)

# Define the numerical derivative of the function
def p_psi(x,param):
    return (-2*param[0]*(x+param[1])-2*param[2]*(x+param[3]))*np.exp(-param[0]*(x+param[1])*(x+param[1])-param[2]*(x+param[3])*(x+param[3]))+(2*param[0]*(-x+param[1])+2*param[2]*(-x+param[3]))*np.exp(-param[0]*(-x+param[1])*(-x+param[1])-param[2]*(-x+param[3])*(-x+param[3]))
    #return psi(x, param)*(-param[1]*x)
def p_psi_sqr(x,param):
    return p_psi(x,param)*p_psi(x,param)
# Define the expectation value of energy
def expectation_e(param):
    # Kinetic expectation val
    e_k = (h_bar*h_bar/(2*m))*quad(p_psi_sqr, -np.inf, np.inf, args = param)[0]
    print('Kinetic Energy: %f' % e_k)
    e_p_sqr_term = -1/2*m*omega*omega*(quad(psi_inner_sqr, -np.inf, np.inf, args = param)[0]) 
    e_p_quad_term = lambd/16*quad(psi_inner_4, -np.inf, np.inf, args = (param))[0]
    print('Integral of psi squared: %f' % (quad(psi_sqr, -np.inf, np.inf, args=param))[0])
    e_p = e_p_quad_term + e_p_sqr_term
    print('Potential Energy: %f' % e_p)
    normalization = abs(1/(quad(psi_sqr, -np.inf, np.inf, args=param))[0])
    print('Normalization: %f' % normalization)
    print('Lowest Energy: %f'% ((e_k + e_p)*normalization))
    print(param)
    print('#############################################')
    return (e_k + e_p)*normalization

# Trying to minimize
#initial_guess =[0.5,0.5,1,1]
bnds = ((0, 10),(-10,10),(0,10),(-10,10))
initial_guess =(1, 1, 1, 1)
x = minimize(expectation_e, initial_guess,bounds=bnds, method='CG')#, options={'maxiter':15000})
print(x)

# e_k = (1/2)*quad(p_psi_sqr, -np.inf, np.inf, args = initial_guess)[0]
# normalization = abs(1/(quad(psi_sqr, -np.inf, np.inf, args=initial_guess))[0])
# print('Kinetic Energy: %f' % e_k)
# e_p = 1/2*(quad(psi_inner_sqr, -np.inf, np.inf, args = initial_guess)[0]) 
# print('Integral of psi squared: %f' % (quad(psi_sqr, -np.inf, np.inf, args=initial_guess))[0])
# print('Potential Energy: %f' % e_p)
# print('Normalization: %f' % normalization)
# print('Lowest Energy: %f'% ((e_k + e_p)*normalization))
# print('#############################################')