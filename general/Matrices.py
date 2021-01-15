import numpy as np
import math
import scipy


def Optomech(params):
    wm=params['wm'] #mechanical oscillator frequency=1
    k=params['k']*wm #light-env coupling
    y=params['y']*wm #mirr-env coupling
    eta=params['eta'] #measurement efficiency on light environment
    g=params['g']*wm #mirror-light coupling
    detuning=params['detuning']*wm#1,-1,0
    ne=params['ne'] #number of mirror phonons
    na=params['na'] #number of light phonons
    phi=params['phi'] #LO phase
    
    cos=np.around(math.cos(phi),15)
    sin=np.around(math.sin(phi),15)

    A=np.array([[-y/2,-wm,0,0],[wm,-y/2,g,0],[0,0,-k/2,detuning],[g,0,-detuning,-k/2]]) 
    D=np.array([[(1+2*ne)*y,0,0,0],[0,(1+2*ne)*y,0,0],[0,0,(1+2*na)*k,0],[0,0,0,(1+2*na)*k]])
    eta=(eta/(1+2*na*eta))**0.5
    b=-(((k*eta)**0.5))
    B=b*np.array([[0,0,0,0],[0,0,0,0],[0,0,cos**2,cos*sin],[0,0,sin*cos,sin**2]])
    E=(1+2*na)*B

    return A,D,B,E

def Cavity(params): #this is less detailed than the optomechanical one
    
    k=params['k'] #loss rate
    eta=params['eta'] #measurement efficiency
    X=params['X_kunit']*k #hamiltonian coupling  
    a=-(X+k/2)
    b=X-k/2
    A=np.array([[a,0],[0,b]])
    D=k*np.identity(2)
    d=-(eta*k)**0.5
    B=d*np.array([[1,0],[0,0]])
    E=B

    return A,D,B,E
    
