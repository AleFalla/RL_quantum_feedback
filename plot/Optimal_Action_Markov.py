import gym
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import sqrtm
import scipy

class Optimal_Agent:
        
    def __init__(self,env):

        #setto la roba dall'ambiente
        self.dt=env.dt
        self.E=env.E
        self.A=env.A
        self.D=env.D
        self.B=env.B
        self.F=env.F
        self.scss=env.sigmacss
        #self.Finv=np.linalg.inv(self.F)
        self.sc=env.sc
        self.Env=env
        A,D,B,E,F,scss=self.A,self.D,self.B,self.E,self.F,self.scss
        
        def purinv(M):
            M=np.reshape(M,(4,4))
            Dyn=(A-(2**0.5)*F@M@(B.T))
            Meas=-2*(((E-scss@B)*(2**(-0.5))+F@M)@(((E-scss@B)*(2**(-0.5))+F@M).T))
            exc=scipy.linalg.solve_continuous_lyapunov(Dyn,Meas)
            su=exc[0:2,0:2]+scss[0:2,0:2]
            purinv=(np.linalg.det(su))**(0.5)
            return(purinv)
        Mguess=-np.identity(4)@(E-scss@B)*(2**(-0.5))
        value=scipy.optimize.minimize(purinv,Mguess.flatten())
        self.Mopt=np.reshape(value.x,(4,4))
        
    def predict(self):
        #sc=self.Env.sc
        
        #E=self.E
        #B=self.B
        #Finv=self.Finv        
        M=self.Mopt
        #M=-Finv.dot(E-sc.dot(B))*(2**(-0.5))
        #M=M#/self.action_scale
        #self.sc=sc
        return M.flatten()
        