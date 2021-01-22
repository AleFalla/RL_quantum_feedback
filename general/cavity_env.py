import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random as rand
import numpy as np
from scipy.linalg import sqrtm
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import math
import scipy
from gym_feedback.envs import Utilities as Tools
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
   
def checkpos(a): #function used to get the positive real parts of the eigenvalues of the Dyn matrix (Hurwitz check)
  for i in range(0,len(a)):
      if a[i]<0:
        a[i]=0
  return a
 
class CavityEnv(gym.Env):
      
      metadata = {'render.modes': ['human']} #non so bene a che serva ma per ora lo tengo
      #provo a definire degli attributi che dovrebbero essere le cose che vanno tenute in memoria in esecuzione
      
      def __init__(self,feedback,rewfunc=Tools.purity_like_rew,u_act=False,F=np.identity(2),q=1e-4,dt=1e-3,plot=False,steadyreset=False,pow=0.5,params={'k':1,'eta':1,'X_kunit':0.499}):
              
              super(CavityEnv, self).__init__() 
              
              #calcolo le matrici e altre cose
              self.u_act=u_act #if True inputs u as action vector (on trajectory only)
              self.feedback=feedback #define what kind of feedback (Bayes or Markov)
              #check selected parameters, if null -> random choice
              k=Tools.check_param(params['k'],1,True)
              eta=Tools.check_param(params['eta'],1,True)
              X=Tools.check_param(params['X_kunit'],0.5,True)
              params={'k':k,'eta':eta,'X_kunit':X}
              self.params=params #save parameters
              self.rewfunc=rewfunc #save reward function, default purity-like. (only for Markovian feedback)
              self.A,self.D,self.B,self.E=Tools.Matrices_Calculator('Cavity',params) #matrices calculation
              A,D,B,E=self.A,self.D,self.B,self.E #save matrices
              self.sigmacss=scipy.linalg.solve_continuous_are(A.T+B@(E.T),B,D-E@(E.T),np.identity(2)) #calculates \sigma_c a steady state
              self.F=F #save F matrix
              self.P=np.array([[1,0],[0,0]]) #define P matrix (used only for the optimal agent calculation)
              self.q=q #save feedback cost q (Bayes)
              self.Q=q*np.identity(2) #define Q matrix (Bayes)
              self.pow=pow #eventual different power for the Markovian reward function
              self.dw=np.random.randn(2)*(dt**0.5) #initialize an extraction of wiener increment
              self.dt=dt #define timestep
              self.plot=plot #plot mode and steadyreset define the reset initialization
              self.steadyreset=steadyreset
              self.time=0 #set time to 0
              self.r=np.zeros(2) #initialize first moments to 0 (reset is the important one)
              self.rmean=np.copy(self.r)
              self.sc=np.identity(2) #initialize \sigma_c to identity (reset is the important one)
              self.exc=np.zeros((2,2)) #initialize excess noise to 0 (reset is the important one
              self.current=-(2**0.5)*(self.B.T)@(self.r)*dt+self.dw #initialize current (Markovian)
              self.reward=0 #initialize reward to 0
              #initialize stuff for render
              self.viewer=None  
              self.ell=None
              self.rmean=np.copy(self.r)
              
              #action is a matrix if u_act is False, else it's a vector
              if u_act==False:
                self.action_space = spaces.Box( 
                  low=-np.inf, high=np.inf,shape=(4,), dtype=np.float32)
                self.observation_space = spaces.Box( 
                  low=-np.inf, high=np.inf,shape=(2,), dtype=np.float32)
              else:
                self.action_space = spaces.Box( 
                  low=-np.inf, high=np.inf,shape=(2,), dtype=np.float32)
                self.observation_space = spaces.Box( 
                  low=-np.inf, high=np.inf,shape=(2,), dtype=np.float32)


      def step(self, action):
              #save some variables to avoid calling always self.
              u_act=self.u_act
              time=self.time
              pow=self.pow
              feedback=self.feedback
              A,D,B,E,F,Q=self.A,self.D,self.B,self.E,self.F,self.Q
              dt=self.dt
              current=self.current
              r=self.r
              sc=self.sc
              exc=self.exc
              dw=self.dw
              rmean=self.rmean

              #save tha action 
              J=np.array(action)

              # in case action is the matrix
              if u_act==False:
                J=np.reshape(J,(2,2)) #make a matrix out of the action
                
                #here we calculate the matrices and the u vector, necessary for later calculation
                if feedback=='Bayes':
                  u=-J@r
                  Dyn=A-F@J
                  L=(E-sc@B)/(2**0.5)

                if feedback=='Markov':
                  u=J@current/dt
                  Dyn=A-(2**0.5)*F@J@(B.T)
                  L=(E-sc@B)/(2**0.5)+F@J

                #here we update variables
                drmean=Dyn@rmean*dt #the unconditional first moment is calculated ignoring the wiener increment part
                rmean=rmean+drmean
                self.rmean=rmean #save the new value
                r,sc=Tools.system_step(r,sc,A,D,B,E,F,dt,dw,u) #evolution of conditional first moments and \sigma_c
                exc=Tools.exc_step(exc,Dyn,L,dt) #excess noise evolution
                if feedback=='Bayes':
                  #here we calculate three quantities necessary for the average reward calculation
                  base=np.outer(rmean.T,rmean)  
                  W=base+exc/2
                  Qcost=self.q*np.trace(J@W@(J.T))
                  rew=-W[0,0]-Qcost#-(r[0]**2+self.q*u@u.T) in case we want to use an on trajectory approach still using matrice action
                if feedback=='Markov':
                  Qcost=0
                  rew=self.rewfunc(r,sc,exc,pow) #purity-like reward function unless otherwise specified in init
                
                #Hurwitz condition check and eventual penalty (different for Markov and Bayes case)
                if np.any(np.real(np.linalg.eigvals(Dyn))>0):
                  if self.feedback=='Markov':
                    rew=rew-np.tanh(np.sum(checkpos(np.linalg.eigvals(Dyn))))
                  elif self.feedback=='Bayes':
                    rew=rew-np.sum(checkpos(np.linalg.eigvals(Dyn)))

                #check on Nan or infinite values, works most of the times, masks invalid with 1e300 and then if one value is 1e300 ends the episode
                check=np.concatenate((self.sc.flatten(),self.exc.flatten(),self.r),axis=0)
                check=np.ma.fix_invalid(check,fill_value=1e300).data

                if np.any(check==1e300):
                  self.r=np.zeros(2)
                  self.sc=np.copy(self.sigmacss)
                  self.exc=np.copy(self.excss)
                  self.Done=True

                #save the updated variable values
                self.sc=sc
                self.r=r
                self.exc=exc
                if feedback=='Bayes':
                    output=W.flatten()#np.concatenate((exc.flatten(),base.flatten()),axis=0)#r in the case of on trajectory with matrice action
                if feedback=='Markov':
                    output=exc.flatten()
                #here we normalize the observation between -1 and 1
                output=np.tanh(output*0.01)   
              

              #here we deal with the choice of u vector as action

              if u_act==True:
                if self.feedback=='Markov':
                  print('with u_act=True only bayesian case is supported')
                  exit
                u=np.array([J[0],J[1]]) #save the action in an array
                r,sc=Tools.system_step(r,sc,A,D,B,E,F,dt,dw,u) #system variables update
                rew=-(r[0]**2+self.q*u@u.T) #define the reward
                Qcost=(u.T)@Q@u

                output=r.flatten() #the observation has to be only r for construction
                #save the updated variables

                self.sc=sc 
                self.r=r
                self.exc=2*np.outer(r,r.T) #this excess noise is the on trajectory quantity to be averaged over N trajectories and then subtract outer(rmean,rmean.T) to recover the actual estimation of the excess noise
              
              #ends the episode at a given time
              if time==1e5:
                self.Done=True
              time+=1
              self.time=time
              
              #here we have the measurement, extraction of wiener increments and current
              self.dw=np.random.randn(2)*(dt**0.5)
              dw=self.dw
              self.current=-(2**0.5)*(self.B.T)@(self.r)*dt+self.dw
              
              #here we save parameters, if needed for eventual training with variable parameters, to be put into the info dictionary
              #parametri=np.array(list(self.params.items()))
              #parametri=parametri[:,1].astype(np.float)
              
              return output , rew , self.Done ,{'M(t)':J,'r':r,'current':self.current,'u':u,'su':self.sc+self.exc,'purity':(np.linalg.det(self.sc+self.exc))**(-0.5),'exc':self.exc,'sc':self.sc,'Qcost':Qcost}#,'params':parametri}
    
      

      def reset(self):
              
              dt=self.dt
              self.time=0 #reset time
              plot=self.plot #save the plot option to avoid self. every time
              
              #if we have the plot option we have fixed out of equilibrium initial conditions at each episode
              if plot==True:
                self.exc=np.zeros((2,2))
                r=np.ones(2)
                n=3
                sc=(2*n+1)*np.identity(2)
                
              #if we disable the plot option we initialize with random out of equilibrium conditions at each episode
              if plot==False:
                #random initial first moment
                r=np.random.uniform(-1,1,2)#np.random.uniform(-0.5,0.5,2)
                #check parameters, the one given as None are random at the beginning of each episode (never used, needs further adjustment)
                k=Tools.check_param(self.params['k'],1,True)
                eta=Tools.check_param(self.params['eta'],1,True)
                X=Tools.check_param(self.params['X_kunit'],0.5,True)
                params={'k':k,'eta':eta,'X_kunit':X}
                self.params=params
                #new calculation of matrices based on new parameters (again needs adjustments)
                self.A,self.D,self.B,self.E=Tools.Matrices_Calculator('Cavity',params)
                A,D,B,E=self.A,self.D,self.B,self.E
                self.sigmacss=scipy.linalg.solve_continuous_are(A.T+B@(E.T),B,D-E@(E.T),np.identity(2))
                #random number of bosons in the system
                n=np.random.uniform(0,3) #here it is possible to include the steady state conditions among the possible initial conditions with negativa lower bound to n. If negative n is extracted we have steady state conditions
                #random matrix
                sctmp=np.random.uniform(-0.1,0.1,(2,2)).diagonal()
                #symplectif form
                Sy=np.array([[0,-1],[1,0]])
                #adds the random matrix to the steady state one and checks if it is physical, namely if Robertson_Schroedinger inqueality holds
                while np.any(np.linalg.eigvals(self.sigmacss+sctmp+1j*Sy)<0):
                    sctmp=np.random.uniform(-0.1,0.1,(2,2)).diagonal()
                sc=self.sigmacss+sctmp
                #initialize the excess noise to zero
                self.exc=np.zeros((2,2))


              #initialize stuff for the render  
              self.viewer=None
              self.ell=None  

              #save the variables
              self.r=r
              self.sc=sc
              
                
              #if steady state option is valid we initialize the system always in steady state feedbackless conditions
              if self.steadyreset or n<0: #if negative n is extracted from the plot False case we have steady state conditions
                    self.r=np.array([0,0]) #initialize first moments to zero (steady state hurwitz initial conditions)
                    self.sigmacss=scipy.linalg.solve_continuous_are(A.T+B@(E.T),B,D-E@(E.T),np.identity(2)) #calculates the steady state \sigma_c
                    self.sc=np.copy(self.sigmacss) #initialize the \sigma_c as the steady state one 
                    self.excss=scipy.linalg.solve_continuous_lyapunov(A,-(E-self.sigmacss@B).dot((E-self.sigmacss@B).T)) #calculates the steady state feedbackless excess noise 
                    self.exc=np.copy(self.excss) #initialize the excess as the steady state feedbackless one
             
              #sets Done as False at the beginning of each episode
              self.Done=False
              #extraction of the first wiener increment of the episode, save r and the current 
              dw=np.random.randn(2)*(dt**0.5)
              self.dw=dw
              r=self.r
              current=-(2**0.5)*(self.B.T)@r*dt+dw
              self.current=current
              
              self.rmean=np.copy(self.r)
              
              if self.u_act==False:
            
                if self.feedback=='Bayes':
                  output=self.exc.flatten()#np.concatenate((np.array(self.exc).flatten(),np.outer(self.r,self.r.T).flatten()),axis=0)
                if self.feedback=='Markov':
                  output=self.exc.flatten()

              else:
                output=self.r #recall that u_act True is possible only with Bayesian feedback as for now

              return output

      #method necessary for rendering
      def _to_rgbarray(self):
        canvas = FigureCanvas(self.viewer)
        canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
        return image

      #render function to make gifs of trained agent
      def render(self,mode='human'):
        #we look at a conditional trajectory...
        r=self.r[0:2]
        sc=self.sc[0:2,0:2]
        exc=self.exc[0:2,0:2]
        #... but for visualization we use the \sigma_u (unconditional) for the covariance ellipse
        su=sc+exc 
        w,v=np.linalg.eig(su)
        theta=0.5*np.arctan(2*su[0,1]/(1e-10+su[0,0]**2-su[1,1]**2)) #stuff for the calculation of the covariance ellipse
        a=0.3 #trasparence
        if self.viewer is None:
            # Here we initialize the plot, 'human' is basically useless in our implementation
            if mode == 'human':
                self.viewer = plt.figure(figsize=[5, 5], dpi=72)
                self.ax=self.viewer.gca()
                self.ell=Ellipse((r[0], r[1]), width=w[0], height=w[1], angle=theta,alpha=a)
                self.ax.add_patch(self.ell)
                plt.ylim([-5,5])
                plt.xlim([-5,5])
            
            elif mode == 'rgb_array':
                self.viewer = plt.figure(figsize=[4,4], dpi=90)
                self.ax=self.viewer.gca()
                self.ell=Ellipse((r[0], r[1]), width=w[0], height=w[1], angle=theta,alpha=a)
                self.ax.add_patch(self.ell)
                plt.ylim([-5,5])
                plt.xlim([-2,2])
        #here we modify the objects initialized in the previous plot
        #ax = self.viewer.gca()
        self.ax.plot(r[0],r[1],'r.-',linewidth=1,markevery=1,alpha=0.6)
        self.ell.set_alpha(a)
        self.ell.set_center((r[0],r[1]))
        self.ell.angle=theta
        self.ell.set_height(w[1])
        self.ell.set_width(w[0])
        #ax.add_patch(Ellipse((r[0], r[1]),
        #width=w[0],#qualche funzione di su
        #height=w[1],
        #angle=theta,alpha=a))

        if mode == 'human':
            return self.viewer
        elif mode == 'rgb_array':
            return self._to_rgbarray()      
          
      
      
      
    

