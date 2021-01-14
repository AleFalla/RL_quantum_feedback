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
#np.seterr(all='raise')
    
def checkpos(a):
  for i in range(0,len(a)):
      if a[i]<0:
        a[i]=0
  return a



  
class OptomechEnv(gym.Env):
      
      metadata = {'render.modes': ['human']} 

      def __init__(self,feedback,P,partialpur=False,rewfunc=Tools.purity_like_rew,F=np.identity(4),q=1e-4,dt=1e-3,plot=False,steadyreset=False,pow=0.5,params={'wm':1,'k':0.5,'y':2e-7,'eta':1,'g':2*0.15,'detuning':1,'ne':3.5e5,'na':0,'phi':math.pi/2}):
              
              super(OptomechEnv, self).__init__()

              self.feedback=feedback #save the kind of feedback we are applying
              self.partialpur=partialpur #if we want to use the partial purity set True (only Markov)
              
              #define stuff for rendering
              self.viewer=None
              self.ell=None
              self.ax=None
              #check None parameters
              wm=Tools.check_param(self.params['wm'],1,True)
              k=Tools.check_param(self.params['k'],3,True)
              g=Tools.check_param(self.params['g'],1,True)
              y=Tools.check_param(self.params['y'],1,True)
              eta=Tools.check_param(self.params['eta'],1,True)
              detuning=Tools.check_param(self.params['detuning'],1,True)
              ne=Tools.check_param(self.params['ne'],0.5,True)
              na=Tools.check_param(self.params['na'],0.5,True)
              phi=Tools.check_param(self.params['phi'],0.5,True)
              params={'wm':wm,'k':k,'y':y,'eta':eta,'g':g,'detuning':detuning,'ne':ne,'na':na,'phi':phi}
              self.params=params #save parameters
              self.rewfunc=rewfunc #set reward function
              #calculate and save matrices
              self.A,self.D,self.B,self.E=Tools.Matrices_Calculator('Optomech',params) 
              A,D,B,E=self.A,self.D,self.B,self.E 
              self.P=P #only for the optimal agent
              self.sigmacss=scipy.linalg.solve_continuous_are(A.T+B@(E.T),B,D-E@(E.T),np.identity(4)) #steady state \sigma_c
              sigmacss=self.sigmacss
              
              self.F=F #save F matrix
              self.q=q #save feedback cost (Bayes)
              self.Q=q*np.identity(4) #define Q matrix
              self.pow=pow #save power, usually default 0.5
              self.dw=np.random.randn(4)*(dt**0.5) #sample wiener increments
              self.dt=dt #save time step interval
              self.plot=plot #save plot option for reset
              self.steadyreset=steadyreset #save steadyreset option for reset
              self.time=0 #initialize time to 0
              self.r=np.zeros(4) #initialize conditional first moments
              self.sc=np.identity(4) #initialize \sigma_c
              self.exc=np.zeros((4,4)) #initialize excess noise
              self.current=-(2**0.5)*(self.B.T)@(self.r)*dt+self.dw #initialize current
              self.rmean=np.copy(self.r) #initialize unconditional first moments
            
              #define action space (4x4 matrix)
              self.action_space = spaces.Box( 
                low=-np.inf, high=np.inf,shape=(16,), dtype=np.float32)
              
              #define different action spaces for Markovian and Bayesian feedback and for partial purity or full purity (only Markov)
              if self.feedback=='Markov':
                  self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
              if self.feedback=='Bayes':
                  self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)
                
              
              
              
      
      def step(self, action):
        #save a number of quantities to avoid writing self.
        rmean=self.rmean
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
        P=self.P
        #turn the action into a numpy array
        J=np.array(action)
        J=np.reshape(J,(4,4))
        
        #calculates u vector and the matrices for update in both Byesian and Markovian case
        if feedback=='Bayes':
          u=-J@r
          Dyn=A-F@J
          L=(E-sc@B)/(2**0.5)
        
        if feedback=='Markov':
          u=J@current/dt
          Dyn=A-(2**0.5)*F@J@(B.T)
          L=(E-sc@B)/(2**0.5)+F@J
        
        #here we update variables
        drmean=Dyn@rmean*dt
        rmean=rmean+drmean
        
        r,sc=Tools.system_step(r,sc,A,D,B,E,F,dt,dw,u)
        exc=Tools.exc_step(exc,Dyn,L,dt)
        
        #reward assignment
        if feedback=='Markov':

          if self.partialpur==True:
            exc2=np.array([[exc[0,0],exc[0,1]],[exc[1,0],exc[1,1]]])
            sc2=np.array([[sc[0,0],sc[0,1]],[sc[1,0],sc[1,1]]])
          else:
            exc2=exc#+(self.q**0.25)*(np.outer(u,u.T)).diagonal()
            sc2=sc
          su2=exc2+sc2
          rew=self.rewfunc(r,sc2,exc2,pow)

        if feedback=='Bayes':

          base=np.outer(rmean.T,rmean)
          W=exc/2+base
          Qcost=self.q*np.trace(J@W@(J.T))
          b=W[0,0]+W[1,1]+Qcost
          rew=-b
        
        #hurwitz condition ckeck, sometimes produces local minima... in that case comment this section
        hurw=np.linalg.norm(checkpos(np.real(np.linalg.eigvals(Dyn)[0:2])))     
        if np.any(np.real(np.linalg.eigvals(Dyn))>0):
          if self.feedback=='Markov':
            rew=np.clip(rew-hurw,0,1)
          elif self.feedback=='Bayes':
            rew=rew-np.sum(checkpos(np.linalg.eigvals(Dyn)))
        
        #check time and stop at the end of each episode
        if time==1e5:
            self.Done=True
        time+=1
        self.time=time #save time variable
        
        #check invalid with masked arrays (avid Nans in training)
        check=np.concatenate((sc.flatten(),exc.flatten(),J.flatten(),r,np.array([rew])),axis=0)
        check=np.array(np.ma.fix_invalid(check,fill_value=1e300).data)

        if np.any(check==1e300):
          self.r=np.zeros(4)
          self.sc=np.copy(self.sigmacss)
          self.exc=np.zeros((4,4))
          rew=0 # -1e6, suggestion is to use 0 only if the reward is positive definite (namely Markov case)
          self.Done=True


        #save variables after update
        self.sc=sc
        self.r=r
        self.rmean=rmean
        self.exc=exc
        
        #here we have the measurement
        self.dw=np.random.randn(4)*(dt**0.5)
        dw=self.dw
        self.current=-(2**0.5)*(self.B.T)@(self.r)*dt+self.dw
        
        #here we save parameters for eventual other applications (useless in the standard one)
        parametri=self.params

        #for Markov case use excess as output, for Bayes use exc and W
        output=exc.flatten()
        if feedback=='Bayes':
          output=np.concatenate((exc.flatten(),W.flatten()),axis=0)
        
        #output normalization
        output=np.tanh(output*0.01)

        #calculation of \sigma_u for purity
        su=exc+sc
        return output , rew , self.Done ,{'M(t)':J,'r':r,'current':self.current,'u':u,'su':sc+exc,'purity':1/((np.linalg.det(su[0:2,0:2]))**0.5),'exc':exc,'sc':sc,'Qcost':self.q*u@u.T,'params':parametri}
    
      

      def reset(self):
              
        #reset time
        self.time=0
        #save the plot option (to avoid self. every time)
        plot=self.plot
        dt=self.dt
        
        #if plot True we have the same fixed out of equilibrium initial conditions
        if plot==True:
          a=1.
          b=1.
          c=1.
          d=1.
          n=3
        
        #if plot False random initial conditions at every time
        if plot==False:
          a=np.random.uniform(-1,1)
          b=np.random.uniform(-1,1)
          c=np.random.uniform(-1,1)
          d=np.random.uniform(-1,1)
          n=np.random.uniform(-0.1,1) #if negative n we have steady state conditions which would otherwise be excluded
          
          #parameters check
          wm=Tools.check_param(self.params['wm'],1,True)
          k=Tools.check_param(self.params['k'],3,True)
          g=Tools.check_param(self.params['g'],1,True)
          y=Tools.check_param(self.params['y'],1,True)
          eta=Tools.check_param(self.params['eta'],1,True)
          detuning=Tools.check_param(self.params['detuning'],1,True)
          ne=Tools.check_param(self.params['ne'],0.5,True)
          na=Tools.check_param(self.params['na'],0.5,True)
          phi=Tools.check_param(self.params['phi'],0.5,True)
          params={'wm':wm,'k':k,'y':y,'eta':eta,'g':g,'detuning':detuning,'ne':ne,'na':na,'phi':phi}
          self.params=params #parameters save
          self.A,self.D,self.B,self.E=Tools.Matrices_Calculator('Optomech',params) #matrices calculation
          A,D,B,E=self.A,self.D,self.B,self.E
          
          
          
        #save initial quantities
        self.r=np.array([a,b,c,d])
        self.sc=(2*n+1)*np.identity(4)
        self.exc=0*np.identity(4)

        #reset stuff for rendering
        self.viewer=None
        self.ell=None
        self.ax=None
        
      
        #if we want feedbackless steady state initial conditions we have to set steadyreset=True or have negative n
        if self.steadyreset or n<0:
              self.r=np.zeros(4)
              self.sigmacss=scipy.linalg.solve_continuous_are(A.T+B@(E.T),B,D-E@(E.T),np.identity(4))
              self.sc=np.copy(self.sigmacss)
              self.excss=scipy.linalg.solve_continuous_lyapunov(A,-(E-self.sigmacss@B).dot((E-self.sigmacss@B).T))
              self.exc=np.copy(self.excss)
        #set Done to initial episode condition False
        self.Done=False
        #sample Wiener increments and calculate current
        dw=np.random.randn(4)*(dt**0.5)
        self.dw=dw
        r=self.r
        current=-(2**0.5)*(self.B.T)@r*dt+dw
        self.current=current
        

        #check invalid values
        check=np.concatenate((self.sc.flatten(),self.exc.flatten(),self.r),axis=0)
        check=np.ma.fix_invalid(check,fill_value=1e300).data

        if np.any(check==1e300):
          self.r=np.zeros(4)
          self.sc=np.copy(self.sigmacss)
          self.exc=np.zeros((4,4))
  
        #different outputs for Bayes and Markov
        if self.feedback=='Bayes':
          output=np.concatenate((self.exc.flatten(),np.zeros((4,4)).flatten()),axis=0)
        if self.feedback=='Markov':
          output=self.exc.flatten()


        return output
          
      
      #function for animation render
      def _to_rgbarray(self):
        canvas = FigureCanvas(self.viewer)
        canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
        return image

      #render method of a conditional trajectory with unconditional covariance
      def render(self,mode='human'):
        r=self.r[0:2]
        sc=self.sc[0:2,0:2]
        exc=self.exc[0:2,0:2]
        su=sc+exc
        w,v=np.linalg.eig(su)
        theta=0.5*np.arctan(2*su[0,1]/(1e-10+su[0,0]**2-su[1,1]**2))
        
        a=0.3
        if self.viewer is None:
            # Here we initialize the plot
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
                plt.xlim([-5,5])
        
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

    

