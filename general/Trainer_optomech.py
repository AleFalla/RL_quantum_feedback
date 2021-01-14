import gym
import numpy as np
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines import PPO2, TRPO,TD3
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
import tensorflow as tf
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from optomech_env import OptomechEnv
from stable_baselines.common.schedules import LinearSchedule
import math
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
import parameters_optomech as par

#here we define a custom network architecture
b=64
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,net_arch=[dict(pi=[b,b],
                                                          vf=[b,b])],feature_extraction="mlp")#,dict(pi=[20], vf=[20])

#here we choose the parameter configuration (see parmeters_optomech.py for the possible configurations)
g,k,mirr=0.9,4,'h'
#here we set if we want to use partial purity or not, and F bound (not full-rank) or not (identity)
partial=False
fbound=True

#choose Bayesian or Markovian config
config=1

if config==0:
    feedback='Bayes'#'Markov' or 'Bayes'
    qs=1e-2
    purity=False
    dirname='Bayesian_cases'

if config==1:
    feedback='Markov'#'Markov' or 'Bayes'
    qs=0
    purity=True
    dirname='Markovian_cases'


e_c=0.002 #here we set the entropy coefficient
steady=True #if True resets always with steady state conditions
plot=False #if True resets always to fixed out of equilibrium conditions
N=1 #number of parallel workers
LRo=2e-4 #define the learning rate     
TIMESTEPS=int(6e6) #training steps
sched_LR=LinearSchedule(1,LRo,0) #schedule for lr reduction
LR=sched_LR.value
clip=LinearSchedule(1,0.2,0).value #schedule for clipping parameter PPO (eventual)


title='feed{}_steady{}_lro{}_ts{}M_N{}_ec{}_{}_{}_{}_partial{}_fbound{}_tanh0.01_pur0.5_hurwseedr0_1e5'.format(feedback,steady,LRo,TIMESTEPS/1e6,N,e_c,k,mirr,g,partial,fbound)
#make checkpoint callback
checkpoint_callback = CheckpointCallback(save_freq=int(1000000/N), save_path='/home/fallani/prova/New/Optomech_checkpoint/{}/{}_q{}'.format(dirname,title,qs))
callback = checkpoint_callback
#set F matrix
zero=np.zeros((2,2))
if fbound==True:
    F=np.block([[zero,zero],[zero,np.identity(2)]]) #custom F matrix
elif fbound==False:
    F=np.identity(4)
P=np.block([[np.identity(2),zero],[zero,zero]])
#set parameters and start training
params=par.parameters(k=k,mirr=mirr,g=g)
#commented parameters from Hammerer, usually useless
#params={'wm':1,'k':0.5,'y':2e-7,'eta':1,'g':0.3,'detuning':-1,'ne':3.5e5,'na':0,'phi':math.pi/2}#{'wm':1,'k':5,'y':1.14e-4,'eta':1,'g':0.095,'detuning':0,'ne':2,'na':0,'phi':math.pi*0.25} #if a parameter is set to None it will be sampled from a uniform distribution at every reset
args={'feedback':feedback,'P':P,'partialpur':partial,'F':F,'q':qs,'steadyreset':steady,'params':params,'plot':plot}#i parametri di default son questi: rewfunc=Tools.purity_like_rew,F=np.identity(4),q=1e-4,dt=1e-3,plot=False,steadyreset=False,pow=0.5
env = make_vec_env(OptomechEnv,n_envs=N,env_kwargs=args) 
#initialize model, if clip=0.2-->clip=clip uses schedule for clipping. For deterministic training set seed
model=PPO2(CustomPolicy,env,n_steps=128,lam=0.95,learning_rate=LR,ent_coef=e_c,cliprange=0.2,cliprange_vf=None ,noptepochs=4,verbose=1,nminibatches=4,tensorboard_log='/home/fallani/prova/New/TRAIN_Optomech/{}/{}_q{}'.format(dirname,title,qs))#,seed=0)
#model training
model.learn(total_timesteps=TIMESTEPS,callback=callback,tb_log_name='{}_q{}'.format(title,qs))
#save the model at a given path
model.save('/home/fallani/prova/New/MODELS/{}_q{}'.format(title,qs))
