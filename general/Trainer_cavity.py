import gym
import numpy as np
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines import PPO1,PPO2
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
import tensorflow as tf
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines.common import make_vec_env
from cavity_env import CavityEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.policies import FeedForwardPolicy, register_policy

#define custom network
b=64
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,net_arch=[dict(pi=[b,b],
                                                          vf=[b,b])],feature_extraction="mlp")

e_c=0.01 #define entropy coeff
feedback='Bayes' #'Markov' or 'Bayes'
steady=True #if True resets always with steady state conditions
N=1 #number of parallel workers
LRo=2.5e-4  #learning rate                        
uact=False #if we want to use u as action (only Bayesian)
TIMESTEPS=int(50e6) #training steps
sched_LR=LinearSchedule(1,LRo,0) #lr schedule
LR=sched_LR.value 
qs=1e-3 #feedback cost (only Bayesian)
dirname='Tesi_bayestraj' #directory name
title='feed{}_steady{}_lro{}_ts{}M_N{}_ec{}_u{}0.35_1e5_hurw_excss'.format(feedback,steady,LRo,TIMESTEPS/1e6,N,e_c,uact)
#make checkpoint callback
checkpoint_callback = CheckpointCallback(save_freq=int(100000/N), save_path='/home/fallani/prova/New/Cavity_checkpoint/{}/{}_q{}'.format(dirname,title,qs))
callback = checkpoint_callback
#set parameters and start training
params={'k':1,'eta':1,'X_kunit':0.35} #if a parameter is set to None it will be sampled from a uniform distribution at every reset
args={'feedback':feedback,'q':qs,'uact':uact,'steadyreset':steady,'pow':0.5,'params':params,'plot':False}#i parametri di default son questi: rewfunc=Tools.purity_like_rew,q=1e-4,dt=1e-3,plot=False,pow=0.5
#instantiate environment
env = make_vec_env(CavityEnv,n_envs=N,env_kwargs=args) 
#instantiate model
model=PPO2(CustomPolicy,env,n_steps=128,learning_rate=LR,lam=0.95,ent_coef=e_c,verbose=1,nminibatches=4,noptepochs=4,tensorboard_log='/home/fallani/prova/New/TRAIN_Cavity/{}/{}_q{}'.format(dirname,title,qs),seed=1)
#train the model
model.learn(total_timesteps=TIMESTEPS,callback=callback,tb_log_name='{}_q{}'.format(title,qs))
#save the trained model at a given path
model.save('/home/fallani/prova/New/MODELS/{}_q{}'.format(title,qs))
        
