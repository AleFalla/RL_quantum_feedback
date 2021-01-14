import math

cases={
    '0.5_h_0.3':{'wm':1,'k':0.5,'y':2e-7,'eta':1,'g':0.3,'detuning':-1,'ne':3.5e5,'na':0,'phi':-0.28*math.pi},
    '0.5_r_0.3':{'wm':1,'k':0.5,'y':1.14e-4,'eta':1,'g':0.3,'detuning':-1,'ne':2,'na':0,'phi':0.72*math.pi},
    '4_h_0.3':{'wm':1,'k':4,'y':2e-7,'eta':1,'g':0.3,'detuning':-1,'ne':3.5e5,'na':0,'phi':0.6*math.pi},
    '4_r_0.3':{'wm':1,'k':4,'y':1.14e-4,'eta':1,'g':0.3,'detuning':-1,'ne':2,'na':0,'phi':0.63*math.pi},
    '16_h_0.3':{'wm':1,'k':16,'y':2e-7,'eta':1,'g':0.3,'detuning':-1,'ne':3.5e5,'na':0,'phi':-0.48*math.pi},#questo fa schifo lo stesso
    '16_r_0.3':{'wm':1,'k':16,'y':1.14e-4,'eta':1,'g':0.3,'detuning':-1,'ne':2,'na':0,'phi':0.52*math.pi},
    '0.5_h_0.9':{'wm':1,'k':0.5,'y':2e-7,'eta':1,'g':0.9,'detuning':-1,'ne':3.5e5,'na':0,'phi':-0.16*math.pi},#da qui sono g=0.9
    '0.5_r_0.9':{'wm':1,'k':0.5,'y':1.14e-4,'eta':1,'g':0.9,'detuning':-1,'ne':2,'na':0,'phi':0.12*math.pi},
    '4_h_0.9':{'wm':1,'k':4,'y':2e-7,'eta':1,'g':0.9,'detuning':-1,'ne':3.5e5,'na':0,'phi':0.56*math.pi}, #questo va bene comunque
    '4_r_0.9':{'wm':1,'k':4,'y':1.14e-4,'eta':1,'g':0.9,'detuning':-1,'ne':2,'na':0,'phi':-0.16*math.pi},
    '16_h_0.9':{'wm':1,'k':16,'y':2e-7,'eta':1,'g':0.9,'detuning':-1,'ne':3.5e5,'na':0,'phi':0.52*math.pi},
    '16_r_0.9':{'wm':1,'k':16,'y':1.14e-4,'eta':1,'g':0.9,'detuning':-1,'ne':2,'na':0,'phi':0.56*math.pi}
}
def parameters(k,mirr,g):
    return cases['{}_{}_{}'.format(k,mirr,g)]
