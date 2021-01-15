import math
cases={
    '0.1_-1_1':{'wm':1,'k':0.1,'y':1.14e-4,'eta':1,'g':1,'detuning':-1,'ne':2,'na':0,'phi':0.12*math.pi},
    '0.1_0_1':{'wm':1,'k':0.1,'y':1.14e-4,'eta':1,'g':1,'detuning':0,'ne':2,'na':0,'phi':-0.88*math.pi},
    '0.1_1_1':{'wm':1,'k':0.1,'y':1.14e-4,'eta':1,'g':1,'detuning':1,'ne':2,'na':0,'phi':0.5*math.pi}, #questo è un troiaio comunque
    '1_-1_1':{'wm':1,'k':1,'y':1.14e-4,'eta':1,'g':1,'detuning':-1,'ne':2,'na':0,'phi':0*math.pi},
    '1_0_1':{'wm':1,'k':1,'y':1.14e-4,'eta':1,'g':1,'detuning':0,'ne':2,'na':0,'phi':-0.24*math.pi},
    '1_1_1':{'wm':1,'k':1,'y':1.14e-4,'eta':1,'g':1,'detuning':1,'ne':2,'na':0,'phi':0.5*math.pi},
    '10_-1_1':{'wm':10,'k':0.1,'y':1.14e-4,'eta':1,'g':1,'detuning':-1,'ne':2,'na':0,'phi':0.7*math.pi},
    '10_0_1':{'wm':10,'k':0.1,'y':1.14e-4,'eta':1,'g':1,'detuning':0,'ne':2,'na':0,'phi':-0.35*math.pi},
    '10_1_1':{'wm':10,'k':0.1,'y':1.14e-4,'eta':1,'g':1,'detuning':1,'ne':2,'na':0,'phi':-0.44*math.pi},
    '0.1_-1_0.5':{'wm':1,'k':0.1,'y':1.14e-4,'eta':1,'g':0.5,'detuning':-1,'ne':2,'na':0,'phi':-0.8*math.pi},#da qui sono g=0.5
    '0.1_0_0.5':{'wm':1,'k':0.1,'y':1.14e-4,'eta':1,'g':0.5,'detuning':0,'ne':2,'na':0,'phi':-0.68*math.pi},
    '0.1_1_0.5':{'wm':1,'k':0.1,'y':1.14e-4,'eta':1,'g':0.5,'detuning':1,'ne':2,'na':0,'phi':0.32*math.pi}, 
    '1_-1_0.5':{'wm':1,'k':1,'y':1.14e-4,'eta':1,'g':0.5,'detuning':-1,'ne':2,'na':0,'phi':0*math.pi}, #questo va bene comunque
    '1_0_0.5':{'wm':1,'k':1,'y':1.14e-4,'eta':1,'g':0.5,'detuning':0,'ne':2,'na':0,'phi':-0.24*math.pi},
    '1_1_0.5':{'wm':1,'k':1,'y':1.14e-4,'eta':1,'g':0.5,'detuning':1,'ne':2,'na':0,'phi':-0.56*math.pi},
    '10_-1_0.5':{'wm':10,'k':0.1,'y':1.14e-4,'eta':1,'g':0.5,'detuning':-1,'ne':2,'na':0,'phi':-0.44*math.pi},
    '10_0_0.5':{'wm':10,'k':0.1,'y':1.14e-4,'eta':1,'g':0.5,'detuning':0,'ne':2,'na':0,'phi':-0.48*math.pi},
    '10_1_0.5':{'wm':10,'k':0.1,'y':1.14e-4,'eta':1,'g':0.5,'detuning':1,'ne':2,'na':0,'phi':0.44*math.pi}
}
cases={
    '0.5_h_0.3':{'wm':1,'k':0.5,'y':2e-7,'eta':1,'g':0.3,'detuning':-1,'ne':3.5e5,'na':0,'phi':-0.28*math.pi},
    '0.5_r_0.3':{'wm':1,'k':0.5,'y':1.14e-4,'eta':1,'g':0.3,'detuning':-1,'ne':2,'na':0,'phi':0.72*math.pi},
    '4_h_0.3':{'wm':1,'k':4,'y':2e-7,'eta':1,'g':0.3,'detuning':-1,'ne':3.5e5,'na':0,'phi':0.6*math.pi},
    '4_r_0.3':{'wm':1,'k':4,'y':1.14e-4,'eta':1,'g':0.3,'detuning':-1,'ne':2,'na':0,'phi':0.63*math.pi},
    '16_h_0.3':{'wm':1,'k':16,'y':2e-7,'eta':1,'g':0.3,'detuning':-1,'ne':3.5e5,'na':0,'phi':-0.48*math.pi},#questo fa schifo
    '16_r_0.3':{'wm':1,'k':16,'y':1.14e-4,'eta':1,'g':0.3,'detuning':-1,'ne':2,'na':0,'phi':0.52*math.pi},
    '0.5_h_0.9':{'wm':1,'k':0.5,'y':2e-7,'eta':1,'g':0.9,'detuning':-1,'ne':3.5e5,'na':0,'phi':-0.16*math.pi},#da qui sono g=0.9
    '0.5_r_0.9':{'wm':1,'k':0.5,'y':1.14e-4,'eta':1,'g':0.9,'detuning':-1,'ne':2,'na':0,'phi':0.12*math.pi},
    '4_h_0.9':{'wm':1,'k':4,'y':2e-7,'eta':1,'g':0.9,'detuning':-1,'ne':3.5e5,'na':0,'phi':0.56*math.pi}, 
    '4_r_0.9':{'wm':1,'k':4,'y':1.14e-4,'eta':1,'g':0.9,'detuning':-1,'ne':2,'na':0,'phi':-0.16*math.pi},
    '16_h_0.9':{'wm':1,'k':16,'y':2e-7,'eta':1,'g':0.9,'detuning':-1,'ne':3.5e5,'na':0,'phi':0.52*math.pi},
    '16_r_0.9':{'wm':1,'k':16,'y':1.14e-4,'eta':1,'g':0.9,'detuning':-1,'ne':2,'na':0,'phi':0.56*math.pi}
}
def parameters(k,mirr,g):
    return cases['{}_{}_{}'.format(k,mirr,g)]

