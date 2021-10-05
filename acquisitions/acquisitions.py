from . import util
import numpy as np

class AcquisitionEI(object):
    def __init__(self, jitter=0.01):
        self.jitter = jitter
    def acq(self, m, s):
        fmin = np.min(m)
        phi, Phi, u = util.get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        return f_acqu

class AcquisitionPI(object):
    def __init__(self, jitter=0.01):
        self.jitter = jitter
    def acq(self, m, s):
        fmin = np.min(m)
        phi, Phi, u = util.get_quantiles(self.jitter, fmin, m, s)
        f_acqu = u
        return f_acqu
    
class AcquisitionGPUCB(object):
    
    def __init__(self, beta=100.):
        self.beta = beta
    def acq(self, m, s):
        return -m + s * np.sqrt(self.beta)
        
class AcquisitionLCB(object):
    def __init__(self, exploration_weight=2):
        self.exploration_weight = exploration_weight
    def acq(self, x):
        f_acqu = -m + self.exploration_weight * s
        return f_acqu