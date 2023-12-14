
class LIF(object):
    def __init__(self, vthr=1.0, tau=0.5):
        self.t = 0.0
        self.vmem = 0.0
        self.vthr = vthr
        self.tau = tau
        self.gamma = 1.0

    def reset(self):
        self.t = 0
        self.vmem = 0.0

    def initMem(self,x):
        self.vmem = x

    def updateMem(self,x):
        self.vmem = x*self.tau
        self.t += 1 
