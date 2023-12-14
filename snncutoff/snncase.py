import torch
import torch.nn as nn


class SNNCASE:
    def __init__(
        self,
        method: str,
        criterion: nn.Module,
        args: dict
    ) -> None:
        """A unified, easy-to-use API for SNN training
        """
        self.method = method
        self.criterion = criterion
        self.args = args
        
    def preprocess(self,x):
        if self.args.add_time_dim:
            x = x.unsqueeze(1)
            x = x.repeat(1,self.args.T,1,1,1)
        return x.transpose(0,1)

    def postprocess(self, x, y):
        if self.args.TET:
            T = self.args.T
            Loss_es = 0
            for t in range(T):
                Loss_es += self.criterion(x[t, ...], y)
            Loss_es = Loss_es / T # L_TET  
            if self.args.lamb != 0:
                MMDLoss = torch.nn.MSELoss()
                y = torch.zeros_like(x).fill_(self.args.means)
                Loss_mmd = MMDLoss(x, y) # L_mse
            else:
                Loss_mmd = 0
            return x.mean(0), (1 - self.args.lamb) * Loss_es + self.args.lamb * Loss_mmd # L_Total
        else:
            if self.args.multistep:
                x = x.mean(0)
            loss = self.criterion(x,y)
            return x,loss

    
