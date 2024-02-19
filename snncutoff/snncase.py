import torch
import torch.nn as nn
from snncutoff.API import get_loss, get_regularizer_loss
from snncutoff.utils import  OutputHook, sethook

class SNNCASE:
    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        args: dict
    ) -> None:
        self.criterion = criterion
        self.snn_loss = get_loss(args.loss,method=args.method)(criterion, args.means,args.lamb)
        self.compute_reg_loss = get_regularizer_loss(args.regularizer,method=args.method)(args).compute_reg_loss
        self.args = args
        self.net = net
        self.loss_reg = 0.0
        self.rcs_n=args.rcs_n
        
    def preprocess(self,x):
        if self.args.add_time_dim:
            x = x.unsqueeze(1)
            x = x.repeat(1,self.args.T,1,1,1)
        return x.transpose(0,1)

    def _forward(self,x,y):
        x = self.preprocess(x)
        x = self.net(x)
        return self.snn_loss(x,y)

    def _forward_regularization(self, x, y): 
        x = self.preprocess(x)
        output_hook = OutputHook()
        self.net = sethook(output_hook)(self.net)
        x = self.net(x)
        cs_mean = torch.stack(output_hook,dim=2).flatten(0, 1).contiguous() 
        loss_reg = self.compute_reg_loss(x,y,cs_mean)
        self.loss_reg = loss_reg
        return self.snn_loss(x,y)[0], self.snn_loss(x,y)[1]+ self.args.alpha*loss_reg
    
    def forward(self, x, y, regularization):
        if regularization:
            return self._forward_regularization(x,y)
        else:
            return self._forward(x,y)

    def get_loss_reg(self):
        return self.loss_reg

    def remove_hook(self):
        output_hook = OutputHook()
        self.net  = sethook(output_hook)(self.net ,remove=True)
