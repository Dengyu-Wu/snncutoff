import torch
import torch.nn as nn
from snncutoff.API import get_loss
from snncutoff.utils import  OutputHook, sethook

class SNNCASE:
    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        args: dict
    ) -> None:
        self.criterion = criterion
        name = 'tet' if args.TET else 'mean'
        self.snn_loss = get_loss(name,method=args.method)(criterion, args.means,args.lamb)
        self.args = args
        self.net = net
        
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
        mask = self.output_mask(x,y)
        cs_mean = torch.stack(output_hook,dim=2).flatten(0, 1).contiguous() 
        mask = torch.unsqueeze(mask,dim=2).flatten(0, 1).contiguous().detach()
        cs_mean = cs_mean*mask
        cs_mean = cs_mean.max(dim=0)[0]
        loss_reg = cs_mean.mean()
        self.loss_reg = loss_reg
        return self.snn_loss(x,y)[0], self.snn_loss(x,y)[1]+ self.args.alpha*loss_reg
    
    def forward(self, x, y, regularization):
        if regularization:
            return self._forward_regularization(x,y)
        else:
            return self._forward(x,y)

    def output_mask(self, x, y):
        _target = torch.unsqueeze(y,dim=0) 
        index = -int(x.shape[0]*0.3)
        right_predict_mask = x[index:].max(-1)[1].eq(_target).to(torch.float32)
        right_predict_mask = right_predict_mask.prod(0,keepdim=True)
        return right_predict_mask 

    def get_loss_reg(self):
        return self.loss_reg

    def remove_hook(self):
        output_hook = OutputHook()
        self.net  = sethook(output_hook)(self.net ,remove=True)
