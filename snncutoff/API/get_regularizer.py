from snncutoff.regularizer import *


class NoneReg(object):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def compute_reg_loss(self, x, y, features):
        return 0.0 

ann_regularizer = {
'none': NoneReg,
'rcs': RCSANN(),
}
snn_regularizer = {
'none': NoneReg,
'rcs': RCSSNN(),
} 

ann_regularizer_loss = {
'none': NoneReg,
'rcs': RCSANNLoss,
}

snn_regularizer_loss = {
'none': NoneReg,
'rcs': RCSSNNLoss,
}

def get_regularizer(name: str, method: str):
    if method == 'ann':
        return ann_regularizer[name]
    elif method == 'snn':
        return snn_regularizer[name]

def get_regularizer_loss(name: str, method: str):
    if method == 'ann':
        return ann_regularizer_loss[name]
    elif method == 'snn':
        return snn_regularizer_loss[name]

