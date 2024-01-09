from snncutoff.loss import *


loss = {
'none': None,
'mean': MeanLoss,
'tet': TETLoss,
}



def get_loss(name: str, method: str,):
    if method == 'ann':
        return loss['mean']
    elif method == 'snn':
        return loss[name]