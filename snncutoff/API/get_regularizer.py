from snncutoff.regularizer import *


ann_regularizer = {
'none': None,
'rcs': RCSANN(),
}
snn_regularizer = {
'none': None,
'rcs': RCSSNN(),
} 

ann_regularizer_loss = {
'none': None,
'rcs': RCSANNLoss,
}

snn_regularizer_loss = {
'none': None,
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