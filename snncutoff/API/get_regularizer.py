from snncutoff.regularizer import *


regularizer = {
'none': None,
'roe': ROE(),
'l2min': L2Min(),
}
snn_regularizer = {
'none': None,
'roe': SNNROE(),
'rcs': SNNRCS(),
} 


def get_regularizer(name: str, method: str):
    if method == 'ann':
        return regularizer[name]
    elif method == 'snn':
        return snn_regularizer[name]