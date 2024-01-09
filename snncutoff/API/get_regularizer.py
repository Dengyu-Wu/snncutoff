from snncutoff.regularizer import *


regularizer = {
'none': None,
'rcs': RCSANN(),
}
snn_regularizer = {
'none': None,
'rcs': RCSSNN(),
'rcsplus': RCSPLUSSNN(),
} 


def get_regularizer(name: str, method: str):
    if method == 'ann':
        return regularizer[name]
    elif method == 'snn':
        return snn_regularizer[name]