from snncutoff.constrs.ann import *
from snncutoff.constrs.snn import *


ann_constrs = {
'baseconstrs': BaseConstrs,
'qcfsconstrs': QCFSConstrs,
'clipconstrs': ClipConstrs,
}

snn_layers = {
'baselayer': BaseLayer,
}


def get_constrs(name: str, method: str):
    if method == 'ann':
        return ann_constrs[name]
    elif method == 'snn':
        return snn_layers[name]