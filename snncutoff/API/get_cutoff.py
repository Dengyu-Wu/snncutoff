from snncutoff.cutoff import *


cutoff_list = {
'timestep': BaseCutoff,
'conf': ConfCutoff,
'topk': TopKCutoff,
}

def get_cutoff(name):
    return cutoff_list[name]