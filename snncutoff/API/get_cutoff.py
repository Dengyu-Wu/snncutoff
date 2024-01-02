from snncutoff.cutoff import *


cutoff_list = {
'timestep': BaseCutoff,
'topk': TopKCutoff,
}

def get_cutoff(name):
    return cutoff_list[name]