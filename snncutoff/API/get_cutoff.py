from snncutoff.cutoff import *


cutoff = {
'timestep': BaseCutoff,
'topk': TopKCutoff,
}



def get_cutoff(args):
    return cutoff[args.cutoff]