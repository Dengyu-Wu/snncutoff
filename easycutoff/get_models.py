
from easycutoff.models.dvs_resnet_models import resnet19
from easycutoff.models.vggsnns import *
from easycutoff.models.vgganns import *
from easycutoff.ann_constrs import *
from easycutoff.regularizer import *
from easycutoff.utils import add_ann_constraints
from easycutoff.models.VGG import *
from easycutoff.models.ResNet import *
from easycutoff.models import sew_resnet


regularizer = {
'none': None,
'roe': ROE(),
}

ann_constrs = {
'baseconstrs': BaseConstrs,
'qcfsconstrs': QCFSConstrs,
'clipconstrs': ClipConstrs,
}

def get_models(args):
    num_classes  = OuputSize(args.data.lower())
    if args.method !='ann' and args.method !='snn':
        AssertionError('Training method is wrong!')
    if InputSize(args.data.lower()) == '2-128-128':
        if args.method=='ann':
            model = ann_models(args.model,num_classes)
            model = add_ann_constraints(model, args.T, args.L, 
                                        ann_constrs=ann_constrs[args.ann_constrs.lower()], 
                                        regularizer=regularizer[args.regularizer.lower()])    
            return model
        elif args.method=='snn':
            return snn_models(args.model,args.T, num_classes)
    elif InputSize(args.data.lower()) == '3-32-32':
        if args.method=='ann':
            return ann_models(args.model,num_classes)
        elif args.method=='snn':
            return snn_models(args.model,num_classes)
    elif InputSize(args.data.lower()) == '3-224-224':
        if args.method=='ann':
            return ann_models(args.model,num_classes)
        elif args.method=='snn':
            return snn_models(args.model,num_classes)
    elif InputSize(args.data.lower()) == '2-240-180':
        if args.method=='ann':
            return VGGANN_NCaltech101()
        elif args.method=='snn':
            return VGGSNN_NCaltech101()
    else:
        NameError("The dataset name is not support!")
        exit(0)
        
    
    return model


# model = resnet19()
# model = modelpool(args.model, args.data)
# model = replace_maxpool2d_by_avgpool2d(model)

# model = sew_resnet.__dict__['sew_resnet34'](T=args.T, connect_f='ADD')


def ann_models( model_name, num_classes):
    if model_name == 'vgg16':
        return vgg16(num_classes=num_classes)
    elif model_name == 'resnet18':
        return resnet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        return resnet34(num_classes=num_classes)
    elif model_name == 'vggann':
        return VGGANN(num_classes=num_classes)
    else:
        AssertionError('The network is not suported!')
        exit(0)

def snn_models(model_name, T, num_classes):
    if model_name == 'VGGSNN':
        return VGGSNN(num_classes=num_classes)
    elif model_name == 'resnet18':
        return resnet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        return resnet34(num_classes=num_classes)
    elif 'sew_resnet' in model_name:
        model = sew_resnet.__dict__['sew_resnet34'](T=T, connect_f='ADD')
        return model
    else:
        AssertionError('SNN is not suported yet!')

def InputSize(name):
    if 'cifar10-dvs' in name.lower() or 'dvs128-gesture' in name.lower():
        return '2-128-128'
    elif 'cifar10' in name.lower() or 'cifar100' in name.lower():
        return '3-32-32'
    elif 'imagenet' in name.lower():
        return '3-224-224'
    elif  'ncaltech101' in name.lower():
        return '2-240-180'
    else:
        NameError('This dataset name is not supported!')

def OuputSize(name):
    if 'cifar10-dvs' in name.lower() or 'cifar10' in name.lower() :
        return 10
    elif  'dvs128-gesture' in name.lower():
        return 11
    elif 'cifar100' in name.lower():
        return 100
    elif 'imagenet' in name.lower():
        return 1000
    elif  'ncaltech101' in name.lower():
        return 101
    else:
        NameError('This dataset name is not supported!')
