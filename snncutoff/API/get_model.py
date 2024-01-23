
from snncutoff.models.vgglike import *
from snncutoff.models.vggsnn import *
from snncutoff.constrs.ann import *
from snncutoff.constrs.snn import *
from snncutoff.utils import add_ann_constraints, add_snn_layers
from snncutoff.models.VGG import VGG
from snncutoff.models.ResNet import get_resnet
from snncutoff.models import sew_resnet
from .get_constrs import get_constrs
from .get_regularizer import get_regularizer

def get_model(args):
    input_size  = InputSize(args.data.lower())
    num_classes  = OuputSize(args.data.lower())
    if args.method !='ann' and args.method !='snn':
        AssertionError('Training method is wrong!')

    if args.method=='ann':
        multistep = args.multistep_ann
        model = ann_models(args.model, input_size, num_classes,multistep)
        model = add_ann_constraints(model, args.T, args.L, args.multistep_ann,
                                    ann_constrs=get_constrs(args.ann_constrs.lower(),args.method), 
                                    regularizer=get_regularizer(args.regularizer.lower(),args.method))    
        return model
    elif args.method=='snn':
        model = ann_models(args.model,input_size,num_classes,multistep=True) if args.arch_conversion else snn_models(args.model,args.T,input_size, num_classes) 
        model = add_snn_layers(model, args.T,
                                snn_layers=get_constrs(args.snn_layers.lower(),args.method), 
                                TEBN=args.TEBN,
                                regularizer=get_regularizer(args.regularizer.lower(),args.method),
                                arch_conversion=args.arch_conversion,
                                )  
        return model
    else:
        NameError("The dataset name is not support!")
        exit(0)

def get_basemodel(name):
    if name.lower() in ['vgg11','vgg13','vgg16','vgg19',]:
        return 'vgg'
    elif name.lower() in ['resnet18','resnet20','resnet34','resnet50','resnet101','resnet152']:
        return 'resnet'
    elif name.lower() in ['sew_resnet18','sew_resnet20','sew_resnet34','sew_resnet50','sew_resnet101','sew_resnet152']:
        return 'sew_resnet'
    else:
        pass

def ann_models( model_name, input_size, num_classes,multistep):
    base_model = get_basemodel(model_name)
    if base_model == 'vgg':
        return VGG(model_name.upper(), num_classes, dropout=0)
    elif base_model == 'resnet':
        return get_resnet(model_name, input_size=input_size, num_classes=num_classes,multistep=multistep)
    elif model_name == 'vggann':
        return VGGANN(num_classes=num_classes)
    elif model_name == 'vgg-gesture':
        return VGG_Gesture()
    elif model_name == 'vgg-ncaltech101':
        return VGGANN_NCaltech101()
    else:
        AssertionError('The network is not suported!')
        exit(0)

def snn_models(model_name, T, num_classes):
    base_model = get_basemodel(model_name)
    if base_model == 'VGGSNN':
        return VGGSNN(num_classes=num_classes)
    elif base_model=='sew_resnet':
        model = sew_resnet.__dict__[model_name](T=T, connect_f='ADD',num_classes=num_classes)
        return model
    else:
        AssertionError('This architecture is not suported yet!')

def InputSize(name):
    if 'cifar10-dvs' in name.lower() or 'dvs128-gesture' in name.lower():
        return 128 #'2-128-128'
    elif 'cifar10' in name.lower() or 'cifar100' in name.lower():
        return 32 #'3-32-32'
    elif 'imagenet' in name.lower():
        return 224 #'3-224-224'
    elif  'ncaltech101' in name.lower():
        return 240 #'2-240-180'
    else:
        NameError('This dataset name is not supported!')

def OuputSize(name):
    if 'cifar10-dvs' == name.lower() or 'cifar10' == name.lower() :
        return 10
    elif 'dvs128-gesture' == name.lower():
        return 11
    elif 'cifar100' == name.lower():
        return 100
    elif 'ncaltech101' == name.lower():
        return 101
    elif 'imagenet-' in name.lower():
        output_size = name.lower().split("-")[-1]
        return int(output_size)
    elif 'imagenet' == name.lower():
        return 1000
    else:
        NameError('This dataset name is not supported!')
