
from snncutoff.models.vgglike import *
from snncutoff.constrs.ann import *
from snncutoff.constrs.snn import *
from snncutoff.utils import add_ann_constraints, add_snn_layers
from snncutoff.models.VGG import *
from snncutoff.models.ResNet import *
from snncutoff.models import sew_resnet
from snncutoff.API import get_regularizer, get_constrs

def get_snn_model(args):
    num_classes  = OuputSize(args.data.lower())
    if args.method !='ann' and args.method !='snn':
        AssertionError('Training method is wrong!')
    if InputSize(args.data.lower()) == '2-128-128':
        if args.method=='ann':
            model = ann_models(args.model,num_classes)
            model = add_ann_constraints(model, args.T, args.L, 
                                        ann_constrs=get_constrs(args.ann_constrs.lower(),args.method), 
                                        regularizer=get_regularizer(args.regularizer.lower(),args.method))    
            return model
        elif args.method=='snn':
            model = ann_models(args.model,num_classes)
            model = add_snn_layers(model, args.T, 
                                    snn_layers=get_constrs(args.snn_layers.lower(),args.method), 
                                    TBN=args.TBN,
                                    regularizer=get_regularizer(args.regularizer.lower(),args.method))  
            return model
            # return snn_models(args.model,args.T, num_classes)
    elif InputSize(args.data.lower()) == '3-32-32':
        if args.method=='ann':
            model = ann_models(args.model,num_classes)
            model = add_ann_constraints(model, args.T, args.L, 
                                        ann_constrs=get_constrs(args.ann_constrs.lower(),args.method), 
                                        regularizer=get_regularizer(args.regularizer.lower(),args.method))   
            return model
        elif args.method=='snn':
            model = ann_models(args.model,num_classes)
            model = nn.Sequential(
                *list(model.children()),  
                ) 
            model = add_snn_layers(model, args.T,
                                    snn_layers=get_constrs(args.snn_layers.lower(),args.method), 
                                    TBN=args.TBN,
                                    regularizer=get_regularizer(args.regularizer.lower(),args.method))  
            return model
    elif InputSize(args.data.lower()) == '3-224-224':
        if args.method=='ann':
            model = ann_models(args.model,num_classes)
            model = add_ann_constraints(model, args.T, args.L, 
                                        ann_constrs=get_constrs(args.ann_constrs.lower(),args.method), 
                                        regularizer=get_regularizer(args.regularizer.lower(),args.method))    
            return model
        elif args.method=='snn':
            return snn_models(args.model,num_classes)
    elif InputSize(args.data.lower()) == '2-240-180':
        if args.method=='ann':
            return VGGANN_NCaltech101()
        elif args.method=='snn':
            model = ann_models(args.model,num_classes)
            model = add_snn_layers(model, args.T, 
                                    snn_layers=get_constrs(args.snn_layers.lower(),args.method), 
                                    TBN=args.TBN,
                                    regularizer=get_regularizer(args.regularizer.lower(),args.method))  
            return model
    else:
        NameError("The dataset name is not support!")
        exit(0)
        
    
    return model


def isVGG(name):
    if name.lower() in ['vgg11','vgg13','vgg16','vgg19',]:
        return True
    return False

def isResNet(name):
    if name.lower() in ['resnet18','resnet20','resnet34','resnet50','resnet101','resnet152']:
        return True
    return False

def ann_models( model_name, num_classes):
    if isVGG(model_name):
        return VGG(model_name.upper(), num_classes, dropout=0)
    elif isResNet(model_name):
        return get_resnet(model_name, num_classes=num_classes)
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
    if 'cifar10-dvs' == name.lower() or 'cifar10' == name.lower() :
        return 10
    elif  'dvs128-gesture' == name.lower():
        return 11
    elif 'cifar100' == name.lower():
        return 100
    elif 'imagenet' == name.lower():
        return 1000
    elif  'ncaltech101' == name.lower():
        return 101
    else:
        NameError('This dataset name is not supported!')
