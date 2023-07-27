import logging
from typing import Tuple, Type, Union

from torch import nn

from models.classifier import Classifier
from models.vid_resnet import C2DResNet50, I3DResNet50, AP3DResNet50, NLResNet50, AP3DNLResNet50

__factory = {
    'c2dres50': C2DResNet50,
    'i3dres50': I3DResNet50,
    'ap3dres50': AP3DResNet50,
    'nlres50': NLResNet50,
    'ap3dnlres50': AP3DNLResNet50,
}

def build_models(config, num_ids: int = 150, train=True):

    if config.MODEL.NAME not in __factory.keys():
        raise KeyError("Invalid model: '{}'".format(config.MODEL.NAME))
    else:
        model = __factory[config.MODEL.NAME](config)
    
    if train:
        id_classifier = Classifier(feature_dim=config.MODEL.APP_FEATURE_DIM,
                                                    num_classes=num_ids)

        return model, id_classifier
    else:
        return model


            
        