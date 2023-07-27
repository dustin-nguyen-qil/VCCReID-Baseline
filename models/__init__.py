import logging
from typing import Tuple, Type, Union

from torch import nn

from models.classifier import Classifier
from models.vid_resnet import *

def build_models(config, num_ids: int = 150, train=True):
    model = C2DResNet50(config)
    
    if train:
        id_classifier = Classifier(feature_dim=config.MODEL.APP_FEATURE_DIM,
                                                    num_classes=num_ids)

        return model, id_classifier
    else:
        return model


            
        