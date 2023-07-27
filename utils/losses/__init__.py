from torch import nn
from utils.losses.cross_entropy_loss_with_label_smooth import CrossEntropyWithLabelSmooth
from utils.losses.triplet_loss import TripletLoss
from utils.losses.contrastive_loss import ContrastiveLoss
from utils.losses.arcface_loss import ArcFaceLoss
from utils.losses.cosface_loss import CosFaceLoss, PairwiseCosFaceLoss
from utils.losses.circle_loss import CircleLoss, PairwiseCircleLoss

def build_losses(config):
    # Build identity classification loss
    if config.LOSS.CLA_LOSS == 'crossentropy':
        criterion_cla = nn.CrossEntropyLoss()
    elif config.LOSS.CLA_LOSS == 'crossentropylabelsmooth':
        criterion_cla = CrossEntropyWithLabelSmooth()
    elif config.LOSS.CLA_LOSS == 'arcface':
        criterion_cla = ArcFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'cosface':
        criterion_cla = CosFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    elif config.LOSS.CLA_LOSS == 'circle':
        criterion_cla = CircleLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    else:
        raise KeyError("Invalid classification loss: '{}'".format(config.LOSS.CLA_LOSS))

    # Build pairwise loss
    if config.LOSS.PAIR_LOSS == 'triplet':
        criterion_pair = TripletLoss(margin=config.LOSS.PAIR_M, distance='cosine')
    elif config.LOSS.PAIR_LOSS == 'contrastive':
        criterion_pair = ContrastiveLoss(scale=config.LOSS.PAIR_S)
    elif config.LOSS.PAIR_LOSS == 'cosface':
        criterion_pair = PairwiseCosFaceLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    elif config.LOSS.PAIR_LOSS == 'circle':
        criterion_pair = PairwiseCircleLoss(scale=config.LOSS.PAIR_S, margin=config.LOSS.PAIR_M)
    else:
        raise KeyError("Invalid pairwise loss: '{}'".format(config.LOSS.PAIR_LOSS))
    
    return criterion_cla, criterion_pair
    

def compute_loss(config,
                 pids,
                 criterion_cla, 
                 criterion_pair,
                 features, 
                 logits, 
                 ):
    
    id_loss = criterion_cla(logits, pids)
    pair_loss = criterion_pair(features, pids)

    loss = config.LOSS.CLA_LOSS_WEIGHT * id_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss 
    
    return loss 