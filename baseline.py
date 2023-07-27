from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.optim import lr_scheduler
from config import CONFIG
from datasets.dataset_loader import build_trainloader
from models import build_models
from utils.losses import build_losses, compute_loss
from utils.multiloss_weighting import MultiNoiseLoss
from torchmetrics import functional as FM
from models.vid_resnet import *

class Baseline(LightningModule):
    def __init__(self) -> None:
        super(Baseline, self).__init__()

        self.trainloader, self.dataset, self.train_sampler = build_trainloader()
        

        # Build model
        self.model, self.id_classifier = build_models(CONFIG, self.dataset.num_pids, train=True)
        # Build losses
        self.criterion_cla, self.criterion_pair = build_losses(CONFIG)

        self.training_step_outputs = []
        self.save_hyperparameters()
        
    def configure_optimizers(self):
        optimizer = optim.Adam(
            params=self.parameters(),
            lr=CONFIG.TRAIN.OPTIMIZER.LR,
            weight_decay=CONFIG.TRAIN.OPTIMIZER.WEIGHT_DECAY
        )
            
        scheduler = lr_scheduler.StepLR(
            optimizer=optimizer, 
            step_size=CONFIG.TRAIN.LR_SCHEDULER.STEPSIZE, 
            gamma=CONFIG.TRAIN.LR_SCHEDULER.DECAY_RATE
        )
        
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return self.trainloader
    
    def on_train_epoch_start(self) -> None:
        # set epoch number so the sampler will shuffle the data before sampling
        self.train_sampler.set_epoch(self.current_epoch)
    
    def forward(self, clip):
        features = self.model(clip)
        logits = self.id_classifier(features)
        return features, logits
    
    def training_step(self, batch, batch_idx):
        clip, pids, _, _ = batch 
        
        features, logits = self.forward(clip)

        loss = compute_loss(CONFIG, pids, self.criterion_cla, self.criterion_pair,
                            features, logits)
        
        acc = FM.accuracy(logits, pids, 'multiclass', average='macro', num_classes=self.dataset.num_pids)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.training_step_outputs.append(loss)
        return loss 
    
    def on_train_epoch_end(self):
        epoch_loss = sum(self.training_step_outputs) / len(self.training_step_outputs)
        self.log('epoch_loss', epoch_loss)
        self.training_step_outputs.clear()                                                                                                                                                                                                                                                                                                                          

class Inference(nn.Module):
    def __init__(self, config) -> None:
        super(Inference, self).__init__()
        
        self.model = build_models(config, train=False)
    
    def forward(self, clip):
        features = self.model(clip)
        return features

