import torch
import os.path as osp
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateFinder
from pytorch_lightning.loggers import TensorBoardLogger
from baseline import Baseline
from config import CONFIG
from utils.utils import build_model_name

logger = TensorBoardLogger(save_dir=CONFIG.METADATA.LOG_PATH)

model = Baseline()

model_name = build_model_name()
print(model_name)

model_checkpoint = ModelCheckpoint(every_n_epochs=5)
early_stopping = EarlyStopping(monitor='epoch_loss', patience=20, mode='min')
# lr_finder = LearningRateFinder(max_lr=0.1, )

trainer = Trainer(
    accelerator='gpu',
    max_epochs=CONFIG.TRAIN.MAX_EPOCH,
    callbacks=[model_checkpoint, early_stopping],
    logger=logger,
    log_every_n_steps=1,
)


if CONFIG.TRAIN.RESUME is not None:
    ckpt_path=CONFIG.TRAIN.RESUME
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    trainer.fit(model=model)#, ckpt_path=CONFIG.TRAIN.RESUME)
else:
    trainer.fit(model=model)

torch.save(model.state_dict(), osp.join(CONFIG.METADATA.SAVE_PATH, model_name))


