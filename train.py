import os
import torch
import pytorch_lightning as pl
from dataset import myData
from metrics import iou_score, dice_score, DiceLoss
from models.ULite import ULite

# Lightning module
class Segmentor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
      
    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        image, y_true = batch
        y_pred = self.model(image)
        loss = DiceLoss()(y_pred, y_true)
        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        return loss, dice, iou

    def training_step(self, batch, batch_idx):
        loss, dice, iou = self._step(batch)
        metrics = {"loss": loss, "train_dice": dice, "train_iou": iou}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dice, iou = self._step(batch)
        metrics = {"val_loss":loss, "val_dice": dice, "val_iou": iou}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, dice, iou = self._step(batch)
        metrics = {"loss":loss, "test_dice": dice, "test_iou": iou}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                         factor = 0.5, patience=10, verbose =True)
        lr_schedulers = {"scheduler": scheduler, "monitor": "val_dice"}
        return [optimizer], lr_schedulers


model = ULite().cuda()
DATA_PATH = ''

# Dataset & Data Loader
train_dataset = myData(type='train', data_path=DATA_PATH, transform=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=2, shuffle=True)

val_dataset = myData(type='test', data_path=DATA_PATH, transform=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)

# Training config
os.makedirs('/content/weights', exist_ok = True)
check_point = pl.callbacks.model_checkpoint.ModelCheckpoint("/content/weights", filename="ckpt{val_dice:0.4f}",
                                                            monitor="val_dice", mode = "max", save_top_k =1,
                                                            verbose=True, save_weights_only=True,
                                                            auto_insert_metric_name=False,)
progress_bar = pl.callbacks.TQDMProgressBar()
PARAMS = {"benchmark": True, "enable_progress_bar" : True,"logger":True,
          "callbacks" : [check_point, progress_bar],
          "log_every_n_steps" :1, "num_sanity_val_steps":0, "max_epochs":200,
          "precision":16,
          }
trainer = pl.Trainer(**PARAMS)
segmentor = Segmentor(model=model)

# Training
trainer.fit(segmentor, train_loader, val_loader)
