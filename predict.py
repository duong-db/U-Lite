import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from dataset import myData
from metrics import iou_score, dice_score
from models.ULite import ULite

model = ULite()
DATA_PATH = ''
CHECKPOINT_PATH = ''

# Lightning module
class Segmentor(pl.LightningModule):
    def __init__(self, model=model):
        super().__init__()
        self.model = model
      
    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        image, y_true = batch
        y_pred = self.model(image)
        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        metrics = {"Test Dice": dice, "Test Iou": iou}
        self.log_dict(metrics, prog_bar=True)
        return metrics

# For visualization
def visualize_prediction(model, dataset, nums):
    plt.figure(figsize=(6, 2*nums), layout='compressed')
    for idx in range(nums):
        x, y = dataset[idx]
        y_pred = model(x.unsqueeze(dim=0)).data.squeeze()
        y_pred[y_pred>0.5] = 1
        y_pred[y_pred<=0.5] = 0

        # convert torch to numpy
        x = x.permute(1, 2, 0).numpy()
        y = y.squeeze().numpy()
        y_pred = y_pred.numpy()
        
        # visualization
        plt.subplot(nums, 3, 3*idx + 1)
        plt.title(f"Image {idx+1}")
        plt.imshow(x, cmap='gray')
        plt.axis('off')

        plt.subplot(nums, 3, 3*idx + 2)
        plt.title(f"Ground truth {idx+1}")
        plt.imshow(y, cmap='gray')
        plt.axis('off')

        plt.subplot(nums, 3, 3*idx + 3)
        plt.title(f"Prediction {idx+1}")
        plt.imshow(y_pred, cmap='gray')
        plt.axis('off')

# Dataset & Data Loader
test_dataset = myData(type='test', data_path=DATA_PATH, transform=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=False)

# Prediction
trainer = pl.Trainer()
segmentor = Segmentor(model).load_from_checkpoint(CHECKPOINT_PATH)
trainer.test(segmentor, test_loader)

# Visualization
visualize_prediction(model=model, dataset=test_dataset, nums=5)