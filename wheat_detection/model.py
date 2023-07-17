import pytorch_lightning as pl
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self, batch):
        images, targets = batch
        loss = self.calc_loss(images, targets)
        return {"loss": loss}

    def calc_loss(self, images, targets):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [lr_scheduler]


class FasterRCNN(Model):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # COCO V1 データで学習済みの重みを使用する
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        # 予測出力部分を差し替える
        in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model = model

    def calc_loss(self, images, targets):
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        return total_loss
