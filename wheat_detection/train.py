import pathlib

from pytorch_lightning import Trainer

from wheat_detection.datamodule import WheatDataModule
from wheat_detection.model import FasterRCNN

data_module = WheatDataModule(img_dir=pathlib.Path("./data/train"), ann_csv_path=pathlib.Path("./data/train.csv"))
model_module = FasterRCNN(num_classes=1)

trainer = Trainer(accelerator="cpu", devices=1, max_epochs=2)
trainer.fit(model_module, datamodule=data_module)
