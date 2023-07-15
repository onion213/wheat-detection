import ast
import csv
import pathlib
import random
from typing import Literal, Optional

import pydantic
import pytorch_lightning as pl
import torch
import torch.utils.data
from PIL import Image

from wheat_detection.logger import Logger
from wheat_detection.transforms import get_transform

logger = Logger()


BBoxForms = Literal["xywh", "xyxy"]


class BBox(pydantic.BaseModel):
    xmin: int
    ymin: int
    width: int
    height: int

    @classmethod
    def from_str(cls, bbox_str: str):
        """parse bbox_str of format: '[875.0, 740.0, 94.0, 61.0]'"""
        # values = list(map(int, bbox_str[1:-1].split(", ")))
        values = list(map(int, ast.literal_eval(bbox_str)))
        return BBox(xmin=values[0], ymin=values[1], width=values[2], height=values[3])

    def to_list(self, form: BBoxForms = "xywh"):
        if form == "xywh":
            return [self.xmin, self.ymin, self.width, self.height]
        elif form == "xyxy":
            return [self.xmin, self.ymin, self.xmin + self.width, self.ymin + self.height]

    @pydantic.validator("*")
    def is_not_negative(cls, value):
        assert value >= 0
        return value


class Annotation(pydantic.BaseModel):
    img_id: str
    img_width: int
    img_height: int
    bbox: BBox
    source: str

    @pydantic.root_validator(pre=True)
    def bbox_is_inside_img(cls, values):
        _bbox = values["bbox"]
        _img_width = values["img_width"]
        _img_height = values["img_height"]
        # MEMO: ignoring for now. Can't pass object to validator?
        if values is not None:
            return values
        assert isinstance(_bbox, BBox)
        assert _bbox.xmin + _bbox.width <= _img_width
        assert _bbox.ymin + _bbox.height <= _img_height
        return values

    @pydantic.validator("bbox")
    def bbox_is_4_ints(cls, value):
        _bbox = value
        assert isinstance(_bbox.xmin, int)
        assert isinstance(_bbox.ymin, int)
        assert isinstance(_bbox.width, int)
        assert isinstance(_bbox.height, int)
        return value


class WheatDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir: pathlib.Path, img_ids: list[str], anns: list[Annotation]):
        # set attributes
        self.img_dir = img_dir
        self.anns = anns
        self.img_ids = img_ids

        # validate image_dir
        if not self.img_dir.exists():
            raise FileNotFoundError(f"directory not found; img_dir: {self.img_dir}")

        # validate annotations
        for img_id in self.img_ids:
            self.validate_img_file_exists(img_id=img_id)

        self.transform = get_transform()

    def validate_img_file_exists(self, img_id: str):
        if not self.img_id_to_path(img_id=img_id).exists():
            raise FileNotFoundError(f"img file not found; img_dir: {self.img_dir}, img_id: {img_id}")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index) -> tuple[Image.Image, dict]:
        img_id = self.img_ids[index]
        img_path = self.img_id_to_path(img_id=img_id)
        img = Image.open(img_path).convert("RGB")
        target_anns = list(ann for ann in self.anns if ann.img_id == img_id)
        boxes = list(ann.bbox.to_list(form="xyxy") for ann in target_anns)
        boxes = torch.as_tensor(boxes, dtype=torch.int64)
        labels = torch.zeros((len(target_anns),), dtype=torch.int64)  # there is only one class in wheat dataset
        target = {"boxes": boxes, "image_id": img_id, "labels": labels}
        img, target = self.transform(img, target)
        return img, target

    def img_id_to_path(self, img_id: str) -> pathlib.Path:
        return self.img_dir / (img_id + ".jpg")


def random_train_valid_split(size: int, train_ratio: float, seed: Optional[int] = None) -> tuple[list[int], list[int]]:
    if seed is not None:
        random.seed(seed)

    train_size = int(size * train_ratio)

    indexes = list(range(size))
    random.shuffle(indexes)
    train_indexes = indexes[:train_size]
    valid_indexes = indexes[train_size:]

    return train_indexes, valid_indexes


class WheatDataModule(pl.LightningDataModule):
    def __init__(
        self, img_dir: pathlib.Path, ann_csv_path: pathlib.Path, train_ratio: float = 0.8, batch_size: int = 32
    ):
        logger.info("Construct DataModule Start")
        super().__init__()
        if not img_dir.exists:
            raise FileNotFoundError(2, f"no such directory; img_dir: {img_dir}")
        self.img_dir = img_dir
        self.imgs, self.anns = parse_ann_csv(ann_csv_path)
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        logger.info("Construct DataModule Done")
        logger.info("Setting up datasets Start")
        self.setup()
        logger.info("Setting up datasets Done")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_size = len(self.imgs)
            train_indexes, valid_indexes = random_train_valid_split(size=full_size, train_ratio=self.train_ratio)
            train_img_ids = list(img for i, img in enumerate(self.imgs) if i in train_indexes)
            valid_img_ids = list(img for i, img in enumerate(self.imgs) if i in valid_indexes)
            self.train_dataset = WheatDataset(img_dir=self.img_dir, img_ids=train_img_ids, anns=self.anns)
            self.valid_dataset = WheatDataset(img_dir=self.img_dir, img_ids=valid_img_ids, anns=self.anns)

    def collate_fn(self, batch):
        """
        convert list[tuple[Image.Image, list[Annotation]]]
        -> tuple[tuple[Image.Image], tuple[list[Annotation]]]
        """
        return tuple(zip(*batch))

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)


def parse_ann_csv(ann_csv_path: pathlib.Path) -> tuple[list[str], list[Annotation]]:
    """expected csv content:
    image_id,width,height,bbox,source
    b6ab77fd7,1024,1024,"[834.0, 222.0, 56.0, 36.0]",usask_1
    ...
    """
    if not ann_csv_path.exists:
        raise FileNotFoundError(2, f"No such file: ${ann_csv_path}")
    with open(ann_csv_path) as f:
        reader = csv.DictReader(f)
        anns = list(
            Annotation(
                img_id=row["image_id"],
                bbox=BBox.from_str(row["bbox"]),
                img_width=int(row["width"]),
                img_height=int(row["height"]),
                source=row["source"],
            )
            for row in reader
        )
        imgs = list(set(ann.img_id for ann in anns))
    return imgs, anns
