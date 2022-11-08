import warnings
from argparse import ArgumentParser
from os import listdir, path

import pytorch_lightning as pl
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader

from nn_models import resnet20, PLWrapper, CustomProgressBar

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

TEST_FACTOR = 0.2


class CaptchaSet(Dataset):
    def __init__(self, root, transform):
        self.transform = transform

        self.root = root
        self.imgs = listdir(root)

    @staticmethod
    def _get_label_from_fn(fn):
        raw_label = fn.split("_")[0]
        labels = [ord(char) - ord("a") for char in raw_label]
        if len(labels) == 4: labels.append(26)
        return labels

    def __getitem__(self, idx):
        img = Image.open(path.join(self.root, self.imgs[idx])).convert("L")

        label = CaptchaSet._get_label_from_fn(self.imgs[idx])

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


class CaptchaDataLoader(pl.LightningDataModule):
    def __init__(self, root: str, batch_size: int):
        super().__init__()
        self.save_hyperparameters("batch_size")

        self.root = root
        self.batch_size = batch_size
        self.transform = None
        self.train_loader = None
        self.val_loader = None
        self.table = [0] * 156 + [1] * 100

    def binarize(self, img):
        return img.point(self.table, "1")

    def setup(self, stage: str):
        transform = transforms.Compose([
            transforms.Lambda(self.binarize),
            transforms.ToTensor(),
        ])
        dataset = CaptchaSet(root=self.root, transform=transform)
        test_count = int(len(dataset) * TEST_FACTOR)
        train_count = len(dataset) - test_count
        train_set, val_set = random_split(dataset, [train_count, test_count])
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def main(hparams):
    pb = CustomProgressBar()
    trainer = pl.Trainer(accelerator=hparams.accelerator, max_epochs=hparams.epochs, log_every_n_steps=10,
                         callbacks=[pb])
    trainer.fit(model=PLWrapper(model=resnet20(), lr=hparams.lr),
                datamodule=CaptchaDataLoader(root=hparams.path, batch_size=hparams.batch_size),
                ckpt_path=hparams.ckpt_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="labelled", help="Path to captcha images")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Recover from checkpoint")
    hparams = parser.parse_args()
    main(hparams)
