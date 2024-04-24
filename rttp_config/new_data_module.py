from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.dataset.rttp_dataset import RttpDataset
from rttp_config.rttp_config import ConfigRttp


class RgbdDataModule(LightningDataModule):
    def __init__(self,dataset_cfg: ConfigRttp):
        super().__init__()
        self.cfg=dataset_cfg


    # def setup(self, stage: str):
    #     # Assign train/val datasets for use in dataloaders
    #     if stage == "fit":
    #         mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
    #         self.mnist_train, self.mnist_val = random_split(
    #             mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
    #         )

    def train_dataloader(self):
        train_set = RttpDataset(self.cfg.dataset, phase='train')
        train_loader = DataLoader(train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                       num_workers=self.cfg.batch_size , pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_set = RttpDataset(self.cfg.dataset, phase='val')
        val_loader = DataLoader(val_set, batch_size=self.cfg.batch_size, shuffle=True,
                                  num_workers=self.cfg.batch_size * 2, pin_memory=True)
        return val_loader

    def test_dataloader(self):
        test_set = RttpDataset(self.cfg.dataset, phase='test')
        test_loader = DataLoader(test_set, batch_size=self.cfg.batch_size, shuffle=True,
                                  num_workers=self.cfg.batch_size * 2, pin_memory=True)
        return test_loader

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=32)