from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.datasets.mnist import MNISTDataModule
from src.models.diffusion.ddpm import DiffusionModel


def main():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # Load config
    with open("configs/ddpm.yaml") as f:
        config = yaml.safe_load(f)

    # Initialize data module
    dm = MNISTDataModule(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # Initialize model
    model = DiffusionModel(
        timesteps=config["model"]["timesteps"],
        beta_start=config["model"]["beta_start"],
        beta_end=config["model"]["beta_end"],
    )

    # Callbacks
    checkpoint_dirpath = Path("./checkpoints/ddpm")
    checkpoint_dirpath.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        filename="ddpm-{epoch:02d}",
        save_last=True,
        save_top_k=0,
    )

    # Logger
    logger = TensorBoardLogger("logs", name="ddpm")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        precision=config["trainer"]["precision"],
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    # Train model
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
