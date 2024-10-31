from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.datasets.mnist import MNISTDataModule
from src.models.gan.dcgan import DCGAN


def main():
    # Load config
    with open("configs/dcgan.yaml") as f:
        config = yaml.safe_load(f)

    # Initialize data module
    dm = MNISTDataModule(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # Initialize model
    model = DCGAN(
        latent_dim=config["model"]["latent_dim"],
        feature_maps=config["model"]["feature_maps"],
        lr=config["model"]["lr"],
        beta1=config["model"]["beta1"],
    )

    # Callbacks
    checkpoint_dirpath = Path("checkpoints/dcgan").mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="g_loss",
        dirpath=checkpoint_dirpath,
        filename="dcgan-{epoch:02d}-{g_loss:.2f}",
        mode="min",
    )

    # Logger
    logger = TensorBoardLogger("logs", name="dcgan")

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
