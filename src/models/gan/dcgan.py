import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, feature_maps: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_maps = feature_maps

        self.main = nn.Sequential(
            # Input is latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # State: (feature_maps*8) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # State: (feature_maps*4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # State: (feature_maps*2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 2, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
            # Output: 1 x 32 x 32
        )

    def forward(self, x):
        x = x.view(x.size(0), self.latent_dim, 1, 1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, feature_maps: int = 64):
        super().__init__()
        self.main = nn.Sequential(
            # Input: 1 x 32 x 32
            nn.Conv2d(1, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: feature_maps x 16 x 16
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*2) x 8 x 8
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*4) x 4 x 4
            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # Output: 1 x 1 x 1
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)


class DCGAN(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int = 100,
        feature_maps: int = 64,
        lr: float = 0.0002,
        beta1: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Networks
        self.generator = Generator(latent_dim, feature_maps)
        self.discriminator = Discriminator(feature_maps)

        # Random noise for visualization
        self.validation_z = torch.randn(8, latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        opt_d, opt_g = self.optimizers()
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)

        # Train Discriminator
        # Generate fake images
        opt_d.zero_grad()
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        fake_imgs = self(z)

        real_labels = torch.ones(batch_size, device=self.device)
        fake_labels = torch.zeros(batch_size, device=self.device)

        # Real images
        real_preds = self.discriminator(real_imgs)
        d_real_loss = self.adversarial_loss(real_preds, real_labels)

        # Fake images
        fake_preds = self.discriminator(fake_imgs.detach())
        d_fake_loss = self.adversarial_loss(fake_preds, fake_labels)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        self.manual_backward(d_loss)
        opt_d.step()

        # Train Generator
        opt_g.zero_grad()
        preds = self.discriminator(fake_imgs)

        g_loss = self.adversarial_loss(preds, real_labels)
        self.manual_backward(g_loss)
        opt_g.step()

        # Logging
        self.log("g_loss", g_loss, prog_bar=True)
        self.log("d_loss", d_loss, prog_bar=True)

        return g_loss

    def validation_step(self, batch, batch_idx):
        z = self.validation_z.to(self.device)
        fake_imgs = self(z)

        # Log sample images
        if batch_idx == 0:
            grid = torchvision.utils.make_grid(fake_imgs, normalize=True)
            self.logger.experiment.add_image(
                "generated_images", grid, self.current_epoch
            )

    def configure_optimizers(self):
        lr = self.hparams.lr
        beta1 = self.hparams.beta1

        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999)
        )
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(beta1, 0.999)
        )

        return [opt_d, opt_g], []
