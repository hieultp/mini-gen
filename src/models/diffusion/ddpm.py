import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DiffusionModel(pl.LightningModule):
    def __init__(
        self,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=32,
        in_channels=1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # UNet for noise prediction
        self.model = SimpleUNet(in_channels=in_channels)

        self.validation_x = torch.randn(
            8,
            self.hparams.in_channels,
            self.hparams.img_size,
            self.hparams.img_size,
        )

    def forward(self, x, t):
        return self.model(x, t)

    def q_sample(self, x_0, t):
        # Sample from q(x_t | x_0)
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t])

        noise = torch.randn_like(x_0)
        return (
            sqrt_alphas_cumprod_t.view(-1, 1, 1, 1) * x_0
            + sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1) * noise
        ), noise

    def on_train_start(self) -> None:
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)

    def on_train_end(self) -> None:
        self.alphas_cumprod = self.alphas_cumprod.cpu()

    @torch.inference_mode()
    def sample(self, batch_size=16, device="cuda"):
        # Start from pure noise
        x = torch.randn(
            batch_size,
            self.hparams.in_channels,
            self.hparams.img_size,
            self.hparams.img_size,
        ).to(device)

        # Reverse diffusion process
        for t in reversed(range(self.hparams.timesteps)):
            # Convert t to tensor
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Get current alpha values
            alpha = self.alphas[t]
            alpha_hat = self.alphas_cumprod[t]

            # Get noise prediction
            predicted_noise = self(x, t_tensor)

            # Get mean for reverse process
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.betas[t])
            else:
                noise = torch.zeros_like(x)
                sigma = 0

            x = (
                1
                / torch.sqrt(alpha)
                * (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * predicted_noise)
                + sigma * noise
            )

        return x.clamp(-1, 1)

    def training_step(self, batch, batch_idx):
        x_0, _ = batch

        # Sample t uniformly
        t = torch.randint(0, len(self.betas), (x_0.shape[0],), device=self.device)

        # Get noisy image and original noise
        x_t, noise = self.q_sample(x_0, t)

        # Predict noise
        pred_noise = self(x_t, t)

        # Calculate loss
        loss = F.mse_loss(pred_noise, noise)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.validation_x.to(self.device)

        # Generate and log sample images every few epochs
        if batch_idx == 0:
            # Reverse diffusion process
            for t in reversed(range(self.hparams.timesteps)):
                # Convert t to tensor
                t_tensor = torch.full(
                    (x.size(0),), t, device=self.device, dtype=torch.long
                )

                # Get current alpha values
                alpha = self.alphas[t]
                alpha_hat = self.alphas_cumprod[t]

                # Get noise prediction
                predicted_noise = self(x, t_tensor)

                # Get mean for reverse process
                if t > 0:
                    noise = torch.randn_like(x)
                    sigma = torch.sqrt(self.betas[t])
                else:
                    noise = torch.zeros_like(x)
                    sigma = 0

                x = (
                    1
                    / torch.sqrt(alpha)
                    * (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * predicted_noise)
                    + sigma * noise
                )

            x = x.clamp(-1, 1)
            grid = torchvision.utils.make_grid(x, normalize=True)
            self.logger.experiment.add_image(
                "generated_images", grid, self.current_epoch
            )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-4)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(32), nn.Linear(32, 32 * 8), nn.ReLU()
        )

        # Encoder
        self.enc1 = self._make_conv_block(in_channels, 64)
        self.enc2 = self._make_conv_block(64, 128)
        self.enc3 = self._make_conv_block(128, 256)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )

        # Decoder
        self.dec3 = self._make_upconv_block(512, 128)
        self.dec2 = self._make_upconv_block(256, 64)
        self.dec1 = self._make_upconv_block(128, 32)

        self.final = nn.Conv2d(32, in_channels, 1)

    def _make_conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def _make_upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x, t):
        # Time embedding
        t = self.time_mlp(t)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Add time embedding
        b = b + t.view(-1, 256, 1, 1)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([b, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        return self.final(d1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
