from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision.utils import make_grid

from utils.checkpoints import save_checkpoint

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def get_vae_trainer(
        vae_model: "nn.Module",
        optimizer: "optim.Optimizer",
        save_path: str, save_period: int,
        perceptual_loss=None, lambda_kl: float = 0.5,
        writer: "SummaryWriter" = False, writer_period: int = 100,
        fixed: "torch.Tensor" = None,
):
    assert writer is None or fixed is not None, \
        "parameters `writer` and `fixed` are mutually inclusive"
    if writer:
        grid_real = make_grid(fixed, nrow=1, normalize=True)
        # writer.add_graph(..., ...)

    def trainer(DATA: dict[str, "torch.Tensor"], step: int) -> dict[str, float]:
        real = DATA["image"]
        fake, mu, logvar = vae_model(real)

        # ===Loss===
        # Reconstruction Loss
        loss_reconstruction = F.mse_loss(fake, real, reduction="none").sum(axis=(1, 2, 3)).mean()
        # KL Divergence Loss
        loss_latent = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(axis=1).mean()
        # Perceptual Loss
        if perceptual_loss is not None:
            loss_perceptual = perceptual_loss(real, fake)
        else:
            loss_perceptual = torch.tensor(0)
        # ---End Loss---

        loss_total = loss_reconstruction + lambda_kl * loss_latent

        # ===Logging===
        if writer and step % writer_period == 0:
            grid_fake = make_grid(fake, nrow=1, normalize=True)
            writer.add_scalar("loss/reconstruction", loss_reconstruction, step)
            writer.add_scalar("loss/latent", loss_latent, step)
            writer.add_scalar("loss/perceptual", loss_perceptual if perceptual_loss is not None else "N/A", step)
            writer.add_scalar("loss/total", loss_total, step)
            writer.add_image("image/real", grid_real, step)
            writer.add_image("image/fake", grid_fake, step)
        # ---End Logging---

        # ===Optimization===
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        # ---End Optimization---

        # ===Saving===
        if step % save_period == 0:
            save_checkpoint(
                save_path,
                {"vae_model": vae_model},
                {"optimizer": optimizer},
                step=step,
            )
        # ---End Saving---

        return {
            "loss/reconstruction": loss_reconstruction.item(),
            "loss/perceptual": loss_perceptual.item(),
            "loss/latent": loss_latent.item(),
            "loss/total": loss_total.item(),
        }

    return trainer


__all__ = [
    "get_vae_trainer",
]
