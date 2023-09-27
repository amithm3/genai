from typing import TYPE_CHECKING

import torch
from torch import nn, optim
from torchvision.utils import make_grid

from .generator import Generator
from .discriminator import Discriminator
from utils.checkpoints import save_checkpoint

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class PerceptualLoss:
    def __init__(self, model: "nn.Module", criterion=None):
        if criterion is None: criterion = nn.L1Loss()
        self.model = model.eval().requires_grad_(False)
        self.criterion = criterion

    def __call__(self, x, y):
        return self.criterion(self.model(x), self.model(y))


def get_cycle_gan_trainer(
        generatorA: "Generator", generatorB: "Generator",
        discriminatorA: "Discriminator", discriminatorB: "Discriminator",
        optimizerG: "optim.Optimizer", optimizerD: "optim.Optimizer",
        save_path: str, save_period: int,
        perceptual_loss=None,
        lambda_cycle: float = 10, lambda_identity: float = 0.5,
        writer: "SummaryWriter" = False, writer_period: int = 100,
        fixedA: "torch.Tensor" = None, fixedB: "torch.Tensor" = None,
):
    assert writer is None or (fixedA is not None and fixedB is not None), \
        "parameters `writer`, `fixedA` and `fixedB` are mutually inclusive"
    L1 = nn.L1Loss()
    MSE = nn.MSELoss()
    if writer:
        grid_realA = make_grid(fixedA, nrow=1, normalize=True)
        grid_realB = make_grid(fixedB, nrow=1, normalize=True)
        # writer.add_graph(..., ...)

    def trainer(DATA: dict[str, dict[str, "torch.Tensor"]], step: int) -> dict[str, float]:
        realA, realB = DATA["domain_0"]["image"], DATA["domain_1"]["image"]
        fakeA, fakeB = generatorA(realB), generatorB(realA)
        backA, backB = generatorA(fakeB), generatorB(fakeA)
        sameA, sameB = generatorA(realA), generatorB(realB)
        pred_realA, pred_realB = discriminatorA(realA), discriminatorB(realB)
        pred_fakeA_true, pred_fakeB_true = discriminatorA(fakeA.detach()), discriminatorB(fakeB.detach())

        # ===Discriminator Loss===
        # Adversarial Loss
        loss_adversarialDA = (MSE(pred_realA, torch.ones_like(pred_realA)) +
                              MSE(pred_fakeA_true, torch.zeros_like(pred_fakeA_true)))
        loss_adversarialDB = (MSE(pred_realB, torch.ones_like(pred_realB)) +
                              MSE(pred_fakeA_true, torch.zeros_like(pred_fakeA_true)))
        loss_adversarialD = (loss_adversarialDA + loss_adversarialDB) / 2
        # Total Loss
        lossD = loss_adversarialD
        # backprop
        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()
        # ---End Discriminator Loss---

        pred_fakeA_false, pred_fakeB_false = discriminatorA(fakeA), discriminatorB(fakeB)
        # ===Generator Loss===
        # Adversarial Loss
        loss_adversarialGA = MSE(pred_fakeA_false, torch.ones_like(pred_fakeA_false))
        loss_adversarialGB = MSE(pred_fakeB_false, torch.ones_like(pred_fakeB_false))
        loss_adversarialG = (loss_adversarialGA + loss_adversarialGB) / 2
        # Cycle Loss
        loss_cycleA = L1(backA, realA)
        loss_cycleB = L1(backB, realB)
        loss_cycle = (loss_cycleA + loss_cycleB) / 2
        # Identity Loss
        loss_identityA = L1(sameA, realA)
        loss_identityB = L1(sameB, realB)
        loss_identity = (loss_identityA + loss_identityB) / 2
        # Perceptual Loss
        if perceptual_loss is not None:
            loss_perceptualA = perceptual_loss(sameA, realA)
            loss_perceptualB = perceptual_loss(sameA, realB)
            loss_perceptual = (loss_perceptualA + loss_perceptualB) / 2
        else:
            loss_perceptual = torch.tensor(0)
        # Total Loss
        lossG = (loss_adversarialG + lambda_cycle * loss_cycle + lambda_identity * loss_identity)
        # backprop
        optimizerG.zero_grad()
        lossG.backward()
        optimizerG.step()
        # ---End Generator Loss---

        loss_total = lossD + lossG

        if writer is not None and step % writer_period == 0:
            writer.add_scalar("loss/adversarial_discriminator", loss_adversarialD.item(), step)
            writer.add_scalar("loss/adversarial_generator", loss_adversarialG.item(), step)
            writer.add_scalar("loss/cycle", loss_cycle.item(), step)
            writer.add_scalar("loss/identity", loss_identity.item(), step)
            writer.add_scalar("loss/perceptual", loss_perceptual.item(), step)
            writer.add_scalar("loss/total", loss_total.item(), step)
            with torch.inference_mode():
                grid_fakeA = make_grid(fakeA := generatorA(fixedB), nrow=1, normalize=True)
                grid_fakeB = make_grid(fakeB := generatorB(fixedA), nrow=1, normalize=True)
                grid_backA = make_grid(generatorA(fakeB), nrow=1, normalize=True)
                grid_backB = make_grid(generatorB(fakeA), nrow=1, normalize=True)
                grid_sameA = make_grid(generatorA(fixedA), nrow=1, normalize=True)
                grid_sameB = make_grid(generatorB(fixedB), nrow=1, normalize=True)
                writer.add_images("images/domainA",
                                  torch.stack([grid_realA, grid_fakeB, grid_backA, grid_sameA]), step)
                writer.add_images("images/domainB",
                                  torch.stack([grid_realB, grid_fakeA, grid_backB, grid_sameB]), step)

        if step % save_period == 0:
            save_checkpoint(
                save_path,
                {
                    "generatorA": generatorA,
                    "generatorB": generatorB,
                    "discriminatorA": discriminatorA,
                    "discriminatorB": discriminatorB,
                },
                {
                    "optimizerG": optimizerG,
                    "optimizerD": optimizerD,
                },
                step=step,
            )

        return {
            "loss": loss_total.item(),
        }

    return trainer


__all__ = [
    "PerceptualLoss",
    "get_cycle_gan_trainer"
]
