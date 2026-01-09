import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10

from src.augmentations import simclr_transform_cifar10
from src.models import SimCLR


class TwoCropsTransform:
    """Apply the same base transform twice to get two correlated views."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        x1 = self.base_transform(x)
        x2 = self.base_transform(x)
        return x1, x2


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    NT-Xent loss used in SimCLR.

    z1, z2: (B, D) L2-normalized embeddings for two augmented views of the same batch.
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    # Cosine similarity matrix (since embeddings are normalized)
    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    # Mask self-similarity
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # Positive pairs: i <-> i+B
    pos_idx = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)

    # Cross-entropy where target is the index of the positive sample
    loss = F.cross_entropy(sim, pos_idx)
    return loss


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_dir", type=str, default="./runs/simclr_cifar10")
    parser.add_argument("--ckpt_path", type=str, default="./runs/simclr_cifar10/checkpoints/last.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Dataset: we use CIFAR-10 but ignore labels (self-supervised)
    base_transform = simclr_transform_cifar10()
    transform = TwoCropsTransform(base_transform)

    train_ds = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,  # important for contrastive loss batching
    )

    model = SimCLR(backbone="resnet18", proj_dim=args.proj_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    writer = SummaryWriter(log_dir=args.log_dir)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for (x1, x2), _ in train_dl:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            z1 = model(x1)
            z2 = model(x2)

            loss = nt_xent_loss(z1, z2, temperature=args.temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if global_step % 50 == 0:
                writer.add_scalar("train/loss_step", loss.item(), global_step)

            global_step += 1

        avg_loss = epoch_loss / len(train_dl)
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)

        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs} | loss={avg_loss:.4f} | time={dt:.1f}s")

        save_checkpoint(args.ckpt_path, model, optimizer, epoch)

    writer.close()
    print("Training finished.")
    print("Checkpoint saved to:", args.ckpt_path)
    print("TensorBoard logs in:", args.log_dir)


if __name__ == "__main__":
    main()
