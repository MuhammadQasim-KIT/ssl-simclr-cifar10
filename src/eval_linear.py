import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from tqdm import tqdm

from src.models import SimCLR


def get_cifar10_supervised_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


@torch.no_grad()
def evaluate(encoder, classifier, loader, device):
    encoder.eval()
    classifier.eval()
    crit = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        h = encoder(x)
        logits = classifier(h)
        loss = crit(logits, y)

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--log_every", type=int, default=50)

    p.add_argument("--mode", choices=["pretrained", "scratch"], default="pretrained")
    p.add_argument("--ckpt_path", type=str, default="./runs/simclr_cifar10/checkpoints/last.pt")
    p.add_argument("--out_dir", type=str, default="./runs/linear_eval")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Mode:", args.mode)

    # Build encoder
    simclr = SimCLR(backbone="resnet18", proj_dim=128)

    if args.mode == "pretrained":
        if not os.path.isfile(args.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        simclr.load_state_dict(ckpt["model"], strict=False)
        print("Loaded SimCLR checkpoint:", args.ckpt_path)
    else:
        print("Using random init encoder (scratch baseline).")

    encoder = simclr.encoder.to(device)

    # Freeze encoder for linear eval
    for p_ in encoder.parameters():
        p_.requires_grad = False
    encoder.eval()

    # Feature dimension
    with torch.no_grad():
        dummy = torch.randn(2, 3, 32, 32).to(device)
        feat_dim = encoder(dummy).shape[1]

    classifier = nn.Linear(feat_dim, 10).to(device)

    # Data
    tfm = get_cifar10_supervised_transforms()
    train_ds = CIFAR10(root=args.data_dir, train=True, download=True, transform=tfm)
    test_ds = CIFAR10(root=args.data_dir, train=False, download=True, transform=tfm)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    run_name = f"{args.mode}_linear_eval_resnet18"
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

    best_acc = -1.0
    best_epoch = 0
    patience_left = args.patience
    global_step = 0

    print(f"\n[Linear Eval] epochs={args.epochs} batch={args.batch_size} lr={args.lr} patience={args.patience}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        classifier.train()
        running = 0.0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch_idx, (x, y) in enumerate(pbar, start=1):
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                h = encoder(x)

            logits = classifier(h)
            loss = crit(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item()
            global_step += 1

            avg_so_far = running / batch_idx
            pbar.set_postfix(loss=f"{avg_so_far:.4f}")

            if batch_idx % args.log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  [epoch {epoch:03d}] batch {batch_idx:04d}/{len(train_dl)} "
                      f"avg_loss={avg_so_far:.4f} lr={lr_now:.5f}")

            writer.add_scalar("train/loss_step", loss.item(), global_step)

        scheduler.step()

        train_loss = running / len(train_dl)
        test_loss, test_acc = evaluate(encoder, classifier, test_dl, device)
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/acc", test_acc, epoch)
        writer.add_scalar("train/lr", lr_now, epoch)

        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"test_loss={test_loss:.4f} | test_acc={test_acc*100:.2f}% | lr={lr_now:.5f} | time={dt:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            patience_left = args.patience
            torch.save({
                "epoch": epoch,
                "classifier": classifier.state_dict(),
                "best_acc": best_acc,
                "mode": args.mode,
                "ckpt_used": args.ckpt_path if args.mode == "pretrained" else None
            }, os.path.join(out_dir, "best_linear.pt"))
        else:
            patience_left -= 1
            print(f"  No improvement. Patience left: {patience_left}")
            if patience_left <= 0:
                print(f"Early stopping. Best acc={best_acc*100:.2f}% at epoch {best_epoch}.")
                break

    writer.close()
    print(f"\nDone. Best linear eval accuracy: {best_acc*100:.2f}% (epoch {best_epoch})")
    print(f"Saved: {os.path.join(out_dir, 'best_linear.pt')}")
    print(f"TensorBoard logs: {os.path.join(out_dir, 'tb')}")


if __name__ == "__main__":
    main()
