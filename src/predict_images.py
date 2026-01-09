import argparse
from pathlib import Path
import random
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# -----------------------------
# CIFAR-10 metadata
# -----------------------------
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


# -----------------------------
# Model definitions
# -----------------------------
def build_resnet18_torchvision(num_classes: int = 10) -> nn.Module:
    from torchvision import models
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    from torchvision.models.resnet import ResNet, BasicBlock

    class CIFARResNet18(ResNet):
        def __init__(self, num_classes: int = 10):
            super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()

    return CIFARResNet18(num_classes=num_classes)


# -----------------------------
# Checkpoint utilities
# -----------------------------
def _strip_prefix(k: str) -> str:
    prefixes = [
        "module.",
        "model.",
        "net.",
        "backbone.",
        "encoder.",
        "student.",
        "online_encoder.",
        "target_encoder.",
    ]
    for p in prefixes:
        if k.startswith(p):
            return k[len(p):]
    return k


def _extract_state_dict(ckpt_obj: object) -> dict:
    if not isinstance(ckpt_obj, dict):
        raise ValueError("Checkpoint is not a dict. Unsupported format.")

    for key in ["model", "state_dict", "model_state_dict", "net", "encoder"]:
        if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
            return ckpt_obj[key]

    tensor_items = {k: v for k, v in ckpt_obj.items() if isinstance(v, torch.Tensor)}
    if len(tensor_items) > 0:
        return tensor_items

    for key in ckpt_obj.keys():
        if isinstance(ckpt_obj[key], dict):
            nested = {k: v for k, v in ckpt_obj[key].items() if isinstance(v, torch.Tensor)}
            if len(nested) > 0:
                return nested

    raise ValueError("Could not find a state_dict inside checkpoint.")


def detect_resnet_variant_from_state(state: dict) -> str:
    # Look for conv1.weight and inspect kernel size
    if "conv1.weight" in state and isinstance(state["conv1.weight"], torch.Tensor):
        kH, kW = int(state["conv1.weight"].shape[2]), int(state["conv1.weight"].shape[3])
        if (kH, kW) == (3, 3):
            return "cifar"
        if (kH, kW) == (7, 7):
            return "torchvision"
    return "torchvision"


def load_full_model_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = _extract_state_dict(ckpt)
    state = {_strip_prefix(k): v for k, v in state.items()}

    model_state = model.state_dict()
    matched = {k: v for k, v in state.items() if (k in model_state and v.shape == model_state[k].shape)}
    ratio = len(matched) / max(1, len(model_state))

    print(f"\n[FINETUNE] Loaded checkpoint file: {ckpt_path}")
    print(f"Checkpoint keys: {len(state)} | Model keys: {len(model_state)}")
    print(f"Matched keys: {len(matched)} ({ratio*100:.1f}%)")

    if ratio < 0.70:
        raise RuntimeError(
            f"❌ Finetune checkpoint does not match model architecture (matched {ratio*100:.1f}%)."
        )

    model.load_state_dict(matched, strict=False)
    print("✅ Finetune weights loaded.")


def load_linear_eval_checkpoint(linear_ckpt_path: Path, device: torch.device):
    """
    Your best_linear.pt contains:
    - one int (epoch)
    - one OrderedDict (linear head state_dict)
    """
    ckpt = torch.load(str(linear_ckpt_path), map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError("Linear eval checkpoint is not a dict.")

    linear_state = None
    epoch_val = None

    for k, v in ckpt.items():
        if isinstance(v, (OrderedDict, dict)):
            linear_state = v
        if isinstance(v, int):
            epoch_val = v

    if linear_state is None:
        raise ValueError("Could not find linear head state_dict in best_linear.pt")

    # strip prefixes on linear keys too
    linear_state = {_strip_prefix(k): v for k, v in linear_state.items()}

    print(f"\n[LINEAR EVAL] Loaded linear checkpoint: {linear_ckpt_path}")
    if epoch_val is not None:
        print(f"Best epoch stored in checkpoint: {epoch_val}")
    print(f"Linear state keys: {len(linear_state)} (showing first 10)")
    print(list(linear_state.keys())[:10])

    return linear_state


# -----------------------------
# Linear-eval model wrapper
# -----------------------------
class LinearEvalModel(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int = 10):
        super().__init__()
        self.encoder = encoder
        # remove classifier head, keep features
        feat_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.encoder(x)
        return self.classifier(feats)


# -----------------------------
# Visualization helpers
# -----------------------------
def unnormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return img_tensor * std + mean


@torch.no_grad()
def run_predictions(model: nn.Module, test_ds, n: int, seed: int, save_path: str, no_save: bool):
    random.seed(seed)
    torch.manual_seed(seed)

    idxs = random.sample(range(len(test_ds)), k=min(n, len(test_ds)))

    cols = 4
    rows = (len(idxs) + cols - 1) // cols
    plt.figure(figsize=(4 * cols, 4 * rows))

    correct = 0

    for i, idx in enumerate(idxs):
        x, y = test_ds[idx]
        x_in = x.unsqueeze(0)

        logits = model(x_in)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred = int(torch.argmax(probs).item())
        conf = float(probs[pred].item())

        correct += int(pred == y)

        x_disp = unnormalize(x).clamp(0, 1)
        img = x_disp.permute(1, 2, 0).numpy()

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"GT: {CIFAR10_CLASSES[y]}\nPred: {CIFAR10_CLASSES[pred]} ({conf*100:.1f}%)", fontsize=10)
        plt.axis("off")

    acc = (correct / len(idxs)) * 100.0
    plt.suptitle(f"Predictions ({correct}/{len(idxs)} correct, sample acc: {acc:.1f}%)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if not no_save:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(sp, dpi=150, bbox_inches="tight")
        print("Saved:", sp)

    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="finetune", choices=["finetune", "linear_eval"])
    parser.add_argument("--ckpt_path", type=str, help="For finetune: full model checkpoint path")

    parser.add_argument("--encoder_ckpt", type=str, default=None,
                        help="For linear_eval: encoder checkpoint (SimCLR). Use NONE for scratch/random encoder.")
    parser.add_argument("--linear_ckpt", type=str, default=None,
                        help="For linear_eval: best_linear.pt path")

    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="results/predictions/preds.png")
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu")  # your setup is CPU
    print("Device:", device)

    # CIFAR-10 test dataset
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    if args.mode == "finetune":
        if not args.ckpt_path:
            raise ValueError("--ckpt_path is required for mode=finetune")

        # detect architecture variant from ckpt
        ckpt_obj = torch.load(args.ckpt_path, map_location="cpu")
        state = _extract_state_dict(ckpt_obj)
        state = {_strip_prefix(k): v for k, v in state.items()}
        variant = detect_resnet_variant_from_state(state)
        print("Detected model variant:", variant)

        model = build_resnet18_cifar(10) if variant == "cifar" else build_resnet18_torchvision(10)
        model.to(device)
        model.eval()

        load_full_model_checkpoint(model, Path(args.ckpt_path), device)

        # run predictions on CPU
        model_cpu = model.cpu()
        run_predictions(model_cpu, test_ds, args.n, args.seed, args.save_path, args.no_save)

    else:
        # linear_eval mode
        if not args.linear_ckpt:
            raise ValueError("--linear_ckpt is required for mode=linear_eval")

        # Determine encoder variant:
        # If encoder_ckpt is NONE → scratch encoder: choose torchvision by default.
        encoder_variant = "torchvision"
        encoder_state = None

        if args.encoder_ckpt and args.encoder_ckpt.upper() != "NONE":
            enc_obj = torch.load(args.encoder_ckpt, map_location="cpu")
            encoder_state = _extract_state_dict(enc_obj)
            encoder_state = {_strip_prefix(k): v for k, v in encoder_state.items()}
            encoder_variant = detect_resnet_variant_from_state(encoder_state)

        print("Encoder variant for linear eval:", encoder_variant)

        encoder = build_resnet18_cifar(10) if encoder_variant == "cifar" else build_resnet18_torchvision(10)

        # Build linear-eval model wrapper (encoder -> features -> linear head)
        model = LinearEvalModel(encoder, num_classes=10)
        model.to(device)
        model.eval()

        # Load encoder weights if provided
        if encoder_state is not None:
            # load only matching encoder keys
            enc_model_state = model.encoder.state_dict()
            matched = {k: v for k, v in encoder_state.items() if (k in enc_model_state and v.shape == enc_model_state[k].shape)}
            ratio = len(matched) / max(1, len(enc_model_state))
            print(f"Encoder keys matched: {len(matched)} ({ratio*100:.1f}%)")
            if ratio < 0.70:
                raise RuntimeError("❌ Encoder checkpoint does not match encoder architecture.")
            model.encoder.load_state_dict(matched, strict=False)
            print("✅ Encoder weights loaded (SimCLR pretrained).")
        else:
            print("Using RANDOM encoder (scratch linear eval).")

        # Load linear head weights from best_linear.pt
        linear_state = load_linear_eval_checkpoint(Path(args.linear_ckpt), device)

        # The linear_state should match either:
        # - classifier.weight / classifier.bias
        # - fc.weight / fc.bias
        # We'll try both mappings safely.
        target_state = model.state_dict()
        mapped = {}

        # 1) Direct match
        for k, v in linear_state.items():
            if k in target_state and v.shape == target_state[k].shape:
                mapped[k] = v

        # 2) If ckpt is bare Linear layer: {'weight','bias'}
        if len(mapped) == 0 and "weight" in linear_state and "bias" in linear_state:
            if ("classifier.weight" in target_state and
                linear_state["weight"].shape == target_state["classifier.weight"].shape):
                mapped["classifier.weight"] = linear_state["weight"]

            if ("classifier.bias" in target_state and
                linear_state["bias"].shape == target_state["classifier.bias"].shape):
                mapped["classifier.bias"] = linear_state["bias"]

        # 3) If ckpt stored as fc.* (rare)
        if len(mapped) == 0:
            for k, v in linear_state.items():
                if k.startswith("fc."):
                    new_k = "classifier." + k[len("fc."):]
                    if new_k in target_state and v.shape == target_state[new_k].shape:
                        mapped[new_k] = v

        if len(mapped) == 0:
            raise RuntimeError(
                "❌ Could not map linear head weights into LinearEvalModel. "
                "Your checkpoint might save the head under different names."
            )

        model.load_state_dict(mapped, strict=False)
        print(f"✅ Linear head loaded. (mapped keys: {list(mapped.keys())})")

        model_cpu = model.cpu()
        run_predictions(model_cpu, test_ds, args.n, args.seed, args.save_path, args.no_save)


if __name__ == "__main__":
    main()
