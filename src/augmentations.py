import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


def simclr_transform_cifar10():
    """
    CPU-friendly SimCLR augmentations for CIFAR-10 (32x32).

    Key ideas preserved:
    - Random crop
    - Random flip
    - Mild color jitter
    - Two different views per image (handled elsewhere)

    Expensive ops (e.g., GaussianBlur) are intentionally removed
    to keep training fast on CPU.
    """
    return T.Compose([
        T.RandomResizedCrop(
            size=32,
            scale=(0.6, 1.0),
            interpolation=InterpolationMode.BILINEAR
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply(
            [T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            )],
            p=0.5
        ),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        ),
    ])
