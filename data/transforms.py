"""Image transforms for the PSD-Few few-shot pipeline."""

from torchvision import transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int = 84):
    """Training augmentation: resize-crop, flip, color jitter, normalize."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def build_val_transform(image_size: int = 84):
    """Validation/test transform: resize-center-crop, normalize."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
