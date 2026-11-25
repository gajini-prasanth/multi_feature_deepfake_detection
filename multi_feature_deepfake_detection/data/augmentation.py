import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Training augmentations inspired by LASTED (compression, blur, noise, flips)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.25),
        A.OneOf([
            A.ImageCompression(quality_range=(50, 95), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
        ], p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Validation and test preprocessing."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_degradation_transform(
    degradation_type: str,
    severity: int | float,
    image_size: int = 224,
) -> A.Compose:
    """Robustness evaluation transforms for Experiment 6."""
    ops = [A.Resize(image_size, image_size)]
    if degradation_type == "jpeg":
        q = int(severity)
        ops.append(A.ImageCompression(quality_range=(q, q), p=1.0))
    elif degradation_type == "blur":
        k = int(severity)
        ops.append(A.GaussianBlur(blur_limit=(k, k), p=1.0))
    elif degradation_type == "noise":
        std = float(severity)
        ops.append(A.GaussNoise(std_range=(std, std), p=1.0))
    elif degradation_type == "resize":
        scale_size = int(severity)
        ops.extend([
            A.Resize(scale_size, scale_size),
            A.Resize(image_size, image_size),
        ])
    else:
        raise ValueError(f"Unknown degradation_type: {degradation_type}")

    ops.extend([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return A.Compose(ops)
