import torchvision.transforms as transforms

mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]


def augment():
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])

    return transform
