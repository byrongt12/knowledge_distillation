import torchvision.transforms as transforms

# transforms.RandomRotation(degrees=(0, 180))
# transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
# transforms.ColorJitter(brightness=.5, hue=.3)
# transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
# transforms.RandomAutocontrast()
# transforms.RandomEqualize()
# transforms.RandAugment()

def transformTrainData():
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32, padding=4),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])

    return transform


def transformTestData():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])

    return transform
