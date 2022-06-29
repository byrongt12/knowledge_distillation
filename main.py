import torch
import torchvision
import torchvision.transforms as transforms

from ResidualBlock import ResidualBlock
from ResNet import ResNet
from Train import train_model, getFeatureMaps
from Test import test_model
from Augmentation import augment
from Print import printModel, printModelSummary, imshow, getModelWeights, printFeatureMaps

if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Image preprocessing modules
    transform = augment()

    # CIFAR-100 dataset
    train_dataset = torchvision.datasets.CIFAR100(root='./data/',
                                                  train=True,
                                                  transform=transform,
                                                  download=True)

    test_dataset = torchvision.datasets.CIFAR100(root='./data/',
                                                 train=False,
                                                 transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=128,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=128,
                                              shuffle=False)

    # RESNET 110

    '''chk_path = "./resnet.ckpt"
    if path.exists(chk_path):
        print("load model")
        model = ResNet(ResidualBlock, [18, 18, 18]).to(device)
        model.load_state_dict(torch.load(chk_path))
    else:'''
    print("build model")
    model = ResNet(ResidualBlock, [18, 18, 18]).to(device)

    firstConvWeights = False
    allConvWeightShape = False
    modelSummary = False
    printModelSummary(model, firstConvWeights, allConvWeightShape, modelSummary)

    train_model(model, device, train_loader, test_loader)

    # test_model(model, device, test_loader)

    # Save the model checkpoint
    torch.save(model.state_dict(), 'resnet.ckpt')
