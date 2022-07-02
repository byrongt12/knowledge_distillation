from os import path

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR100

from Augmentation import transformTrainData, transformTestData
from DeviceLoader import get_device, to_device, ToDeviceLoader
from Print import printModelSummary, plot_acc, printFeatureMaps, printModel, show_batch
from ResNet import ResNet
from ResidualBlock import ResidualBlock
from Train import train_model, evaluate

if __name__ == '__main__':
    # Device configuration
    device = get_device()

    # Image preprocessing modules
    transformTrainData = transformTrainData()
    transformTestData = transformTestData()

    # CIFAR-100 dataset
    BATCH_SIZE = 100
    train_dataset = CIFAR100(root='./data/', train=True, transform=transformTrainData, download=True)
    test_dataset = CIFAR100(root='./data/', train=False, transform=transformTestData)

    # Data loader
    train_dl = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, )
    test_dl = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, )

    '''show_batch(train_dl)
    exit()'''

    # RESNET 110
    '''chk_path = "./resnet110.ckpt"
    if path.exists(chk_path):
        print("load model")
        model = ResNet(ResidualBlock, [18, 18, 18])
        model.load_state_dict(torch.load(chk_path))
    else:'''
    print("build model")
    model = ResNet(ResidualBlock, [18, 18, 18])

    model = to_device(model, device)
    train_dl = ToDeviceLoader(train_dl, device)
    test_dl = ToDeviceLoader(test_dl, device)

    firstConvWeights = False
    allConvWeightShape = False
    modelSummary = False
    printModelSummary(model, firstConvWeights, allConvWeightShape, modelSummary)

    epochs = 80
    optimizer = torch.optim.Adam
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    scheduler = torch.optim.lr_scheduler.OneCycleLR

    history = [evaluate(model, test_dl)]
    history += train_model(epochs=epochs, train_dl=train_dl, test_dl=test_dl, model=model, optimizer=optimizer,
                           max_lr=max_lr,
                           grad_clip=grad_clip, weight_decay=weight_decay,
                           scheduler=torch.optim.lr_scheduler.OneCycleLR)

    print("Hyper parameters:")
    print("Number of epochs: " + str(epochs))
    print("Optimizer: " + str(optimizer))
    print("max learning rate: " + str(max_lr))
    print("gradient clip value: " + str(grad_clip))
    print("weight decay: " + str(weight_decay))
    print("scheduler: " + str(scheduler))

    plot_acc(history)
    # Save the model checkpoint
    torch.save(model.state_dict(), 'resnet110.ckpt')
