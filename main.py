from os import path

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR100
import torch.nn.functional as F

from Augmentation import transformTrainData, transformTestData
from DeviceLoader import get_device, to_device, ToDeviceLoader
from Helper import getNumberOfConvolutionLayers
from Print import printModelSummary, plot_acc, printFeatureMaps, printModel, show_batch
from ResNet import ResNet
from ResidualBlock import ResidualBlock
from Train import train_model, evaluate, train_model_with_distillation
from Helper import getFeatureMaps

if __name__ == '__main__':

    teacher_model_number = 18  # ResNet 110
    student_model_number = 3  # ResNet 20

    epochs = 20
    BATCH_SIZE = 100

    optimizer = torch.optim.Adam
    max_lr = 0.01

    distill_optimizer = torch.optim.SGD
    distill_lr = 0.1

    grad_clip = 0
    weight_decay = 0
    scheduler = torch.optim.lr_scheduler.OneCycleLR
    kd_loss_type = 'ssim'

    # Device configuration
    device = get_device()

    # Image preprocessing modules
    transformTrainData = transformTrainData()
    transformTestData = transformTestData()

    # CIFAR-100 dataset
    train_dataset = CIFAR100(root='./data/', train=True, transform=transformTrainData, download=True)
    test_dataset = CIFAR100(root='./data/', train=False, transform=transformTestData)

    # Data loader
    train_dl = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    '''show_batch(train_dl)
    exit()'''

    # RESNET 110
    chk_path = "./resnet110_0.7018.ckpt"
    teacher_model = None
    if path.exists(chk_path):
        print("Loaded teacher model.")
        teacher_model = ResNet(ResidualBlock, [teacher_model_number, teacher_model_number, teacher_model_number])
        teacher_model.load_state_dict(torch.load(chk_path))
    else:
        print("No teacher model found.")
        exit()

    student_model = ResNet(ResidualBlock, [student_model_number, student_model_number, student_model_number])

    teacher_model = to_device(teacher_model, device)
    student_model = to_device(student_model, device)
    train_dl = ToDeviceLoader(train_dl, device)
    test_dl = ToDeviceLoader(test_dl, device)

    # Train student:
    '''firstConvWeights = False
    allConvWeightShape = False
    modelSummary = True
    printModelSummary(student_model, firstConvWeights, allConvWeightShape, modelSummary)
    exit()'''

    history = [evaluate(student_model, test_dl)]

    numOfFeatureMapsForTeacher = getNumberOfConvolutionLayers(teacher_model)
    numOfFeatureMapsForStudent = getNumberOfConvolutionLayers(student_model)
    heuristicToStudentDict = {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5,
        'f': 6,
        'g': 7,
        'h': 8,
        'i': 9,
        'j': 10,
        'k': 11,
        'l': 12,
        'm': 13,
        'n': 14,
        'o': 15,
        'p': 16,
        'q': 17,
        'r': 18,
    }
    # Get GA string and pass to function below.
    # del model
    heuristicString = "abcdefghijklmnopqr"
    history += train_model_with_distillation(heuristicString=heuristicString, heuristicToStudentDict=heuristicToStudentDict, epochs=epochs, train_dl=train_dl, test_dl=test_dl,
                                             student_model=student_model, student_model_number=student_model_number,
                                             teacher_model=teacher_model, teacher_model_number=teacher_model_number,
                                             device=device,
                                             optimizer=optimizer,
                                             max_lr=max_lr,
                                             weight_decay=weight_decay,
                                             scheduler=scheduler,
                                             kd_loss_type=kd_loss_type, distill_optimizer=distill_optimizer, distill_lr=distill_lr, grad_clip=grad_clip,)

    print("Hyper parameters:")
    print("Number of epochs: " + str(epochs))
    print("Batch size: " + str(BATCH_SIZE))
    print("Optimizer: " + str(optimizer))
    print("Max learning rate: " + str(max_lr))
    print("Gradient clip value: " + str(grad_clip))
    print("Weight decay: " + str(weight_decay))
    print("Scheduler: " + str(scheduler))
    print("KD loss type: " + str(kd_loss_type))
    print("Distill  optimizer : " + str(distill_optimizer))
    print("Distill  optimizer learning rate: " + str(distill_lr))
    print("Transforms on training data: ", end="")
    print(transformTrainData)

    plot_acc(history)
    # Save the model checkpoint
    torch.save(student_model.state_dict(), 'resnet20.ckpt')
