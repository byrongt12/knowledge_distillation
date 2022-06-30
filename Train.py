import torch
import torch.nn as nn

from Print import getModelWeights, printFeatureMaps
from Test import test_model


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def getFeatureMaps(model, device, train_loader):
    dataIter = iter(train_loader)
    imgs, labels = dataIter.next()

    image = imgs[0]

    # print(f"Image shape before: {image.shape}")
    image = image.unsqueeze(0)
    # print(f"Image shape after: {image.shape}")
    image = image.to(device)

    outputs = []
    names = []
    model_weights, conv_layers = getModelWeights(model)

    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))
    # print(len(outputs))
    # print feature_maps
    # for feature_map in outputs:
    #     print(feature_map.shape)

    return outputs


def train_model(model, device, train_loader, test_loader):
    # Hyper-parameters
    num_epochs = 80
    learning_rate = 0.05

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizerType = "sgd"

    if optimizerType == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    elif optimizerType == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)
    else:
        optimizer = None
        print("Please select correct optimizer type.")
        exit()

    print("Starting training with:")
    print("Epochs: " + str(num_epochs))
    print("Learning rate: " + str(learning_rate))
    print(optimizer)

    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # print("epoch: " + str(i))
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        if (epoch + 1) % 2 == 0:
            # Test during training
            print("Training data", end=" ")
            test_model(model, device, train_loader)
            print("Testing data", end=" ")
            test_model(model, device, test_loader)

        '''# Decay learning rate
        if (epoch + 1) % 15 == 0:
            curr_lr /= 5
            update_lr(optimizer, curr_lr)'''

        # Decay learning rate
        if (epoch + 1) % 25 == 0:
            curr_lr /= 10
            update_lr(optimizer, curr_lr)


