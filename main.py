import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from os import path
from torchviz import make_dot

from residualblock import ResidualBlock
from resnet import ResNet


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def printModel(model, t_loader):
    batch = next(iter(t_loader))
    yhat = model(batch[0].cuda())  # Give dummy batch to forward()
    make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz.png")


def getModelWeights(model):
    # save the convolutional layer weights
    m_weights = []
    # save the convolutional layers
    c_layers = []
    # get all the model children as list
    model_children = list(model.children())
    # counter to keep count of the conv layers
    counter = 0

    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            m_weights.append(model_children[i].weight)
            c_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        m_weights.append(child.weight)
                        c_layers.append(child)

    return m_weights, c_layers


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    num_epochs = 8
    learning_rate = 0.001

    # Image preprocessing modules
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])

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
                                               batch_size=100,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100,
                                              shuffle=False)

    # RESNET 110
    chk_path = "./resnet.ckpt"
    if path.exists(chk_path):
        print("load model")
        model = ResNet(ResidualBlock, [18, 18, 18]).to(device)
        model.load_state_dict(torch.load(chk_path))
    else:
        print("build model")
        model = ResNet(ResidualBlock, [18, 18, 18]).to(device)

    # Print model in PDF
    # printModel(model, train_loader)

    # Get model weights
    '''
    model_weights, conv_layers = getModelWeights(model)
    print(f"Total convolution layers: {len(conv_layers)}")
    print("conv_layers")
    print(len(conv_layers))
    print(model_weights[0])
    exit()
    '''
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

        # Decay learning rate
        if (epoch + 1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'resnet.ckpt')
