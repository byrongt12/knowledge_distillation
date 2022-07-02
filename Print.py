import pylab as pl
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchviz import make_dot
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

'''
dataIter = iter(train_loader)
imgs, labels = dataIter.next()

for i in range(2):
    imshow(torchvision.utils.make_grid(imgs[i]))
'''


def imshow(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


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


def printModelSummary(nn_model, firstConvWeight=False, allWeightsShape=False, summaryDisplay=True):
    if firstConvWeight or allWeightsShape:
        model_weights, conv_layers = getModelWeights(nn_model)
        print(f"Total convolution layers: {len(conv_layers)}")
        print("conv_layers")
        print(len(conv_layers))
        if firstConvWeight:
            print(model_weights[0])
        if allWeightsShape:
            for i in range(len(model_weights)):
                print(model_weights[i].shape)

    if summaryDisplay:
        summary(nn_model, (3, 32, 32))


def printFeatureMaps(model, device, train_loader):
    dataIter = iter(train_loader)
    imgs, labels = dataIter.next()

    image = imgs[0]

    print(f"Image shape before: {image.shape}")
    image = image.unsqueeze(0)
    print(f"Image shape after: {image.shape}")
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
    for feature_map in outputs:
        print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    for fm in processed:
        print(fm.shape)

    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(10, 11, i + 1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=30)
    plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')


def show_batch(dl):
    for batch in dl:
        images, labels = batch
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(make_grid(images[:20], nrow=5).permute(1, 2, 0))
        pl.show()
        break


def plot_acc(history):
    plt.plot([x["val_acc"] for x in history], "-x")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

def plot_loss(history):
    plt.plot([x.get("train_loss") for x in history], "-bx")
    plt.plot([x["val_loss"] for x in history], "-rx")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train loss", "val loss"])
    plt.show()

def plot_lrs(history):
    plt.plot(np.concatenate([x.get("lrs", []) for x in history]))
    plt.xlabel("Batch number")
    plt.ylabel("Learning rate")
    plt.show()
