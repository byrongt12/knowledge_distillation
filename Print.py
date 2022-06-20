import torch.nn as nn
from torchviz import make_dot
from torchsummary import summary

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