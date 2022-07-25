import torch  # need this for eval function
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision.transforms as transforms


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


def getNumberOfConvolutionLayers(nn_model):
    model_weights, conv_layers = getModelWeights(nn_model)
    return len(conv_layers)


def getFeatureMaps(model, device, image):
    # dataIter = iter(train_loader)
    # imgs, labels = next(dataIter)

    # image = imgs[0]

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

def printSingularFeatureMap(featureMap):
    feature_map = featureMap.squeeze(0)
    gray_scale = torch.sum(feature_map, 0)
    gray_scale = gray_scale / feature_map.shape[0]

    plt.imshow(gray_scale.data.cpu().numpy())
    plt.show()

def distill(featureMapNumForTeacher, featureMapNumForStudent, device, teacher_model, student_model, student_model_number, batch):

    student_model.train()  # put the model in train mode

    # Get optimizer set up for the student model.

    layer = None
    block = None
    conv = None

    divisor = student_model_number * 2
    quotient = featureMapNumForStudent // divisor
    if featureMapNumForStudent % divisor == 0:
        layer = quotient
    else:
        layer = quotient + 1

    if featureMapNumForStudent % divisor == 0:
        block = student_model_number
        conv = 2
    else:
        if (featureMapNumForStudent % divisor) % 2 == 0:
            block = ((featureMapNumForStudent % divisor) // 2)
        else:
            block = ((featureMapNumForStudent % divisor) // 2) + 1

    if conv is None:
        if ((featureMapNumForStudent % divisor) % 2) == 0:
            conv = 2
        else:
            conv = 1

    '''print(layer)
    print(block)
    print(conv)'''

    if layer is None or block is None or conv is None:
        print("Layer or block or conv is Null")
        exit()

    executeStr = 'torch.optim.SGD(list(student_model.layer' + str(layer) + '[' + str(block - 1) + '].conv' + str(
        conv) + '.parameters()), lr=0.3)'
    # torch.optim.SGD(list(student_model.layer2[0].conv2.parameters()), lr=0.3)  # The 8th conv layer.

    distill_optimizer = eval(executeStr)

    images, labels = batch

    for image in images:
        featureMapForTeacher = getFeatureMaps(teacher_model, device, image)[featureMapNumForTeacher]
        featureMapForStudent = getFeatureMaps(student_model, device, image)[featureMapNumForStudent]

        A = featureMapForTeacher.detach().clone()
        B = featureMapForStudent.detach().clone()

        # If matrices have different shapes: downsize to small one + shave off values make matrix size identical.
        if featureMapForTeacher.size() != featureMapForStudent.size():

            '''torch.set_printoptions(profile="full")
            print(A)
            print(B)

            print(A.size())
            print(B.size())'''

            if A.size() < B.size():  # if the total Student tensor is bigger but inner tensors smaller
                A = transforms.functional.resize(A, B.size()[3])
                B = B.narrow(1, 0, A.size()[1])

            elif A.size() > B.size():  # if the total Teacher tensor is bigger but inner tensors smaller
                B = transforms.functional.resize(B, A.size()[3])
                A = A.narrow(1, 0, B.size()[1])

        distill_loss = F.cosine_similarity(A.reshape(1, -1),
                                           B.reshape(1, -1))

        print("worked")
        exit()
        distill_loss.backward()
        distill_optimizer.step()
        distill_optimizer.zero_grad()
        break
