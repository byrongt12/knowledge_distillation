import torch
import torch.nn as nn

from Print import getModelWeights, printFeatureMaps
from Test import test_model

# Speed up training
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


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


def evaluate(model, test_dl):
    model.eval()
    outputs = [model.validation_step(batch) for batch in test_dl]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(epochs, train_dl, test_dl, model, optimizer, max_lr, weight_decay, scheduler, grad_clip=None):
    torch.cuda.empty_cache()
    history = []

    optimizer = optimizer(model.parameters(), max_lr, weight_decay=weight_decay)
    scheduler = scheduler(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_dl))

    for epoch in range(epochs):
        model.train()  # put the model in train mode
        train_loss = []
        train_acc = []
        lrs = []

        for batch in train_dl:
            loss, acc = model.training_step(batch)
            train_loss.append(loss)
            train_acc.append(acc)

            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            # optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None

            scheduler.step()
            lrs.append(get_lr(optimizer))
            
        result = evaluate(model, test_dl)
        result["train_loss"] = torch.stack(train_loss).mean().item()
        result["train_acc"] = torch.stack(train_acc).mean().item()
        result["lrs"] = lrs

        model.epoch_end(epoch, result)
        history.append(result)

    return history
