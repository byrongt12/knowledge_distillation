import torch
import torch.nn as nn
import torch.nn.functional as F

from Helper import getModelWeights, getFeatureMaps, distill
from Print import printFeatureMaps

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


def train_model_with_distillation(epochs, train_dl, test_dl, student_model, student_model_number, teacher_model, device, optimizer, max_lr,
                                  weight_decay, scheduler,
                                  grad_clip=None):
    torch.cuda.empty_cache()
    history = []

    optimizer = optimizer(student_model.parameters(), max_lr, weight_decay=weight_decay)
    scheduler = scheduler(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_dl))

    for epoch in range(epochs):
        student_model.train()  # put the model in train mode
        train_loss = []
        train_acc = []
        lrs = []

        for batch in train_dl:

            '''# Normal error and update
            loss, acc = student_model.training_step(batch)
            train_loss.append(loss)
            train_acc.append(acc)

            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(student_model.parameters(), grad_clip)

            optimizer.step()
            # optimizer.zero_grad()
            for param in student_model.parameters():
                param.grad = None'''

            # Distillation across feature map error - update only from the current layer backward
            # For each batch, step through GA string.

            featureMapNumForTeacher = 56
            featureMapNumForStudent = 1

            distill(featureMapNumForTeacher, featureMapNumForStudent, device, teacher_model, student_model,student_model_number, batch)

            '''# Step scheduler
            scheduler.step()
            lrs.append(get_lr(optimizer))'''

        result = evaluate(student_model, test_dl)
        # result["train_loss"] = torch.stack(train_loss).mean().item()
        # result["train_acc"] = torch.stack(train_acc).mean().item()
        # result["lrs"] = lrs
        print(result)
        # student_model.epoch_end(epoch, result)
        history.append(result)

    return history
